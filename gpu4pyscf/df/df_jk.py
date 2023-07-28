#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

import copy
import cupy
import numpy
from pyscf import lib, scf, __config__
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.df import df_jk, addons
from gpu4pyscf.lib.cupy_helper import contract, solve_triangular, unpack_tril, take_last2d, transpose_sum, load_library, get_avail_mem
from gpu4pyscf.dft import rks, numint
from gpu4pyscf.scf import hf
from gpu4pyscf.df import df, int3c2e

libcupy_helper = load_library('libcupy_helper')

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def init_workflow(mf, dm0=None):
    # build CDERI for omega = 0 and omega ! = 0
    def build_df():
        mf.with_df.build()
        if hasattr(mf, '_numint'):
            omega, _, _ = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
            if abs(omega) <= 1e-10: return
            key = '%.6f' % omega
            if key in mf.with_df._rsh_df:
                rsh_df = mf.with_df._rsh_df[key]
            else:
                rsh_df = mf.with_df._rsh_df[key] = copy.copy(mf.with_df).reset()
            rsh_df.build(omega=omega)
        return

    # pre-compute h1e and s1e and cderi for async workflow
    with lib.call_in_background(build_df) as build:
        build()
        mf.s1e = cupy.asarray(mf.get_ovlp(mf.mol))
        mf.h1e = cupy.asarray(mf.get_hcore(mf.mol))
        # for DFT object
        if hasattr(mf, '_numint'):
            ni = mf._numint
            rks.initialize_grids(mf, mf.mol, dm0)
            ni.build(mf.mol, mf.grids.coords)
            mf._numint.xcfuns = numint._init_xcfuns(mf.xc)
            mf._numint.use_sparsity = 0
    dm0 = cupy.asarray(dm0)
    return

def _density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.
    Args:
        mf : an SCF object
    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        only_dfj : str
            Compute Coulomb integrals only and no approximation for HF
            exchange. Same to RIJONX in ORCA
    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K
    Examples:
    '''

    assert isinstance(mf, scf.hf.SCF)
    
    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf_class = mf.__class__

    if isinstance(mf, df_jk._DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = copy.copy(mf)
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    class DensityFitting(df_jk._DFHF, mf_class):
        __doc__ = '''
        Density fitting SCF class
        Attributes for density-fitting SCF:
            auxbasis : str or basis dict
                Same format to the input attribute mol.basis.
                The default basis 'weigend+etb' means weigend-coulomb-fit basis
                for light elements and even-tempered basis for heavy elements.
            with_df : DF object
                Set mf.with_df = None to switch off density fitting mode.
        See also the documents of class %s for other SCF attributes.
        ''' % mf_class
        
        def __init__(self, mf, dfobj, only_dfj):
            self.__dict__.update(mf.__dict__)
            self._eri = None
            self.rhoj = None
            self.rhok = None
            self.direct_scf = False
            self.with_df = dfobj
            self.only_dfj = only_dfj
            self._keys = self._keys.union(['with_df', 'only_dfj'])
            
        init_workflow = init_workflow

        def reset(self, mol=None):
            self.with_df.reset(mol)
            return mf_class.reset(self, mol)

        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                   omega=None):
            if dm is None: dm = self.make_rdm1()
            if self.with_df and self.only_dfj:
                vj = vk = None
                if with_j:
                    vj, vk = self.with_df.get_jk(dm, hermi, True, False,
                                                 self.direct_scf_tol, omega)
                if with_k:
                    vk = mf_class.get_jk(self, mol, dm, hermi, False, True, omega)[1]
            elif self.with_df:
                vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                             self.direct_scf_tol, omega)
            else:
                vj, vk = mf_class.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
            return vj, vk
        
        def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
            '''
            effective potential
            '''
            if mol is None: mol = self.mol
            if dm is None: dm = self.make_rdm1()
            
            # for DFT
            if mf_class == rks.RKS:
                return rks._get_veff(self, dm=dm)

            if self.direct_scf:
                ddm = cupy.asarray(dm) - dm_last
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                return vhf_last + vj - vk * .5
            else:
                vj, vk = self.get_jk(mol, dm, hermi=hermi)
                return vj - vk * .5
        
        def energy_elec(self, dm=None, h1e=None, vhf=None):
            '''
            electronic energy
            '''
            if dm is None: dm = self.make_rdm1()
            if h1e is None: h1e = self.get_hcore()
            if vhf is None: vhf = self.get_veff(self.mol, dm)
            # for DFT
            if mf_class == rks.RKS:
                e1 = cupy.sum(h1e*dm)
                ecoul = self.ecoul
                exc = self.exc
                e2 = ecoul + exc        
                #logger.debug(self, f'E1 = {e1}, Ecoul = {ecoul}, Exc = {exc}')
                return e1+e2, e2
            
            e1 = cupy.einsum('ij,ji->', h1e, dm).real
            e_coul = cupy.einsum('ij,ji->', vhf, dm).real * .5
            self.scf_summary['e1'] = e1
            self.scf_summary['e2'] = e_coul
            #logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
            return e1+e_coul, e_coul

        def energy_tot(self, dm, h1e, vhf=None):
            '''
            compute tot energy
            '''
            nuc = self.energy_nuc()
            e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc
            self.scf_summary['nuc'] = nuc.real
            return e_tot
        
        def nuc_grad_method(self):
            if mf_class == rks.RKS:
                from gpu4pyscf.df.grad import rks as rks_grad
                return rks_grad.Gradients(self)
            if mf_class == hf.RHF:
                from gpu4pyscf.df.grad import rhf as rhf_grad
                return rhf_grad.Gradients(self)
            raise NotImplementedError()
        

        def Hessian(self):
            from gpu4pyscf.df.hessian import rhf, rks
            if isinstance(self, scf.rhf.RHF):
                if isinstance(self, scf.hf.KohnShamDFT):
                    return rks.Hessian(self)
                else:
                    return rhf.Hessian(self)
            else:
                raise NotImplementedError

        # for pyscf 1.0, 1.1 compatibility
        @property
        def _cderi(self):
            naux = self.with_df.get_naoaux()
            return next(self.with_df.loop(blksize=naux))
        @_cderi.setter
        def _cderi(self, x):
            self.with_df._cderi = x

        @property
        def auxbasis(self):
            return getattr(self.with_df, 'auxbasis', None)
    
    return DensityFitting(mf, with_df, only_dfj)

def get_jk(dfobj, dms_tag, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-14, omega=None):
    '''
    get jk with density fitting
    outputs and input are on the same device
    '''
    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    out_shape = dms_tag.shape
    out_cupy = isinstance(dms_tag, cupy.ndarray)
    if not isinstance(dms_tag, cupy.ndarray):
        dms_tag = cupy.asarray(dms_tag)

    assert(with_j or with_k)
    if dms_tag is None: logger.error("dm is not given")
    nao = dms_tag.shape[-1]
    dms = dms_tag.reshape([-1,nao,nao])
    nset = dms.shape[0]
    t0 = (logger.process_clock(), logger.perf_counter())
    if dfobj._cderi is None: 
        log.warn('CDERI not found, build...')    
        dfobj.build(direct_scf_tol=direct_scf_tol, omega=omega)

    nao, naux = dfobj.nao, dfobj.naux
    vj = None; vk = None
    ao_idx = dfobj.intopt.sph_ao_idx
    dms = take_last2d(dms, ao_idx)

    t1 = log.timer_debug1('init jk', *t0)
    if with_j:
        tril_row = dfobj.tril_row
        tril_col = dfobj.tril_col
        dm_tril = dms[:, tril_row, tril_col]
        dm_tril[:, dfobj.diag_idx] *= .5
        rhoj = 2.0*dm_tril.dot(dfobj._cderi.T)
        vj_tril = cupy.dot(rhoj, dfobj._cderi)
        dm_tril = None
        
        vj = cupy.empty_like(dms)
        vj[:,tril_row, tril_col] = vj_tril
        vj[:,tril_col, tril_row] = vj_tril
        vj_tril = None
    if with_k:
        vk = cupy.zeros_like(dms)
        # SCF K matrix with occ
        if nset == 1 and hasattr(dms_tag, 'occ_coeff'):
            occ_coeff = cupy.asarray(dms_tag.occ_coeff[ao_idx, :], order='C')
            nocc = occ_coeff.shape[1]
            blksize = dfobj.get_blksize(extra=nao*nocc)
            cderi_buf = cupy.empty([blksize, nao, nao])
            for p0, p1, cderi_tril in dfobj.loop(blksize=blksize):
                unpack_tril(cderi_tril, cderi_buf)
                # leading dimension is 1
                rhok = contract('Lij,jk->Lki', cderi_buf[:p1-p0], occ_coeff)
                vk[0] += contract('Lki,Lkj->ij', rhok, rhok)
                #contract('Lki,Lkj->ij', rhok, rhok, alpha=1.0, beta=1.0, out=vk[0])
            vk *= 2.0
        # CP-HF K matrix
        elif hasattr(dms_tag, 'mo1'):
            mo1 = dms_tag.mo1[:,ao_idx,:]
            nocc = mo1.shape[2]
            occ_coeff = dms_tag.occ_coeff[ao_idx,:] * 2.0  # due to rhok and rhok1, put it here for symmetry
            blksize = dfobj.get_blksize(extra=2*nao*nocc)
            cderi_buf = cupy.empty([blksize, nao, nao])
            for p0, p1, cderi_tril in dfobj.loop(blksize=blksize):
                unpack_tril(cderi_tril, cderi_buf)
                rhok = contract('Lij,jk->Lki', cderi_buf[:p1-p0], occ_coeff)
                for i in range(mo1.shape[0]):
                    rhok1 = contract('Lij,jk->Lki', cderi_buf[:p1-p0], mo1[i])
                    vk[i] += contract('Lki,Lkj->ij', rhok, rhok1)
                    #contract('Lki,Lkj->ij', rhok, rhok1, alpha=1.0, beta=1.0, out=vk[i])
            cderi_buf = occ_coeff = rhok1 = rhok = mo1 = None
            #vk *= 2.0
            vk = vk + vk.transpose(0,2,1)
        # general K matrix with density matrix
        else:
            blksize = dfobj.get_blksize()
            cderi_buf = cupy.empty([blksize, nao, nao])
            for p0, p1, cderi_tril in dfobj.loop(blksize=blksize):
                unpack_tril(cderi_tril, cderi_buf)
                for k in range(nset):
                    rhok = contract('Lij,jk->Lki', cderi_buf[:p1-p0], dms[k])
                    vk[k] += contract('Lki,Lkj->ij', cderi_buf[:p1-p0], rhok)
            rhok = cderi_buf = None

    rev_ao_idx = numpy.argsort(dfobj.intopt.sph_ao_idx)
    if with_j:
        vj = take_last2d(vj, rev_ao_idx)
        vj = vj.reshape(out_shape)
    if with_k:
        vk = take_last2d(vk, rev_ao_idx)
        vk = vk.reshape(out_shape)
    t1 = log.timer_debug1('vj and vk', *t1)
    if out_cupy:
        return vj, vk
    else:
        return vj.get() if vj else None, vk.get() if vk else None

def _get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True,
            direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
            omega=None):
    if omega is None:
        return get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)

    # A temporary treatment for RSH-DF integrals
    key = '%.6f' % omega
    if key in dfobj._rsh_df:
        rsh_df = dfobj._rsh_df[key]
    else:
        rsh_df = dfobj._rsh_df[key] = copy.copy(dfobj).reset()
        logger.info(dfobj, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)
    
    with rsh_df.mol.with_range_coulomb(omega):
        return get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13):
    intopt = getattr(dfobj, 'intopt', None)
    if intopt is None:
        dfobj.build(direct_scf_tol=direct_scf_tol)
        intopt = dfobj.intopt
    j2c = dfobj.j2c
    rhoj = int3c2e.get_j_int3c2e_pass1(intopt, dm)
    if dfobj.cd_low.tag == 'eig':
        rhoj = cupy.linalg.lstsq(j2c, rhoj)
    else:
        rhoj = cupy.linalg.solve(j2c, rhoj)

    rhoj *= 2.0
    vj = int3c2e.get_j_int3c2e_pass2(intopt, rhoj)
    return vj