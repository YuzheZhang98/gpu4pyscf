# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

'''
Hessian of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
from pyscf import lib, gto
from gpu4pyscf import scf
from gpu4pyscf.solvent.pcm import PI
from gpu4pyscf.solvent.grad.pcm import grad_qv, grad_solver, grad_nuc
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.hessian.jk import _ao2mo

def hess_nuc(pcmobj):
    if not pcmobj._intermediates:
        pcmobj.build()
    mol = pcmobj.mol
    q_sym        = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    # nuclei potential response
    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    v_ng_ip1ip2 = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol).reshape([3,3,mol.natm,-1])
    dv_g = numpy.einsum('n,xyng->ngxy', atom_charges, v_ng_ip1ip2)
    dv_g = numpy.einsum('ngxy,g->ngxy', dv_g, q_sym)

    de = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for ia in range(mol.natm):
        p0, p1 = gridslice[ia]
        de_tmp = numpy.sum(dv_g[:,p0:p1], axis=1)
        de[:,ia] -= de_tmp
        #de[ia,:] -= de_tmp.transpose([0,2,1])


    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    v_ng_ip1ip2 = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol, fakemol_nuc).reshape([3,3,-1,mol.natm])
    dv_g = numpy.einsum('n,xygn->gnxy', atom_charges, v_ng_ip1ip2)
    dv_g = numpy.einsum('gnxy,g->gnxy', dv_g, q_sym)

    for ia in range(mol.natm):
        p0, p1 = gridslice[ia]
        de_tmp = numpy.sum(dv_g[p0:p1], axis=0)
        de[ia,:] -= de_tmp
        #de[ia,:] -= de_tmp.transpose([0,2,1])

    int2c2e_ipip1 = mol._add_suffix('int2c2e_ipip1')
    v_ng_ipip1 = gto.mole.intor_cross(int2c2e_ipip1, fakemol_nuc, fakemol).reshape([3,3,mol.natm,-1])
    dv_g = numpy.einsum('g,xyng->nxy', q_sym, v_ng_ipip1)
    for ia in range(mol.natm):
        de[ia,ia] -= dv_g[ia] * atom_charges[ia]

    v_ng_ipip1 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc).reshape([3,3,-1,mol.natm])
    dv_g = numpy.einsum('n,xygn->gxy', atom_charges, v_ng_ipip1)
    dv_g = numpy.einsum('g,gxy->gxy', q_sym, dv_g)
    for ia in range(mol.natm):
        p0, p1 = gridslice[ia]
        de[ia,ia] -= numpy.sum(dv_g[p0:p1], axis=0)

    return de

def hess_qv(pcmobj, dm, verbose=None):
    raise NotImplementedError("PCM analytical hessian is not tested")
    if not pcmobj._intermediates or 'q_sym' not in pcmobj._intermediates:
        pcmobj._get_vind(dm)
    gridslice    = pcmobj.surface['gslice_by_atom']
    q_sym        = pcmobj._intermediates['q_sym']

    intopt = pcmobj.intopt
    intopt.clear()
    # rebuild with aosym
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False)
    coeff = intopt.coeff
    dm_cart = coeff @ dm @ coeff.T
    #dm_cart = cupy.einsum('pi,ij,qj->pq', coeff, dm, coeff)

    dvj, _ = int3c2e.get_int3c2e_ipip1_hjk(intopt, q_sym, None, dm_cart, with_k=False)
    dq, _ = int3c2e.get_int3c2e_ipvip1_hjk(intopt, q_sym, None, dm_cart, with_k=False)
    dvj, _ = int3c2e.get_int3c2e_ip1ip2_hjk(intopt, q_sym, None, dm_cart, with_k=False)
    dq, _ = int3c2e.get_int3c2e_ipip2_hjk(intopt, q_sym, None, dm_cart, with_k=False)

    cart_ao_idx = intopt.cart_ao_idx
    rev_cart_ao_idx = numpy.argsort(cart_ao_idx)
    dvj = dvj[:,rev_cart_ao_idx]

    aoslice = intopt.mol.aoslice_by_atom()
    dq = cupy.asarray([cupy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
    dvj= 2.0 * cupy.asarray([cupy.sum(dvj[:,p0:p1], axis=1) for p0,p1 in aoslice[:,2:]])
    de = dq + dvj
    return de.get()

def hess_elec(pcmobj, dm, verbose=None):
    '''
    slow version with finite difference
    TODO: use analytical hess_nuc
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    coords = mol.atom_coords(unit='Bohr')

    def pcm_grad_scanner(mol):
        # TODO: use more analytical forms
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        #return grad_elec(pcmobj, dm)
        pcm_grad = grad_nuc(pcmobj, dm)
        pcm_grad+= grad_solver(pcmobj, dm)
        pcm_grad+= grad_qv(pcmobj, dm)
        return pcm_grad

    mol.verbose = 0
    de = numpy.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-3
    for ia in range(mol.natm):
        for ix in range(3):
            dv = numpy.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            g0 = pcm_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            g1 = pcm_grad_scanner(mol)
            de[ia,:,ix] = (g0 - g1)/2.0/eps
    t1 = log.timer_debug1('solvent energy', *t1)
    pcmobj.reset(pmol)
    return de

def fd_grad_vmat(pcmobj, dm, mo_coeff, mo_occ, atmlst=None, verbose=None):
    '''
    dv_solv / da
    slow version with finite difference
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    if atmlst is None:
        atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    coords = mol.atom_coords(unit='Bohr')
    def pcm_vmat_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        return v

    mol.verbose = 0
    vmat = cupy.empty([len(atmlst), 3, nao, nocc])
    eps = 1e-3
    for i0, ia in enumerate(atmlst):
        for ix in range(3):
            dv = numpy.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            vmat0 = pcm_vmat_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            vmat1 = pcm_vmat_scanner(mol)

            grad_vmat = (vmat0 - vmat1)/2.0/eps
            grad_vmat = contract("ij,jq->iq", grad_vmat, mocc)
            grad_vmat = contract("iq,ip->pq", grad_vmat, mo_coeff)
            vmat[i0,ix] = grad_vmat
    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    pcmobj.reset(pmol)
    return vmat

"""
def analytic_grad_vmat(pcmobj, mo_coeff, mo_occ, atmlst=None, verbose=None):
    '''
    dv_solv / da
    slow version with finite difference
    '''
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    if atmlst is None:
        atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    dm = cupy.dot(mocc, mocc.T) * 2
    coords = mol.atom_coords(unit='Bohr')

    # TODO: add those contributions
    # contribution due to _get_v
    # contribution due to linear solver
    # contribution due to _get_vmat

    vmat = cupy.zeros([len(atmlst), 3, nao, nocc])

    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    pcmobj.reset(pmol)
    return vmat
"""

def make_hess_object(hess_method):
    if hess_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    name = (hess_method.base.with_solvent.__class__.__name__
            + hess_method.__class__.__name__)
    return lib.set_class(WithSolventHess(hess_method),
                         (WithSolventHess, hess_method.__class__), name)

class WithSolventHess:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.hessian import pcm           # type: ignore
        hess_method = self.undo_solvent().to_cpu()
        return pcm.make_hess_object(hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        is_equilibrium = self.base.with_solvent.equilibrium_solvation
        self.base.with_solvent.equilibrium_solvation = True
        self.de_solvent = hess_elec(self.base.with_solvent, dm, verbose=self.verbose)
        #self.de_solvent+= hess_nuc(self.base.with_solvent)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent
        self.base.with_solvent.equilibrium_solvation = is_equilibrium
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1(ao_repr=True)
            dv = fd_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1(ao_repr=True)
            dm = dm[0] + dm[1]
            dva = fd_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose)
            dvb = fd_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')
        
    def get_veff_resp_mo(self, mol, dms, mo_coeff, mo_occ, hermi=1):
        v1vo = super().get_veff_resp_mo(mol, dms, mo_coeff, mo_occ, hermi=hermi)
        if not self.base.with_solvent.equilibrium_solvation:
            return v1vo
        v_solvent = self.base.with_solvent._B_dot_x(dms)
        if isinstance(self.base, scf.uhf.UHF):
            n_dm = dms.shape[1]
            mocca = mo_coeff[0][:,mo_occ[0]>0]
            moccb = mo_coeff[1][:,mo_occ[1]>0]
            moa, mob = mo_coeff
            nmoa = moa.shape[1]
            nocca = mocca.shape[1]
            v1vo_sol = v_solvent[0] + v_solvent[1]
            v1vo[:,:nmoa*nocca] += _ao2mo(v1vo_sol, mocca, moa).reshape(n_dm,-1)
            v1vo[:,nmoa*nocca:] += _ao2mo(v1vo_sol, moccb, mob).reshape(n_dm,-1)
        elif isinstance(self.base, scf.hf.RHF):
            n_dm = dms.shape[0]
            mocc = mo_coeff[:,mo_occ>0]
            v1vo += _ao2mo(v_solvent, mocc, mo_coeff).reshape(n_dm,-1)
        else:
            raise NotImplementedError('Base object is not supported')
        return v1vo
    
    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass


