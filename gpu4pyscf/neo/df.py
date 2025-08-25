#!/usr/bin/env python

'''
Density-fitting for interaction Coulomb in (C)NEO
'''

import cupy
from gpu4pyscf import neo, __config__
from gpu4pyscf.df import df, df_jk, int3c2e
from gpu4pyscf.neo import hf, ks
from pyscf import lib as pyscf_lib


def dot_cderi_dm(intopt1, intopt2, eri1, eri2, dm1, dm2_len):
    nao1 = dm1.shape[-1]
    dms1 = dm1.reshape([-1,nao1,nao1])
    dms1 = intopt1.sort_orbitals(dms1, axis=[1,2])

    rows1 = intopt1.cderi_row
    cols1 = intopt1.cderi_col
    dm_sparse = dms1[:,rows1,cols1]
    dm_sparse *= 2
    dm_sparse[:, intopt1.cderi_diag] *= .5

    rhoj = dm_sparse.dot(eri1.T)
    vj_sparse = cupy.dot(rhoj, eri2)

    vj = cupy.zeros((1, dm2_len, dm2_len))
    rows2 = intopt2.cderi_row
    cols2 = intopt2.cderi_col
    vj[:,rows2,cols2] = vj_sparse
    vj[:,cols2,rows2] = vj_sparse
    vj = intopt2.unsort_orbitals(vj, axis=[1,2])
    vj = vj.reshape((dm2_len, dm2_len))
    return vj

def density_fit(mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
    assert isinstance(mf, neo.HF)

    if not isinstance(mf.components['e'], df_jk._DFHF):
        # Need to undo component before df because gpu df_jk._DFHF will
        # override get_veff. In pyscf, df_jk._DFHF won't.
        obj = mf.components['e'].undo_component()
        obj = df_jk.density_fit(obj, auxbasis=auxbasis, only_dfj=ee_only_dfj)
        mf.components['e'] = hf.general_scf(obj, charge=1.)

    if isinstance(mf, _DFNEO):
        return mf

    dfmf = _DFNEO(mf, auxbasis, ee_only_dfj, df_ne)
    if df_ne:
        name = _DFNEO.__name_mixin__ + '-EE&NE-' + mf.__class__.__name__
    else:
        name = _DFNEO.__name_mixin__ + '-EE-' + mf.__class__.__name__
    return pyscf_lib.set_class(dfmf, (_DFNEO, mf.__class__), name)

class DFInteractionCoulomb(hf.InteractionCoulomb):
    def __init__(self, *args, df_ne=False, auxbasis=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_ne = df_ne
        self.auxbasis = auxbasis
        self._cderi = None
        self._low = None
        self.intopt = None
        self.direct_scf_tol = getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)

    def get_vint(self, dm):
        if not self.df_ne or not \
           ((self.mf1_type.startswith('n') and self.mf2_type.startswith('e')) or
            (self.mf1_type.startswith('e') and self.mf2_type.startswith('n'))):
            return super().get_vint(dm)

        assert isinstance(dm, dict)
        assert self.mf1_type in dm or self.mf2_type in dm
        # Spin-insensitive interactions: sum over spin in dm first
        dm1 = dm.get(self.mf1_type)
        dm2 = dm.get(self.mf2_type)

        # Get total densities if the dm has two spin channels
        if dm1 is not None and self.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]

        if dm2 is not None and self.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]

        mol1 = self.mf1.mol
        mol2 = self.mf2.mol
        mol = mol1.super_mol
        assert mol == mol2.super_mol

        vj = {}
        if self.mf1_type == 'e':
            mf_e = self.mf1
            dm_e = dm1
            mf_n = self.mf2
            dm_n = dm2
            t_n = self.mf2_type
        else:
            mf_e = self.mf2
            dm_e = dm2
            mf_n = self.mf1
            dm_n = dm1
            t_n = self.mf1_type

        assert isinstance(mf_e, df_jk._DFHF)

        if mf_e.with_df.cd_low is None:
            mf_e.with_df.build(direct_scf_tol=self.direct_scf_tol)
        self._low = mf_e.with_df.cd_low

        if self._cderi is None:
            intopt = int3c2e.VHFOpt(mf_n.mol, mf_e.with_df.auxmol, 'int2e')
            intopt.build(self.direct_scf_tol, diag_block_with_triu=False, aosym=True,
                         group_size=df.GROUP_SIZE, group_size_aux=df.GROUP_SIZE)
            self.intopt = intopt
            self._cderi = df.cholesky_eri_gpu(intopt, mf_n.mol,
                                              mf_e.with_df.auxmol, self._low)
            assert len(self._cderi) == 1, "DF NEO for multiple gpus not implemented yet"
        if dm_e is not None:
            vj[t_n] = dot_cderi_dm(mf_e.with_df.intopt, self.intopt,
                                   mf_e.with_df._cderi[0], self._cderi[0],
                                   dm_e, mf_n.mol.nao_nr())

        if dm_n is not None:
            vj['e'] = dot_cderi_dm(self.intopt, mf_e.with_df.intopt,
                                   self._cderi[0], mf_e.with_df._cderi[0],
                                   dm_n, mf_e.mol.nao_nr())

        charge_product = self.mf1.charge * self.mf2.charge

        if self.mf1_type in vj:
            vj[self.mf1_type] *= charge_product

        if self.mf2_type in vj:
            vj[self.mf2_type] *= charge_product

        return vj

class DFInteractionCorrelation(ks.InteractionCorrelation, DFInteractionCoulomb):
    def __init__(self, *args, df_ne=False, auxbasis=None, epc=None, **kwargs):
        super().__init__(*args, epc=epc, **kwargs)
        DFInteractionCoulomb.__init__(self, *args, df_ne=df_ne,
                                      auxbasis=auxbasis, **kwargs)

class _DFNEO:
    __name_mixin__ = 'DF'

    _keys = {'ee_only_dfj', 'df_ne', 'auxbasis'}

    def __init__(self, mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
        self.__dict__.update(mf.__dict__)
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne

        self.auxbasis = auxbasis

        if isinstance(mf, neo.KS):
            self.interactions = hf.generate_interactions(
                self.components, DFInteractionCorrelation,
                self.max_memory, df_ne=self.df_ne,
                auxbasis=self.auxbasis, epc=mf.epc)
        else:
            self.interactions = hf.generate_interactions(
                self.components, DFInteractionCoulomb,
                self.max_memory, df_ne=self.df_ne,
                auxbasis=self.auxbasis)

    def nuc_grad_method(self):
        import pyscf.neo.df_grad
        return pyscf.neo.df_grad.Gradients(self)

    Gradients = pyscf_lib.alias(nuc_grad_method, alias_name='Gradients')

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        super().reset(mol)
        # Need this because components and interactions can be
        # completely destroyed in super().reset
        if not isinstance(self.components['e'], df_jk._DFHF):
            obj = self.components['e'].undo_component()
            obj = df_jk.density_fit(self.components['e'],
                                    auxbasis=self.auxbasis,
                                    only_dfj=self.ee_only_dfj)
            self.components['e'] = hf.general_scf(obj, charge=1.)
            if isinstance(self, neo.KS):
                self.interactions = hf.generate_interactions(
                    self.components, DFInteractionCorrelation,
                    self.max_memory, df_ne=self.df_ne,
                    auxbasis=self.auxbasis, epc=self.epc)
            else:
                self.interactions = hf.generate_interactions(
                    self.components, DFInteractionCoulomb,
                    self.max_memory, df_ne=self.df_ne,
                    auxbasis=self.auxbasis)
        if self.components['e'].with_df is not None:
            self.components['e'].with_df.cd_low = None
        for t, comp in self.interactions.items():
            comp._cderi = None
            comp._low = None
        return self
