#!/usr/bin/env python

'''
Density-fitting for interaction Coulomb in (C)NEO
'''

import cupy
from gpu4pyscf import scf, neo, __config__
from gpu4pyscf.df import df, df_jk, int3c2e
from gpu4pyscf.neo import hf
from pyscf import lib as pyscf_lib

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

class _DFNEO:
    __name_mixin__ = 'DF'

    _keys = {'ee_only_dfj', 'df_ne', 'auxbasis'}

    def __init__(self, mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
        self.__dict__.update(mf.__dict__)
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne
        self.auxbasis = auxbasis
        self.cderi_n = None
        self.intopt_n = None

    def get_vint(self, mol=None, dm=None, init_guess=False, **kwargs):
        if not self.df_ne:
            return super().get_vint(mol=mol, dm=dm, **kwargs)
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vint = {}
        mf_n = {}

        for t in self.components.keys():
            vint[t] = 0
        for (t1, t2), interaction in self.interactions.items():
            mf_n[t1] = interaction.mf1
            mf_n[t2] = interaction.mf2
            if t1.startswith('n') and t2.startswith('n') and not init_guess:
                v = interaction.get_vint(dm, **kwargs)
                vint[t1] += v[t1]
                vint[t2] += v[t2]
        del mf_n['e']

        mf_e = self.components['e']
        direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)
        if mf_e.with_df.cd_low is None:
            mf_e.with_df.build(direct_scf_tol=direct_scf_tol)
        low = mf_e.with_df.cd_low
        assert len(mf_e.with_df._cderi) == 1, "NEO DF NE for multiple gpus not supported yet"

        if self.cderi_n is None or self.intopt_n is None:
            cderi_n = {}
            intopt_n = {}
            for t, mf in mf_n.items():
                intopt_n[t] = int3c2e.VHFOpt(mf.mol, mf_e.with_df.auxmol, 'int2e')
                intopt_n[t].build(direct_scf_tol, diag_block_with_triu=False, aosym=True,
                                group_size=df.GROUP_SIZE, group_size_aux=df.GROUP_SIZE)
                cderi_n[t] = df.cholesky_eri_gpu(intopt_n[t], mf.mol,
                                                mf_e.with_df.auxmol, low)[0]
            self.cderi_n = cderi_n
            self.intopt_n = intopt_n

        dm_e = dm.get('e')
        if dm_e is not None and isinstance(mf_e, scf.uhf.UHF):
            assert dm_e.ndim > 2 and dm_e.shape[0] == 2
            dm_e = dm_e[0] +dm_e[1]

        if dm_e is not None:
            intopt = mf_e.with_df.intopt
            nao = dm_e.shape[-1]
            dm_e = dm_e.reshape([-1,nao,nao])
            dm_e = intopt.sort_orbitals(dm_e, axis=[1,2])
            rows = intopt.cderi_row
            cols = intopt.cderi_col
            dm_sparse = dm_e[:,rows,cols]
            dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5
            rhoj = dm_sparse.dot(mf_e.with_df._cderi[0].T)

            for t, cderi in self.cderi_n.items():
                vj_sparse = cupy.dot(rhoj, cderi)
                nao = mf_n[t].mol.nao_nr()
                vj = cupy.zeros((1, nao, nao))
                rows = self.intopt_n[t].cderi_row
                cols = self.intopt_n[t].cderi_col
                vj[:,rows,cols] = vj_sparse
                vj[:,cols,rows] = vj_sparse
                vj = self.intopt_n[t].unsort_orbitals(vj, axis=[1,2])
                vint[t] += vj.reshape((nao, nao)) * mf_n[t].charge
            dm_sparse = rhoj = vj = vj_sparse = None
            if init_guess:
                return vint

        rhoj = 0
        for t in mf_n.keys():
            dm_n = dm.get(t)
            if dm_n is not None:
                intopt = self.intopt_n[t]
                nao = dm_n.shape[-1]
                dm_n = dm_n.reshape([-1,nao,nao])
                dm_n = intopt.sort_orbitals(dm_n, axis=[1,2])
                rows = intopt.cderi_row
                cols = intopt.cderi_col
                dm_sparse = dm_n[:,rows,cols]
                dm_sparse *= 2
                dm_sparse[:, intopt.cderi_diag] *= .5
                rhoj += dm_sparse.dot(self.cderi_n[t].T) * mf_n[t].charge

        vj_sparse = cupy.dot(rhoj, mf_e.with_df._cderi[0])
        nao = mf_e.mol.nao_nr()
        vj = cupy.zeros((1, nao, nao))
        intopt = mf_e.with_df.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        vj[:,rows,cols] = vj_sparse
        vj[:,cols,rows] = vj_sparse
        vj = intopt.unsort_orbitals(vj, axis=[1,2])
        vint['e'] += vj.reshape((nao, nao))
        dm_sparse = rhoj = vj = vj_sparse = None

        for t in self.components.keys():
            self.components[t]._vint = vint[t]
            vint[t] = 0
        return vint

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        '''modified from hf.HF.get_init_guess'''
        if not self.df_ne:
            return super().get_init_guess(mol=mol, key=key, **kwargs)
        dm_guess = {}
        if not isinstance(key, str):
            if isinstance(key, dict): # several components are given
                dm_guess = key
            else: # cupy.ndarray
                dm_guess['e'] = key   # only e_guess is provided
            key = 'minao' # for remaining components, use default minao guess
        if mol is None: mol = self.mol
        if 'e' not in dm_guess:
            # Note that later in the code, super_mol is used instead of mol.components['e'],
            # because e_comp will have zero charges for quantum nuclei, but we want a
            # classical HF initial guess here
            e_guess = self.components['e'].get_init_guess(mol.components['e'], key, **kwargs)
            # alternatively, try the mixed initial guess
            # e_guess = init_guess_mixed(mol)
            dm_guess['e'] = e_guess

        if 'p' in self.components and 'p' not in dm_guess:
            p_guess = self.components['p'].get_init_guess(mol.components['p'], key, **kwargs)
            dm_guess['p'] = p_guess

        # Nuclear guess is obtained from the electron guess from a pure classical nuclei
        # calculation, then only set one nucleus to be quantum and obtain its solution
        vint = self.get_vint(dm=dm_guess, init_guess=True)
        for t, comp in self.components.items():
            if t.startswith('n') and t not in dm_guess:
                mol_tmp = neo.Mole()
                # Do not invoke possibly expensive QMMM during init guess
                mol_tmp.build(quantum_nuc=[comp.mol.atom_index],
                              nuc_basis=mol.nuclear_basis,
                              mm_mol=None, dump_input=False, parse_arg=False,
                              verbose=mol.verbose, output=mol.output,
                              max_memory=mol.max_memory, atom=mol.atom, unit=mol.unit,
                              nucmod=mol.nucmod, ecp=mol.ecp, pseudo=mol.pseudo,
                              charge=mol.charge, spin=mol.spin, symmetry=mol.symmetry,
                              symmetry_subgroup=mol.symmetry_subgroup, cart=mol.cart,
                              magmom=mol.magmom)
                if hasattr(mol.components[t], 'intor_symmetric_original'):
                    assert t in mol_tmp.components
                    mol_tmp.components[t].nao = mol.components[t].nao
                    mol_tmp.components[t].intor_symmetric_original = \
                            mol.components[t].intor_symmetric_original
                    mol_tmp.components[t].intor_symmetric = \
                            mol.components[t].intor_symmetric
                h_core = comp.get_hcore(mol_tmp.components[t])
                s = comp.get_ovlp(mol_tmp.components[t])
                mo_energy, mo_coeff = comp.eig(h_core + vint[t], s)
                mo_occ = comp.get_occ(mo_energy, mo_coeff)
                dm_guess[t] = comp.make_rdm1(mo_coeff, mo_occ)
        return dm_guess

    def nuc_grad_method(self):
        import gpu4pyscf.neo.df_grad
        return gpu4pyscf.neo.df_grad.Gradients(self)

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
        if self.components['e'].with_df is not None:
            self.components['e'].with_df.cd_low = None
        self.cderi_n = None
        self.intopt_n = None
        return self
