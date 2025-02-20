#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
EPC Not Supported
'''

from gpu4pyscf import dft, lib
from gpu4pyscf.lib import logger
from gpu4pyscf.neo import hf
class InteractionCorrelation(hf.InteractionCoulomb):
    '''Fake Inter-component Coulomb and correlation, EPC not supported yet'''
    def __init__(self, *args, epc=None, **kwargs):
        super().__init__(*args, **kwargs)
        if epc is not None:
            raise NotImplementedError('Electron-proton correlation is not supported on GPU yet.')

class KS(hf.HF):
    '''
    Examples::

    >>> from gpu4pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.KS(mol, xc='b3lyp5')
    >>> mf.max_cycle = 100
    >>> mf.scf()
    '''

    def __init__(self, mol, *args, xc=None, epc=None, **kwargs):
        super().__init__(mol, *args, **kwargs)
        # NOTE: To prevent user error, require xc to be explicitly provided
        if xc is None:
            raise RuntimeError('Please provide electronic xc via "xc" kwarg!')
        self.xc_e = xc # Electron xc functional
        if epc is not None:
            raise NotImplementedError('Electron-proton correlation is not supported on GPU yet.')
        self.epc = epc # Electron-proton correlation

        for t, comp in self.mol.components.items():
            if not t.startswith('n'):
                if self.unrestricted:
                    mf = dft.UKS(comp, xc=self.xc_e)
                else:
                    if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                        mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        mf = dft.RKS(comp, xc=self.xc_e)
                if self.df_ee:
                    mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = hf.general_scf(mf, charge=charge)
        self.interactions = hf.generate_interactions(self.components, InteractionCorrelation,
                                                     epc=self.epc)

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if vint is None: vint = self.get_vint(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['coul'] = 0
        self.scf_summary['exc'] = 0
        e_elec = 0
        e2 = 0
        for t, comp in self.components.items():
            logger.debug(self, f'Component: {t}')
            # Assign epc correlation energy to electrons
            if hasattr(vhf[t], 'exc') and hasattr(vint[t], 'exc'):
                vhf[t].exc += vint[t].exc
            if hasattr(vint[t], 'vj'):
                vj = vint[t].vj
            else:
                vj = vint[t]
            # vj acts as if a spin-insensitive one-body Hamiltonian
            # .5 to remove double-counting
            e_elec_t, e2_t = comp.energy_elec(dm[t], h1e[t] + vj * .5, vhf[t])
            e_elec += e_elec_t
            e2 += e2_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            # Nucleus is RHF and its scf_summary does not have coul or exc
            if hasattr(vhf[t], 'exc'):
                self.scf_summary['coul'] += comp.scf_summary['coul']
                self.scf_summary['exc'] += comp.scf_summary['exc']
        return e_elec, e2

    def get_vint_slow(self, mol=None, dm=None):
        '''Inter-type Coulomb and possible epc, slow version'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        # Old code that works with InteractionCorrelation.get_vint
        vint = {}
        for t in self.components.keys():
            vint[t] = 0
        for t_pair, interaction in self.interactions.items():
            v = interaction.get_vint(dm)
            for t in t_pair:
                # Take care of tag_array, accumulate exc and vj
                # NOTE: tag_array is scheduled to be removed in the future
                v_has_tag = hasattr(v[t], 'exc')
                vint_has_tag = hasattr(vint[t], 'exc')
                if v_has_tag:
                    if vint_has_tag:
                        exc = vint[t].exc + v[t].exc
                        vj = vint[t].vj + v[t].vj
                    else:
                        exc = v[t].exc
                        vj = v[t].vj
                    vint[t] = lib.tag_array(vint[t] + v[t], exc=exc, vj=vj)
                else:
                    if vint_has_tag:
                        vint[t] = lib.tag_array(vint[t] + v[t], exc=vint[t].exc, vj=vint[t].vj + v[t])
                    else:
                        vint[t] += v[t]
        return vint

    def get_vint_fast(self, mol=None, dm=None):
        '''Inter-type Coulomb and possible epc'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        # For better performance, avoid duplicated elec calculations for multiple protons

        # First get Vj
        # NOTE: super().get_vint just uses that function form. This does not mean
        # pure Coulomb will be used. The interaction class is still the Correlation one.
        # Therefore, we need to be able to get Coulomb within the Correlation class.
        # vj = super().get_vint(mol, dm, no_epc=True)
        vj = super().get_vint(mol, dm)
        if self.epc is None:
            return vj
        else:
            raise NotImplementedError('Electron-proton correlation not implemented on GPU.')

    get_vint = get_vint_fast

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        old_keys = sorted(self.components.keys())
        super().reset(mol=mol)
        if old_keys != sorted(self.components.keys()):
            # quantum nuc is different, need to rebuild
            for t, comp in self.mol.components.items():
                if not t.startswith('n'):
                    if self.unrestricted:
                        mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                            mf = dft.UKS(comp, xc=self.xc_e)
                        else:
                            mf = dft.RKS(comp, xc=self.xc_e)
                    if self.df_ee:
                        mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                    charge = 1.
                    if t.startswith('p'):
                        charge = -1.
                    self.components[t] = hf.general_scf(mf, charge=charge)
            self.interactions = hf.generate_interactions(self.components,
                                                         InteractionCorrelation,
                                                         epc=self.epc)
        return self

if __name__ == '__main__':
    from gpu4pyscf import neo
    mol = neo.M(atom='H 0 0 0', basis='ccpvdz', nuc_basis='pb4d', verbose=5, spin=1)
    mf = neo.KS(mol, xc='PBE')
    mf.scf()