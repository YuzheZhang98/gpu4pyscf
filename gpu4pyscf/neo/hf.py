#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import cupy
import numpy
import scipy
import warnings
from pyscf import lib as pyscf_lib
from pyscf import gto
from pyscf.scf import hf as hf_cpu
from gpu4pyscf import lib, neo, scf
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import to_cupy, contract
from pyscf.data import nist
from gpu4pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL


def init_guess_mixed(mol, mixing_parameter = cupy.pi/4):
    ''' Copy from pyscf/examples/scf/56-h2_symm_breaking.py

    Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

    psi_1a = cupy.cos(q)*psi_homo + cupy.sin(q)*psi_lumo
    psi_1b = cupy.cos(q)*psi_homo - cupy.sin(q)*psi_lumo

    psi_2a = -cupy.sin(q)*psi_homo + cupy.cos(q)*psi_lumo
    psi_2b =  cupy.sin(q)*psi_homo + cupy.cos(q)*psi_lumo

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi

    # based on init_guess_by_1e
    h1e = to_cupy(hf_cpu.get_hcore(mol))
    s1e = to_cupy(hf_cpu.get_ovlp(mol))
    mo_energy, mo_coeff = hf_cpu.eig(h1e, s1e)
    mf = scf.HF(mol)
    mo_occ = mf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx = 0
    lumo_idx = 1

    for i in range(len(mo_occ)-1):
        if mo_occ[i] > 0 and mo_occ[i+1] < 0:
            homo_idx = i
            lumo_idx = i+1

    psi_homo = mo_coeff[:, homo_idx]
    psi_lumo = mo_coeff[:, lumo_idx]

    Ca = cupy.zeros_like(mo_coeff)
    Cb = cupy.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    q = mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:,k] = cupy.cos(q)*psi_homo + cupy.sin(q)*psi_lumo
            Cb[:,k] = cupy.cos(q)*psi_homo - cupy.sin(q)*psi_lumo
            continue
        if k == lumo_idx:
            Ca[:,k] = -cupy.sin(q)*psi_homo + cupy.cos(q)*psi_lumo
            Cb[:,k] =  cupy.sin(q)*psi_homo + cupy.cos(q)*psi_lumo
            continue
        Ca[:,k] = mo_coeff[:,k]
        Cb[:,k] = mo_coeff[:,k]

    dm = scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm

def as_scanner(mf):
    '''Generating a scanner/solver for (C)NEO PES.
    Copied from scf.hf.as_scanner
    '''
    if isinstance(mf, pyscf_lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + CNEO_Scanner.__name_mixin__
    return pyscf_lib.set_class(CNEO_Scanner(mf), (CNEO_Scanner, mf.__class__), name)

class CNEO_Scanner(pyscf_lib.SinglePointScanner):
    def __init__(self, mf_obj):
        self.__dict__.update(mf_obj.__dict__)
        self._last_mol_fp = mf_obj.mol.ao_loc

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, neo.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # Cleanup intermediates associated to the pervious mol object
        self.reset(mol)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0') # this can be electronic only or full
        elif self.mo_coeff is None:
            dm0 = None
        else:
            dm0 = None
            # dm0 form last calculation cannot be used in the current
            # calculation if a completely different system is given.
            # Obviously, the systems are very different if the number of
            # basis functions are different.
            # TODO: A robust check should include more comparison on
            # various attributes between current `mol` and the `mol` in
            # last calculation.
            if cupy.array_equal(self._last_mol_fp, mol.ao_loc):
                dm0 = self.make_rdm1()
            # TODO: chkfile support
            # Currently mo_coeff dumped is a dict, can't go through from_chk
            #elif self.chkfile and h5py.is_hdf5(self.chkfile):
            #    dm0 = self.from_chk(self.chkfile)
        for _, comp in self.components.items():
            comp.mo_coeff = None
        e_tot = self.kernel(dm0=dm0, **kwargs)
        self._last_mol_fp = mol.ao_loc
        return e_tot

def hcore_qmmm(mol, mm_mol, direct_scf_tol):
    '''From qmmm/itrf'''
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()

    v = 0
    if mm_mol.charge_model == 'gaussian':
        expnts = mm_mol.get_zetas()
        fakemol = gto.fakemol_for_charges(coords, expnts)
        intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
        intopt.build(direct_scf_tol, diag_block_with_triu=False, aosym=True,
                     group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
        v += int3c2e.get_j_int3c2e_pass2(intopt, -cupy.asarray(charges))
    else: # point-charge model
        nao = mol.nao
        max_memory = mol.super_mol.max_memory - pyscf_lib.current_memory()[0]
        blksize = int(min(max_memory*1e6/8/nao**2, 200))
        blksize = max(blksize, 1)
        for i0, i1 in pyscf_lib.prange(0, charges.size, blksize):
            j3c = mol.intor('int1e_grids', hermi=1, grids=coords[i0:i1])
            v += contract('kpq,k->pq', j3c, -charges[i0:i1])
    return v.get()

def general_scf(method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
    '''Modify SCF (HF and DFT) method to support for general charge
    and general mass, such that positrons and nuclei can be calculated.

    Args:
        charge : float
            Charge of the particle. 1 means electron, -1 means positron.
        mass : float
            Mass of the particle in a.u. Nuclei will have high mass
        is_nucleus : bool
            If the particle is nucleus. Nucleus won't see PP and is
            considered a distinguishable single particle
        nuc_occ_state : int
            Select the nuclear orbital that is occupied. For Delta-SCF.
    '''
    assert (isinstance(method, scf.hf.SCF))
    if isinstance(method, Component):
        method.charge = charge
        method.mass = mass
        method.is_nucleus = is_nucleus
        method.nuc_occ_state = nuc_occ_state
        return method
    return pyscf_lib.set_class(ComponentSCF(method, charge, mass, is_nucleus, nuc_occ_state),
                         (ComponentSCF, method.__class__))

class Component:
    __name_mixin__ = 'Component'

class ComponentSCF(Component):
    _keys = {'charge', 'mass', 'is_nucleus', 'nuc_occ_state'}
    def __init__(self, method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
        self.__dict__.update(method.__dict__)
        self.charge = charge
        self.mass = mass
        self.is_nucleus = is_nucleus
        self.nuc_occ_state = nuc_occ_state

    def undo_component(self):
        obj = lib.view(self, pyscf_lib.drop_class(self.__class__, Component))
        return obj

    def get_hcore(self, mol=None):
        '''Core Hamiltonian'''
        if mol is None: mol = self.mol
        h = mol.intor_symmetric('int1e_kin') / self.mass

        if mol._pseudo and not self.is_nucleus:
            # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
            # may exist if mol is converted from cell object.
            from pyscf.gto import pp_int
            h += pp_int.get_gth_pp(mol) * self.charge
        else:
            h += mol.intor_symmetric('int1e_nuc') * self.charge

        if len(mol._ecpbas) > 0 and not self.is_nucleus:
            h += mol.intor_symmetric('ECPscalar') * self.charge

        if mol.super_mol.mm_mol is not None:
            h += hcore_qmmm(mol, mol.super_mol.mm_mol, self.direct_scf_tol) * self.charge
        return to_cupy(h)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if self.is_nucleus: # Nucleus does not have self-type interaction
            if mol is None:
                mol = self.mol
            return cupy.zeros((mol.nao, mol.nao))
        else:
            if abs(self.charge) != 1.:
                raise NotImplementedError('General charge J/K with tag_array')
            return super().get_veff(mol, dm, dm_last, vhf_last, hermi)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''Support fractional occupation. For nucleus, make sure it is a single particle'''
        if mo_energy is None: mo_energy = self.mo_energy
        if self.is_nucleus:
            e_idx = cupy.argsort(mo_energy)
            mo_occ = cupy.zeros(mo_energy.size)
            mo_occ[e_idx[self.nuc_occ_state]] = self.mol.nnuc # 1 or fractional
            return mo_occ
        else:
            if self.mol.nhomo is None:
                return super().get_occ(mo_energy, mo_coeff)
            else:
                assert isinstance(self, scf.uhf.UHF)
                e_idx_a = cupy.argsort(mo_energy[0])
                e_idx_b = cupy.argsort(mo_energy[1])
                e_sort_a = mo_energy[0][e_idx_a]
                e_sort_b = mo_energy[1][e_idx_b]
                nmo = mo_energy[0].size
                n_a, n_b = self.nelec
                mo_occ = cupy.zeros_like(mo_energy)
                # change the homo occupation to fractional
                if n_a > n_b or n_b <= 0:
                    mo_occ[0,e_idx_a[:n_a - 1]] = 1
                    mo_occ[0,e_idx_a[n_a - 1]] = self.mol.nhomo
                    mo_occ[1,e_idx_b[:n_b]] = 1
                else:
                    mo_occ[1,e_idx_b[:n_b - 1]] = 1
                    mo_occ[1,e_idx_b[n_b - 1]] = self.mol.nhomo
                    mo_occ[0,e_idx_a[:n_a]] = 1
                if self.verbose >= logger.INFO and n_a < nmo and n_b > 0 and n_b < nmo:
                    if e_sort_a[n_a-1]+1e-3 > e_sort_a[n_a]:
                        logger.warn(mf, 'alpha nocc = %d  HOMO %.15g >= LUMO %.15g',
                                    n_a, e_sort_a[n_a-1], e_sort_a[n_a])
                    else:
                        logger.info(mf, '  alpha nocc = %d  HOMO = %.15g  LUMO = %.15g',
                                    n_a, e_sort_a[n_a-1], e_sort_a[n_a])

                    if e_sort_b[n_b-1]+1e-3 > e_sort_b[n_b]:
                        logger.warn(mf, 'beta  nocc = %d  HOMO %.15g >= LUMO %.15g',
                                    n_b, e_sort_b[n_b-1], e_sort_b[n_b])
                    else:
                        logger.info(mf, '  beta  nocc = %d  HOMO = %.15g  LUMO = %.15g',
                                    n_b, e_sort_b[n_b-1], e_sort_b[n_b])

                    if e_sort_a[n_a-1]+1e-3 > e_sort_b[n_b]:
                        logger.warn(mf, 'system HOMO %.15g >= system LUMO %.15g',
                                    e_sort_b[n_a-1], e_sort_b[n_b])

                    cupy.set_printoptions(threshold=nmo)
                    logger.debug(self, '  alpha mo_energy =\n%s', mo_energy[0])
                    logger.debug(self, '  beta  mo_energy =\n%s', mo_energy[1])
                    cupy.set_printoptions(threshold=1000)

                if mo_coeff is not None and self.verbose >= logger.DEBUG:
                    ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                              mo_coeff[1][:,mo_occ[1]>0]), self.get_ovlp())
                    logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
                return mo_occ

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        if self.is_nucleus:
            # Nuclear initial DM cannot be generated here. It needs electronic DM
            # for the initial diagonalization.
            return 0
        else:
            # For electrons, use super_mol to generate the guess, because e_mol
            # will have zero charges for quantum nuclei, but we want a classical
            # HF initial guess here.
            # For positrons, just use the electron guess as a bad guess, but
            # better than nothing.
            if mol is None:
                mol = self.mol
            charge = self.charge
            self.charge = abs(charge)
            dm = super().get_init_guess(mol.super_mol, key, **kwargs)
            self.charge = charge
            return dm

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        raise AttributeError('energy_tot should not be called from component SCF')


class InteractionCoulomb:
    '''Inter-component Coulomb interactions'''
    def __init__(self, mf1_type, mf1, mf2_type, mf2):
        self.mf1_type = mf1_type
        self.mf1 = mf1
        self.mf1_unrestricted = isinstance(self.mf1, scf.uhf.UHF)
        self.mf2_type = mf2_type
        self.mf2 = mf2
        self.mf2_unrestricted = isinstance(self.mf2, scf.uhf.UHF)

    def get_vint(self, dm):
        '''Obtain vj for both components'''
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
        # set not needed dm to zero matrix
        if dm1 is None:
            dm1 = cupy.zeros((mol1.nao, mol1.nao))
        if dm2 is None:
            dm2 = cupy.zeros((mol2.nao, mol2.nao))
        vj[self.mf1_type], vj[self.mf2_type] = scf.hf.jk.get_j_4fold(mol1, mol2,
                                                                     dm1, dm2, hermi=1)
        charge_product = self.mf1.charge * self.mf2.charge
        if self.mf1_type in vj:
            vj[self.mf1_type] *= charge_product
        if self.mf2_type in vj:
            vj[self.mf2_type] *= charge_product
        return vj

def generate_interactions(components, interaction_class, **kwagrs):
    keys = sorted(components.keys())

    interactions = {}
    for i, p1 in enumerate(keys):
        for p2 in keys[i+1:]:
            if p1 != p2:
                interaction = interaction_class(p1, components[p1],
                                                p2, components[p2],
                                                **kwagrs)
                interactions[(p1, p2)] = interaction

    return interactions

def get_fock(mf, h1e=None, s1e=None, vhf=None, vint=None, dm=None, cycle=-1,
             diis=None, diis_start_cycle=None, level_shift_factor=None,
             damp_factor=None, fock_last=None, diis_pos='both', diis_type=3):
    if h1e is None: h1e = mf.get_hcore()
    if dm is None: dm = mf.make_rdm1()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if vint is None: vint = mf.get_vint(mf.mol, dm)
    f = {}
    for t in mf.components.keys():
        f[t] = cupy.asarray(h1e[t]) + vhf[t] + cupy.asarray(vint[t])
        if mf.unrestricted and not t.startswith('n') and f[t].ndim == 2:
            f[t] = cupy.asarray((f[t],) * 2)

    # CNEO constraint term
    # NOTE: even if not using DIIS, we still optimize f.
    # This helps with final extra cycle convergence.
    f0 = None
    if isinstance(mf, neo.CDFT):
        if diis_pos == 'pre' or diis_pos == 'both' or (cycle < 0 and diis is None):
            # optimize f in cNEO
            for t, comp in mf.components.items():
                if t.startswith('n'):
                    ia = comp.mol.atom_index
                    opt = scipy.optimize.root(mf.position_analysis, mf.f[ia].get(),
                                              args=(comp, f[t], s1e[t]),
                                              method='hybr')
                    logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                 (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                    logger.debug(mf, 'Position deviation: %s', opt.fun)

        # For DIIS type 1, preserve original matrices
        if diis_type == 1:
            f0 = f.copy()

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] += fock_add[t]

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()

    # TODO: unpack level_shift_factor and damp_factor

    for t in mf.components.keys():
        if mf.unrestricted and not t.startswith('n') and \
                isinstance(dm[t], cupy.ndarray) and dm[t].ndim == 2:
            dm[t] = cupy.asarray((dm[t]*0.5,) * 2)

    if damp_factor is not None and abs(damp_factor) > 1e-4:
        warnings.warn('Damping for multi-component SCF is not yet implemented.')
    # if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:

    if diis is not None and cycle >= diis_start_cycle:
        if isinstance(mf, neo.CDFT):
            # if CNEO, needs to manually use lib.diis and pack/unpack
            keys = sorted(f.keys())
            shapes = {k: f[k].shape for k in keys}
            if diis_type != 1:
                f_flat = cupy.concatenate([f[k].ravel() for k in keys])

            if diis_type == 1:
                f0_flat = cupy.concatenate([f0[k].ravel() for k in keys])
                f_flat = diis.update(f0_flat, scf.diis.get_err_vec(s1e, dm, f))
            elif diis_type == 2:
                f_flat = diis.update(f_flat)
            elif diis_type == 3:
                f_flat = diis.update(f_flat, scf.diis.get_err_vec(s1e, dm, f))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")

            # Reconstruct dictionary
            offset = 0
            f_new = {}
            for k in keys:
                size = numpy.prod(shapes[k])
                f_new[k] = f_flat[offset:offset+size].reshape(shapes[k])
                offset += size
            f = f_new

            if diis_type == 1:
                for t in fock_add:
                    f[t] += fock_add[t]
        else:
            # if not CNEO, directly use the scf.diis object provided
            f = diis.update(s1e, dm, f)
            # WARNING: CDIIS only. Using EDIIS or ADIIS will cause errors

    if level_shift_factor is not None and abs(level_shift_factor) > 1e-4:
        warnings.warn('Level shift for multi-component SCF is not yet implemented.')
    # if abs(level_shift_factor) > 1e-4:

    # Post-DIIS CDFT optimization
    if isinstance(mf, neo.CDFT) and (diis_pos == 'post' or diis_pos == 'both'):
        f0 = {}
        for t in f:
            if t.startswith('n'):
                f0[t] = f[t] - fock_add[t]
            else:
                f0[t] = f[t]

        for t, comp in mf.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                opt = scipy.optimize.root(mf.position_analysis, mf.f[ia].get(),
                                          args=(comp, f0[t], s1e[t]), method='hybr')
                logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                             (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                logger.debug(mf, 'Position deviation: %s', opt.fun)

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] = f0[t] + fock_add[t]
    return f

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = cupy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, **kwargs)
    else:
        dm = mf.get_init_guess(mol, dm0, **kwargs)

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    vint = mf.get_vint(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf, vint)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)  # = h1e + vhf + vint, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if isinstance(mf, neo.CDFT):
        # mf_diis needs to be the raw lib.diis.DIIS() for CNEO
        mf_diis = lib.diis.DIIS()
        mf_diis.space = 8

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    fock_last = None
    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, vint, dm, cycle, mf_diis, fock_last=fock_last)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        vint = mf.get_vint(mol, dm)
        e_tot = mf.energy_tot(dm, h1e, vhf, vint)

        # Here Fock matrix is h1e + vhf + vint, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)  # = h1e + vhf + vint, no DIIS
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {}
        for t in grad.keys():
            norm_gorb[t] = cupy.linalg.norm(grad[t])
            if not TIGHT_GRAD_CONV_TOL:
                norm_gorb[t] = norm_gorb[t] / cupy.sqrt(norm_gorb[t].size)
        norm_ddm = {}
        for t in dm.keys():
            norm_ddm[t] = cupy.linalg.norm(dm[t]-dm_last[t])
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad.keys():
            if not t.startswith('e'):
                logger.info(mf, f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g',
                            norm_gorb[t], norm_ddm[t])

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb['e'] < conv_tol_grad:
            scf_conv = True

        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    mf.cycles = cycle + 1
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        vint = mf.get_vint(mol, dm)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf, vint), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {}
        for t in grad.keys():
            norm_gorb[t] = cupy.linalg.norm(grad[t])
            if not TIGHT_GRAD_CONV_TOL:
                norm_gorb[t] = norm_gorb[t] / cupy.sqrt(norm_gorb[t].size)
        norm_ddm = {}
        for t in dm.keys():
            norm_ddm[t] = cupy.linalg.norm(dm[t]-dm_last[t])

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb['e'] < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad.keys():
            if not t.startswith('e'):
                logger.info(mf, f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g',
                            norm_gorb[t], norm_ddm[t])
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

class HF(scf.hf.SCF):
    '''Multicomponent Hartree-Fock

    Examples::

    >>> from gpu4pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz',
    >>>             nuc_basis='pb4d')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    -99.98104139461894
    '''
    def __init__(self, mol, unrestricted=False, df_ee=False,
                 auxbasis_e=None, only_dfj_e=False):
        super().__init__(mol)
        self.df_ee = df_ee
        self.auxbasis_e = auxbasis_e
        self.only_dfj_e = only_dfj_e
        if self.mol.components['e'].nhomo is not None or self.mol.spin != 0:
            unrestricted = True
        # When either electron or positron wave function is unrestricted,
        # both will be unrestricted.
        if self.mol.components.get('p') is not None and self.mol.components['p'].spin != 0:
            unrestricted = True
        self.unrestricted = unrestricted
        self.components = {}
        for t, comp in self.mol.components.items():
            if t.startswith('n'):
                self.components[t] = general_scf(scf.RHF(comp),
                                                 charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                 mass=self.mol.mass[comp.atom_index]
                                                      * nist.ATOMIC_MASS / nist.E_MASS,
                                                 is_nucleus=True,
                                                 nuc_occ_state=0)
            else:
                if self.unrestricted:
                    mf = scf.UHF(comp)
                else:
                    mf = scf.RHF(comp)
                if self.df_ee:
                    mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = general_scf(mf, charge=charge)
        self.interactions = generate_interactions(self.components, InteractionCoulomb)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__.__name__)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Charge      Location')
                coords = self.mol.mm_mol.atom_coords()
                charges = self.mol.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug(self, '%.9g    %s', z, coords[i])
        return self

    def check_sanity(self):
        return self.components['e'].check_sanity()

    def eig(self, h, s):
        e = {}
        c = {}
        for t, comp in self.components.items():
            e[t], c[t] = comp.eig(h[t], s[t])
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = {}
        for t, comp in mol.components.items():
            hcore[t] = self.components[t].get_hcore(mol=comp)
        return hcore

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        ovlp = {}
        for t, comp in mol.components.items():
            ovlp[t] = self.components[t].get_ovlp(mol=comp)
        return ovlp

    get_fock = get_fock

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mo_occ = {}
        for t, comp in self.components.items():
            coeff = mo_coeff.get(t) if mo_coeff is not None and \
                    isinstance(mo_coeff, dict) else None
            mo_occ[t] = comp.get_occ(mo_energy[t], coeff)
        return mo_occ

    def get_grad(self, mo_coeff, mo_occ, fock):
        grad = {}
        for t, comp in self.components.items():
            grad[t] = comp.get_grad(mo_coeff[t], mo_occ[t], fock[t])
        return grad

    def get_init_guess(self, mol=None, key='minao', **kwargs):
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
                if hasattr(comp, 'intor_symmetric_original'):
                    assert t in mol_tmp.components
                    mol_tmp.components[t].nao = mol.components[t].nao
                    mol_tmp.components[t].intor_symmetric_original = \
                            mol.components[t].intor_symmetric_original
                    mol_tmp.components[t].intor_symmetric = \
                            mol.components[t].intor_symmetric
                h_core = comp.get_hcore(mol_tmp.components[t])
                s = comp.get_ovlp(mol_tmp.components[t])
                vint = 0
                for t_pair, interaction in self.interactions.items():
                    # Only get e Coulomb on n
                    if t in t_pair and 'e' in t_pair:
                        vint += interaction.get_vint(dm_guess)[t]
                mo_energy, mo_coeff = comp.eig(h_core + vint, s)
                mo_occ = comp.get_occ(mo_energy, mo_coeff)
                dm_guess[t] = comp.make_rdm1(mo_coeff, mo_occ)
        return dm_guess

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm = {}
        for t, comp in self.components.items():
            coeff = comp.mo_coeff if mo_coeff is None else mo_coeff[t]
            occ = comp.mo_occ if mo_occ is None else mo_occ[t]
            dm[t] = comp.make_rdm1(coeff, occ)
        return dm

    def make_rdm2(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm2 = {}
        for t, comp in self.components.items():
            coeff = comp.mo_coeff if mo_coeff is None else mo_coeff[t]
            occ = comp.mo_occ if mo_occ is None else mo_occ[t]
            dm2[t] = comp.make_rdm2(coeff, occ)
        return dm2

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if vint is None: vint = self.get_vint(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['e2'] = 0
        e_elec = 0
        e_coul = 0
        for t, comp in self.components.items():
            logger.debug(self, f'Component: {t}')
            # vint acts as if a spin-insensitive one-body Hamiltonian
            # .5 to remove double-counting
            e_elec_t, e_coul_t = comp.energy_elec(dm[t], h1e[t] + vint[t] * .5, vhf[t])
            e_elec += e_elec_t
            e_coul += e_coul_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            self.scf_summary['e2'] += comp.scf_summary['e2']
        return e_elec, e_coul

    def energy_tot(self, dm=None, h1e=None, vhf=None, vint=None):
        nuc = self.energy_nuc()
        self.scf_summary['nuc'] = nuc.real

        e_tot = self.energy_elec(dm, h1e, vhf, vint)[0] + nuc
        if self.disp is not None:
            self.components['e'].disp = self.disp
        if self.components['e'].do_disp():
            if 'dispersion' in self.components['e'].scf_summary:
                e_tot += self.components['e'].scf_summary['dispersion']
            else:
                e_disp = self.components['e'].get_dispersion()
                self.components['e'].scf_summary['dispersion'] = e_disp
                e_tot += e_disp
            self.scf_summary['dispersion'] = self.components['e'].scf_summary['dispersion']

        return e_tot

    def energy_nuc(self):
        nuc = self.components['e'].energy_nuc()
        if self.mol.mm_mol is not None:
            # interactions between QM nuclei and MM particles
            mm_mol = self.mol.mm_mol
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            mol_e = self.components['e'].mol
            for j in range(mol_e.natm):
                q2, r2 = mol_e.atom_charge(j), mol_e.atom_coord(j)
                r = pyscf_lib.norm(r2-coords, axis=1)
                nuc += q2*(charges/r).sum()
        return nuc

    def scf(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = self.make_rdm1()

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
            # Distribute mo_energy, mo_coeff and mo_occ
            for t, comp in self.components.items():
                comp.mo_energy = self.mo_energy[t]
                comp.mo_coeff = self.mo_coeff[t]
                comp.mo_occ = self.mo_occ[t]
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'Multicomponent-SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = pyscf_lib.alias(scf, alias_name='kernel')

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        raise AttributeError('get_jk should not be called from multi-component SCF')

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_j should not be called from multi-component SCF')

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_k should not be called from multi-component SCF')

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Self-type JK'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vhf = {}
        for t, comp in self.components.items():
            dm_last_t = dm_last.get(t, 0) if isinstance(dm_last, dict) else 0
            vhf_last_t = vhf_last.get(t, 0) if isinstance(vhf_last, dict) else 0
            vhf[t] = comp.get_veff(mol.components[t], dm[t], dm_last_t, vhf_last_t, hermi)
        return vhf

    def get_vint(self, mol=None, dm=None, **kwargs):
        '''Inter-type Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vint = {}
        for t in self.components.keys():
            vint[t] = 0
        for t_pair, interaction in self.interactions.items():
            v = interaction.get_vint(dm, **kwargs)
            for t in t_pair:
                vint[t] += v[t]
        return vint

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        return self.components['e'].mulliken_pop(mol, dm, s, verbose)

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=None, s=None):
        if pre_orth_method is not None:
            return self.components['e'].mulliken_meta(mol, dm, verbose, pre_orth_method, s)
        return self.components['e'].mulliken_meta(mol, dm, verbose, s=s)

    def canonicalize(self, mo_coeff, mo_occ, fock=None):
        raise NotImplementedError

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE, **kwargs):
        raise NotImplementedError # CDFT will have this

    def quad_moment(self, mol=None, dm=None, unit='DebyeAngstrom', origin=None,
                    verbose=logger.NOTE, **kwargs):
        raise NotImplementedError

    def _is_mem_enough(self):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        raise AttributeError('density_fit should not be used. Check df_ee option.')

    def sfx2c1e(self):
        raise NotImplementedError

    def newton(self):
        raise NotImplementedError

    def remove_soscf(self):
        raise NotImplementedError

    def stability(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise AttributeError('Only CNEO-DFT has gradients.')

    def update_(self, chkfile=None):
        raise NotImplementedError

    as_scanner = as_scanner

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        if mol is not None:
            self.mol = mol
        super().reset(mol=mol)
        if sorted(self.components.keys()) == sorted(self.mol.components.keys()):
            # quantum nuc is the same, reset each component
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
            for t, comp in self.interactions.items():
                comp._eri = None
        else:
            # quantum nuc is different, need to rebuild
            self.components.clear()
            for t, comp in self.mol.components.items():
                if t.startswith('n'):
                    self.components[t] = general_scf(scf.RHF(comp),
                                                     charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                     mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS / nist.E_MASS,
                                                     is_nucleus=True,
                                                     nuc_occ_state=0)
                else:
                    if self.unrestricted:
                        mf = scf.UHF(comp)
                    else:
                        mf = scf.RHF(comp)
                    if self.df_ee:
                        mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                    charge = 1.
                    if t.startswith('p'):
                        charge = -1.
                    self.components[t] = general_scf(mf, charge=charge)
            self.interactions = generate_interactions(self.components, InteractionCoulomb)

        return self

    def apply(self, fn, *args, **kwargs):
        raise NotImplementedError

    def to_ks(self, xc='HF'):
        raise NotImplementedError

    def to_gpu(self):
        raise NotImplementedError

if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0', basis='ccpvdz', nuc_basis='pb4d', verbose=5, spin=1)
    mf = neo.HF(mol)
    mf.scf()