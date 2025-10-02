#!/usr/bin/env python

'''
Analytic nuclear gradient for constrained nuclear-electronic orbital
'''

import numpy
import cupy
import warnings
from gpu4pyscf import neo, scf
from gpu4pyscf.df import int3c2e
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf
from gpu4pyscf.lib.cupy_helper import tag_array, contract, ensure_numpy, get_avail_mem
from pyscf.scf.jk import get_jk
from pyscf import gto
from pyscf import lib as pyscf_lib
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.qmmm.pbc.tools import get_multipole_tensors_pp, get_multipole_tensors_pg


def general_grad(grad_method):
    '''Modify gradient method to support general charge and mass.
    Similar to general_scf decorator in neo/hf.py
    '''
    if isinstance(grad_method, ComponentGrad):
        return grad_method

    assert (isinstance(grad_method.base, scf.hf.SCF) and
            isinstance(grad_method.base, neo.hf.Component))

    return grad_method.view(pyscf_lib.make_class((ComponentGrad, grad_method.__class__)))

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of general charged particle gradients
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ndim = dm0.ndim

    if ndim > 2:
        dm0_org = dm0
        dm0 = dm0[0] + dm0[1]
        dme0 = dme0[0] + dme0[1]

    # (\nabla i | hcore | j) - (\nabla i | j)
    h1 = cupy.asarray(mf_grad.get_hcore(mol))
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))

    # (i | \nabla hcore | j)
    t3 = log.init_timer()
    dh1e = int3c2e.get_dh1e(mol, dm0) * mf.charge

    # Calculate ECP contributions in (i | \nabla hcore | j) and
    # (\nabla i | hcore | j) simultaneously
    if mol.has_ecp():
        # TODO: slice ecp_atoms
        ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1 -= h1_ecp.sum(axis=0)

        dh1e[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm0)
    t3 = log.timer_debug1('gradients of h1e', *t3)

    if ndim > 2:
        dvhf = mf_grad.get_veff(mol, dm0_org)
    else:
        dvhf = mf_grad.get_veff(mol, dm0)
    log.timer_debug1('gradients of veff', *t3)
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')

    extra_force = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += ensure_numpy(mf_grad.extra_force(ia, locals()))

    log.timer_debug1('gradients of 2e part', *t3)

    dh = contract('xij,ij->xi', h1, dm0)
    ds = contract('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = ensure_numpy(2.0 * dvhf + dh1e + delec)
    de += extra_force

    log.timer_debug1('gradients of electronic part', *t0)
    return de

class ComponentGrad:
    __name_mixin__ = 'Component'

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def get_hcore(self, mol=None):
        '''Core Hamiltonian first derivatives for general charged particle'''
        if mol is None: mol = self.mol

        # Kinetic and nuclear potential derivatives
        h = -mol.intor('int1e_ipkin', comp=3) / self.base.mass
        if mol._pseudo:
            raise NotImplementedError('Nuclear gradients for GTH PP')
        else:
            h -= mol.intor('int1e_ipnuc', comp=3) * self.base.charge
        if mol.has_ecp():
            h -= mol.intor('ECPscalar_ipnuc', comp=3) * self.base.charge

        # Add MM contribution if present
        if mol.super_mol.mm_mol is not None:
            mm_mol = mol.super_mol.mm_mol
            rcut_hcore = mm_mol.rcut_hcore
            coords = cupy.asarray(mm_mol.atom_coords())
            charges = cupy.asarray(mm_mol.atom_charges())

            Ls = cupy.asarray(mm_mol.get_lattice_Ls())

            qm_center = cupy.mean(cupy.asarray(mol.atom_coords()), axis=0)
            all_coords = (coords[None,:,:] + Ls[:,None,:]).reshape(-1,3)
            all_charges = cupy.hstack([charges] * len(Ls))
            dist2 = all_coords - qm_center
            dist2 = contract('ix,ix->i', dist2, dist2)

            # charges within rcut_hcore exactly go into hcore
            mask = dist2 <= rcut_hcore**2
            charges = all_charges[mask]
            coords = all_coords[mask]
            if mm_mol.charge_model == 'gaussian' and len(coords) != 0:
                expnts = cupy.hstack([mm_mol.get_zetas()] * len(Ls))[mask]
                v = int1e_grids_ip1(mol, coords, charges=charges, charge_exponents=expnts)
            elif mm_mol.charge_model == 'point' and len(coords) != 0:
                raise RuntimeError("Not tested yet")
                max_memory = scf_grad.max_memory - lib.current_memory()[0]
                blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
                blksize = max(blksize, 1)
                coords = coords.get()
                for i0, i1 in lib.prange(0, len(coords), blksize):
                    j3c = cp.asarray(mol.intor('int1e_grids_ip', grids=coords[i0:i1]))
                    g_qm += contract('ikpq,k->ipq', j3c, charges[i0:i1])
            else: # len(coords) == 0
                pass
            h += self.base.charge * v.get()
        return h

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            raise NotImplementedError('X2C not supported')
        else:
            with_ecp = mol.has_ecp()
            if with_ecp:
                ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
            else:
                ecp_atoms = ()
            aoslices = mol.aoslice_by_atom()
            h1 = self.get_hcore(mol)
            def hcore_deriv(atm_id):
                p0, p1 = aoslices[atm_id][2:]
                # External potential gradient
                if not mol.super_mol._quantum_nuc[atm_id]:
                    with mol.with_rinv_at_nucleus(atm_id):
                        vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                        vrinv *= -mol.atom_charge(atm_id)
                        if with_ecp and atm_id in ecp_atoms:
                            vrinv += mol.intor('ECPscalar_iprinv', comp=3)
                        vrinv *= self.base.charge
                else:
                    vrinv = numpy.zeros((3, mol.nao, mol.nao))
                # Hcore gradient
                vrinv[:,p0:p1] += h1[:,p0:p1]
                return vrinv + vrinv.transpose(0,2,1)
        return hcore_deriv

    def get_veff(self, mol=None, dm=None):
        '''
        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix,
        here it returns the first-order derivatives of the energy contributions from Veff
        per atom.
        '''
        if self.base.is_nucleus: # Nucleus does not have self-type interaction
            if mol is None:
                mol = self.mol
            return cupy.zeros((mol.natm, 3))
        else:
            assert abs(self.base.charge) == 1
            return super().get_veff(mol, dm)

    grad_elec = grad_elec

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise AttributeError

def grad_pair_int(mol1, mol2, dm1, dm2, charge1, charge2, atmlst):
    de = numpy.zeros((len(atmlst),3))
    aoslices1 = mol1.aoslice_by_atom()
    aoslices2 = mol2.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices1[ia]
        # Derivative w.r.t. mol1
        if shl1 > shl0:
            shls_slice = (shl0, shl1) + (0, mol1.nbas) + (0, mol2.nbas)*2
            v1 = get_jk((mol1, mol1, mol2, mol2),
                        dm2.get(), scripts='ijkl,lk->ij',
                        intor='int2e_ip1', aosym='s2kl', comp=3,
                        shls_slice=shls_slice)
            de[i0] -= 2. * charge1 * charge2 * \
                      numpy.einsum('xij,ij->x', v1, dm1[p0:p1].get())
        shl0, shl1, p0, p1 = aoslices2[ia]
        # Derivative w.r.t. mol2
        if shl1 > shl0:
            shls_slice = (shl0, shl1) + (0, mol2.nbas) + (0, mol1.nbas)*2
            v1 = get_jk((mol2, mol2, mol1, mol1),
                        dm1.get(), scripts='ijkl,lk->ij',
                        intor='int2e_ip1', aosym='s2kl', comp=3,
                        shls_slice=shls_slice)
            de[i0] -= 2. * charge1 * charge2 * \
                      numpy.einsum('xij,ij->x', v1, dm2[p0:p1].get())
    return de

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calculate gradient for inter-component Coulomb interactions'''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    de = numpy.zeros((len(atmlst),3))

    for (t1, t2), interaction in mf.interactions.items():
        comp1 = mf.components[t1]
        comp2 = mf.components[t2]
        dm1 = dm0[t1]
        if interaction.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        dm2 = dm0[t2]
        if interaction.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        mol1 = comp1.mol
        mol2 = comp2.mol
        de += grad_pair_int(mol1, mol2, dm1, dm2,
                            comp1.charge, comp2.charge, atmlst)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)
    return de

def grad_hcore_mm(mf_grad, dm=None, mol=None):
    '''Calculate QMMM gradient for MM atoms'''
    if mol is None:
        mol = mf_grad.mol
    mm_mol = mol.mm_mol
    if mm_mol is None:
        warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
        return None
    mf = mf_grad.base
    if dm is None:
        dm = mf.make_rdm1()

    rcut_hcore = mm_mol.rcut_hcore

    coords = cupy.asarray(mm_mol.atom_coords())
    charges = cupy.asarray(mm_mol.atom_charges())
    expnts = cupy.asarray(mm_mol.get_zetas())

    Ls = cupy.asarray(mm_mol.get_lattice_Ls())

    qm_center = cupy.mean(cupy.asarray(mol.atom_coords()), axis=0)
    all_coords = (coords[None,:,:] + Ls[:,None,:]).reshape(-1,3)
    all_charges = cupy.hstack([charges] * len(Ls))
    all_expnts = cupy.hstack([expnts] * len(Ls))
    dist2 = all_coords - qm_center
    dist2 = contract('ix,ix->i', dist2, dist2)

    # charges within rcut_hcore exactly go into hcore
    mask = dist2 <= rcut_hcore**2
    charges = all_charges[mask]
    coords = all_coords[mask]
    expnts = all_expnts[mask]
    g = cupy.zeros_like(all_coords)

    if len(coords) != 0:
        expnts = cupy.hstack([mm_mol.get_zetas()] * len(Ls))[mask]
        # Handle each charged component's interaction with MM
        for t, comp in mf.components.items():
            mol_comp = comp.mol
            dm_comp = dm[t]
            g[mask] += (comp.charge *
                       int1e_grids_ip2(mol_comp, coords, dm=dm_comp,
                                       charges=charges, charge_exponents=expnts).T)
        g = g.reshape(len(Ls), -1, 3)
        g = cupy.sum(g, axis=0)

    return g.get()

def grad_nuc_mm(mf_grad, mol=None):
    '''Nuclear gradients of the QM-MM nuclear energy
    (in the form of point charge Coulomb interactions)
    with respect to MM atoms.
    '''
    if hasattr(mf_grad, 'de_nuc_mm'):
        return mf_grad.de_nuc_mm

    from scipy.special import erf
    if mol is None:
        mol = mf_grad.mol
    mm_mol = mol.mm_mol
    if mm_mol is None:
        warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
        return None
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    Ls = mm_mol.get_lattice_Ls()
    qm_center = numpy.mean(mol.atom_coords(), axis=0)
    all_coords = pyscf_lib.direct_sum('ix+Lx->Lix',
            mm_mol.atom_coords(), Ls).reshape(-1,3)
    all_charges = numpy.hstack([mm_mol.atom_charges()] * len(Ls))
    all_expnts = numpy.hstack([numpy.sqrt(mm_mol.get_zetas())] * len(Ls))
    dist2 = all_coords - qm_center
    dist2 = pyscf_lib.einsum('ix,ix->i', dist2, dist2)
    mask = dist2 <= mm_mol.rcut_hcore**2
    charges = all_charges[mask]
    coords = all_coords[mask]
    expnts = all_expnts[mask]

    g_mm = numpy.zeros_like(all_coords)
    mol_e = mol.components['e']
    for i in range(mol_e.natm):
        q1 = mol_e.atom_charge(i)
        r1 = mol_e.atom_coord(i)
        r = pyscf_lib.norm(r1-coords, axis=1)
        g_mm[mask] += q1 * pyscf_lib.einsum('i,ix,i->ix', charges, r1-coords, erf(expnts*r)/r**3)
        g_mm[mask] -= q1 * pyscf_lib.einsum('i,ix,i->ix', charges * expnts * 2 / numpy.sqrt(numpy.pi),
                                            r1-coords, numpy.exp(-expnts**2 * r**2)/r**2)
    g_mm = g_mm.reshape(len(Ls), -1, 3)
    g_mm = numpy.sum(g_mm, axis=0)
    return g_mm

def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    This is different from GradientsBase.as_scanner because CNEO uses two
    layers of mole objects.

    Copied from grad.rhf.as_scanner
    '''
    if mf_grad.base.mol.mm_mol is not None:
        raise NotImplementedError('(C)NEO PBC QM/MM scanner not implemneted.')
    if isinstance(mf_grad, pyscf_lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + CNEO_GradScanner.__name_mixin__
    return pyscf_lib.set_class(CNEO_GradScanner(mf_grad),
                               (CNEO_GradScanner, mf_grad.__class__), name)

class CNEO_GradScanner(pyscf_lib.GradScanner):
    def __init__(self, g):
        pyscf_lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, neo.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
            e_tot = mf_scanner(mol, dm0=dm0)
        else:
            e_tot = mf_scanner(mol)

        for t in mf_scanner.components.keys():
            if isinstance(mf_scanner.components[t], hf.KohnShamDFT):
                if getattr(self.components[t], 'grids', None):
                    self.components[t].grids.reset(mol.components[t])
                if getattr(self.components[t], 'nlcgrids', None):
                    self.components[t].nlcgrids.reset(mol.components[t])

        de = self.kernel(**kwargs)
        return e_tot, de

class Gradients(rhf_grad.GradientsBase):
    '''Analytic gradients for CDFT

    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; H 0 0 0.917', basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2')
    >>> mf.kernel()
    >>> g = neo.Gradients(mf)
    >>> g.kernel()
    '''
    def __init__(self, mf):
        super().__init__(mf)
        self.grid_response = None

        # Get base gradient for each component
        self.components = {}
        for t, comp in self.base.components.items():
            self.components[t] = general_grad(comp.nuc_grad_method())
        self._keys = self._keys.union(['grid_response', 'components'])

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__.__name__)
            if self.verbose >= logger.DEBUG1:
                logger.debug1(self, 'Charge      Location')
                coords = self.mol.mm_mol.atom_coords()
                charges = self.mol.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug1(self, '%.9g    %s', z, coords[i])
            return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(self.mol)
        if sorted(self.components.keys()) == sorted(self.mol.components.keys()):
            # quantum nuc is the same, reset each component
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
        else:
            # quantum nuc is different, need to rebuild
            self.components.clear()
            for t, comp in self.base.components.items():
                self.components[t] = general_grad(comp.nuc_grad_method())
        return self

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g_qm = self.components['e'].grad_nuc(mol.components['e'], atmlst)
        if mol.mm_mol is not None:
            from scipy.special import erf
            assert atmlst is None
            mm_mol = mol.mm_mol
            coords = cupy.asarray(mm_mol.atom_coords())
            charges = cupy.asarray(mm_mol.atom_charges())
            Ls = cupy.asarray(mm_mol.get_lattice_Ls())
            qm_center = cupy.mean(cupy.asarray(mol.atom_coords()), axis=0)
            all_coords = (coords[None, :, :] + Ls[:, None, :]).reshape(-1, 3)
            all_charges = cupy.hstack([charges] * len(Ls))
            all_expnts = cupy.hstack([cupy.sqrt(cupy.asarray(mm_mol.get_zetas()))]
                                     * len(Ls))
            dist2 = all_coords - qm_center
            dist2 = contract('ix,ix->i', dist2, dist2)
            mask = dist2 <= mm_mol.rcut_hcore**2
            charges = all_charges[mask]
            coords = all_coords[mask]
            expnts = all_expnts[mask]
            g_mm = cupy.zeros_like(all_coords)
            mol_e = mol.components['e']
            for i in range(mol_e.natm):
                q1 = mol_e.atom_charge(i)
                r1 = cupy.asarray(mol_e.atom_coord(i))
                r = cupy.linalg.norm(r1-coords, axis=1)
                g_mm_ = q1 * contract('ix,i->ix', r1-coords,
                                      charges * erf(expnts*r)/r**3)
                g_mm_ -= q1 * contract('ix,i->ix', r1-coords,
                                       charges * expnts * 2 / numpy.sqrt(numpy.pi)
                                       * cupy.exp(-expnts**2 * r**2)/r**2)
                g_mm[mask] += g_mm_
                g_qm[i] -= cupy.sum(g_mm_, axis=0).get()
            g_mm = g_mm.reshape(len(Ls), -1, 3)
            g_mm = cupy.sum(g_mm, axis=0)
            self.de_nuc_mm = g_mm.get()

        return g_qm

    def symmetrize(self, de, atmlst=None):
        return rhf_grad.symmetrize(self.mol.components['e'], de, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # Get inter-component interaction gradient first
        de = self.grad_int(mo_energy, mo_coeff, mo_occ, atmlst)
        # add gradient from each component
        for t, comp in self.components.items():
            if self.grid_response is not None and isinstance(comp.base, hf.KohnShamDFT):
                comp.grid_response = self.grid_response
            de += comp.grad_elec(mo_energy=mo_energy[t], mo_coeff=mo_coeff[t],
                                 mo_occ=mo_occ[t], atmlst=atmlst)

        # Add EPC contribution if needed
        if hasattr(self.base, 'epc') and self.base.epc is not None:
            raise NotImplementedError('Electron-proton correlation is not supported on GPU yet.')
            # de += self.grad_epc(mo_energy, mo_coeff, mo_occ, atmlst)

        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        if self.base.do_disp():
            self.de += self.components['e'].get_dispersion()
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = pyscf_lib.alias(kernel, alias_name='grad')

    def grad_ewald(self, dm=None, with_mm=False, mm_ewald_pot=None, qm_ewald_pot=None):
        '''PBC correction energy grad w.r.t. qm and mm atom positions
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        if dm is None:
            dm = self.base.make_rdm1()
        mol = self.base.mol
        cell = self.base.mol.mm_mol
        assert cell is not None and cell.dimension == 3
        qm_charges = self.base.get_qm_charges(dm)
        qm_dipoles = self.base.get_qm_dipoles(dm)
        qm_quads = self.base.get_qm_quadrupoles(dm)
        qm_coords = cupy.asarray(mol.atom_coords())
        mm_charges = cupy.asarray(cell.atom_charges())
        mm_coords = cupy.asarray(cell.atom_coords())

        # nuc grad due to qm multipole change due to ovlp change
        # only electron has this part
        qm_multipole_grad = cupy.zeros_like(qm_coords)

        if mm_ewald_pot is None:
            if self.base.mm_ewald_pot is not None:
                mm_ewald_pot = self.base.mm_ewald_pot
            else:
                mm_ewald_pot = self.base.get_mm_ewald_pot(mol, cell)
        if qm_ewald_pot is None:
            qm_ewald_pot = self.base.get_qm_ewald_pot(mol, dm, self.base.qm_ewald_hess)
        ewald_pot = \
            mm_ewald_pot[0] + qm_ewald_pot[0], \
            mm_ewald_pot[1] + qm_ewald_pot[1], \
            mm_ewald_pot[2] + qm_ewald_pot[2]

        mol_e = mol.components['e']
        dEds = cupy.zeros((mol_e.nao, mol_e.nao))
        dEdsr = cupy.zeros((3, mol_e.nao, mol_e.nao))
        dEdsrr = cupy.zeros((3, 3, mol_e.nao, mol_e.nao))
        s1 = cupy.asarray(self.get_ovlp(mol_e)) # = -mol.intor('int1e_ipovlp')
        s1r = list()
        s1rr = list()
        bas_atom = mol_e._bas[:,gto.ATOM_OF]
        aoslices = mol_e.aoslice_by_atom()
        for iatm in range(mol_e.natm):
            v0 = cupy.asarray(ewald_pot[0][iatm])
            v1 = cupy.asarray(ewald_pot[1][iatm])
            v2 = cupy.asarray(ewald_pot[2][iatm])
            p0, p1 = aoslices[iatm, 2:]

            dEds[p0:p1] -= v0 * dm['e'][p0:p1]
            dEdsr[:,p0:p1] -= contract('x,uv->xuv', v1, dm['e'][p0:p1])
            dEdsrr[:,:,p0:p1] -= contract('xy,uv->xyuv', v2, dm['e'][p0:p1])

            b0, b1 = numpy.where(bas_atom == iatm)[0][[0,-1]]
            shlslc = (b0, b1+1, 0, mol_e.nbas)
            with mol_e.with_common_orig(qm_coords[iatm].get()):
                # s1r[a,x,u,v] = \int phi_u (r_a-Ri_a) (-\nabla_x phi_v) dr
                s1r.append(
                    cupy.asarray(-mol_e.intor('int1e_irp', shls_slice=shlslc).
                                 reshape(3, 3, -1, mol_e.nao)))
                # s1rr[a,b,x,u,v] =
                # \int phi_u [3/2*(r_a-Ri_a)(r_b-Ri_b)-1/2*(r-Ri)^2 delta_ab] (-\nable_x phi_v) dr
                s1rr_ = cupy.asarray(-mol_e.intor('int1e_irrp', shls_slice=shlslc).
                                     reshape(3, 3, 3, -1, mol_e.nao))
                s1rr_trace = cupy.einsum('aaxuv->xuv', s1rr_)
                s1rr_ *= 3 / 2
                for k in range(3):
                    s1rr_[k,k] -= 0.5 * s1rr_trace
                s1rr.append(s1rr_)

        for jatm in range(mol_e.natm):
            p0, p1 = aoslices[jatm, 2:]

            # d E_qm_pc / d Ri with fixed ewald_pot
            qm_multipole_grad[jatm] += \
                contract('uv,xuv->x', dEds[p0:p1], s1[:,p0:p1]) \
              - contract('uv,xuv->x', dEds[:,p0:p1], s1[:,:,p0:p1])

            # d E_qm_dip / d Ri
            qm_multipole_grad[jatm] -= \
                 contract('auv,axuv->x', dEdsr[:,p0:p1], s1r[jatm])
            s1r_ = list()
            for iatm in range(mol_e.natm):
                s1r_.append(s1r[iatm][...,p0:p1])
            s1r_ = cupy.concatenate(s1r_, axis=-2)
            qm_multipole_grad[jatm] += contract('auv,axuv->x', dEdsr[...,p0:p1], s1r_)

            # d E_qm_quad / d Ri
            qm_multipole_grad[jatm] -= \
                    contract('abuv,abxuv->x', dEdsrr[:,:,p0:p1], s1rr[jatm])
            s1rr_ = list()
            for iatm in range(mol_e.natm):
                s1rr_.append(s1rr[iatm][...,p0:p1])
            s1rr_ = cupy.concatenate(s1rr_, axis=-2)
            qm_multipole_grad[jatm] += contract('abuv,abxuv->x', dEdsrr[...,p0:p1], s1rr_)

        cput1 = logger.timer(self, 'grad_ewald pulay', *cput0)
        s1 = s1r = s1rr = dEds = dEdsr = dEdsrr = None

        ew_eta, ew_cut = cell.get_ewald_params()

        # ---------------------------------------------- #
        # -------- Ewald real-space gradient ----------- #
        # ---------------------------------------------- #

        Lall = cupy.asarray(cell.get_lattice_Ls())

        from pyscf import pbc
        rmax_qm = max(cupy.linalg.norm(qm_coords - cupy.mean(qm_coords, axis=0), axis=-1))
        qm_ewovrl_grad = cupy.zeros_like(qm_coords)

        grad_Tij = lambda R, r: get_multipole_tensors_pp(R, [1,2,3], r)
        grad_kTij = lambda R, r, eta: get_multipole_tensors_pg(R, eta, [1,2,3], r)

        def grad_qm_multipole(Tija, Tijab, Tijabc,
                              qm_charges, qm_dipoles, qm_quads,
                              mm_charges):
            Tc   = contract('ijx,j->ix', Tija, mm_charges)
            res  = contract('i,ix->ix', qm_charges, Tc)
            Tc   = contract('ijxa,j->ixa', Tijab, mm_charges)
            res += contract('ia,ixa->ix', qm_dipoles, Tc)
            Tc   = contract('ijxab,j->ixab', Tijabc, mm_charges)
            res += contract('iab,ixab->ix', qm_quads, Tc) / 3
            return res

        def grad_mm_multipole(Tija, Tijab, Tijabc,
                              qm_charges, qm_dipoles, qm_quads,
                              mm_charges):
            Tc  = contract('i,ijx->jx', qm_charges, Tija)
            Tc += contract('ia,ijxa->jx', qm_dipoles, Tijab)
            Tc += contract('iab,ijxab->jx', qm_quads, Tijabc) / 3
            return contract('jx,j->jx', Tc, mm_charges)

        #------ qm - mm clasiical ewald energy gradient ------#
        all_mm_coords = (mm_coords[None,:,:] - Lall[:,None,:]).reshape(-1,3)
        all_mm_charges = cupy.hstack([mm_charges] * len(Lall))
        dist2 = all_mm_coords - cupy.mean(qm_coords, axis=0)[None]
        dist2 = contract('jx,jx->j', dist2, dist2)
        if with_mm:
            mm_ewovrl_grad = numpy.zeros_like(all_mm_coords)
        mem_avail = get_avail_mem()
        blksize = int(mem_avail/64/3/len(all_mm_coords))
        if blksize == 0:
            raise RuntimeError(f"Not enough GPU memory, mem_avail = {mem_avail}, blkszie = {blksize}")
        for i0, i1 in pyscf_lib.prange(0, mol.natm, blksize):
            R = qm_coords[i0:i1,None,:] - all_mm_coords[None,:,:]
            r = cupy.linalg.norm(R, axis=-1)
            r[r<1e-16] = cupy.inf

            # subtract the real-space Coulomb within rcut_hcore
            mask = dist2 <= cell.rcut_hcore**2
            Tija, Tijab, Tijabc = grad_Tij(R[:,mask], r[:,mask])
            qm_ewovrl_grad[i0:i1] -= grad_qm_multipole(Tija, Tijab, Tijabc,
                    qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])
            if with_mm:
                mm_ewovrl_grad[mask] += grad_mm_multipole(Tija, Tijab, Tijabc,
                        qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])

            # difference between MM gaussain charges and MM point charges
            mask = dist2 > cell.rcut_hcore**2
            zetas = cupy.asarray(cell.get_zetas())
            min_expnt = cupy.min(zetas)
            max_ewrcut = pbc.gto.cell._estimate_rcut(min_expnt, 0, 1., cell.precision)
            cut2 = (max_ewrcut + rmax_qm)**2
            mask = mask & (dist2 <= cut2)
            expnts = cupy.hstack([cupy.sqrt(zetas)] * len(Lall))[mask]
            if expnts.size != 0:
                Tija, Tijab, Tijabc = grad_kTij(R[:,mask], r[:,mask], expnts)
                qm_ewovrl_grad[i0:i1] -= grad_qm_multipole(Tija, Tijab, Tijabc,
                        qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])
                if with_mm:
                    mm_ewovrl_grad[mask] += grad_mm_multipole(Tija, Tijab, Tijabc,
                            qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])

            # ewald real-space sum
            cut2 = (ew_cut + rmax_qm)**2
            mask = dist2 <= cut2
            Tija, Tijab, Tijabc = grad_kTij(R[:,mask], r[:,mask], ew_eta)
            qm_ewovrl_grad[i0:i1] += grad_qm_multipole(Tija, Tijab, Tijabc,
                    qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])
            if with_mm:
                mm_ewovrl_grad[mask] -= grad_mm_multipole(Tija, Tijab, Tijabc,
                        qm_charges[i0:i1], qm_dipoles[i0:i1], qm_quads[i0:i1], all_mm_charges[mask])

        if with_mm:
            mm_ewovrl_grad = mm_ewovrl_grad.reshape(len(Lall), -1, 3)
            mm_ewovrl_grad = cupy.sum(mm_ewovrl_grad, axis=0)
        all_mm_coords = all_mm_charges = None

        #------ qm - qm clasiical ewald energy gradient ------#
        R = qm_coords[:,None,:] - qm_coords[None,:,:]
        r = numpy.sqrt(contract('ijx,ijx->ij', R, R))
        r[r<1e-16] = 1e100

        # subtract the real-space Coulomb within rcut_hcore
        Tija, Tijab, Tijabc = grad_Tij(R, r)
        #qm_ewovrl_grad -= cp.einsum('i,ijx,j->ix', qm_charges, Tija, qm_charges)
        #qm_ewovrl_grad += cp.einsum('i,ijxa,ja->ix', qm_charges, Tijab, qm_dipoles)
        #qm_ewovrl_grad -= cp.einsum('i,ijxa,ja->jx', qm_charges, Tijab, qm_dipoles) #
        #qm_ewovrl_grad += cp.einsum('ia,ijxab,jb->ix', qm_dipoles, Tijabc, qm_dipoles)
        #qm_ewovrl_grad -= cp.einsum('i,ijxab,jab->ix', qm_charges, Tijabc, qm_quads) / 3
        #qm_ewovrl_grad += cp.einsum('i,ijxab,jab->jx', qm_charges, Tijabc, qm_quads) / 3 #
        temp = contract('ijx,j->ix', Tija, qm_charges)
        qm_ewovrl_grad -= contract('i,ix->ix', qm_charges, temp)
        temp = contract('ijxa,ja->ix', Tijab, qm_dipoles)
        qm_ewovrl_grad += contract('i,ix->ix', qm_charges, temp)
        temp = contract('i,ijxa->jxa', qm_charges, Tijab)
        qm_ewovrl_grad -= contract('jxa,ja->jx', temp, qm_dipoles) #
        temp = contract('ijxab,jb->ixa', Tijabc, qm_dipoles)
        qm_ewovrl_grad += contract('ia,ixa->ix', qm_dipoles, temp)
        temp = contract('ijxab,jab->ix', Tijabc, qm_quads)
        qm_ewovrl_grad -= contract('i,ix->ix', qm_charges, temp) / 3
        temp = contract('ijxab,jab->ijx', Tijabc, qm_quads)
        qm_ewovrl_grad += contract('i,ijx->jx', qm_charges, temp) / 3 #
        temp = None

        # ewald real-space sum
        # NOTE here I assume ewald real-space sum is over all qm images
        # consistent with mm_mole.get_ewald_pot
        R = (R[:,:,None,:] - Lall[None,None]).reshape(len(qm_coords), len(Lall)*len(qm_coords), 3)
        r = numpy.sqrt(contract('ijx,ijx->ij', R, R))
        r[r<1e-16] = 1e100
        Tija, Tijab, Tijabc = grad_kTij(R, r, ew_eta)
        Tija = Tija.reshape(len(qm_coords), len(qm_coords), len(Lall), 3)
        Tijab = Tijab.reshape(len(qm_coords), len(qm_coords), len(Lall), 3, 3)
        Tijabc = Tijabc.reshape(len(qm_coords), len(qm_coords), len(Lall), 3, 3, 3)
        #qm_ewovrl_grad += cp.einsum('i,ijLx,j->ix', qm_charges, Tija, qm_charges)
        #qm_ewovrl_grad -= cp.einsum('i,ijLxa,ja->ix', qm_charges, Tijab, qm_dipoles)
        #qm_ewovrl_grad += cp.einsum('i,ijLxa,ja->jx', qm_charges, Tijab, qm_dipoles) #
        #qm_ewovrl_grad -= cp.einsum('ia,ijLxab,jb->ix', qm_dipoles, Tijabc, qm_dipoles)
        #qm_ewovrl_grad += cp.einsum('i,ijLxab,jab->ix', qm_charges, Tijabc, qm_quads) / 3
        #qm_ewovrl_grad -= cp.einsum('i,ijLxab,jab->jx', qm_charges, Tijabc, qm_quads) / 3 #
        temp = contract('ijLx,j->ix', Tija, qm_charges)
        qm_ewovrl_grad += contract('i,ix->ix', qm_charges, temp)
        temp = contract('ijLxa,ja->ix', Tijab, qm_dipoles)
        qm_ewovrl_grad -= contract('i,ix->ix', qm_charges, temp)
        temp = contract('i,ijLxa->jxa', qm_charges, Tijab)
        qm_ewovrl_grad += contract('jxa,ja->jx', temp, qm_dipoles) #
        temp = contract('ijLxab,jb->ixa', Tijabc, qm_dipoles)
        qm_ewovrl_grad -= contract('ia,ixa->ix', qm_dipoles, temp)
        temp = contract('ijLxab,jab->ix', Tijabc, qm_quads)
        qm_ewovrl_grad += contract('i,ix->ix', qm_charges, temp) / 3
        temp = contract('i,ijLxab->jxab', qm_charges, Tijabc)
        qm_ewovrl_grad -= contract('jxab,jab->jx', temp, qm_quads) / 3 #

        cput2 = logger.timer(self, 'grad_ewald real-space', *cput1)

        # ---------------------------------------------- #
        # ---------- Ewald k-space gradient ------------ #
        # ---------------------------------------------- #

        mesh = cell.mesh
        Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
        Gv = cupy.asarray(Gv)
        absG2 = contract('gi,gi->g', Gv, Gv)
        absG2[absG2==0] = 1e200
        coulG = 4*numpy.pi / absG2
        coulG *= weights
        Gpref = cupy.exp(-absG2/(4*ew_eta**2)) * coulG

        GvRmm = contract('gx,ix->ig', Gv, mm_coords)
        cosGvRmm = cupy.cos(GvRmm)
        sinGvRmm = cupy.sin(GvRmm)
        zcosGvRmm = contract("i,ig->g", mm_charges, cosGvRmm)
        zsinGvRmm = contract("i,ig->g", mm_charges, sinGvRmm)

        GvRqm = contract('gx,ix->ig', Gv, qm_coords)
        cosGvRqm = cupy.cos(GvRqm)
        sinGvRqm = cupy.sin(GvRqm)
        zcosGvRqm = contract("i,ig->g", qm_charges, cosGvRqm)
        zsinGvRqm = contract("i,ig->g", qm_charges, sinGvRqm)
        #DGcosGvRqm = cp.einsum("ia,ga,ig->g", qm_dipoles, Gv, cosGvRqm)
        #DGsinGvRqm = cp.einsum("ia,ga,ig->g", qm_dipoles, Gv, sinGvRqm)
        #TGGcosGvRqm = cp.einsum("iab,ga,gb,ig->g", qm_quads, Gv, Gv, cosGvRqm)
        #TGGsinGvRqm = cp.einsum("iab,ga,gb,ig->g", qm_quads, Gv, Gv, sinGvRqm)
        temp = contract('ia,ig->ag', qm_dipoles, cosGvRqm)
        DGcosGvRqm = contract('ga,ag->g', Gv, temp)
        temp = contract('ia,ig->ag', qm_dipoles, sinGvRqm)
        DGsinGvRqm = contract('ga,ag->g', Gv, temp)
        temp = contract('iab,ig->abg', qm_quads, cosGvRqm)
        temp = contract('abg,ga->bg', temp, Gv)
        TGGcosGvRqm = contract('gb,bg->g', Gv, temp)
        temp = contract('iab,ig->abg', qm_quads, sinGvRqm)
        temp = contract('abg,ga->bg', temp, Gv)
        TGGsinGvRqm = contract('gb,bg->g', Gv, temp)

        qm_ewg_grad = cupy.zeros_like(qm_coords)
        if with_mm:
            mm_ewg_grad = cupy.zeros_like(mm_coords)

        # qm pc - mm pc
        #p = ['einsum_path', (3, 4), (1, 3), (1, 2), (0, 1)]
        #qm_ewg_grad -= cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, sinGvRqm, zcosGvRmm, Gpref, optimize=p)
        #qm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, cosGvRqm, zsinGvRmm, Gpref, optimize=p)
        temp = contract('g,g->g', zcosGvRmm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', sinGvRqm, temp)
        qm_ewg_grad -= contract('i,ix->ix', qm_charges, temp)
        temp = contract('g,g->g', zsinGvRmm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', cosGvRqm, temp)
        qm_ewg_grad += contract('i,ix->ix', qm_charges, temp)
        if with_mm:
            #p = ['einsum_path', (0, 2), (1, 2), (0, 2), (0, 1)]
            #mm_ewg_grad -= cp.einsum('i,gx,ig,g,g->ix', mm_charges, Gv, sinGvRmm, zcosGvRqm, Gpref, optimize=p)
            #mm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', mm_charges, Gv, cosGvRmm, zsinGvRqm, Gpref, optimize=p)
            temp = contract('i,ig->gi', mm_charges, sinGvRmm)
            temp2 = contract('g,g->g', zcosGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad -= contract('gi,gx->ix', temp, temp2)
            temp = contract('i,ig->gi', mm_charges, cosGvRmm)
            temp2 = contract('g,g->g', zsinGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad += contract('gi,gx->ix', temp, temp2)
        # qm dip - mm pc
        #p = ['einsum_path', (4, 5), (1, 4), (0, 1), (0, 2), (0, 1)]
        #qm_ewg_grad -= cp.einsum('ia,gx,ga,ig,g,g->ix', qm_dipoles, Gv, Gv, sinGvRqm, zsinGvRmm, Gpref, optimize=p)
        #qm_ewg_grad -= cp.einsum('ia,gx,ga,ig,g,g->ix', qm_dipoles, Gv, Gv, cosGvRqm, zcosGvRmm, Gpref, optimize=p)
        temp = contract('g,g->g', zsinGvRmm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp2 = contract('ia,ga->gi', qm_dipoles, Gv)
        temp2 = contract('ig,gi->ig', sinGvRqm, temp2)
        qm_ewg_grad -= contract('gx,ig->ix', temp, temp2)
        temp = contract('g,g->g', zcosGvRmm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp2 = contract('ia,ga->gi', qm_dipoles, Gv)
        temp2 = contract('ig,gi->ig', cosGvRqm, temp2)
        qm_ewg_grad -= contract('gx,ig->ix', temp, temp2)
        if with_mm:
            #p = ['einsum_path', (1, 3), (0, 2), (0, 2), (0, 1)]
            #mm_ewg_grad += cp.einsum('g,j,gx,jg,g->jx', DGcosGvRqm, mm_charges, Gv, cosGvRmm, Gpref, optimize=p)
            #mm_ewg_grad += cp.einsum('g,j,gx,jg,g->jx', DGsinGvRqm, mm_charges, Gv, sinGvRmm, Gpref, optimize=p)
            temp = contract('j,jg->gj', mm_charges, cosGvRmm)
            temp2 = contract('g,g->g', DGcosGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad += contract('gj,gx->jx', temp, temp2)
            temp = contract('j,jg->gj', mm_charges, sinGvRmm)
            temp2 = contract('g,g->g', DGsinGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad += contract('gj,gx->jx', temp, temp2)
        # qm quad - mm pc
        #p = ['einsum_path', (5, 6), (0, 5), (0, 2), (2, 3), (1, 2), (0, 1)]
        #qm_ewg_grad += cp.einsum('ga,gb,iab,gx,ig,g,g->ix', Gv, Gv, qm_quads, Gv, sinGvRqm, zcosGvRmm, Gpref, optimize=p) / 3
        #qm_ewg_grad -= cp.einsum('ga,gb,iab,gx,ig,g,g->ix', Gv, Gv, qm_quads, Gv, cosGvRqm, zsinGvRmm, Gpref, optimize=p) / 3
        temp = contract('g,g->g', zcosGvRmm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp2 = contract('gb,gx->gbx', Gv, Gv)
        temp = contract('ag,gbx->abgx', temp, temp2)
        temp = contract('ig,abgx->iabx', sinGvRqm, temp)
        qm_ewg_grad += contract('iab,iabx->ix', qm_quads, temp) / 3
        temp = contract('g,g->g', zsinGvRmm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp2 = contract('gb,gx->gbx', Gv, Gv)
        temp = contract('ag,gbx->abgx', temp, temp2)
        temp = contract('ig,abgx->iabx', cosGvRqm, temp)
        qm_ewg_grad -= contract('iab,iabx->ix', qm_quads, temp) / 3
        if with_mm:
            #p = ['einsum_path', (1, 3), (0, 2), (0, 2), (0, 1)]
            #mm_ewg_grad += cp.einsum('g,j,gx,jg,g->jx', TGGcosGvRqm, mm_charges, Gv, sinGvRmm, Gpref, optimize=p) / 3
            #mm_ewg_grad -= cp.einsum('g,j,gx,jg,g->jx', TGGsinGvRqm, mm_charges, Gv, cosGvRmm, Gpref, optimize=p) / 3
            temp = contract('j,jg->gj', mm_charges, sinGvRmm)
            temp2 = contract('g,g->g', TGGcosGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad += contract('gj,gx->jx', temp, temp2) / 3
            temp = contract('j,jg->gj', mm_charges, cosGvRmm)
            temp2 = contract('g,g->g', TGGsinGvRqm, Gpref)
            temp2 = contract('gx,g->gx', Gv, temp2)
            mm_ewg_grad -= contract('gj,gx->jx', temp, temp2) / 3

        # qm pc - qm pc
        #p = ['einsum_path', (3, 4), (1, 3), (1, 2), (0, 1)]
        #qm_ewg_grad -= cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, sinGvRqm, zcosGvRqm, Gpref, optimize=p)
        #qm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, cosGvRqm, zsinGvRqm, Gpref, optimize=p)
        temp = contract('g,g->g', zcosGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', sinGvRqm, temp)
        qm_ewg_grad -= contract('i,ix->ix', qm_charges, temp)
        temp = contract('g,g->g', zsinGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', cosGvRqm, temp)
        qm_ewg_grad += contract('i,ix->ix', qm_charges, temp)
        # qm pc - qm dip
        #qm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, cosGvRqm, DGcosGvRqm, Gpref, optimize=p)
        #qm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, sinGvRqm, DGsinGvRqm, Gpref, optimize=p)
        temp = contract('g,g->g', DGcosGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', cosGvRqm, temp)
        qm_ewg_grad += contract('i,ix->ix', qm_charges, temp)
        temp = contract('g,g->g', DGsinGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', sinGvRqm, temp)
        qm_ewg_grad += contract('i,ix->ix', qm_charges, temp)
        #p = ['einsum_path', (3, 5), (1, 4), (1, 3), (1, 2), (0, 1)]
        #qm_ewg_grad -= cp.einsum('ja,ga,gx,g,jg,g->jx', qm_dipoles, Gv, Gv, zsinGvRqm, sinGvRqm, Gpref, optimize=p)
        #qm_ewg_grad -= cp.einsum('ja,ga,gx,g,jg,g->jx', qm_dipoles, Gv, Gv, zcosGvRqm, cosGvRqm, Gpref, optimize=p)
        temp = contract('g,g->g', zsinGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp = contract('gx,ag->axg', Gv, temp)
        temp = contract('jg,axg->ajx', sinGvRqm, temp)
        qm_ewg_grad -= contract('ja,ajx->jx', qm_dipoles, temp)
        temp = contract('g,g->g', zcosGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp = contract('gx,ag->axg', Gv, temp)
        temp = contract('jg,axg->ajx', cosGvRqm, temp)
        qm_ewg_grad -= contract('ja,ajx->jx', qm_dipoles, temp)
        # qm dip - qm dip
        #p = ['einsum_path', (4, 5), (1, 4), (1, 3), (1, 2), (0, 1)]
        #qm_ewg_grad -= cp.einsum('ia,ga,gx,ig,g,g->ix', qm_dipoles, Gv, Gv, sinGvRqm, DGcosGvRqm, Gpref, optimize=p)
        #qm_ewg_grad += cp.einsum('ia,ga,gx,ig,g,g->ix', qm_dipoles, Gv, Gv, cosGvRqm, DGsinGvRqm, Gpref, optimize=p)
        temp = contract('g,g->g', DGcosGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp = contract('gx,ag->axg', Gv, temp)
        temp = contract('ig,axg->aix', sinGvRqm, temp)
        qm_ewg_grad -= contract('ia,aix->ix', qm_dipoles, temp)
        temp = contract('g,g->g', DGsinGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp = contract('gx,ag->axg', Gv, temp)
        temp = contract('ig,axg->aix', cosGvRqm, temp)
        qm_ewg_grad += contract('ia,aix->ix', qm_dipoles, temp)
        # qm pc - qm quad
        #p = ['einsum_path', (3, 4), (1, 3), (1, 2), (0, 1)]
        #qm_ewg_grad += cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, sinGvRqm, TGGcosGvRqm, Gpref, optimize=p) / 3
        #qm_ewg_grad -= cp.einsum('i,gx,ig,g,g->ix', qm_charges, Gv, cosGvRqm, TGGsinGvRqm, Gpref, optimize=p) / 3
        temp = contract('g,g->g', TGGcosGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', sinGvRqm, temp)
        qm_ewg_grad += contract('i,ix->ix', qm_charges, temp) / 3
        temp = contract('g,g->g', TGGsinGvRqm, Gpref)
        temp = contract('gx,g->gx', Gv, temp)
        temp = contract('ig,gx->ix', cosGvRqm, temp)
        qm_ewg_grad -= contract('i,ix->ix', qm_charges, temp) / 3
        #p = ['einsum_path', (4, 6), (1, 5), (1, 2), (2, 3), (1, 2), (0, 1)]
        #qm_ewg_grad += cp.einsum('jab,ga,gb,gx,g,jg,g->jx', qm_quads, Gv, Gv, Gv, zcosGvRqm, sinGvRqm, Gpref, optimize=p) / 3
        #qm_ewg_grad -= cp.einsum('jab,ga,gb,gx,g,jg,g->jx', qm_quads, Gv, Gv, Gv, zsinGvRqm, cosGvRqm, Gpref, optimize=p) / 3
        temp = contract('g,g->g', zcosGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp2 = contract('gb,gx->bgx', Gv, Gv)
        temp = contract('ag,bgx->abgx', temp, temp2)
        temp = contract('jg,abgx->abjx', sinGvRqm, temp)
        qm_ewg_grad += contract('jab,abjx->jx', qm_quads, temp) / 3
        temp = contract('g,g->g', zsinGvRqm, Gpref)
        temp = contract('ga,g->ag', Gv, temp)
        temp2 = contract('gb,gx->bgx', Gv, Gv)
        temp = contract('ag,bgx->abgx', temp, temp2)
        temp = contract('jg,abgx->abjx', cosGvRqm, temp)
        qm_ewg_grad -= contract('jab,abjx->jx', qm_quads, temp) / 3

        logger.timer(self, 'grad_ewald k-space', *cput2)
        logger.timer(self, 'grad_ewald', *cput0)
        if not with_mm:
            return (qm_multipole_grad + qm_ewovrl_grad + qm_ewg_grad).get()
        else:
            return (qm_multipole_grad + qm_ewovrl_grad + qm_ewg_grad).get(), \
                   (mm_ewovrl_grad + mm_ewg_grad).get()

    grad_int = grad_int
    grad_hcore_mm = grad_hcore_mm
    grad_nuc_mm = grad_nuc_mm

    def grad_mm(self, dm=None, mol=None):
        g_mm = self.grad_hcore_mm(dm, mol) + self.grad_nuc_mm(mol)
        if self.base.mol.mm_mol is not None:
            g_mm += self.de_ewald_mm
        return g_mm

    def _finalize(self):
        if self.base.mol.mm_mol is not None:
            g_ewald_qm, self.de_ewald_mm = self.grad_ewald(with_mm=True)
            self.de += g_ewald_qm
        super()._finalize()


    as_scanner = as_scanner

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def to_gpu(self):
        raise NotImplementedError

Grad = Gradients

# Inject to CDFT class
neo.cdft.CDFT.Gradients = pyscf_lib.class_as_method(Gradients)

if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', nuc_basis='pb4d', verbose=5)
    mf = neo.CDFT(mol, xc='PBE')
    mf.scf()
    mf.nuc_grad_method().grad()
