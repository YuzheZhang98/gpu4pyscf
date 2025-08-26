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
from gpu4pyscf.lib.cupy_helper import tag_array, contract, ensure_numpy
from pyscf.scf.jk import get_jk
from pyscf import gto
from pyscf import lib as pyscf_lib


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
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            if mm_mol.charge_model == 'gaussian':
                expnts = mm_mol.get_zetas()
                fakemol = gto.fakemol_for_charges(coords, expnts)

                intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
                intopt.build(self.base.direct_scf_tol, diag_block_with_triu=True, aosym=False,
                                group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
                v = cupy.zeros((3, mol.nao, mol.nao))
                for i0,i1,j0,j1,k0,k1,j3c in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
                    v[:,i0:i1,j0:j1] += contract('xkji,k->xij', j3c, cupy.asarray(charges[k0:k1]))
                v = intopt.unsort_orbitals(v, axis=[1,2])
            else:
                nao = mol.nao
                max_memory = mol.super_mol.max_memory - pyscf_lib.current_memory()[0]
                blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
                blksize = max(blksize, 1)
                v = 0
                for i0, i1 in pyscf_lib.prange(0, charges.size, blksize):
                    j3c = mol.intor('int1e_grids_ip', grids=coords[i0:i1])
                    v += contract('ikpq,k->ipq', j3c, charges[i0:i1])
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

    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    g = numpy.zeros_like(coords)
    mf = mf_grad.base
    if dm is None:
        dm = mf.make_rdm1()

    # Handle each charged component's interaction with MM
    for t, comp in mf.components.items():
        mol_comp = comp.mol
        dm_comp = dm[t]
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            fakemol = gto.fakemol_for_charges(coords, expnts)
            intopt = int3c2e.VHFOpt(mol_comp, fakemol, 'int2e')
            intopt.build(mf_grad.base.direct_scf_tol, diag_block_with_triu=True, aosym=False,
                         group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
            dm_ = intopt.sort_orbitals(dm_comp, axis=[0,1])
            for i0,i1,j0,j1,k0,k1,j3c in int3c2e.loop_int3c2e_general(intopt, ip_type='ip2'):
                j3c = contract('xkji,k->xkji', j3c, charges[k0:k1])
                g[k0:k1] += (contract('xkji,ij->kx', j3c, dm_[i0:i1,j0:j1])).get() * comp.charge
        else:
            # From examples/qmmm/30-force_on_mm_particles.py
            # The interaction between electron density and MM particles
            # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
            #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
            for i, q in enumerate(charges):
                with mol_comp.with_rinv_origin(coords[i]):
                    v = mol_comp.intor('int1e_iprinv')
                g[i] += (contract('ij,xji->x', dm_comp, v) +
                         contract('ij,xij->x', dm_comp, v.conj())).get() \
                        * -q * comp.charge

def grad_nuc_mm(mf_grad, mol=None):
    '''Nuclear gradients of the QM-MM nuclear energy
    (in the form of point charge Coulomb interactions)
    with respect to MM atoms.
    '''
    if mol is None:
        mol = mf_grad.mol
    mm_mol = mol.mm_mol
    if mm_mol is None:
        warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
        return None
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    g_mm = numpy.zeros_like(coords)
    mol_e = mol.components['e']
    for i in range(mol_e.natm):
        q1 = mol_e.atom_charge(i)
        r1 = mol_e.atom_coord(i)
        r = pyscf_lib.norm(r1-coords, axis=1)
        g_mm += q1 * numpy.einsum('i,ix,i->ix', charges, r1-coords, 1/r**3)
    return g_mm

def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    This is different from GradientsBase.as_scanner because CNEO uses two
    layers of mole objects.

    Copied from grad.rhf.as_scanner
    '''
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
            coords = mol.mm_mol.atom_coords()
            charges = mol.mm_mol.atom_charges()
            # nuclei lattice interaction
            mol_e = mol.components['e']
            g_mm = numpy.empty((mol_e.natm,3))
            for i in range(mol_e.natm):
                q1 = mol_e.atom_charge(i)
                r1 = mol_e.atom_coord(i)
                r = pyscf_lib.norm(r1-coords, axis=1)
                g_mm[i] = -q1 * numpy.einsum('i,ix,i->x', charges, r1-coords, 1/r**3)
            if atmlst is not None:
                g_mm = g_mm[atmlst]
        else:
            g_mm = 0
        return g_qm + g_mm

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

        # Get gradient from each component
        de = 0
        for t, comp in self.components.items():
            if self.grid_response is not None and isinstance(comp.base, hf.KohnShamDFT):
                comp.grid_response = self.grid_response
            de += comp.grad_elec(mo_energy=mo_energy[t], mo_coeff=mo_coeff[t],
                                 mo_occ=mo_occ[t], atmlst=atmlst)

        # Add inter-component interaction gradient
        de += self.grad_int(mo_energy, mo_coeff, mo_occ, atmlst)

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

    grad_int = grad_int
    grad_hcore_mm = grad_hcore_mm
    grad_nuc_mm = grad_nuc_mm

    def grad_mm(self, dm=None, mol=None):
        return self.grad_hcore_mm(dm, mol) + self.grad_nuc_mm(mol)

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
