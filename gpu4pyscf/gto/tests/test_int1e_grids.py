# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib, gto, df
from gpu4pyscf.gto.moleintor import intor

def setUpModule():
    global mol_sph, mol_cart, grid_points, integral_threshold, density_contraction_threshold, charge_contraction_threshold
    atom = '''
O	0.0000	0.7375	-0.0528
O	0.0000	-0.7375	-0.0528
H	0.8190	0.8170	0.4220
H	-0.8190	-0.8170	0.4220
'''
    bas='def2-qzvpp'

    mol_sph = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.verbose = 0
    mol_sph.build()

    mol_cart = pyscf.M(atom=atom, basis=bas, max_memory=32000, cart=True)
    mol_cart.output = '/dev/null'
    mol_cart.verbose = 0
    mol_cart.build()

    xs = np.arange(-2.01, 2.0, 0.5)
    ys = np.arange(-2.02, 2.0, 0.5)
    zs = np.arange(-2.03, 2.0, 0.5)
    grid_points = lib.cartesian_prod([xs, ys, zs])

    # All of the following thresholds bound the max value of the corresponding matrix / tensor.
    integral_threshold = 1e-12
    density_contraction_threshold = 1e-10
    charge_contraction_threshold = 1e-12

def tearDownModule():
    global mol_sph, mol_cart, grid_points
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart, grid_points

class KnownValues(unittest.TestCase):
    '''
    Values are compared to PySCF CPU intor() function
    '''
    def test_int1e_grids_full_tensor_cart(self):
        ref_int1e = mol_cart.intor('int1e_grids', grids=grid_points)
        test_int1e = intor(mol_cart, 'int1e_grids', grid_points)
        np.testing.assert_allclose(ref_int1e, test_int1e, atol = integral_threshold)

    def test_int1e_grids_full_tensor_sph(self):
        ref_int1e = mol_sph.intor('int1e_grids', grids=grid_points)
        test_int1e = intor(mol_sph, 'int1e_grids', grid_points)
        np.testing.assert_allclose(ref_int1e, test_int1e, atol = integral_threshold)

    def test_int1e_grids_density_contracted_cart_symmetric(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_cart.nao, mol_cart.nao))
        dm = 0.5 * (dm + dm.T)
        ref_int1e_dot_D = np.einsum('pij,ij->p', mol_cart.intor('int1e_grids', grids=grid_points), dm)
        test_int1e_dot_D = intor(mol_cart, 'int1e_grids', grid_points, dm = dm)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_density_contracted_sph_symmetric(self):
        np.random.seed(12346)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        dm = 0.5 * (dm + dm.T)
        ref_int1e_dot_D = np.einsum('pij,ij->p', mol_sph.intor('int1e_grids', grids=grid_points), dm)
        test_int1e_dot_D = intor(mol_sph, 'int1e_grids', grid_points, dm = dm)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_density_contracted_cart_asymmetric(self):
        np.random.seed(12347)
        dm = np.random.uniform(-2.0, 2.0, (mol_cart.nao, mol_cart.nao))
        ref_int1e_dot_D = np.einsum('pij,ij->p', mol_cart.intor('int1e_grids', grids=grid_points), dm)
        test_int1e_dot_D = intor(mol_cart, 'int1e_grids', grid_points, dm = dm)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_density_contracted_sph_asymmetric(self):
        np.random.seed(12348)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        ref_int1e_dot_D = np.einsum('pij,ij->p', mol_sph.intor('int1e_grids', grids=grid_points), dm)
        test_int1e_dot_D = intor(mol_sph, 'int1e_grids', grid_points, dm = dm)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_charge_contracted_cart(self):
        np.random.seed(12347)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        ref_int1e_dot_q = np.einsum('pij,p->ij', mol_cart.intor('int1e_grids', grids=grid_points), charges)
        test_int1e_dot_q = intor(mol_cart, 'int1e_grids', grid_points, charges = charges)
        assert isinstance(test_int1e_dot_q, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_q, test_int1e_dot_q, atol = charge_contraction_threshold)

    def test_int1e_grids_charge_contracted_sph(self):
        np.random.seed(12348)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        ref_int1e_dot_q = np.einsum('pij,p->ij', mol_sph.intor('int1e_grids', grids=grid_points), charges)
        test_int1e_dot_q = intor(mol_sph, 'int1e_grids', grid_points, charges = charges)
        assert isinstance(test_int1e_dot_q, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_q, test_int1e_dot_q, atol = charge_contraction_threshold)

    # Range separated integrals

    def test_int1e_grids_full_tensor_omega(self):
        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        ref_int1e = mol_sph_omega.intor('int1e_grids', grids=grid_points)
        test_int1e = intor(mol_sph_omega, 'int1e_grids', grid_points)
        np.testing.assert_allclose(ref_int1e, test_int1e, atol = integral_threshold)

    def test_int1e_grids_density_contracted_omega(self):
        np.random.seed(12349)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        ref_int1e_dot_D = np.einsum('pij,ij->p', mol_sph_omega.intor('int1e_grids', grids=grid_points), dm)
        test_int1e_dot_D = intor(mol_sph_omega, 'int1e_grids', grid_points, dm = dm)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_charge_contracted_omega(self):
        np.random.seed(12349)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        ref_int1e_dot_q = np.einsum('pij,p->ij', mol_sph_omega.intor('int1e_grids', grids=grid_points), charges)
        test_int1e_dot_q = intor(mol_sph_omega, 'int1e_grids', grid_points, charges = charges)
        assert isinstance(test_int1e_dot_q, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_q, test_int1e_dot_q, atol = charge_contraction_threshold)

    # Gaussian charges

    def test_int1e_grids_full_tensor_guassian_charge(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        int3c2e = mol_sph._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph._atm, mol_sph._bas, mol_sph._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        test_int1e = intor(mol_sph, 'int1e_grids', grid_points, charge_exponents = charge_exponents)
        np.testing.assert_allclose(ref_int1e, test_int1e, atol = integral_threshold)

    def test_int1e_grids_density_contracted_guassian_charge(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        int3c2e = mol_sph._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph._atm, mol_sph._bas, mol_sph._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        ref_int1e_dot_D = np.einsum('pij,ij->p', ref_int1e, dm)
        test_int1e_dot_D = intor(mol_sph, 'int1e_grids', grid_points, dm = dm, charge_exponents = charge_exponents)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_charge_contracted_guassian_charge(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        int3c2e = mol_sph._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph._atm, mol_sph._bas, mol_sph._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        ref_int1e_dot_q = np.einsum('pij,p->ij', ref_int1e, charges)
        test_int1e_dot_q = intor(mol_sph, 'int1e_grids', grid_points, charges = charges, charge_exponents = charge_exponents)
        assert isinstance(test_int1e_dot_q, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_q, test_int1e_dot_q, atol = charge_contraction_threshold)

    def test_int1e_grids_full_tensor_guassian_charge_omega(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        int3c2e = mol_sph_omega._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph_omega._atm, mol_sph_omega._bas, mol_sph_omega._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph_omega, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        test_int1e = intor(mol_sph_omega, 'int1e_grids', grid_points, charge_exponents = charge_exponents)
        np.testing.assert_allclose(ref_int1e, test_int1e, atol = integral_threshold)

    def test_int1e_grids_density_contracted_guassian_charge_omega(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        int3c2e = mol_sph_omega._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph_omega._atm, mol_sph_omega._bas, mol_sph_omega._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph_omega, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        ref_int1e_dot_D = np.einsum('pij,ij->p', ref_int1e, dm)
        test_int1e_dot_D = intor(mol_sph_omega, 'int1e_grids', grid_points, dm = dm, charge_exponents = charge_exponents)
        assert isinstance(test_int1e_dot_D, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_D, test_int1e_dot_D, atol = density_contraction_threshold)

    def test_int1e_grids_charge_contracted_guassian_charge_omega(self):
        np.random.seed(12351)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        int3c2e = mol_sph_omega._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol_sph_omega._atm, mol_sph_omega._bas, mol_sph_omega._env, int3c2e)
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)
        ref_int1e = df.incore.aux_e2(mol_sph_omega, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        ref_int1e = ref_int1e.transpose((2,0,1))

        ref_int1e_dot_q = np.einsum('pij,p->ij', ref_int1e, charges)
        test_int1e_dot_q = intor(mol_sph_omega, 'int1e_grids', grid_points, charges = charges, charge_exponents = charge_exponents)
        assert isinstance(test_int1e_dot_q, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dot_q, test_int1e_dot_q, atol = charge_contraction_threshold)

if __name__ == "__main__":
    print("Full Tests for One Electron Coulomb Integrals")
    unittest.main()
