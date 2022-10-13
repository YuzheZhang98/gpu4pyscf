/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GPU4PYSCF_RYS_ROOTS_CUH
#define GPU4PYSCF_RYS_ROOTS_CUH

__device__ void GINTrys_root2(double x, double *rw);
__device__ void GINTrys_root3(double x, double *rw);
__device__ void GINTrys_root4(double x, double *rw);
__device__ void GINTrys_root5(double x, double *rw);

#endif //GPU4PYSCF_RYS_ROOTS_CUH