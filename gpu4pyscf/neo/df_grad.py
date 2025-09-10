#!/usr/bin/env python

'''
Analytic gradient for density-fitting Coulomb interaction in CNEO-DFT
'''

from concurrent.futures import ThreadPoolExecutor
import cupy
import numpy
from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf.neo import grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.lib.cupy_helper import contract, ensure_numpy, reduce_to_device
from gpu4pyscf.df.int3c2e import get_int3c2e_ip_jk, VHFOpt, _split_tasks
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, num_devices


def _j_ip_task(intopt, rhoj_cart, dm_cart, task_list, device_id=0, omega=None):
    '''
    Modified from gpu4pyscf.df.grad.jk._jk_ip_task
    '''
    mol = intopt.mol
    natm = mol.natm
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, mol.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        cart_aux_loc = intopt.cart_aux_loc
        ej = ek = ejaux = ekaux = None
        rhoj_cart = cupy.asarray(rhoj_cart)
        dm_cart = cupy.asarray(dm_cart)
        ej = cupy.zeros((natm,3), order='C')
        ejaux = cupy.zeros((natm,3))

        for cp_kl_id in task_list:
            k0, k1 = cart_aux_loc[cp_kl_id], cart_aux_loc[cp_kl_id+1]
            rhoj_tmp = rhok_tmp = None
            rhoj_tmp = rhoj_cart[k0:k1]

            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip1', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            ej += ej_tmp
            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip2', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            ejaux += ej_tmp

            rhoj_tmp = rhok_tmp = ej_tmp = ek_tmp = None
            t0 = log.timer_debug1(f'calculate {cp_kl_id:3d} / {len(intopt.aux_log_qs):3d}, {k1-k0:3d} slices', *t0)
    return ej, ek, ejaux, ekaux

def get_grad_vj(with_df, mol, auxmol, rhoj_cart, dm_cart, omega=None):
    '''
    Calculate vj    = (i'j|L)(L|kl)(ij)(kl)
              vjaux = (ij|L')(L|kl)(ij)(kl)
    Modified from gpu4pyscf.df.grad.jk.get_grad_vjk
    '''
    nao_cart = dm_cart.shape[0]
    block_size = with_df.get_blksize(nao=nao_cart)

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size_aux=block_size, verbose=0)

    aux_ao_loc = numpy.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _j_ip_task, intopt, rhoj_cart, dm_cart, task_list[device_id],
                device_id=device_id, omega=omega)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    vjaux_total = []
    vkaux_total = []
    for future in futures:
        rhoj, rhok, vjaux, vkaux = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)
        vjaux_total.append(vjaux)
        vkaux_total.append(vkaux)

    rhoj = rhok = vjaux = vkaux = None
    rhoj = reduce_to_device(rhoj_total)
    vjaux = reduce_to_device(vjaux_total)
    return rhoj, vjaux

def get_rhoj(intopt, dm, cderi):
    dm = cupy.asarray(dm)
    rows = intopt.cderi_row
    cols = intopt.cderi_col
    dm_sparse = dm[rows, cols]
    dm_sparse[intopt.cderi_diag] *= .5

    rhoj = 2.0 * dm_sparse.dot(cderi.T)
    return rhoj

def get_cross_j(with_df, mol_e, mol_n, intopt_e, intopt_n, dm_e, dm_n,
                cderi_e, cderi_n, low, charge_n):
    assert (isinstance(mol_n, dict) and isinstance(intopt_n, dict) and
            isinstance(intopt_n, dict) and isinstance(dm_n, dict) and
            isinstance(cderi_n, dict) and isinstance(charge_n, dict))
    dm_e = cupy.asarray(dm_e)
    dm_e = intopt_e.sort_orbitals(dm_e, axis=[0,1])
    for t in dm_n.keys():
        dm_n[t] = cupy.asarray(dm_n[t])
        dm_n[t] = intopt_n[t].sort_orbitals(dm_n[t], axis=[0,1])

    rhoj_e = get_rhoj(intopt_e, dm_e, cderi_e)
    rhoj_n = {}
    for t in intopt_n.keys():
        rhoj_n[t] = get_rhoj(intopt_n[t], dm_n[t], cderi_n[t]) * charge_n[t]

    auxmol_e = with_df.auxmol
    int2c_e1 = cupy.asarray(auxmol_e.intor('int2c2e_ip1'))
    rhoj_cart_e = rhoj_cart_n = None
    auxslices = auxmol_e.aoslice_by_atom()
    aux_cart2sph_e = intopt_e.aux_cart2sph
    low_t = low.T.copy()

    if low.tag == 'eig':
        rhoj_e = cupy.dot(low_t.T, rhoj_e)
        for t, rhoj in rhoj_n.items():
            rhoj_n[t] = cupy.dot(low_t.T, rhoj)
    elif low.tag == 'cd':
        rhoj_e = solve_triangular(low_t, rhoj_e, lower=False, overwrite_b=True)
        for t, rhoj in rhoj_n.items():
            rhoj_n[t] = solve_triangular(low_t, rhoj, lower=False, overwrite_b=True)
    if not auxmol_e.cart:
        rhoj_cart_e = contract('pq,q->p', aux_cart2sph_e, rhoj_e)
        rhoj_cart_n = {t: contract('pq,q->p', intopt_n[t].aux_cart2sph, rhoj_n[t])
                       for t in rhoj_n.keys()}
    else:
        rhoj_cart_e = rhoj_e
        rhoj_cart_n = rhoj_n

    rhoj_e = intopt_e.unsort_orbitals(rhoj_e, aux_axis=[0])
    rhoj_n_total = 0
    for t, rhoj in rhoj_n.items():
        rhoj_n_total += intopt_n[t].unsort_orbitals(rhoj, aux_axis=[0])
    tmp = contract('xpq,q->xp', int2c_e1, rhoj_e)
    vjaux = -contract('xp,p->xp', tmp, rhoj_n_total)
    tmp = contract('xpq,q->xp', int2c_e1, rhoj_n_total)
    vjaux += -contract('xp,p->xp', tmp, rhoj_e)
    # (d/dX P|Q)
    ejaux = cupy.array([-vjaux[:,p0:p1].sum(axis=1) for p0, p1 in auxslices[:,2:]])
    rhoj_e = rhoj_n = rhoj_n_total = vjaux = tmp = low_t = int2c_e1 = None

    dm_cart_e = dm_e
    dm_cart_n = dm_n
    if not mol_e.cart:
        cart2sph_e = intopt_e.cart2sph
        dm_cart_e = cart2sph_e @ dm_e @ cart2sph_e.T

    for t in dm_n.keys():
        if not mol_n[t].cart:
            cart2sph_n = intopt_n[t].cart2sph
            dm_cart_n[t] = cart2sph_n @ dm_n[t] @ cart2sph_n.T

    rhoj_cart_n_total = 0
    for rhoj_cart in rhoj_cart_n.values():
        rhoj_cart_n_total += rhoj_cart
    ej_e, ejaux_3c_e = get_grad_vj(with_df, mol_e, auxmol_e, rhoj_cart_n_total, dm_cart_e, omega=None)
    ej_n = ejaux_3c_n = 0
    for t, dm_cart in dm_cart_n.items():
        ej_temp, ejaux_3c_temp = get_grad_vj(with_df, mol_n[t], auxmol_e, rhoj_cart_e, dm_cart, omega=None)
        ej_n, ejaux_3c_n = ej_n + ej_temp * charge_n[t], ejaux_3c_n + ejaux_3c_temp * charge_n[t]
    ej = ej_e + ej_n
    ejaux_3c = ejaux_3c_e + ejaux_3c_n
    ej_e = ej_n = ejaux_3c_e = ejaux_3c_n = ej_temp = ejaux_3c_temp = None

    ej = -2.0 * ej  # need to double here because in dft code *2 is processed elsewhere
    ejaux -= ejaux_3c
    return ensure_numpy(ej + ejaux)

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calcuate gradient for inter-component density-fitting Coulomb interactions'''
    mf = mf_grad.base
    mol = mf_grad.mol

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    de = numpy.zeros((len(atmlst), 3))

    mol_n = {}
    dm_n = {}
    intopt_n = mf.intopt_n
    cderi_n = mf.cderi_n
    charge_n = {}
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

        # 0: not ne; 1: t1 is e t2 is n; 2: t1 is n t2 is e
        check_ne = 0
        if mf_grad.base.df_ne:
            if t1 == 'e' and t2.startswith('n'):
                check_ne = 1
            elif t2 == 'e' and t1.startswith('n'):
                check_ne = 2

        if check_ne != 0:
            if check_ne == 1:
                mol_e = mol1
                mol_n[t2] = mol2
                with_df = comp1.with_df
                dm_e = dm1
                dm_n[t2] = dm2
                charge_n[t2] = comp2.charge
            else:
                mol_e = mol2
                mol_n[t1] = mol1
                with_df = comp2.with_df
                dm_e = dm2
                dm_n[t1] = dm1
                charge_n[t1] = comp1.charge

        else:
            de += grad.grad_pair_int(mol1, mol2, dm1, dm2,
                                     comp1.charge, comp2.charge, atmlst)

    if mol_n and dm_n and intopt_n and cderi_n and charge_n:
        de += get_cross_j(with_df, mol_e, mol_n, with_df.intopt, intopt_n,
                          dm_e, dm_n, with_df._cderi[0], cderi_n,
                          with_df.cd_low, charge_n)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)

    return de

class Gradients(grad.Gradients):
    '''Analtic graident for density-fitting CDFT'''

    def __init__(self, mf):
        super().__init__(mf)

    grad_int = grad_int

Grad = Gradients
