import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cusparse
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
from gamma.simulator.func import elastic_stiff_matrix, constitutive_problem, transformation, disp_match
cp.cuda.Device(3).use()
import pyvista as pv
import vtk
import os

os.makedirs("results_box", exist_ok=True)

def cg_gpu(A, b, x0=None, tol=1e-8, maxiter=1000):
    """Conjugate Gradient solver using CuPy only (for GPU sparse matrix A)"""
    n = b.size
    if x0 is None:
        x = cp.zeros_like(b)
    else:
        x = x0

    r = b - A @ x
    p = r.copy()
    rs_old = cp.inner(r, r)

    for i in range(maxiter):
        Ap = A @ p
        alpha = rs_old / cp.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = cp.inner(r, r)
        if cp.sqrt(rs_new) < tol:
            return x, 0  # converged
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, 1  # did not converge


def save_vtk(filename):
    n_e_save = cp.sum(domain.active_elements)
    n_n_save = cp.sum(domain.active_nodes)
    active_elements = domain.elements[domain.active_elements].tolist()
    active_cells = np.array([item for sublist in active_elements for item in [8] + sublist])
    active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements))
    points = domain.nodes[0:n_n_save].get() + 5 * U[0:n_n_save].get()
    Sv = transformation(cp.sqrt(1 / 2 * ((S[0:n_e_save, :, 0] - S[0:n_e_save, :, 1]) ** 2 +
                                         (S[0:n_e_save, :, 1] - S[0:n_e_save, :, 2]) ** 2 +
                                         (S[0:n_e_save, :, 2] - S[0:n_e_save, :, 0]) ** 2 +
                                         6 * (S[0:n_e_save, :, 3] ** 2 + S[0:n_e_save, :, 4] ** 2 + S[0:n_e_save, :, 5] ** 2))),
                      domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S11 = transformation(S[0:n_e_save, :, 0], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S22 = transformation(S[0:n_e_save, :, 1], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S33 = transformation(S[0:n_e_save, :, 2], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S12 = transformation(S[0:n_e_save, :, 3], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S23 = transformation(S[0:n_e_save, :, 4], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    S13 = transformation(S[0:n_e_save, :, 5], domain.elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)

    active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
    active_grid.point_data['temp'] = heat_solver.temperature[0:n_n_save].get()
    active_grid.point_data['S_von'] = Sv.get()
    active_grid.point_data['S11'] = S11.get()
    active_grid.point_data['S22'] = S22.get()
    active_grid.point_data['S33'] = S33.get()
    active_grid.point_data['S12'] = S12.get()
    active_grid.point_data['S23'] = S23.get()
    active_grid.point_data['S13'] = S13.get()
    active_grid.point_data['U1'] = U[0:n_n_save, 0].get()
    active_grid.point_data['U2'] = U[0:n_n_save, 1].get()
    active_grid.point_data['U3'] = U[0:n_n_save, 2].get()
    active_grid.save(filename)

# 경로 변경된 부분
domain = domain_mgr(filename='box.k', toolpathdir='box_toolpath.crs')
heat_solver = heat_solve_mgr(domain)
endtime = float(domain.toolpath[:, 0].max())
n_n = len(domain.nodes)
n_e = len(domain.elements)
n_q = 8
n_int = n_e * n_q
file_num = 0

poisson = 0.3
a1 = 10000
young1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Young_Debroy.txt')[:,1]/1e6)
temp_young1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Young_Debroy.txt')[:,0])
Y1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Yield_Debroy.txt')[:,1]/1e6*np.sqrt(2/3))
temp_Y1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Yield_Debroy.txt')[:,0])
scl1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Alpha_Debroy.txt')[:,1])
temp_scl1 = cp.array(np.loadtxt('/home/ftk3187/github/GAMMA/DED_GAMMA_solver/examples/0_properties/TI64_Alpha_Debroy.txt')[:,0])

T_Ref = domain.ambient

E = cp.zeros((n_e, n_q, 6))
S = cp.zeros((n_e, n_q, 6))
Ep_prev = cp.zeros((n_e, n_q, 6))
Hard_prev = cp.zeros((n_e, n_q, 6))
U = cp.zeros((n_n, 3))
dU = cp.zeros((n_n, 3))
F = cp.zeros((n_n, 3))
f = cp.zeros((n_n, 3))
alpha_Th = cp.zeros((n_e, n_q, 6))
idirich = cp.array(domain.nodes[:, 2] == -15.0)
n_e_old = cp.sum(domain.element_birth < 1e-5)
n_n_old = cp.sum(domain.node_birth < 1e-5)

nodes_pos = domain.nodes[domain.elements]
Jac = cp.matmul(domain.Bip_ele, nodes_pos[:, cp.newaxis, :, :].repeat(8, axis=1))
ele_detJac = cp.linalg.det(Jac)

tol = 1.0e-6
Maxit = 100

t = 0
last_mech_time = 0
output_timestep = 10
save_vtk(f'results_box/box_{file_num}.vtk')
file_num += 1

while domain.current_sim_time < endtime - domain.dt:

    t += 1
    heat_solver.time_integration()
    if t % 5000 == 0:
        cp.get_default_memory_pool().free_all_blocks()
        print(f"Current time: {domain.current_sim_time}, Done: {100 * domain.current_sim_time / domain.end_time:.2f}%")

    n_e_active = cp.sum(domain.element_birth < domain.current_sim_time)
    n_n_active = cp.sum(domain.node_birth < domain.current_sim_time)

    implicit_timestep = 0.1 if heat_solver.laser_state == 0 and n_e_active == n_e_old else 0.05

    if domain.current_sim_time >= last_mech_time + implicit_timestep:
        active_eles = domain.elements[0:n_e_active]
        active_nodes = domain.nodes[0:n_n_active]

        if n_n_active > n_n_old:
            if domain.nodes[n_n_old:n_n_active, 2].max() > domain.nodes[0:n_n_old, 2].max():
                U = disp_match(domain.nodes, U, n_n_old, n_n)

        temperature_nodes = heat_solver.temperature[domain.elements]
        temperature_ip = (domain.Nip_ele[:, cp.newaxis, :] @ temperature_nodes[:, cp.newaxis, :, cp.newaxis].repeat(8, axis=1))[:, :, 0, 0]
        temperature_ip = cp.clip(temperature_ip, 300, 2300)

        Q = cp.zeros(domain.nodes.shape, dtype=bool)
        Q[0:n_n_active, :] = 1
        Q[idirich, :] = 0

        young = cp.interp(temperature_ip, temp_young1, young1)
        shear = young / (2 * (1 + poisson))
        bulk = young / (3 * (1 - 2 * poisson))
        scl = cp.interp(temperature_ip, temp_scl1, scl1)
        a = a1 * cp.ones_like(young)
        alpha_Th[:, :, 0:3] = scl[:, :, cp.newaxis].repeat(3, axis=2)
        Y = cp.interp(temperature_ip, temp_Y1, Y1)

        K_elast, B, D_elast, _, _, iD, jD, ele_detJac = elastic_stiff_matrix(active_eles, active_nodes, domain.Bip_ele, shear[0:n_e_active], bulk[0:n_e_active])

        for beta in [1.0, 0.5, 0.3, 0.1]:
            U_it = U[0:n_n_active]
            for it in range(Maxit):
                E[0:n_e_active] = cp.reshape(B @ U_it.flatten(), (-1, 8, 6))
                E[0:n_e_active] -= (temperature_ip[0:n_e_active, :, cp.newaxis].repeat(6, axis=2) - T_Ref) * alpha_Th[0:n_e_active]
                S, DS, IND_p, _, _ = constitutive_problem(E[0:n_e_active], Ep_prev[0:n_e_active], Hard_prev[0:n_e_active], shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active])
                vD = ele_detJac[:, :, cp.newaxis, cp.newaxis].repeat(6, axis=2).repeat(6, axis=3) * DS
                D_p = cusparse.csr_matrix((cp.ndarray.flatten(vD), (cp.ndarray.flatten(iD), cp.ndarray.flatten(jD))), shape=D_elast.shape, dtype=cp.float_)
                K_tangent = K_elast + B.T @ (D_p - D_elast) @ B
                F = B.T @ ((ele_detJac[:, :, cp.newaxis] * S).reshape(-1))
                # dU[Q], error = cusparse.linalg.cg(K_tangent[Q[0:n_n_active].flatten()][:, Q[0:n_n_active].flatten()], -F[Q[0:n_n_active].flatten()], tol=tol)
                dU_sol, error = cg_gpu(
                      K_tangent[Q[0:n_n_active].flatten()][:, Q[0:n_n_active].flatten()],
                      -F[Q[0:n_n_active].flatten()],
                      tol=tol,
                      maxiter=1000
                      )
                dU[Q] = dU_sol


                U_new = U_it + beta * dU[0:n_n_active]
                q1 = beta**2 * dU[0:n_n_active].flatten() @ K_elast @ dU[0:n_n_active].flatten()
                q2 = U_it.flatten() @ K_elast @ U_it.flatten()
                q3 = U_new.flatten() @ K_elast @ U_new.flatten()
                criterion = 0 if q2 == 0 and q3 == 0 else q1 / (q2 + q3)
                print(f"  stopping criterion = {criterion:.2e}")
                U_it = cp.array(U_new)
                if criterion < tol:
                    print("F =", cp.linalg.norm(F[Q[0:n_n_active].flatten()]))
                    break
                print(f"  Newton it={it}, beta={beta:.2f}, criterion={criterion:.2e}")
            else:
                continue
            break
        else:
            raise Exception(f'Newton solver did not converge at timestep {t}')

        U[0:n_n_active] = U_it
        E[0:n_e_active] = cp.reshape(B @ U_it.flatten(), (-1, 8, 6))
        E[0:n_e_active] -= (temperature_ip[0:n_e_active, :, cp.newaxis].repeat(6, axis=2) - T_Ref) * alpha_Th[0:n_e_active]
        S, DS, IND_p, Ep, Hard = constitutive_problem(E[0:n_e_active], Ep_prev[0:n_e_active], Hard_prev[0:n_e_active], shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active])
        Ep_prev[0:n_e_active] = Ep
        Hard_prev[0:n_e_active] = Hard
        n_e_old = n_e_active
        n_n_old = n_n_active
        last_mech_time = domain.current_sim_time

        if domain.current_sim_time >= file_num * output_timestep:
            save_vtk(f'results_box/box_{file_num}.vtk')
            file_num += 1

