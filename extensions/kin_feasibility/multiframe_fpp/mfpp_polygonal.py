import copy
from typing import List
import scipy

import cvxpy as cp
import numpy as np

from ..cvx_mfpp_tools import get_aux_frame_idx, create_cvx_norm_eq_relaxation
from ...traversable_regions import TraversableRegions

b_debug = False

if b_debug:
    from ruamel.yaml import YAML
    import os
    import sys
    cwd = os.getcwd()
    sys.path.append(cwd)


def solve_min_reach_iris_distance(reach: dict[dict],
                                  traversable_regions: TraversableRegions,
                                  aux_frames=None,
                                  weights_rigid: np.array = None) -> tuple:
    iris_regions = traversable_regions.regions        # list[dict{'A', 'b'}]
    iris_seq = traversable_regions.IRIS_seq           # {frame: [[seg0_idxs], ...]}
    safe_points_list = traversable_regions.safe_points_lst  # list[dict{frame: pos}]

    if weights_rigid is None:
        weights_rigid = np.array([1, 0, 0])

    d = 3
    n_f = len(iris_seq)
    frame_names = list(iris_seq.keys())
    first_frame = frame_names[0]
    n_phases = len(iris_seq[first_frame])

    # Total IRIS waypoint intervals per frame (same for all frames by construction)
    num_iris_tot = sum(len(iris_seq[first_frame][p]) for p in range(n_phases))

    # Optimization variable layout:
    #   x = [frame0_pt0, ..., frame0_pt{num_iris_tot},
    #         frame1_pt0, ..., frame1_pt{num_iris_tot}, ...]
    # Each frame has (num_iris_tot + 1) d-dim points.
    x = cp.Variable(d * n_f * (num_iris_tot + 1))

    # ---- Safe-point equality constraints -----------------------------------------
    # For each frame, constrain trajectory variable at every safe-point boundary.
    contact_constr = []
    x_init_idx = 0
    for f_name in frame_names:
        for seg_idx in range(len(safe_points_list)):          # len = n_phases + 1
            if f_name in safe_points_list[seg_idx]:
                contact_constr.append(
                    x[x_init_idx: x_init_idx + d] == safe_points_list[seg_idx][f_name]
                )
            if seg_idx < len(safe_points_list) - 1:
                next_seg_len = len(iris_seq[f_name][seg_idx])
            else:
                next_seg_len = 1          # final safe-point occupies one slot
            x_init_idx += d * next_seg_len

    # ---- IRIS containment constraints --------------------------------------------
    iris_constr = {fn: [] for fn in frame_names}
    x_init_idx = 0
    for frame in frame_names:
        # Initial position: contained in the first IRIS region of phase 0
        ir_idx = iris_seq[frame][0][0]
        iris_cur = iris_regions[ir_idx]
        iris_constr[frame].append(iris_cur['A'] @ x[x_init_idx: x_init_idx + d] <= iris_cur['b'])
        x_init_idx += d

        for seg_idx in range(n_phases):
            seg_idxs = iris_seq[frame][seg_idx]
            curr_seg_len = len(seg_idxs)
            for ir_seg_count in range(curr_seg_len):
                ir_idx = seg_idxs[ir_seg_count]
                iris_cur = iris_regions[ir_idx]

                if curr_seg_len == 1:
                    A_c = iris_cur['A']
                    b_c = iris_cur['b']
                else:
                    if ir_seg_count < curr_seg_len - 1:
                        # Transition point: must lie in intersection of consecutive regions
                        ir_next_idx = seg_idxs[ir_seg_count + 1]
                        iris_next = iris_regions[ir_next_idx]
                        A_c = np.vstack([iris_cur['A'], iris_next['A']])
                        b_c = np.concatenate([iris_cur['b'], iris_next['b']])
                    else:
                        A_c = iris_cur['A']
                        b_c = iris_cur['b']

                iris_constr[frame].append(A_c @ x[x_init_idx: x_init_idx + d] <= b_c)
                x_init_idx += d

    # ---- Reachability constraints ------------------------------------------------
    reach_constr = []
    if reach is not None:
        x_curr_idx = 0
        for frame_idx, frame in enumerate(frame_names):
            for ti in range(num_iris_tot + 1):
                if frame == 'torso':
                    x_curr_idx += d
                    continue

                t_curr_idx = 0 * (num_iris_tot + 1) * d + d * ti
                z_t = x[t_curr_idx: t_curr_idx + d]

                ee_curr_idx = frame_idx * (num_iris_tot + 1) * d + d * ti
                z_ee = x[ee_curr_idx: ee_curr_idx + d]

                if frame not in reach:
                    x_curr_idx += d
                    continue
                coeffs = reach[frame]
                H = coeffs['H']
                d_vec = np.reshape(coeffs['d'], (len(H),))
                reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                x_curr_idx += d

    # ---- Rigid-link SOC constraints ---------------------------------------------
    frame_list = list(safe_points_list[0].keys())
    cost_log_abs = 0.
    soc_constraint = []
    A_soc_debug, d_soc_debug, cost_log_abs_list = [], [], []
    link_threshold = 0.05
    if aux_frames is not None:
        for aux_fr in aux_frames:
            prox_idx, dist_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot + 1)
            if not np.isnan(prox_idx):
                link_length += link_threshold
                A_soc_aux, d_soc_aux = create_cvx_norm_eq_relaxation(
                    prox_idx, dist_idx, link_length, d, num_iris_tot + 1, x)
                A_soc_debug += copy.deepcopy(A_soc_aux)
                d_soc_debug += copy.deepcopy(d_soc_aux)

        for Ai, di in zip(A_soc_debug, d_soc_debug):
            soc_constraint.append(cp.SOC(di, Ai @ x))
            for i in range(3):
                if weights_rigid[i] != 0.:
                    cost_log_abs_list.append(weights_rigid[i] * cp.log(Ai[i] @ x))
        cost_log_abs = -(cp.sum(cost_log_abs_list))

    # ---- Minimum-distance cost --------------------------------------------------
    cost = 0
    for fr in range(n_f):
        start_idx = fr * d * (num_iris_tot + 1)
        end_idx = start_idx + d * (num_iris_tot + 1)
        x_fr = x[start_idx: end_idx]
        p_fr_t = cp.reshape(x_fr, [d, num_iris_tot + 1], order='F')
        cost += cp.sum(cp.norm(p_fr_t[:, 1:] - p_fr_t[:, :-1], axis=1))

    # ---- Solve ------------------------------------------------------------------
    all_iris_constr = []
    for constr_list in iris_constr.values():
        all_iris_constr += constr_list

    prob = cp.Problem(
        cp.Minimize(cost + cost_log_abs),
        reach_constr + contact_constr + all_iris_constr + soc_constraint,
    )
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('Polygonal problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=0.05, eps_abs=0.05)

        for frame, constr_list in iris_constr.items():
            residuals = [scipy.linalg.norm(ic.residual) for ic in constr_list]
        rc_residuals = [scipy.linalg.norm(rc.residual) for rc in reach_constr]

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    if aux_frames is not None and A_soc_debug:
        for Ai in A_soc_debug:
            _ = np.linalg.norm(Ai @ traj) - (link_threshold + 0.)

    return traj, length, solver_time
