import copy
import time
from typing import List

import cvxpy as cp
import numpy as np
from scipy.special import binom

from ..cvx_mfpp_tools import get_aux_frame_idx, \
    create_bezier_cvx_norm_eq_relaxation, add_vel_acc_constr
from extensions.path_parameterization import BezierCurve, CompositeBezierCurve
from extensions.traversable_regions import TraversableRegions


def has_safe_point_at(points_sequence_order: List[np.ndarray],
                      num_iris_tot: int,
                      safe_points_lst: List[dict[str: np.ndarray]],
                      index: int,
                      frame_name:str):
    # determine the current segment based on the index
    ir_idx = index % num_iris_tot
    counter, seg_idx = 0, 0
    for dur in points_sequence_order:
        # check if the current index is within the segment
        if counter < ir_idx:
            counter += len(dur)
            seg_idx += 1
        else:
            break

    b_end_of_current_seg = ir_idx == counter
    if ir_idx == 0:
        raise ValueError('Initial positions are handled separately before here.')
    else:
        # apply desired safe points at the end of each segment
        if b_end_of_current_seg:
            # check if the current index is within the segment
            if frame_name in safe_points_lst[seg_idx].keys():
                return safe_points_lst[seg_idx][frame_name]

    return [False]

def optimize_multiple_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  traversable_regions: TraversableRegions,
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  fixed_frames=None,
                                  b_final_vel_constr=False,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  weights_rigid_link=None,
                                  b_use_knees_in_smooth_plan=True,
                                  n_points=None, **kwargs):
    if weights_rigid_link is None:
        weights_rigid_link = np.array([3500., 0.5, 10.])     # default for g1

    iris_seq = traversable_regions.IRIS_seq
    safe_points_lst = traversable_regions.safe_points_lst

    # number of frames
    n_frames = len(safe_points_lst[0].keys())

    # point segment order
    point_seg_order = []
    for dseg_dict in durations:
        point_seg_order.append(dseg_dict['torso'])

    # Problem size. Assume for now same number of boxes for all frames
    d = traversable_regions.regions[0]['A'].shape[1]
    num_iris_tot = 0
    for seg_dur in durations:
        num_iris_tot += len(seg_dur[next(iter(seg_dur))])
    D = max(alpha)

    # default number of points for Bezier curve
    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for k in range(num_iris_tot * n_frames):
        points[k] = {}
        for i in range(D + 1):
            size = (n_points - i, d)
            points[k][i] = cp.Variable(size)

    frame_list = list(safe_points_lst[0].keys())
    constraints = []

    # Loop through boxes.
    cost = 0
    continuity = {}
    frame_idx, fr_seg_k_box = 0, 0
    seg_idx, k = 0, 0
    for k in range(num_iris_tot * n_frames):
        continuity[k] = {}

        # Update frame name and number of boxes within segment/interval
        f_name = frame_list[frame_idx]
        sequenced_idx = iris_seq[f_name][seg_idx][fr_seg_k_box]
        A = traversable_regions.regions[sequenced_idx]['A']
        b = traversable_regions.regions[sequenced_idx]['b']
        b = np.reshape(b, (len(b), 1))
        b = np.repeat(b, n_points, axis=1)
        constraints.append(A @ points[k][0].T <= b)
        num_iris_current = len(iris_seq[f_name][seg_idx])

        # Enforce given positions
        if k % num_iris_tot == 0:          # initial position for each frame
            # if also a fixed frame, repeat for entire segment duration
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][:-1] == fixed_frame_pos_mat)
            else:   # assign for just the first time instant
                constraints.append(points[k][0][0] == safe_points_lst[0][f_name])   # initial position
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_iris_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])
        elif (k + 1) % num_iris_tot == 0:  # final position for each frame
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                constraints.append(points[k][0][0] == safe_pnt) # pos
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            else:
                constraints.append(points[k][0][-1] == safe_points_lst[-1][f_name])
        else:       # safe and fixed positions at other times
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                constraints.append(points[k][0][0] == safe_pnt) # pos
                if b_final_vel_constr and (k-1) % num_iris_tot != 0:
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx-1], points[k-1], constraints, False)
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-2, axis=0)
                constraints.append(points[k][0][1:-1] == fixed_frame_pos_mat)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[seg_idx][f_name][fr_seg_k_box] / h
            constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])

        # if we are in the same frame, enforce dynamics, continuity, differentiability, and cost
        if (k+1) % num_iris_tot != 0:
            # Continuity and differentiability.
            if fr_seg_k_box < num_iris_current:
                for i in range(D + 1):
                    constraints.append(points[k][i][-1] == points[k + 1][i][0])
                    if i > 0:
                        continuity[k][i] = constraints[-1]

        # Cost function
        for i, ai in alpha.items():
            h = n_points - 1 - i
            A_cost = np.zeros((h + 1, h + 1))
            for m in range(h + 1):
                for n in range(h + 1):
                    A_cost[m, n] = binom(h, m) * binom(h, n) / binom(2 * h, m + n)
            A_cost *= durations[seg_idx][f_name][fr_seg_k_box] / (2 * h + 1)
            A_cost = np.kron(A_cost, np.eye(d))
            p = cp.vec(points[k][i], order='C')
            cost += ai * cp.quad_form(p, A_cost)

        # Adjust frame name, segment and box numbers
        if (k+1) % num_iris_tot == 0:
            frame_idx += 1
            seg_idx = 0
            fr_seg_k_box = 0
        else:           # move to next segment if this is the last box
            if fr_seg_k_box == (num_iris_current - 1):
                fr_seg_k_box = 0        # reset the box count
                seg_idx += 1            # increase segment
            else:
                fr_seg_k_box += 1

    # Reachability constraints
    reach_constr = []
    if reach_region is not None:
        for fr_idx, frame_name in enumerate(frame_list):
            if frame_name == 'torso':
                continue

            coeffs = reach_region[frame_name]
            H = coeffs['H']
            d_vec = np.reshape(coeffs['d'], (len(H), 1))
            d_mat = np.repeat(d_vec, n_points, axis=1)
            for ti in range(num_iris_tot):
                # torso index
                z_t = points[0 * num_iris_tot + ti][0]

                # current frame index
                z_ee_seg = points[fr_idx * num_iris_tot + ti][0]

                # reachable constraint
                if frame_name == 'LF' or frame_name == 'RF' or frame_name == 'LH' or frame_name == 'RH':
                    reach_constr.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
                if b_use_knees_in_smooth_plan:
                    if frame_name == 'L_knee' or frame_name == 'R_knee':
                        reach_constr.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)

    # Rigid links (e.g., shin link length) constraint relaxation
    soc_constraint, cost_log_abs = [], []
    cost_log_abs_sum = 0.
    link_threshold = 0.01
    if bool(aux_frames):
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            link_length += link_threshold
            for nb in range(1, num_iris_tot-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    create_bezier_cvx_norm_eq_relaxation(link_length, link_proximal_point,
                                             link_distal_point, soc_constraint, cost_log_abs,
                                                         wi=weights_rigid_link)

        cost_log_abs_sum = -(cp.sum(cost_log_abs))

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + reach_constr + soc_constraint)
    try:
        prob.solve(solver='CLARABEL')
    except Exception as e:
        print("[MFPP Smooth WARNING] ", e)
        prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print(f'{"*" * 5} Smooth Problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=5e-1, eps_abs=5e-1)
        if prob.status == 'infeasible':
            print(f'{"*" * 5} Smooth Problem was infeasible. Retrying without reachability constraints.')
            prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
            prob.solve(solver='CLARABEL')
            if prob.status == 'infeasible':
                print('***** Smooth Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
                prob.solve(solver='SCS', eps_rel=5e-2, eps_abs=5e-2)
                if prob.status == 'infeasible':
                    print('***** Smooth (2nd Attempt) Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
                    prob.solve(solver='SCS', eps_rel=5e-1, eps_abs=5e-1)

    # check link constraints values (debug)
    if bool(aux_frames):
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)
            link_length += link_threshold
            for nb in range(1, num_iris_tot-1):
                for pnt in range(1):
                    _ = points[prox_fr_idx+nb][0][pnt]
                    _ = points[dist_fr_idx+nb][0][pnt]

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_iris_tot * n_frames):
        num_iris_current = len(iris_seq[frame_name][seg_idx])
        # move on to next segment after the current number of safe boxes
        if (fr_seg_k_box != 0) and fr_seg_k_box % num_iris_current == 0 and seg_idx != (num_iris_tot-1):
            seg_idx += 1
            fr_seg_k_box = 0

        # move on to next frame after all boxes processed for each frame
        if k != 0 and (k % num_iris_tot) == 0:
            frame_idx += 1
            frame_name = frame_list[frame_idx]
            fr_seg_k_box = 0

        b_time = a + durations[seg_idx][frame_name][fr_seg_k_box]
        beziers.append(BezierCurve(points[k][0].value, a, b_time))
        a = b_time
        fr_seg_k_box += 1
        # skip the final positions, those are assigned later
        if (k + 1) % num_iris_tot == 0:
            fr_seg_k_box = 0
            seg_idx = 0
            path.append(copy.deepcopy(CompositeBezierCurve(beziers)))
            beziers.clear()
            a = 0

    retiming_weights = {}
    cost_breakdown = {}

    print(f"[Smooth] Cost: {cost.value:.3f}")
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights
    dual_vars = {}
    dual_vars['lam_g0'] = prob.solution.dual_vars
    dual_vars['constraints_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in constraints
    )
    dual_vars['reach_constr_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in reach_constr
    )
    dual_vars['soc_constr_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in soc_constraint
    )
    dual_vars['lam_x0'] = np.zeros(prob.size_metrics.num_scalar_variables)

    return path, sol_stats, points, dual_vars


def unpack_sol_to_points(x_sol, num_iris_all_frames, n_points, D, d):
    sol_points = [None] * num_iris_all_frames
    for k in range(num_iris_all_frames):
        sol_points[k] = [None] * (D + 1)
    curr_idx = 0
    for k in range(num_iris_all_frames):
        for i in range(D + 1):
            next_idx = curr_idx + (n_points - i) * d
            sol_points[k][i] = x_sol[curr_idx:next_idx].reshape(n_points-i, d, order='F')
            curr_idx = next_idx

    return sol_points


def get_ci_from_global_ir_idx(total_iris_regions, k_ir, durations, f_name, h):
    c_seq, curr_ir_num = 0, 0
    for k in range(total_iris_regions):
        if k == k_ir:
            return durations[c_seq][f_name][curr_ir_num] / h
        else:
            if curr_ir_num < len(durations[c_seq][f_name]) - 1:
                curr_ir_num += 1
            else:
                c_seq += 1
                curr_ir_num = 0

    return durations[c_seq][f_name][curr_ir_num] / h
