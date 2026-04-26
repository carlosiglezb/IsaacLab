import copy
from typing import List

import numpy as np

from .mfpp_polygonal import solve_min_reach_iris_distance
from .mfpp_smooth import optimize_multiple_bezier_iris
from ..planner_surface_contact import get_contact_seq_from_fixed_frames_seq
from extensions.traversable_regions import TraversableRegions


def plan_multiple_iris(traversable_regions: TraversableRegions,
                       T: float,
                       alpha: List[float],
                       verbose: bool = True,
                       A=None,
                       fixed_frames=None,
                       motion_frames_seq=None,
                       sca_robot_geometry=None,
                       env_geometry=None,
                       w_rigid=None,
                       w_rigid_poly=None,
                       b_use_knees_in_smooth_plan: bool = False,
                       b_final_vel_constr: bool = False):
    solver_stats = {}

    iris_seq = traversable_regions.IRIS_seq
    safe_pnt_lst = traversable_regions.safe_points_lst
    R = traversable_regions.reach

    # Stage 1: minimum-reach-distance polygonal solve
    traj, length, solver_time = solve_min_reach_iris_distance(
        traversable_regions, aux_frames=A, weights_rigid=w_rigid_poly)
    solver_stats['min_reach_iris_distance_cvxpy_time'] = solver_time
    if verbose:
        print(f"[Compute Time] Min. distance solve time: {solver_time}")

    alpha = {i + 1: ai for i, ai in enumerate(alpha)}

    d = 3
    frame_names = list(iris_seq.keys())
    first_frame = frame_names[0]
    n_phases = len(iris_seq[first_frame])
    n_f = len(frame_names)
    n_poly_points = round(len(traj) / n_f)
    durations = []
    ir_i = 0
    for phase_idx in range(n_phases):
        durations.append({})
        num_iris = len(iris_seq[first_frame][phase_idx])
        for frame_idx, (frame, phase_iris_lists) in enumerate(iris_seq.items()):
            ir = phase_iris_lists[phase_idx]
            # Collect boundary-point indices for all num_iris+1 waypoints of this
            # frame/phase.  Each 3-D point occupies d consecutive elements in traj.
            for b in range(num_iris + 1):
                first_idx = n_poly_points * frame_idx + (b + ir_i) * d
                last_idx = first_idx + d - 1
                if b == 0:
                    ee_traj_idx = np.linspace(first_idx, last_idx, d).astype(int)
                else:
                    ee_traj_idx = np.vstack((ee_traj_idx, np.linspace(first_idx, last_idx, d).astype(int)))

            ee_traj_change = traj[ee_traj_idx[1:]] - traj[ee_traj_idx[:-1]]
            seg_lengths = np.linalg.norm(ee_traj_change, axis=1)   # (num_iris,)
            total = seg_lengths.sum()
            if total < 1e-10:
                # Degenerate: all waypoints coincide — distribute time evenly.
                durations[phase_idx][frame] = np.full(num_iris, T / num_iris)
            else:
                durations[phase_idx][frame] = seg_lengths * (T / total)
        ir_i += num_iris

    # Stage 2: smooth Bezier solve
    parsed_contact_seq = get_contact_seq_from_fixed_frames_seq(fixed_frames)
    surface_normals_lst = motion_frames_seq.get_contact_surfaces()

    paths, sol_stats, points, dvars = optimize_multiple_bezier_iris(
        A, traversable_regions, durations, alpha,
        fixed_frames=fixed_frames,
        contact_sequence=parsed_contact_seq,
        surface_normals_lst=surface_normals_lst,
        weights_rigid_link=w_rigid,
        b_use_knees_in_smooth_plan=b_use_knees_in_smooth_plan,
        b_final_vel_constr=b_final_vel_constr,
    )
    solver_stats['multiple_bezier_iris_cvxpy_time'] = sol_stats['runtime']
    if verbose:
        print(f"[Compute Time] Bezier solve time: {sol_stats['runtime']}")

    return paths, iris_seq, points, safe_pnt_lst, solver_stats
