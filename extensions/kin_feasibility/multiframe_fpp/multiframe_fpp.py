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
    num_iris_tot = sum(len(iris_seq[first_frame][p]) for p in range(n_phases))

    # Build per-frame, per-segment duration arrays from the polygonal solution.
    # traj layout: [frame0_pt0..frame0_pt{num_iris_tot}, frame1_pt0..., ...]
    # Each frame has (num_iris_tot + 1) d-dim points.
    n_poly_points = num_iris_tot + 1

    durations = []
    for seg_idx in range(n_phases):
        durations.append({})

    for frame_idx, frame in enumerate(frame_names):
        frame_start = frame_idx * n_poly_points * d

        # Collect all (num_iris_tot + 1) waypoints for this frame
        frame_traj = traj[frame_start: frame_start + n_poly_points * d]
        pts = frame_traj.reshape(n_poly_points, d)           # (N+1, 3)
        diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)  # (N,)

        # Guard against zero total length
        total = diffs.sum()
        if total < 1e-8:
            diffs = np.ones(num_iris_tot) / num_iris_tot

        # Normalise so that all durations across all phases sum to T
        diffs = diffs * T / diffs.sum()

        # Distribute into per-phase arrays matching iris_seq layout
        ir_offset = 0
        for seg_idx in range(n_phases):
            n_ir = len(iris_seq[frame][seg_idx])
            durations[seg_idx][frame] = diffs[ir_offset: ir_offset + n_ir]
            ir_offset += n_ir

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
