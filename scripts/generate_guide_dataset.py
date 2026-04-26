"""Generate the guide dataset .npz file for the knee-knocker residual-policy env.

This script is standalone (no Isaac Sim required). It:
  1. Builds axis-aligned-box IRIS polytopes from ResidualGuideTrackingEnvCfg
  2. Wraps them in a TraversableRegions object (with root-to-torso offset and
     G1 reachability halfspaces, matching test_mfpp_smooth.py conventions)
  3. Builds a MotionFrameSequencer from SAFE_PNT_LST and FIXED_FRAMES
  4. Instantiates LocomanipulationFramePlanner with aux_frames loaded from yaml
  5. Calls generate_guide_dataset, which sweeps (n_alpha × n_w_rigid) random
     (alpha, w_rigid) combinations.  Each guide independently samples a
     traversal duration T ~ U(T_min, T_max) to increase speed diversity.

Usage (from repo root):
    python scripts/generate_guide_dataset.py \\
        --save_path guide_dataset.npz \\
        --T_min 2.0 --T_max 3.0 --n_alpha 20 --n_w_rigid 10 --n_waypoints 150
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure repo root is on the path before any local-extension imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from extensions.iris.default_plans import get_on_knocker_balanced_contact_sequence
from extensions.traversable_regions import TraversableRegions
from extensions.g1_residual_control.guide_dataset import generate_guide_dataset
from extensions.g1_residual_control.g1_planner_constants import (
    IRIS_LST, IRIS_SEQ, SAFE_PNT_LST, FIXED_FRAMES, ROOT_TO_TORSO_OFFSET,
)
from extensions.kin_feasibility.locomanipulation_frame_planner import LocomanipulationFramePlanner

cwd = os.getcwd()
_REACH_DIR = os.path.join(cwd, 'extensions', 'kin_feasibility', 'reachability')
aux_frames_path = os.path.join(_REACH_DIR, 'g1_aux_frames.yaml')
reach_path = os.path.join(_REACH_DIR, 'g1_')


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------

def build_planner() -> tuple[LocomanipulationFramePlanner, dict[str, np.ndarray]]:
    """Assemble the LocomanipulationFramePlanner from g1_planner_constants."""
    # Build reachability halfspace path dict for all non-torso frames.
    ee_halfspace_params = {
        fr: reach_path + fr + '.yaml'
        for fr in IRIS_SEQ.keys()
        if fr != 'torso'
    }

    starting_pos = SAFE_PNT_LST[0]

    traversable_regions = TraversableRegions(
        IRIS_LST,
        IRIS_SEQ,
        SAFE_PNT_LST,
        convex_hull_halfspace_path_dict=ee_halfspace_params,
        root_to_torso_pos=ROOT_TO_TORSO_OFFSET,
    )

    _, MOTION_SEQ = get_on_knocker_balanced_contact_sequence(starting_pos)

    # aux_frames_path loads the G1 shin-link rigid-link descriptors from yaml,
    # matching the aux_frames used in test_mfpp_smooth.py test_with_aux_frames_*.
    planner = LocomanipulationFramePlanner(
        traversable_regions_list=traversable_regions,
        aux_frames_path=aux_frames_path,
        fixed_frames=FIXED_FRAMES,
        motion_frames_seq=MOTION_SEQ,
    )
    return planner, starting_pos


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate guide-dataset .npz for the knee-knocker residual env"
    )
    parser.add_argument("--save_path", default="guide_dataset.npz",
                        help="Output .npz file path (default: guide_dataset.npz)")
    parser.add_argument("--T_min", type=float, default=2.8,
                        help="Minimum traversal duration in seconds (default: 2.0)")
    parser.add_argument("--T_max", type=float, default=3.0,
                        help="Maximum traversal duration in seconds (default: 3.0)")
    parser.add_argument("--n_alpha", type=int, default=10,
                        help="Number of alpha samples (default: 20)")
    parser.add_argument("--n_w_rigid", type=int, default=5,
                        help="Number of w_rigid samples (default: 10)")
    parser.add_argument("--n_waypoints", type=int, default=150,
                        help="Waypoints per guide trajectory (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print("Building planner ...")
    planner, p_init = build_planner()

    total = args.n_alpha * args.n_w_rigid
    print(f"Generating {args.n_alpha} × {args.n_w_rigid} = {total} guides  "
          f"(T ~ U({args.T_min}, {args.T_max})s, {args.n_waypoints} waypoints) ...")

    generate_guide_dataset(
        planner=planner,
        p_init_nominal=p_init,
        T_min=args.T_min,
        T_max=args.T_max,
        n_alpha=args.n_alpha,
        n_w_rigid=args.n_w_rigid,
        n_waypoints=args.n_waypoints,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
