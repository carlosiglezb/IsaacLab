"""Load guide_dataset.npz and visualize a sampled subset of trajectories.

Uses GuideDataset.sample to draw random guides and GuideDataset.query_targets
to sweep time, so the same runtime code-path is exercised during validation.

Usage (from repo root):
    python scripts/visualize_guide_dataset.py --dataset guide_dataset.npz \\
        --n_guides 4 --n_queries 50 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure repo root is on path before any local-extension imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from extensions.g1_residual_control.guide_dataset import GuideDataset
from extensions.g1_residual_control.g1_planner_constants import IRIS_LST, SAFE_PNT_LST

_BASE_TORSO_XY = SAFE_PNT_LST[0]["torso"][:2].copy()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_BOX_COLORS = ['#aec6cf', '#b5ead7', '#ffdac1']   # pastel blue / green / orange


def _draw_box_wireframe(ax, b_vec, color='lightgrey', label=None):
    """Draw an axis-aligned box from b_vec = [+x,+y,+z,−x,−y,−z]."""
    lo = np.array([-b_vec[3], -b_vec[4], -b_vec[5]])
    hi = np.array([ b_vec[0],  b_vec[1],  b_vec[2]])
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for k, (a, b) in enumerate(edges):
        ax.plot([corners[a,0], corners[b,0]],
                [corners[a,1], corners[b,1]],
                [corners[a,2], corners[b,2]],
                color=color, linewidth=0.6, alpha=0.5,
                label=label if k == 0 else None)


# ---------------------------------------------------------------------------
# Validation print
# ---------------------------------------------------------------------------

def print_dataset_summary(dataset: GuideDataset) -> None:
    print("\n=== GuideDataset summary ===")
    s = dataset.ctrl_pts.shape
    print(f"  ctrl_pts shape : {tuple(s)}  "
          f"(n_guides={s[0]} × n_frames={s[1]} × n_segments={s[2]} × (degree+1)={s[3]} × 3)")
    print(f"  Bezier degree  : {dataset.degree}")
    print(f"  frames         : {dataset.frame_names}")
    print(f"  T_plan (mean)  : {dataset.T_plan:.3f} s")
    t = dataset.T_plan_arr.numpy()
    print(f"  T_plan range   : [{t.min():.3f}, {t.max():.3f}] s")
    if dataset.hand_mask is not None:
        contact = [n for n, m in zip(dataset.frame_names, dataset.hand_mask.tolist()) if m]
        print(f"  contact frames : {contact}")
    torso_xy = dataset.p_init_nominal_torso_arr[:, :2].numpy()
    print(f"  torso XY range : "
          f"x=[{torso_xy[:, 0].min():+.3f}, {torso_xy[:, 0].max():+.3f}]  "
          f"y=[{torso_xy[:, 1].min():+.3f}, {torso_xy[:, 1].max():+.3f}]")


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize(dataset_path: str, n_guides: int, n_queries: int, seed: int,
              save_path: str | None) -> None:
    dataset = GuideDataset(dataset_path, device="cpu")
    print_dataset_summary(dataset)

    torch.manual_seed(seed)

    # --- Select guides at their generated positions (no runtime shift) ------
    # We index the guide tensors directly so trajectories are displayed exactly
    # as produced by the planner. query_targets() is still exercised on the result.
    idx = torch.randint(0, dataset.n_guides, (n_guides,))
    ctrl_pts    = dataset.ctrl_pts[idx].clone()           # (G, F, S, D+1, 3)
    trans_times = dataset.transition_times[idx].clone()   # (G, S+1)
    T_per_env   = dataset.T_plan_arr[idx].clone()
    torso_nominals = dataset.p_init_nominal_torso_arr[idx]  # (G, 3)

    print(f"\nSampled {n_guides} guides  |  "
          f"T per guide: {T_per_env.numpy().round(3).tolist()}")
    for g_i, (g_idx, p_nom) in enumerate(zip(idx.tolist(), torso_nominals.numpy())):
        xy_off = p_nom[:2] - _BASE_TORSO_XY
        print(f"  guide {g_i} (dataset idx {g_idx:3d}): "
              f"torso xy nominal = [{p_nom[0]:+.3f}, {p_nom[1]:+.3f}]  "
              f"(offset [{xy_off[0]:+.3f}, {xy_off[1]:+.3f}])")

    frame_names = dataset.frame_names
    n_frames    = dataset.n_frames
    colors      = plt.cm.tab10(np.linspace(0, 1, n_frames))

    fig = plt.figure(figsize=(10, 5 * n_guides))

    for g in range(n_guides):
        ax = fig.add_subplot(n_guides, 1, g + 1, projection='3d')

        # ---- IRIS box wireframes -----------------------------------------
        for reg_idx, reg in enumerate(IRIS_LST):
            _draw_box_wireframe(ax, reg['b'], color=_BOX_COLORS[reg_idx],
                                label=f"IRIS {reg_idx}" if g == 0 else None)

        # ---- Safe-point markers (shifted to match this guide's XY offset) --
        xy_off = torso_nominals[g, :2].numpy() - _BASE_TORSO_XY
        first_sp = True
        for sp_dict in SAFE_PNT_LST:
            for pos in sp_dict.values():
                shifted = np.asarray(pos).copy()
                shifted[0] += xy_off[0]
                shifted[1] += xy_off[1]
                ax.scatter(*shifted, marker='*', s=60,
                           color='red', zorder=5, depthshade=False,
                           label='safe pts' if g == 0 and first_sp else None)
                first_sp = False

        # ---- Query guide trajectory via GuideDataset.query_targets --------
        T_g = float(T_per_env[g])
        t_vals      = torch.linspace(0.0, T_g, n_queries)                        # (Q,)
        ctrl_pts_g  = ctrl_pts[g:g+1].expand(n_queries, -1, -1, -1, -1)         # (Q, F, S, D+1, 3)
        trans_g     = trans_times[g:g+1].expand(n_queries, -1)                   # (Q, S+1)
        targets     = dataset.query_targets(ctrl_pts_g, trans_g, t_vals)         # (Q, F, 3)
        pts_np      = targets.numpy()                                             # (Q, F, 3)

        for f_idx, fname in enumerate(frame_names):
            traj = pts_np[:, f_idx, :]                                       # (Q, 3)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    '-o', color=colors[f_idx], markersize=2, linewidth=1.5,
                    label=fname if g == 0 else None)
            # Mark start and end
            ax.scatter(*traj[0],  marker='>', s=40, color=colors[f_idx], zorder=6)
            ax.scatter(*traj[-1], marker='s', s=40, color=colors[f_idx], zorder=6)

        p_nom = torso_nominals[g].numpy()
        xy_off_g = p_nom[:2] - _BASE_TORSO_XY
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_title(
            f"Guide {g}  |  T = {T_g:.2f} s  |  "
            f"offset = [{xy_off_g[0]:+.3f}, {xy_off_g[1]:+.3f}] m",
            fontsize=10,
        )
        if g == 0:
            ax.legend(loc='upper left', fontsize=7, ncol=3)

    plt.suptitle(f"{dataset_path}  —  {n_guides} sampled guides", fontsize=11)
    plt.tight_layout()

    out = save_path or os.path.splitext(dataset_path)[0] + "_preview.png"
    plt.savefig(out, dpi=130)
    print(f"\nSaved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a subset of guides from a guide_dataset.npz file"
    )
    parser.add_argument("--dataset", default="guide_dataset.npz",
                        help="Path to the .npz dataset (default: guide_dataset.npz)")
    parser.add_argument("--n_guides", type=int, default=2,
                        help="Number of guides to sample and plot (default: 2)")
    parser.add_argument("--n_queries", type=int, default=350,
                        help="Time steps queried via query_targets (default: 350)")
    parser.add_argument("--seed", type=int, default=10,
                        help="RNG seed for guide sampling (default: 0)")
    parser.add_argument("--save_path", default=None,
                        help="Output PNG path (default: <dataset>_preview.png)")
    args = parser.parse_args()

    visualize(
        dataset_path=args.dataset,
        n_guides=args.n_guides,
        n_queries=args.n_queries,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
