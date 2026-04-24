"""Unit tests for optimize_multiple_bezier_iris (mfpp_smooth.py).

Run from repo root:
    python -m pytest extensions/kin_feasibility/test/test_mfpp_smooth.py -v

Set B_VISUALIZE = True to open a matplotlib 3D window after each test.
"""
import sys
import os
import unittest
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from extensions.traversable_regions import TraversableRegions
from extensions.kin_feasibility.multiframe_fpp.mfpp_smooth import optimize_multiple_bezier_iris

B_VISUALIZE = False  # flip to True for interactive 3D plots

# ---------------------------------------------------------------------------
# Shared fixtures (same IRIS geometry as test_mfpp_polygonal.py)
# ---------------------------------------------------------------------------

A_MAT = np.vstack([np.eye(3), -np.eye(3)])

_B0 = np.array([0.3,  0.8,  1.2,  1.6, 0.8,  0.0])   # approach box
_B1 = np.array([1.6,  0.37, 1.2,  1.6, 0.37, -0.4])  # doorway box
_B2 = np.array([1.6,  0.8,  1.2, -0.41, 0.8,  0.0])  # exit box

IRIS_LST = [
    {'A': A_MAT, 'b': _B0},
    {'A': A_MAT, 'b': _B1},
    {'A': A_MAT, 'b': _B2},
]

IRIS_SEQ = OrderedDict([
    ("torso",  [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("LF",     [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("RF",     [[0], [0, 1], [1, 1, 1], [1, 2], [2]]),
    ("L_knee", [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("R_knee", [[0], [0, 1], [1, 1, 1], [2, 2], [2]]),
    ("LH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("RH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
])

SAFE_PNT_LST = [
    {'LF':     np.array([0.03357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.21127486,  0.15165375, 0.77523073]),
     'L_knee': np.array([0.18606585,  0.1186009,  0.31920651]),
     'RF':     np.array([0.03357225, -0.11850645, 0.043254  ]),
     'RH':     np.array([0.21127486, -0.15164375, 0.77523073]),
     'R_knee': np.array([0.18606585, -0.1186009,  0.31920651]),
     'torso':  np.array([-0.0339635,  0.,          0.844    ])},
    {'LF':     np.array([0.03357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.34,  0.35, 1.  ]),
     'L_knee': np.array([0.18606585,  0.1186009,  0.31920651]),
     'RF':     np.array([0.03357225, -0.11850645, 0.043254  ]),
     'RH':     np.array([0.34, -0.35,  1.  ]),
     'R_knee': np.array([0.18606585, -0.1186009,  0.31920651])},
    {'LF':     np.array([0.03357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.34,  0.35, 1.  ]),
     'L_knee': np.array([0.18606585,  0.1186009,  0.31920651]),
     'RF':     np.array([0.35,       -0.11850645, 0.44      ]),
     'RH':     np.array([0.34, -0.35,  1.  ]),
     'R_knee': np.array([0.5,        -0.11850645, 0.72      ])},
    {'LF':     np.array([0.47357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.34,  0.35, 1.  ]),
     'L_knee': np.array([0.62357225,  0.11850645, 0.323254  ]),
     'RF':     np.array([0.35,       -0.11850645, 0.44      ]),
     'RH':     np.array([0.34, -0.35,  1.  ]),
     'R_knee': np.array([0.5,        -0.11850645, 0.72      ])},
    {'LF':     np.array([0.47357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.65127486,  0.15165375, 0.77523073]),
     'L_knee': np.array([0.62357225,  0.11850645, 0.323254  ]),
     'RF':     np.array([0.47357225, -0.11850645, 0.043254  ]),
     'RH':     np.array([0.34, -0.35,  1.  ]),
     'R_knee': np.array([0.62357225, -0.11850645, 0.323254  ]),
     'torso':  np.array([0.4060365,   0.,          0.844    ])},
    {'LF':     np.array([0.47357225,  0.11850645, 0.043254  ]),
     'LH':     np.array([0.65127486,  0.15165375, 0.77523073]),
     'L_knee': np.array([0.62357225,  0.11850645, 0.323254  ]),
     'RF':     np.array([0.47357225, -0.11850645, 0.043254  ]),
     'RH':     np.array([0.65127486, -0.15164375, 0.77523073]),
     'R_knee': np.array([0.62357225, -0.11850645, 0.323254  ]),
     'torso':  np.array([0.4060365,   0.,          0.844    ])},
]


def _make_traversable_regions():
    return TraversableRegions(IRIS_LST, IRIS_SEQ, SAFE_PNT_LST)


def _make_durations(iris_seq, T=1.0):
    """Build uniform duration array per box from IRIS_SEQ."""
    frame_names = list(iris_seq.keys())
    first_frame = frame_names[0]
    durations = []
    for phase_boxes in iris_seq[first_frame]:
        n_boxes = len(phase_boxes)
        seg_dict = {name: np.full(n_boxes, T) for name in frame_names}
        durations.append(seg_dict)
    return durations


def _num_iris_tot(iris_seq):
    first_frame = next(iter(iris_seq))
    return sum(len(phase) for phase in iris_seq[first_frame])


def _visualize_smooth(path, frame_list, iris_regions, safe_pnt_lst, title="", n_samples=30):
    """3D matplotlib plot of the smooth Bezier solution."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(frame_list)))

    box_colors = ['lightblue', 'lightgreen', 'lightyellow']
    for reg_idx, reg in enumerate(iris_regions):
        b = reg['b']
        lo = np.array([-b[3], -b[4], -b[5]])
        hi = np.array([ b[0],  b[1],  b[2]])
        _draw_box_wireframe(ax, lo, hi, color=box_colors[reg_idx % len(box_colors)],
                            label=f"IRIS {reg_idx}" if reg_idx < 3 else None)

    for i, (fname, composite_bez) in enumerate(zip(frame_list, path)):
        t_vals = np.linspace(composite_bez.a, composite_bez.b, n_samples)
        pts = np.array([composite_bez(t) for t in t_vals])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                '-o', color=colors[i], markersize=2, linewidth=1.5, label=fname)

    for sp_dict in safe_pnt_lst:
        for fname, pos in sp_dict.items():
            pos = np.asarray(pos)
            ax.scatter(*pos, marker='*', s=120, zorder=5, color='red')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title or "optimize_multiple_bezier_iris solution")
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()


def _draw_box_wireframe(ax, lo, hi, color='blue', label=None):
    """Draw the 12 edges of an axis-aligned box."""
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
                color=color, linewidth=0.8,
                label=label if k == 0 else None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimizeMultipleBezierIris(unittest.TestCase):

    def setUp(self):
        self.tr = _make_traversable_regions()
        self.frame_list = list(SAFE_PNT_LST[0].keys())   # order from safe_points_lst[0]
        self.n_iris = _num_iris_tot(IRIS_SEQ)             # 9
        self.durations = _make_durations(IRIS_SEQ, T=1.0)
        self.alpha = {2: 1.0}                             # minimize integrated acceleration squared
        self.fixed_frames = [None] * 5                    # no fixed frames

    # ------------------------------------------------------------------
    # Test 1: no reach, no aux-frames, no fixed frames
    # ------------------------------------------------------------------
    def test_feasible_no_reach_no_aux(self):
        """Smooth solver returns a feasible solution with reach=None, aux=None."""
        path, sol_stats, points, dual_vars = optimize_multiple_bezier_iris(
            reach_region=None,
            aux_frames=None,
            traversable_regions=self.tr,
            durations=self.durations,
            alpha=self.alpha,
            fixed_frames=self.fixed_frames,
        )

        self.assertIsNotNone(path, "Path should not be None")
        self.assertEqual(len(path), len(self.frame_list),
                         f"Expected {len(self.frame_list)} CompositeBezierCurves, got {len(path)}")

        # Each frame's composite bezier must contain exactly num_iris_tot pieces.
        for fname, composite_bez in zip(self.frame_list, path):
            self.assertEqual(len(composite_bez.beziers), self.n_iris,
                             f"Frame {fname}: expected {self.n_iris} bezier pieces, "
                             f"got {len(composite_bez.beziers)}")

        self.assertFalse(np.isnan(sol_stats['cost']), "Cost contains NaN")
        self.assertGreater(sol_stats['runtime'], 0.0)

        self._check_safe_points(path, tol=0.05)

        if B_VISUALIZE:
            _visualize_smooth(path, self.frame_list, IRIS_LST, SAFE_PNT_LST,
                              title="test_feasible_no_reach_no_aux")

    # ------------------------------------------------------------------
    # Test 2: varied alpha (derivative order / cost weight) configurations
    # ------------------------------------------------------------------
    def test_different_alpha_values(self):
        """Solver stays feasible across a range of alpha cost configurations."""
        alpha_cases = [
            {1: 1.0},            # minimize integrated velocity squared
            {2: 1.0},            # minimize integrated acceleration squared
            {1: 0.5, 2: 0.5},   # mixed velocity + acceleration
            {2: 0.1},            # small acceleration weight
        ]

        for alpha in alpha_cases:
            with self.subTest(alpha=alpha):
                path, sol_stats, _, _ = optimize_multiple_bezier_iris(
                    reach_region=None,
                    aux_frames=None,
                    traversable_regions=self.tr,
                    durations=self.durations,
                    alpha=alpha,
                    fixed_frames=self.fixed_frames,
                )

                self.assertIsNotNone(path, f"Path is None for alpha={alpha}")
                self.assertEqual(len(path), len(self.frame_list))
                self.assertFalse(np.isnan(sol_stats['cost']),
                                 f"Cost is NaN for alpha={alpha}")

                self._check_safe_points(path, tol=0.05)

                if B_VISUALIZE:
                    _visualize_smooth(path, self.frame_list, IRIS_LST, SAFE_PNT_LST,
                                     title=f"alpha={alpha}")

    # ------------------------------------------------------------------
    # Test 3: some frames are pinned to safe points for entire segments
    # ------------------------------------------------------------------
    def test_with_fixed_frames(self):
        """Solver remains feasible when feet are fixed in segment 0 and hands in segment 3."""
        # Fixing a frame in the last segment (4) overrides the final safe-point constraint
        # with SAFE_PNT_LST[4] instead of SAFE_PNT_LST[-1], causing check failures when the
        # two differ. Use only intermediate segments (0–3) to avoid this.
        fixed_frames = [
            ['LF', 'RF'],   # fix feet in phase 0
            None,
            None,
            ['LH', 'RH'],  # fix hands in phase 3 (not the final phase)
            None,
        ]

        path, sol_stats, _, _ = optimize_multiple_bezier_iris(
            reach_region=None,
            aux_frames=None,
            traversable_regions=self.tr,
            durations=self.durations,
            alpha=self.alpha,
            fixed_frames=fixed_frames,
        )

        self.assertIsNotNone(path)
        self.assertFalse(np.isnan(sol_stats['cost']))

        self._check_safe_points(path, tol=0.05)

        if B_VISUALIZE:
            _visualize_smooth(path, self.frame_list, IRIS_LST, SAFE_PNT_LST,
                              title="test_with_fixed_frames")

    # ------------------------------------------------------------------
    # Test 4: G1 env fixed-frame schedule from g1_env_cfg.py
    # ------------------------------------------------------------------
    def test_g1_fixed_frames(self):
        """Smooth solver is feasible with the exact FIXED_FRAMES from ResidualGuideTrackingEnvCfg.

        FIXED_FRAMES[4] (last segment) only contains frames whose SAFE_PNT_LST[4] value
        equals SAFE_PNT_LST[-1], so the final safe-point check still passes.
        """
        # Copied verbatim from g1_env_cfg.py :: ResidualGuideTrackingEnvCfg.FIXED_FRAMES
        g1_fixed_frames = [
            ['LF', 'RF', 'L_knee', 'R_knee'],       # seg 0: lower body stationary
            ['LF', 'L_knee', 'LH', 'RH'],            # seg 1: right side steps through
            ['RF', 'R_knee', 'LH', 'RH'],            # seg 2: left side steps through
            ['LF', 'L_knee', 'RH'],                  # seg 3: partial re-plant
            ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'],  # seg 4: square up
        ]

        path, sol_stats, _, _ = optimize_multiple_bezier_iris(
            reach_region=None,
            aux_frames=None,
            traversable_regions=self.tr,
            durations=self.durations,
            alpha=self.alpha,
            fixed_frames=g1_fixed_frames,
        )

        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(self.frame_list))
        self.assertFalse(np.isnan(sol_stats['cost']))
        self.assertGreater(sol_stats['runtime'], 0.0)

        self._check_safe_points(path, tol=0.05)

        if B_VISUALIZE:
            _visualize_smooth(path, self.frame_list, IRIS_LST, SAFE_PNT_LST,
                              title="test_g1_fixed_frames")

    # ------------------------------------------------------------------
    # Test 5: C0 continuity at bezier segment junctions
    # ------------------------------------------------------------------
    def test_bezier_c0_continuity(self):
        """End-point of each Bezier piece matches start-point of the next."""
        path, _, _, _ = optimize_multiple_bezier_iris(
            reach_region=None,
            aux_frames=None,
            traversable_regions=self.tr,
            durations=self.durations,
            alpha=self.alpha,
            fixed_frames=self.fixed_frames,
        )

        self.assertIsNotNone(path)

        tol = 1e-3
        for fname, composite_bez in zip(self.frame_list, path):
            for i in range(len(composite_bez.beziers) - 1):
                end_pt = composite_bez.beziers[i].end_point()
                start_pt = composite_bez.beziers[i + 1].start_point()
                np.testing.assert_allclose(
                    end_pt, start_pt, atol=tol,
                    err_msg=f"C0 discontinuity at junction {i}→{i+1} for frame '{fname}'"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_safe_points(self, path, tol=0.05):
        """Assert that initial and final safe-point boundary constraints hold within tol."""
        frame_to_idx = {name: i for i, name in enumerate(self.frame_list)}

        for fname in self.frame_list:
            fr_idx = frame_to_idx[fname]
            composite_bez = path[fr_idx]
            t_start = composite_bez.a
            t_end = composite_bez.b

            if fname in SAFE_PNT_LST[0]:
                expected = np.asarray(SAFE_PNT_LST[0][fname])
                actual = composite_bez(t_start)
                np.testing.assert_allclose(
                    actual, expected, atol=tol,
                    err_msg=f"Initial safe-point mismatch for frame '{fname}'"
                )

            if fname in SAFE_PNT_LST[-1]:
                expected = np.asarray(SAFE_PNT_LST[-1][fname])
                actual = composite_bez(t_end)
                np.testing.assert_allclose(
                    actual, expected, atol=tol,
                    err_msg=f"Final safe-point mismatch for frame '{fname}'"
                )


if __name__ == '__main__':
    B_VISUALIZE = True
    unittest.main()
