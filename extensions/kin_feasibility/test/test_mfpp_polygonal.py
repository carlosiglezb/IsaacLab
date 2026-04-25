"""Unit tests for solve_min_reach_iris_distance (mfpp_polygonal.py).

Run from repo root:
    python -m pytest extensions/kin_feasibility/test/test_mfpp_polygonal.py -v

Set B_VISUALIZE = True to open a matplotlib 3D window after each test.
"""
import sys
import os
import unittest
from collections import OrderedDict

import numpy as np

# Ensure repo root is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ruamel.yaml import YAML

from extensions.traversable_regions import TraversableRegions
from extensions.kin_feasibility.multiframe_fpp.mfpp_polygonal import solve_min_reach_iris_distance

B_VISUALIZE = True   # flip to True for interactive 3D plots

# Paths to per-frame reachability halfspace yaml files and aux-frame descriptor.
# The test file lives 3 levels below the repo root (test/ → kin_feasibility/ → extensions/ → IsaacLab/).
_ISAACLAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_REACH_DIR = os.path.join(_ISAACLAB_ROOT, 'extensions', 'kin_feasibility', 'reachability')
_REACH_BASE = os.path.join(_REACH_DIR, 'g1_')          # + '<frame>.yaml'
_AUX_FRAMES_PATH = os.path.join(_REACH_DIR, 'g1_aux_frames.yaml')

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
ROOT_TO_TORSO_OFFSET = [-0.0039635, 0.0, 0.164]

# Correct 6×3 halfspace matrix: [I3; -I3] encodes  x_min ≤ x ≤ x_max
A_MAT = np.vstack([np.eye(3), -np.eye(3)])

# b-vectors from ResidualGuideTrackingEnvCfg (layout: [+x,+y,+z,−x,−y,−z])
_B0 = np.array([0.3,  0.8,  1.2,  1.6, 0.8,  0.0])   # approach box
_B1 = np.array([1.6,  0.37, 1.2,  1.6, 0.37, -0.4])  # doorway box
_B2 = np.array([1.6,  0.8,  1.2, -0.41, 0.8,  0.0])  # exit box

IRIS_LST = [
    {'A': A_MAT, 'b': _B0},
    {'A': A_MAT, 'b': _B1},
    {'A': A_MAT, 'b': _B2},
]

# Per-frame IRIS region index sequences across 5 motion phases.
# Matches ResidualGuideTrackingEnvCfg.IRIS_seq exactly.
IRIS_SEQ = OrderedDict([
    ("torso",  [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("LF",     [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("RF",     [[0], [0, 1], [1, 1, 1], [1, 2], [2]]),
    ("L_knee", [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("R_knee", [[0], [0, 1], [1, 1, 1], [2, 2], [2]]),
    ("LH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("RH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
])

# Safe contact points at each phase boundary (6 entries for 5 phases).
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


def _make_traversable_regions(reach_paths: dict):
    return TraversableRegions(IRIS_LST, IRIS_SEQ, SAFE_PNT_LST, reach_paths, root_to_torso_pos=ROOT_TO_TORSO_OFFSET)


def _load_reach_paths(frame_names):
    """Load G1 convex-hull reachability halfspaces from yaml files."""
    reach_paths = {}
    for frame_name in frame_names:
        if frame_name == 'torso':
            continue
        path = _REACH_BASE + frame_name + '.yaml'
        reach_paths[frame_name] = path
    return reach_paths


def _load_aux_frames():
    """Load G1 shin-link aux-frame descriptors from yaml."""
    aux_frames = []
    with open(_AUX_FRAMES_PATH, 'r') as f:
        yml = YAML().load(f)
        for fr in yml:
            aux_frames.append(dict(fr))
    return aux_frames


def _unpack_traj(traj, frame_names, num_iris_tot, d=3):
    """Return dict {frame_name: (num_iris_tot+1, 3) array} from flat traj."""
    n_pts = num_iris_tot + 1
    result = {}
    for i, name in enumerate(frame_names):
        start = i * n_pts * d
        pts = traj[start: start + n_pts * d].reshape(n_pts, d)
        result[name] = pts
    return result


def _num_iris_tot(iris_seq):
    first_frame = next(iter(iris_seq))
    return sum(len(phase) for phase in iris_seq[first_frame])


def _visualize(traj_dict, iris_regions, iris_seq, safe_pnt_lst, title=""):
    """3D matplotlib scatter of the polygonal solution."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    frame_names = list(traj_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(frame_names)))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw IRIS region boxes as wireframes
    box_colors = ['lightblue', 'lightgreen', 'yellow']
    for reg_idx, reg in enumerate(iris_regions):
        b = reg['b']
        # bounds: x in [-b[3], b[0]], y in [-b[4], b[1]], z in [-b[5], b[2]]
        lo = np.array([-b[3], -b[4], -b[5]])
        hi = np.array([ b[0],  b[1],  b[2]])
        _draw_box_wireframe(ax, lo, hi, color=box_colors[reg_idx % len(box_colors)],
                            label=f"IRIS {reg_idx}" if reg_idx < 3 else None)

    # Draw per-frame trajectories
    for i, (fname, pts) in enumerate(traj_dict.items()):
        c = colors[i]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                '-o', color=c, markersize=3, linewidth=1.5, label=fname)

    # Mark safe points
    for sp_dict in safe_pnt_lst:
        for fname, pos in sp_dict.items():
            pos = np.asarray(pos)
            ax.scatter(*pos, marker='*', s=120, zorder=5, color='red')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title or "solve_min_reach_iris_distance solution")
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()


def _draw_box_wireframe(ax, lo, hi, color='blue', label=None):
    """Draw the 12 edges of an axis-aligned box."""
    from itertools import combinations
    corners = np.array([[lo[0], lo[1], lo[2]],
                        [hi[0], lo[1], lo[2]],
                        [hi[0], hi[1], lo[2]],
                        [lo[0], hi[1], lo[2]],
                        [lo[0], lo[1], hi[2]],
                        [hi[0], lo[1], hi[2]],
                        [hi[0], hi[1], hi[2]],
                        [lo[0], hi[1], hi[2]]])
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

class TestSolveMinReachIrisDistance(unittest.TestCase):

    def setUp(self):
        reach_paths = _load_reach_paths(list(IRIS_SEQ.keys()))
        self.tr = _make_traversable_regions(reach_paths)
        self.frame_names = list(IRIS_SEQ.keys())
        self.n_iris = _num_iris_tot(IRIS_SEQ)   # = 9

    # ------------------------------------------------------------------
    # Test 1: no reach, no aux-frames, default w_rigid
    # ------------------------------------------------------------------
    def test_feasible_no_reach_no_aux(self):
        """Solver must return a feasible solution with reach=None, aux=None."""
        traj, length, solve_time = solve_min_reach_iris_distance(
            traversable_regions=self.tr,
            aux_frames=None,
            weights_rigid=None,
        )

        self.assertIsNotNone(traj, "Trajectory should not be None")
        self.assertFalse(np.isnan(traj).any(), "Trajectory contains NaN values")

        expected_len = 3 * len(self.frame_names) * (self.n_iris + 1)
        self.assertEqual(traj.shape[0], expected_len,
                         f"Expected {expected_len} decision variables, got {traj.shape[0]}")
        self.assertGreater(length, 0.0, "Objective value must be positive")
        self.assertGreater(solve_time, 0.0)

        # Safe-point equality constraints: every prescribed boundary point
        # must be satisfied to within solver tolerance (1 cm).
        traj_dict = _unpack_traj(traj, self.frame_names, self.n_iris)
        self._check_safe_points(traj_dict, tol=0.05)

        if B_VISUALIZE:
            _visualize(traj_dict, IRIS_LST, IRIS_SEQ, SAFE_PNT_LST,
                       title="test_feasible_no_reach_no_aux")

    # ------------------------------------------------------------------
    # Test 2: varied w_rigid values
    # ------------------------------------------------------------------
    def test_different_w_rigid_values(self):
        """Solver must stay feasible across a range of w_rigid weights."""
        w_rigid_cases = [
            np.array([0.0,   0.0,   0.0  ]),   # no regularisation
            np.array([1.0,   0.0,   1.0  ]),   # x/z only
            np.array([1.621, 0.0,   0.808]),   # default-like
            np.array([5.0,   0.0,   5.0  ]),   # heavy regularisation
        ]

        for w_rigid in w_rigid_cases:
            with self.subTest(w_rigid=w_rigid):
                traj, length, solve_time = solve_min_reach_iris_distance(
                    traversable_regions=self.tr,
                    aux_frames=None,
                    weights_rigid=w_rigid,
                )

                self.assertIsNotNone(traj,
                    f"Trajectory is None for w_rigid={w_rigid}")
                self.assertFalse(np.isnan(traj).any(),
                    f"NaN in trajectory for w_rigid={w_rigid}")
                self.assertGreater(length, 0.0)

                traj_dict = _unpack_traj(traj, self.frame_names, self.n_iris)
                self._check_safe_points(traj_dict, tol=0.05)

                if B_VISUALIZE:
                    _visualize(traj_dict, IRIS_LST, IRIS_SEQ, SAFE_PNT_LST,
                               title=f"w_rigid={np.round(w_rigid, 2)}")

    # ------------------------------------------------------------------
    # Test 3: G1 reachability constraints, no aux frames
    # ------------------------------------------------------------------
    def test_with_reach_only(self):
        """Solver returns a feasible solution when G1 reachability halfspaces are active."""

        traj, length, solve_time = solve_min_reach_iris_distance(
            traversable_regions=self.tr,
            aux_frames=None,
            weights_rigid=None,
        )

        self.assertIsNotNone(traj, "Trajectory is None with reach constraints")
        self.assertFalse(np.isnan(traj).any(), "NaN in trajectory with reach constraints")
        self.assertGreater(solve_time, 0.0)

        traj_dict = _unpack_traj(traj, self.frame_names, self.n_iris)
        self._check_safe_points(traj_dict, tol=0.05)

        if B_VISUALIZE:
            _visualize(traj_dict, IRIS_LST, IRIS_SEQ, SAFE_PNT_LST,
                       title="test_with_reach_only")

    # ------------------------------------------------------------------
    # Test 4: G1 shin-link aux-frame constraints, no reach region
    # ------------------------------------------------------------------
    def test_with_aux_frames_only(self):
        """Solver returns a feasible solution when shin-link rigid constraints are active."""
        aux_frames = _load_aux_frames()

        traj, length, solve_time = solve_min_reach_iris_distance(
            traversable_regions=self.tr,
            aux_frames=aux_frames,
            weights_rigid=np.array([1.621, 0.0, 0.808]),
        )

        self.assertIsNotNone(traj, "Trajectory is None with aux_frames")
        self.assertFalse(np.isnan(traj).any(), "NaN in trajectory with aux_frames")
        self.assertGreater(solve_time, 0.0)

        traj_dict = _unpack_traj(traj, self.frame_names, self.n_iris)
        self._check_safe_points(traj_dict, tol=0.05)

        if B_VISUALIZE:
            _visualize(traj_dict, IRIS_LST, IRIS_SEQ, SAFE_PNT_LST,
                       title="test_with_aux_frames_only")


    # ------------------------------------------------------------------
    # Test 6: full G1 config — reach + aux frames together
    # ------------------------------------------------------------------
    def test_with_reach_and_aux_frames(self):
        """Solver returns a feasible solution with both reach and shin-link constraints."""
        aux_frames = _load_aux_frames()

        traj, length, solve_time = solve_min_reach_iris_distance(
            traversable_regions=self.tr,
            aux_frames=aux_frames,
            weights_rigid=np.array([1.621, 0.0, 0.808]),
        )

        self.assertIsNotNone(traj, "Trajectory is None with reach+aux_frames")
        self.assertFalse(np.isnan(traj).any(), "NaN in trajectory with reach+aux_frames")
        self.assertGreater(solve_time, 0.0)

        traj_dict = _unpack_traj(traj, self.frame_names, self.n_iris)
        self._check_safe_points(traj_dict, tol=0.05)

        if B_VISUALIZE:
            _visualize(traj_dict, IRIS_LST, IRIS_SEQ, SAFE_PNT_LST,
                       title="test_with_reach_and_aux_frames")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_safe_points(self, traj_dict, tol=0.05):
        """Assert that safe-point boundary constraints hold within tol."""
        frame_names = list(IRIS_SEQ.keys())
        n_iris = self.n_iris

        # Reconstruct the index each safe-point corresponds to in the
        # flat trajectory for each frame.  The layout is:
        #   frame_pts[0] = initial (safe_pnt_lst[0])
        #   frame_pts[seg_cumlen] = boundary after segment seg_idx
        #   frame_pts[n_iris] = final (safe_pnt_lst[-1])
        for frame in frame_names:
            pts = traj_dict[frame]
            # initial point
            if frame in SAFE_PNT_LST[0]:
                expected = np.asarray(SAFE_PNT_LST[0][frame])
                np.testing.assert_allclose(
                    pts[0], expected, atol=tol,
                    err_msg=f"Initial safe-point mismatch for frame {frame}")

            # final point
            if frame in SAFE_PNT_LST[-1]:
                expected = np.asarray(SAFE_PNT_LST[-1][frame])
                np.testing.assert_allclose(
                    pts[-1], expected, atol=tol,
                    err_msg=f"Final safe-point mismatch for frame {frame}")


if __name__ == '__main__':
    B_VISUALIZE = True
    unittest.main()
