"""Planner-level geometric constants for the G1 knee-knocker environment.

Kept in a standalone module (no IsaacLab / Isaac Sim imports) so that
offline scripts such as generate_guide_dataset.py can import them without
requiring a full Isaac Sim installation.

g1_env_cfg.py imports from here so the values stay in one place.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Root-to-torso offset
# ---------------------------------------------------------------------------
# Translation from the G1 pelvis (root) frame to the torso frame origin,
# used to shift reachability halfspace offsets into the root frame.
ROOT_TO_TORSO_OFFSET: list[float] = [-0.0039635, 0.0, 0.164]

# ---------------------------------------------------------------------------
# IRIS collision-free regions
# ---------------------------------------------------------------------------
# Three axis-aligned boxes around the Navy-door knee-knocker obstacle.
# Constraint: A_mat @ p <= b_vec  where  A_mat = [I3; -I3]
# encodes  x_min <= p <= x_max  element-wise.
_A_MAT = np.vstack([np.eye(3), -np.eye(3)])

# IRIS 0: approach side   −1.6 < x < 0.30,  −0.80 < y < 0.80,  0 < z < 1.2
_B0 = np.array([0.3,  0.8,  1.2,  1.6,  0.8,  0.0])
# IRIS 1: doorway passage −1.6 < x < 1.60,  −0.37 < y < 0.37,  0.4 < z < 1.2
_B1 = np.array([1.6,  0.37, 1.2,  1.6,  0.37, -0.4])
# IRIS 2: exit side        0.41 < x < 1.60, −0.80 < y < 0.80,  0 < z < 1.2
_B2 = np.array([1.6,  0.8,  1.2, -0.41, 0.8,  0.0])

IRIS_0: dict = {'A': _A_MAT, 'b': _B0}
IRIS_1: dict = {'A': _A_MAT, 'b': _B1}
IRIS_2: dict = {'A': _A_MAT, 'b': _B2}

IRIS_LST: list[dict] = [IRIS_0, IRIS_1, IRIS_2]

# ---------------------------------------------------------------------------
# IRIS contact sequence
# ---------------------------------------------------------------------------
# Per-body list of IRIS region indices across motion phases.
# Phases: [pre-approach, approach, crossing, exit, post-exit]
IRIS_SEQ: OrderedDict = OrderedDict([
    ("torso",  [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("LF",     [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("RF",     [[0], [0, 1], [1, 1, 1], [1, 2], [2]]),
    ("L_knee", [[0], [0, 0], [0, 1, 2], [2, 2], [2]]),
    ("R_knee", [[0], [0, 1], [1, 1, 1], [2, 2], [2]]),
    ("LH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
    ("RH",     [[1], [1, 1], [1, 1, 1], [1, 1], [1]]),
])

# ---------------------------------------------------------------------------
# Safe-point waypoints
# ---------------------------------------------------------------------------
# Ordered list of dicts: one per contact-phase boundary.
# Values are world-frame 3-D positions [x, y, z] (metres).
SAFE_PNT_LST: list[dict[str, np.ndarray]] = [
    {'LF':     np.array([ 0.03357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.21127486,  0.15165375,  0.77523073]),
     'L_knee': np.array([ 0.18606585,  0.1186009,   0.31920651]),
     'RF':     np.array([ 0.03357225, -0.11850645,  0.043254  ]),
     'RH':     np.array([ 0.21127486, -0.15164375,  0.77523073]),
     'R_knee': np.array([ 0.18606585, -0.1186009,   0.31920651]),
     'torso':  np.array([-0.0339635,   0.,           0.844    ])},
    {'LF':     np.array([ 0.03357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.34,        0.35,         1.        ]),
     'L_knee': np.array([ 0.18606585,  0.1186009,   0.31920651]),
     'RF':     np.array([ 0.03357225, -0.11850645,  0.043254  ]),
     'RH':     np.array([ 0.34,       -0.35,         1.        ]),
     'R_knee': np.array([ 0.18606585, -0.1186009,   0.31920651])},
    {'LF':     np.array([ 0.03357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.34,        0.35,         1.        ]),
     'L_knee': np.array([ 0.18606585,  0.1186009,   0.31920651]),
     'RF':     np.array([ 0.35,       -0.11850645,  0.44      ]),
     'RH':     np.array([ 0.34,       -0.35,         1.        ]),
     'R_knee': np.array([ 0.5,        -0.11850645,  0.72      ])},
    {'LF':     np.array([ 0.47357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.34,        0.35,         1.        ]),
     'L_knee': np.array([ 0.62357225,  0.11850645,  0.323254  ]),
     'RF':     np.array([ 0.35,       -0.11850645,  0.44      ]),
     'RH':     np.array([ 0.34,       -0.35,         1.        ]),
     'R_knee': np.array([ 0.5,        -0.11850645,  0.72      ])},
    {'LF':     np.array([ 0.47357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.65127486,  0.15165375,  0.77523073]),
     'L_knee': np.array([ 0.62357225,  0.11850645,  0.323254  ]),
     'RF':     np.array([ 0.47357225, -0.11850645,  0.043254  ]),
     'RH':     np.array([ 0.34,       -0.35,         1.        ]),
     'R_knee': np.array([ 0.62357225, -0.11850645,  0.323254  ]),
     'torso':  np.array([ 0.4060365,   0.,           0.844    ])},
    {'LF':     np.array([ 0.47357225,  0.11850645,  0.043254  ]),
     'LH':     np.array([ 0.65127486,  0.15165375,  0.77523073]),
     'L_knee': np.array([ 0.62357225,  0.11850645,  0.323254  ]),
     'RF':     np.array([ 0.47357225, -0.11850645,  0.043254  ]),
     'RH':     np.array([ 0.65127486, -0.15164375,  0.77523073]),
     'R_knee': np.array([ 0.62357225, -0.11850645,  0.323254  ]),
     'torso':  np.array([ 0.4060365,   0.,           0.844    ])},
]

# ---------------------------------------------------------------------------
# Fixed-frame schedule
# ---------------------------------------------------------------------------
# Per-phase list of frame names whose positions are held fixed by the planner.
FIXED_FRAMES: list[list[str]] = [
    ['LF', 'RF', 'L_knee', 'R_knee'],
    ['LF', 'L_knee', 'LH', 'RH'],
    ['RF', 'R_knee', 'LH', 'RH'],
    ['LF', 'L_knee', 'RH'],
    ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'],
]
