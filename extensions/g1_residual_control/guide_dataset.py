"""Offline guide dataset generation and GPU-resident runtime sampling.

Workflow
--------
Offline (run once, before training):

    from extensions.g1_residual_control.guide_dataset import generate_guide_dataset
    generate_guide_dataset(planner, p_init_nominal, iris_seq_cfg,
                           safe_pnt_lst, fixed_frames, T=15.0,
                           save_path="guides.npz")

Online (env reset, zero solve overhead):

    dataset = GuideDataset("guides.npz", device="cuda")
    # at reset:
    guides = dataset.sample(num_envs, p_torso_actual)   # (N, F, W, 3)
    # each policy step:
    targets = dataset.query_targets(guides, t_elapsed)  # (N, F, 3)

Shift semantics
---------------
All non-hand frames (torso, feet, knees) are translated by
    delta = p_torso_actual - p_torso_nominal
so the guide follows the robot's actual spawn position.
Hand frames (LH, RH) are **not** shifted because they are in contact with
fixed environment geometry; shifting would cause penetration.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Name mappings
# ---------------------------------------------------------------------------

# Planner frame name → URDF body name
FRAME_TO_BODY: dict[str, str] = {
    "torso":  "torso_link",
    "LF":     "left_ankle_roll_link",
    "RF":     "right_ankle_roll_link",
    "L_knee": "left_knee_link",
    "R_knee": "right_knee_link",
    "LH":     "left_rubber_hand",
    "RH":     "right_rubber_hand",
}
BODY_TO_FRAME: dict[str, str] = {v: k for k, v in FRAME_TO_BODY.items()}

# Frames in contact with the environment: their guides are never shifted.
_CONTACT_FRAMES: frozenset[str] = frozenset({"LH", "RH"})


# ---------------------------------------------------------------------------
# Offline generation
# ---------------------------------------------------------------------------

def _sample_paths(paths: list, n_waypoints: int, T_plan: float) -> np.ndarray:
    """Evaluate each CompositeBezierCurve at n_waypoints uniform time steps.

    Returns
    -------
    np.ndarray, shape (n_frames, n_waypoints, 3), float32
    """
    t_samples = np.linspace(0.0, T_plan, n_waypoints)
    out = np.zeros((len(paths), n_waypoints, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        t_clamped = np.clip(t_samples, path.a, path.b)
        for j, t in enumerate(t_clamped):
            out[i, j] = path(t)
    return out


def generate_guide_dataset(
    planner,
    p_init_nominal: dict[str, np.ndarray],
    T_min: float = 2.0,
    T_max: float = 3.0,
    n_alpha: int = 20,
    n_w_rigid: int = 10,
    w_rigid_mid_fixed: float = 0.0,
    n_waypoints: int = 150,
    seed: int = 42,
    save_path: str = "guide_dataset.npz",
) -> None:
    """Generate and save a dataset of kinematic guides.

    Samples ``n_alpha`` alpha vectors and ``n_w_rigid`` w_rigid vectors
    independently at random (Latin-Hypercube-like uniform coverage), then
    solves the two-stage SOCP planner for each of the
    ``n_alpha × n_w_rigid`` (alpha, w_rigid) combinations.

    Each guide independently draws a traversal duration
    T_i ~ U(T_min, T_max), increasing speed diversity in the dataset.
    Pass ``T`` (a fixed float) for backward-compatible behaviour.

    Parameter ranges
    ----------------
    alpha   : (n_alpha, 3) — entries [0,1] drawn from U(0, 1), entry [2] from U(0, 0.2)
    w_rigid : (n_w_rigid, 3) — entries [0] and [2] drawn from U(0, 1);
              entry [1] held fixed at ``w_rigid_mid_fixed``

    All guides are generated at ``p_init_nominal`` (the shift reference).
    The per-env rigid offset is applied at runtime by ``GuideDataset.sample``.

    Parameters
    ----------
    planner : LocomanipulationFramePlanner
        Fully initialised planner (IRIS regions and motion frame sequence loaded).
    p_init_nominal : dict[str, np.ndarray]
        Nominal world-frame starting positions keyed by planner frame name.
        The torso entry is stored as the shift reference.
    T_min : float
        Lower bound of the uniform duration distribution (default 2.0 s).
    T_max : float
        Upper bound of the uniform duration distribution (default 3.0 s).
    w_rigid_mid_fixed : float
        Value for the fixed middle entry of w_rigid (index 1).
    n_waypoints : int
        Uniform time samples per guide.
    seed : int
        RNG seed for reproducibility.
    save_path : str
        Destination ``.npz`` file.
    """
    rng = np.random.default_rng(seed)

    # --- Parameter grids --------------------------------------------------
    alpha_grid = np.zeros((n_alpha, 3), dtype=np.float64)
    alpha_grid[:, 0] = rng.uniform(0.0, 1.0, n_alpha)
    alpha_grid[:, 1] = rng.uniform(0.0, 1.0, n_alpha)
    alpha_grid[:, 2] = rng.uniform(0.0, 0.2, n_alpha)

    w_rigid_grid = np.zeros((n_w_rigid, 3), dtype=np.float64)
    w_rigid_grid[:, 0] = rng.uniform(0.0, 1.0, n_w_rigid)
    w_rigid_grid[:, 1] = w_rigid_mid_fixed
    w_rigid_grid[:, 2] = rng.uniform(0.0, 1.0, n_w_rigid)

    # --- Metadata ---------------------------------------------------------
    frame_names: list[str] = list(p_init_nominal.keys())
    body_names:  list[str] = [FRAME_TO_BODY.get(fn, fn) for fn in frame_names]
    hand_mask = np.array([fn in _CONTACT_FRAMES for fn in frame_names], dtype=bool)
    p_nominal_torso = p_init_nominal["torso"].astype(np.float32)

    # --- Generation loop --------------------------------------------------
    guides_list:   list[np.ndarray] = []
    alpha_used:    list[np.ndarray] = []
    w_rigid_used:  list[np.ndarray] = []
    T_used:        list[float] = []

    total = n_alpha * n_w_rigid
    for i, alpha in enumerate(alpha_grid):
        for j, w_rigid in enumerate(w_rigid_grid):
            count = i * n_w_rigid + j + 1
            T_i = float(rng.uniform(T_min, T_max))
            print(f"[{count:3d}/{total}]  alpha={np.round(alpha, 3)}  "
                  f"w_rigid={np.round(w_rigid, 3)}  T={T_i:.2f}s", end="  ")
            try:
                planner.plan_iris(
                    p_init_nominal, T_i,
                    alpha=alpha.tolist(),
                    w_rigid=w_rigid,
                )
                n_phases = len(planner.fixed_frames)
                wp = _sample_paths(planner.path, n_waypoints, T_i * n_phases)  # (F, W, 3)
                guides_list.append(wp)
                alpha_used.append(alpha.copy())
                w_rigid_used.append(w_rigid.copy())
                T_used.append(T_i)
                print("ok")
            except Exception as exc:
                print(f"FAILED ({exc})")

    if not guides_list:
        raise RuntimeError("Every guide generation attempt failed. "
                           "Check planner, IRIS regions, and safe_pnt_lst.")

    guides_arr = np.stack(guides_list, axis=0)  # (n_guides, F, W, 3)
    T_plan_arr = np.array(T_used, dtype=np.float32)

    np.savez(
        save_path,
        guides=guides_arr.astype(np.float32),
        alpha_grid=np.array(alpha_used, dtype=np.float32),
        w_rigid_grid=np.array(w_rigid_used, dtype=np.float32),
        p_init_nominal_torso=p_nominal_torso,
        T_plan=T_plan_arr.mean(),      # scalar mean for backward compat
        T_plan_arr=T_plan_arr,         # per-guide durations (n_guides,)
        frame_names=np.array(frame_names),
        body_names=np.array(body_names),
        hand_mask=hand_mask,
    )
    n = len(guides_list)
    mb = guides_arr.nbytes / 1e6
    print(f"\nSaved {n}/{total} guides → {save_path}  "
          f"(shape {guides_arr.shape}, {mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Runtime dataset
# ---------------------------------------------------------------------------

class GuideDataset:
    """GPU-resident guide library with O(1) sample-and-shift at env reset.

    The full guide tensor is loaded onto the GPU once and stays there for the
    entire training run.  All operations (random selection, rigid shift,
    time-indexed lookup) are differentiable GPU tensor operations with no
    CPU round-trips.

    Attributes
    ----------
    guides : torch.Tensor, shape (n_guides, n_frames, n_waypoints, 3)
        Pre-computed guide waypoints for all (alpha, w_rigid) combinations.
    p_init_nominal_torso : torch.Tensor, shape (3,)
        Nominal torso position used as shift reference during generation.
    T_plan : float
        Guide duration in seconds.
    frame_names : list[str]
        Planner frame names in dataset index order.
    body_names : list[str]
        Corresponding URDF body names.
    body_name_to_idx : dict[str, int]
        Lookup from URDF body name to frame axis index.
    shift_mask : torch.Tensor, shape (n_frames,)
        1.0 for non-contact frames (shift applied), 0.0 for contact frames.
    """

    def __init__(self, path: str, device: str = "cuda"):
        data = np.load(path, allow_pickle=True)

        self.guides = torch.from_numpy(data["guides"]).float().to(device)
        self.p_init_nominal_torso = (
            torch.from_numpy(data["p_init_nominal_torso"]).float().to(device)
        )
        self.T_plan = float(data["T_plan"])
        if "T_plan_arr" in data:
            self.T_plan_arr = torch.from_numpy(data["T_plan_arr"]).float().to(device)
        else:
            self.T_plan_arr = None

        frame_names: list[str] = data["frame_names"].tolist()
        body_names:  list[str] = data["body_names"].tolist()
        hand_mask_np: np.ndarray = data["hand_mask"]

        self.frame_names = frame_names
        self.body_names  = body_names
        self.frame_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(frame_names)}
        self.body_name_to_idx:  dict[str, int] = {n: i for i, n in enumerate(body_names)}

        # 1.0 → apply shift;  0.0 → keep fixed (contact frames)
        self.shift_mask = (
            ~torch.from_numpy(hand_mask_np)
        ).float().to(device)                           # (F,)

        self.n_guides, self.n_frames, self.n_waypoints, _ = self.guides.shape
        self.device = device

    # ------------------------------------------------------------------
    # Env-reset: sample + shift
    # ------------------------------------------------------------------

    def sample(
        self,
        num_envs: int,
        p_init_torso_actual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and shift guides for a batch of environments.

        A guide is drawn uniformly at random for each environment.
        Non-contact frames (torso, feet, knees) are translated by the
        vector from the nominal torso spawn position to the actual one.
        Contact frames (LH, RH) are left unchanged.

        Parameters
        ----------
        num_envs : int
            Number of environments (or reset environments) to sample for.
        p_init_torso_actual : torch.Tensor, shape (num_envs, 3)
            Actual world-frame torso position at env spawn, on GPU.

        Returns
        -------
        guides : torch.Tensor, shape (num_envs, n_frames, n_waypoints, 3)
        T_per_env : torch.Tensor, shape (num_envs,)
            Per-environment traversal duration (seconds), drawn from the
            stored ``T_plan_arr`` if available, else broadcast ``T_plan``.
        """
        idx = torch.randint(0, self.n_guides, (num_envs,), device=self.device)
        guides = self.guides[idx].clone()  # (E, F, W, 3)

        # Translation offset per environment
        shift = p_init_torso_actual - self.p_init_nominal_torso  # (E, 3)

        # Expand shift and mask for broadcasting:
        #   shift  : (E, 3) → (E, 1, 1, 3)
        #   mask   : (F,)   → (1, F, 1, 1)
        shift_b = shift[:, None, None, :].expand_as(guides)       # (E, F, W, 3)
        mask_b  = self.shift_mask[None, :, None, None]            # (1, F, 1, 1)
        guides  = guides + shift_b * mask_b

        if self.T_plan_arr is not None:
            T_per_env = self.T_plan_arr[idx]                      # (E,)
        else:
            T_per_env = torch.full((num_envs,), self.T_plan, device=self.device)

        return guides, T_per_env

    # ------------------------------------------------------------------
    # Per-step: time-indexed position lookup
    # ------------------------------------------------------------------

    def query_targets(
        self,
        guides: torch.Tensor,
        t_elapsed: torch.Tensor,
        T_per_env: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-frame position targets at the current episode time.

        Uses nearest-waypoint indexing (no interpolation).

        Parameters
        ----------
        guides : torch.Tensor, shape (num_envs, n_frames, n_waypoints, 3)
        t_elapsed : torch.Tensor, shape (num_envs,)
            Time elapsed since episode start, in seconds.
        T_per_env : torch.Tensor, shape (num_envs,), optional
            Per-environment traversal duration returned by ``sample``.
            Falls back to the scalar ``self.T_plan`` when not provided.

        Returns
        -------
        torch.Tensor, shape (num_envs, n_frames, 3)
        """
        T = T_per_env if T_per_env is not None else self.T_plan
        t_norm  = (t_elapsed / T).clamp(0.0, 1.0)                 # (E,)
        wp_idx  = (t_norm * (self.n_waypoints - 1)).long()         # (E,)

        # gather along waypoint dim (dim=2)
        idx_exp = wp_idx[:, None, None, None].expand(
            -1, self.n_frames, 1, 3
        )                                                           # (E, F, 1, 3)
        targets = guides.gather(2, idx_exp).squeeze(2)             # (E, F, 3)
        return targets
