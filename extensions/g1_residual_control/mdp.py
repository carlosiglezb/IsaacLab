"""MDP terms for the G1 residual guide-tracking environment.

Guide terms are implemented against ``GuideDataset``, which is loaded once at
env startup and stored on ``env.guide_dataset``.  The per-env guide assignment
tensors ``env.guide_ctrl_pts`` and ``env.guide_transition_times`` are
populated at env reset by the ``reset_guide_assignment`` event term
and queried each policy step via vectorised Bezier evaluation.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .g1_planner_constants import CONTACT_SURFACES, N_PHASES, TARGET_TO_CURRENT_TORSO_OFFSET
from .guide_dataset import FRAME_TO_BODY


# ---------------------------------------------------------------------------
# Event terms
# ---------------------------------------------------------------------------

def startup_guide_init(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Load the guide dataset and initialise per-env guide buffers.

    Called once with ``mode="startup"`` before any resets occur.
    Attaches ``env.guide_dataset`` (GuideDataset),
    ``env.guide_ctrl_pts`` (num_envs, n_frames, n_segments, degree+1, 3), and
    ``env.guide_transition_times`` (num_envs, n_segments+1) to the env.
    """
    from .guide_dataset import GuideDataset

    dataset_path: str = env.cfg.GUIDE_DATASET_PATH
    env.guide_dataset = GuideDataset(dataset_path, device=str(env.device))

    # Allocate the per-env guide buffers
    ds = env.guide_dataset
    env.guide_ctrl_pts = torch.zeros(
        env.num_envs, ds.n_frames, ds.n_segments, ds.degree_plus_1, 3,
        device=env.device,
    )
    env.guide_transition_times = torch.zeros(
        env.num_envs, ds.n_segments + 1,
        device=env.device,
    )

    # Allocate per-env duration buffer
    env.guide_T_per_env = torch.full(
        (env.num_envs,), ds.T_plan, device=env.device
    )

    # Sample initial guides for all environments
    robot: Articulation = env.scene["robot"]
    torso_idx = robot.find_bodies("torso_link")[0][0]
    p_torso_w = robot.data.body_pos_w[:, torso_idx, :3]        # (num_envs, 3)
    p_torso = p_torso_w - env.scene.env_origins                 # env-local frame
    ctrl_pts, trans_times, T_per_env = ds.sample(env.num_envs, p_torso)
    # Shift from env-local → world frame so guide targets match body_pos_w.
    ctrl_pts = ctrl_pts + env.scene.env_origins[:, None, None, None, :]  # (E, 1, 1, 1, 3)
    env.guide_ctrl_pts[:] = ctrl_pts
    env.guide_transition_times[:] = trans_times
    env.guide_T_per_env[:] = T_per_env

    # Temporary offset correction: torso guide is off by TARGET_TO_CURRENT_TORSO_OFFSET.
    # Remove once the dataset generation applies this offset at source.
    _torso_fidx = ds.frame_name_to_idx["torso"]
    _torso_offset = torch.tensor(TARGET_TO_CURRENT_TORSO_OFFSET, device=env.device, dtype=torch.float32)
    env.guide_ctrl_pts[:, _torso_fidx] += _torso_offset


def reset_guide_assignment(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Resample guides for environments that just reset.

    Called with ``mode="reset"`` at each episode boundary.  Only the
    environments in ``env_ids`` receive new guide assignments; all others
    keep their current guides unchanged.
    """
    if not hasattr(env, "guide_dataset"):
        return

    n_reset = len(env_ids)
    robot: Articulation = env.scene["robot"]
    torso_idx = robot.find_bodies("torso_link")[0][0]
    p_torso_w = robot.data.body_pos_w[env_ids, torso_idx, :3]  # (n_reset, 3)
    p_torso = p_torso_w - env.scene.env_origins[env_ids]        # env-local frame

    new_ctrl_pts, new_trans_times, new_T = env.guide_dataset.sample(n_reset, p_torso)
    # Shift from env-local → world frame so guide targets match body_pos_w.
    new_ctrl_pts = new_ctrl_pts + env.scene.env_origins[env_ids, None, None, None, :]
    env.guide_ctrl_pts[env_ids] = new_ctrl_pts
    env.guide_transition_times[env_ids] = new_trans_times
    env.guide_T_per_env[env_ids] = new_T

    # Temporary offset correction: torso guide is off by TARGET_TO_CURRENT_TORSO_OFFSET.
    # Remove once the dataset generation applies this offset at source.
    _torso_fidx = env.guide_dataset.frame_name_to_idx["torso"]
    _torso_offset = torch.tensor(TARGET_TO_CURRENT_TORSO_OFFSET, device=env.device, dtype=torch.float32)
    env.guide_ctrl_pts[env_ids, _torso_fidx] += _torso_offset


    if not hasattr(env, '_reset_alignment_logged'):
        env._reset_alignment_logged = True
        from .guide_dataset import FRAME_TO_BODY

        t0 = torch.zeros(len(env_ids), device=env.device)
        ctrl_sub = env.guide_ctrl_pts[env_ids]
        trans_sub = env.guide_transition_times[env_ids]
        targets_t0 = env.guide_dataset.query_targets(ctrl_sub, trans_sub, t0)
        print("\n[Reset alignment] actual body pos vs guide t=0 target (env-local frame):")
        for i, (frame_name, body_name) in enumerate(FRAME_TO_BODY.items()):
            body_ids, _ = robot.find_bodies(body_name)
            actual_w = robot.data.body_pos_w[env_ids[0], body_ids[0], :3]
            guide_w = targets_t0[0, i]
            err = (actual_w - guide_w).norm().item()
            actual_loc = actual_w - env.scene.env_origins[env_ids[0]]
            guide_loc = guide_w - env.scene.env_origins[env_ids[0]]
            print(f"  {frame_name:8s}: actual={actual_loc.tolist()}  guide={guide_loc.tolist()}  err={err:.4f}m")
        print()


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------

def guide_body_target_pos(
    env: ManagerBasedRLEnv,
    body_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame guide position target for a single body link.

    Returns
    -------
    torch.Tensor, shape (num_envs, 3)
    """
    if not hasattr(env, "guide_dataset"):
        return torch.zeros(env.num_envs, 3, device=env.device)

    ds = env.guide_dataset
    frame_idx: int = ds.body_name_to_idx[body_name]

    t_elapsed = env.episode_length_buf * env.step_dt
    targets = ds.query_targets(
        env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
    )
    return targets[:, frame_idx, :]


def guide_tracking_error(
    env: ManagerBasedRLEnv,
    body_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position tracking error (current body pos − guide target) for one or more links.

    Concatenates errors along the last dimension.

    Returns
    -------
    torch.Tensor, shape (num_envs, 3 * len(body_names))
    """
    if not hasattr(env, "guide_dataset"):
        return torch.zeros(env.num_envs, 3 * len(body_names), device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    t_elapsed = env.episode_length_buf * env.step_dt
    all_targets = ds.query_targets(
        env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
    )

    errors: list[torch.Tensor] = []
    for body_name in body_names:
        frame_idx = ds.body_name_to_idx[body_name]
        body_idx  = robot.find_bodies(body_name)[0][0]
        current   = robot.data.body_pos_w[:, body_idx, :3]          # (num_envs, 3)
        target    = all_targets[:, frame_idx, :]                     # (num_envs, 3)
        errors.append(current - target)

    return torch.cat(errors, dim=-1)                                 # (num_envs, 3*N)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------

def guide_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    body_name: str,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential position-tracking reward for a single body link.

    Reward = exp(-||p_current - p_guide||² / sigma²)

    Returns 1.0 at zero error, decaying smoothly to 0 as the error grows.

    Parameters
    ----------
    body_name : str
        URDF link name, e.g. ``'left_palm_link'``.
    sigma : float
        Kernel width in metres.  Smaller → sharper peak.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    if not hasattr(env, "guide_dataset"):
        return torch.zeros(env.num_envs, device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    frame_idx = ds.body_name_to_idx[body_name]
    body_idx  = robot.find_bodies(body_name)[0][0]

    t_elapsed   = env.episode_length_buf * env.step_dt
    all_targets = ds.query_targets(
        env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
    )

    current = robot.data.body_pos_w[:, body_idx, :3]                # (num_envs, 3)
    target  = all_targets[:, frame_idx, :]                          # (num_envs, 3)

    sq_err = torch.sum((current - target) ** 2, dim=-1)             # (num_envs,)
    return torch.exp(-sq_err / (sigma ** 2))                        # (num_envs,)


def guide_vel_tracking_exp(
    env: ManagerBasedRLEnv,
    body_name: str,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential velocity-tracking reward for a single body link.

    Reward = exp(-||v_current - v_guide||² / sigma²)

    ``v_guide`` is the analytical time derivative of the Bezier curve evaluated
    at the current episode time (world frame, m/s).  ``v_current`` is the
    world-frame linear velocity of the body's centre of mass.

    A wider sigma (default 0.25 m/s) is appropriate here because velocity
    errors are naturally larger in scale than position errors and the reward
    should provide useful gradient several tenths of a m/s away from the target.

    Parameters
    ----------
    body_name : str
        URDF link name, e.g. ``'left_ankle_roll_link'``.
    sigma : float
        Kernel width in m/s.  Smaller → sharper peak.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    if not hasattr(env, "guide_dataset"):
        return torch.zeros(env.num_envs, device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    frame_idx = ds.body_name_to_idx[body_name]
    body_idx  = robot.find_bodies(body_name)[0][0]

    t_elapsed  = env.episode_length_buf * env.step_dt
    all_vel    = ds.query_velocities(
        env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
    )  # (num_envs, n_frames, 3)

    current_vel = robot.data.body_lin_vel_w[:, body_idx, :]          # (num_envs, 3)
    target_vel  = all_vel[:, frame_idx, :]                           # (num_envs, 3)

    sq_err = torch.sum((current_vel - target_vel) ** 2, dim=-1)      # (num_envs,)
    return torch.exp(-sq_err / (sigma ** 2))                         # (num_envs,)


def body_position_progress(
    env: ManagerBasedRLEnv,
    body_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense position-progress reward across multiple tracked bodies.

    Implements the RRL/ReLIC progress signal:

        r_t = Σ_i [ d_i(t-1) − d_i(t) ]

    where d_i(t) = ‖p_i(t) − g_i(t)‖₂ is the L2 distance from body i's
    world-frame position to its time-interpolated guide target.  The reward is
    positive when the bodies collectively move closer to the guide and negative
    when they drift away.

    Returns 0 on the first step of every episode so that the stale previous
    buffer from a prior episode cannot corrupt the signal after a reset.

    Parameters
    ----------
    body_names : list[str]
        URDF link names to include in the progress sum, e.g.
        ``["torso_link", "left_rubber_hand", ...]``.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    if not hasattr(env, "guide_dataset"):
        return torch.zeros(env.num_envs, device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    t_elapsed = env.episode_length_buf * env.step_dt
    all_targets = ds.query_targets(
        env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
    )  # (num_envs, n_frames, 3)

    # Compute current L2 distance for each tracked body
    curr_dists = torch.zeros(env.num_envs, len(body_names), device=env.device)
    for k, body_name in enumerate(body_names):
        frame_idx = ds.body_name_to_idx[body_name]
        body_idx = robot.find_bodies(body_name)[0][0]
        current = robot.data.body_pos_w[:, body_idx, :3]   # (num_envs, 3)
        target = all_targets[:, frame_idx, :]               # (num_envs, 3)
        curr_dists[:, k] = torch.norm(current - target, dim=-1)

    # Lazy-initialise the buffer; return zero reward on the very first call.
    if not hasattr(env, "_pos_prog_prev_dists"):
        env._pos_prog_prev_dists = curr_dists.clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev_dists = env._pos_prog_prev_dists  # (num_envs, n_bodies)

    # Summed progress; positive = getting closer to the guide.
    progress = (prev_dists - curr_dists).sum(dim=-1)  # (num_envs,)

    # Suppress the reward on the first step after each per-env reset so the
    # stale prev_dists from the previous episode cannot generate a spurious
    # signal.  episode_length_buf is incremented before reward computation, so
    # it equals 1 on the first step of a new episode.
    progress = progress.masked_fill(env.episode_length_buf == 1, 0.0)

    # Advance the buffer for all environments.
    env._pos_prog_prev_dists = curr_dists.clone()

    return progress


def torso_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for forward (world-x) linear velocity of the robot root.

    Only positive x-velocity is rewarded — backward motion returns 0.0.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(robot.data.root_lin_vel_w[:, 0], min=0.0)


def goal_not_reached_penalty(
    env: ManagerBasedRLEnv,
    body_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminal penalty proportional to remaining L2 distance to final guide goals.

    Fires only at episode end (timeout or early termination).  Final goal
    positions are read from the last waypoint of each env's assigned guide,
    so they automatically incorporate per-guide XY offsets.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    if not hasattr(env, "guide_ctrl_pts"):
        return torch.zeros(env.num_envs, device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    # Last control point of last segment = endpoint of the Bezier trajectory.
    goal_positions = env.guide_ctrl_pts[:, :, -1, -1, :]           # (E, F, 3)

    total_dist = torch.zeros(env.num_envs, device=env.device)
    for body_name in body_names:
        frame_idx = ds.body_name_to_idx[body_name]
        body_idx  = robot.find_bodies(body_name)[0][0]
        current   = robot.data.body_pos_w[:, body_idx, :3]
        goal      = goal_positions[:, frame_idx, :]
        total_dist = total_dist + torch.norm(current - goal, dim=-1)

    return total_dist * env.termination_manager.dones.float()


def contact_schedule_violation(
    env: ManagerBasedRLEnv,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise feet that remain near the floor during their swing phase.

    ``CONTACT_SURFACES[phase]`` lists the bodies that are braced against a
    physical surface in phase *p*.  Any foot (LF/RF) absent from that dict is
    in swing and should be elevated.  This term fires an exponential penalty
    proportional to how close the swing foot is to the floor:

        penalty_f = exp(-dz_f² / σ²),   dz_f = foot_z_world − floor_z_world

    The floor reference height (0.043 m) matches the ankle contact value in
    ``CONTACT_SURFACES``.  The penalty is 1.0 when the foot is at floor level
    and decays to ~0.018 at dz = 2σ (≈10 cm for the default σ = 5 cm).

    Parameters
    ----------
    sigma : float
        Kernel half-width in metres.  Smaller → gradient concentrated near floor.

    Returns
    -------
    torch.Tensor, shape (num_envs,)
        Positive values; set a negative ``weight`` in RewardsCfg to penalise.
    """
    if not hasattr(env, "guide_T_per_env"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]

    t_elapsed   = env.episode_length_buf * env.step_dt
    T_per_phase = env.guide_T_per_env / N_PHASES
    phase_idx   = (t_elapsed / T_per_phase).long().clamp(0, N_PHASES - 1)   # (E,)

    # Ankle height at floor contact (from CONTACT_SURFACES floor entries).
    _FLOOR_Z_LOCAL = 0.043
    floor_z_w = env.scene.env_origins[:, 2] + _FLOOR_Z_LOCAL               # (E,)

    FOOT_FRAMES = {"LF": FRAME_TO_BODY["LF"], "RF": FRAME_TO_BODY["RF"]}

    penalty = torch.zeros(env.num_envs, device=env.device)

    for p, surfaces in enumerate(CONTACT_SURFACES):
        env_mask = (phase_idx == p)                                          # (E,) bool
        if not env_mask.any():
            continue

        for frame_name, body_name in FOOT_FRAMES.items():
            if frame_name in surfaces:
                continue   # foot is braced against a surface this phase — not in swing

            body_idx = robot.find_bodies(body_name)[0][0]
            foot_z_w = robot.data.body_pos_w[:, body_idx, 2]                # (E,)

            dz = foot_z_w - floor_z_w                                       # (E,), ≥ 0
            penalty += torch.exp(-(dz ** 2) / sigma ** 2) * env_mask.float()

    return penalty


def contact_surface_proximity(
    env: ManagerBasedRLEnv,
    sigma: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phase-gated surface proximity reward for hands and feet.

    For each motion phase the planner defines which bodies should be braced
    against a physical surface (``CONTACT_SURFACES`` in g1_planner_constants).
    This reward fires an exponential kernel along the single constrained axis
    for each such body:

        r = Σ_{b ∈ contacts(phase)} exp(-(body_coord[axis] - surface_w[axis])² / σ²)

    where ``surface_w`` is the surface coordinate converted to world frame by
    adding ``env.scene.env_origins[:, axis]``.  The sum is over all contact
    bodies active in the current phase; the caller controls the overall scale
    via the ``RewardsCfg`` weight.

    Parameters
    ----------
    sigma : float
        Kernel half-width in metres.  Reward = 1.0 at zero error, ~0.37 at
        one ``sigma``, ~0.018 at two ``sigma``.  Default 0.05 m (5 cm).

    Returns
    -------
    torch.Tensor, shape (num_envs,)
    """
    if not hasattr(env, "guide_T_per_env"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]

    t_elapsed = env.episode_length_buf * env.step_dt          # (E,)
    T_per_phase = env.guide_T_per_env / N_PHASES              # (E,)
    phase_idx = (t_elapsed / T_per_phase).long().clamp(0, N_PHASES - 1)  # (E,)

    reward = torch.zeros(env.num_envs, device=env.device)

    for p, surfaces in enumerate(CONTACT_SURFACES):
        env_mask = (phase_idx == p)                           # (E,) bool
        if not env_mask.any():
            continue

        for frame_name, surface in surfaces.items():
            body_name = FRAME_TO_BODY[frame_name]
            body_idx = robot.find_bodies(body_name)[0][0]

            # Body coordinate along the constrained axis in world frame.
            body_coord = robot.data.body_pos_w[:, body_idx, surface.axis]  # (E,)

            # Surface coordinate in world frame.
            surface_w = env.scene.env_origins[:, surface.axis] + surface.value  # (E,)

            d_sq = (body_coord - surface_w) ** 2              # (E,)
            reward += torch.exp(-d_sq / sigma**2) * env_mask.float()

    return reward
