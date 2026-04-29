"""MDP terms for the G1 residual guide-tracking environment.

Guide terms are implemented against ``GuideDataset``, which is loaded once at
env startup and stored on ``env.guide_dataset``.  The per-env guide assignment
tensor ``env.guide_waypoints`` is populated at env reset by the
``reset_guide_assignment`` event term and queried each policy step.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


# ---------------------------------------------------------------------------
# Event terms
# ---------------------------------------------------------------------------

def startup_guide_init(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Load the guide dataset and initialise per-env guide buffers.

    Called once with ``mode="startup"`` before any resets occur.
    Attaches ``env.guide_dataset`` (GuideDataset) and
    ``env.guide_waypoints`` (num_envs, n_frames, n_waypoints, 3) to the env.
    """
    from .guide_dataset import GuideDataset

    dataset_path: str = env.cfg.GUIDE_DATASET_PATH
    env.guide_dataset = GuideDataset(dataset_path, device=str(env.device))

    # Allocate the per-env guide buffer
    ds = env.guide_dataset
    env.guide_waypoints = torch.zeros(
        env.num_envs, ds.n_frames, ds.n_waypoints, 3,
        device=env.device,
    )

    # Allocate per-env duration buffer
    env.guide_T_per_env = torch.full(
        (env.num_envs,), ds.T_plan, device=env.device
    )

    # Sample initial guides for all environments
    robot: Articulation = env.scene["robot"]
    torso_idx = robot.find_bodies("torso_link")[0][0]
    p_torso = robot.data.body_pos_w[:, torso_idx, :3]  # (num_envs, 3)
    guides, T_per_env = ds.sample(env.num_envs, p_torso)
    env.guide_waypoints[:] = guides
    env.guide_T_per_env[:] = T_per_env


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
    p_torso = robot.data.body_pos_w[env_ids, torso_idx, :3]  # (n_reset, 3)

    new_guides, new_T = env.guide_dataset.sample(n_reset, p_torso)
    env.guide_waypoints[env_ids] = new_guides
    env.guide_T_per_env[env_ids] = new_T


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
        env.guide_waypoints, t_elapsed, T_per_env=env.guide_T_per_env
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
        env.guide_waypoints, t_elapsed, T_per_env=env.guide_T_per_env
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
        env.guide_waypoints, t_elapsed, T_per_env=env.guide_T_per_env
    )

    current = robot.data.body_pos_w[:, body_idx, :3]                # (num_envs, 3)
    target  = all_targets[:, frame_idx, :]                          # (num_envs, 3)

    sq_err = torch.sum((current - target) ** 2, dim=-1)             # (num_envs,)
    return torch.exp(-sq_err / (sigma ** 2))                        # (num_envs,)


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
    if not hasattr(env, "guide_waypoints"):
        return torch.zeros(env.num_envs, device=env.device)

    ds = env.guide_dataset
    robot: Articulation = env.scene[asset_cfg.name]

    # Last waypoint encodes the final goal position for each env's guide.
    goal_positions = env.guide_waypoints[:, :, -1, :]              # (E, F, 3)

    total_dist = torch.zeros(env.num_envs, device=env.device)
    for body_name in body_names:
        frame_idx = ds.body_name_to_idx[body_name]
        body_idx  = robot.find_bodies(body_name)[0][0]
        current   = robot.data.body_pos_w[:, body_idx, :3]
        goal      = goal_positions[:, frame_idx, :]
        total_dist = total_dist + torch.norm(current - goal, dim=-1)

    return total_dist * env.termination_manager.dones.float()
