"""Time-varying commands derived from the Bezier guide dataset.

GuideTorsoVelocityCommand: replaces the fixed UniformVelocityCommandCfg so
the base policy tracks a velocity consistent with the guide trajectory.

GuideHandPositionCommand: replaces the fixed UniformPoseCommandCfg so the
base policy is given the hand target position read directly from the guide
trajectory (expressed in the robot's heading-aligned body frame).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GuideTorsoVelocityCommand(CommandTerm):
    """Base velocity command derived from the analytical Bezier torso guide velocity.

    At each policy step the torso guide velocity is evaluated by differentiating
    the Bezier curve (see GuideDataset.query_velocities), then projected from
    the world XY plane into the robot's heading-aligned body frame:

        v_x_body =  cos(yaw) * v_x_world + sin(yaw) * v_y_world
        v_y_body = -sin(yaw) * v_x_world + cos(yaw) * v_y_world

    The command tensor (num_envs, 3) has the same layout as the standard
    UniformVelocityCommand: [v_x_body, v_y_body, omega_z].  The angular
    component is always 0 because the guide assumes straight-ahead traversal.

    Once the guide trajectory ends the command drops to zero so the base
    policy is no longer driven forward after the crossing is complete.
    """

    cfg: GuideTorsoVelocityCommandCfg

    def __init__(self, cfg: GuideTorsoVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(env.num_envs, 3, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """Velocity command in robot body frame.  Shape (num_envs, 3)."""
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # Command is fully determined by the guide at each step; nothing to resample.
        pass

    def _update_command(self):
        if not hasattr(self._env, "guide_dataset"):
            return

        ds = self._env.guide_dataset
        torso_frame_idx = ds.body_name_to_idx["torso_link"]

        t_elapsed = self._env.episode_length_buf * self._env.step_dt   # (E,)

        # World-frame guide velocity for all frames, shape (E, n_frames, 3).
        vel_w = ds.query_velocities(
            self._env.guide_ctrl_pts,
            self._env.guide_transition_times,
            t_elapsed,
        )
        v_world_xy = vel_w[:, torso_frame_idx, :2]   # (E, 2)

        # Rotate world-frame XY velocity into the robot's heading-aligned body frame.
        yaw     = self.robot.data.heading_w            # (E,)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        v_x_b = cos_yaw * v_world_xy[:, 0] + sin_yaw * v_world_xy[:, 1]
        v_y_b = -sin_yaw * v_world_xy[:, 0] + cos_yaw * v_world_xy[:, 1]

        clamp = self.cfg.max_speed
        self._command[:, 0] = v_x_b.clamp(-clamp, clamp)
        self._command[:, 1] = v_y_b.clamp(-clamp, clamp)
        self._command[:, 2] = 0.0


@configclass
class GuideTorsoVelocityCommandCfg(CommandTermCfg):
    """Configuration for GuideTorsoVelocityCommand."""

    class_type: type = GuideTorsoVelocityCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never resample — the command is derived from the guide at every step."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    max_speed: float = 1.5
    """Safety clamp on the commanded body-frame speed [m/s]."""


class GuideHandPositionCommand(CommandTerm):
    """End-effector position command derived from the guide trajectory.

    At each policy step the guide position for the specified hand body is
    queried from the dataset via ``GuideDataset.query_targets``, then
    expressed in the robot's heading-aligned body frame:

        dp_world = p_hand_world - p_base_world
        p_x_body =  cos(yaw) * dp_x + sin(yaw) * dp_y
        p_y_body = -sin(yaw) * dp_x + cos(yaw) * dp_y
        p_z_body =  dp_z   (unchanged)

    The command tensor is (num_envs, 7) = [pos_x, pos_y, pos_z, quat_w, quat_x,
    quat_y, quat_z], matching the UniformPoseCommand layout exactly so that the
    base policy checkpoint sees the same observation dimensionality.  The
    orientation is kept as the identity quaternion (w=1, xyz=0) because the
    guide only prescribes positions.

    Once the guide trajectory ends the command freezes at the final guide
    position (``query_targets`` clamps t to [0, T_total]).
    """

    cfg: "GuideHandPositionCommandCfg"

    def __init__(self, cfg: "GuideHandPositionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] — identity orientation
        self._command = torch.zeros(env.num_envs, 7, device=env.device)
        self._command[:, 3] = 1.0   # quat_w = 1 (identity)

    @property
    def command(self) -> torch.Tensor:
        """Hand pose in robot heading-aligned body frame.  Shape (num_envs, 7)."""
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        if not hasattr(self._env, "guide_dataset"):
            return

        ds = self._env.guide_dataset
        frame_idx = ds.body_name_to_idx[self.cfg.body_name]

        t_elapsed = self._env.episode_length_buf * self._env.step_dt   # (E,)

        # World-frame guide positions for all frames, shape (E, n_frames, 3).
        pos_w = ds.query_targets(
            self._env.guide_ctrl_pts,
            self._env.guide_transition_times,
            t_elapsed,
        )
        p_hand_w = pos_w[:, frame_idx, :]   # (E, 3)

        # Position of the robot base in world frame.
        p_base_w = self.robot.data.root_pos_w   # (E, 3)

        # Relative position in world frame.
        dp = p_hand_w - p_base_w   # (E, 3)

        # Rotate XY into the robot's heading-aligned body frame.
        yaw     = self.robot.data.heading_w    # (E,)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        self._command[:, 0] =  cos_yaw * dp[:, 0] + sin_yaw * dp[:, 1]
        self._command[:, 1] = -sin_yaw * dp[:, 0] + cos_yaw * dp[:, 1]
        self._command[:, 2] =  dp[:, 2]
        # Indices 3-6 remain the identity quaternion set at construction.


@configclass
class GuideHandPositionCommandCfg(CommandTermCfg):
    """Configuration for GuideHandPositionCommand."""

    class_type: type = GuideHandPositionCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Never resample — the command is derived from the guide at every step."""

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    body_name: str = MISSING
    """URDF body name whose guide position to track (e.g. 'left_rubber_hand')."""
