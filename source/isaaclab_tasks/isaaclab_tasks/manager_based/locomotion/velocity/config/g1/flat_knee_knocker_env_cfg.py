# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_knee_knocker_env_cfg import G1RoughKneeKnockerEnvCfg


@configclass
class G1FlatKneeKnockerEnvCfg(G1RoughKneeKnockerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # reduce sim time to prevent avoiding obstacle
        self.episode_length_s = 7.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 0.1
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -1e-5
        self.rewards.dof_acc_l2.weight = -1.0e-8
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.feet_air_time.params["threshold"] = 1.5
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*"]
        )
        self.rewards.joint_deviation_arms.weight = -0.001
        self.rewards.track_lin_vel_xy_exp.weight = 5.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.05

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.02, 0.02)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.02, 0.02)


class G1FlatKneeKnockerEnvCfg_PLAY(G1FlatKneeKnockerEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.episode_length_s = 5.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
