# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

# from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.rough_env_cfg import DigitRewards, DigitRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg

from isaaclab.managers import TerminationTermCfg

ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_.*"
]


LEG_JOINT_NAMES = [
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_.*",
]

@configclass
class G1LocoManipRewards(G1Rewards):
    joint_deviation_arms = None
    no_jumps = RewTerm(
        func=mdp.desired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )


    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*"])},
    )

    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "command_name": "right_ee_pose",
        },
    )

    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "command_name": "right_ee_pose",
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )


@configclass
class G1LocoManipObservations:
    """Configuration for the G1 Locomanipulation environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class G1LocoManipCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 5.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(0.05, 0.30),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(-0.30, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class G1Events(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class G1LocoManipEnvCfg(G1RoughEnvCfg):
    rewards: G1LocoManipRewards = G1LocoManipRewards()
    observations: G1LocoManipObservations = G1LocoManipObservations()
    commands: G1LocoManipCommands = G1LocoManipCommands()

    def __post_init__(self):
        super().__post_init__()

        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt

        # Custom terminations
        self.terminations.torso_height.minimum_height = 0.2
        self.terminations.base_contact = None
        # self.terminations.base_orientation = TerminationTermCfg(
        #     func=mdp.bad_orientation,
        #     params={"limit_angle": 0.7},
        # )

        self.episode_length_s = 7.0

        # Rewards:
        self.rewards.flat_orientation_l2.weight = -10.5
        self.rewards.termination_penalty.weight = -100.0
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None
        # Remove pushing robot
        self.events.push_robot = None


class G1LocoManipEnvCfg_PLAY(G1LocoManipEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
