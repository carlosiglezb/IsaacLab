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
from isaaclab.envs.mdp.rewards import feet_flat_orientation
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

# from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.rough_env_cfg import DigitRewards, DigitRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg

from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains.config.mildly_rough import MILDLY_ROUGH_TERRAINS_CFG  # isort: skip

from isaaclab.envs.mdp.rewards import primitive_distance_penalty

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

    flat_feet = RewTerm(
        func=feet_flat_orientation,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
        }
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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
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
        body_name="left_rubber_hand",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35),
            pos_y=(0.05, 0.35),
            pos_z=(-0.10, 0.40),
            roll=(-0.5, 0.5),
            pitch=(-0.5, 0.5),
            yaw=(-math.pi / 2.0 - 0.1, 0.5),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_rubber_hand",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35),
            pos_y=(-0.35, -0.05),
            pos_z=(-0.10, 0.40),
            roll=(-0.5, 0.5),
            pitch=(-0.5, 0.5),
            yaw=(-0.5, math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class G1Events(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "force_range": (-5.0, 5.0),
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


        self.rewards.torso_larm_self_collision = RewTerm(
            func=primitive_distance_penalty,
            weight=-0.1,
            params={
                "pair_1_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "pair_2_cfg": SceneEntityCfg("robot", body_names="left_shoulder_yaw_link"),
                "radius_1": 0.11,
                "radius_2": 0.03,
                "offset_1": [0, 0, 0.13],
                "offset_2": [0, 0, 0.02],
                "length_1": 0.33,
                "length_2": 0.1,
            },
        )

        self.rewards.torso_rarm_self_collision = RewTerm(
            func=primitive_distance_penalty,
            weight=-0.1,
            params={
                "pair_1_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "pair_2_cfg": SceneEntityCfg("robot", body_names="right_shoulder_yaw_link"),
                "radius_1": 0.11,
                "radius_2": 0.03,
                "offset_1": [0, 0, 0.13],
                "offset_2": [0, 0, 0.02],
                "length_1": 0.33,
                "length_2": 0.1,
            },
        )

        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt

        # Custom terminations
        self.terminations.torso_height.minimum_height = 0.2
        self.terminations.base_contact = None
        # self.terminations.base_orientation = TerminationTermCfg(
        #     func=mdp.bad_orientation,
        #     params={"limit_angle": 0.7},
        # )

        self.episode_length_s = 14.0

        # Rewards:
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -20.5
        self.rewards.termination_penalty.weight = -100.0
        self.rewards.joint_deviation_fingers = None
        # self.rewards.joint_deviation_torso = None
        self.rewards.joint_deviation_torso.params['asset_cfg'].joint_names = 'waist_pitch_joint'
        self.rewards.lin_vel_z_l2 = None

        # Customize terrain
        self.scene.terrain.terrain_type = "plane"   # generator
        self.scene.terrain.terrain_generator = None #MILDLY_ROUGH_TERRAINS_CFG
        # Remove height scanner.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None
        # Add back pushing robot
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
        self.events.base_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
            },
        )
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)


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
        # Planar terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None

