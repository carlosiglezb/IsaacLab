import os
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

cwd = os.getcwd()

G1_PRIMITIVE_COLLISIONS = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(asset_path=f"{cwd}/source/isaaclab_assets/isaaclab_assets/robots/g1/g1_simple_collisions.urdf",
                                fix_base=False,
                                # collision_from_visuals=True,
                                # collider_type='convex_decomposition',
                                activate_contact_sensors=True,
                                self_collision=True,
                                merge_fixed_joints=False,
                                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None,
                                                                                              damping=None)
                                ),
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    disable_gravity=False,
                                    retain_accelerations=False,
                                    linear_damping=0.0,
                                    angular_damping=0.0,
                                    max_linear_velocity=1000.0,
                                    max_angular_velocity=1000.0,
                                    max_depenetration_velocity=1.0,
                                ),
                                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                                    enabled_self_collisions=True,
                                    solver_position_iteration_count=8,
                                    solver_velocity_iteration_count=4),
                                ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.03, 0.0, 0.68),
        joint_pos={
            # ".*_hip_pitch_joint": -np.pi / 6,
            # ".*_knee_joint": np.pi / 3,
            # ".*_ankle_pitch_joint": -np.pi / 6,
            # ".*_elbow_joint": 0.87,
            ".*_hip_pitch_joint": -0.697,
            ".*_knee_joint": 1.23,
            ".*_ankle_pitch_joint": -0.53,
            ".*_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
                "waist_roll_joint": 0.01,
                "waist_pitch_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=2.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot with customized simplified collisions."""
