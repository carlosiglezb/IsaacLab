# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for residual policy learning on the G1.

Architecture
------------
  base policy (pre-trained loco-manipulation checkpoint)
      ↓  q_base  (full-body joint targets)
  residual policy (what we are training here)
      ↓  δq       (learned correction, scaled by ``JointResidualActionCfg.scale``)
  q_cmd = q_base + δq

The residual policy observes proprioceptive state plus position-tracking
errors between the current body poses and guide trajectories generated
by the user's planner package.  Guide-related MDP terms live in ``mdp.py``
and are currently stubs; fill them in once the planner is wired in.
"""

import os
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_assets import G1_PRIMITIVE_COLLISIONS

from .actions_cfg import JointResidualActionCfg
from .commands import GuideTorsoVelocityCommandCfg
from . import mdp as residual_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

cwd = os.getcwd()


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """G1 scene with the Navy-door knee-knocker obstacle."""

    hole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KneeKnocker",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{cwd}/source/my_usds/navy_door.urdf",
            fix_base=None,
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.0),
            rot=(0.707, 0, 0, 0.707),
        ),
    )

    robot: ArticulationCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Single residual action term over the full joint space.

    ``JointResidualAction.apply_action()`` is responsible for:
      1. Loading the base loco-manipulation policy from ``base_policy_path``.
      2. Running a forward pass using the ``base_policy`` observation group.
      3. Adding the residual network output (scaled by ``scale``) to q_base.

    The base policy path is set in ``ResidualGuideTrackingEnvCfg.__post_init__``
    so it can be overridden without touching this dataclass.
    """

    joint_residual = JointResidualActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.4,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the residual policy network.

        Comprises three groups:
          (a) Proprioceptive robot state
          (b) Guide trajectory targets for the seven tracked bodies
          (c) Position tracking errors (current pose – guide target)
          (d) Previous residual action
        """

        # ---- (a) Proprioception ----------------------------------------
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_lin_vel = ObsTerm(
            func=base_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ---- (b) Guide trajectory targets (world-frame position, 3-D each)
        guide_left_hand_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "left_rubber_hand"},
        )
        guide_right_hand_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "right_rubber_hand"},
        )
        guide_torso_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "torso_link"},
        )
        guide_left_knee_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "left_knee_link"},
        )
        guide_right_knee_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "right_knee_link"},
        )
        guide_left_foot_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "left_ankle_roll_link"},
        )
        guide_right_foot_pos = ObsTerm(
            func=residual_mdp.guide_body_target_pos,
            params={"body_name": "right_ankle_roll_link"},
        )

        # ---- (c) Tracking errors (current body pos − guide target, per link)
        tracking_error_hands = ObsTerm(
            func=residual_mdp.guide_tracking_error,
            params={"body_names": ["left_rubber_hand", "right_rubber_hand"]},
        )
        tracking_error_torso = ObsTerm(
            func=residual_mdp.guide_tracking_error,
            params={"body_names": ["torso_link"]},
        )
        tracking_error_knees = ObsTerm(
            func=residual_mdp.guide_tracking_error,
            params={"body_names": ["left_knee_link", "right_knee_link"]},
        )
        tracking_error_feet = ObsTerm(
            func=residual_mdp.guide_tracking_error,
            params={"body_names": ["left_ankle_roll_link", "right_ankle_roll_link"]},
        )

        # ---- (d) Commands (same signals the base policy conditions on)
        velocity_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        ee_left_hand_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_left_hand"},
        )
        ee_right_hand_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_right_hand"},
        )

        # ---- (e) Previous residual action
        last_residual_action = ObsTerm(
            func=base_mdp.last_action,
            params={"action_name": "joint_residual"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class BasePolicyCfg(ObsGroup):
        """Observations fed to the frozen base loco-manipulation policy.

        Mirrors the exact observation format that the base
        checkpoint was trained on.  The current layout follows the agile
        locomotion policy convention as a reasonable starting point.
        Update joint_names and any extra terms once the base policy's
        training config is confirmed.
        """

        base_lin_vel = ObsTerm(
            func=base_mdp.base_lin_vel,
        )
        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel,
        )
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        left_ee_pose_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_left_hand"},
        )
        right_ee_pose_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_right_hand"},
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    base_policy: BasePolicyCfg = BasePolicyCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward terms for the residual guide-tracking task.

    Tracking rewards use an exponential kernel  exp(-||e||² / σ²),
    which gives a smooth gradient everywhere and equals 1.0 at zero error.
    Weights and σ values below are initial guesses — tune via sweep.
    """

    # ---- Guide position tracking -----------------------------------------
    track_left_hand_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=2.0,
        params={"body_name": "left_rubber_hand", "sigma": 0.10},
    )
    track_right_hand_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=2.0,
        params={"body_name": "right_rubber_hand", "sigma": 0.10},
    )
    track_torso_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=1.5,
        params={"body_name": "torso_link", "sigma": 0.15},
    )
    track_left_knee_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=1.0,
        params={"body_name": "left_knee_link", "sigma": 0.25},
    )
    track_right_knee_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=1.0,
        params={"body_name": "right_knee_link", "sigma": 0.25},
    )
    track_left_foot_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=1.0,
        params={"body_name": "left_ankle_roll_link", "sigma": 0.25},
    )
    track_right_foot_pos = RewTerm(
        func=residual_mdp.guide_pos_tracking_exp,
        weight=1.0,
        params={"body_name": "right_ankle_roll_link", "sigma": 0.25},
    )

    # ---- Guide velocity tracking (supplementary) -------------------------
    # sigma=0.25 m/s gives useful gradient several tenths of a m/s from the
    # target, appropriate for deliberate stepping speeds (0.1–0.5 m/s).
    # Weights are ~25% of the matching position terms to keep velocity as a
    # temporal-coherence hint rather than the dominant training signal.
    track_left_hand_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.5,
        params={"body_name": "left_rubber_hand", "sigma": 0.25},
    )
    track_right_hand_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.5,
        params={"body_name": "right_rubber_hand", "sigma": 0.25},
    )
    track_torso_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.4,
        params={"body_name": "torso_link", "sigma": 0.25},
    )
    track_left_knee_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.25,
        params={"body_name": "left_knee_link", "sigma": 0.25},
    )
    track_right_knee_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.25,
        params={"body_name": "right_knee_link", "sigma": 0.25},
    )
    track_left_foot_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.25,
        params={"body_name": "left_ankle_roll_link", "sigma": 0.25},
    )
    track_right_foot_vel = RewTerm(
        func=residual_mdp.guide_vel_tracking_exp,
        weight=0.25,
        params={"body_name": "right_ankle_roll_link", "sigma": 0.25},
    )

    # ---- Goal-reaching rewards -------------------------------------------
    # Dense position-progress signal (ReLIC-style): reward = Σ_i [d_i(t-1) − d_i(t)]
    # summed over all seven tracked frames.  Positive when bodies collectively
    # move toward the guide, negative when they drift away.
    position_progress = RewTerm(
        func=residual_mdp.body_position_progress,
        weight=2.0,
        params={
            "body_names": [
                "torso_link",
                "left_rubber_hand",
                "right_rubber_hand",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
            ]
        },
    )
    # Sparse terminal penalty: sum of L2 distances to final guide positions
    # across key bodies.  Fires once per episode at timeout or fall.
    goal_not_reached = RewTerm(
        func=residual_mdp.goal_not_reached_penalty,
        weight=-20.0,
        params={
            "body_names": [
                "torso_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_knee_link",
                "right_knee_link",
                "left_rubber_hand",
                "right_rubber_hand",
            ]
        },
    )

    # ---- Contact sequence -----------------------------------------------
    # Reward proximity to the physical surface (floor/wall) that each hand
    # or foot should brace against during its FIXED phase.  sigma = 5 cm
    # gives a meaningful gradient up to ~10 cm away from the surface.
    contact_surface_proximity = RewTerm(
        func=residual_mdp.contact_surface_proximity,
        weight=1.5,
        params={"sigma": 0.1},
    )

    # ---- Regularization --------------------------------------------------
    # Penalise the magnitude of the residual δq — keeps the residual small
    # so the base policy remains in control and the correction is surgical.
    residual_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.05)
    # Penalise large changes in the residual to keep actions smooth.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    # Penalise high joint velocities to reduce wear-and-tear behaviour.
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ---- Alive bonus -----------------------------------------------------
    # Small constant reward for surviving each timestep; prevents the policy
    # from learning to terminate early to avoid tracking penalties.
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Terminate if the pelvis drops below 0.4 m (robot has fallen or deeply collapsed).
    robot_fell = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Terminate if the torso tilts more than ~20° from upright (catches sideways/forward falls
    # that may keep the pelvis above the height threshold).
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class EventsCfg:
    """Lifecycle event hooks for guide dataset management."""

    # Load GuideDataset from disk and allocate env.guide_waypoints once.
    startup_guide_init = EventTerm(
        func=residual_mdp.startup_guide_init,
        mode="startup",
    )

    # Reset root pose and velocity to init_state (no randomisation — fixed obstacle task).
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset joint positions and velocities to their defaults.
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Resample guides for environments that just reset.
    reset_guide_assignment = EventTerm(
        func=residual_mdp.reset_guide_assignment,
        mode="reset",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@configclass
class CommandsCfg:
    """Commands passed to the base policy.

    base_velocity is derived from the analytical Bezier torso guide velocity
    at each policy step, expressed in the robot's heading-aligned body frame.
    This replaces the fixed 0.4 m/s command that conflicted with the careful
    stepping behaviour required for the knee-knocker crossing.

    Hand pose targets are expressed in the robot's base frame, matching the
    frame convention the base checkpoint was trained with.
    """

    base_velocity = GuideTorsoVelocityCommandCfg(
        asset_name="robot",
        max_speed=1.5,
    )

    # Left palm: forward and slightly left, arms tucked for narrow passage.
    # pos_x/y/z are in the robot base frame (metres); roll/pitch/yaw in rad.
    ee_left_hand = base_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_rubber_hand",
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=base_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.25),
            pos_y=(0.15, 0.15),
            pos_z=(0.05, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    ee_right_hand = base_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_rubber_hand",
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=base_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.25),
            pos_y=(-0.15, -0.15),
            pos_z=(0.05, 0.05),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Main environment configuration
# ---------------------------------------------------------------------------

@configclass
class ResidualGuideTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """RL environment for training a residual policy over a pre-trained
    loco-manipulation base on the G1 with the knee-knocker obstacle.

    The robot must track guide trajectories for its hands, torso, knees,
    and feet while the base policy handles gross locomotion.

    IRIS collision regions
    ---------------------
    Three overlapping axis-aligned boxes (defined in the robot's local
    world frame) represent the collision-free navigation corridors around
    the Navy door.  A_mat encodes both ≤ and ≥ bounds via sign flip.

    The IRIS_seq dict maps body names (as used by the planner) to the
    sequence of region indices each body should pass through.
    """

    # ---- Managers --------------------------------------------------------
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum = None

    # ---- Base policy ----------------------------------------------------
    # Path to the pre-trained loco-manipulation policy checkpoint.
    # Override this in a subclass or set it before instantiating the env.
    BASE_POLICY_PATH: str = "logs/rsl_rl/g1_locomanipulation/policy.pt"

    # ---- Guide dataset --------------------------------------------------
    # Path to the .npz file produced by generate_guide_dataset().
    GUIDE_DATASET_PATH: str = "guide_dataset.npz"

    def __post_init__(self):
        self.scene.robot = G1_PRIMITIVE_COLLISIONS.replace(prim_path="{ENV_REGEX_NS}/Robot")

        """Post-initialisation: sim timing and action-term configuration."""
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 200.0   # 200 Hz physics
        self.sim.render_interval = 2

        # Forward the base policy path into the action term so
        # JointResidualAction can load the checkpoint at startup.
        self.actions.joint_residual.base_policy_path = self.BASE_POLICY_PATH


@configclass
class ResidualGuideTrackingEnvCfg_PLAY(ResidualGuideTrackingEnvCfg):
    """Play/evaluation variant: single environment, rendering enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        # Render every physics step so video capture is smooth.
        self.sim.render_interval = 1
