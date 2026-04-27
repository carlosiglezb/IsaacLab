from isaaclab.utils import configclass

from isaaclab.managers import ActionTermCfg

from .residual_action import JointResidualAction


@configclass
class JointResidualActionCfg(ActionTermCfg):
    class_type = JointResidualAction
    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    scale: float = 0.1
    offset: float = 0.0

    # Path to the pre-trained base loco-manipulation policy checkpoint (.pt).
    # Set in the environment's __post_init__ so it can be overridden per variant.
    base_policy_path: str = ""

    # Observation group name whose concatenated tensor is fed to the base policy.
    base_policy_obs_group: str = "base_policy"

    # Per-body IK weights used when falling back to the weighted Jacobian nominal.
    # Keys must match link names in the robot URDF.
    task_weights: dict[str, float] = {
        "pelvis": 1.0,
        "left_palm_link": 0.2,
        "right_palm_link": 0.2,
        "right_knee_link": 0.2,
        "left_knee_link": 0.2,
        "left_ankle_roll_link": 0.8,
        "right_ankle_roll_link": 0.8,
    }
