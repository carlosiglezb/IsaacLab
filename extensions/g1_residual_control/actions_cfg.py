from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from .residual_action import JointResidualAction


@configclass
class JointResidualActionCfg(JointPositionActionCfg):
    """Configuration for the residual joint-position action term.

    The action term loads a frozen JIT-exported loco-manipulation policy from
    ``base_policy_path``, runs a forward pass each step using the observation
    group named ``base_policy_obs_group``, then adds the residual network output
    (scaled by ``scale``) before applying the final joint position target.
    """

    class_type = JointResidualAction

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]

    # Residual scale: keep δq small so the base policy stays in control.
    scale: float = 0.15

    # Path to the JIT-exported base loco-manipulation policy (.pt).
    # Set in ResidualGuideTrackingEnvCfg.__post_init__ so it can be
    # overridden per variant without touching this dataclass.
    base_policy_path: str = ""

    # Observation group whose concatenated tensor is fed to the base policy.
    base_policy_obs_group: str = "base_policy"
