from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from .residual_action import JointResidualAction


@configclass
class JointResidualActionCfg(JointPositionActionCfg):
    """Configuration for the residual joint-position action term.

    Two base-controller modes are supported (IK takes priority over frozen policy):

    **IK mode** (set ``ik_body_names``):
        A stacked multi-body position IK is solved each step against the active
        guide targets.  ``ik_lambda`` controls DLS damping (larger = more
        conservative joint motion) and ``n_ik_iters`` controls how many
        linearised Newton steps are taken per policy step.

    **Frozen-policy mode** (set ``base_policy_path``, leave ``ik_body_names`` empty):
        Loads a JIT-exported loco-manipulation checkpoint from ``base_policy_path``
        and evaluates it each step using the ``base_policy_obs_group`` observation
        group (legacy behaviour; preserved for backward compatibility).

    In both modes the learned residual δq is added on top of q_base and scaled by
    ``scale``.
    """

    class_type = JointResidualAction

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]

    # The residual δq is an absolute correction in joint space, not a delta
    # from the default pose.  The parent class adds default_joint_pos as an
    # offset when use_default_offset=True, which would double-count the
    # default configuration and send joints past their limits.
    use_default_offset: bool = False

    # Residual scale: keep δq small relative to the base controller's output.
    scale: float = 0.15

    # ------------------------------------------------------------------
    # IK-based base controller (new)
    # ------------------------------------------------------------------

    # URDF body names whose world-frame positions the IK controller targets.
    # When non-empty, the frozen-policy path below is ignored.
    ik_body_names: list[str] = []

    # DLS damping coefficient λ.  Larger values trade tracking accuracy for
    # smoother, more conservative joint motion near singularities.
    ik_lambda: float = 0.05

    # Number of linearised Newton steps per policy step.  1 is sufficient
    # for the residual-learning use case; increase for tighter IK tracking.
    n_ik_iters: int = 1

    # ------------------------------------------------------------------
    # Frozen-policy base controller (legacy)
    # ------------------------------------------------------------------

    # Path to the JIT-exported base loco-manipulation policy (.pt).
    # Ignored when ik_body_names is non-empty.
    base_policy_path: str = ""

    # Observation group whose concatenated tensor is fed to the base policy.
    base_policy_obs_group: str = "base_policy"

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    # When True, renders red spheres at the current body positions and green
    # spheres at the guide target positions each step.  Intended for play mode;
    # leave False during training to avoid USD overhead.
    debug_vis: bool = False
