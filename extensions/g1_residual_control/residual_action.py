from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .actions_cfg import JointResidualActionCfg


class JointResidualAction(JointPositionAction):
    """Joint position action that adds a learned residual to a frozen base policy.

    Each step:
      1. The frozen base loco-manipulation policy is queried with the
         ``cfg.base_policy_obs_group`` observation group → ``q_base``.
      2. The residual policy output (scaled by ``cfg.scale`` via the parent's
         ``process_actions``) is added: ``q_cmd = q_base + δq``.
      3. ``q_cmd`` is applied as a joint position target.

    When ``cfg.base_policy_path`` is empty the term falls back to the robot's
    default joint positions so the environment still steps without a checkpoint.
    """

    cfg: JointResidualActionCfg

    def __init__(self, cfg: JointResidualActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._base_policy: torch.jit.ScriptModule | None = None
        if cfg.base_policy_path:
            if not check_file_path(cfg.base_policy_path):
                raise FileNotFoundError(
                    f"Base policy checkpoint not found: '{cfg.base_policy_path}'"
                )
            file_bytes = read_file(cfg.base_policy_path)
            self._base_policy = torch.jit.load(file_bytes).to(env.device)
            self._base_policy.eval()

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    def apply_actions(self) -> None:
        # 1. Nominal: run the frozen base loco-manipulation policy.
        if self._base_policy is not None:
            base_obs = self._env.observation_manager.compute_group(
                self.cfg.base_policy_obs_group
            )
            with torch.no_grad():
                q_base = self._base_policy(base_obs)
        else:
            # No checkpoint provided — hold the robot's default pose.
            q_base = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # 2. Add the scaled residual correction.
        #    processed_actions = raw_residual * cfg.scale  (computed by parent).
        q_cmd = q_base + self.processed_actions

        # 3. Apply joint position targets.
        self._asset.set_joint_position_target(q_cmd, joint_ids=self._joint_ids)
