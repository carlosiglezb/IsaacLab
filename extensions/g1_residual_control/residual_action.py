from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .actions_cfg import JointResidualActionCfg


# Two-prototype marker config: index 0 = current (red), index 1 = desired (green).
_GUIDE_VIS_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/GuideTracking",
    markers={
        "current": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        ),
        "desired": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
        ),
    },
)


class JointResidualAction(JointPositionAction):
    """Joint position action with a pluggable base controller + learned residual.

    Base controller selection (evaluated in priority order):

    1. **IK base** (``cfg.ik_body_names`` non-empty): A multi-body position IK
       is solved each step against the active guide targets using a stacked
       Damped Least Squares (DLS) Jacobian.  ``cfg.n_ik_iters`` Newton steps
       are taken starting from the current joint configuration.  This is the
       recommended mode for residual learning without a pre-trained checkpoint.

    2. **Frozen-policy base** (``cfg.base_policy_path`` non-empty, IK disabled):
       A JIT-exported loco-manipulation checkpoint is evaluated each step using
       the ``cfg.base_policy_obs_group`` observation group.

    3. **Default pose** (fallback): Holds the robot at its default joint
       positions when neither of the above is configured.

    In all cases the learned residual correction is applied on top:

        q_cmd = q_base + δq

    where δq is the residual output scaled by ``cfg.scale``.
    """

    cfg: JointResidualActionCfg

    def __init__(self, cfg: JointResidualActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        # --- Frozen-policy base (legacy) ------------------------------------
        self._base_policy: torch.jit.ScriptModule | None = None
        if cfg.base_policy_path and not cfg.ik_body_names:
            if not check_file_path(cfg.base_policy_path):
                raise FileNotFoundError(
                    f"Base policy checkpoint not found: '{cfg.base_policy_path}'"
                )
            file_bytes = read_file(cfg.base_policy_path)
            self._base_policy = torch.jit.load(file_bytes).to(env.device)
            self._base_policy.eval()

        # --- IK-based base --------------------------------------------------
        # Body/joint indices resolved here; guide frame indices resolved lazily
        # on first call (GuideDataset is attached after the startup event).
        self._ik_body_indices: list[int] = []
        self._ik_jacobi_body_indices: list[int] = []
        self._ik_jacobi_joint_ids: list[int] = []
        self._ik_frame_indices: list[int] | None = None  # None = not yet resolved

        if cfg.ik_body_names:
            self._setup_ik()

    # ------------------------------------------------------------------
    # IK helpers
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        """Resolve per-body and per-joint PhysX indices for the IK solve."""
        robot = self._asset

        for body_name in self.cfg.ik_body_names:
            body_ids, _ = robot.find_bodies(body_name)
            if len(body_ids) != 1:
                raise ValueError(
                    f"IK base: expected exactly one body matching '{body_name}', "
                    f"got {len(body_ids)}"
                )
            body_idx = int(body_ids[0])
            self._ik_body_indices.append(body_idx)
            # For fixed-base robots PhysX excludes the root body from the Jacobian,
            # so the index is shifted by -1.  Floating-base uses the direct index.
            jacobi_body_idx = body_idx - 1 if robot.is_fixed_base else body_idx
            self._ik_jacobi_body_indices.append(jacobi_body_idx)

        # PhysX places 6 base-DOF columns before the joint columns for
        # floating-base robots; fixed-base robots have no offset.
        joint_id_offset = 0 if robot.is_fixed_base else 6
        if isinstance(self._joint_ids, slice):
            raw_ids = list(range(robot.num_joints))
        else:
            raw_ids = list(self._joint_ids)
        self._ik_jacobi_joint_ids = [j + joint_id_offset for j in raw_ids]

    def _resolve_ik_frame_indices(self) -> list[int]:
        """Map each IK body name to its guide-dataset frame index."""
        ds = self._env.guide_dataset
        return [ds.body_name_to_idx[name] for name in self.cfg.ik_body_names]

    def _compute_ik_base(self) -> torch.Tensor:
        """Multi-body position IK against the current guide targets.

        Builds a stacked position Jacobian across all ``cfg.ik_body_names``
        bodies and solves with Damped Least Squares (DLS):

            Δq = J^T (J J^T + λ²I)^{-1} e

        ``cfg.n_ik_iters`` Newton steps are taken, each reusing the Jacobian
        linearised at the current simulation state (the robot has not moved).

        Returns
        -------
        torch.Tensor, shape (num_envs, n_joints)
        """
        env = self._env
        robot = self._asset

        # Fall back to current joint positions if guide data is not yet ready.
        if not hasattr(env, "guide_ctrl_pts"):
            return robot.data.joint_pos[:, self._joint_ids].clone()

        # Lazily resolve guide frame indices on first call.
        if self._ik_frame_indices is None:
            self._ik_frame_indices = self._resolve_ik_frame_indices()

        t_elapsed = env.episode_length_buf * env.step_dt
        all_targets = env.guide_dataset.query_targets(
            env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
        )  # (N, n_frames, 3)

        # Start from current joint positions.
        q = robot.data.joint_pos[:, self._joint_ids].clone()  # (N, n_joints)
        lam2 = self.cfg.ik_lambda ** 2
        n_bodies = len(self._ik_body_indices)
        m = 3 * n_bodies  # rows of the stacked Jacobian
        lam2_I = lam2 * torch.eye(m, device=self.device)

        for _ in range(self.cfg.n_ik_iters):
            # Full PhysX Jacobian: (N, n_physx_bodies, 6, n_dofs)
            full_J = robot.root_physx_view.get_jacobians()

            J_blocks: list[torch.Tensor] = []
            e_blocks: list[torch.Tensor] = []
            for body_idx, j_body_idx, frame_idx in zip(
                self._ik_body_indices,
                self._ik_jacobi_body_indices,
                self._ik_frame_indices,
            ):
                # Position-only Jacobian w.r.t. controlled joints: (N, 3, n_joints)
                J_b = full_J[:, j_body_idx, :3, :][:, :, self._ik_jacobi_joint_ids]
                J_blocks.append(J_b)

                current_pos = robot.data.body_pos_w[:, body_idx, :3]  # (N, 3)
                target_pos = all_targets[:, frame_idx, :]              # (N, 3)
                e_blocks.append(target_pos - current_pos)

            J_stack = torch.cat(J_blocks, dim=1)               # (N, 3*n_bodies, n_joints)
            e_stack = torch.cat(e_blocks, dim=-1).unsqueeze(-1) # (N, 3*n_bodies, 1)

            # DLS solve: Δq = J^T (J J^T + λ²I)^{-1} e
            JJT_reg = torch.bmm(J_stack, J_stack.transpose(-1, -2)) + lam2_I
            rhs = torch.linalg.solve(JJT_reg, e_stack)          # (N, m, 1)
            delta_q = torch.bmm(J_stack.transpose(-1, -2), rhs).squeeze(-1)  # (N, n_joints)

            q = q + delta_q

        return q

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _visualize_guide_tracking(self) -> None:
        """Render red spheres at current body positions and green spheres at guide targets.

        Only active when ``cfg.debug_vis`` is True and IK body names are configured.
        The ``VisualizationMarkers`` instance is created lazily on the first call.
        """
        if not hasattr(self._env, "guide_ctrl_pts") or self._ik_frame_indices is None:
            return

        # Lazily create the USD instancer the first time we need it.
        if not hasattr(self, "_vis_markers"):
            self._vis_markers = VisualizationMarkers(_GUIDE_VIS_CFG)

        robot = self._asset
        env = self._env

        t_elapsed = env.episode_length_buf * env.step_dt
        all_targets = env.guide_dataset.query_targets(
            env.guide_ctrl_pts, env.guide_transition_times, t_elapsed
        )  # (N, n_frames, 3)

        # Stack positions across all tracked bodies: (N, B, 3)
        curr_pos = torch.stack(
            [robot.data.body_pos_w[:, idx, :3] for idx in self._ik_body_indices], dim=1
        )
        des_pos = torch.stack(
            [all_targets[:, idx, :] for idx in self._ik_frame_indices], dim=1
        )

        N, B = curr_pos.shape[:2]
        # Concatenate: first N*B current (prototype 0), then N*B desired (prototype 1).
        translations = torch.cat(
            [curr_pos.reshape(N * B, 3), des_pos.reshape(N * B, 3)], dim=0
        )
        marker_indices = torch.zeros(2 * N * B, dtype=torch.long, device=self.device)
        marker_indices[N * B :] = 1

        self._vis_markers.visualize(translations=translations, marker_indices=marker_indices)

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    def apply_actions(self) -> None:
        # 1. Compute nominal base configuration.
        if self.cfg.ik_body_names:
            q_base = self._compute_ik_base()
        elif self._base_policy is not None:
            base_obs = self._env.observation_manager.compute_group(
                self.cfg.base_policy_obs_group
            )
            with torch.no_grad():
                q_base = self._base_policy(base_obs)
        else:
            q_base = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # 2. Add the scaled residual correction.
        #    processed_actions = raw_residual * cfg.scale  (computed by parent).
        q_cmd = q_base + self.processed_actions

        # 3. Apply joint position targets.
        self._asset.set_joint_position_target(q_cmd, joint_ids=self._joint_ids)

        # 4. Optional per-step visualisation of current vs. desired body positions.
        if self.cfg.debug_vis and self._ik_body_indices:
            self._visualize_guide_tracking()
