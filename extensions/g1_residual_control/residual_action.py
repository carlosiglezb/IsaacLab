import torch
from omni.isaac.lab.managers.action_manager import JointPositionAction
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import quat_mul, quat_inv, quat_rotate

from extensions.g1_residual_control.actions_cfg import JointResidualActionCfg


class JointResidualAction(JointPositionAction):
    def __init__(self, cfg: JointResidualActionCfg, env):
        super().__init__(cfg, env)

        # Identify the body indices for the robot's end-effectors (e.g., torso or hands)
        self.body_indices = {
            name: self.env.scene[self.cfg.asset_name].find_bodies(name)[0]
            for name in ["pelvis", "left_hand", "right_hand", "left_knee", "right_knee"]
        }

    def apply_action(self):
        asset = self.env.scene[self.cfg.asset_name]

        # 1. Analytic: Get the kinematically consistent seed
        task_targets = self.env.get_spline_targets()
        # FIXME: run base policy checkpoint instead of Jacobian IK
        q_nom = self.compute_weighted_ik(asset, task_targets)

        # 2. Learned: Add the residual from the policy
        res_action = self.processed_actions
        self._raw_targets = q_nom + res_action

        # 3. Execution
        asset.set_joint_position_target(self._raw_targets, joint_ids=self.joint_ids)


    def compute_weighted_ik(self, asset, task_targets):
        """
        task_targets: dict of {body_name: target_pose_tensor}
        """
        all_jacobians = []
        all_errors = []
        all_weights = []

        # Access the full Jacobian for all bodies in one call for speed
        full_jacobians = asset.root_physx_view.get_jacobians()

        for body_name, target_pose in task_targets.items():
            # 1. Get body data
            idx = self.body_indices[body_name]

            # Extract 6D Jacobian and Pose Error
            J = full_jacobians[:, idx, :, :]

            # 2. Compute Pose Error (6D: pos + small-angle rot)
            curr_pose = asset.data.body_state_w[:, idx, :7]
            error = self._compute_pose_error(curr_pose, target_pose)  # (N, 6)

            # 3. Get weight from config
            weight = self.cfg.task_weights[body_name]

            # 4. Store
            all_jacobians.append(J * weight)
            all_errors.append(error * weight)

        # --- Stacking ---
        J_stack = torch.cat(all_jacobians, dim=1)  # (N, 6*num_tasks, n_joints)
        e_stack = torch.cat(all_errors, dim=1).unsqueeze(-1)  # (N, 6*num_tasks)

        # --- Solve via Damped Least Squares ---
        # (J^T J + lambda*I) dq = J^T e
        lmbda = 0.01  # Damping factor
        J_t = J_stack.transpose(1, 2)

        # Left hand side: J^T * J + damping
        lhs = torch.bmm(J_t, J_stack) + lmbda * torch.eye(J_stack.shape[-1], device=self.device)
        # Right hand side: J^T * e
        rhs = torch.bmm(J_t, e_stack.unsqueeze(-1))

        dq = torch.linalg.solve(lhs, rhs).squeeze(-1)
        return asset.data.joint_pos + dq

    def _compute_pose_error(self, curr_pose, target_pose):
        """
        Computes the 6D error [pos_err, rot_err] between current and target poses.
        curr_pose/target_pose: (N, 7) -> [x, y, z, qw, qx, qy, qz]
        """
        # 1. Position Error (Delta P)
        pos_error = target_pose[:, :3] - curr_pose[:, :3]

        # 2. Orientation Error (Delta Phi)
        # Formula: q_error = q_target * inv(q_curr)
        q_curr = curr_pose[:, 3:]
        q_target = target_pose[:, 3:]

        # Calculate the relative rotation quaternion
        q_error = quat_mul(q_target, quat_inv(q_curr))

        # Convert quaternion error to a rotation vector (axis-angle)
        # For small errors, the vector part [qx, qy, qz] is proportional to the error
        # We use 2 * q_xyz * sign(qw) to handle the double-cover property
        rot_error = 2.0 * q_error[:, 1:] * torch.sign(q_error[:, 0:1])

        # 3. Concatenate into 6D vector (N, 6)
        return torch.cat([pos_error, rot_error], dim=-1)