# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configuration for the G1 residual guide-tracking policy."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1ResidualGuideTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the residual policy that corrects a frozen loco-manipulation base.

    Design choices
    --------------
    init_noise_std=0.1
        The residual should stay small relative to the base policy output.
        A tight initial distribution prevents the residual from swamping the
        base trajectory in early training.

    obs_groups
        Both actor and critic consume the same "policy" observation group.
        The "base_policy" group is consumed internally by JointResidualAction
        and never exposed to the RL runner.

    num_steps_per_env=48
        Each episode lasts 1000 control steps (20 s at 50 Hz).  Using 48
        rollout steps (≈1 s of experience per env per update) gives the
        advantage estimator enough context while keeping update frequency high.

    actor_obs_normalization=True
        Inputs span heterogeneous scales: joint angles (~rad), world-frame body
        positions (~m), and velocity commands — online normalisation accelerates
        convergence without hand-tuning observation scales.
    """

    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_residual_guide_tracking"

    # Explicit obs → algorithmic-set mapping.  "base_policy" is internal to
    # JointResidualAction and must NOT appear here.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
