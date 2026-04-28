# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 residual guide-tracking extension.

Usage
-----
Import this module **before** calling ``gym.make`` so that the task IDs are
registered.  The easiest way is to add the extensions directory to
``PYTHONPATH`` and then add the following import in ``train.py`` after the
``# PLACEHOLDER: Extension template`` comment::

    import g1_residual_control  # noqa: F401

Train::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Residual-GuideTracking-G1-v0 --num_envs 512

Play::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Residual-GuideTracking-G1-Play-v0
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Residual-GuideTracking-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_env_cfg:ResidualGuideTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualGuideTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Residual-GuideTracking-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_env_cfg:ResidualGuideTrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualGuideTrackingPPORunnerCfg",
    },
)
