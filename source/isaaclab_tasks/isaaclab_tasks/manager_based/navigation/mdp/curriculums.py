# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def knee_knocker_levels_pose(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
): # -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the height of the knee knocker when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    door: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("pose_command")
    # compute the distance in the x-direction that the robot walked
    final_pos_error = torch.norm(command[env_ids, :2], dim=1)
    # robots that walked far enough/pass door progress to harder terrains
    move_up = final_pos_error < 0.1
    # robots that did not walk close enough to target go to simpler terrains
    move_down = ~move_up
    # update terrain levels
    door.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(door.terrain_levels.float())

def terrain_levels_pose(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
): # -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move to a desired pose.

    This term is used to increase the height of the knee knocker when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("pose_command")
    # compute the distance in the x-direction that the robot walked
    final_pos_error = torch.norm(command[env_ids, :2], dim=1)
    # robots that walked far enough/pass door progress to harder terrains
    move_up = final_pos_error < 0.1
    # robots that did not walk close enough to target go to simpler terrains
    move_down = ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
