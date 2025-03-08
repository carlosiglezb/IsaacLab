# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

NAVY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=10.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.04),
            noise_step=0.02,
            border_width=0.25
        ),
        "knee_knocker": terrain_gen.MeshKneeKnockerTerrainCfg(
            proportion=0.8,
            step_height_range=(0.0, 0.4),
            step_width=0.765,
            door_width=3.0,
            x_offset=0.3,
        ),
        # "knee_knocker_mid_low": terrain_gen.MeshKneeKnockerTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.1, 0.2),
        #     step_width=0.765,
        #     door_width=3.0,
        #     x_offset=0.3,
        # ),
        # "knee_knocker_mid_high": terrain_gen.MeshKneeKnockerTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.2, 0.3),
        #     step_width=0.765,
        #     door_width=3.0,
        #     x_offset=0.3,
        # ),
        # "knee_knocker_high": terrain_gen.MeshKneeKnockerTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.3, 0.4),
        #     step_width=0.765,
        #     door_width=3.0,
        #     x_offset=0.3,
        # ),
    },
)
"""Navy terrains configuration."""
