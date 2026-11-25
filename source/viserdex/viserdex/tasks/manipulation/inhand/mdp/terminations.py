# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Task-specific terminations retained from Isaac Lab's previous in-hand task.

The v3.0 reorient package no longer exports these two terms. Preserve the
original thresholds/reference frame while reading the public ProxyArray API.
"""

import torch
from isaaclab.managers import SceneEntityCfg


def max_consecutive_success(env, num_success: int, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return command.metrics["consecutive_success"] >= num_success


def object_away_from_robot(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    return torch.linalg.vector_norm(robot.data.root_pos_w.torch - obj.data.root_pos_w.torch, dim=1) > threshold
