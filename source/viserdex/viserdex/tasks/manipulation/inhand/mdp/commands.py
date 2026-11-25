# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""ViserDex's released goal distribution with the Isaac Lab v3 command lifecycle."""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab_tasks.core.reorient.mdp import ReorientCommand, ReorientCommandCfg


class ViserDexReorientCommand(ReorientCommand):
    """Sample q_x(alpha) * q_y(beta), with both angles uniform in [-pi, pi].

    Matches Isaac Lab 2.3.2's InHandReOrientationCommand used by the public
    ViserDex release, rather than v3's uniform SO(3) sampler. Quaternion
    storage and all other command behavior follow v3's XYZW convention.
    """

    def _resample_command(self, env_ids: Sequence[int] | slice):
        # Preserve v3's goal accounting and reset guard; only the distribution changes.
        self._success.record_goal_reached(env_ids)
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        rand_floats = 2.0 * torch.rand((num_envs, 2), device=self.device) - 1.0
        x_axis = rand_floats.new_tensor((1.0, 0.0, 0.0)).expand(num_envs, -1)
        y_axis = rand_floats.new_tensor((0.0, 1.0, 0.0)).expand(num_envs, -1)
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, x_axis),
            math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, y_axis),
        )
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class ViserDexReorientCommandCfg(ReorientCommandCfg):
    """Use the released ViserDex goal sampler in all in-hand task variants."""

    class_type: type[ReorientCommand] = ViserDexReorientCommand
