"""Wuji-style home-relative position targets with EMA and reset warmup.

Task-local implementation on the official action API. No solver modifications.
"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg

class WujiHomeOffsetAction(EMAJointPositionToLimitsAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.home = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self.warmup_steps = max(0, math.ceil(cfg.warmup_time_s / env.step_dt - 1e-9))
        self._prev_applied_actions.copy_(self.home)
        self._processed_actions.copy_(self.home)

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        raw = self.home + self.cfg.offset_scale * actions.clamp(-1., 1.)
        raw = raw.clamp(limits[..., 0], limits[..., 1])
        filtered = self._alpha * raw + (1 - self._alpha) * self._prev_applied_actions
        target = torch.where((self._env.episode_length_buf < self.warmup_steps)[:, None], self.home, filtered)
        self._processed_actions.copy_(target.clamp(limits[..., 0], limits[..., 1]))
        self._prev_applied_actions.copy_(self._processed_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._prev_applied_actions[ids] = self.home[ids]
        self._processed_actions[ids] = self.home[ids]

@configclass
class WujiHomeOffsetActionCfg(EMAJointPositionToLimitsActionCfg):
    class_type: type = WujiHomeOffsetAction
    rescale_to_limits: bool = False
    alpha: float = 0.5
    offset_scale: float = 0.5
    warmup_time_s: float = 0.4
