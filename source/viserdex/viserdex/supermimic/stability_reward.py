"""Stationary, positive pose-and-speed reward for persistent holding."""
import math
import torch
from .quiet_hold import angular_excess

DENSE_REWARD_PROTOCOL = 'dense_pose_positive_stability_v1'
ORIENTATION_HOLD_REWARD_PROTOCOL = 'dense_orientation_speed_positive_stability_v2'
STABILITY_CONFIG_FIELDS = ('stability_reward_scale', 'stability_angle_sigma_rad',
                           'stability_position_sigma_m', 'stability_angular_speed_scale',
                           'stability_linear_speed_scale')


def validate_stability_reward(cfg):
    for name in STABILITY_CONFIG_FIELDS:
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0 or (name != 'stability_reward_scale' and value == 0):
            raise ValueError(f'{name} must be finite and positive (reward scale may be zero)')


def stability_reward(angle, distance, angular_speed, linear_speed, cfg):
    """Reward correct, quiet poses continuously; no clock or success latch."""
    exponent = (angular_excess(angle, cfg) / cfg.stability_angle_sigma_rad).square()
    if cfg.track_object_position:
        excess = (distance - cfg.position_tolerance_m).clamp_min(0)
        exponent = exponent + (excess / cfg.stability_position_sigma_m).square()
    exponent = (exponent + (angular_speed / cfg.stability_angular_speed_scale).square()
                + (linear_speed / cfg.stability_linear_speed_scale).square())
    return cfg.stability_reward_scale * torch.exp(-exponent)


def task_reward_bound(cfg):
    """Per-control-step upper bound used to normalize the constrained objective."""
    position_scale = cfg.position_reward_scale if cfg.track_object_position else 0.
    return cfg.control_dt * (cfg.orientation_reward_scale + position_scale + cfg.stability_reward_scale)
