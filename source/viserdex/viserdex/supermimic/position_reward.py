"""Position tracking in meters, shared by physics and imagined rollouts."""
import math
import torch

POSITION_REWARD_PROTOCOL = 'tolerant_position_gaussian_v1'
POSITION_CONFIG_FIELDS = ('track_object_position', 'position_reward_scale', 'position_tolerance_m', 'position_sigma_m',
                          'hold_position_threshold_m', 'hold_linear_speed_threshold_m_s')


def validate_position_reward(cfg):
    if type(cfg.track_object_position) is not bool:
        raise ValueError('track_object_position must be boolean')
    for name in POSITION_CONFIG_FIELDS:
        if name == 'track_object_position':
            continue
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0 or (name != 'position_tolerance_m' and value == 0):
            raise ValueError(f'{name} must be finite and positive (tolerance may be zero)')
    if cfg.hold_position_threshold_m < cfg.position_tolerance_m:
        raise ValueError('hold_position_threshold_m must cover the reward tolerance')


def position_reward(distance_m, cfg):
    """Broad attraction with a flat acceptable region; no action processing."""
    if not cfg.track_object_position:
        return torch.zeros_like(distance_m)
    excess = (distance_m - cfg.position_tolerance_m).clamp_min(0)
    return cfg.position_reward_scale * torch.exp(-(excess / cfg.position_sigma_m).square())


def position_hold_valid(distance_m, linear_speed_m_s, cfg):
    valid = linear_speed_m_s < cfg.hold_linear_speed_threshold_m_s
    if cfg.track_object_position:
        valid = valid & (distance_m < cfg.hold_position_threshold_m)
    return valid
