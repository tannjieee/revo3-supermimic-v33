"""Bounded SO(3) task reward, shared by physics callbacks and learned rollouts."""
import math
import torch
from .quiet_hold import angular_excess

ORIENTATION_REWARD_PROTOCOL = "two_scale_so3_gaussian_v1"


def validate_orientation_reward(cfg):
    for name in ('orientation_reward_scale', 'orientation_coarse_sigma_rad', 'orientation_fine_sigma_rad'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if cfg.orientation_fine_sigma_rad > cfg.orientation_coarse_sigma_rad:
        raise ValueError('orientation_fine_sigma_rad must not exceed the coarse scale')
    if not 0 <= cfg.orientation_coarse_fraction <= 1:
        raise ValueError('orientation_coarse_fraction must be in [0, 1]')


def orientation_reward(error_rad, cfg):
    """10 * (0.3 exp(-(theta/90deg)^2) + 0.7 exp(-(theta/10deg)^2)).

    theta is the sign-invariant shortest SO(3) angle, not Euler differences.
    No 1/2 in the exponent: sigma denotes the width at exp(-1).
    The coarse term avoids the almost-zero far-field gradient of a narrow
    Gaussian. Both terms have zero derivative at the target.
    Quiet holding replaces theta with max(theta - tolerance, 0), making the
    entire accepted neighborhood a maximum-reward plateau.
    Fresh quiet-hold runs use only the coarse exponential (mix=1).
    """
    error_rad = angular_excess(error_rad, cfg)
    mix = cfg.orientation_coarse_fraction
    if mix == 1.0:
        return cfg.orientation_reward_scale * torch.exp(-(error_rad / cfg.orientation_coarse_sigma_rad).square())
    if mix == 0.0:
        return cfg.orientation_reward_scale * torch.exp(-(error_rad / cfg.orientation_fine_sigma_rad).square())
    coarse = torch.exp(-(error_rad / cfg.orientation_coarse_sigma_rad).square())
    fine = torch.exp(-(error_rad / cfg.orientation_fine_sigma_rad).square())
    return cfg.orientation_reward_scale * (mix * coarse + (1 - mix) * fine)
