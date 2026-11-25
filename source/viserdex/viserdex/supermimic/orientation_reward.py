"""Bounded SO(3) task reward, shared by physics callbacks and learned rollouts."""
import math
import torch
from .quiet_hold import angular_excess

ORIENTATION_REWARD_PROTOCOL = "two_scale_so3_gaussian_v1"
LINEAR_FINE_ORIENTATION_PROTOCOL = "linear_lorentzian_so3_v2"


def validate_orientation_reward(cfg):
    mode = getattr(cfg, 'orientation_reward_mode', 'gaussian')
    if mode not in ('gaussian', 'linear_fine'):
        raise ValueError(f'Unknown orientation_reward_mode: {mode}')
    for name in ('orientation_reward_scale', 'orientation_coarse_sigma_rad', 'orientation_fine_sigma_rad'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if cfg.orientation_fine_sigma_rad > cfg.orientation_coarse_sigma_rad:
        raise ValueError('orientation_fine_sigma_rad must not exceed the coarse scale')
    if not 0 <= cfg.orientation_coarse_fraction <= 1:
        raise ValueError('orientation_coarse_fraction must be in [0, 1]')
    prog_scale = getattr(cfg, 'orientation_progress_scale', 0.0)
    prog_speed = getattr(cfg, 'orientation_progress_speed_rad_s', 2.0)
    if not math.isfinite(prog_scale) or prog_scale < 0:
        raise ValueError('orientation_progress_scale must be finite and nonnegative')
    if not math.isfinite(prog_speed) or prog_speed <= 0:
        raise ValueError('orientation_progress_speed_rad_s must be finite and positive')


def orientation_progress_reward(prev_error_rad, error_rad, dt, cfg):
    """Smooth bounded angular progress rate reward (per second).

    Pays +scale * tanh((theta_t - theta_{t+1}) / (dt * speed)) so turning toward
    the target yields immediate dense rollout credit inside short imagination
    horizons while remaining strictly bounded in [-scale, +scale] and telescoping
    to zero under symmetric oscillation.
    """
    scale = getattr(cfg, 'orientation_progress_scale', 0.0)
    if not scale:
        return torch.zeros_like(error_rad)
    speed = getattr(cfg, 'orientation_progress_speed_rad_s', 2.0)
    return scale * torch.tanh((prev_error_rad - error_rad) / (dt * speed))


def orientation_reward(error_rad, cfg):
    """Bounded SO(3) orientation reward in [0, orientation_reward_scale].

    Modes:
    - 'gaussian' (legacy V25-V33): 10 * (mix * exp(-(theta/sigma_c)^2) + (1-mix) * exp(-(theta/sigma_f)^2)).
    - 'linear_fine' (V34): strong far-field linear gradient plus sharp near-target Lorentzian pull:
      scale * (0.6 * clamp(1 - theta/pi, 0) + 0.4 / (1 + (theta / 0.25)^2)).
      Strict upper bound equals orientation_reward_scale at theta = 0; avoids both far-field
      Gaussian saturation and the ~26 deg near-goal plateau.
    """
    error_rad = angular_excess(error_rad, cfg)
    mode = getattr(cfg, 'orientation_reward_mode', 'gaussian')
    if mode == 'linear_fine':
        linear = (1.0 - error_rad / math.pi).clamp_min(0.0)
        fine = (1.0 + (error_rad / 0.25).square()).reciprocal()
        return cfg.orientation_reward_scale * (0.6 * linear + 0.4 * fine)
    mix = cfg.orientation_coarse_fraction
    if mix == 1.0:
        return cfg.orientation_reward_scale * torch.exp(-(error_rad / cfg.orientation_coarse_sigma_rad).square())
    if mix == 0.0:
        return cfg.orientation_reward_scale * torch.exp(-(error_rad / cfg.orientation_fine_sigma_rad).square())
    coarse = torch.exp(-(error_rad / cfg.orientation_coarse_sigma_rad).square())
    fine = torch.exp(-(error_rad / cfg.orientation_fine_sigma_rad).square())
    return cfg.orientation_reward_scale * (mix * coarse + (1 - mix) * fine)
