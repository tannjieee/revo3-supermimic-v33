"""Explicit playback-only command-shaping experiment, not a trained policy contract.

Bounds target slew, not actual joint velocity. Caller must pass the resulting
normalized action to the simulator so observation history reflects applied commands.
"""
import math
import torch
from .policy_contract import absolute_joint_target


class TargetShaper:
    def __init__(self, limits, dt, velocity_deg_s, tau_s=0.):
        if not all(math.isfinite(v) for v in (dt, velocity_deg_s, tau_s)) or dt <= 0 or velocity_deg_s <= 0 or tau_s < 0:
            raise ValueError('Require positive dt/velocity and nonnegative finite tau')
        if not torch.isfinite(limits).all() or not (limits[..., 1] > limits[..., 0]).all():
            raise ValueError('Invalid joint position limits')
        self.limits = limits
        self.max_step = math.radians(velocity_deg_s) * dt
        self.alpha = 1. if tau_s == 0 else -math.expm1(-dt / tau_s)
        self.previous = None

    def reset(self, measured_encoder):
        if measured_encoder.shape != self.limits.shape[:-1] or not torch.isfinite(measured_encoder).all():
            raise ValueError('Invalid measured encoder reset state')
        lo, hi = self.limits.unbind(-1)
        self.previous = measured_encoder.clamp(lo, hi).clone()

    def __call__(self, raw):
        if self.previous is None:
            raise RuntimeError('Reset target shaper from measured encoder first')
        if raw.shape != self.previous.shape or not torch.isfinite(raw).all() or (raw.abs() > 1 + 1e-6).any():
            raise ValueError('Expected finite tanh action')
        lo, hi = self.limits.unbind(-1)
        desired = absolute_joint_target(raw.clamp(-1, 1), self.limits)
        delta = ((desired - self.previous) * self.alpha).clamp(-self.max_step, self.max_step)
        target = (self.previous + delta).clamp(lo, hi)
        self.previous = target.clone()
        return (2 * (target - lo) / (hi - lo) - 1).clamp(-1, 1)
