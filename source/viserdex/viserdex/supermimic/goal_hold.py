"""Pure, shared sampled-time hold protocol. No simulator or learned surrogate."""
import math
import torch


def required_steps(dt, duration=0.5):
    if dt <= 0 or duration <= 0:
        raise ValueError("Control dt and hold duration must be positive")
    n = round(duration / dt)
    if n < 1 or not math.isclose(n * dt, duration, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Hold duration must be a positive integer number of control steps")
    return n


def hold_transition(previous, error, speed, dt, *, angle_threshold=0.1,
                    speed_threshold=0.5, duration=0.5, blocked=None):
    """One post-physics sample; pulse once on completion and clear the counter.

    previous is the number of consecutive qualifying samples before this step.
    A reset/termination/timeout cannot earn a new success or carry partial credit.
    Thresholds are strict; no interpolation between control samples is claimed.
    """
    n = required_steps(dt, duration)
    valid = torch.isfinite(error) & torch.isfinite(speed)
    valid = valid & (error < angle_threshold) & (speed < speed_threshold)
    if blocked is not None:
        valid = valid & ~blocked
    ticks = torch.where(valid, previous + 1, torch.zeros_like(previous))
    completed = ticks >= n
    return torch.where(completed, torch.zeros_like(ticks), ticks), completed


def hold_kwargs(cfg=None):
    if cfg is None:
        return dict(angle_threshold=0.1, speed_threshold=0.5, duration=0.5)
    return dict(angle_threshold=cfg.hold_angle_threshold,
                speed_threshold=cfg.hold_speed_threshold, duration=cfg.hold_duration)


def persistent_hold_transition(previous, error, speed, dt, *, already_completed,
                               angle_threshold=0.1, speed_threshold=0.5,
                               duration=0.5, blocked=None):
    """Continuous hold progress and a one-time diagnostic event per command."""
    n = required_steps(dt, duration)
    valid = torch.isfinite(error) & torch.isfinite(speed) & (error < angle_threshold) & (speed < speed_threshold)
    if blocked is not None:
        valid = valid & ~blocked
    ticks = torch.where(valid, (previous + 1).clamp_max(n), torch.zeros_like(previous))
    return ticks, (ticks >= n) & ~already_completed
