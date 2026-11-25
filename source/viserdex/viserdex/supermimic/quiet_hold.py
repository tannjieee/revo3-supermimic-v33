"""Tolerant arrival and orientation-gated motion costs, expressed per second.

Real and imagined rewards each multiply by control_dt exactly once. Saved
checkpoints missing these fields retain the original reward.
"""
import math
import torch

PROTOCOL = 'tolerant_arrival_quiet_hold_v1'
SINGLE_EXP_PROTOCOL = 'single_exp_arrival_quiet_hold_v2'
HOME_POSE_PROTOCOL = 'single_exp_quiet_hold_home_pose_v3'
LEGACY_DEFAULTS = dict(
    quiet_hold_enabled=False, orientation_tolerance_rad=0.0,
    hold_joint_motion_weight=0.0, hold_target_motion_weight=0.0,
    hold_joint_speed_scale_rad_s=math.pi / 9,
    hold_target_speed_scale_rad_s=math.pi / 9,
    home_pose_weight=0.0, home_pose_deadband_rad=math.pi / 12,
    home_pose_scale_rad=math.pi / 6,
)
QUIET_HOLD_CONFIG_FIELDS = tuple(LEGACY_DEFAULTS)


def fresh_options(enabled=True):
    return dict(LEGACY_DEFAULTS, **(dict(
        quiet_hold_enabled=True, orientation_tolerance_rad=math.radians(10.),
        orientation_coarse_fraction=1.0,
        hold_joint_motion_weight=1.0, hold_target_motion_weight=1.0,
    ) if enabled else {}))


def reward_protocol(cfg):
    if cfg.home_pose_weight:
        return HOME_POSE_PROTOCOL
    # Saved v1 runs retain their serialized two-scale mixture.
    return SINGLE_EXP_PROTOCOL if cfg.orientation_coarse_fraction == 1.0 else PROTOCOL


def validate_quiet_hold(cfg):
    if type(cfg.quiet_hold_enabled) is not bool:
        raise ValueError('quiet_hold_enabled must be boolean')
    for name in ('orientation_tolerance_rad', 'hold_joint_motion_weight', 'hold_target_motion_weight'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if cfg.orientation_tolerance_rad >= math.pi:
        raise ValueError('orientation_tolerance_rad must be below pi')
    for name in ('hold_joint_speed_scale_rad_s', 'hold_target_speed_scale_rad_s'):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(cfg.home_pose_weight) or cfg.home_pose_weight < 0:
        raise ValueError('home_pose_weight must be finite and nonnegative')
    if not math.isfinite(cfg.home_pose_deadband_rad) or cfg.home_pose_deadband_rad < 0:
        raise ValueError('home_pose_deadband_rad must be finite and nonnegative')
    if not math.isfinite(cfg.home_pose_scale_rad) or cfg.home_pose_scale_rad <= 0:
        raise ValueError('home_pose_scale_rad must be finite and positive')
    if cfg.home_pose_weight and not cfg.quiet_hold_enabled:
        raise ValueError('HOME posture preference requires quiet holding')
    if not cfg.quiet_hold_enabled and any(getattr(cfg, n) != LEGACY_DEFAULTS[n] for n in
            ('orientation_tolerance_rad', 'hold_joint_motion_weight', 'hold_target_motion_weight')):
        raise ValueError('Quiet holding parameters require quiet_hold_enabled')


def angular_excess(angle, cfg):
    return ((angle - cfg.orientation_tolerance_rad).clamp_min(0)
            if getattr(cfg, 'quiet_hold_enabled', False) else angle)


def motion_penalty(angle, joint_velocity, target, previous_target, dt, cfg):
    """Use executed target changes, never absolute targets or raw logits."""
    gate = torch.exp(-(angular_excess(angle, cfg) / cfg.stability_angle_sigma_rad).square())
    joint = (joint_velocity / cfg.hold_joint_speed_scale_rad_s).square().mean(-1)
    command = ((target - previous_target) / (dt * cfg.hold_target_speed_scale_rad_s)).square().mean(-1)
    return gate * (cfg.hold_joint_motion_weight * joint + cfg.hold_target_motion_weight * command)


def home_pose_penalty(joints, home, cfg):
    """All-phase preference on geometric joint angles; free motion inside the band."""
    excess = ((joints - home).abs() - cfg.home_pose_deadband_rad).clamp_min(0)
    return cfg.home_pose_weight * (excess / cfg.home_pose_scale_rad).square().mean(-1)


def imagined_motion_penalty(state, next_physical, action, angle, cfg, *, limits=None, encoder_bias=None):
    from .core import action_target
    from .state_layout import COMPACT
    # WM CUDA capture supplies resident constants; standalone audits may construct them.
    if limits is None:
        limits = action.new_tensor(cfg.network_joint_limits)
    target = action_target(state, action, limits, cfg)
    if cfg.dex_reference:
        from .dex_reference import EMA, ZERO
        previous = state[..., EMA] - state[..., ZERO]
    else:
        previous = action_target(state, state[..., COMPACT.last_action], limits, cfg)
        if encoder_bias is None:
            encoder_bias = state.new_tensor(cfg.encoder_bias_rad)
        initial = state[..., :21] - encoder_bias
        previous = torch.where(state[..., COMPACT.delay_ready].bool(), previous, initial)
    return motion_penalty(angle, next_physical[..., 21:42], target, previous, cfg.control_dt, cfg)


def validate_episode_timing(dt, steps, *, compact):
    if min(abs(dt - hz**-1) for hz in (20, 25, 30)) > 1e-8:
        raise ValueError('Expected 20, 25 or 30 Hz')
    durations = (10., 20.) if compact else (20.,)
    if min(abs(steps * dt - seconds) for seconds in durations) > 1e-8:
        raise ValueError('Expected a 10/20 second compact episode or 20 second legacy episode')


def configure_fresh_environment(env_cfg, *, enabled=True, duration=10., stability_scale=None):
    if duration not in (10., 20.):
        raise ValueError('Collection episode must be 10 or 20 seconds')
    command = env_cfg.commands.object_pose
    for name, value in fresh_options(enabled).items():
        setattr(command, name, value)
    if enabled:
        command.stability_reward_scale = 10. if stability_scale is None else stability_scale
    validate_quiet_hold(command)
    env_cfg.episode_length_s = duration
