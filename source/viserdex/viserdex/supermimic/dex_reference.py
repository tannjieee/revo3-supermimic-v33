"""Version 33: explicit encoder EMA memory and geometric, delay-free dynamics."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch

STATE_PROTOCOL = 'revo3_geometric_physics_encoder_ema_bias_wrench_v33'
OBSERVATION_PROTOCOL = 'revo3_joint_tracking_error_relative_rot6d_history3_v33'
EMA = slice(83, 104)
ZERO = slice(104, 125)
WRENCH = slice(125, 131)
DEFAULT_PROFILE = Path(__file__).resolve().parents[4] / 'configs/supermimic_v33/control.yaml'


def load_profile(path=DEFAULT_PROFILE):
    import yaml
    raw = Path(path).read_bytes()
    profile = yaml.safe_load(raw)
    validate_profile(profile)
    return profile, hashlib.sha256(raw).hexdigest()


def validate_profile(p):
    from .revised_contract import HARDWARE_JOINT_NAMES, CENTER_REVISION, urdf_limits
    if p['schema'] != 'revo3_ema_soft_limits_v33' or set(p['soft_limits_deg']) != set(HARDWARE_JOINT_NAMES):
        raise ValueError('v33 needs exactly the 21 named Revo3 soft limits')
    if p['history'] != 3 or p['fixed_encoder_bias_deg'] != 0 or not 0 < p['ema_alpha'] <= 1:
        raise ValueError('v33 requires three frames, disabled fixed offset and a valid EMA')
    bias = np.radians(p['zero_bias_deg'])
    if bias.shape != (2,) or not np.isfinite(bias).all() or bias[0] > bias[1]:
        raise ValueError('Invalid encoder zero randomization')
    limits = np.radians([p['soft_limits_deg'][n] for n in HARDWARE_JOINT_NAMES])
    _, asset = urdf_limits(CENTER_REVISION)
    hard = np.array(asset['geometric_limits_rad'])
    if (limits.shape != (21, 2) or not np.isfinite(limits).all() or (limits[:, 0] >= limits[:, 1]).any()
            or (limits[:, 0]-bias[1] < hard[:, 0]-2e-7).any()
            or (limits[:, 1]-bias[0] > hard[:, 1]+2e-7).any()):
        raise ValueError('Soft targets minus every allowed zero offset must fit mechanical limits')
    for name in ('wrench_force_max_N', 'wrench_torque_max_Nm', 'contact_loss_weight'):
        if not math.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f'Invalid {name}')
    if not 0 <= p['wrench_probability'] <= 1:
        raise ValueError('Invalid wrench probability')
    for name in ('target_slew_rate_deg_s', 'home_pose_weight', 'home_pose_deadband_deg', 'hold_target_motion_weight',
                 'object_motion_angular_speed_rad_s', 'object_motion_linear_speed_m_s'):
        if name in p and (not math.isfinite(p[name]) or p[name] < 0):
            raise ValueError(f'Invalid {name}')
    if 'home_pose_scale_deg' in p and (not math.isfinite(p['home_pose_scale_deg']) or p['home_pose_scale_deg'] <= 0):
        raise ValueError('Invalid home_pose_scale_deg')
    if bool(p.get('object_motion_angular_speed_rad_s', 0)) != bool(p.get('object_motion_linear_speed_m_s', 0)):
        raise ValueError('Global object speed costs require both thresholds')
    # Optional V34 optimization profile overrides (omitted in legacy V33 YAML profiles).
    if 'wm_dynamics_mode' in p and p['wm_dynamics_mode'] not in ('delta54', 'velocity_integrated'):
        raise ValueError(f"Invalid wm_dynamics_mode: {p['wm_dynamics_mode']}")
    if 'wm_integration_coeff' in p and (not math.isfinite(p['wm_integration_coeff']) or not 0 < p['wm_integration_coeff'] <= 1):
        raise ValueError('Invalid wm_integration_coeff')
    if 'wm_loss_normalization' in p and p['wm_loss_normalization'] not in ('running_std', 'bryson'):
        raise ValueError(f"Invalid wm_loss_normalization: {p['wm_loss_normalization']}")
    if 'wm_bryson_scales' in p:
        scales = p['wm_bryson_scales']
        if not isinstance(scales, (list, tuple)) or len(scales) != 6 or any(not math.isfinite(v) or v <= 0 for v in scales):
            raise ValueError('wm_bryson_scales must contain 6 finite positive nominal errors')
    if 'wm_activation' in p and p['wm_activation'] not in ('silu', 'gelu'):
        raise ValueError(f"Invalid wm_activation: {p['wm_activation']}")
    if 'contact_loss_reduction' in p and p['contact_loss_reduction'] not in ('sum', 'mean'):
        raise ValueError(f"Invalid contact_loss_reduction: {p['contact_loss_reduction']}")
    for name in ('wm_reward_loss_weight', 'wm_observation_loss_weight', 'wm_cost_loss_weight'):
        if name in p and (not math.isfinite(p[name]) or p[name] < 0):
            raise ValueError(f'Invalid {name}')
    if 'wm_holdout_fraction' in p and (not math.isfinite(p['wm_holdout_fraction']) or not 0 <= p['wm_holdout_fraction'] < 0.5):
        raise ValueError('Invalid wm_holdout_fraction')
    if 'replay_sampling_mode' in p and p['replay_sampling_mode'] not in ('legacy', 'uniform_failure'):
        raise ValueError(f"Invalid replay_sampling_mode: {p['replay_sampling_mode']}")
    if 'failure_sample_fraction' in p and (not math.isfinite(p['failure_sample_fraction']) or not 0 <= p['failure_sample_fraction'] <= 1):
        raise ValueError('Invalid failure_sample_fraction')
    for name in ('grouped_physical_loss', 'deterministic_imagination', 'target_rate_pre_slew_cost',
                 'critic_layer_norm', 'critic_value_cap', 'critic_mean_loss',
                 'sequential_goals', 'compact_critic_costs',
                 'target_slew_ste', 'actor_lambda_return', 'goal_curriculum_enabled',
                 'wm_residual', 'actor_layer_norm'):
        if name in p and type(p[name]) is not bool:
            raise ValueError(f'{name} must be boolean')
    if 'identified_hand_profile' in p:
        from .identified_hand import SUPPORTED_PROFILES
        if p['identified_hand_profile'] not in SUPPORTED_PROFILES:
            raise ValueError(f"Invalid identified_hand_profile: {p['identified_hand_profile']}")
    if 'actor_activation' in p and p['actor_activation'] not in ('elu', 'silu', 'gelu'):
        raise ValueError(f"Invalid actor_activation: {p['actor_activation']}")
    for name in ('wm_huber_delta', 'weight_decay', 'critic_huber_delta', 'exploration_min_std_deg',
                 'wm_input_noise_std'):
        if name in p and (not math.isfinite(p[name]) or p[name] < 0):
            raise ValueError(f'Invalid {name}')
    if 'wm_step_decay' in p and (not math.isfinite(p['wm_step_decay']) or not 0 < p['wm_step_decay'] <= 1):
        raise ValueError('Invalid wm_step_decay')
    if 'actor_failure_sample_fraction' in p and (
            not math.isfinite(p['actor_failure_sample_fraction']) or not 0 <= p['actor_failure_sample_fraction'] <= 0.5):
        raise ValueError('Invalid actor_failure_sample_fraction')
    if 'critic_ema_tau' in p and (not math.isfinite(p['critic_ema_tau']) or not 0 <= p['critic_ema_tau'] <= 1):
        raise ValueError('Invalid critic_ema_tau')
    if 'wm_ema_tau' in p and (not math.isfinite(p['wm_ema_tau']) or not 0 <= p['wm_ema_tau'] <= 1):
        raise ValueError('Invalid wm_ema_tau')
    if 'wm_grad_clip' in p and (not math.isfinite(p['wm_grad_clip']) or p['wm_grad_clip'] <= 0):
        raise ValueError('Invalid wm_grad_clip')
    if 'lr_min_ratio' in p and (not math.isfinite(p['lr_min_ratio']) or not 0 < p['lr_min_ratio'] <= 1):
        raise ValueError('Invalid lr_min_ratio')
    for name in ('critic_ensemble_size', 'lr_decay_iters', 'wm_updates'):
        if name in p and (type(p[name]) is not int or p[name] < 1):
            raise ValueError(f'Invalid {name}')
    if 'exploration_decay_iters' in p and (type(p['exploration_decay_iters']) is not int or p['exploration_decay_iters'] < 0):
        raise ValueError('Invalid exploration_decay_iters')
    if 'orientation_reward_mode' in p and p['orientation_reward_mode'] not in ('gaussian', 'linear_fine'):
        raise ValueError(f"Invalid orientation_reward_mode: {p['orientation_reward_mode']}")
    if 'orientation_tolerance_deg' in p and (not math.isfinite(p['orientation_tolerance_deg']) or not 0 <= p['orientation_tolerance_deg'] < 180):
        raise ValueError('Invalid orientation_tolerance_deg')
    if 'orientation_progress_scale' in p and (not math.isfinite(p['orientation_progress_scale']) or p['orientation_progress_scale'] < 0):
        raise ValueError('Invalid orientation_progress_scale')
    if 'orientation_progress_speed_rad_s' in p and (not math.isfinite(p['orientation_progress_speed_rad_s']) or p['orientation_progress_speed_rad_s'] <= 0):
        raise ValueError('Invalid orientation_progress_speed_rad_s')
    if 'actor_lr_scale' in p and (not math.isfinite(p['actor_lr_scale']) or not 0 < p['actor_lr_scale'] <= 1):
        raise ValueError('Invalid actor_lr_scale')
    for name in ('actor_ramp_iters', 'goal_curriculum_cooldown_iters'):
        if name in p and (type(p[name]) is not int or p[name] < 0):
            raise ValueError(f'Invalid {name}')
    for name in ('actor_updates', 'critic_epochs', 'recent_window_steps'):
        if name in p and (type(p[name]) is not int or p[name] < 1):
            raise ValueError(f'Invalid {name}')
    if 'recent_sample_fraction' in p and (not math.isfinite(p['recent_sample_fraction']) or not 0 <= p['recent_sample_fraction'] <= 1):
        raise ValueError('Invalid recent_sample_fraction')
    if 'exploration_position_std_deg' in p and (not math.isfinite(p['exploration_position_std_deg']) or p['exploration_position_std_deg'] <= 0):
        raise ValueError('Invalid exploration_position_std_deg')
    if 'goal_curriculum_start_deg' in p and (not math.isfinite(p['goal_curriculum_start_deg']) or not 5.0 <= p['goal_curriculum_start_deg'] <= 180.0):
        raise ValueError('Invalid goal_curriculum_start_deg')
    if 'goal_curriculum_step_deg' in p and (not math.isfinite(p['goal_curriculum_step_deg']) or not 1.0 <= p['goal_curriculum_step_deg'] <= 180.0):
        raise ValueError('Invalid goal_curriculum_step_deg')
    if 'goal_curriculum_threshold_ratio' in p and (not math.isfinite(p['goal_curriculum_threshold_ratio']) or not 0 < p['goal_curriculum_threshold_ratio'] < 1):
        raise ValueError('Invalid goal_curriculum_threshold_ratio')
    if 'action_residual_deg' in p:
        res = p['action_residual_deg']
        if isinstance(res, (int, float)):
            if not math.isfinite(res) or res <= 0:
                raise ValueError('action_residual_deg must be finite and positive')
        elif isinstance(res, dict):
            unknown = set(res) - set(HARDWARE_JOINT_NAMES) - {'default'}
            if unknown:
                raise ValueError(f'Unknown joint names in action_residual_deg: {sorted(unknown)}')
            if 'default' not in res and set(res) != set(HARDWARE_JOINT_NAMES):
                raise ValueError('action_residual_deg dict requires either all 21 joints or a default entry')
            for k, v in res.items():
                if not math.isfinite(v) or v <= 0:
                    raise ValueError(f'Invalid action_residual_deg[{k}]')
        else:
            raise ValueError('action_residual_deg must be a positive number or joint-name mapping')
    for name in ('ema_delta_penalty_weight', 'joint_speed_penalty_weight', 'cage_penalty_weight', 'cage_safe_radius_m',
                 'object_angular_speed_penalty_weight', 'object_angular_speed_limit_rad_s',
                 'object_linear_speed_penalty_weight', 'object_linear_speed_limit_m_s'):
        if name in p and (not math.isfinite(p[name]) or p[name] < 0):
            raise ValueError(f'Invalid {name}')
    for name in ('cage_scale_m', 'object_angular_speed_scale_rad_s', 'object_linear_speed_scale_m_s'):
        if name in p and (not math.isfinite(p[name]) or p[name] <= 0):
            raise ValueError(f'Invalid {name}')
    return limits, asset


def action_residual_vector_rad(p):
    """Return 21-element tuple of per-joint residual radii in hardware order, or () if disabled."""
    if not p or 'action_residual_deg' not in p:
        return ()
    from .revised_contract import HARDWARE_JOINT_NAMES
    res = p['action_residual_deg']
    if isinstance(res, (int, float)):
        return tuple(math.radians(float(res)) for _ in HARDWARE_JOINT_NAMES)
    default = res.get('default')
    return tuple(math.radians(float(res.get(n, default))) for n in HARDWARE_JOINT_NAMES)


def home_pose_options(p):
    return dict(home_pose_weight=p.get('home_pose_weight', 0.),
                home_pose_deadband_rad=math.radians(p.get('home_pose_deadband_deg', 15.)),
                home_pose_scale_rad=math.radians(p.get('home_pose_scale_deg', 30.)))


def reward_options(p):
    """Optional profile overrides shared by real reward and imagined reward.

    An omitted quiet-target weight retains the existing quiet-hold defaults.
    This is independent of the global target-rate cost and controller slew limit.
    """
    options = home_pose_options(p)
    if 'hold_target_motion_weight' in p:
        options['hold_target_motion_weight'] = p['hold_target_motion_weight']
    if 'orientation_reward_mode' in p:
        options['orientation_reward_mode'] = p['orientation_reward_mode']
    if 'orientation_tolerance_deg' in p:
        options['orientation_tolerance_rad'] = math.radians(p['orientation_tolerance_deg'])
    for key in ('ema_delta_penalty_weight', 'joint_speed_penalty_weight',
                'cage_penalty_weight', 'cage_safe_radius_m', 'cage_scale_m',
                'object_angular_speed_penalty_weight', 'object_angular_speed_limit_rad_s', 'object_angular_speed_scale_rad_s',
                'object_linear_speed_penalty_weight', 'object_linear_speed_limit_m_s', 'object_linear_speed_scale_m_s',
                'sequential_goals',
                'orientation_progress_scale', 'orientation_progress_speed_rad_s'):
        if key in p:
            options[key] = p[key]
    return options


def profile_options(p, digest):
    limits, _ = validate_profile(p)
    from .revised_contract import HARDWARE_JOINT_NAMES
    options = dict(dex_reference=True, state_protocol=STATE_PROTOCOL, observation_protocol=OBSERVATION_PROTOCOL,
        control_profile=p, control_profile_sha256=digest, network_joint_limits=tuple(map(tuple, limits.tolist())),
        encoder_bias_rad=(0.,)*21, ema_alpha=p['ema_alpha'], history=3, include_delay_ready=False,
        grouped_physical_loss=p.get('grouped_physical_loss', False), contact_loss_weight=p['contact_loss_weight'],
        contact_body_names=('world',)+tuple(n.replace('_joint', '_link') for n in HARDWARE_JOINT_NAMES),
        zero_bias_range_rad=tuple(math.radians(v) for v in p['zero_bias_deg']),
        wrench_probability=p['wrench_probability'], wrench_force_max=p['wrench_force_max_N'],
        wrench_torque_max=p['wrench_torque_max_Nm'],
        target_slew_rate_rad_s=math.radians(p.get('target_slew_rate_deg_s', 0.)),
        object_motion_angular_speed_rad_s=p.get('object_motion_angular_speed_rad_s', 0.),
        object_motion_linear_speed_m_s=p.get('object_motion_linear_speed_m_s', 0.),
        **reward_options(p))
    for key in ('wm_dynamics_mode', 'wm_integration_coeff', 'wm_loss_normalization', 'wm_activation',
                'contact_loss_reduction', 'wm_reward_loss_weight', 'wm_observation_loss_weight',
                'wm_cost_loss_weight', 'wm_holdout_fraction', 'replay_sampling_mode',
                'failure_sample_fraction', 'deterministic_imagination', 'target_rate_pre_slew_cost',
                'critic_layer_norm', 'critic_value_cap', 'critic_mean_loss', 'compact_critic_costs',
                'target_slew_ste', 'actor_lr_scale', 'actor_ramp_iters', 'actor_updates',
                'critic_epochs', 'actor_lambda_return', 'recent_sample_fraction', 'recent_window_steps',
                'goal_curriculum_enabled', 'goal_curriculum_start_deg', 'goal_curriculum_step_deg',
                'goal_curriculum_threshold_ratio', 'goal_curriculum_cooldown_iters',
                'identified_hand_profile', 'wm_huber_delta', 'wm_residual', 'weight_decay',
                'critic_ensemble_size', 'critic_ema_tau', 'critic_huber_delta',
                'exploration_decay_iters', 'lr_decay_iters', 'lr_min_ratio', 'wm_updates',
                'wm_step_decay', 'wm_input_noise_std', 'actor_layer_norm', 'actor_activation',
                'actor_failure_sample_fraction', 'wm_ema_tau', 'wm_grad_clip'):
        if key in p:
            options[key] = p[key]
    if 'exploration_position_std_deg' in p:
        options['exploration_position_std_rad'] = math.radians(p['exploration_position_std_deg'])
    if 'exploration_min_std_deg' in p:
        options['exploration_min_std_rad'] = math.radians(p['exploration_min_std_deg'])
    if 'wm_bryson_scales' in p:
        options['wm_bryson_scales'] = tuple(float(v) for v in p['wm_bryson_scales'])
    residual = action_residual_vector_rad(p)
    if residual:
        options['dex_action_residual_rad'] = residual
    return options


def raw_encoder_target(action, limits, *, home=None, residual_rad=None):
    """Decode tanh action to encoder target: either full soft limits or HOME-local residual."""
    if residual_rad is None or len(residual_rad) == 0:
        from .policy_contract import absolute_joint_target
        return absolute_joint_target(action, limits)
    home_t = home if isinstance(home, torch.Tensor) else action.new_tensor(home)
    res_t = residual_rad if isinstance(residual_rad, torch.Tensor) else action.new_tensor(residual_rad)
    lower, upper = limits[..., 0], limits[..., 1]
    return (home_t + res_t * action).clamp(lower, upper)


def filter_target(previous, raw, alpha, max_delta=0., *, ste=False):
    """Rate-limit the executed EMA command; preserve exact legacy math when off.

    When ste=True, the forward command is identically rate-limited while the
    backward pass preserves straight-through gradients through the clamp.
    """
    encoded = alpha * raw + (1 - alpha) * previous
    if max_delta:
        clipped = previous + (encoded - previous).clamp(-max_delta, max_delta)
        encoded = encoded + (clipped - encoded).detach() if ste else clipped
    return encoded


def target(state, action, limits, alpha, max_delta=0., *, home=None, residual_rad=None, ste=False):
    raw = raw_encoder_target(action, limits, home=home, residual_rad=residual_rad)
    encoded = filter_target(state[..., EMA], raw, alpha, max_delta, ste=ste)
    return encoded-state[..., ZERO]


def unclipped_target(state, action, limits, alpha, *, home=None, residual_rad=None):
    """Pre-slew EMA geometric target so rate-limit costs retain nonzero gradients."""
    raw = raw_encoder_target(action, limits, home=home, residual_rad=residual_rad)
    encoded = alpha * raw + (1 - alpha) * state[..., EMA]
    return encoded - state[..., ZERO]


def observation(joints_normalized, state, limits, position, orientation):
    from .core import rotation_features, quat_mul, quat_conjugate
    half = (limits[..., 1]-limits[..., 0])*.5
    target_normalized = (state[..., EMA]-limits.mean(-1))/half
    # Use the same noisy encoder reading for position and tracking error.
    return torch.cat((joints_normalized, joints_normalized-target_normalized, state[..., 55:76],
        position, state[..., 76:79], rotation_features(quat_mul(orientation, quat_conjugate(state[..., 79:83])))), -1)


def sample_wrench(reference, cfg):
    active = torch.rand(*reference.shape[:-1], 1, device=reference.device) < cfg.wrench_probability
    unit = 2*torch.rand(*reference.shape[:-1], 6, device=reference.device)-1
    # Scalar multiplies avoid constructing/copying a CPU constant during capture.
    return torch.cat((unit[..., :3]*cfg.wrench_force_max,
                      unit[..., 3:]*cfg.wrench_torque_max), -1)*active


def configure_environment(env_cfg, p, digest):
    from .revised_contract import configure_environment as configure_base, CENTER_REVISION
    configure_base(env_cfg, CENTER_REVISION)
    limits, _ = validate_profile(p)
    hand_profile = p.get('identified_hand_profile', 'fit_03')
    if hand_profile != 'fit_03':
        from .identified_hand import actuator_config
        env_cfg.scene.robot.actuators = {
            'fingers': actuator_config(profile=hand_profile, use_identified_effort_limits=True)
        }
    action = env_cfg.actions.joint_pos
    action.dex_reference_profile = p
    action.training_limits_sha256 = digest
    action.position_limits_rad = tuple(map(tuple, limits.tolist()))
    action.encoder_bias_rad = (0.,)*21
    action.alpha = p['ema_alpha']
    for actuator in env_cfg.scene.robot.actuators.values():
        actuator.min_delay = actuator.max_delay = 0


def prepare_saved_identified_hand(env_cfg, saved):
    """Instantiate non-default identified actuator schema before from_dict merge on resume/playback."""
    dex_profile = saved.get('actions', {}).get('joint_pos', {}).get('dex_reference_profile')
    if isinstance(dex_profile, dict):
        hand_profile = dex_profile.get('identified_hand_profile', 'fit_03')
        if hand_profile != 'fit_03':
            from .identified_hand import actuator_config
            env_cfg.scene.robot.actuators = {
                'fingers': actuator_config(profile=hand_profile, use_identified_effort_limits=True)
            }
            for actuator in env_cfg.scene.robot.actuators.values():
                actuator.min_delay = actuator.max_delay = 0


def state_fields():
    from .state_layout import fields
    result = fields()
    result.pop('actuator_delay_ready')
    result.update(ema_encoder_target_rad=[83,104], episode_encoder_zero_rad=[104,125],
                  current_object_wrench_base_N_Nm=[125,131])
    return result


def observation_fields():
    return dict(joint_position_normalized=[0,21], measured_minus_previous_ema_normalized=[21,42],
        last_raw_action=[42,63], object_position_base_m=[63,66], goal_position_base_m=[66,69],
        relative_rotation_6d=[69,75])
