"""Differentiable task algebra. Quaternions are XYZW; spatial state is in hand base.

The compact 84D state holds the 55D physical core, one last action, persistent
pose goal and actuator-delay reset flag. Legacy 173D tasks keep
their separate historical layout.
"""

from .state_layout import layout, COMPACT, REFERENCE, STATE_PROTOCOL
from . import dex_reference as dex
from .goal_hold import hold_transition, persistent_hold_transition, hold_kwargs, required_steps
from .policy_contract import LEGACY_OBSERVATION_PROTOCOL, DIRECT_OBSERVATION_PROTOCOL, absolute_joint_target
from .orientation_reward import orientation_reward, validate_orientation_reward
from .position_reward import position_reward, position_hold_valid, validate_position_reward
from .stability_reward import stability_reward, validate_stability_reward
from dataclasses import dataclass
import math
import torch
from viserdex.distillation.noise import (
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_unique,
    rotation_vector_quat,
    pose_features,
)

Q = slice(0, 21)
QD = slice(21, 42)
POS = slice(42, 45)
ROT = slice(45, 49)
VEL = slice(49, 52)
OMEGA = slice(52, 55)
EMA = slice(55, 76)
ACTIONS = slice(76, 160)
GOAL_POS = slice(160, 163)
GOAL_ROT = slice(163, 167)
TIME = slice(167, 168)
COUNT = slice(168, 169)
BASE_POS = slice(169, 172)
HOLD = slice(172, 173)  # consecutive qualifying control samples (privileged task state)
STATE_DIM = 173
OBS_DIM = 124


@dataclass
class Config:
    dex_reference: bool = False
    control_profile: dict | None = None
    control_profile_sha256: str = ""
    zero_bias_range_rad: tuple = (0., 0.)
    contact_body_names: tuple = ()
    contact_loss_weight: float = .1
    wrench_probability: float = 0.
    wrench_force_max: float = 0.
    wrench_torque_max: float = 0.
    observation_protocol: str = LEGACY_OBSERVATION_PROTOCOL
    state_protocol: str = "legacy173"
    encoder_bias_rad: tuple = ()
    joint_velocity_scale: float = 0.1
    include_object_position: bool = True
    goal_mode: str = "resample_on_success"
    track_object_position: bool = True  # Missing field in historical checkpoints means pose tracking.
    goal_evaluation_s: float = 10.0  # Diagnostics only, never a policy input or terminal boundary.
    stability_reward_scale: float = 3.0
    stability_angle_sigma_rad: float = math.pi / 12
    stability_position_sigma_m: float = 0.02
    stability_angular_speed_scale: float = 0.5
    stability_linear_speed_scale: float = 0.05
    quiet_hold_enabled: bool = False  # Missing in saved runs: preserve historical rewards.
    orientation_tolerance_rad: float = 0.0
    hold_joint_motion_weight: float = 0.0
    hold_target_motion_weight: float = 0.0
    hold_joint_speed_scale_rad_s: float = math.pi / 9
    hold_target_speed_scale_rad_s: float = math.pi / 9
    home_pose_weight: float = 0.0
    home_pose_deadband_rad: float = math.pi / 12
    home_pose_scale_rad: float = math.pi / 6
    position_reward_scale: float = 2.0
    position_tolerance_m: float = 0.01
    position_sigma_m: float = 0.05
    hold_position_threshold_m: float = 0.02
    hold_linear_speed_threshold_m_s: float = 0.05
    orientation_reward_mode: str = "gaussian"
    orientation_reward_scale: float = 10.0
    orientation_coarse_sigma_rad: float = math.pi / 2
    orientation_fine_sigma_rad: float = math.pi / 18
    orientation_coarse_fraction: float = 0.3
    geometry_enabled: bool = False  # enabled explicitly by the Geometry-v1 entrypoint
    compile_geometry: bool = False
    geometry_loss_weight: float = 0.5
    grouped_physical_loss: bool = False
    control_dt: float = 1 / 30
    ema_alpha: float = 0.95
    target_slew_rate_rad_s: float = 0.0  # Disabled for historical checkpoints.
    action_mode: str = "limits"
    action_home: tuple = ()
    action_offset_scale: float = 0.5
    action_warmup_s: float = 0.4
    success_bonus: float = 250.0
    action_rate_weight: float = 0.01
    joint_vel_weight: float = 2.5e-5
    action_l2_weight: float = 1e-4
    action_second_diff_weight: float = 0.0
    hand_pose_weight: float = 0.0
    hold_angle_threshold: float = 0.1
    hold_speed_threshold: float = 0.5
    hold_duration: float = 0.5
    actor_lr_scale: float = 1.0
    history: int = 10
    wm_window: int = 8
    horizon: int = 32
    wm_updates: int = 10
    wm_batch: int = 2048
    actor_batch: int = 4096
    critic_epochs: int = 5
    critic_minibatches: int = 4
    microbatch: int = 128
    wm_microbatch: int | None = None
    actor_microbatch: int | None = None
    critic_microbatch: int | None = None
    amp: str = "none"
    buffer_steps: int = 1000
    warmup_steps: int = 64
    adaptive_replay: bool = False
    adaptive_grasps: bool = False
    event_capacity: int = 4096
    event_fraction: float = 0.25
    actor_ramp_iters: int = 1000  # restore the validated Geometry-v1 policy LR ramp after WM warmup
    wm_only_iters: int = 500
    utd: float = 0.4
    learning_rate: float = 2e-4
    lr_decay_iters: int = 250000
    gamma: float = 0.99
    lam: float = 0.95
    exploration: float = 0.1
    wm_hidden: tuple = (2048, 2048, 2048)
    actor_hidden: tuple = (2048, 1024, 512, 256)
    critic_hidden: tuple = (256, 128)
    # Revo3-specific soft survival width, in meters. Hard real termination is unchanged.
    survival_width: float = 0.01
    # Missing in v27 and earlier checkpoints: preserve their analytic survival.
    wm_task_losses: bool = False
    wm_reward_loss_weight: float = 1.0
    wm_observation_loss_weight: float = 1.0
    wm_termination_loss_weight: float = 1.0
    wm_cost_loss_weight: float = 1.0
    # Legacy defaults keep version-25 checkpoints exactly loadable.
    network_revision: str = "legacy"
    network_joint_limits: tuple = ()
    include_delay_ready: bool = True
    hand_center_base_m: tuple = (0.,0.,0.)
    drop_radius_m: float = .3
    # Missing fields preserve the historical root-position sphere on resume.
    drop_boundary_mode: str = 'root_sphere'
    drop_ellipsoid_center_m: tuple = (0., 0., 0.)
    drop_ellipsoid_semi_axes_m: tuple = (.3, .3, .3)
    drop_object_com_local_m: tuple = (0., 0., 0.)
    drop_boundary_sha256: str = ''
    constraints_enabled: bool = False
    object_hold_costs_enabled: bool = False  # Historical checkpoints have only two joint costs.
    cost_object_hold_angle_sigma_rad: float = math.pi / 12
    object_motion_angular_speed_rad_s: float = 0.0
    object_motion_linear_speed_m_s: float = 0.0
    cost_velocity_limit_rad_s: float = math.pi / 2
    cost_target_rate_limit_rad_s: float = math.pi / 3
    cost_penalty: float = 1.
    # Missing in v28 and earlier: retain their original value/cost semantics.
    stable_value_learning: bool = False
    paper_critic: bool = False  # Missing in old checkpoints: preserve real-TD/BCE/EMA.
    critic_target_tau: float = .02
    critic_td_steps: int = 1  # Missing in historical checkpoints: one-step TD.
    input_joint_velocity_std_floor_rad_s: float = 0.0  # Historical normalization unchanged.
    # v30 retains absolute position commands. Missing fields preserve v29.
    absolute_policy_optimization: bool = False
    exploration_position_std_rad: float = math.pi / 360  # 0.5 degrees
    exploration_correlation_s: float = .2
    warmup_joint_amplitude_rad: float = 0.0  # Opt-in bounded HOME excitation; real pretraining collection only.
    warmup_joint_segment_steps: int = 30
    actor_logit_threshold: float = 2.
    actor_logit_penalty: float = .05
    actor_gradient_audit_interval: int = 50
    input_position_scale: float = 1.0
    input_linear_velocity_scale: float = 1.0
    input_angular_velocity_scale: float = 1.0
    # V34 Phase 0/1 optimization flags (all disabled/legacy by default for V33 compatibility).
    wm_dynamics_mode: str = "delta54"
    wm_integration_coeff: float = 0.55
    wm_loss_normalization: str = "running_std"
    wm_bryson_scales: tuple = (0.05, 1.5, 0.005, 0.03, 0.10, 0.50)
    wm_activation: str = "silu"
    contact_loss_reduction: str = "sum"
    wm_holdout_fraction: float = 0.0
    replay_sampling_mode: str = "legacy"
    failure_sample_fraction: float = 0.25
    deterministic_imagination: bool = False
    target_rate_pre_slew_cost: bool = False
    critic_layer_norm: bool = False
    critic_value_cap: bool = False
    critic_mean_loss: bool = False
    # V34 Phase 2 optimization flags (control space, sequential goals, compact Critic, smooth/cage regs).
    dex_action_residual_rad: tuple = ()
    sequential_goals: bool = False
    compact_critic_costs: bool = False
    ema_delta_penalty_weight: float = 0.0
    joint_speed_penalty_weight: float = 0.0
    cage_penalty_weight: float = 0.0
    cage_safe_radius_m: float = 0.045
    cage_scale_m: float = 0.025

    @property
    def uses_v34_optimizations(self):
        return (
            self.wm_dynamics_mode != "delta54"
            or self.wm_loss_normalization != "running_std"
            or self.wm_activation != "silu"
            or self.contact_loss_reduction != "sum"
            or self.wm_holdout_fraction > 0.0
            or self.replay_sampling_mode != "legacy"
            or self.deterministic_imagination
            or self.target_rate_pre_slew_cost
            or self.critic_layer_norm
            or self.critic_value_cap
            or self.critic_mean_loss
            or self.orientation_reward_mode != "gaussian"
            or bool(self.dex_action_residual_rad)
            or self.sequential_goals
            or self.compact_critic_costs
            or self.ema_delta_penalty_weight > 0.0
            or self.joint_speed_penalty_weight > 0.0
            or self.cage_penalty_weight > 0.0
            or (self.wm_task_losses and min(
                self.wm_reward_loss_weight, self.wm_observation_loss_weight, self.wm_cost_loss_weight) == 0.0)
        )

    @property
    def uses_target_critic(self):
        return self.stable_value_learning and not self.paper_critic

    def __post_init__(self):
        for name in ('target_slew_rate_rad_s', 'object_motion_angular_speed_rad_s', 'object_motion_linear_speed_m_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.target_slew_rate_rad_s and not self.dex_reference:
            raise ValueError('Target slew limit requires explicit v33 controller memory')
        if bool(self.object_motion_angular_speed_rad_s) != bool(self.object_motion_linear_speed_m_s):
            raise ValueError('Global object speed costs require both thresholds')
        if self.object_motion_angular_speed_rad_s and not self.object_hold_costs_enabled:
            raise ValueError('Global object speed costs require object cost heads')
        if self.home_pose_weight and (len(self.action_home) != 21 or not all(math.isfinite(x) for x in self.action_home)):
            raise ValueError('HOME posture preference requires 21 finite geometric joint angles')
        if self.dex_reference != (self.state_protocol == dex.STATE_PROTOCOL):
            raise ValueError('v33 state protocol and profile flag must agree')
        if self.dex_reference:
            expected = dex.profile_options(self.control_profile, self.control_profile_sha256)
            v34_ablation_fields = {
                'wm_dynamics_mode', 'wm_integration_coeff', 'wm_loss_normalization', 'wm_bryson_scales',
                'wm_activation', 'grouped_physical_loss', 'contact_loss_reduction',
                'wm_reward_loss_weight', 'wm_observation_loss_weight', 'wm_cost_loss_weight',
                'wm_holdout_fraction', 'replay_sampling_mode', 'failure_sample_fraction',
                'deterministic_imagination', 'target_rate_pre_slew_cost',
                'critic_layer_norm', 'critic_value_cap', 'critic_mean_loss',
                'orientation_reward_mode', 'orientation_tolerance_rad',
                'dex_action_residual_rad', 'sequential_goals', 'compact_critic_costs',
                'ema_delta_penalty_weight', 'joint_speed_penalty_weight',
                'cage_penalty_weight', 'cage_safe_radius_m', 'cage_scale_m',
            }
            for name, value in expected.items():
                if name in v34_ablation_fields:
                    continue
                actual = getattr(self, name)
                if isinstance(value, tuple):
                    actual = tuple(tuple(v) if isinstance(v, list) else v for v in actual)
                if actual != value:
                    raise ValueError(f"v33 profile mismatch: {name}")
            if not self.include_object_position or not self.wm_task_losses:
                raise ValueError("v33 requires object position and WM task supervision")
        if type(self.paper_critic) is not bool:
            raise ValueError('paper_critic must be boolean')
        if self.paper_critic and (not self.stable_value_learning or not self.absolute_policy_optimization):
            raise ValueError('Paper Critic requires the current absolute-position risk profile')
        if self.paper_critic and self.critic_td_steps != 1:
            raise ValueError('Paper Critic uses imagined lambda returns, not a real TD window')
        if (not math.isfinite(self.warmup_joint_amplitude_rad) or self.warmup_joint_amplitude_rad < 0
                or type(self.warmup_joint_segment_steps) is not int or self.warmup_joint_segment_steps < 2):
            raise ValueError('Warmup excitation requires finite nonnegative amplitude and at least two segment steps')
        if self.warmup_joint_amplitude_rad and (not self.absolute_policy_optimization or self.wm_only_iters < 2):
            raise ValueError('HOME excitation requires absolute policy optimization and at least two warmup iterations')
        if (not math.isfinite(self.input_joint_velocity_std_floor_rad_s)
                or self.input_joint_velocity_std_floor_rad_s < 0):
            raise ValueError('Joint velocity normalization floor must be finite and nonnegative')
        if self.input_joint_velocity_std_floor_rad_s and not self.absolute_policy_optimization:
            raise ValueError('Physical velocity normalization floor requires the v30 absolute policy protocol')
        if self.input_joint_velocity_std_floor_rad_s and (not math.isfinite(self.joint_velocity_scale)
                                                         or self.joint_velocity_scale <= 0):
            raise ValueError('Physical velocity normalization requires a positive velocity input scale')
        if type(self.critic_td_steps) is not int or not 1 <= self.critic_td_steps <= self.buffer_steps:
            raise ValueError('Critic TD steps must be a positive integer within replay capacity')
        if self.critic_td_steps > 1 and not self.absolute_policy_optimization:
            raise ValueError('Multi-step real Critic requires the v30 absolute policy protocol')
        if type(self.absolute_policy_optimization) is not bool:
            raise ValueError('absolute_policy_optimization must be boolean')
        if self.absolute_policy_optimization:
            if not self.stable_value_learning or self.action_mode != 'direct_limits':
                raise ValueError('Absolute policy optimization requires stable values and direct absolute targets')
            if len(self.action_home) != 21 or len(self.network_joint_limits) != 21:
                raise ValueError('Absolute policy initialization requires HOME and joint limits')
            if len(self.encoder_bias_rad) != 21:
                raise ValueError('Absolute HOME initialization requires 21 encoder offsets')
            for home, bias, (lo, hi) in zip(self.action_home, self.encoder_bias_rad, self.network_joint_limits):
                q = home - bias
                if not math.isfinite(q) or not lo < q < hi:
                    raise ValueError('HOME must lie strictly inside encoder limits')
        for name in ('exploration_position_std_rad', 'exploration_correlation_s',
                     'actor_logit_threshold', 'actor_logit_penalty'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.actor_gradient_audit_interval) is not int or self.actor_gradient_audit_interval < 1:
            raise ValueError('Actor gradient audit interval must be a positive integer')
        if type(self.stable_value_learning) is not bool:
            raise ValueError('stable_value_learning must be boolean')
        if self.stable_value_learning and not (self.wm_task_losses and self.object_hold_costs_enabled):
            raise ValueError('Stable values require learned termination and all five hand-center costs')
        if not math.isfinite(self.critic_target_tau) or not 0 < self.critic_target_tau <= 1:
            raise ValueError('critic_target_tau must be in (0, 1]')
        if type(self.wm_task_losses) is not bool:
            raise ValueError('wm_task_losses must be boolean')
        if self.wm_task_losses and (self.network_revision != 'hardware_center25_v3' or not self.constraints_enabled):
            raise ValueError('WM task supervision requires the hand-center constrained profile')
        for name in ('wm_reward_loss_weight', 'wm_observation_loss_weight',
                     'wm_termination_loss_weight', 'wm_cost_loss_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.wm_dynamics_mode not in ('delta54', 'velocity_integrated'):
            raise ValueError(f'Unknown wm_dynamics_mode: {self.wm_dynamics_mode}')
        if not math.isfinite(self.wm_integration_coeff) or not 0 < self.wm_integration_coeff <= 1:
            raise ValueError('wm_integration_coeff must be in (0, 1]')
        if self.wm_loss_normalization not in ('running_std', 'bryson'):
            raise ValueError(f'Unknown wm_loss_normalization: {self.wm_loss_normalization}')
        if (len(self.wm_bryson_scales) != 6
                or any(not math.isfinite(v) or v <= 0 for v in self.wm_bryson_scales)):
            raise ValueError('wm_bryson_scales must contain 6 finite positive nominal errors')
        if self.wm_activation not in ('silu', 'gelu'):
            raise ValueError(f'Unknown wm_activation: {self.wm_activation}')
        if self.contact_loss_reduction not in ('sum', 'mean'):
            raise ValueError(f'Unknown contact_loss_reduction: {self.contact_loss_reduction}')
        if not math.isfinite(self.wm_holdout_fraction) or not 0 <= self.wm_holdout_fraction < 0.5:
            raise ValueError('wm_holdout_fraction must be in [0, 0.5)')
        if self.replay_sampling_mode not in ('legacy', 'uniform_failure'):
            raise ValueError(f'Unknown replay_sampling_mode: {self.replay_sampling_mode}')
        if not math.isfinite(self.failure_sample_fraction) or not 0 <= self.failure_sample_fraction <= 1:
            raise ValueError('failure_sample_fraction must be in [0, 1]')
        for name in ('deterministic_imagination', 'target_rate_pre_slew_cost',
                     'critic_layer_norm', 'critic_value_cap', 'critic_mean_loss',
                     'sequential_goals', 'compact_critic_costs'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.dex_action_residual_rad:
            if not self.dex_reference:
                raise ValueError('dex_action_residual_rad requires dex_reference')
            if len(self.dex_action_residual_rad) != 21 or any(not math.isfinite(v) or v <= 0 for v in self.dex_action_residual_rad):
                raise ValueError('dex_action_residual_rad must contain 21 finite positive radii')
        if self.compact_critic_costs and not self.object_hold_costs_enabled:
            raise ValueError('compact_critic_costs requires object_hold_costs_enabled')
        for name in ('ema_delta_penalty_weight', 'joint_speed_penalty_weight', 'cage_penalty_weight', 'cage_safe_radius_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.cage_scale_m) or self.cage_scale_m <= 0:
            raise ValueError('cage_scale_m must be finite and positive')
        if self.orientation_reward_mode not in ('gaussian', 'linear_fine'):
            raise ValueError(f'Unknown orientation_reward_mode: {self.orientation_reward_mode}')
        if self.network_revision not in ('legacy', 'urdf_little_v2', 'hardware_center25_v3'):
            raise ValueError('Unknown network revision')
        if self.network_revision != 'legacy':
            if not self.compact_state or self.geometry_enabled or self.compile_geometry:
                raise ValueError('Revised contract requires compact state and no network geometry')
            if self.include_delay_ready and not (self.absolute_policy_optimization
                                                and self.network_revision == 'hardware_center25_v3'):
                raise ValueError('Revised delay-ready input requires the v31 absolute policy protocol')
            if len(self.network_joint_limits) != 21:
                raise ValueError('Revised input scaling requires 21 joint bounds')
            for lo, hi in self.network_joint_limits:
                if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                    raise ValueError('Invalid network scaling bounds')
        if len(self.hand_center_base_m) != 3 or not all(math.isfinite(x) for x in self.hand_center_base_m):
            raise ValueError('Hand center must be three finite base-frame meters')
        from .position_boundary import validate_boundary
        validate_boundary(self)
        if self.constraints_enabled and not self.compact_state:
            raise ValueError('Constrained rollout requires stationary compact state')
        if type(self.object_hold_costs_enabled) is not bool:
            raise ValueError('object_hold_costs_enabled must be boolean')
        if self.object_hold_costs_enabled:
            if not self.constraints_enabled or self.track_object_position:
                raise ValueError('Object holding costs require constrained orientation-only holding')
            if self.stability_reward_scale != 0 and not self.quiet_hold_enabled:
                raise ValueError('Object holding costs replace the stability reward; set its scale to zero')
        for scale in (self.drop_radius_m, self.cost_velocity_limit_rad_s, self.cost_target_rate_limit_rad_s, self.cost_penalty):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Constraint scales must be finite and positive')
        if not math.isfinite(self.cost_object_hold_angle_sigma_rad) or self.cost_object_hold_angle_sigma_rad <= 0:
            raise ValueError('Object holding cost angle width must be finite and positive')
        for scale in (self.input_position_scale, self.input_linear_velocity_scale, self.input_angular_velocity_scale):
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError('Input scales must be finite and positive')
        if self.state_protocol not in ("legacy173", STATE_PROTOCOL, dex.STATE_PROTOCOL):
            raise ValueError("Unknown state protocol")
        if self.compact_state and (self.goal_mode != "single_dense_hold" or self.action_second_diff_weight
                                  or self.action_mode != "direct_limits"
                                  or self.observation_protocol != (dex.OBSERVATION_PROTOCOL if self.dex_reference else DIRECT_OBSERVATION_PROTOCOL)):
            raise ValueError("Compact state requires persistent goal and first-order action cost only")
        if self.encoder_bias_rad and (len(self.encoder_bias_rad) != 21 or not all(math.isfinite(x) for x in self.encoder_bias_rad)):
            raise ValueError("Expected 21 finite encoder offsets")
        if self.control_dt <= 0 or not 0 < self.ema_alpha <= 1:
            raise ValueError("Invalid control period or action filter")
        if self.observation_protocol not in (LEGACY_OBSERVATION_PROTOCOL, DIRECT_OBSERVATION_PROTOCOL, dex.OBSERVATION_PROTOCOL):
            raise ValueError("Unknown observation protocol")
        if self.goal_mode not in ("resample_on_success", "single_dense_hold"):
            raise ValueError("Unknown goal mode")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be strictly between zero and one")
        if self.goal_mode == "single_dense_hold":
            from .quiet_hold import validate_quiet_hold
            validate_quiet_hold(self)
            if self.quiet_hold_enabled and self.uses_target_critic:
                raise ValueError('Quiet holding has signed rewards; use paper_critic instead of bounded target Critic')
            if self.quiet_hold_enabled and (not self.compact_state or self.action_mode != 'direct_limits'
                                            or len(self.network_joint_limits) != 21):
                raise ValueError('Quiet holding requires compact state and 21 explicit action limits')
            validate_orientation_reward(self)
            validate_position_reward(self)
            validate_stability_reward(self)
            if not math.isfinite(self.goal_evaluation_s) or self.goal_evaluation_s <= 0:
                raise ValueError("Goal evaluation time must be finite and positive")
            if self.action_l2_weight or self.hand_pose_weight or self.success_bonus:
                raise ValueError("Single-goal task removes action_l2, hand_pose and success_bonus")
        if not math.isfinite(self.joint_velocity_scale) or self.joint_velocity_scale <= 0:
            raise ValueError("joint_velocity_scale must be finite and positive")
        if self.action_mode not in ("limits", "home_offset", "direct_limits"):
            raise ValueError("Unknown action mapping")
        if self.action_mode == "direct_limits" and not self.dex_reference and (self.ema_alpha != 1 or self.action_warmup_s != 0):
            raise ValueError("Direct targets require no smoothing and no action warmup")
        if self.action_mode == "home_offset":
            if len(self.action_home) != 21 or not all(math.isfinite(x) for x in self.action_home):
                raise ValueError("21 finite home joint angles required")
            if self.action_offset_scale <= 0 or self.action_warmup_s < 0:
                raise ValueError("Invalid home action range or warmup")
        if not math.isfinite(self.hand_pose_weight) or self.hand_pose_weight < 0:
            raise ValueError("hand_pose_weight must be finite and nonnegative")
        if self.hand_pose_weight and (len(self.action_home) != 21 or not all(math.isfinite(x) for x in self.action_home)):
            raise ValueError("Hand-pose penalty requires 21 finite default joint angles")
        required_steps(self.control_dt, self.hold_duration)
        if self.hold_angle_threshold <= 0 or self.hold_speed_threshold <= 0:
            raise ValueError("Hold thresholds must be positive")
        if not 0 < self.actor_lr_scale <= 1:
            raise ValueError("Invalid actor learning-rate multiplier")
        if min(self.joint_vel_weight, self.action_l2_weight, self.action_rate_weight, self.action_second_diff_weight) < 0:
            raise ValueError("Penalty magnitudes must be nonnegative")
        if self.geometry_loss_weight < 0:
            raise ValueError("geometry_loss_weight must be nonnegative")
        if self.event_capacity < 0 or not 0 <= self.event_fraction <= 1 or self.actor_ramp_iters < 0:
            raise ValueError("Invalid event archive or actor ramp configuration")
        if self.amp not in ("none", "bf16"):
            raise ValueError("amp must be none or bf16")
        for name in ("wm_microbatch", "actor_microbatch", "critic_microbatch"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when specified")
        for name in (
            "history",
            "wm_window",
            "horizon",
            "wm_updates",
            "wm_batch",
            "actor_batch",
            "critic_epochs",
            "critic_minibatches",
            "microbatch",
            "buffer_steps",
            "warmup_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.buffer_steps < self.wm_window or self.warmup_steps < self.wm_window:
            raise ValueError("Replay/warmup must cover the WM window")
        if self.warmup_steps < self.critic_td_steps:
            raise ValueError("Replay warmup must cover the Critic TD window")
        if self.actor_batch * self.horizon < self.critic_minibatches or self.utd <= 0:
            raise ValueError("Invalid update scheduling")

    @property
    def compact_state(self):
        return self.state_protocol in (STATE_PROTOCOL, dex.STATE_PROTOCOL)

    @property
    def state_dim(self):
        return REFERENCE.dim if self.dex_reference else COMPACT.dim if self.compact_state else STATE_DIM

    @property
    def critic_state_dim(self):
        return 135 if self.dex_reference else (87 + int(self.include_delay_ready)) if self.compact_state else 177

    @property
    def observation_dim(self):
        if self.dex_reference:
            return 75
        if self.observation_protocol == LEGACY_OBSERVATION_PROTOCOL:
            return 124
        return 78 + 3 * int(self.include_object_position)


def rotation_log(q):
    q = quat_unique(q)
    length = q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., 3:].clamp_min(0))
    # The selected small-angle branch avoids a 0/0 in both forward and backward.
    scale = torch.where(length > 1e-6, angle / length.clamp_min(1e-6), 2 + length.square() / 3)
    return q[..., :3] * scale


def rotation_features(q):
    basis = torch.eye(3, device=q.device, dtype=q.dtype)[:2]
    return quat_apply(q[..., None, :].expand(*q.shape[:-1], 2, 4), basis.expand(*q.shape[:-1], 2, 3)).flatten(-2)


def physical_features(s):
    return torch.cat((s[..., :45], rotation_features(s[..., ROT]), s[..., 49:55]), -1)


def full_features(s):
    if layout(s) == REFERENCE:
        return torch.cat((physical_features(s), s[..., 55:79],
            rotation_features(quat_mul(s[..., ROT], quat_conjugate(s[..., REFERENCE.goal_rot]))), s[..., 83:]), -1)
    if layout(s) == COMPACT:
        return torch.cat((physical_features(s), s[..., 55:79],
                          rotation_features(quat_mul(s[..., ROT], quat_conjugate(s[..., COMPACT.goal_rot]))),
                          s[..., COMPACT.delay_ready]), -1)
    return torch.cat((physical_features(s), s[..., 55:163], rotation_features(s[..., GOAL_ROT]), s[..., 167:]), -1)


def physical_delta(s, nxt):
    return torch.cat(
        (
            nxt[..., :45] - s[..., :45],
            rotation_log(quat_mul(nxt[..., ROT], quat_conjugate(s[..., ROT]))),
            nxt[..., 49:55] - s[..., 49:55],
        ),
        -1,
    )


def integrate_physical(s, delta):
    return torch.cat(
        (
            s[..., :45] + delta[..., :45],
            quat_unique(quat_mul(rotation_vector_quat(delta[..., 45:48]), s[..., ROT])),
            s[..., 49:55] + delta[..., 48:54],
        ),
        -1,
    )


def integrate_velocity_physical(s, delta_vel, dt, c=0.55):
    """Official SuperMimic GenCoord-style velocity-increment integration.

    Predicts 27D velocity increments [dqd(21), dv(3), dw(3)] and derives next
    joint angles, object position, and SO(3) orientation by analytic integration:
        v_next = v + dv
        q_next = q + dt * (v + c * dv)
    where c = (decimation + 1) / (2 * decimation) (= 0.55 for decimation=10).
    Object angular velocity is in hand-base frame, so orientation left-multiplies.
    """
    dqd = delta_vel[..., :21]
    dv = delta_vel[..., 21:24]
    dw = delta_vel[..., 24:27]
    q_next = s[..., Q] + dt * (s[..., QD] + c * dqd)
    qd_next = s[..., QD] + dqd
    pos_next = s[..., POS] + dt * (s[..., VEL] + c * dv)
    rot_step = dt * (s[..., OMEGA] + c * dw)
    rot_next = quat_unique(quat_mul(rotation_vector_quat(rot_step), s[..., ROT]))
    vel_next = s[..., VEL] + dv
    omega_next = s[..., OMEGA] + dw
    return torch.cat((q_next, qd_next, pos_next, rot_next, vel_next, omega_next), -1)


def ema_target(s, action, limits, alpha=0.95):
    lower, upper = limits[..., 0], limits[..., 1]
    target = lower + (action.clamp(-1, 1) + 1) * 0.5 * (upper - lower)
    return (alpha * target + (1 - alpha) * s[..., EMA]).clamp(lower, upper)


def action_target(s, action, limits, cfg, *, home=None, residual_rad=None):
    """Exact real/imagined control contract; targets retain their own EMA memory."""
    if cfg.dex_reference:
        res = residual_rad if residual_rad is not None else (cfg.dex_action_residual_rad or None)
        h = home if home is not None else (cfg.action_home if res is not None else None)
        return dex.target(
            s, action, limits, cfg.ema_alpha, cfg.target_slew_rate_rad_s * cfg.control_dt,
            home=h, residual_rad=res,
        )
    if cfg.action_mode == "direct_limits":
        return absolute_joint_target(action, limits)
    if cfg.action_mode == "limits":
        return ema_target(s, action, limits, cfg.ema_alpha)
    home = action.new_tensor(cfg.action_home)
    lower, upper = limits[..., 0], limits[..., 1]
    raw = (home + cfg.action_offset_scale * action.clamp(-1, 1)).clamp(lower, upper)
    filtered = cfg.ema_alpha * raw + (1 - cfg.ema_alpha) * s[..., EMA]
    # TIME is remaining fraction of the fixed 20-second episode. Round recovers
    # the exact integer counter despite floating-point representation of 0.05.
    ticks = torch.round((1 - s[..., TIME]) * round(20 / cfg.control_dt))
    warmup_steps = math.ceil(cfg.action_warmup_s / cfg.control_dt - 1e-9)
    return torch.where(ticks < warmup_steps, home, filtered).clamp(lower, upper)


def angle_error(s):
    return rotation_log(quat_mul(s[..., ROT], quat_conjugate(s[..., layout(s).goal_rot]))).norm(dim=-1)


def goal_event(s, physical_next, dt, error=None, speed=None, hold_cfg=None, episode_steps=None, blocked=None):
    if layout(s) in (COMPACT, REFERENCE):
        raise ValueError("Compact success counters live in the simulator command tracker, outside learned state")
    if error is None:
        error = rotation_log(quat_mul(physical_next[..., ROT], quat_conjugate(s[..., layout(s).goal_rot]))).norm(dim=-1)
    if speed is None:
        speed = physical_next[..., OMEGA].norm(dim=-1)
    episode_steps = episode_steps or round(20 / dt)
    if hold_cfg is not None and hold_cfg.goal_mode == "single_dense_hold":
        if blocked is None:
            blocked = physical_next[..., POS].norm(dim=-1) > 0.3
        position_ok = position_hold_valid(
            (physical_next[..., POS] - s[..., layout(s).goal_pos]).norm(dim=-1),
            physical_next[..., VEL].norm(dim=-1), hold_cfg)
        blocked = blocked | ~position_ok
        return persistent_hold_transition(s[..., HOLD].squeeze(-1), error, speed, dt,
            already_completed=s[..., COUNT].squeeze(-1) > 0,
            blocked=blocked, **hold_kwargs(hold_cfg))
    if blocked is None:
        blocked = ((s[..., TIME].squeeze(-1) <= 1 / episode_steps + 1e-7)
                   | (physical_next[..., POS].norm(dim=-1) > 0.3)
                   | (s[..., COUNT].squeeze(-1) >= 50))
    return hold_transition(s[..., HOLD].squeeze(-1), error, speed, dt,
                           blocked=blocked, **hold_kwargs(hold_cfg))


def reward(s, physical_next, action, dt=1 / 30, success_bonus=250.0, action_rate_weight=0.01, joint_vel_weight=2.5e-5, action_l2_weight=1e-4, action_second_diff_weight=0.0, *, hold_cfg=None, limits=None, encoder_bias=None, home=None, residual_rad=None):
    error = rotation_log(quat_mul(physical_next[..., ROT], quat_conjugate(s[..., layout(s).goal_rot]))).norm(dim=-1)
    return reward_from_orientation_error(s, physical_next, action, error, dt, success_bonus, action_rate_weight, joint_vel_weight, action_l2_weight, action_second_diff_weight, hold_cfg=hold_cfg, limits=limits, encoder_bias=encoder_bias, home=home, residual_rad=residual_rad)


def reward_from_orientation_error(s, physical_next, action, error, dt=1 / 30, success_bonus=250.0, action_rate_weight=0.01, joint_vel_weight=2.5e-5, action_l2_weight=1e-4, action_second_diff_weight=0.0, *, hold_cfg=None, success_event=None, limits=None, encoder_bias=None, home=None, residual_rad=None):
    if hold_cfg is not None and hold_cfg.goal_mode == "single_dense_hold":
        distance = (physical_next[..., POS] - s[..., layout(s).goal_pos]).norm(dim=-1)
        quiet_penalty = 0.0
        pose_penalty = 0.0
        if hold_cfg.home_pose_weight:
            from .quiet_hold import home_pose_penalty
            home = physical_next.new_tensor(hold_cfg.action_home) if home is None else home
            pose_penalty = home_pose_penalty(physical_next[..., Q], home, hold_cfg)
        if hold_cfg.quiet_hold_enabled:
            from .quiet_hold import imagined_motion_penalty
            quiet_penalty = imagined_motion_penalty(s, physical_next, action, error, hold_cfg,
                limits=limits, encoder_bias=encoder_bias, home=home, residual_rad=residual_rad)
        reg_penalty = 0.0
        if (hold_cfg.ema_delta_penalty_weight or hold_cfg.joint_speed_penalty_weight
                or hold_cfg.cage_penalty_weight):
            from .quiet_hold import imagined_dex_regularization
            reg_penalty = imagined_dex_regularization(
                s, physical_next, action, distance, hold_cfg, limits=limits, home=home, residual_rad=residual_rad)
        return dt * (
            orientation_reward(error, hold_cfg) + position_reward(distance, hold_cfg)
            + stability_reward(error, distance, physical_next[..., OMEGA].norm(dim=-1),
                               physical_next[..., VEL].norm(dim=-1), hold_cfg)
            - quiet_penalty
            - pose_penalty
            - reg_penalty
            - joint_vel_weight * physical_next[..., QD].square().sum(-1)
            - action_rate_weight * (action - s[..., layout(s).last_action]).square().sum(-1)
        )
    if success_event is None:
        _, success_event = goal_event(s, physical_next, dt, error=error, hold_cfg=hold_cfg)
    pose_penalty = 0.0
    if hold_cfg is not None and hold_cfg.hand_pose_weight:
        # Evaluate post-physics, pre-reset actual/predicted joints, not actions or EMA targets.
        home = physical_next.new_tensor(hold_cfg.action_home)
        pose_penalty = hold_cfg.hand_pose_weight * (physical_next[..., Q] - home).square().sum(-1)
    return dt * (
        1 / (error + 0.1)
        - pose_penalty
        + success_bonus * success_event.to(s.dtype)
        - joint_vel_weight * physical_next[..., QD].square().sum(-1)
        - action_l2_weight * action.square().sum(-1)
        - action_rate_weight * (action - s[..., layout(s).last_action]).square().sum(-1)
        - (action_second_diff_weight * (action - 2 * s[..., 76:97] + s[..., 97:118]).square().sum(-1)
               if action_second_diff_weight else 0.)
    )


def sample_xy_goal(reference):
    angles = (2 * torch.rand(*reference.shape[:-1], 2, device=reference.device) - 1) * math.pi
    x = torch.zeros_like(reference[..., :3])
    y = torch.zeros_like(x)
    x[..., 0] = angles[..., 0]
    y[..., 1] = angles[..., 1]
    return quat_mul(rotation_vector_quat(x), rotation_vector_quat(y))


def advance(s, physical_next, action, target, sampled_goal_base, episode_steps=600, *, hold_cfg=None, event_override=None):
    if layout(s) == REFERENCE:
        return torch.cat((physical_next, action, s[..., 76:83], target+s[..., dex.ZERO], s[..., dex.ZERO],
                          dex.sample_wrench(s, hold_cfg)), -1)
    if layout(s) == COMPACT:
        return torch.cat((physical_next, action, s[..., COMPACT.goal_pos], s[..., COMPACT.goal_rot],
                          torch.ones_like(s[..., COMPACT.delay_ready])), -1)
    ticks, reached = goal_event(s, physical_next, 20 / episode_steps, hold_cfg=hold_cfg, episode_steps=episode_steps) if event_override is None else event_override
    goal = torch.where(reached[..., None], sampled_goal_base, s[..., GOAL_ROT])
    history = torch.cat((action, s[..., 76:139]), -1)
    return torch.cat(
        (
            physical_next,
            target,
            history,
            s[..., layout(s).goal_pos],
            goal,
            (s[..., TIME] - 1 / episode_steps).clamp_min(0),
            s[..., COUNT] + reached[..., None],
            s[..., BASE_POS],
            ticks[..., None],
        ),
        -1,
    )


def push_history(history, observation, reset=None):
    updated = torch.cat((history[:, 1:], observation[:, None]), 1)
    if reset is not None:
        # Repeating the first sample introduces no fictitious previous-episode measurements.
        updated = torch.where(reset[:, None, None], observation[:, None].expand_as(updated), updated)
    return updated


def config20(**kwargs):
    """Hold-Penalty2x-v1: double all four Hold-v1 motion penalties.

    Revo3 holdouts show growing physical errors beyond the eight-step WM
    training window. Bootstrap the remaining return from the online Critic;
    keep full closed-loop BPTT through all eight predicted transitions.
    """
    from .wuji_pose import HOME_POSITION, OFFSET_SCALE, WARMUP_SECONDS
    return Config(control_dt=1/20, ema_alpha=.5, action_mode="home_offset",
                  action_home=HOME_POSITION, action_offset_scale=OFFSET_SCALE,
                  action_warmup_s=WARMUP_SECONDS,
                  hand_pose_weight=kwargs.pop("hand_pose_weight", 0.2),
                  success_bonus=250*(20/30), action_rate_weight=.01*(20/30)**2*2.5,
                  joint_vel_weight=2.5e-5*2.5, action_l2_weight=1e-4*2.5,
                  action_second_diff_weight=.01*(20/30)**2*2.5,
                  gamma=.99**1.5, lam=.95**1.5, horizon=8, **kwargs)


def manual_convex_config(**kwargs):
    """81D stationary single-pose Actor; 20s is only a collection cutoff."""
    from dataclasses import replace
    from .identified_hand import load_identified_hand
    parameters, _ = load_identified_hand()
    settings = dict(state_protocol=STATE_PROTOCOL, encoder_bias_rad=tuple(parameters["encoder_bias_rad"]),
                    action_second_diff_weight=0., observation_protocol=DIRECT_OBSERVATION_PROTOCOL,
                    action_mode="direct_limits", ema_alpha=1.0, action_warmup_s=0.0,
                    goal_mode="single_dense_hold", action_l2_weight=0., hand_pose_weight=0.,
                    success_bonus=0., gamma=0.99, position_reward_scale=1.0)
    settings.update(kwargs)
    return replace(config20(), **settings)


def revised_manual_config(**kwargs):
    """Fresh-training profile; not a shape-compatible resume of version 25."""
    from dataclasses import replace
    from .revised_contract import REVISION, reorder, urdf_limits
    base = manual_convex_config()
    limits, _ = urdf_limits()
    settings = dict(network_revision=REVISION, network_joint_limits=tuple(map(tuple, limits.tolist())),
        actor_hidden=(1024,512,256), critic_hidden=(512,256,128), geometry_enabled=False,
        compile_geometry=False, geometry_loss_weight=0., include_delay_ready=False,
        input_position_scale=10., input_linear_velocity_scale=2., input_angular_velocity_scale=.1,
        encoder_bias_rad=tuple(reorder(base.encoder_bias_rad)), action_home=tuple(reorder(base.action_home)))
    settings.update(kwargs)
    return replace(base, **settings)


def hand_center_config(**kwargs):
    """v27: hardware order, 25 Hz, shared hand-center scaling, constrained BPTT."""
    from dataclasses import replace
    from .revised_contract import CENTER_REVISION, HARDWARE_JOINT_NAMES, urdf_limits
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    base = manual_convex_config()
    order = [REVO3_JOINT_NAMES.index(n) for n in HARDWARE_JOINT_NAMES]
    limits, _ = urdf_limits(CENTER_REVISION)
    settings = dict(network_revision=CENTER_REVISION, control_dt=.04, hold_duration=.52,
        track_object_position=False,
        network_joint_limits=tuple(map(tuple, limits.tolist())),
        encoder_bias_rad=tuple(base.encoder_bias_rad[i] for i in order),
        action_home=tuple(base.action_home[i] for i in order),
        hand_center_base_m=(.055,-.010,.100), constraints_enabled=True,
        object_hold_costs_enabled=True, stability_reward_scale=0.,
        joint_vel_weight=0., action_rate_weight=0.)
    settings.update(kwargs)
    return replace(revised_manual_config(), **settings)
