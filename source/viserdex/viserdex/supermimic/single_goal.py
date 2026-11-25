"""Single object pose command: stationary dense reward and indefinite holding."""
import math
import torch
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_mul
from .hold_command import HoldReorientCommand, HoldReorientCommandCfg
from .goal_hold import hold_transition, persistent_hold_transition
from .core import sample_xy_goal
from .orientation_reward import orientation_reward, validate_orientation_reward
from .position_reward import position_reward, position_hold_valid, validate_position_reward
from .stability_reward import stability_reward, validate_stability_reward
from .quiet_hold import validate_quiet_hold, motion_penalty, dex_regularization_penalty


class SingleGoalCommand(HoldReorientCommand):
    def __init__(self, cfg, env):
        validate_orientation_reward(cfg)
        validate_position_reward(cfg)
        validate_stability_reward(cfg)
        validate_quiet_hold(cfg)
        super().__init__(cfg, env)
        self.goal_start_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.robot = env.scene['robot']
        if cfg.goal_position_base_m is not None:
            p = torch.as_tensor(cfg.goal_position_base_m, device=self.device, dtype=torch.float32)
            if p.shape != (3,) or not torch.isfinite(p).all():
                raise ValueError('goal_position_base_m must contain three finite meters')
            self.pos_command_w.copy_(self.robot.data.root_pos_w.torch
                + quat_apply(self.robot.data.root_quat_w.torch, p.expand(self.num_envs, -1)))
            self.pos_command_e.copy_(self.pos_command_w - env.scene.env_origins)
        self.default_pos_command_e = self.pos_command_e.clone()
        self.last_position_error = torch.zeros(self.num_envs, device=self.device)
        self.last_linear_speed = torch.zeros(self.num_envs, device=self.device)

    @property
    def elapsed_seconds(self):
        # Evaluation metadata only; never consumed by reward, state or Actor.
        return (self._env.common_step_counter - self.goal_start_step) * self._env.step_dt

    def evaluate_step(self):
        if self._evaluated_step == self._env.common_step_counter:
            return self.just_completed
        self._evaluated_step = self._env.common_step_counter
        self.last_angle.copy_(quat_error_magnitude(self.object.data.root_quat_w.torch, self.quat_command_w))
        self.last_speed.copy_(self.object.data.root_ang_vel_w.torch.norm(dim=-1))
        self.last_position_error.copy_((self.object.data.root_pos_w.torch - self.pos_command_w).norm(dim=-1))
        self.last_linear_speed.copy_(self.object.data.root_lin_vel_w.torch.norm(dim=-1))
        blocked = self._env.reset_buf | ~position_hold_valid(
            self.last_position_error, self.last_linear_speed, self.cfg)
        if getattr(self.cfg, 'sequential_goals', False):
            ticks, event = hold_transition(
                self.hold_ticks, self.last_angle, self.last_speed, self._env.step_dt,
                angle_threshold=self.cfg.orientation_success_threshold,
                speed_threshold=self.cfg.hold_speed_threshold, duration=self.cfg.hold_duration,
                blocked=blocked,
            )
        else:
            ticks, event = persistent_hold_transition(
                self.hold_ticks, self.last_angle, self.last_speed, self._env.step_dt,
                already_completed=self.metrics['consecutive_success'] > 0,
                angle_threshold=self.cfg.orientation_success_threshold,
                speed_threshold=self.cfg.hold_speed_threshold, duration=self.cfg.hold_duration,
                blocked=blocked,
            )
        self.hold_ticks.copy_(ticks)
        self.just_completed.copy_(event)
        self.metrics['consecutive_success'].add_(event)
        return event

    def compute(self, dt):
        # Official task-local command lifecycle. Sequential goals resample immediately
        # after each completed hold; legacy single-goal keeps one goal per episode.
        self._update_metrics()
        if getattr(self.cfg, 'sequential_goals', False):
            self._update_command()

    def _update_command(self):
        if not getattr(self.cfg, 'sequential_goals', False):
            return
        earned = self.just_completed & ~self._env.reset_buf
        ids = earned.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self._resample_command(ids, preserve_success=True)

    def _resample_command(self, env_ids, *, preserve_success=False):
        self.quat_command_w[env_ids] = sample_xy_goal(self.quat_command_w[env_ids])
        self.pos_command_e[env_ids] = self.default_pos_command_e[env_ids]
        self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
        self._restart_goal(env_ids, preserve_success=preserve_success)

    def _restart_goal(self, env_ids, *, preserve_success=False):
        self.goal_start_step[env_ids] = self._env.common_step_counter
        self.hold_ticks[env_ids] = 0
        self.just_completed[env_ids] = False
        if not preserve_success:
            self.metrics['consecutive_success'][env_ids] = 0

    def set_goal(self, env_ids, quaternion_xyzw, position_base_m):
        """Explicit hand-base-frame pose; reissuing it restarts evaluation statistics only."""
        q = torch.as_tensor(quaternion_xyzw, device=self.device, dtype=torch.float32)
        p = torch.as_tensor(position_base_m, device=self.device, dtype=torch.float32)
        if q.shape != self.quat_command_w[env_ids].shape or not torch.isfinite(q).all():
            raise ValueError('Expected one finite XYZW quaternion per selected environment')
        if not torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-4, rtol=0):
            raise ValueError('Expected unit XYZW quaternions')
        if p.shape != self.pos_command_w[env_ids].shape or not torch.isfinite(p).all():
            raise ValueError('Expected one finite base-frame XYZ position per selected environment')
        base_q = self.robot.data.root_quat_w.torch[env_ids]
        p_w = self.robot.data.root_pos_w.torch[env_ids] + quat_apply(base_q, p)
        self.quat_command_w[env_ids] = quat_mul(base_q, q)
        self.pos_command_w[env_ids] = p_w
        self.pos_command_e[env_ids] = p_w - self._env.scene.env_origins[env_ids]
        self._restart_goal(env_ids)
        adapter = getattr(self._env, 'supermimic_adapter', None)
        if adapter is not None:
            adapter.refresh_task_goal(env_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # The parent tracker assumes each reached goal immediately creates a
        # replacement. Here each episode has exactly one command instead.
        self.metrics['success_rate'][ids] = (self.metrics['consecutive_success'][ids] > 0).float()
        result = CommandTerm.reset(self, ids)
        self._env.extras.setdefault('log', {})['Metrics/success_rate'] = result.pop('success_rate')
        return result


@configclass
class SingleGoalCommandCfg(HoldReorientCommandCfg):
    class_type: type = SingleGoalCommand
    update_goal_on_success: bool = False
    sequential_goals: bool = False
    goal_evaluation_s: float = 10.0
    track_object_position: bool = True  # Preserve saved environments predating orientation-only holding.
    stability_reward_scale: float = 3.0
    stability_angle_sigma_rad: float = math.pi / 12
    stability_position_sigma_m: float = 0.02
    stability_angular_speed_scale: float = 0.5
    stability_linear_speed_scale: float = 0.05
    quiet_hold_enabled: bool = False
    orientation_tolerance_rad: float = 0.0
    hold_joint_motion_weight: float = 0.0
    hold_target_motion_weight: float = 0.0
    hold_joint_speed_scale_rad_s: float = math.pi / 9
    hold_target_speed_scale_rad_s: float = math.pi / 9
    home_pose_weight: float = 0.0
    home_pose_deadband_rad: float = math.pi / 12
    home_pose_scale_rad: float = math.pi / 6
    ema_delta_penalty_weight: float = 0.0
    joint_speed_penalty_weight: float = 0.0
    cage_penalty_weight: float = 0.0
    cage_safe_radius_m: float = 0.045
    cage_scale_m: float = 0.025
    orientation_reward_mode: str = "gaussian"
    orientation_reward_scale: float = 10.0
    orientation_coarse_sigma_rad: float = math.pi / 2
    orientation_fine_sigma_rad: float = math.pi / 18
    orientation_coarse_fraction: float = 0.3
    goal_position_base_m: tuple[float, float, float] | None = None
    position_reward_scale: float = 1.0
    position_tolerance_m: float = 0.01
    position_sigma_m: float = 0.05
    hold_position_threshold_m: float = 0.02
    hold_linear_speed_threshold_m_s: float = 0.05


def dense_orientation_reward(env, command_name='object_pose'):
    command = env.command_manager.get_term(command_name)
    command.evaluate_step()
    return orientation_reward(command.last_angle, command.cfg)


def dense_position_reward(env, command_name='object_pose'):
    command = env.command_manager.get_term(command_name)
    command.evaluate_step()
    return position_reward(command.last_position_error, command.cfg)


def dense_stability_reward(env, command_name='object_pose'):
    command = env.command_manager.get_term(command_name)
    command.evaluate_step()
    value = stability_reward(command.last_angle, command.last_position_error,
                             command.last_speed, command.last_linear_speed, command.cfg)
    if getattr(command.cfg, 'quiet_hold_enabled', False):
        action = env.action_manager.get_term('joint_pos')
        velocity = action._asset.data.joint_vel
        velocity = velocity if isinstance(velocity, torch.Tensor) else velocity.torch
        value = value - motion_penalty(command.last_angle, velocity[:, action._joint_ids],
            action.processed_actions, action.previous_processed_actions, env.step_dt, command.cfg)
    if getattr(command.cfg, 'home_pose_weight', 0.):
        from .quiet_hold import home_pose_penalty
        action = env.action_manager.get_term('joint_pos')
        data = action._asset.data
        joints, home = data.joint_pos, data.default_joint_pos
        joints = joints if isinstance(joints, torch.Tensor) else joints.torch
        home = home if isinstance(home, torch.Tensor) else home.torch
        value = value - home_pose_penalty(joints[:, action._joint_ids], home[:, action._joint_ids], command.cfg)
    if (getattr(command.cfg, 'ema_delta_penalty_weight', 0.)
            or getattr(command.cfg, 'joint_speed_penalty_weight', 0.)
            or getattr(command.cfg, 'cage_penalty_weight', 0.)):
        action = env.action_manager.get_term('joint_pos')
        velocity = action._asset.data.joint_vel
        velocity = velocity if isinstance(velocity, torch.Tensor) else velocity.torch
        unclipped = getattr(action, 'unclipped_processed_actions', action.processed_actions)
        value = value - dex_regularization_penalty(
            command.last_position_error, velocity[:, action._joint_ids],
            unclipped, action.previous_processed_actions, env.step_dt, command.cfg)
    return value
