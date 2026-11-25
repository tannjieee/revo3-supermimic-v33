"""Task-local hold command using the official manager lifecycle; no engine patches."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from viserdex.tasks.manipulation.inhand.mdp.commands import ViserDexReorientCommand, ViserDexReorientCommandCfg
from .goal_hold import hold_transition, required_steps


class HoldReorientCommand(ViserDexReorientCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        required_steps(env.step_dt, cfg.hold_duration)
        self.hold_ticks = torch.zeros(self.num_envs, device=self.device)
        self.just_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_angle = torch.zeros(self.num_envs, device=self.device)
        self.last_speed = torch.zeros(self.num_envs, device=self.device)
        self._evaluated_step = -1

    def evaluate_step(self):
        # Reward callbacks and diagnostics may read this repeatedly: count once.
        if self._evaluated_step == self._env.common_step_counter:
            return self.just_completed
        self._evaluated_step = self._env.common_step_counter
        self.last_angle.copy_(quat_error_magnitude(self.object.data.root_quat_w.torch, self.quat_command_w))
        self.last_speed.copy_(self.object.data.root_ang_vel_w.torch.norm(dim=-1))
        ticks, event = hold_transition(self.hold_ticks, self.last_angle, self.last_speed,
            self._env.step_dt, angle_threshold=self.cfg.orientation_success_threshold,
            speed_threshold=self.cfg.hold_speed_threshold, duration=self.cfg.hold_duration,
            blocked=self._env.reset_buf)
        self.hold_ticks.copy_(ticks)
        self.just_completed.copy_(event)
        self.metrics['consecutive_success'].add_(event)
        return self.just_completed

    def _update_metrics(self):
        # The parent counts instantaneous angular crossings; do not call it here.
        self.metrics['orientation_error'] = quat_error_magnitude(self.object.data.root_quat_w.torch, self.quat_command_w)
        self.metrics['position_error'] = (self.object.data.root_pos_w.torch - self.pos_command_w).norm(dim=-1)

    def _update_command(self):
        # evaluate_step ran before reset. Resets clear their pulse, so freshly
        # randomized poses cannot earn a sample or an unearned goal here.
        earned = self._success.earned(self.just_completed)
        self._resample(earned.nonzero(as_tuple=False).squeeze(-1))

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.hold_ticks[env_ids] = 0
        self.just_completed[env_ids] = False

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.hold_ticks[ids] = 0
        self.just_completed[ids] = False
        return result


@configclass
class HoldReorientCommandCfg(ViserDexReorientCommandCfg):
    class_type: type = HoldReorientCommand
    hold_speed_threshold: float = 0.5
    hold_duration: float = 0.5


def hold_success_bonus(env, command_name='object_pose'):
    return env.command_manager.get_term(command_name).evaluate_step().float()
