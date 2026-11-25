"""Read task goal changes through a Gym wrapper without changing simulation state."""

import json
from pathlib import Path

import gymnasium as gym
import torch


class GoalCountingWrapper(gym.Wrapper):
    """Count earned goal resamples, excluding the first goal assigned on resets."""

    def __init__(self, env, directory):
        super().__init__(env)
        assert self.unwrapped.num_envs == 1
        self.command = self.unwrapped.command_manager.get_term('object_pose')
        assert self.command.cfg.update_goal_on_success
        assert min(self.command.cfg.resampling_time_range) > 1000
        self.directory = Path(directory)
        self.teacher_iteration = json.loads((self.directory / 'preview_selection.json').read_text())['iteration']
        self.stream = (self.directory / 'success_steps.jsonl').open('x')
        self.previous_counter = int(self.command.command_counter[0])
        self.steps = 0
        self.successes = 0
        self.episode_successes = 0
        self.events = []
        self.episodes = []
        self.termination_counts = {}
        self._counting_closed = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.previous_counter = int(self.command.command_counter[0])
        self.episode_successes = 0
        return result

    def step(self, action):
        old_goal = self.command.command[0, 3:7].detach().clone()
        result = self.env.step(action)
        _, _, terminated, truncated, _ = result
        done = bool((terminated | truncated)[0])
        counter = int(self.command.command_counter[0])
        gained = 0 if done else counter - self.previous_counter
        assert gained in (0, 1), (counter, self.previous_counter, done)
        new_goal = self.command.command[0, 3:7]
        goal_changed = not torch.equal(old_goal, new_goal)
        error = float(self.command.metrics['orientation_error'][0])
        if gained:
            assert goal_changed and error < self.command.cfg.orientation_success_threshold
        if not done:
            assert goal_changed == bool(gained)
        else:
            # ReorientCommand suppresses success resampling on the auto-reset step.
            assert counter == 1
        self.steps += 1
        self.successes += gained
        self.episode_successes += gained
        row = {
            'step': self.steps, 'video_frame': self.steps - 1,
            'time_s': self.steps * self.unwrapped.step_dt,
            'new_successes': gained, 'successes_total': self.successes,
            'episode_successes': self.episode_successes,
            'command_counter': counter, 'done': done,
            'orientation_error_before_goal_switch_rad': error,
            'old_goal_xyzw': old_goal.cpu().tolist(),
            'new_goal_xyzw': new_goal.cpu().tolist(),
            'object_quaternion_xyzw': self.unwrapped.scene['object'].data.root_quat_w.torch[0].cpu().tolist(),
        }
        if gained:
            self.events.append(row.copy())
        if done:
            self.episodes.append({'end_step': self.steps, 'successes': self.episode_successes})
            self.episode_successes = 0
        for name in self.unwrapped.termination_manager.active_terms:
            if bool(self.unwrapped.termination_manager.get_term(name)[0]):
                self.termination_counts[name] = self.termination_counts.get(name, 0) + 1
        self.previous_counter = counter
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        if self.steps % 30 == 0:
            self.save_summary()
        return result

    def save_summary(self):
        summary = {
            'steps': self.steps, 'duration_s': self.steps * self.unwrapped.step_dt,
            'num_envs': 1, 'policy_hz': 1 / self.unwrapped.step_dt,
            'successes_total': self.successes,
            'success_threshold_rad': self.command.cfg.orientation_success_threshold,
            'definition': 'Earned goal resamples from the public command_counter; initial/reset goals excluded.',
            'success_events': self.events,
            'finished_episodes': self.episodes,
            'current_episode_successes': self.episode_successes,
            'termination_counts': self.termination_counts,
            'teacher_iteration': self.teacher_iteration,
            'is_teacher_state_policy': True,
            'is_rgb_closed_loop': False,
        }
        target = self.directory / 'success_summary.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(target)
        return summary

    def close(self):
        if not self._counting_closed:
            self.save_summary()
            self.stream.close()
            self._counting_closed = True
        return self.env.close()
