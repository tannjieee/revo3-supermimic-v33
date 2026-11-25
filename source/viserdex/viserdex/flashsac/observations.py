"""One sensor advancement per transition, including pre-autoreset actor history."""
import torch
from torch import nn
from viserdex.supermimic.core import full_features, push_history
from viserdex.supermimic.networks import input_affine
from viserdex.supermimic.sensors import observe, reset_sensors


def transition_history(history, sensors, terminal_state, reset_state, done, limits, noise, cfg):
    terminal_obs, advanced = observe(terminal_state, sensors, limits, noise, cfg)
    terminal_history = push_history(history, terminal_obs)
    behavior_obs = terminal_obs.clone()
    behavior_sensors = reset_sensors(advanced, done, noise)
    if done.any():
        ids = done.nonzero().flatten()
        reset_obs, reset_values = observe(reset_state[ids],
            {k: v[ids] for k, v in behavior_sensors.items()}, limits[ids], noise, cfg)
        behavior_obs[ids] = reset_obs
        for key, value in reset_values.items():
            behavior_sensors[key][ids] = value
    behavior_history = push_history(history, behavior_obs, done)
    return terminal_history, behavior_history, behavior_obs, behavior_sensors


class ObservationEncoder(nn.Module):
    actor_dim = 810
    observation_dim = 963

    def __init__(self, cfg, context_scale):
        super().__init__()
        if cfg.history != 10 or cfg.observation_dim != 81 or not cfg.compact_state:
            raise ValueError('FlashSAC v1 requires compact state and 10 x 81 actor history')
        scale, offset = input_affine(cfg, 81, actor=True)
        self.register_buffer('history_scale', scale)
        self.register_buffer('history_offset', offset)
        # full_features always includes ready; do not duplicate it.
        scale, offset = input_affine(cfg, 153, critic=True)
        self.register_buffer('state_scale', scale)
        self.register_buffer('state_offset', offset)
        self.register_buffer('context_scale', context_scale.clone())

    def actor(self, history):
        return (history * self.history_scale + self.history_offset).flatten(1)

    def forward(self, history, state, context):
        x = torch.cat((full_features(state), context / self.context_scale), -1)
        if x.shape[-1] != 153:
            raise ValueError(f'Privileged schema changed: {x.shape}')
        result = torch.cat((self.actor(history), x*self.state_scale+self.state_offset), -1)
        if result.shape[-1] != self.observation_dim or not torch.isfinite(result).all():
            raise ValueError('Invalid FlashSAC observation')
        return result
