"""Offline/deployment inference for versions 25 through 33; no hardware connection.

v25-v32 return encoder targets unchanged. v33 takes measured encoder angles,
returns geometric targets after EMA and the caller-provided zero offset, and
advances EMA memory only on commit. Reset initializes it from the first reading.
The velocity argument is retained for API compatibility; v33 Actor ignores it.
"""

from pathlib import Path
import torch
from .core import Config, Q, QD, POS, ROT, push_history
from .state_layout import COMPACT, STATE_PROTOCOL
from .networks import Actor
from .policy_contract import DIRECT_OBSERVATION_PROTOCOL, absolute_joint_target
from .sensors import initial_sensors, observe
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES


def validate_payload(payload):
    if payload.get('version') not in (25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
        raise ValueError('A version-25 or later direct pose-goal Actor (supported: v25-v34) is required')
    cfg = Config(**payload['config'])
    from .trainer import checkpoint_version
    from .revised_contract import joint_names
    if payload['version'] != checkpoint_version(cfg):
        raise ValueError('Network revision does not match checkpoint version')
    meta = payload['metadata']
    from . import dex_reference as dex
    if (cfg.state_protocol != (dex.STATE_PROTOCOL if cfg.dex_reference else STATE_PROTOCOL)
            or cfg.observation_protocol != (dex.OBSERVATION_PROTOCOL if cfg.dex_reference else DIRECT_OBSERVATION_PROTOCOL)
            or cfg.goal_mode != 'single_dense_hold'
            or cfg.action_mode != 'direct_limits' or (not cfg.dex_reference and cfg.ema_alpha != 1) or cfg.action_warmup_s != 0):
        raise ValueError('A version-25 or later direct pose-goal Actor (supported: v25-v33) is required')
    if (meta.get('state_protocol') != cfg.state_protocol
            or meta.get('observation_protocol') != cfg.observation_protocol
            or meta.get('observation_dim') != cfg.observation_dim
            or meta.get('quaternion_order') != 'XYZW'
            or meta.get('joint_names') != list(joint_names(cfg))):
        raise ValueError('Checkpoint observation/joint/quaternion contract mismatch')
    limits = torch.tensor(meta['joint_limits_rad'], dtype=torch.float32)
    if limits.shape != (21, 2) or not torch.isfinite(limits).all() or not (limits[:, 1] > limits[:, 0]).all():
        raise ValueError('Checkpoint has invalid asset joint limits')
    if cfg.network_revision != 'legacy':
        torch.testing.assert_close(limits, limits.new_tensor(cfg.network_joint_limits), atol=2e-7, rtol=0)
    return cfg, limits


def export_actor(checkpoint, output):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg, _ = validate_payload(payload)
    Actor(cfg).load_state_dict(payload['actor'], strict=True)
    exported = {key: payload[key] for key in ('version', 'iteration', 'config', 'noise_config', 'actor', 'metadata')}
    exported['metadata'] = dict(exported['metadata'], actor_only_export=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)


class ManualConvexPolicy:
    def __init__(self, checkpoint, device='cpu'):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.cfg, limits = validate_payload(payload)
        self.metadata = payload['metadata']
        self.device = torch.device(device)
        self.limits = limits.to(self.device)[None]
        self.actor = Actor(self.cfg).to(self.device).eval()
        self.actor.load_state_dict(payload['actor'], strict=True)
        self.actor.amp = 'none'
        self.noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
        self.reset()

    def reset(self, zero_bias_rad=None):
        self.state = torch.zeros(1, self.cfg.state_dim, device=self.device)
        if self.cfg.dex_reference:
            from .dex_reference import ZERO
            if zero_bias_rad is not None:
                self.state[:, ZERO] = self._vector(zero_bias_rad, 21)
        self.sensors = initial_sensors(1, self.device, self.noise)
        self.history = None
        self.has_goal = False
        self.pending = False

    def _vector(self, value, n):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.shape != (n,) or not torch.isfinite(value).all():
            raise ValueError(f'Expected {n} finite values')
        return value

    def set_goal(self, goal_quaternion_xyzw, goal_position_base_m, *, quaternion_order='XYZW'):
        """Set one base-frame pose command, persistent until explicitly replaced."""
        if quaternion_order != 'XYZW':
            raise ValueError('Convert WXYZ to XYZW explicitly at the external API boundary')
        if self.pending:
            raise RuntimeError('Previous action has not been committed')
        q = self._vector(goal_quaternion_xyzw, 4)
        if abs(float(q.norm()) - 1) > 1e-4:
            raise ValueError('Expected a unit XYZW quaternion')
        p = self._vector(goal_position_base_m, 3)
        self.state[:, COMPACT.goal_rot] = q
        self.state[:, COMPACT.goal_pos] = p
        self.has_goal = True

    @torch.inference_mode()
    def propose(self, joint_position_rad, joint_velocity_rad_s, object_position_base_m,
                object_quaternion_xyzw, goal_quaternion_xyzw=None, *, goal_position_base_m=None,
                quaternion_order='XYZW'):
        if quaternion_order != 'XYZW':
            raise ValueError('Convert WXYZ to XYZW explicitly at the external API boundary')
        if self.pending:
            raise RuntimeError('Previous action has not been committed')
        if goal_quaternion_xyzw is not None or goal_position_base_m is not None:
            if goal_quaternion_xyzw is None or goal_position_base_m is None:
                raise ValueError('Supply both goal orientation and goal position for a pose command')
            q = self._vector(goal_quaternion_xyzw, 4)
            p = self._vector(goal_position_base_m, 3)
            if abs(float(q.norm()) - 1) > 1e-4:
                raise ValueError('Expected a unit XYZW quaternion')
            old = self.state[0, COMPACT.goal_rot]
            same = torch.allclose(q, old, atol=1e-6, rtol=0) or torch.allclose(-q, old, atol=1e-6, rtol=0)
            same = same and torch.allclose(p, self.state[0, COMPACT.goal_pos], atol=1e-6, rtol=0)
            if not self.has_goal or not same:
                self.set_goal(q, p)
        if not self.has_goal:
            raise ValueError('Set one goal before requesting an action')
        for key, values, n in ((Q, joint_position_rad, 21), (QD, joint_velocity_rad_s, 21),
                               (POS, object_position_base_m, 3), (ROT, object_quaternion_xyzw, 4)):
            value = self._vector(values, n)
            if n == 4 and abs(float(value.norm()) - 1) > 1e-4:
                raise ValueError('Expected a unit XYZW quaternion')
            self.state[:, key] = value
        if self.cfg.dex_reference:
            from .dex_reference import ZERO, EMA
            if self.history is None:
                self.state[:, EMA] = self.state[:, Q]
            self.state[:, Q] -= self.state[:, ZERO]
        else:
            self.state[:, Q] += self.state.new_tensor(self.cfg.encoder_bias_rad)
        obs, self.sensors = observe(self.state, self.sensors, self.limits, self.noise, self.cfg)
        self.history = (obs[:, None].expand(-1, self.cfg.history, -1).clone() if self.history is None
                        else push_history(self.history, obs))
        self.raw = self.actor(self.history)
        if self.cfg.dex_reference:
            from .core import action_target
            target = action_target(self.state, self.raw, self.limits, self.cfg)
            self.proposed_target = target
        else:
            target = absolute_joint_target(self.raw, self.limits)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite Actor target')
        self.pending = True
        return target[0].cpu().numpy(), self.raw[0].cpu().numpy(), obs[0].cpu().numpy()

    def commit(self):
        if not self.pending:
            raise RuntimeError('No proposed action')
        self.state[:, COMPACT.last_action] = self.raw
        if self.cfg.dex_reference:
            from .dex_reference import EMA, ZERO
            self.state[:, EMA] = self.proposed_target+self.state[:, ZERO]
        else:
            self.state[:, COMPACT.delay_ready] = 1.
        self.pending = False
