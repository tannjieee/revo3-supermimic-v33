"""Official action-manager term: tanh decoded once to training encoder limits."""

import math
import torch
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
from .policy_contract import absolute_joint_target


class DirectJointPositionAction(JointPositionToLimitsAction):
    def __init__(self, cfg, env):
        if cfg.clip is not None or cfg.scale != 1.0 or cfg.rescale_to_limits:
            raise ValueError('Direct actions prohibit extra clipping/scaling/filtering')
        super().__init__(cfg, env)
        self.encoder_bias = self._raw_actions.new_tensor(cfg.encoder_bias_rad or [0.] * self.action_dim)
        # Diagnostic/reward history only; does not filter or change control.
        self.previous_processed_actions = torch.zeros_like(self._raw_actions)
        self._quiet_target_ready = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=self._raw_actions.device)
        from .sdk_limits import validate_limits
        from .training_limits import load_training_limits
        if cfg.dex_reference_profile is not None:
            from .dex_reference import validate_profile
            expected, source = validate_profile(cfg.dex_reference_profile)
            if list(cfg.joint_names) != source['joint_names']:
                raise ValueError('v33 hardware joint order changed')
            limits = validate_limits(cfg.position_limits_rad)
            if not (limits == expected).all():
                raise ValueError('v33 action limits differ from YAML snapshot')
            self.position_limits = self._raw_actions.new_tensor(limits)[None].expand(env.num_envs, -1, -1)
            self.zero_bias = torch.zeros_like(self._raw_actions)
            self.ema_encoder = torch.zeros_like(self._raw_actions)
            self.target_max_delta = math.radians(cfg.dex_reference_profile.get('target_slew_rate_deg_s', 0.)) * env.step_dt
            self.reset()
            return
        if cfg.limit_contract in ('urdf_little_v2', 'hardware_center25_v3'):
            from .revised_contract import urdf_limits, JOINT_NAMES
            expected, source = urdf_limits(cfg.limit_contract)
            if list(cfg.joint_names) != source['joint_names']:
                raise ValueError('URDF action joint order changed')
        elif cfg.limit_contract == 'legacy':
            expected, source = load_training_limits()
        else:
            raise ValueError('Unknown limit contract')
        limits = validate_limits(cfg.position_limits_rad)
        if (cfg.software_limits_sha256 != source['sdk_software_limits']['sha256']
                or cfg.training_limits_sha256 != source['sha256'] or (limits != expected).any()):
            raise ValueError('Action limits differ from the independent training configuration')
        self.position_limits = self._raw_actions.new_tensor(limits)[None].expand(env.num_envs, -1, -1)

    def process_actions(self, actions):
        if self.cfg.dex_reference_profile is not None:
            initial = self.ema_encoder - self.zero_bias
        else:
            measured = self._asset.data.joint_pos
            measured = measured if isinstance(measured, torch.Tensor) else measured.torch
            initial = measured[:, self._joint_ids] - self.encoder_bias
        self.previous_processed_actions.copy_(torch.where(
            self._quiet_target_ready, self._processed_actions, initial))
        self._raw_actions.copy_(actions)
        raw = absolute_joint_target(actions, self.position_limits)
        if self.cfg.dex_reference_profile is not None:
            from .dex_reference import filter_target
            self.ema_encoder = filter_target(self.ema_encoder, raw, self.cfg.alpha, getattr(self, 'target_max_delta', 0.))
            raw = self.ema_encoder-self.zero_bias
        self._processed_actions.copy_(raw)
        self._quiet_target_ready.fill_(True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, '_quiet_target_ready'):
            ids = slice(None) if env_ids is None else env_ids
            self._quiet_target_ready[ids] = False
            self.previous_processed_actions[ids] = 0.
        if self.cfg.dex_reference_profile is not None and hasattr(self, "zero_bias"):
            ids = slice(None) if env_ids is None else env_ids
            lo, hi = [math.radians(v) for v in self.cfg.dex_reference_profile["zero_bias_deg"]]
            self.zero_bias[ids] = lo+(hi-lo)*torch.rand_like(self.zero_bias[ids])
            home = self._asset.data.default_joint_pos
            home = home if isinstance(home, torch.Tensor) else home.torch
            self.ema_encoder[ids] = home[ids][:, self._joint_ids]+self.zero_bias[ids]
            self._processed_actions[ids] = home[ids][:, self._joint_ids]

    def apply_actions(self):
        # PACE encoder = geometric angle - fixed bias. This translates coordinates;
        # the policy output and returned absolute encoder command stay untouched.
        self._asset.set_joint_position_target_index(
            target=self.processed_actions + self.encoder_bias, joint_ids=self._joint_ids)


@configclass
class DirectJointPositionActionCfg(JointPositionToLimitsActionCfg):
    class_type: type = DirectJointPositionAction
    scale: float = 1.0
    rescale_to_limits: bool = False

    dex_reference_profile: dict | None = None
    alpha: float = 1.0
    encoder_bias_rad: tuple = ()
    position_limits_rad: tuple = ()
    software_limits_sha256: str = ''
    training_limits_sha256: str = ''
    limit_contract: str = 'legacy'
