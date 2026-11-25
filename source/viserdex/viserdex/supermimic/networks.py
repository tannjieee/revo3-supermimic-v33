"""Three networks; normalizers are updated explicitly from real simulation only."""

import torch
from torch import nn
from .core import physical_features, full_features, physical_delta, integrate_physical
from .state_layout import COMPACT
from .geometry import HandObjectGeometry


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-2, std_floor=None):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("variance", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        if std_floor is not None:
            floor = torch.as_tensor(std_floor, dtype=torch.float32)
            if floor.shape != (dim,) or not torch.isfinite(floor).all() or (floor < 0).any():
                raise ValueError('Normalization floor must be a finite nonnegative vector')
            self.register_buffer('std_floor', floor.clone())

    @torch.no_grad()
    def update(self, x, distributed=None):
        if self.count >= 2e8:
            return
        x = x.reshape(-1, x.shape[-1])
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite real normalization sample")
        n = len(x)
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        if distributed is not None:
            n, mean, var = distributed.moments(n, mean, var)
        total = self.count + n
        delta = mean - self.mean
        self.variance.copy_((self.variance * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * (n / total))
        self.count.copy_(total)

    @property
    def std(self):
        std = (self.variance + self.eps**2).sqrt()
        return torch.maximum(std, self.std_floor) if hasattr(self, 'std_floor') else std

    def forward(self, x):
        return (x - self.mean) / self.std


def input_normalizer(cfg, dim):
    """Same qdot units/floor in Actor, WM and Critic; never clips observations."""
    floor = None
    if cfg.input_joint_velocity_std_floor_rad_s:
        floor = torch.zeros(dim)
        floor[21:42] = cfg.joint_velocity_scale * cfg.input_joint_velocity_std_floor_rad_s
    return RunningNorm(dim, std_floor=floor)


def mlp(input_dim, hidden, output_dim, layer_norm=False):
    layers = []
    for width in hidden:
        layers.append(nn.Linear(input_dim, width))
        if layer_norm:
            layers.extend((nn.LayerNorm(width), nn.SiLU()))
        else:
            layers.append(nn.ELU())
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def run_mlp(net, x, amp):
    if amp == "none":
        return net(x)
    # Match the reference: autocast only the MLP, return to FP32 before
    # physical integration, rotations, rewards, or observation processing.
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        return net(x).float()


def input_affine(cfg, width, *, actor=False, critic=False):
    scale, offset = torch.ones(width), torch.zeros(width)
    def position(start):
        scale[start:start+3] = cfg.input_position_scale
        offset[start:start+3] = -torch.tensor(cfg.hand_center_base_m) * cfg.input_position_scale
    if actor:
        start = 63
        if cfg.include_object_position:
            position(start)
            start += 3
        position(start)
        return scale, offset
    limits = torch.tensor(cfg.network_joint_limits)
    half = (limits[:,1]-limits[:,0])*.5
    center = limits.mean(-1)
    bias = torch.tensor(cfg.encoder_bias_rad)
    scale[:21] = half.reciprocal()
    offset[:21] = -(center+bias)/half
    scale[21:42] = cfg.joint_velocity_scale
    position(42)
    scale[51:54] = cfg.input_linear_velocity_scale
    scale[54:57] = cfg.input_angular_velocity_scale
    if critic:
        position(78)
    else:
        scale[57:78] = half.reciprocal()
        offset[57:78] = -center/half
    return scale, offset


class Dynamics(nn.Module):
    def __init__(self, context_dim, cfg, geometry_spec=None):
        super().__init__()
        self.amp = cfg.amp
        self.dex_reference = cfg.dex_reference
        self.contact_count = len(cfg.contact_body_names) if cfg.dex_reference else 0
        self.compact_state = cfg.compact_state
        self.include_delay_ready = cfg.include_delay_ready
        self.geometry = HandObjectGeometry(geometry_spec) if cfg.geometry_enabled else None
        extra = self.geometry.feature_dim if self.geometry is not None else 0
        width = 57 + 21 + (6 if cfg.dex_reference else 21 + int(cfg.include_delay_ready) if cfg.compact_state else 0) + context_dim + extra
        self.input_norm = input_normalizer(cfg, width)
        if cfg.network_revision != 'legacy':
            scale, offset = input_affine(cfg, len(self.input_norm.mean))
            self.register_buffer('input_scale', scale)
            self.register_buffer('input_offset', offset)
        self.delta_norm = RunningNorm(54, eps=1e-4)
        self.register_buffer("context_scale", torch.ones(context_dim))
        self.net = mlp(width, cfg.wm_hidden, 54 + self.contact_count, True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.learned_termination = cfg.wm_task_losses
        if self.learned_termination:
            self.termination_net = mlp(57, (64, 64), 1)
            nn.init.zeros_(self.termination_net[-1].weight)
            # Initial prior only, subsequently learned from unweighted hard labels.
            nn.init.constant_(self.termination_net[-1].bias, -4.59511985)
            self.register_buffer('termination_temperature', torch.ones(()))
            self.register_buffer('termination_bias', torch.zeros(()))

    def termination_logits(self, physical):
        if not self.learned_termination:
            raise ValueError('This historical WM has no learned termination head')
        features = physical_features(physical) * self.input_scale[:57] + self.input_offset[:57]
        features = (features - self.input_norm.mean[:57]) / self.input_norm.std[:57]
        return run_mlp(self.termination_net, features, self.amp).squeeze(-1)

    def termination_probability(self, physical):
        logits = self.termination_logits(physical)
        return torch.sigmoid(logits / self.termination_temperature + self.termination_bias)

    def inputs(self, s, target, context):
        pieces = [physical_features(s), target]
        if self.dex_reference:
            from .dex_reference import WRENCH
            pieces.append(s[..., WRENCH])
        elif self.compact_state:
            pieces.append(s[..., COMPACT.last_action])
            if self.include_delay_ready:
                pieces.append(s[..., COMPACT.delay_ready])
        if self.geometry is not None:
            pieces.append(self.geometry(s))
        result = torch.cat((*pieces, context / self.context_scale), -1)
        if hasattr(self, 'input_scale'):
            result = result * self.input_scale + self.input_offset
        return result

    def predict(self, s, target, context):
        output = run_mlp(self.net, self.input_norm(self.inputs(s, target, context)), self.amp)
        physical = integrate_physical(s, output[..., :54] * self.delta_norm.std + self.delta_norm.mean)
        return physical, output[..., 54:]

    def forward(self, s, target, context):
        return self.predict(s, target, context)[0]


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.amp = cfg.amp
        self.obs_norm = (RunningNorm(cfg.observation_dim) if cfg.dex_reference
                         else input_normalizer(cfg, cfg.observation_dim))
        if cfg.network_revision != 'legacy':
            scale, offset = input_affine(cfg, cfg.observation_dim, actor=True)
            self.register_buffer('observation_scale', scale)
            if cfg.network_revision == 'hardware_center25_v3':
                self.register_buffer('observation_offset', offset)
        self.net = mlp(cfg.observation_dim * cfg.history, cfg.actor_hidden, 21)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
            if cfg.absolute_policy_optimization:
                limits = torch.tensor(cfg.network_joint_limits)
                home = torch.tensor(cfg.action_home) - torch.tensor(cfg.encoder_bias_rad)
                normalized = 2*(home-limits[:, 0])/(limits[:, 1]-limits[:, 0])-1
                # A constant legal absolute HOME target independent of input
                # normalization at startup. Earlier layers learn after step one.
                self.net[-1].weight.zero_()
                self.net[-1].bias.copy_(torch.atanh(normalized))

    def scaled_observation(self, observation):
        result = observation * self.observation_scale if hasattr(self, 'observation_scale') else observation
        return result + self.observation_offset if hasattr(self, 'observation_offset') else result

    def logits(self, history):
        return run_mlp(self.net, self.obs_norm(self.scaled_observation(history)).flatten(1), self.amp)

    def forward(self, history):
        # Exactly one output tanh. The direct task decodes it affinely to absolute
        # asset limits; legacy tasks retain their original action transform.
        return self.logits(history).tanh()

    def sample(self, history, noise_std):
        # Perturb latent commands, not already bounded actions. Replay and
        # observation history both receive the command actually sent to Newton.
        logits = self.logits(history)
        return (logits + noise_std * torch.randn_like(logits)).tanh()


class Critic(nn.Module):
    def __init__(self, context_dim, cfg, geometry_spec=None):
        super().__init__()
        self.amp = cfg.amp
        self.geometry = HandObjectGeometry(geometry_spec) if cfg.geometry_enabled else None
        extra = self.geometry.feature_dim if self.geometry is not None else 0
        self.include_delay_ready = cfg.include_delay_ready
        self.dex_reference = cfg.dex_reference
        self.input_norm = input_normalizer(cfg, cfg.critic_state_dim + context_dim + extra)
        if cfg.network_revision != 'legacy':
            scale, offset = input_affine(cfg, len(self.input_norm.mean), critic=True)
            self.register_buffer('input_scale', scale)
            self.register_buffer('input_offset', offset)
        self.register_buffer("context_scale", torch.ones(context_dim))
        self.constrained = cfg.constraints_enabled
        self.paper_critic = cfg.paper_critic
        self.stable_values = cfg.uses_target_critic
        from .constraints import cost_names
        self.net = mlp(cfg.critic_state_dim + context_dim + extra, cfg.critic_hidden,
                       1 + len(cost_names(cfg)) if self.constrained else 1)
        if self.stable_values:
            from .stable_values import value_bounds
            self.register_buffer('value_bounds', torch.tensor(value_bounds(cfg)))
            # A low initial risk prior; logits are trained with soft-label BCE,
            # so a saturated value retains a corrective supervised gradient.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, -4.59511985)
        elif self.paper_critic:
            # Unbounded linear values, trained in return units by Eq. (3).
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif self.constrained:
            with torch.no_grad():
                self.net[-1].bias[1:] = -5.


    def inputs(self, s, context):
        features = full_features(s)
        if not self.include_delay_ready and not self.dex_reference:
            features = features[..., :-1]
        pieces = [features]
        if self.geometry is not None:
            pieces.append(self.geometry(s))
        result = torch.cat((*pieces, context / self.context_scale), -1)
        if hasattr(self, 'input_scale'):
            result = result * self.input_scale + self.input_offset
        return result

    def logits(self, s, context):
        return run_mlp(self.net, self.input_norm(self.inputs(s, context)), self.amp)

    def forward(self, s, context):
        value = self.logits(s, context)
        if self.paper_critic:
            return value
        if self.stable_values:
            return value.sigmoid() * self.value_bounds
        if self.constrained:
            return torch.cat((value[..., :1], torch.nn.functional.softplus(value[..., 1:])), -1)
        return value.squeeze(-1)
