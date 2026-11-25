"""Pinned FlashSAC with an explicit HOME/angle-noise initialization adaptation."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import torch
from torch import nn
import gymnasium as gym
import numpy as np
from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig, _sample_flashsac_actions, _update_networks
from flash_rl.agents.flashSAC.update import _select_min_q_log_probs
from flash_rl.agents.flashSAC.layer import NormalTanhPolicy
from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler

VERSION = 'flashsac_revo3_die47_hist25_s0_v1'
UPSTREAM = '87edc9061150ae9e962dd84e6544e27a1554b3ab'


def configuration(device, seed, updates, *, capacity=1_000_000, warmup=100_352, batch=2048):
    return FlashSACConfig(seed=seed, device_type=device, buffer_device_type=device,
        buffer_max_length=capacity, buffer_min_length=warmup, sample_batch_size=batch,
        normalize_reward=True, normalized_G_max=5., asymmetric_observation=True,
        learning_rate_init=3e-4, learning_rate_peak=3e-4, learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6, learning_rate_warmup_step=int(1e-6*updates),
        learning_rate_decay_rate=1., learning_rate_decay_step=max(1, updates),
        actor_num_blocks=2, actor_hidden_dim=128, actor_bc_alpha=0.,
        actor_noise_zeta_mu=2., actor_noise_zeta_max=16, actor_update_period=2,
        critic_num_blocks=2, critic_hidden_dim=256, critic_num_bins=101,
        critic_min_v=-5., critic_max_v=5., critic_target_update_tau=.01,
        temp_initial_value=.01, temp_target_sigma=.15, temp_target_entropy=0.,
        gamma=.99, n_step=1, use_compile=False, compile_mode='default', use_amp=False,
        load_optimizer=True, load_reward_normalizer=True)


class HomeNormalTanhPolicy(NormalTanhPolicy):
    """Independent gains retain unit-normalized output weights and exact initial HOME.

    Gains are learned, never normalized or clamped. This changes initialization
    and parameterization, not tanh density, support or the absolute command map.
    """
    def __init__(self, hidden_dim, policy_cfg, initial_std_deg=.5):
        super().__init__(hidden_dim, 21)
        self.mean_gain = nn.Parameter(torch.zeros(21))
        self.std_gain = nn.Parameter(torch.zeros(21))
        limits = torch.tensor(policy_cfg.network_joint_limits)
        half = (limits[:, 1]-limits[:, 0])*.5
        home = torch.tensor(policy_cfg.action_home)-torch.tensor(policy_cfg.encoder_bias_rad)
        a = (home-limits.mean(-1))/half
        sigma = math.radians(initial_std_deg)/(half*(1-a.square()))
        raw_std = torch.atanh((sigma.log()+10)/6-1)
        if not torch.isfinite(raw_std).all() or not (a.abs() < 1).all():
            raise ValueError('HOME/noise cannot be represented by the policy')
        with torch.no_grad():
            self.mean_bias.copy_(torch.atanh(a))
            self.std_bias.copy_(raw_std)

    def get_mean_and_std(self, x, training):
        mean = self.mean_gain*self.mean_w(x)+self.mean_bias
        raw = self.std_gain*self.std_w(x)+self.std_bias
        return mean, torch.exp(-10+6*(1+torch.tanh(raw)))


class RateLimitedActor(FlashSACActor):
    """Network sees history only; controller uses previous target or reset encoder.

    Entropy is defined in local tanh coordinates, not global action coordinates.
    This same differentiable map is used by collection, actor loss and TD targets.
    """
    def configure_rate(self, cfg):
        limits = torch.tensor(cfg.network_joint_limits)
        self.register_buffer('rate_delta', cfg.cost_target_rate_limit_rad_s*cfg.control_dt /
                             ((limits[:, 1]-limits[:, 0])*.5))

    def map_action(self, local, observations):
        previous = torch.where(observations[:, 897:898].bool(),
                               observations[:, 867:888], observations[:, 810:831])
        lower = torch.maximum(previous-self.rate_delta, previous.new_tensor(-1.))
        upper = torch.minimum(previous+self.rate_delta, previous.new_tensor(1.))
        return (lower+upper)*.5+(upper-lower)*.5*local

    def forward(self, observations, training):
        local, info = super().forward(observations[:, :810], training)
        return self.map_action(local, observations), info


class RevoFlashSAC(FlashSACAgent):
    def __init__(self, cfg, policy_cfg, initial_std_deg=.5, action_mode="s0"):
        if cfg.n_step != 1 or cfg.use_amp or cfg.use_compile:
            raise ValueError('v1 is audited only for 1-step eager FP32')
        super().__init__(gym.spaces.Box(-np.inf, np.inf, (963,), dtype=np.float32),
            gym.spaces.Box(-1., 1., (21,), dtype=np.float32), {'actor_observation_size': (810,)}, cfg)
        if action_mode not in ('s0', 's1'):
            raise ValueError('Unknown action contract')
        self.action_mode = action_mode
        self.version = VERSION if action_mode == 's0' else 'flashsac_revo3_die47_hist25_s1_v1'
        self.target_rate_deg_s = math.degrees(policy_cfg.cost_target_rate_limit_rad_s)
        if action_mode == 's1' and not math.isclose(self.target_rate_deg_s, 60.):
            self.version = f'flashsac_revo3_die47_hist25_s1_rate{self.target_rate_deg_s:g}_v1'
        network = self._actor.network
        if action_mode == 's1':
            replacement = RateLimitedActor(cfg.actor_num_blocks, 810, cfg.actor_hidden_dim, 21).to(self._device)
            replacement.load_state_dict(network.state_dict())
            replacement.configure_rate(policy_cfg)
            network = replacement.to(self._device)
        network.predictor = HomeNormalTanhPolicy(cfg.actor_hidden_dim, policy_cfg, initial_std_deg).to(self._device)
        if action_mode == 's1':
            sigma = math.radians(initial_std_deg)/(policy_cfg.cost_target_rate_limit_rad_s*policy_cfg.control_dt)
            with torch.no_grad():
                network.predictor.mean_bias.zero_()
                network.predictor.std_bias.fill_(math.atanh((math.log(sigma)+10)/6-1))
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate_peak,
                                     fused=self._device.type == 'cuda')
        schedule = warmup_cosine_decay_scheduler(cfg.learning_rate_init, cfg.learning_rate_peak,
            cfg.learning_rate_end, cfg.learning_rate_warmup_step, cfg.learning_rate_decay_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(step)/cfg.learning_rate_peak)
        self._actor = Network(network, optimizer, scheduler, use_weight_normalization=True)
        self._actor.normalize_parameters()
        self.initial_std_deg = initial_std_deg

    @torch.no_grad()
    def act(self, observation, *, training):
        n = len(observation)
        if self._cached_noise.shape != (n, 21):
            self._cached_noise = torch.randn(n, 21, device=self._device)
            self._cur_noise_repeat_count.zero_()
        self._cached_noise, action, self._cur_noise_repeat_count, self._cur_noise_repeat_n = _sample_flashsac_actions(
            self._actor, self._cached_noise, observation[:, :810], float(training),
            self._cur_noise_repeat_count, self._cur_noise_repeat_n, self._zeta_cdf)
        return self._actor.network.map_action(action, observation) if self.action_mode == "s1" else action

    @torch.no_grad()
    def reset_noise(self, done):
        # Keep upstream global duration; independently refresh only reset slots.
        if done.any() and self._cached_noise.ndim == 2:
            self._cached_noise[done] = torch.randn(int(done.sum()), 21, device=self._device)

    def update(self):
        # Same upstream update call/order; observe the actual TD target without
        # another sample, RNG draw or BatchNorm forward.
        batch = {k:v.to(self._device, non_blocking=True) for k,v in self._replay_buffer.sample().items()}
        batch['actor_observation'] = batch['observation'][:, :963 if self.action_mode == 's1' else 810]
        batch['actor_next_observation'] = batch['next_observation'][:, :963 if self.action_mode == 's1' else 810]
        batch['reward'] = self.reward_normalizer.normalize_rewards(batch['reward'])
        captured = {}
        def actor_hook(module, inputs, output):
            if not torch.is_grad_enabled():
                captured['next_log_pi'] = output[1]['log_prob'].detach().clone()
        def target_hook(module, inputs, output):
            captured['qs'] = output[0].detach().clone().chunk(2, dim=1)[1]
            captured['log_probs'] = output[1]['log_prob'].detach().clone().chunk(2, dim=1)[1]
        handles = [self._actor.network.register_forward_hook(actor_hook),
                   self._target_critic.network.register_forward_hook(target_hook)]
        try:
            result = _update_networks(batch, self._actor, self._critic, self._target_critic, self._temperature,
                self._cfg, self._update_step % self._cfg.actor_update_period == 0, self._device, self._grad_scaler)
        finally:
            for handle in handles: handle.remove()
        self._update_step += 1
        with torch.no_grad():
            probabilities = _select_min_q_log_probs(captured['qs'], captured['log_probs']).exp()
            support = torch.linspace(-5, 5, 101, device=self._device)
            entropy_term = self._temperature()*captured['next_log_pi']
            targets = batch['reward'][:, None] + self._cfg.gamma*(1-batch['terminated'][:, None])*(support-entropy_term[:,None])
            result['diagnostic/target_outside_support_mass'] = (probabilities*((targets < -5)|(targets > 5))).sum(-1).mean()
            result['diagnostic/target_below_support_mass'] = (probabilities*(targets < -5)).sum(-1).mean()
            result['diagnostic/target_above_support_mass'] = (probabilities*(targets > 5)).sum(-1).mean()
            result['diagnostic/target_lower_edge_probability'] = probabilities[:,0].mean()
            result['diagnostic/target_upper_edge_probability'] = probabilities[:,-1].mean()
            result['diagnostic/target_q_mean'] = captured['qs'].mean()
            result['diagnostic/next_logpi_mean'] = captured['next_log_pi'].mean()
            result['diagnostic/alpha'] = self._temperature().squeeze()
            result['diagnostic/normalized_reward_mean'] = batch['reward'].mean()
            result['diagnostic/target_edge_probability'] = (probabilities[:,0]+probabilities[:,-1]).mean()
            result['diagnostic/twin_q_gap'] = (captured['qs'][0]-captured['qs'][1]).abs().mean()
            result['diagnostic/alpha_logpi_abs_mean'] = entropy_term.abs().mean()
            norm = self.reward_normalizer
            result['diagnostic/reward_denominator'] = torch.maximum(torch.sqrt(norm.G_rms.var+norm.epsilon),norm.G_r_max/norm.G_max).squeeze()
            bn = self._actor.network.embedder.norm
            z = (batch['actor_observation'][:, :810]-bn.running_mean)/torch.sqrt(bn.running_var+bn.eps)
            result['diagnostic/velocity_bn_abs_p99'] = torch.quantile(z.reshape(-1,10,81)[:,:,21:42].abs().flatten(),.99)
        result = {k:float(v.detach()) if isinstance(v,torch.Tensor) else float(v) for k,v in result.items()}
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError('Nonfinite FlashSAC update metrics')
        for network in (self._actor, self._critic, self._target_critic, self._temperature):
            if any(not torch.isfinite(p).all() for p in network.network.parameters()):
                raise FloatingPointError('Nonfinite FlashSAC parameter')
        return result


def tree_hash(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def save_checkpoint(path, agent, encoder, metadata, *, replay=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    agent.save(str(path))
    torch.save(encoder.state_dict(), path/'encoder.pt')
    torch.save(dict(cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        noise=agent._cached_noise, repeat_count=agent._cur_noise_repeat_count,
        repeat_n=agent._cur_noise_repeat_n), path/'collection_state.pt')
    if replay:
        agent.save_replay_buffer(str(path))
    payload = dict(metadata, version=agent.version, upstream=UPSTREAM, action_mode=agent.action_mode, algorithm=asdict(agent._cfg),
                   initial_std_deg=agent.initial_std_deg, q_updates=agent._update_step,
                   resume_semantics='evaluation only; exact simulator resume not implemented')
    (path/'metadata.json').write_text(json.dumps(payload, indent=2)+'\n')
    (path/'hashes.json').write_text(json.dumps(tree_hash(path), indent=2)+'\n')


def load_evaluation(path, agent, encoder, contract):
    path = Path(path)
    data = json.loads((path/'metadata.json').read_text())
    if data['version'] != agent.version or data['upstream'] != UPSTREAM or data['environment_contract'] != contract:
        raise ValueError('Checkpoint protocol/environment mismatch')
    expected = json.loads((path/'hashes.json').read_text())
    actual = tree_hash(path); actual.pop('hashes.json')
    if expected != actual:
        raise ValueError('Checkpoint integrity mismatch')
    agent.load(str(path))
    encoder.load_state_dict(torch.load(path/'encoder.pt', map_location=agent._device, weights_only=True), strict=True)
    return data
