"""Autoregressive WM fitting, differentiable actor learning, and lambda Critic.

Microbatches accumulate gradients into the requested logical batch. WM/actor
optimizer update counts do not depend on microbatch size. Critic targets are
computed once with the pre-update Critic, then held fixed for all five epochs.
"""

from contextlib import contextmanager
import copy
from dataclasses import asdict
import math
from pathlib import Path
import torch
from torch import nn
from .core import *
from .networks import Actor, Critic, Dynamics
from .sensors import observe
from .distributed import Distributed

CHECKPOINT_VERSION = 17
DIRECT_CHECKPOINT_VERSION = 25
IMAGINATION_PROTOCOL = "bounded_actor_imagined_lambda_online_critic_v1"
MICROBATCH_FIELDS = ("microbatch", "wm_microbatch", "actor_microbatch", "critic_microbatch")


def imagination_protocol(cfg):
    if cfg.dex_reference:
        return "encoder_EMA_zero_bias_contacts_independent_failure_v33"
    if cfg.paper_critic:
        return 'absolute_position_imagined_lambda_linear_online_critic_mse_v32'
    if cfg.absolute_policy_optimization:
        base = 'absolute_position_correlated_exploration_bounded_values'
        if cfg.input_joint_velocity_std_floor_rad_s:
            base += '_physical_velocity_norm'
        return (base + (f'_real_td{cfg.critic_td_steps}' if cfg.critic_td_steps > 1 else '')
                + ('_delay_ready_v31' if cfg.include_delay_ready else '_v30'))
    return ('bounded_failure_risk_target_critic_real_TD_v29' if cfg.stable_value_learning
            else IMAGINATION_PROTOCOL)


def checkpoint_version(cfg):
    if cfg.dex_reference:
        return 33
    if cfg.paper_critic:
        return 32
    if cfg.absolute_policy_optimization:
        return 31 if cfg.include_delay_ready else 30
    if cfg.stable_value_learning:
        return 29
    if cfg.wm_task_losses:
        return 28
    if cfg.network_revision == 'hardware_center25_v3':
        return 27
    if cfg.network_revision != 'legacy':
        return 26
    return DIRECT_CHECKPOINT_VERSION if cfg.observation_protocol == DIRECT_OBSERVATION_PROTOCOL else CHECKPOINT_VERSION


def imagined_continuation(state, next_physical, survival_width, single_goal=False, cfg=None):
    """Analytic survival from predicted position; differentiable across 0.3 m.

    Like the reference SuperMimic rollouts, imagined drops discount future
    return softly. Actual simulator termination remains hard in real replay.
    The task's success-count limit is an administrative episode boundary.
    """
    from .spatial_contract import drop_distance
    distance = next_physical[:, POS].norm(dim=-1) if cfg is None else drop_distance(next_physical, cfg)
    radius = .3 if cfg is None else cfg.drop_radius_m
    survival = torch.sigmoid((radius - distance) / survival_width)
    if cfg is not None and cfg.drop_boundary_mode == 'com_ellipsoid':
        survival = torch.where(distance > radius, 0., survival)
    return survival if single_goal else survival * (state[:, COUNT].squeeze(-1) < 50)


@contextmanager
def frozen(*models):
    saved = [(p, p.requires_grad) for m in models for p in m.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, requires in saved:
            p.requires_grad_(requires)


def lambda_returns(rewards, continuation, values, gamma, lam):
    result = []
    acc = values[-1]
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * continuation[t] * ((1 - lam) * values[t + 1] + lam * acc)
        result.append(acc)
    return torch.stack(result[::-1])


def checked_step(model, optimizer, value_clip=False, distributed=None, *, tensor_norm=False):
    if distributed is not None:
        distributed.gradients(model)
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if value_clip:
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
    optimizer.step()
    return norm.detach() if tensor_norm else float(norm)


def scalar_metrics(metrics):
    """Transfer detached scalar diagnostics together, outside optimizer updates."""
    names = [name for name, value in metrics.items() if isinstance(value, torch.Tensor)]
    if not names:
        return dict(metrics)
    values = torch.stack([metrics[name].detach().to(torch.float64).reshape(()) for name in names]).cpu().tolist()
    return {**metrics, **dict(zip(names, values))}


def actor_ramp(iteration, warmup, duration):
    if iteration < warmup:
        return 0.0
    return min((iteration - warmup + 1) / max(1, duration), 1.0)


def validate_resume_actor_ramp(schedule, iteration):
    """Validate optional runtime scheduling without changing the Config contract."""
    if schedule is None:
        return None
    if (type(schedule) is not dict or set(schedule) != {"start_iteration", "duration"}
            or type(iteration) is not int or iteration < 0):
        raise ValueError("Resume actor ramp requires an iteration-bound schedule")
    start, duration = schedule["start_iteration"], schedule["duration"]
    if (type(start) is not int or not 0 <= start <= iteration
            or type(duration) is not int or duration < 1):
        raise ValueError("Resume actor ramp requires 0 <= start <= iteration and a positive integer duration")
    return dict(start_iteration=start, duration=duration)


class Trainer:
    @property
    def sync_models(self):
        models = (self.wm, self.actor, self.critic)
        return models + ((self.target_critic,) if self.cfg.uses_target_critic else ())

    def __init__(self, cfg, noise_cfg, context, limits, base_inverse, device, episode_steps=600, distributed=None, geometry_spec=None, wm_cuda_graph=False):
        self.cfg, self.noise_cfg = cfg, noise_cfg
        self.context, self.limits, self.base_inverse = context, limits, base_inverse
        self.device, self.episode_steps = device, episode_steps
        self.distributed = distributed or Distributed(device)
        if cfg.geometry_enabled and geometry_spec is None:
            raise ValueError("Geometry training requires actual-asset geometry metadata")
        self.geometry_spec = geometry_spec
        self.wm = Dynamics(context.shape[-1], cfg, geometry_spec).to(device)
        self.actor = Actor(cfg).to(device)
        self.critic = Critic(context.shape[-1], cfg, geometry_spec).to(device)
        if cfg.compile_geometry:
            if not cfg.geometry_enabled:
                raise ValueError("compile_geometry requires geometry_enabled")
            self.wm.geometry.enable_compilation()
            self.critic.geometry.enable_compilation()
        # Dimensionless domain parameters: otherwise a generic 0.01 normalization
        # epsilon would almost erase kg*m^2 inertia and gram-scale masses.
        with torch.no_grad():
            scale = self.distributed.reduce(context.abs().mean(0)).clamp_min(1e-12)
            self.wm.context_scale.copy_(scale)
            self.critic.context_scale.copy_(scale)
        self.distributed.broadcast_models(self.wm, self.actor, self.critic)
        if cfg.uses_target_critic:
            self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        wm_parameters = (p for n, p in self.wm.named_parameters()
                         if not cfg.dex_reference or not n.startswith("termination_net."))
        self.opt_wm = torch.optim.Adam(wm_parameters, lr=cfg.learning_rate)
        if cfg.dex_reference:
            self.opt_failure = torch.optim.Adam(self.wm.termination_net.parameters(), lr=cfg.learning_rate)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.iteration = 0
        self.resume_actor_ramp = None
        self.step_credit = 0.0
        self.real_steps = 0
        self.optimizer_steps = dict(wm=0, actor=0, critic=0)
        if cfg.dex_reference:
            self.optimizer_steps["failure"] = 0
        self.grasp_manager = None
        self.wm_cuda_graph = wm_cuda_graph
        self._wm_capture = None
        # Fixed physical-unit floors are allocated once, also safe for capture.
        self._wm_floor = torch.tensor(
            [0.003] * 21 + [0.05] * 21 + [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3, device=self.device
        )
        self._wm_encoder_bias = torch.tensor(cfg.encoder_bias_rad, device=self.device)
        self._wm_home = torch.tensor(cfg.action_home, device=self.device)
        self._wm_hand_center = torch.tensor(cfg.hand_center_base_m, device=self.device)

    def arm_resume_actor_ramp(self, duration):
        """Start an explicit Actor-only LR ramp on the next normal update.

        Parameters, normalizers and all Adam state/LRs stay untouched here.
        Restoring a checkpoint preserves its existing start; only this explicit
        call restarts the runtime schedule.
        """
        self.resume_actor_ramp = validate_resume_actor_ramp(
            dict(start_iteration=self.iteration, duration=duration), self.iteration)

    def continuation(self, state, next_physical):
        if self.cfg.wm_task_losses:
            continuation = 1 - self.wm.termination_probability(next_physical)
            if self.cfg.stable_value_learning or self.cfg.drop_boundary_mode == 'com_ellipsoid':
                from .spatial_contract import drop_distance
                # The known simulator boundary takes precedence over a false
                # negative from the learned classifier. Inside it stays soft.
                continuation = torch.where(drop_distance(next_physical, self.cfg) > self.cfg.drop_radius_m,
                                           0., continuation)
            return continuation
        return imagined_continuation(state, next_physical, self.cfg.survival_width,
                                     single_goal=self.cfg.goal_mode == 'single_dense_hold', cfg=self.cfg)

    @torch.no_grad()
    def update_real_statistics(self, state, target, nxt, observation, *, policy_statistics=True):
        self.wm.input_norm.update(self.wm.inputs(state, target, self.context), self.distributed)
        self.wm.delta_norm.update(physical_delta(state, nxt), self.distributed)
        if policy_statistics:
            self.actor.obs_norm.update(self.actor.scaled_observation(observation), self.distributed)
            self.critic.input_norm.update(self.critic.inputs(state, self.context), self.distributed)
            if self.cfg.uses_target_critic:
                from .stable_values import sync_value_buffers
                sync_value_buffers(self)

    def _wm_loss(self, w):
        """Pure device computation, shared by eager execution and CUDA capture."""
        cfg = self.cfg
        physical_totals = torch.zeros(6, device=self.device)
        geometry_totals = torch.zeros(5, device=self.device)
        from .wm_supervision import task_losses, loss_names
        task_totals = torch.zeros(len(loss_names(cfg)), device=self.device)
        scale = torch.maximum(self.wm.delta_norm.std, self._wm_floor)
        context = self.context[w["env_id"]]
        limits = self.limits[w["env_id"]]
        s = w["state"][:, 0]
        loss = 0.0
        failure_loss = 0.0
        sequence_error = torch.zeros(len(s), device=self.device)
        weights = w.get("importance_weight", torch.ones_like(sequence_error))
        for t in range(cfg.wm_window):
            active = w['mask'][:, t].bool()
            # A reset tail is unrelated to this rollout. Exclude it before
            # model/readout evaluation: NaN * 0 would still poison backward.
            action = torch.where(active[:, None], w['action'][:, t], 0.)
            real_state = torch.where(active[:, None], w['state'][:, t], s.detach())
            real_next = torch.where(active[:, None], w['next_physical'][:, t], s[:, :55].detach())
            if cfg.dex_reference:
                # Only the force realized for this transition is known; no future force leakage.
                s = torch.cat((s[..., :125], real_state[..., 125:131]), -1)
            target = action_target(s, action, limits, cfg)
            if cfg.dex_reference:
                pred, contact_logits = self.wm.predict(s, target, context)
            else:
                pred = self.wm(s, target, context)
            # Error in physical coordinates, including SO(3) orientation residual.
            error = physical_delta(real_next, pred) / scale
            groups = torch.stack([part.square().mean(-1) for part in error.split((21, 21, 3, 3, 3, 3), -1)], -1)
            physical_loss = groups.mean(-1) if cfg.grouped_physical_loss else error.square().mean(-1)
            step_loss = physical_loss
            if cfg.wm_task_losses:
                auxiliary, components = task_losses(self, s, pred, real_state,
                    real_next, action, limits,
                    torch.where(active, w['reward'][:, t], 0.), w['terminated'][:, t] & active)
                step_loss = step_loss + auxiliary
                if cfg.dex_reference:
                    from torch.nn import functional as F
                    labels = torch.where(active[:, None], w['contact'][:, t], 0.).float()
                    contact_loss = F.binary_cross_entropy_with_logits(contact_logits, labels, reduction='none').sum(-1)
                    step_loss = step_loss + cfg.contact_loss_weight*contact_loss
                    components[:, -1] = contact_loss.detach()
                    terminal = (w['terminated'][:, t] & active).float()
                    independent = (F.binary_cross_entropy_with_logits(self.wm.termination_logits(real_next.detach()), terminal, reduction='none')
                        + F.binary_cross_entropy_with_logits(self.wm.termination_logits(pred.detach()), terminal, reduction='none'))
                    failure_loss = failure_loss + (independent*w['mask'][:, t]*weights).sum()/cfg.wm_batch
                    components[:, 2] = independent.detach()
                task_totals += (components.detach() * (w['mask'][:, t] * weights)[:, None]).sum(0) / cfg.wm_batch
            physical_totals += (groups.detach() * (w['mask'][:, t] * weights)[:, None]).sum(0) / cfg.wm_batch
            if self.wm.geometry is not None:
                geometric = self.wm.geometry.supervised_errors(pred, real_next)
                step_loss = step_loss + cfg.geometry_loss_weight * geometric.mean(-1)
                geometry_totals += (geometric.detach() * (w['mask'][:, t] * weights)[:, None]).sum(0) / cfg.wm_batch
            masked = step_loss * w["mask"][:, t]
            sequence_error += masked.detach()
            loss = loss + (masked * weights).sum() / cfg.wm_batch
            # No intermediate physical teacher forcing. The pending last command and reset flag enter the delayed-actuator WM.
            advanced = (advance(s, pred, action, target, s[:, layout(s).goal_rot], self.episode_steps, hold_cfg=cfg)
                        if cfg.compact_state else torch.cat((pred, target, s[:, 76:]), -1))
            s = torch.where((active & ~w['done'][:, t])[:, None], advanced, s)
        if cfg.dex_reference:
            self._independent_failure_loss = failure_loss
        return loss, sequence_error / w["mask"].sum(-1).clamp_min(1), physical_totals, geometry_totals, task_totals

    def wm_update(self, replay, *, tensor_metrics=False):
        cfg = self.cfg
        chunk_size = cfg.wm_microbatch or cfg.microbatch
        batch = (replay.sample(cfg.wm_batch, cfg.wm_window, include_events=False) if cfg.wm_task_losses
                 else replay.sample(cfg.wm_batch, cfg.wm_window))
        if cfg.wm_task_losses:
            if 'reward' not in batch or 'terminated' not in batch:
                raise ValueError('WM task supervision requires real reward and terminated labels')
            # A calibration is tied to frozen weights. Any refit invalidates it.
            self.wm.termination_temperature.fill_(1.)
            self.wm.termination_bias.zero_()
        if cfg.dex_reference:
            self.opt_failure.zero_grad(set_to_none=True)
        sequence_errors = []
        # FP64 scalar accumulation matches the former Python-float diagnostics;
        # none of these detached accumulators participate in backward.
        loss_total = torch.zeros((), device=self.device, dtype=torch.float64)
        physical_totals = torch.zeros(6, device=self.device)
        geometry_totals = torch.zeros(5, device=self.device)
        from .wm_supervision import loss_names
        task_totals = torch.zeros(len(loss_names(cfg)), device=self.device)
        if self.wm_cuda_graph:
            if chunk_size < cfg.wm_batch:
                raise ValueError("WM CUDA graph requires a full logical batch")
            if self._wm_capture is None:
                from .wm_cuda_graph import CapturedWMLoss
                self._wm_capture = CapturedWMLoss(self, batch)
            loss, errors, physical_totals, geometry_totals, task_totals = self._wm_capture(batch)
            loss_total = loss.detach().to(torch.float64)
            sequence_errors.append(errors)
        else:
            self.opt_wm.zero_grad(set_to_none=True)
            for begin in range(0, cfg.wm_batch, chunk_size):
                w = {k: v[begin : begin + chunk_size] for k, v in batch.items()}
                loss, errors, physical, geometric, auxiliary = self._wm_loss(w)
                loss.backward()
                if cfg.dex_reference:
                    self._independent_failure_loss.backward()
                    self._independent_failure_loss = None
                loss_total += loss.detach()
                sequence_errors.append(errors)
                physical_totals += physical
                geometry_totals += geometric
                task_totals += auxiliary
        if cfg.dex_reference:
            fgn = checked_step(self.wm.termination_net, self.opt_failure, distributed=self.distributed, tensor_norm=True)
            self.opt_failure.zero_grad(set_to_none=True)
            self.optimizer_steps["failure"] += 1
        gn = checked_step(self.wm, self.opt_wm, distributed=self.distributed, tensor_norm=True)
        replay.update_priorities(batch, torch.cat(sequence_errors))
        self.optimizer_steps["wm"] += 1
        metrics = {"wm_loss": loss_total / cfg.wm_window, "wm_grad_norm": gn}
        if cfg.dex_reference:
            metrics["failure_grad_norm"] = fgn
            valid = batch['mask'][..., None]
            positives = (batch['contact'].float()*valid).sum()
            metrics['wm_contact_positive_fraction'] = positives/(valid.sum()*len(cfg.contact_body_names)).clamp_min(1)
            metrics['failure_loss'] = task_totals[2].detach()/cfg.wm_window
        names = ("joint_position", "joint_velocity", "object_position", "object_rotation", "object_velocity", "object_omega")
        metrics.update({"wm_loss_" + name: value.detach().to(torch.float64) / cfg.wm_window for name, value in zip(names, physical_totals)})
        if cfg.wm_task_losses:
            metrics.update({'wm_loss_' + name: value.detach().to(torch.float64) / cfg.wm_window
                            for name, value in zip(loss_names(cfg), task_totals)})
        if self.wm.geometry is not None:
            names = ("point_position", "point_velocity", "relative_position", "relative_velocity", "surface_gap")
            metrics.update({"wm_loss_" + name: value.detach().to(torch.float64) / cfg.wm_window for name, value in zip(names, geometry_totals)})
        return metrics if tensor_metrics else scalar_metrics(metrics)

    def update_world_model(self, replay, *, updates=None):
        """Refit fresh replay without advancing training time or policy optimizers.

        The caller owns data collection and held-out readiness decisions. This
        method changes only WM parameters/Adam and its optimizer-step counter;
        it deliberately leaves iteration, step credit, actor ramp and actor /
        Critic parameters, normalizers and optimizer state untouched.
        """
        count = self.cfg.wm_updates if updates is None else updates
        if type(count) is not int or count < 1:
            raise ValueError("WM readiness updates must be a positive integer")
        for _ in range(count):
            # Preserve historical metrics: report the last WM optimizer step.
            metrics = self.wm_update(replay, tensor_metrics=True)
        return scalar_metrics({**metrics, "readiness_wm_updates": count})

    def imagine(self, batch):
        if self.cfg.stable_value_learning:
            from .stable_values import imagine_stable
            return imagine_stable(self, batch)
        if self.cfg.constraints_enabled:
            from .constraints import imagine_constrained
            return imagine_constrained(self, batch)
        cfg = self.cfg
        s = batch["state"]
        history = batch["history"]
        sensors = {k[7:]: v for k, v in batch.items() if k.startswith("sensor/")}
        ids = batch["env_id"]
        context = self.context[ids]
        limits = self.limits[ids]
        base_inv = self.base_inverse[ids]
        states = [s]
        rewards = []
        continuations = []
        values = [self.critic(s, context).detach()]
        valid = []
        alive = torch.ones(len(s), device=self.device)
        objective = torch.zeros_like(alive)
        ended = torch.zeros_like(alive, dtype=torch.bool)
        for t in range(cfg.horizon):
            action = self.actor(history)
            target = action_target(s, action, limits, cfg)
            nxt = self.wm(s, target, context)
            r = reward(s, nxt, action, cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight, cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight, hold_cfg=cfg)
            # The reset boundary is not simulated. At a timeout, bootstrap V once and stop.
            single = cfg.goal_mode == "single_dense_hold"
            # The goal persists indefinitely. The real 20s cutoff
            # only refreshes training samples; no terminal boundary enters this MDP.
            timeout = torch.zeros_like(ended) if single else s[:, TIME].squeeze(-1) <= (1 / self.episode_steps + 1e-7)
            continuation = self.continuation(s, nxt)
            count_limit = torch.zeros_like(ended) if single else s[:, COUNT].squeeze(-1) >= 50
            goal = s[:, layout(s).goal_rot] if single else quat_unique(quat_mul(base_inv, sample_xy_goal(s[:, GOAL_ROT])))
            s = advance(s, nxt, action, target, goal, self.episode_steps, hold_cfg=cfg)
            obs, sensors = observe(s, sensors, limits, self.noise_cfg, cfg)
            history = push_history(history, obs)
            value = self.critic(s, context)
            target_value = value.detach()
            objective = objective + cfg.gamma**t * alive * r
            objective = objective + cfg.gamma ** (t + 1) * alive * continuation * timeout * value
            active = ~ended
            valid.append(active)
            # Timeout bootstrap belongs in targets but not a rollout crossing into another episode.
            rewards.append(torch.where(active, r + cfg.gamma * continuation * timeout * target_value, 0.0))
            continuations.append(continuation * (~timeout) * active)
            # Keep all imagined physical states, including after likely drops,
            # as in the references. Only known episode limits stop the sequence.
            ended = ended | timeout | count_limit
            alive = alive * continuation * (~timeout)
            states.append(s)
            values.append(target_value)
        objective = objective + cfg.gamma**cfg.horizon * alive * value
        # Freeze these targets before ANY Critic optimizer step.
        with torch.no_grad():
            targets = lambda_returns(
                torch.stack(rewards), torch.stack(continuations), torch.stack(values), cfg.gamma, cfg.lam
            )
        return -objective.mean(), torch.stack(states[:-1]).detach(), targets.detach(), context, torch.stack(valid)

    def policy_update(self, replay):
        cfg = self.cfg
        actor_chunk = cfg.actor_microbatch or cfg.microbatch
        start = replay.sample(cfg.actor_batch, include_events=False) if cfg.stable_value_learning else replay.sample(cfg.actor_batch)
        if cfg.uses_target_critic:
            from .stable_values import fit_real_critic
            critic_batch = (start if cfg.critic_td_steps == 1 else replay.sample(
                cfg.actor_batch, window=cfg.critic_td_steps, include_events=False, prioritized=False))
            real_metrics = fit_real_critic(self, critic_batch)
        self.opt_wm.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_actor.zero_grad(set_to_none=True)
        datasets = []
        cost_metrics = {}
        objective = torch.zeros((), device=self.device, dtype=torch.float64)
        gradient_metrics = {}
        audit_policy_step = (cfg.absolute_policy_optimization
                             and self.iteration % cfg.actor_gradient_audit_interval == 0)
        if audit_policy_step:
            # Fixed real histories isolate the optimizer's change from physics,
            # exploration and normalization updates. No policy/action clipping.
            with torch.no_grad():
                audit_history = start['history'][:min(actor_chunk, 256)]
                audit_action_before = self.actor(audit_history)
                audit_scaled = self.actor.scaled_observation(audit_history)
                audit_velocity = self.actor.obs_norm(audit_scaled)[..., 21:42]
                gradient_metrics.update(
                    actor_input_velocity_abs_p99=torch.quantile(audit_velocity.abs(), .99),
                    actor_input_velocity_std_min=self.actor.obs_norm.std[21:42].min())
        with frozen(self.wm, self.critic):
            for begin in range(0, cfg.actor_batch, actor_chunk):
                batch = {k: v[begin : begin + actor_chunk] for k, v in start.items()}
                loss, states, targets, context, valid = self.imagine(batch)
                if audit_policy_step and begin == 0:
                    from .absolute_policy import objective_gradient_metrics
                    gradient_metrics.update(objective_gradient_metrics(self.actor,
                        {**self._actor_objective_terms, **self._actor_reward_components}))
                    gradient_metrics['actor_gradient_audit_samples'] = len(batch['state'])
                weight = len(batch["state"]) / cfg.actor_batch
                (loss * weight).backward()
                objective += loss.detach().to(torch.float64) * weight
                if cfg.constraints_enabled:
                    for name, value in self._cost_metrics.items():
                        cost_metrics[name] = cost_metrics.get(name, 0.) + value * weight
                if not cfg.uses_target_critic:
                    datasets.append((states[valid], targets[valid], context[None].expand(cfg.horizon, -1, -1)[valid]))
        gn = checked_step(self.actor, self.opt_actor, True, distributed=self.distributed, tensor_norm=True)
        self.optimizer_steps["actor"] += 1
        if audit_policy_step:
            with torch.no_grad():
                audit_limits = self.limits[start['env_id'][:len(audit_history)]]
                half_range = (audit_limits[..., 1]-audit_limits[..., 0])/2
                delta = (self.actor(audit_history)-audit_action_before)*half_range
                gradient_metrics.update(actor_update_target_rms_deg=delta.square().mean().sqrt()*(180/math.pi),
                    actor_update_target_max_deg=delta.abs().max()*(180/math.pi),
                    actor_update_audit_samples=len(audit_history))
        if cfg.uses_target_critic:
            metrics = real_metrics
            valid_fraction = cost_metrics['imagined_alive_mean']
        else:
            states, targets, context = (torch.cat([d[i] for d in datasets]) for i in range(3))
            metrics = self.fit_critic((states, targets, context), rollout_count=cfg.actor_batch)
            valid_fraction = len(states) / (cfg.actor_batch * cfg.horizon)
        if self.cfg.constraints_enabled:
            metrics.update(cost_metrics)
        if cfg.absolute_policy_optimization:
            # Release the last rollout graph after the optimizer step.
            self._actor_objective_terms = None
            self._actor_reward_components = None
            if cfg.dex_reference:
                for suffix in ('abs_p99', 'std_min'):
                    key = 'actor_input_velocity_'+suffix
                    if key in gradient_metrics:
                        gradient_metrics['actor_input_tracking_error_'+suffix] = gradient_metrics.pop(key)
            metrics.update(gradient_metrics)
        return scalar_metrics({"actor_loss": objective, "actor_grad_norm": gn,
                "imagined_valid_fraction": valid_fraction, **metrics})

    def fit_critic(self, imagined, *, rollout_count=None):
        # All targets come from the pre-update online Critic and are fixed for
        # every epoch. Match the references: no real TD mixture or EMA model.
        states, targets, context = imagined
        if targets.requires_grad:
            raise ValueError("Critic targets must be detached before fitting")
        loss_total = torch.zeros((), device=self.device, dtype=torch.float64)
        # Reward and violation returns can differ sharply early in training.
        # Balance regression heads with fixed detached per-rollout scales; this
        # changes neither value units nor the physical zero-cost budgets.
        target_scale = (targets.detach().std(0, unbiased=False).clamp_min(1.)
                        if self.cfg.constraints_enabled and not self.cfg.paper_critic else 1.)
        # Eq. (3): expectation over starts, SUM over imagined time, not mean
        # over time. Preserve this scale even when known-terminal tails are absent.
        time_scale = 1.
        if self.cfg.paper_critic:
            if rollout_count is None or rollout_count < 1:
                raise ValueError('Paper Critic requires the number of imagined rollout starts')
            counts = self.distributed.reduce(states.new_tensor([len(states), rollout_count]), 'sum')
            time_scale = counts[0] / counts[1]
        updates = 0
        for _ in range(self.cfg.critic_epochs):
            splits = torch.tensor_split(torch.randperm(len(states), device=self.device),
                                        self.cfg.critic_minibatches)
            for idx in splits:
                self.opt_critic.zero_grad(set_to_none=True)
                denominator = self.distributed.batch_denominator(len(idx))
                for chunk in idx.split(self.cfg.critic_microbatch or self.cfg.microbatch):
                    loss = ((self.critic(states[chunk], context[chunk]) - targets[chunk]) / target_scale).square().sum() / denominator * time_scale
                    loss.backward()
                    loss_total += loss.detach()
                if not len(idx):
                    (self.critic(states[:1], context[:1]).sum() * 0).backward()
                # Keep the per-step nonfinite check and clipping; the norm is
                # unused, so do not copy it to the CPU for every minibatch.
                checked_step(self.critic, self.opt_critic, distributed=self.distributed, tensor_norm=True)
                self.optimizer_steps["critic"] += 1
                updates += 1
        return scalar_metrics({"critic_loss": loss_total / updates})

    def update(self, replay):
        fraction = min(self.iteration / self.cfg.lr_decay_iters, 1.0)
        lr = self.cfg.learning_rate * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for optimizer in (self.opt_wm, self.opt_actor, self.opt_critic) + ((self.opt_failure,) if self.cfg.dex_reference else ()):
            for group in optimizer.param_groups:
                group["lr"] = lr
        global_ramp = actor_ramp(self.iteration, self.cfg.wm_only_iters, self.cfg.actor_ramp_iters)
        resume_ramp = (actor_ramp(self.iteration, self.resume_actor_ramp["start_iteration"],
                                  self.resume_actor_ramp["duration"])
                       if self.resume_actor_ramp is not None else 1.0)
        ramp = min(global_ramp, resume_ramp)
        for group in self.opt_actor.param_groups:
            group["lr"] = lr * ramp * self.cfg.actor_lr_scale
        metrics = {"learning_rate": lr, "actor_learning_rate": lr * ramp * self.cfg.actor_lr_scale, "actor_ramp": ramp,
                   "resume_actor_ramp": resume_ramp, "event_archive_size": replay.archive_size}
        for _ in range(self.cfg.wm_updates):
            metrics.update(self.wm_update(replay, tensor_metrics=True))
        if self.iteration >= self.cfg.wm_only_iters:
            metrics.update(self.policy_update(replay))
        elif self.cfg.uses_target_critic:
            from .stable_values import fit_real_critic
            metrics.update(fit_real_critic(self, replay.sample(self.cfg.actor_batch,
                window=self.cfg.critic_td_steps, include_events=False, prioritized=False)))
        self.iteration += 1
        return scalar_metrics(metrics)

    def save(self, path, metadata):
        if self.distributed.rank != 0:
            raise RuntimeError("Only rank zero writes shared checkpoints")
        cuda_rng = []
        if torch.cuda.is_available():
            if not self.distributed.enabled:
                cuda_rng = torch.cuda.get_rng_state_all()
            elif torch.device(self.device).type == "cuda":
                cuda_rng = [torch.cuda.get_rng_state(self.device)]
        payload = dict(
            version=checkpoint_version(self.cfg),
            imagination_protocol=imagination_protocol(self.cfg),
            geometry_spec=self.geometry_spec,
            config=asdict(self.cfg),
            noise_config=asdict(self.noise_cfg),
            metadata=metadata,
            iteration=self.iteration,
            resume_actor_ramp=validate_resume_actor_ramp(self.resume_actor_ramp, self.iteration),
            real_steps=self.real_steps,
            step_credit=self.step_credit,
            optimizer_steps=self.optimizer_steps,
            grasp_sampling=self.grasp_manager.state_dict() if self.grasp_manager else None,
            rng_cpu=torch.get_rng_state(),
            rng_cuda=cuda_rng,
        )
        for name in ("wm", "actor", "critic", "opt_wm", "opt_actor", "opt_critic"):
            payload[name] = getattr(self, name).state_dict()
        if self.cfg.dex_reference:
            payload["opt_failure"] = self.opt_failure.state_dict()
        if self.cfg.uses_target_critic:
            payload['target_critic'] = self.target_critic.state_dict()
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def restore(self, payload, metadata, restore_rng=True, allow_microbatch_override=False, allow_gamma_override=False,
                solver_iteration_override=None, allow_speed_cost_override=False):
        self._wm_capture = None
        if payload.get("version") != checkpoint_version(self.cfg):
            raise ValueError(f"Version {checkpoint_version(self.cfg)} required for this policy observation/action contract")
        if payload.get("imagination_protocol") != imagination_protocol(self.cfg):
            raise ValueError("Resume imagination protocol changed")
        if payload.get("geometry_spec") != self.geometry_spec:
            raise ValueError("Resume geometry contract changed")
        resume_schedule = validate_resume_actor_ramp(payload.get("resume_actor_ramp"), payload["iteration"])
        saved_cfg = Config(**payload["config"])
        if allow_speed_cost_override:
            if not (saved_cfg.constraints_enabled and self.cfg.constraints_enabled):
                raise ValueError('Speed-cost override requires existing cost heads')
            for name in ('cost_velocity_limit_rad_s', 'cost_target_rate_limit_rad_s'):
                value = getattr(self.cfg, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('Speed-cost thresholds must be finite and positive')
                setattr(saved_cfg, name, value)
        if allow_gamma_override:
            if self.cfg.stable_value_learning:
                raise ValueError('v29 return bounds depend on gamma; start fresh to change gamma')
            if any(payload["optimizer_steps"][name] != 0 for name in ("actor", "critic")):
                raise ValueError("Gamma override requires an untrained actor and critic")
            if not 0 < self.cfg.gamma < 1:
                raise ValueError("Gamma must be between zero and one")
            saved_cfg.gamma = self.cfg.gamma
        if allow_microbatch_override:
            # Execution partitioning only: logical batches, update counts,
            # precision, task, physics and sampling contracts remain strict.
            for name in MICROBATCH_FIELDS:
                value = getattr(self.cfg, name)
                if value is not None and (type(value) is not int or value < 1):
                    raise ValueError("Microbatch overrides must be positive integers")
                setattr(saved_cfg, name, value)
        if asdict(saved_cfg) != asdict(self.cfg) or payload["noise_config"] != asdict(self.noise_cfg):
            raise ValueError("Resume configuration changed")
        for key in ("environment_contract", "joint_names", "context_schema", "observation_protocol"):
            # Required runtime metadata; task ID is checked by the train entrypoint.
            if key == 'environment_contract' and solver_iteration_override is not None:
                from .solver_budget import validate_solver_override
                validate_solver_override(solver_iteration_override, payload['metadata'][key], metadata[key])
                continue
            if payload["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume contract changed: {key}")
        if self.cfg.compact_state:
            for key in ("state_protocol", "context_manifest", "identified_hand", "joint_limits_rad", "sdk_software_limits", "training_soft_limits"):
                if (key == "identified_hand" and solver_iteration_override is not None
                        and payload["metadata"].get(key) != metadata.get(key)):
                    from .solver_budget import validate_identified_substep_override
                    validate_identified_substep_override(payload["metadata"].get(key), metadata.get(key), solver_iteration_override)
                    continue
                if payload["metadata"].get(key) != metadata.get(key):
                    raise ValueError(f"Resume compact contract changed: {key}")
        if payload["metadata"].get("environment_binding", {}).get("assets") != metadata.get(
            "environment_binding", {}
        ).get("assets"):
            raise ValueError("Resume asset fingerprint changed")
        saved_grasps = payload.get("grasp_sampling")
        if (saved_grasps is None) != (self.grasp_manager is None):
            raise ValueError("Resume requires the same grasp bank mode")
        if saved_grasps is not None:
            self.grasp_manager.load_state_dict(saved_grasps)
        for name in ("wm", "actor", "critic", "opt_wm", "opt_actor", "opt_critic"):
            getattr(self, name).load_state_dict(payload[name])
        if self.cfg.dex_reference:
            self.opt_failure.load_state_dict(payload["opt_failure"])
        if self.cfg.uses_target_critic:
            self.target_critic.load_state_dict(payload['target_critic'], strict=True)
        for name in ("iteration", "real_steps", "step_credit", "optimizer_steps"):
            setattr(self, name, payload[name])
        self.optimizer_steps = dict(self.optimizer_steps)
        self.resume_actor_ramp = resume_schedule
        # Simulator/replay are deliberately re-warmed; this is not bitwise continuation.
        if restore_rng:
            torch.set_rng_state(payload["rng_cpu"].cpu())
            if payload["rng_cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng_cuda"]])
