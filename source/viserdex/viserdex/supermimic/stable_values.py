"""v29: bounded episodic risk, real-transition TD, and a frozen target Critic.

Raw physical violations retain their original thresholds and logging. Running
risk is c/(1+c), in [0,1]. Failure ends the rollout and pays 1/(1-gamma) once
PER cost head: the exact return of an absorbing worst-case violation state.
Thus failure cannot reduce any cost by removing its future, while no dynamics
or value is evaluated on an unobserved post-failure trajectory.
"""
import torch
from torch.nn import functional as F
from .core import action_target, advance, reward, push_history
from .constraints import cost_names, violation_costs
from .state_layout import COMPACT
from .stability_reward import task_reward_bound


def value_bounds(cfg):
    return [task_reward_bound(cfg) / (1-cfg.gamma)] + [1/(1-cfg.gamma)] * len(cost_names(cfg))


def cap_values(value, bounds):
    """Apply theoretical upper bound V_max (and nonnegative lower bound on risk heads).

    Mirrors official SuperMimic's upper-bound clamp without squashing interior values
    or restricting signed reward lower bounds when penalties are active.
    """
    reward = value[..., :1].clamp_max(bounds[0])
    if value.shape[-1] == 1:
        return reward
    costs = torch.minimum(value[..., 1:].clamp_min(0.0), bounds[1:])
    return torch.cat((reward, costs), -1)


def running_risk(raw_cost):
    # Stable even for large finite raw violations; zero and the deadband remain
    # exactly zero. The drop head represents failure, not distance overshoot.
    bounded = 1 - (1 + raw_cost[..., :-1]).reciprocal()
    return torch.cat((bounded, torch.zeros_like(raw_cost[..., -1:])), -1)


def failure_risk_cost(raw_cost, failure_probability, cfg):
    p = failure_probability[..., None].to(raw_cost.dtype)
    return (1-p) * running_risk(raw_cost) + p / (1-cfg.gamma)


@torch.no_grad()
def sync_value_buffers(trainer):
    # Both networks use exactly the same real-data coordinate normalization.
    for target, current in zip(trainer.target_critic.buffers(), trainer.critic.buffers()):
        target.copy_(current)


@torch.no_grad()
def update_target(trainer):
    tau = (trainer.cfg.critic_ema_tau if trainer.cfg.uses_paper_target_critic
           else trainer.cfg.critic_target_tau)
    for target, current in zip(trainer.target_critic.parameters(), trainer.critic.parameters()):
        target.lerp_(current, tau)
    sync_value_buffers(trainer)


@torch.no_grad()
def real_td_batch(trainer, batch):
    """Use pre-reset physical targets; a time limit is NOT a failure label."""
    if not all(key in batch for key in ('state', 'next_physical', 'action', 'reward', 'terminated', 'env_id')):
        raise ValueError('v29 Critic requires complete real transition and terminated labels')
    cfg = trainer.cfg
    if cfg.critic_td_steps > 1:
        return real_multistep_td_batch(trainer, batch)
    state, physical, action = batch['state'], batch['next_physical'], batch['action']
    context, limits = trainer.context[batch['env_id']], trainer.limits[batch['env_id']]
    target = action_target(state, action, limits, cfg)
    nxt = advance(state, physical, action, target, state[:, COMPACT.goal_rot],
                  trainer.episode_steps, hold_cfg=cfg)
    terminated = batch['terminated'].bool()
    raw, _ = violation_costs(state, physical, action, limits, cfg)
    cost = failure_risk_cost(raw, terminated, cfg)
    immediate = torch.cat((batch['reward'][:, None], cost), -1)
    # Do not query the Critic on failed next states, even when its value would
    # subsequently be multiplied by zero. Reset observations never enter here.
    safe_next = torch.where(terminated[:, None], state, nxt)
    bootstrap = trainer.target_critic(safe_next, context)
    targets = immediate + cfg.gamma * (~terminated)[:, None] * bootstrap
    bounds = trainer.critic.value_bounds
    unit_targets = targets / bounds
    if not (torch.isfinite(unit_targets).all() and (unit_targets >= -1e-6).all()
            and (unit_targets <= 1+1e-6).all()):
        raise ValueError('Real TD targets violate the declared reward/risk bounds')
    return state, context, unit_targets.clamp(0, 1), cost


def real_prefix(batch):
    """Include the first ending transition, then exclude all reset-episode data."""
    done = batch['done'].bool() | batch['terminated'].bool()
    valid = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)
    return valid, valid.sum(1)


@torch.no_grad()
def real_multistep_td_batch(trainer, batch):
    """Finite real behavior return + one frozen bootstrap, never crossing reset.

    These are replay behavior returns, not importance-corrected on-policy
    estimates. Use short windows and uniform starts, not WM error priorities.
    A timeout bootstraps its PRE-reset physical next state. A true failure
    pays absorbing risk once and has no continuation value.
    """
    cfg = trainer.cfg
    state = batch['state']
    if state.ndim != 3 or state.shape[1] != cfg.critic_td_steps or 'done' not in batch:
        raise ValueError('Multi-step TD requires a complete real window with done boundaries')
    valid, length = real_prefix(batch)
    n, steps, width = state.shape
    rows = torch.arange(n, device=state.device)
    last = length-1
    context, limits = trainer.context[batch['env_id']], trainer.limits[batch['env_id']]
    # Exclude invalid tails BEFORE evaluating costs. Multiplication by a mask
    # afterward would still propagate NaN/Inf from unrelated reset episodes.
    safe_state = torch.where(valid[..., None], state, state[:, :1])
    safe_physical = torch.where(valid[..., None], batch['next_physical'], state[:, :1, :55])
    safe_action = torch.where(valid[..., None], batch['action'], 0.)
    raw, _ = violation_costs(safe_state.reshape(-1, width), safe_physical.reshape(-1, 55),
        safe_action.reshape(-1, 21), limits[:, None].expand(-1, steps, -1, -1).reshape(-1, 21, 2), cfg)
    cost = failure_risk_cost(raw.reshape(n, steps, -1), batch['terminated'].bool() & valid, cfg)
    immediate = torch.cat((batch['reward'][..., None], cost), -1)
    immediate = torch.where(valid[..., None], immediate, 0.)
    discounts = cfg.gamma ** torch.arange(steps, device=state.device, dtype=state.dtype)
    targets = (immediate * discounts[None, :, None]).sum(1)
    end_state, end_physical, end_action = state[rows, last], batch['next_physical'][rows, last], batch['action'][rows, last]
    terminated = batch['terminated'][rows, last].bool()
    target = action_target(end_state, end_action, limits, cfg)
    nxt = advance(end_state, end_physical, end_action, target, end_state[:, COMPACT.goal_rot],
                  trainer.episode_steps, hold_cfg=cfg)
    safe_next = torch.where(terminated[:, None], state[:, 0], nxt)
    bootstrap = trainer.target_critic(safe_next, context)
    targets += (cfg.gamma ** length)[:, None] * (~terminated)[:, None] * bootstrap
    unit_targets = targets / trainer.critic.value_bounds
    if not (torch.isfinite(unit_targets).all() and (unit_targets >= -1e-6).all()
            and (unit_targets <= 1+1e-6).all()):
        raise ValueError('Real multi-step TD targets violate the declared reward/risk bounds')
    return state[:, 0], context, unit_targets.clamp(0, 1), cost


def fit_real_critic(trainer, batch):
    """Soft-label BCE estimates bounded conditional means, on REAL states only.

    Targets are frozen before all epochs. Their scales are fixed by mathematical
    return bounds, never by the current batch variance. BCEWithLogits avoids a
    vanishing supervised correction when a bounded output saturates.
    """
    from .trainer import checked_step, scalar_metrics
    state, context, labels, costs = real_td_batch(trainer, batch)
    cfg = trainer.cfg
    total = torch.zeros((), dtype=torch.float64, device=state.device)
    updates = 0
    for _ in range(cfg.critic_epochs):
        for ids in torch.tensor_split(torch.randperm(len(state), device=state.device), cfg.critic_minibatches):
            trainer.opt_critic.zero_grad(set_to_none=True)
            denominator = trainer.distributed.batch_denominator(len(ids))
            for chunk in ids.split(cfg.critic_microbatch or cfg.microbatch):
                logits = trainer.critic.logits(state[chunk], context[chunk])
                loss = F.binary_cross_entropy_with_logits(logits, labels[chunk], reduction='sum') / denominator
                loss.backward()
                total += loss.detach()
            if not len(ids):
                (trainer.critic(state[:1], context[:1]).sum() * 0).backward()
            checked_step(trainer.critic, trainer.opt_critic, distributed=trainer.distributed, tensor_norm=True)
            trainer.optimizer_steps['critic'] += 1
            updates += 1
    update_target(trainer)
    with torch.no_grad():
        values = trainer.critic(state, context) / trainer.critic.value_bounds
        terminated = batch['terminated'].bool()
        timeout = batch.get('done', terminated).bool() & ~terminated
        metrics = dict(critic_loss=total/updates,
                       critic_real_normalized_mse=(values-labels).square().mean(),
                       critic_target_tau=cfg.critic_target_tau,
                       critic_td_steps=cfg.critic_td_steps)
        if cfg.critic_td_steps > 1:
            valid, length = real_prefix(batch)
            # Rates per valid real transition, not per padded/reset window.
            metrics['critic_real_mean_td_steps'] = length.float().mean()
            denominator = valid.sum().clamp_min(1)
            metrics['critic_real_terminated_fraction'] = (terminated & valid).sum()/denominator
            metrics['critic_real_timeout_fraction'] = (timeout & valid).sum()/denominator
        else:
            metrics['critic_real_terminated_fraction'] = terminated.float().mean()
        if 'done' in batch:
            if cfg.critic_td_steps == 1:
                metrics['critic_real_timeout_fraction'] = timeout.float().mean()
        for i, name in enumerate(('reward', *cost_names(cfg))):
            metrics[f'critic_value_mean_{name}'] = values[:, i].mean()
            metrics[f'critic_value_max_{name}'] = values[:, i].max()
            metrics[f'critic_target_mean_{name}'] = labels[:, i].mean()
            metrics[f'critic_td_mae_{name}'] = (values[:, i]-labels[:, i]).abs().mean()
    return scalar_metrics(metrics)


def imagine_stable(trainer, batch):
    from .sensors import observe
    from .trainer import lambda_returns
    cfg = trainer.cfg
    state, history = batch['state'], batch['history']
    sensors = {k[7:]: v for k, v in batch.items() if k.startswith('sensor/')}
    context, limits = trainer.context[batch['env_id']], trainer.limits[batch['env_id']]
    alive = torch.ones(len(state), device=state.device)
    objective = torch.zeros_like(alive)
    names = cost_names(cfg)
    running = torch.zeros(len(state), len(names), device=state.device)
    terminal = torch.zeros_like(running)
    use_lam = bool(getattr(cfg, 'actor_lambda_return', False))
    actor_value_model = trainer.critic if cfg.paper_critic else trainer.target_critic
    target_value_model = trainer.target_critic if cfg.has_target_critic else trainer.critic
    cap_bounds = state.new_tensor(value_bounds(cfg)) if cfg.critic_value_cap else None
    init_state = state
    post_states, post_alives = [], []
    immediate, continuations, valid, alive_history = [], [], [], []
    regularization = torch.zeros_like(alive)
    if cfg.absolute_policy_optimization:
        from .absolute_policy import explore_absolute, logit_regularizer
        noise = batch['exploration_noise_rad']
    for t in range(cfg.horizon):
        active = alive > 0
        alive_history.append(alive.detach())
        if cfg.absolute_policy_optimization:
            logits = trainer.actor.logits(history)
            if cfg.deterministic_imagination:
                action, next_noise = logits.tanh(), noise
            else:
                action, next_noise = explore_absolute(logits.tanh(), noise, limits, cfg)
            # On real replay start states, always give extreme logits a direct
            # corrective gradient, even if the WM predicts an immediate drop.
            if t == 0:
                regularization = logit_regularizer(logits, cfg)
        else:
            action = trainer.actor(history)
        target = action_target(state, action, limits, cfg,
                               home=trainer._wm_home, residual_rad=trainer._wm_action_residual)
        nxt = trainer.wm(state, target, context)
        r = reward(state, nxt, action, cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight,
                   cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight, hold_cfg=cfg,
                   limits=limits, encoder_bias=trainer._wm_encoder_bias, home=trainer._wm_home,
                   residual_rad=trainer._wm_action_residual)
        raw, _ = violation_costs(state, nxt, action, limits, cfg,
                                 encoder_bias=trainer._wm_encoder_bias, hand_center=trainer._wm_hand_center,
                                 home=trainer._wm_home, residual_rad=trainer._wm_action_residual)
        continuation = torch.where(active, trainer.continuation(state, nxt), 0.)
        p = 1-continuation
        step_discount = (cfg.gamma * cfg.lam) ** t if use_lam else cfg.gamma ** t
        discount_alive = step_discount * alive
        objective = objective + discount_alive * r
        running = running + discount_alive[:, None] * continuation[:, None] * running_risk(raw)
        terminal = terminal + discount_alive[:, None] * p[:, None] / (1-cfg.gamma)
        cost = failure_risk_cost(raw, p, cfg)
        alive = alive * continuation
        advanced = advance(state, nxt, action, target, state[:, COMPACT.goal_rot], trainer.episode_steps, hold_cfg=cfg)
        # A known failure freezes this sample at its last valid state. Future
        # contributions are zero; no post-drop imagined state feeds the models.
        survives = alive > 0
        if cfg.absolute_policy_optimization:
            noise = torch.where(survives[:, None], next_noise, noise)
        state = torch.where(survives[:, None], advanced, state)
        obs, next_sensors = observe(state, sensors, limits, trainer.noise_cfg, cfg)
        history = torch.where(survives[:, None, None], push_history(history, obs), history)
        sensors = {k: torch.where(survives.reshape(-1, *([1]*(v.ndim-1))), v, sensors[k])
                   for k, v in next_sensors.items()}
        post_states.append(state)
        post_alives.append(alive)
        immediate.append(torch.cat((r[:, None], cost), -1) * active[:, None])
        continuations.append(continuation[:, None].expand(-1, 1+len(names)))
        valid.append(active)
    # Single fused Critic forward over all H post-transition imagined states (exact equivalence, 8x fewer launches).
    H, B = cfg.horizon, len(context)
    stacked_post = torch.stack(post_states)
    flat_post = stacked_post.reshape(H * B, -1)
    flat_ctx = context[None].expand(H, -1, -1).reshape(H * B, -1)
    all_raw_values = actor_value_model(flat_post, flat_ctx).reshape(H, B, -1)
    all_values = cap_values(all_raw_values, cap_bounds) if cap_bounds is not None else all_raw_values
    raw_value = all_raw_values[-1]
    value = all_values[-1]
    reward_rollout = (1-cfg.gamma) * objective.mean() / task_reward_bound(cfg)
    if use_lam:
        alives_t = torch.stack(post_alives)
        t_idx = torch.arange(H, device=state.device, dtype=state.dtype)
        boot_w = (cfg.gamma ** (t_idx + 1)) * (cfg.lam ** t_idx) * torch.where(t_idx == H - 1, 1.0, 1.0 - cfg.lam)
        w_alive = boot_w[:, None] * alives_t
        reward_boot_acc = (w_alive * all_values[..., 0]).sum(0)
        cost_boot_acc = (w_alive[..., None] * all_values[..., 1:]).sum(0)
        reward_bootstrap = (1-cfg.gamma) * reward_boot_acc.mean() / task_reward_bound(cfg)
        bootstrap = cost_boot_acc
    else:
        reward_bootstrap = (1-cfg.gamma) * (cfg.gamma**cfg.horizon * alive * value[:, 0]).mean() / task_reward_bound(cfg)
        bootstrap = cfg.gamma**cfg.horizon * alive[:, None] * value[:, 1:]
    pieces = [(1-cfg.gamma)*part.mean(0) for part in (running, terminal, bootstrap)]
    normalized = sum(pieces)
    loss = -reward_rollout-reward_bootstrap + cfg.cost_penalty*normalized.sum()
    if cfg.absolute_policy_optimization:
        trainer._actor_objective_terms = dict(reward=-reward_rollout-reward_bootstrap,
            cost=cfg.cost_penalty*normalized.sum(), regularizer=regularization.mean())
        trainer._actor_reward_components = dict(reward_rollout=-reward_rollout, reward_bootstrap=-reward_bootstrap)
        loss = loss + regularization.mean()
    with torch.no_grad():
        if target_value_model is actor_value_model:
            init_v = actor_value_model(init_state, context)
            if cap_bounds is not None:
                init_v = cap_values(init_v, cap_bounds)
            all_target_values = torch.cat((init_v[None], all_values.detach()), dim=0)
        else:
            all_s = torch.cat((init_state[None], stacked_post.detach()), dim=0)
            flat_all_s = all_s.reshape((H + 1) * B, -1)
            flat_all_c = context[None].expand(H + 1, -1, -1).reshape((H + 1) * B, -1)
            tv = target_value_model(flat_all_s, flat_all_c).reshape(H + 1, B, -1)
            all_target_values = cap_values(tv, cap_bounds) if cap_bounds is not None else tv
        targets = lambda_returns(torch.stack(immediate), torch.stack(continuations),
                                 all_target_values, cfg.gamma, cfg.lam)
        if cap_bounds is not None:
            targets = cap_values(targets, cap_bounds)
    states_for_critic = torch.cat((init_state[None], stacked_post[:-1]), dim=0).detach()
    metrics = dict(actor_reward_rollout=reward_rollout.detach(), actor_reward_bootstrap=reward_bootstrap.detach(),
                   imagined_alive_mean=torch.stack(alive_history).mean(),
                   imagined_terminal_probability=(1-alive).mean().detach(),
                   actor_cost_bootstrap_fraction=(pieces[2].sum()/normalized.sum().clamp_min(1e-12)).detach())
    if cap_bounds is not None:
        metrics['critic_reward_cap_hit_fraction'] = (raw_value[:, 0] >= cap_bounds[0]).float().mean().detach()
        metrics['critic_cost_cap_hit_fraction'] = (raw_value[:, 1:] >= cap_bounds[1:]).float().mean().detach()
    for i, name in enumerate(names):
        metrics[f'imagined_cost_{name}'] = normalized[i].detach()
        for label, piece in zip(('running', 'terminal', 'bootstrap'), pieces):
            metrics[f'actor_cost_{label}_{name}'] = piece[i].detach()
    if cfg.absolute_policy_optimization:
        metrics.update(actor_logit_regularization=regularization.mean().detach(),
                       actor_cost_weight=cfg.cost_penalty,
                       actor_weighted_cost=(cfg.cost_penalty*normalized.sum()).detach())
    trainer._cost_metrics = metrics
    return loss, states_for_critic, targets.detach(), context, torch.stack(valid)
