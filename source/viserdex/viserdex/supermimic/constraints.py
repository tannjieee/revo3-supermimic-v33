"""Dimensionless violation costs for constrained differentiable rollouts.

This borrows N-P3O's reward/cost separation, not its PPO likelihood-ratio loss.
Budgets are zero normalized discounted violation. This is a soft learned
constraint, not a per-tick hardware speed guarantee.
"""
import torch
from .core import action_target, rotation_log, quat_mul, quat_conjugate, ROT, VEL, OMEGA
from .state_layout import COMPACT
from .stability_reward import task_reward_bound
from .spatial_contract import drop_distance

COST_NAMES = ('joint_speed', 'target_rate')
OBJECT_COST_NAMES = ('object_angular_speed', 'object_linear_speed', 'drop')


def cost_names(cfg):
    if getattr(cfg, 'compact_critic_costs', False) and cfg.object_hold_costs_enabled:
        return ('drop',)
    return COST_NAMES + OBJECT_COST_NAMES if cfg.object_hold_costs_enabled else COST_NAMES


def object_hold_gate(state, next_physical, cfg):
    """Continuous orientation proximity, independent of clocks and success latches."""
    angle = rotation_log(quat_mul(next_physical[..., ROT],
                                 quat_conjugate(state[..., COMPACT.goal_rot]))).norm(dim=-1)
    return torch.exp(-(angle / cfg.cost_object_hold_angle_sigma_rad).square())


def violation_costs(state, next_physical, action, limits, cfg, *, encoder_bias=None, hand_center=None, home=None, residual_rad=None):
    if getattr(cfg, 'compact_critic_costs', False) and cfg.object_hold_costs_enabled:
        drop_ratio = (drop_distance(next_physical, cfg, origin=hand_center) / cfg.drop_radius_m)[..., None]
        return torch.relu(drop_ratio - 1).square(), drop_ratio
    target = action_target(state, action, limits, cfg, home=home, residual_rad=residual_rad)
    rate_target = target
    previous = action_target(state, state[..., COMPACT.last_action], limits, cfg, home=home, residual_rad=residual_rad)
    # An empty reset queue has no previous policy command: use the measured
    # encoder position as the first target-rate reference, as in playback.
    bias = state.new_tensor(cfg.encoder_bias_rad) if encoder_bias is None else encoder_bias
    initial = state[..., :21] - bias
    if cfg.dex_reference:
        from .dex_reference import EMA, ZERO, unclipped_target
        previous = state[..., EMA]-state[..., ZERO]
        if getattr(cfg, 'target_rate_pre_slew_cost', False) and cfg.target_slew_rate_rad_s:
            # Penalize the pre-slew EMA command intent so hard slew clamping does not zero gradients.
            res_ref = residual_rad if residual_rad is not None else (cfg.dex_action_residual_rad or None)
            home_ref = (home if home is not None else cfg.action_home) if res_ref is not None else None
            rate_target = unclipped_target(
                state, action, limits, cfg.ema_alpha, home=home_ref, residual_rad=res_ref,
            )
    else:
        previous = torch.where(state[..., COMPACT.delay_ready].bool(), previous, initial)
    velocity_ratio = next_physical[..., 21:42].abs() / cfg.cost_velocity_limit_rad_s
    target_ratio = ((rate_target-previous) / cfg.control_dt).abs() / cfg.cost_target_rate_limit_rad_s
    ratios = torch.stack((velocity_ratio.amax(-1), target_ratio.amax(-1)), -1)
    costs = torch.relu(ratios-1).square()
    if cfg.object_hold_costs_enabled:
        object_ratios = torch.stack((next_physical[..., OMEGA].norm(dim=-1) / cfg.hold_speed_threshold,
                                    next_physical[..., VEL].norm(dim=-1) / cfg.hold_linear_speed_threshold_m_s,
                                    drop_distance(next_physical, cfg, origin=hand_center) / cfg.drop_radius_m), -1)
        # Returning these raw ratios keeps actual threshold exceedance visible
        # in logs even while the orientation-dependent speed weight is small.
        raw_costs = torch.relu(object_ratios-1).square()
        gate = object_hold_gate(state, next_physical, cfg)
        weights = torch.stack((gate, gate, torch.ones_like(gate)), -1)
        weighted = weights * raw_costs
        if cfg.object_motion_angular_speed_rad_s:
            moving_ratios = torch.stack((next_physical[..., OMEGA].norm(dim=-1) / cfg.object_motion_angular_speed_rad_s,
                                        next_physical[..., VEL].norm(dim=-1) / cfg.object_motion_linear_speed_m_s), -1)
            # Full-time moving limits plus the tighter existing near-goal limits.
            # Reuse the same two cost heads without double-counting a violation.
            moving_cost = torch.relu(moving_ratios - 1).square()
            weighted = torch.cat((torch.maximum(weighted[..., :2], moving_cost), weighted[..., 2:]), -1)
        costs = torch.cat((costs, weighted), -1)
        ratios = torch.cat((ratios, object_ratios), -1)
    return costs, ratios


def imagine_constrained(trainer, batch):
    from .core import reward, advance, push_history
    from .sensors import observe
    from .trainer import imagined_continuation, lambda_returns
    cfg = trainer.cfg
    state, history = batch['state'], batch['history']
    sensors = {k[7:]: v for k,v in batch.items() if k.startswith('sensor/')}
    context, limits = trainer.context[batch['env_id']], trainer.limits[batch['env_id']]
    states, values = [state], [trainer.critic(state,context).detach()]
    rewards, continuations = [], []
    alive = torch.ones(len(state),device=state.device)
    objective = torch.zeros_like(alive)
    names = cost_names(cfg)
    cost_return = torch.zeros(len(state),len(names),device=state.device)
    for t in range(cfg.horizon):
        action = trainer.actor(history)
        target = action_target(state,action,limits,cfg)
        nxt = trainer.wm(state,target,context)
        r = reward(state,nxt,action,cfg.control_dt,cfg.success_bonus,cfg.action_rate_weight,
                   cfg.joint_vel_weight,cfg.action_l2_weight,cfg.action_second_diff_weight,hold_cfg=cfg)
        costs,_ = violation_costs(state,nxt,action,limits,cfg)
        continuation = trainer.continuation(state,nxt)
        objective = objective + cfg.gamma**t * alive * r
        # Do not let predicted drops erase violations. Cost rollout and
        # cost bootstrap continue independently of reward survival weighting.
        cost_return = cost_return + cfg.gamma**t * costs
        alive = alive * continuation
        state = advance(state,nxt,action,target,state[:,COMPACT.goal_rot],trainer.episode_steps,hold_cfg=cfg)
        obs,sensors = observe(state,sensors,limits,trainer.noise_cfg,cfg)
        history = push_history(history,obs)
        value = trainer.critic(state,context)
        rewards.append(torch.cat((r[:,None],costs),-1))
        continuations.append(torch.cat((continuation[:,None],torch.ones_like(costs)),-1))
        states.append(state);values.append(value.detach())
    objective = objective + cfg.gamma**cfg.horizon * alive * value[:,0]
    cost_return = cost_return + cfg.gamma**cfg.horizon * value[:,1:]
    # Fixed physical scales, never subtract a batch mean from the violation.
    # The normalized discounted budget is zero for every nonnegative cost.
    normalized_cost = (1-cfg.gamma)*cost_return.mean(0)
    reward_bound = task_reward_bound(cfg)
    loss = -(1-cfg.gamma)*objective.mean()/reward_bound + cfg.cost_penalty*normalized_cost.relu().sum()
    with torch.no_grad():
        targets = lambda_returns(torch.stack(rewards),torch.stack(continuations),torch.stack(values),cfg.gamma,cfg.lam)
    trainer._cost_metrics = {f'imagined_cost_{name}': value.detach() for name,value in zip(names,normalized_cost)}
    valid = torch.ones(cfg.horizon,len(state),dtype=torch.bool,device=state.device)
    return loss,torch.stack(states[:-1]).detach(),targets.detach(),context,valid
