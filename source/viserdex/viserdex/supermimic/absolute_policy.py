"""v30 absolute commands, physical-unit exploration and anti-saturation loss.

No target integrator, slew limiter, smoothing of the policy command or reduced
joint range is introduced. Only the additive exploration offset has memory.
The same function runs in real collection and differentiable imagination.
"""
import math
import torch


def scheduled_exploration_std_rad(cfg, iteration):
    """Cosine-anneal exploration std from exploration_position_std_rad to exploration_min_std_rad."""
    start = cfg.exploration_position_std_rad
    end = getattr(cfg, 'exploration_min_std_rad', 0.0)
    decay = getattr(cfg, 'exploration_decay_iters', 0)
    if end <= 0.0 or decay <= 0 or end >= start:
        return start
    joint_iter = max(0, int(iteration) - cfg.wm_only_iters)
    frac = min(joint_iter / decay, 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))


def explore_absolute(action, previous_noise_rad, limits, cfg, *, innovation=None, std_rad=None):
    """AR(1) angle perturbation, reflected at encoder limits without clipping.

    sigma is the stationary physical-angle standard deviation before reflection;
    rho=exp(-dt/tau). Reflection avoids a point mass at the boundary and leaves
    a nonzero local action derivative away from the reflection cusps. This does
    not bound an arbitrary change of the policy's absolute target.
    """
    if previous_noise_rad.shape != action.shape:
        raise ValueError('Exploration memory must match the absolute action shape')
    innovation = torch.randn_like(action) if innovation is None else innovation
    if innovation.shape != action.shape:
        raise ValueError('Exploration innovation must match the action shape')
    rho = math.exp(-cfg.control_dt / cfg.exploration_correlation_s)
    sigma = cfg.exploration_position_std_rad if std_rad is None else std_rad
    noise = rho * previous_noise_rad + sigma * math.sqrt(1-rho*rho) * innovation
    lower, upper = limits[..., 0], limits[..., 1]
    if getattr(cfg, 'dex_action_residual_rad', ()):
        half = action.new_tensor(cfg.dex_action_residual_rad)
    else:
        half = (upper-lower)*.5
    perturbed = action + noise/half
    # Fold the normalized coordinate into [-1,1]; inference remains plain tanh.
    reflected = 1 - torch.abs(torch.remainder(perturbed+1, 4)-2)
    return reflected, noise


def logit_regularizer(logits, cfg):
    """Zero in the usable interior; corrects extreme logits before tanh."""
    return cfg.actor_logit_penalty * torch.relu(logits.abs()-cfg.actor_logit_threshold).square().mean(-1)


def objective_gradient_metrics(actor, terms):
    """Read-only, pre-Adam gradients on one declared microbatch, before clipping."""
    params = tuple(actor.parameters())
    vectors = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                  for p, g in zip(params, grads)]).detach()
    reward, cost = vectors['reward'], vectors['cost']
    result = {f'actor_gradient_{name}_norm': vector.norm() for name, vector in vectors.items()}
    result['actor_gradient_cost_reward_ratio'] = cost.norm()/reward.norm().clamp_min(1e-12)
    result['actor_gradient_cost_reward_cosine'] = (cost@reward)/(cost.norm()*reward.norm()).clamp_min(1e-12)
    if 'reward_rollout' in vectors and 'reward_bootstrap' in vectors:
        short, tail = vectors['reward_rollout'], vectors['reward_bootstrap']
        result['actor_gradient_bootstrap_rollout_ratio'] = tail.norm()/short.norm().clamp_min(1e-12)
        result['actor_gradient_bootstrap_rollout_cosine'] = (tail@short)/(tail.norm()*short.norm()).clamp_min(1e-12)
    return result
