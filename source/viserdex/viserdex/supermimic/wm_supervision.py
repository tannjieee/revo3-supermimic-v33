"""Task-relevant WM supervision on pre-reset physical targets.

Reward/observation/cost are exact differentiable readouts, not extra learned
approximations. Their consistency losses train dynamics; termination has a
learned head. The analytic-readout losses extend SuperMimic Eq. 1, which omits
losses for analytic heads. Sensor noise and time-limit truncation are not labels.
"""
import torch
from torch.nn import functional as F
from .core import Q, QD, POS, ROT, reward
from .state_layout import COMPACT
from .policy_contract import direct_observation
from .constraints import violation_costs, cost_names
from .stability_reward import task_reward_bound

PROTOCOL = 'analytic_task_consistency_learned_termination_v1'


def clean_next_observation(state, physical, action, limits, cfg, *, encoder_bias=None):
    if cfg.dex_reference:
        from .dex_reference import ZERO, observation
        from .core import action_target, advance
        target = action_target(state, action, limits, cfg)
        nxt = advance(state, physical, action, target, state[..., COMPACT.goal_rot], hold_cfg=cfg)
        joints = 2*(physical[..., Q]+state[..., ZERO]-limits[..., 0])/(limits[..., 1]-limits[..., 0])-1
        return observation(joints, nxt, limits, physical[..., POS], physical[..., ROT])
    bias = physical.new_tensor(cfg.encoder_bias_rad) if encoder_bias is None else encoder_bias
    joints = 2 * (physical[..., Q] - bias - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1
    return direct_observation(joints, physical[..., QD], action, state[..., COMPACT.goal_rot],
                              physical[..., POS], physical[..., ROT], state[..., COMPACT.goal_pos],
                              velocity_scale=cfg.joint_velocity_scale,
                              include_object_position=cfg.include_object_position)


def task_reward(state, physical, action, cfg, *, limits=None, encoder_bias=None, home=None):
    return reward(state, physical, action, cfg.control_dt, cfg.success_bonus,
                  cfg.action_rate_weight, cfg.joint_vel_weight, cfg.action_l2_weight,
                  cfg.action_second_diff_weight, hold_cfg=cfg, limits=limits, encoder_bias=encoder_bias, home=home)


def loss_names(cfg):
    return ('reward', 'observation', 'termination', 'cost',
            *('cost_' + name for name in cost_names(cfg)), *(('contact',) if cfg.dex_reference else ()))


def task_losses(trainer, state, prediction, real_state, real_next, action, limits, real_reward, terminated):
    """Per-sample losses; replay masking/importance weights are applied by caller."""
    cfg = trainer.cfg
    with torch.set_grad_enabled(torch.is_grad_enabled() and cfg.wm_reward_loss_weight > 0):
        predicted_reward = task_reward(state, prediction, action, cfg,
            limits=limits, encoder_bias=trainer._wm_encoder_bias, home=trainer._wm_home)
        reward_error = ((predicted_reward - real_reward) / task_reward_bound(cfg)).square()
    with torch.set_grad_enabled(torch.is_grad_enabled() and cfg.wm_observation_loss_weight > 0):
        predicted_obs = clean_next_observation(state, prediction, action, limits, cfg, encoder_bias=trainer._wm_encoder_bias)
        actual_obs = clean_next_observation(real_state, real_next, action, limits, cfg, encoder_bias=trainer._wm_encoder_bias)
        # Fixed Actor affine in both paths; no sensor sampling or learned norm drift.
        obs_error = (trainer.actor.scaled_observation(predicted_obs) -
                     trainer.actor.scaled_observation(actual_obs)).square().mean(-1)
    with torch.set_grad_enabled(torch.is_grad_enabled() and cfg.wm_cost_loss_weight > 0):
        constants = dict(encoder_bias=trainer._wm_encoder_bias, hand_center=trainer._wm_hand_center)
        predicted_cost, _ = violation_costs(state, prediction, action, limits, cfg, **constants)
        actual_cost, _ = violation_costs(real_state, real_next, action, limits, cfg, **constants)
        # Costs are dimensionless but unbounded. Log1p balances large violations
        # without clipping targets or changing the actor/critic cost definition.
        cost_errors = (torch.log1p(predicted_cost) - torch.log1p(actual_cost)).square()
        cost_error = cost_errors.mean(-1)
    terminal_error = (torch.zeros_like(reward_error) if cfg.dex_reference else
        F.binary_cross_entropy_with_logits(
            trainer.wm.termination_logits(prediction), terminated.float(), reduction='none'))
    total = (
        (cfg.wm_reward_loss_weight * reward_error if cfg.wm_reward_loss_weight else 0.0)
        + (cfg.wm_observation_loss_weight * obs_error if cfg.wm_observation_loss_weight else 0.0)
        + (cfg.wm_termination_loss_weight * terminal_error if cfg.wm_termination_loss_weight and not cfg.dex_reference else 0.0)
        + (cfg.wm_cost_loss_weight * cost_error if cfg.wm_cost_loss_weight else 0.0)
    )
    if not isinstance(total, torch.Tensor):
        total = torch.zeros_like(reward_error)
    components = torch.cat((torch.stack((reward_error, obs_error, terminal_error, cost_error), -1),
                            cost_errors), -1)
    if cfg.dex_reference:
        components = torch.cat((components, torch.zeros_like(components[..., :1])), -1)
    return total, components
