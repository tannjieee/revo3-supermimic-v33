"""Preserve task reward and all five v32 bounded risk definitions in real TD."""
import torch
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.stable_values import failure_risk_cost
from viserdex.supermimic.stability_reward import task_reward_bound


def real_transition(record, context, encoder, limits, cfg):
    nxt = torch.cat((record['next_physical'], record['next_aux']), -1)
    if not torch.equal(nxt[:, 76:83], record['state'][:, 76:83]):
        raise ValueError('Reset goal leaked into terminal state')
    if not torch.equal(nxt[:, 55:76], record['action']) or not (nxt[:, 83] == 1).all():
        raise ValueError('Pre-reset command or delay-ready lost')
    raw, ratios = violation_costs(record['state'], record['next_physical'], record['action'], limits, cfg)
    risks = failure_risk_cost(raw, record['terminated'], cfg)
    reward = record['reward']/task_reward_bound(cfg)-cfg.cost_penalty*risks.sum(-1)
    transition = dict(observation=encoder(record['history'], record['state'], context).detach(),
        next_observation=encoder(record['next_history'], nxt, context).detach(),
        action=record['action'].detach(), reward=reward.detach(),
        terminated=record['terminated'].detach(), truncated=record['truncated'].detach())
    return transition, raw, risks, ratios
