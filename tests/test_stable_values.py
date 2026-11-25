"""v29 Bellman bounds, failure semantics, real TD provenance and gradients."""
from dataclasses import replace
import copy
import math
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from viserdex.supermimic.trainer import checkpoint_version, frozen
from viserdex.supermimic.sensors import initial_sensors, observe
from viserdex.supermimic.stable_values import (
    running_risk, failure_risk_cost, value_bounds, real_td_batch,
    fit_real_critic, update_target, sync_value_buffers,
)
from test_wm_task_supervision import setup


def fixture(n=4, **kwargs):
    tr, state, limits = setup(n, stable_value_learning=True, **kwargs)
    sensors = initial_sensors(n, tr.device, tr.noise_cfg)
    obs, sensors = observe(state, sensors, limits, tr.noise_cfg, tr.cfg)
    batch = dict(state=state, history=obs[:, None].expand(-1, 10, -1).clone(),
                 next_physical=state[:, :55].clone(), action=torch.zeros(n, 21, device=state.device),
                 reward=torch.full((n,), .4, device=state.device),
                 terminated=torch.zeros(n, dtype=torch.bool, device=state.device),
                 done=torch.zeros(n, dtype=torch.bool, device=state.device),
                 env_id=torch.arange(n, device=state.device))
    batch.update({'sensor/'+k:v for k,v in sensors.items()})
    return tr, batch


def test_bounded_risk_retains_deadband_and_monotonic_gradient():
    raw = torch.tensor([[0., 1., 3., 1e12, 6.]], requires_grad=True)
    bounded = running_risk(raw)
    torch.testing.assert_close(bounded, torch.tensor([[0., .5, .75, 1., 0.]]))
    bounded.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad[0, :4] > 0).all() and raw.grad[0, 4] == 0


@pytest.mark.parametrize('gamma', [.9, .99, .999])
def test_failure_is_paid_once_and_cannot_buy_cheaper_cost_return(gamma):
    tr, _ = fixture(gamma=gamma)
    raw = torch.tensor([[0., .3, 3., 1e6, 0.]]).expand(9, -1)
    p = torch.linspace(0, 1, 9).requires_grad_()
    maximum = 1/(1-gamma)
    cost = failure_risk_cost(raw, p, tr.cfg)
    # Even the largest admissible future value cannot make failing cheaper.
    q = cost + gamma*(1-p[:, None])*maximum
    assert (q <= maximum+1e-3).all()
    q.sum().backward()
    assert (p.grad >= -1e-3).all()
    torch.testing.assert_close(cost[-1], torch.full((5,), maximum))


def test_real_td_uses_pre_reset_state_and_terminated_not_timeout():
    tr, b = fixture()
    b['terminated'][0] = True
    b['done'][:2] = True  # index 1 is a timeout
    b['next_physical'][:, 42] += .02
    seen = []
    class Fixed(nn.Module):
        def forward(self, state, context):
            seen.append(state.clone())
            return state.new_tensor([4., 20., 20., 20., 20., 20.]).expand(len(state), -1)
    tr.target_critic = Fixed()
    _, _, targets, _ = real_td_batch(tr, b)
    bounds = tr.critic.value_bounds
    torch.testing.assert_close(targets[0, 1:], torch.ones(5))
    assert targets[0, 0] == pytest.approx(.4/bounds[0].item())
    torch.testing.assert_close(targets[1:, 1:], torch.full((3, 5), .99*20/100))
    torch.testing.assert_close(seen[0][0], b['state'][0])
    torch.testing.assert_close(seen[0][1:, :55], b['next_physical'][1:])
    b.pop('terminated')
    with pytest.raises(ValueError, match='terminated'):
        real_td_batch(tr, b)


def test_target_is_independent_frozen_and_ema_buffers_use_real_coordinates():
    tr, b = fixture()
    assert tr.target_critic is not tr.critic
    assert all(not p.requires_grad for p in tr.target_critic.parameters())
    before = [p.clone() for p in tr.target_critic.parameters()]
    with torch.no_grad():
        for p in tr.critic.parameters(): p.add_(1.)
        tr.critic.input_norm.mean.fill_(.7)
    update_target(tr)
    for old, target in zip(before, tr.target_critic.parameters()):
        torch.testing.assert_close(target, old+tr.cfg.critic_target_tau)
    torch.testing.assert_close(tr.target_critic.input_norm.mean, tr.critic.input_norm.mean)


def test_extreme_logits_stay_bounded_and_have_corrective_bce_gradients():
    tr, b = fixture()
    with torch.no_grad():
        tr.critic.net[-1].bias.copy_(torch.tensor([-100., 100., -100., 100., -100., 100.]))
    v = tr.critic(b['state'], tr.context)
    assert (v >= 0).all() and (v <= tr.critic.value_bounds).all()
    logits = tr.critic.logits(b['state'], tr.context)
    F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, .5)).backward()
    grad = tr.critic.net[-1].bias.grad
    assert (grad.abs() > .01).all()
    assert torch.equal(grad.sign(), tr.critic.net[-1].bias.sign())


def test_critic_fits_real_data_without_world_model_or_actor_updates():
    tr, b = fixture(learning_rate=.003, critic_target_tau=.1)
    before_actor = copy.deepcopy(tr.actor.state_dict())
    before_wm = copy.deepcopy(tr.wm.state_dict())
    initial = tr.critic(b['state'], tr.context).detach()/tr.critic.value_bounds
    for _ in range(60): metrics = fit_real_critic(tr, b)
    final = tr.critic(b['state'], tr.context).detach()/tr.critic.value_bounds
    assert final[:, 0].mean() > initial[:, 0].mean()
    assert final[:, 1:].mean() < initial[:, 1:].mean()
    assert math.isfinite(metrics['critic_loss'])
    assert tr.optimizer_steps == dict(wm=0, actor=0, critic=60)
    for model, saved in ((tr.actor, before_actor), (tr.wm, before_wm)):
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, saved[k], atol=0, rtol=0)
    assert all(p.grad is None for p in tr.target_critic.parameters())


def test_known_drop_overrides_false_negative_and_never_propagates_dropped_state():
    tr, b = fixture()
    seen = []
    center = b['state'][:, 42:45].clone()
    class Drops(nn.Module):
        def forward(self, state, target, context):
            seen.append(state[:, 42:45].clone())
            nxt = state[:, :55].clone()
            nxt[:, 42] += .31
            return nxt
        def termination_probability(self, physical):
            return physical.new_zeros(len(physical))
    tr.wm = Drops()
    loss, _, targets, _, valid = tr.imagine(b)
    assert math.isfinite(loss.item()) and loss.item() <= 5.
    assert valid[0].all() and not valid[1:].any()
    for positions in seen: torch.testing.assert_close(positions, center)
    for name in ('joint_speed','target_rate','object_angular_speed','object_linear_speed','drop'):
        assert tr._cost_metrics['imagined_cost_'+name] == pytest.approx(1.)
        assert tr._cost_metrics['actor_cost_terminal_'+name] == pytest.approx(1.)
        assert tr._cost_metrics['actor_cost_bootstrap_'+name] == 0
    torch.testing.assert_close(targets[0, :, 1:], torch.full((4, 5), 100.))


def test_full_actor_gradients_and_objective_decomposition():
    tr, b = fixture()
    torch.nn.init.normal_(tr.wm.net[-1].weight, std=.005)
    torch.nn.init.normal_(tr.target_critic.net[-1].weight, std=.01)
    with frozen(tr.wm, tr.critic):
        loss, _, targets, _, _ = tr.imagine(b)
        loss.backward()
    assert -1-1e-6 <= loss.item() <= 5+1e-6
    assert not targets.requires_grad
    assert sum(p.grad.abs().sum().item() for p in tr.actor.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for m in (tr.wm,tr.critic,tr.target_critic) for p in m.parameters())
    m = tr._cost_metrics
    cost = sum(v for k,v in m.items() if k.startswith('imagined_cost_'))
    torch.testing.assert_close(loss.detach(), cost-m['actor_reward_rollout']-m['actor_reward_bootstrap'])
    for name in ('joint_speed','target_rate','object_angular_speed','object_linear_speed','drop'):
        parts = sum(m[f'actor_cost_{part}_{name}'] for part in ('running','terminal','bootstrap'))
        torch.testing.assert_close(parts, m['imagined_cost_'+name])


def test_v29_roundtrip_retains_target_network_and_rejects_v28_resume(tmp_path):
    tr, b = fixture()
    fit_real_critic(tr, b)
    metadata=dict(environment_contract='test', joint_names=[], context_schema={}, observation_protocol='test')
    tr.save(tmp_path/'model.pt', metadata)
    payload=torch.load(tmp_path/'model.pt', weights_only=False)
    new, _=fixture()
    new.restore(payload, metadata, restore_rng=False)
    assert payload['version'] == checkpoint_version(tr.cfg) == 29
    assert new.optimizer_steps == tr.optimizer_steps
    for k, v in tr.target_critic.state_dict().items(): torch.testing.assert_close(v, new.target_critic.state_dict()[k])
    old=copy.deepcopy(payload);old['version']=28
    with pytest.raises(ValueError, match='Version 29'):
        new.restore(old,metadata,restore_rng=False)
    del payload['target_critic']
    with pytest.raises(KeyError, match='target_critic'):
        new.restore(payload,metadata,restore_rng=False)


def test_warmup_updates_world_model_and_real_critic_but_keeps_actor_frozen():
    tr, b = fixture()
    from test_wm_task_supervision import window
    w=window(tr,b['state'])
    class Buffer:
        archive_size=0
        def sample(self,batch,window=1,**kwargs): return w if window>1 else b
        def update_priorities(self,*args): return 0
    before=copy.deepcopy(tr.actor.state_dict())
    metrics=tr.update(Buffer())
    assert tr.optimizer_steps['wm'] == tr.cfg.wm_updates
    assert tr.optimizer_steps['critic'] == 1 and tr.optimizer_steps['actor'] == 0
    assert math.isfinite(metrics['critic_loss'])
    for k,v in tr.actor.state_dict().items(): torch.testing.assert_close(v,before[k])


@pytest.mark.parametrize('kwargs', [dict(wm_task_losses=False),dict(object_hold_costs_enabled=False),
                                    dict(critic_target_tau=0),dict(critic_target_tau=float('nan'))])
def test_invalid_stable_value_contract_rejected(kwargs):
    tr,_=fixture()
    with pytest.raises(ValueError): replace(tr.cfg,**kwargs)
