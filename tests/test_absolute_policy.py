"""Physical-unit exploration, absolute output contract, gradients and resume."""
from dataclasses import replace
import copy
import math
import pytest
import torch
from viserdex.supermimic.absolute_policy import explore_absolute, logit_regularizer, objective_gradient_metrics
from viserdex.supermimic.core import action_target
from viserdex.supermimic.trainer import checkpoint_version, frozen
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.replay import Replay
from test_stable_values import fixture as stable_fixture


def fixture(n=4, **kwargs):
    tr,b=stable_fixture(n,absolute_policy_optimization=True,cost_penalty=.05,**kwargs)
    b['exploration_noise_rad']=torch.zeros(n,21)
    return tr,b


def test_fresh_absolute_home_matches_physical_home_including_cmp_bias():
    tr,b=fixture()
    for h in (torch.zeros_like(b['history']),torch.randn_like(b['history'])*10):
        a=tr.actor(h)
        target=action_target(b['state'],a,tr.limits,tr.cfg)
        expected=torch.tensor(tr.cfg.action_home)-torch.tensor(tr.cfg.encoder_bias_rad)
        torch.testing.assert_close(target,expected.expand_as(target),atol=2e-7,rtol=0)
        assert a.abs().max()<.98
    assert checkpoint_version(tr.cfg)==30
    assert tr.actor.net[-1].weight.count_nonzero()==0


def test_no_noise_preserves_absolute_target_and_reflection_remains_bounded():
    tr,b=fixture()
    a=torch.tensor([-.9,.9,0.,.5])[:,None].expand(4,21).clone().requires_grad_()
    sampled,_=explore_absolute(a,torch.zeros_like(a),tr.limits,tr.cfg,innovation=torch.zeros_like(a))
    torch.testing.assert_close(sampled,a,atol=2e-7,rtol=0)
    sampled.sum().backward();torch.testing.assert_close(a.grad,torch.ones_like(a))
    extreme=torch.full_like(a,100.)
    reflected,_=explore_absolute(a,extreme,tr.limits,tr.cfg,innovation=extreme)
    assert reflected.abs().max()<=1
    grad,=torch.autograd.grad(reflected.sum(),a)
    torch.testing.assert_close(grad.abs(),torch.ones_like(grad))


def test_physical_noise_scale_temporal_correlation_and_rate_conflict():
    tr,_=fixture();torch.manual_seed(440)
    n=1024;limits=tr.limits[:1].expand(n,-1,-1);a=torch.zeros(n,21);noise=torch.zeros_like(a)
    hist=[]
    for i in range(180):
        out,noise=explore_absolute(a,noise,limits,tr.cfg)
        if i>=20:hist.append(out)
    angles=torch.stack(hist)*(limits[...,1]-limits[...,0])/2
    sigma=angles.std().item();rho=math.exp(-tr.cfg.control_dt/tr.cfg.exploration_correlation_s)
    empirical=(angles[1:]*angles[:-1]).mean()/angles.square().mean()
    assert sigma==pytest.approx(tr.cfg.exploration_position_std_rad,rel=.025)
    assert empirical.item()==pytest.approx(rho,abs=.01)
    violations=((angles[1:]-angles[:-1]).abs().amax(-1)>tr.cfg.cost_target_rate_limit_rad_s*tr.cfg.control_dt).float().mean()
    assert violations<1e-4


def test_logit_regularization_has_direct_corrective_gradient_at_saturation():
    tr,_=fixture();z=torch.tensor([[-8.,8.,-1.,1.]+[0.]*17],requires_grad=True)
    loss=logit_regularizer(z,tr.cfg).mean();loss.backward()
    assert z.grad[0,0]<0 and z.grad[0,1]>0
    assert z.grad[0,2:].count_nonzero()==0
    assert z.grad.abs().max()>.01


def test_replay_preserves_noise_memory_and_realized_absolute_action():
    tr,b=fixture();b.pop('env_id');b['exploration_noise_rad'].fill_(.003)
    noise=b['exploration_noise_rad'].clone()
    b['action'],_=explore_absolute(tr.actor(b['history']),noise,tr.limits,tr.cfg,innovation=torch.ones_like(noise))
    replay=Replay(4,4,'cpu',event_capacity=0)
    replay.add(b);s=replay.sample(16,include_events=False)
    torch.testing.assert_close(s['exploration_noise_rad'],torch.full((16,21),.003))
    torch.testing.assert_close(s['action'],b['action'][s['env_id']])


def test_closed_loop_gradients_frozen_models_and_diagnostic_do_not_mutate_grads():
    tr,b=fixture();torch.nn.init.normal_(tr.wm.net[-1].weight,std=.01)
    with frozen(tr.wm,tr.critic):
        loss,*_=tr.imagine(b)
        parts=tr._actor_objective_terms
        torch.testing.assert_close(loss,sum(parts.values()))
        torch.testing.assert_close(parts['reward'],sum(tr._actor_reward_components.values()))
        metrics=objective_gradient_metrics(tr.actor,{**parts,**tr._actor_reward_components})
        torch.testing.assert_close(metrics['actor_gradient_bootstrap_rollout_ratio'],
            metrics['actor_gradient_reward_bootstrap_norm']/metrics['actor_gradient_reward_rollout_norm'].clamp_min(1e-12))
        assert all(p.grad is None for p in tr.actor.parameters())
        assert all(torch.isfinite(v) for v in metrics.values())
        assert metrics['actor_gradient_reward_norm']>0
        loss.backward()
    assert all(p.grad is None for m in (tr.wm,tr.critic,tr.target_critic) for p in m.parameters())
    assert sum(p.grad.abs().sum() for p in tr.actor.parameters())>0
    del b['exploration_noise_rad']
    with pytest.raises(KeyError):tr.imagine(b)


def test_noise_memory_changes_imagined_targets_reproducibly():
    tr,b=fixture();torch.nn.init.normal_(tr.wm.net[-1].weight,std=.01)
    torch.manual_seed(7);x=tr.imagine(b)[0]
    torch.manual_seed(7);y=tr.imagine(b)[0]
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    altered={**b,'exploration_noise_rad':torch.full_like(b['exploration_noise_rad'],.02)}
    torch.manual_seed(7);z=tr.imagine(altered)[0]
    assert not torch.allclose(x,z,atol=1e-7,rtol=0)


def test_v30_policy_update_roundtrip_and_v29_strict_separation(tmp_path):
    tr,b=fixture();tr.iteration=500
    class Buffer:
        def sample(self,*args,**kwargs):return b
    m=tr.policy_update(Buffer())
    assert math.isfinite(m['actor_loss']) and 'actor_gradient_cost_reward_ratio' in m
    assert m['actor_update_target_max_deg'] >= m['actor_update_target_rms_deg'] > 0
    assert m['actor_update_audit_samples'] == len(b['state'])
    assert math.isfinite(m['actor_input_velocity_abs_p99']) and m['actor_input_velocity_std_min']>0
    assert tr._actor_objective_terms is None and tr._actor_reward_components is None
    assert tr.optimizer_steps==dict(wm=0,actor=1,critic=1)
    meta=dict(environment_contract='test',joint_names=[],context_schema={},observation_protocol='test')
    tr.save(tmp_path/'v30.pt',meta);payload=torch.load(tmp_path/'v30.pt',weights_only=False)
    new,_=fixture();new.restore(payload,meta,restore_rng=False)
    for name in ('actor','wm','critic','target_critic'):
        for k,v in getattr(tr,name).state_dict().items():torch.testing.assert_close(v,getattr(new,name).state_dict()[k])
    payload['version']=29
    with pytest.raises(ValueError,match='Version 30'):new.restore(payload,meta)
    old,_=stable_fixture()
    assert checkpoint_version(old.cfg)==29


@pytest.mark.parametrize('kwargs',[dict(stable_value_learning=False),dict(exploration_position_std_rad=0),
    dict(exploration_correlation_s=float('nan')),dict(actor_logit_penalty=-1),dict(actor_gradient_audit_interval=0),
    dict(action_home=tuple([100.]*21))])
def test_invalid_absolute_policy_settings(kwargs):
    tr,_=fixture()
    with pytest.raises(ValueError):replace(tr.cfg,**kwargs)


def test_shared_physical_velocity_std_floor_survives_static_warmup_without_clipping():
    tr,b=fixture(input_joint_velocity_std_floor_rad_s=math.pi/2)
    normalizers=(tr.actor.obs_norm,tr.wm.input_norm,tr.critic.input_norm,tr.target_critic.input_norm)
    expected=tr.cfg.joint_velocity_scale*math.pi/2
    for norm in normalizers:
        norm.update(torch.zeros(12,len(norm.mean)))
        torch.testing.assert_close(norm.std[21:42],torch.full((21,),expected))
        x=torch.zeros(3,len(norm.mean));x[:,21:42]=10*expected;x.requires_grad_()
        y=norm(x)
        torch.testing.assert_close(y[:,21:42],torch.full((3,21),10.))
        y.sum().backward()
        torch.testing.assert_close(x.grad[:,21:42],torch.full((3,21),1/expected))
        # Large observed variance remains data-driven; the floor is not a cap.
        norm.update(torch.cat((torch.ones(100,len(norm.mean)), -torch.ones(100,len(norm.mean)))))
        assert (norm.std[21:42]>expected).all()
        assert norm.std_floor[:21].count_nonzero()==0 and norm.std_floor[42:].count_nonzero()==0
    old,_=fixture()
    assert 'std_floor' not in old.actor.obs_norm.state_dict()


def test_velocity_std_floor_checkpoint_roundtrip_and_contract_rejection(tmp_path):
    tr,b=fixture(input_joint_velocity_std_floor_rad_s=math.pi/2,critic_td_steps=8)
    meta=dict(environment_contract='test',joint_names=[],context_schema={},observation_protocol='test')
    tr.save(tmp_path/'floor.pt',meta)
    payload=torch.load(tmp_path/'floor.pt',weights_only=False)
    new,_=fixture(input_joint_velocity_std_floor_rad_s=math.pi/2,critic_td_steps=8)
    new.restore(payload,meta,restore_rng=False)
    for name in ('actor','wm','critic','target_critic'):
        for k,v in getattr(tr,name).state_dict().items():torch.testing.assert_close(v,getattr(new,name).state_dict()[k])
    old,_=fixture(critic_td_steps=8)
    with pytest.raises(ValueError,match='protocol'):old.restore(payload,meta,restore_rng=False)


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_invalid_physical_normalization_floor(value):
    tr,_=fixture()
    with pytest.raises(ValueError,match='normalization'):replace(tr.cfg,input_joint_velocity_std_floor_rad_s=value)
