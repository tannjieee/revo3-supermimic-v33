"""Semantic tests for the real-transition FlashSAC bridge (no physics startup)."""
import json
from pathlib import Path
import copy
import math
import pytest
import torch
from viserdex.supermimic.core import Config
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.sensors import initial_sensors, reset_sensors, observe
from viserdex.flashsac.agent import configuration, RevoFlashSAC, save_checkpoint, load_evaluation, tree_hash
from viserdex.flashsac.observations import ObservationEncoder, transition_history
from viserdex.flashsac.exploration import audit_action, ActionRejected
from viserdex.flashsac.rewards import real_transition
from flash_rl.agents.flashSAC.update import _compute_categorical_td_target, _select_min_q_log_probs

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture
def task():
    data=json.loads((ROOT/'configs/flashsac_die47/task.json').read_text())
    return Config(**data['config']),PerceptionNoiseCfg(**data['noise_config'])


def states(cfg,n=8):
    s=torch.zeros(n,84); s[:,:21]=torch.tensor(cfg.action_home)
    s[:,42:45]=torch.tensor([.03425,-.00856,.10169]);s[:,48]=1;s[:,82]=1
    s[:,76:79]=s[:,42:45]
    limits=torch.tensor(cfg.network_joint_limits).expand(n,-1,-1).clone()
    home=(2*(s[:,:21]-torch.tensor(cfg.encoder_bias_rad)-limits[:,:,0])/(limits[:,:,1]-limits[:,:,0])-1)
    return s,limits,home


def test_sensor_terminal_history_matches_behavior_and_keeps_old_goal(task):
    cfg,noise=task;s,limits,a=states(cfg)
    sensor=reset_sensors(initial_sensors(8,'cpu',noise),torch.ones(8,dtype=torch.bool),noise)
    first,sensor=observe(s,sensor,limits,noise,cfg)
    history=first[:,None].repeat(1,10,1)
    terminal=s.clone();terminal[:,55:76]=a;terminal[:,83]=1
    reset=s.clone();reset[0,79:83]=torch.tensor([0.,0.,1.,0.])
    done=torch.tensor([True]+[False]*7)
    next_h,behavior_h,obs,after=transition_history(history,sensor,terminal,reset,done,limits,noise,cfg)
    assert torch.equal(next_h[~done],behavior_h[~done])
    assert torch.equal(next_h[:,:-1],history[:,1:])
    assert torch.equal(behavior_h[0],obs[0].expand(10,-1))
    assert not torch.equal(next_h[0],behavior_h[0])
    assert torch.equal(after['age'][1:],sensor['age'][1:]+1)


def test_absolute_guard_uses_measured_reset_then_last_command(task):
    cfg,_=task;s,limits,a=states(cfg)
    assert audit_action(a,s,limits,cfg)['max_delta_deg']<1e-4
    bad=a.clone();bad[:,15]=1
    with pytest.raises(ActionRejected):audit_action(bad,s,limits,cfg)
    s[:,55:76]=a;s[:,83]=1
    half=(limits[:,:,1]-limits[:,:,0])*.5
    assert audit_action(a+math.radians(2.3)/half,s,limits,cfg)['max_delta_deg']<2.4
    with pytest.raises(ActionRejected):audit_action(a+math.radians(2.5)/half,s,limits,cfg)


def test_home_init_normalization_probability_and_first_update(task):
    cfg,_=task;s,limits,home=states(cfg)
    agent=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg)
    obs=torch.randn(8,963)
    expected=home
    assert torch.allclose(agent.act(obs,training=False),expected,atol=1e-6)
    agent._actor.normalize_parameters()
    assert torch.allclose(agent.act(obs*50,training=False),expected,atol=1e-6)
    a,info=agent._actor(observations=obs[:,:810],training=True)
    assert torch.isfinite(info['log_prob']).all()
    for _ in range(2):
        agent.process_transition(dict(observation=obs, next_observation=obs+.1,action=a.detach(),
            reward=torch.ones(8)*.03,terminated=torch.zeros(8),truncated=torch.zeros(8)))
    before={k:v.clone() for k,v in agent._actor.network.state_dict().items()}
    result=agent.update()
    assert all(math.isfinite(x) for x in result.values())
    assert agent._update_step==1
    assert any(not torch.equal(v,before[k]) for k,v in agent._actor.network.state_dict().items())
    assert torch.isfinite(agent.act(obs,training=False)).all()
    assert not hasattr(agent,'wm')
    assert math.isclose(agent._cfg.temp_target_entropy,-10.0418104853,abs_tol=1e-8)


def test_distribution_target_terminal_timeout_and_min_selection():
    # A delta mass at +1. Terminal maps to reward 0; timeout continues to .99.
    logits=torch.tensor([[float('-inf'),float('-inf'),0.]]).expand(2,-1)
    result=_compute_categorical_td_target(logits,torch.zeros(2),torch.tensor([1.,0.]),
        torch.zeros(2),.99,3,-1.,1.)
    assert torch.allclose(result.sum(-1),torch.ones(2))
    assert torch.allclose(result[0],torch.tensor([0.,1.,0.]))
    assert torch.allclose(result[1],torch.tensor([0.,.01,.99]),atol=1e-6)
    qs=torch.tensor([[1.,0.],[0.,1.]])
    lp=torch.arange(12.).reshape(2,2,3)
    chosen=_select_min_q_log_probs(qs,lp)
    assert torch.equal(chosen,torch.stack((lp[1,0],lp[0,1])))


def test_rewards_keep_five_absorbing_costs_and_prereset(task):
    cfg,_=task;s,limits,a=states(cfg)
    nxt=s.clone();nxt[:,55:76]=a;nxt[:,83]=1
    context=torch.ones(8,65);encoder=ObservationEncoder(cfg,torch.ones(65))
    h=torch.zeros(8,10,81)
    record=dict(state=s,history=h,next_history=h,action=a,next_physical=nxt[:,:55],next_aux=nxt[:,55:],
        reward=torch.ones(8)*.4,terminated=torch.tensor([True]+[False]*7),
        truncated=torch.tensor([False,True]+[False]*6))
    t,raw,risks,ratios=real_transition(record,context,encoder,limits,cfg)
    assert t['observation'].shape==(8,963)
    assert torch.equal(t['observation'][:,:810],encoder.actor(h))
    assert torch.allclose(risks[0],torch.ones(5)*100)
    assert t['reward'][0]==pytest.approx(-24.)
    assert t['reward'][1]==pytest.approx(1.)
    assert not t['terminated'][1] and t['truncated'][1]
    record['next_aux']=record['next_aux'].clone();record['next_aux'][0,24]=1
    with pytest.raises(ValueError):real_transition(record,context,encoder,limits,cfg)


def test_checkpoint_roundtrip_and_protocol_rejection(task,tmp_path):
    cfg,_=task
    agent=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg)
    encoder=ObservationEncoder(cfg,torch.ones(65))
    path=tmp_path/'checkpoint'
    save_checkpoint(path,agent,encoder,dict(environment_contract='test'))
    before=tree_hash(path);obs=torch.randn(8,963);expected=agent.act(obs,training=False)
    restored=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg)
    load_evaluation(path,restored,encoder,'test')
    assert torch.equal(expected,restored.act(obs,training=False))
    assert tree_hash(path)==before
    with pytest.raises(ValueError):load_evaluation(path,restored,encoder,'other')


def test_diagnostics_preserve_exact_upstream_update_and_bn(task):
    from flash_rl.agents.flashSAC.agent import FlashSACAgent
    cfg,_=task
    kwargs=configuration('cpu',42,20,capacity=32,warmup=8,batch=8)
    torch.manual_seed(813);audited=RevoFlashSAC(kwargs,cfg)
    torch.manual_seed(813);reference=RevoFlashSAC(kwargs,cfg)
    transition=dict(observation=torch.randn(16,963),next_observation=torch.randn(16,963),
        action=torch.rand(16,21)*2-1,reward=torch.randn(16),
        terminated=torch.tensor([1.]+[0.]*15),truncated=torch.tensor([0.,1.]+[0.]*14))
    for agent in (audited,reference):agent.process_transition(transition)
    rng=torch.get_rng_state()
    metrics=audited.update()
    torch.set_rng_state(rng);FlashSACAgent.update(reference)
    for name in ('_actor','_critic','_target_critic','_temperature'):
        actual=getattr(audited,name).network.state_dict()
        expected=getattr(reference,name).network.state_dict()
        for key in actual:assert torch.equal(actual[key],expected[key]),(name,key)
    assert 0 <= metrics['diagnostic/target_outside_support_mass'] <= 1
    # Target BN learns on concatenated current/next batch; it is not frozen.
    assert torch.count_nonzero(audited._target_critic.network.embedder.norm.running_mean)>0


def test_s1_mapping_reset_bounds_gradients_and_update(task, tmp_path):
    cfg,_=task
    s,limits,home=states(cfg)
    encoder=ObservationEncoder(cfg,torch.ones(65))
    h=torch.randn(8,10,81)
    obs=encoder(h,s,torch.ones(8,65))
    agent=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg,action_mode='s1')
    net=agent._actor.network
    assert torch.allclose(agent.act(obs,training=False),home,atol=1e-6)
    for local in (-torch.ones(8,21),torch.ones(8,21),torch.rand(8,21)*2-1):
        action=net.map_action(local,obs)
        audit_action(action,s,limits,cfg)
    # At joint limits use one-sided support. Reset ignores stale previous target.
    s[:,55:76]=torch.where(torch.arange(21)%2==0,1.,-1.)
    s[:,83]=1
    obs=encoder(h,s,torch.ones(8,65))
    local=torch.zeros(8,21,requires_grad=True)
    a=net.map_action(local,obs)
    audit_action(a,s,limits,cfg)
    a.sum().backward()
    assert torch.allclose(local.grad,net.rate_delta.expand_as(local)*.5,atol=1e-7)
    # Privileged context cannot affect network output or controller mapping.
    other=obs.clone();other[:,898:]=torch.randn_like(other[:,898:])*100
    assert torch.equal(agent.act(obs,training=False),agent.act(other,training=False))
    # Local log density retained, mapped global action is supplied to the critic.
    torch.manual_seed(123)
    local,local_info=super(type(net),net).forward(obs[:,:810],False)
    torch.manual_seed(123)
    mapped,info=net(obs,False)
    assert torch.equal(info['log_prob'],local_info['log_prob'])
    assert torch.equal(mapped,net.map_action(local,obs))
    nxt=s.clone();nxt[:,55:76]=mapped.detach()
    next_obs=encoder(h,nxt,torch.ones(8,65))
    agent.process_transition(dict(observation=obs,next_observation=next_obs,action=mapped.detach(),
        reward=torch.ones(8)*.03,terminated=torch.zeros(8),truncated=torch.zeros(8)))
    result=agent.update()
    assert all(math.isfinite(x) for x in result.values())
    audit_action(agent.act(next_obs,training=True),nxt,limits,cfg)
    checkpoint=tmp_path/'s1'
    save_checkpoint(checkpoint,agent,encoder,dict(environment_contract='test'))
    restored=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg,action_mode='s1')
    load_evaluation(checkpoint,restored,encoder,'test')
    assert torch.equal(agent.act(obs,training=False),restored.act(obs,training=False))
    old=RevoFlashSAC(configuration('cpu',42,20,capacity=32,warmup=8,batch=8),cfg)
    with pytest.raises(ValueError,match='protocol'):
        load_evaluation(checkpoint,old,encoder,'test')


def test_s1_rate120_contract_and_presend_guard(task, tmp_path):
    cfg,_=task
    data=json.loads((ROOT/'configs/flashsac_die47/task_rate120.json').read_text())
    base=json.loads((ROOT/'configs/flashsac_die47/task.json').read_text())
    assert {k for k in base['config'] if base['config'][k]!=data['config'][k]}=={'cost_target_rate_limit_rad_s'}
    assert base['noise_config']==data['noise_config']
    fast=Config(**data['config'])
    s,limits,home=states(fast)
    encoder=ObservationEncoder(fast,torch.ones(65))
    obs=encoder(torch.zeros(8,10,81),s,torch.ones(8,65))
    config=configuration('cpu',42,20,capacity=32,warmup=8,batch=8)
    agent=RevoFlashSAC(config,fast,action_mode='s1')
    assert agent.version=='flashsac_revo3_die47_hist25_s1_rate120_v1'
    assert torch.allclose(agent.act(obs,training=False),home,atol=1e-6)
    for sign in (-1.,1.):
        action=agent._actor.network.map_action(torch.full((8,21),sign),obs)
        report=audit_action(action,s,limits,fast)
        assert abs(report['max_delta_deg']-4.8)<1e-4
    half=(limits[:,:,1]-limits[:,:,0])*.5
    audit_action(home+math.radians(4.7)/half,s,limits,fast)
    with pytest.raises(ActionRejected):
        audit_action(home+math.radians(4.9)/half,s,limits,fast)
    checkpoint=tmp_path/'rate120'
    save_checkpoint(checkpoint,agent,encoder,dict(environment_contract='test'))
    old=RevoFlashSAC(config,cfg,action_mode='s1')
    with pytest.raises(ValueError,match='protocol'):
        load_evaluation(checkpoint,old,encoder,'test')
