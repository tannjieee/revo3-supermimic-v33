"""Control, contact supervision, gradient isolation and deploy/replay parity for v33."""
from dataclasses import replace, asdict
import copy
import math
import pytest
import torch
from viserdex.supermimic import dex_reference as dex
from viserdex.supermimic.core import hand_center_config, action_target, advance, physical_features
from viserdex.supermimic.networks import Actor, Dynamics, Critic
from viserdex.supermimic.trainer import Trainer, checkpoint_version, frozen
from viserdex.supermimic.sensors import initial_sensors, observe
from viserdex.supermimic.contact_prediction import reduce_contacts
from viserdex.supermimic.wm_supervision import clean_next_observation, task_reward
from viserdex.distillation.config import PerceptionNoiseCfg


def config(small=True):
    p, h = dex.load_profile()
    c = hand_center_config(wm_task_losses=True, stable_value_learning=True,
        absolute_policy_optimization=True, paper_critic=True)
    c = replace(c, **dex.profile_options(p, h))
    if small:
        c = replace(c, wm_hidden=(32,32), actor_hidden=(32,), critic_hidden=(32,),
            wm_batch=4, actor_batch=4, microbatch=4, wm_window=3, horizon=3,
            critic_minibatches=1, critic_epochs=1, wm_only_iters=0)
    return c


def fixture():
    c = config()
    limits = torch.tensor(c.network_joint_limits).expand(4,-1,-1)
    s = torch.zeros(4, c.state_dim)
    s[:, :21] = torch.tensor(c.action_home)
    s[:, 48] = s[:, 82] = 1
    s[:, 42:45] = torch.tensor(c.hand_center_base_m)
    s[:, 76:79] = s[:, 42:45]
    s[:, dex.ZERO] = torch.linspace(-math.pi/180,math.pi/180,4)[:,None]
    s[:, dex.EMA] = s[:, :21]+s[:, dex.ZERO]
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False, joint_delay_max_steps=0)
    t = Trainer(c, noise, torch.ones(4,65), limits, torch.tensor([[0.,0.,0.,1.]]).expand(4,-1), 'cpu')
    return t, s


def window(t,s):
    k=t.cfg.wm_window
    sensors=initial_sensors(4,'cpu',t.noise_cfg)
    obs,sensors=observe(s,sensors,t.limits,t.noise_cfg,t.cfg)
    a=t.actor(obs[:,None].expand(-1,3,-1)).detach()
    return dict(state=s[:,None].expand(-1,k,-1).clone(),action=a[:,None].expand(-1,k,-1).clone(),
        next_physical=s[:,:55][:,None].expand(-1,k,-1).clone(),
        contact=torch.zeros(4,k,22,dtype=torch.bool),terminated=torch.zeros(4,k,dtype=torch.bool),
        done=torch.zeros(4,k,dtype=torch.bool),mask=torch.ones(4,k),env_id=torch.arange(4),
        reward=task_reward(s,s[:,:55],a,t.cfg)[:,None].expand(-1,k).clone())


def test_speed_cost_resume_override_preserves_state_and_rejects_other_changes(tmp_path):
    t,s=fixture();w=window(t,s)
    class Buffer:
        def sample(self,*args,**kw):return w
        def update_priorities(self,*args):pass
    t.wm_update(Buffer())
    t.iteration=8000
    meta=dict(environment_contract='test',joint_names=[],context_schema={},
        observation_protocol=t.cfg.observation_protocol,state_protocol=t.cfg.state_protocol)
    t.save(tmp_path/'checkpoint.pt',meta)
    payload=torch.load(tmp_path/'checkpoint.pt',weights_only=False)
    other,_=fixture()
    other.cfg=replace(other.cfg,cost_velocity_limit_rad_s=math.radians(120),
                      cost_target_rate_limit_rad_s=math.radians(180))
    with pytest.raises(ValueError,match='configuration changed'):
        other.restore(payload,meta,restore_rng=False)
    other.restore(payload,meta,restore_rng=False,allow_speed_cost_override=True)
    assert other.iteration==8000 and other.optimizer_steps==t.optimizer_steps
    assert other.cfg.input_joint_velocity_std_floor_rad_s==t.cfg.input_joint_velocity_std_floor_rad_s
    for name in ('wm','actor','critic','opt_wm','opt_actor','opt_critic','opt_failure'):
        torch.testing.assert_close(getattr(other,name).state_dict(),getattr(t,name).state_dict(),rtol=0,atol=0)
    other.cfg=replace(other.cfg,cost_penalty=other.cfg.cost_penalty*2)
    with pytest.raises(ValueError,match='configuration changed'):
        other.restore(payload,meta,restore_rng=False,allow_speed_cost_override=True)


def test_soft_limits_and_all_zero_offsets_are_mechanically_legal():
    p,_=dex.load_profile(); limits,asset=dex.validate_profile(p)
    assert p['soft_limits_deg']['right_thumb_MCP_joint'][0]==4
    assert p['soft_limits_deg']['right_little_MPR_joint'][0]==-29
    bad=copy.deepcopy(p);bad['soft_limits_deg']['right_thumb_MCP_joint'][0]=0
    with pytest.raises(ValueError,match='mechanical'):dex.validate_profile(bad)
    c=config();assert checkpoint_version(c)==33
    assert type(c)(**asdict(c))==c


def test_production_shapes_and_physical_wm_ignores_goal_raw_and_controller_memory():
    c=config(False);a=Actor(c);w=Dynamics(65,c);v=Critic(65,c)
    assert a.net[0].in_features==225 and sum(p.numel() for p in a.parameters())==892949
    assert w.net[0].in_features==149 and w.net[-1].out_features==76
    assert v.net[0].in_features==200
    t,s=fixture();target=action_target(s,torch.zeros(4,21),t.limits,t.cfg)
    original=t.wm.inputs(s,target,t.context)
    altered=s.clone();altered[:,55:125]=torch.randn_like(altered[:,55:125])
    torch.testing.assert_close(t.wm.inputs(altered,target,t.context),original,atol=0,rtol=0)
    altered[:,dex.WRENCH]=.1
    assert not torch.equal(t.wm.inputs(altered,target,t.context),original)


def test_ema_tracking_observation_and_rate_cost_use_previous_actual_target():
    t,s=fixture();a=torch.ones(4,21)
    target=action_target(s,a,t.limits,t.cfg)
    expected=.51*t.limits[...,1]+.49*s[:,dex.EMA]-s[:,dex.ZERO]
    torch.testing.assert_close(target,expected)
    nxt=advance(s,s[:,:55],a,target,s[:,79:83],hold_cfg=t.cfg)
    torch.testing.assert_close(nxt[:,dex.EMA],target+s[:,dex.ZERO])
    obs,_=observe(nxt,initial_sensors(4,'cpu',t.noise_cfg),t.limits,t.noise_cfg,t.cfg)
    half=(t.limits[...,1]-t.limits[...,0])/2
    torch.testing.assert_close(obs[:,21:42],(nxt[:,:21]+nxt[:,dex.ZERO]-nxt[:,dex.EMA])/half)
    clean=clean_next_observation(s,s[:,:55],a,t.limits,t.cfg)
    torch.testing.assert_close(clean,obs)
    from viserdex.supermimic.constraints import violation_costs
    _,ratios=violation_costs(s,s[:,:55],a,t.limits,t.cfg)
    expected_rate=((target-(s[:,dex.EMA]-s[:,dex.ZERO]))/.04).abs().amax(-1)/t.cfg.cost_target_rate_limit_rad_s
    torch.testing.assert_close(ratios[:,1],expected_rate)


def test_contact_reduction_handles_geometry_order_worlds_inactive_and_duplicates():
    channels=torch.tensor([0,0,1,-1,-1]);obj=torch.tensor([False,False,False,True,False])
    pairs=torch.tensor([[3,0],[1,3],[3,2],[2,3],[4,3],[0,2],[-1,3]])
    worlds=torch.tensor([0,0,1,0,0,0,0]);active=torch.tensor([1,1,1,0,1,1,1],dtype=torch.bool)
    actual=reduce_contacts(pairs,worlds,active,channels,obj,2,2)
    assert actual.tolist()==[[True,False],[False,True]]
    torch.testing.assert_close(reduce_contacts(pairs.flip(-1),worlds,active,channels,obj,2,2),actual)


def test_failure_optimizer_is_disjoint_and_prediction_is_detached():
    t,s=fixture();w=window(t,s);w['terminated'][0,0]=True
    loss,*_=t._wm_loss(w)
    failure=t._independent_failure_loss
    loss.backward()
    assert all(p.grad is None for p in t.wm.termination_net.parameters())
    before=[None if p.grad is None else p.grad.clone() for p in t.wm.net.parameters()]
    failure.backward()
    for saved,p in zip(before,t.wm.net.parameters()):torch.testing.assert_close(saved,p.grad)
    assert t.wm.termination_net[-1].bias.grad.abs().sum()>0
    wm_ids={id(p) for g in t.opt_wm.param_groups for p in g['params']}
    f_ids={id(p) for g in t.opt_failure.param_groups for p in g['params']}
    assert not wm_ids & f_ids
    with torch.no_grad():torch.nn.init.normal_(t.wm.termination_net[-1].weight)
    x=s[:,:55].clone().requires_grad_()
    with frozen(t.wm):t.wm.termination_probability(x).sum().backward()
    assert x.grad.abs().sum()>0


def test_wm_mask_excludes_nan_reset_tails_and_records_current_wrench():
    t,s=fixture();w=window(t,s)
    w['mask'][:,1:]=0;w['done'][:,0]=True
    for name in ('state','next_physical','action','reward'):
        w[name][:,1:]=float('nan')
    seen=[]
    handle=t.wm.input_norm.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    w['state'][:,0,dex.WRENCH]=.005
    loss,*_=t._wm_loss(w);loss.backward();t._independent_failure_loss.backward();handle.remove()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in t.wm.parameters())
    torch.testing.assert_close(seen[0][:,78:84],torch.full((4,6),.005))


def test_updates_checkpoint_and_export_sequence_parity(tmp_path):
    t,s=fixture();w=window(t,s)
    class Buffer:
        def sample(self,*args,**kw):return w
        def update_priorities(self,*args):pass
    metrics=t.wm_update(Buffer());assert math.isfinite(metrics['wm_loss_contact'])
    assert t.optimizer_steps['failure']==t.optimizer_steps['wm']==1
    sensors=initial_sensors(4,'cpu',t.noise_cfg);obs,sensors=observe(s,sensors,t.limits,t.noise_cfg,t.cfg)
    b=dict(state=s,history=obs[:,None].expand(-1,3,-1).clone(),env_id=torch.arange(4),exploration_noise_rad=torch.zeros(4,21))
    b.update({'sensor/'+k:v for k,v in sensors.items()})
    class PolicyBuffer:
        def sample(self,*args,**kw):return b
    metrics=t.policy_update(PolicyBuffer());assert math.isfinite(metrics['actor_loss'])
    from viserdex.supermimic.revised_contract import joint_names
    meta=dict(environment_contract='test',joint_names=list(joint_names(t.cfg)),context_schema={},
        observation_protocol=t.cfg.observation_protocol,state_protocol=t.cfg.state_protocol,
        observation_dim=75,quaternion_order='XYZW',joint_limits_rad=t.cfg.network_joint_limits)
    t.save(tmp_path/'checkpoint.pt',meta)
    other,_=fixture();other.restore(torch.load(tmp_path/'checkpoint.pt',weights_only=False),meta,restore_rng=False)
    assert other.optimizer_steps==t.optimizer_steps
    from viserdex.supermimic.inference import export_actor,ManualConvexPolicy
    export_actor(tmp_path/'checkpoint.pt',tmp_path/'actor.pt');policy=ManualConvexPolicy(tmp_path/'actor.pt')
    policy.reset(s[0,dex.ZERO]);policy.set_goal(s[0,79:83],s[0,76:79])
    state=s[:1].clone();history=None
    for step in range(4):
        measured=state[0,:21]+state[0,dex.ZERO]
        target,raw,seen=policy.propose(measured,state[0,21:42],state[0,42:45],state[0,45:49])
        clean,_=observe(state,initial_sensors(1,'cpu',t.noise_cfg),t.limits[:1],t.noise_cfg,t.cfg)
        history=clean[:,None].expand(-1,3,-1).clone() if history is None else torch.cat((history[:,1:],clean[:,None]),1)
        action=t.actor(history);expected=action_target(state,action,t.limits[:1],t.cfg)
        torch.testing.assert_close(torch.tensor(target),expected[0]);torch.testing.assert_close(torch.tensor(seen),clean[0])
        policy.commit();state=advance(state,state[:,:55],action,expected,state[:,79:83],hold_cfg=t.cfg).detach()


@pytest.mark.parametrize('adaptive',[False,True])
def test_real_replay_retains_contact_labels_in_windows(adaptive):
    from viserdex.supermimic.replay import Replay
    t,s=fixture();w=window(t,s)
    replay=Replay(8,4,'cpu',event_capacity=0,event_window=3,adaptive=adaptive)
    for k in range(5):
        record={key:value[:,0].clone() for key,value in w.items() if key not in ('env_id','mask')}
        record['contact'][:,k%22]=True
        replay.add(record)
    b=replay.sample(4,3,include_events=False)
    assert b['contact'].shape==(4,3,22)
    assert b['contact'].any(-1).all()
    assert torch.isfinite(t._wm_loss(b)[0])


def test_new_action_defaults_preserve_old_environment_contract_hash():
    from viserdex.distillation.environment import contract_sha256
    scene={n:{'spawn':{}} for n in ('robot','object')}
    old=dict(actions={'joint_pos':{'class_type':'viserdex.supermimic.direct_actions:DirectJointPositionAction',
                                  'limit_contract':'hardware_center25_v3'}},
             commands={'object_pose':{}},sim={'physics':{}},scene=scene,
             observations={'policy':{},'privileged':{}})
    new=copy.deepcopy(old);new['actions']['joint_pos'].update(alpha=1.,dex_reference_profile=None)
    assert contract_sha256(old)==contract_sha256(new)
    p,_=dex.load_profile();new['actions']['joint_pos'].update(alpha=.51,dex_reference_profile=p)
    assert contract_sha256(old)!=contract_sha256(new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
def test_cuda_graph_preserves_separate_failure_optimizer_and_changing_inputs():
    cpu,state=fixture()
    eager=Trainer(cpu.cfg,cpu.noise_cfg,cpu.context.cuda(),cpu.limits.cuda(),cpu.base_inverse.cuda(),'cuda')
    captured=copy.deepcopy(eager);captured.wm_cuda_graph=True
    w={k:v.cuda() for k,v in window(cpu,state).items()}
    w['terminated'][0,0]=True
    w['done'][1,0]=True;w['mask'][1,1:]=0
    for key in ('state','next_physical','action','reward'):
        w[key][1,1:]=float('nan')
    class Buffer:
        def sample(self,*args,**kw):return w
        def update_priorities(self,*args):pass
    for step in range(3):
        w['contact'][:,:,step]=True
        w['state'][:,0,125:131]=step*.001
        a=eager.wm_update(Buffer());b=captured.wm_update(Buffer())
        for name in a:assert b[name]==pytest.approx(a[name],rel=2e-5,abs=2e-6),name
        for name,x in eager.wm.state_dict().items():
            torch.testing.assert_close(x,captured.wm.state_dict()[name],rtol=2e-5,atol=2e-6)
        for opt in ('opt_wm','opt_failure'):
            x,y=getattr(eager,opt).state_dict(),getattr(captured,opt).state_dict()
            for key,values in x['state'].items():
                for field,value in values.items():
                    torch.testing.assert_close(value,y['state'][key][field],rtol=3e-5,atol=2e-6)
        for t in (eager,captured):
            t.wm.zero_grad(set_to_none=True)
            t.wm.delta_norm.mean.add_(.00001)
            t.wm.input_norm.variance.mul_(.99)
    assert eager.optimizer_steps==captured.optimizer_steps
    assert captured.optimizer_steps['failure']==3
