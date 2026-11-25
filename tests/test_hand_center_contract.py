from dataclasses import replace
import math
import torch
from viserdex.supermimic.core import hand_center_config, physical_delta, integrate_physical
from viserdex.supermimic.revised_contract import joint_names, urdf_limits, CENTER_REVISION
from viserdex.supermimic.networks import Actor,Dynamics,Critic
from viserdex.supermimic.sensors import observe,initial_sensors
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.trainer import Trainer,frozen,imagined_continuation,checkpoint_version
from viserdex.supermimic.spatial_contract import drop_distance
from viserdex.distillation.config import PerceptionNoiseCfg


def fixture(n=3):
    cfg=hand_center_config();limits=torch.tensor(cfg.network_joint_limits,dtype=torch.float32)[None].expand(n,-1,-1)
    state=torch.zeros(n,84);state[:,:21]=limits.mean(-1)+torch.tensor(cfg.encoder_bias_rad)
    state[:,45:49]=state[:,79:83]=torch.tensor([0.,0.,0.,1.])
    state[:,42:45]=state[:,76:79]=torch.tensor(cfg.hand_center_base_m)
    state[:,83]=1
    return cfg,state,limits


def test_hardware_order_and_limits_bias_are_not_only_cosmetic():
    cfg,s,limits=fixture()
    assert [n.split('_')[2] for n in joint_names(cfg)[16:]]==['MCP','PIP','DIP','CMP','CMR']
    assert cfg.encoder_bias_rad[19]==math.pi/180 and sum(x!=0 for x in cfg.encoder_bias_rad)==1
    _,src=urdf_limits(CENTER_REVISION)
    torch.testing.assert_close(limits[0]+torch.tensor(cfg.encoder_bias_rad)[:,None],torch.tensor(src['geometric_limits_rad'],dtype=torch.float32))
    assert cfg.control_dt==.04 and cfg.hold_duration==.52 and checkpoint_version(cfg)==27


def test_same_physical_quantities_have_same_scaled_values_in_all_networks():
    cfg,s,limits=fixture();s[:,21:42]=torch.linspace(-2,2,21)
    noise=PerceptionNoiseCfg(enabled=False,joint_enabled=False)
    obs,_=observe(s,initial_sensors(3,'cpu',noise),limits,noise,cfg)
    a=Actor(cfg).scaled_observation(obs)
    w=Dynamics(65,cfg).inputs(s,limits.mean(-1),torch.ones(3,65))
    v=Critic(65,cfg).inputs(s,torch.ones(3,65))
    torch.testing.assert_close(a[:,:42],w[:,:42],atol=2e-7,rtol=0)
    torch.testing.assert_close(a[:,63:66],w[:,42:45],atol=0,rtol=0)
    torch.testing.assert_close(a[:,66:69],v[:,78:81],atol=0,rtol=0)
    assert a[:,63:69].abs().max()<1e-7
    torch.testing.assert_close(w[:,:57],v[:,:57])


def test_drop_frame_and_physical_increment_translation_invariance():
    cfg,s,limits=fixture();assert drop_distance(s,cfg).max()==0
    far=s.clone();far[:,42]+=.31
    assert (drop_distance(far,cfg)>.3).all()
    near_c=imagined_continuation(s,s[:,:55],.01,True,cfg)
    far_c=imagined_continuation(s,far[:,:55],.01,True,cfg)
    assert (near_c>.99).all() and (far_c<.5).all()
    delta=physical_delta(s,far[:,:55]);torch.testing.assert_close(integrate_physical(s,delta),far[:,:55])
    shifted=s.clone();shifted[:,42:45]-=torch.tensor(cfg.hand_center_base_m)
    target=far[:,:55].clone();target[:,42:45]-=torch.tensor(cfg.hand_center_base_m)
    torch.testing.assert_close(delta,physical_delta(shifted,target),atol=1e-7,rtol=0)


def test_cost_threshold_units_and_gradients():
    cfg,s,limits=fixture();action=torch.zeros(3,21,requires_grad=True)
    nxt=s[:,:55].clone();nxt[:,21]=2*cfg.cost_velocity_limit_rad_s
    cost,ratio=violation_costs(s,nxt,action,limits,cfg)
    torch.testing.assert_close(cost[:,0],torch.ones(3))
    assert cost[:,1].max()==0
    action2=torch.full((3,21),.5,requires_grad=True)
    c,r=violation_costs(s,nxt,action2,limits,cfg);c[:,1].sum().backward()
    assert torch.isfinite(action2.grad).all() and action2.grad.abs().sum()>0
    assert math.isclose(cfg.cost_velocity_limit_rad_s,math.radians(90))
    assert math.isclose(cfg.cost_target_rate_limit_rad_s,math.radians(60))


def test_cost_rollout_and_value_fit_are_differentiable_and_frozen_correctly():
    cfg,s,limits=fixture(2);cfg=replace(cfg,wm_hidden=(16,),actor_hidden=(16,),critic_hidden=(16,),
        actor_batch=2,actor_microbatch=2,horizon=3,critic_epochs=1,critic_minibatches=1)
    noise=PerceptionNoiseCfg(enabled=False,joint_enabled=False,joint_delay_max_steps=0)
    tr=Trainer(cfg,noise,torch.ones(2,65),limits,torch.tensor([[0.,0.,0.,1.]]).expand(2,-1),'cpu',500)
    torch.nn.init.normal_(tr.wm.net[-1].weight,std=.001)
    batch=dict(state=s,history=torch.zeros(2,10,81),env_id=torch.arange(2))
    batch.update({'sensor/'+k:v for k,v in initial_sensors(2,'cpu',noise).items()})
    with frozen(tr.wm,tr.critic):
        loss,states,targets,context,valid=tr.imagine(batch);loss.backward()
    assert torch.isfinite(loss) and targets.shape==(3,2,6) and not targets.requires_grad
    assert sum(float(p.grad.abs().sum())for p in tr.actor.parameters()if p.grad is not None)>0
    assert all(p.grad is None for p in tr.wm.parameters()) and all(p.grad is None for p in tr.critic.parameters())
    result=tr.fit_critic((states.flatten(0,1),targets.flatten(0,1),context[None].expand(3,-1,-1).flatten(0,1)))
    assert math.isfinite(result['critic_loss'])
