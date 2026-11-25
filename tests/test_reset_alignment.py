"""Reset memory is part of dynamics/value state, without changing Actor inputs."""
from dataclasses import replace
import torch
from test_absolute_policy import fixture
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.networks import Actor, Dynamics, Critic
from viserdex.supermimic.trainer import checkpoint_version, imagination_protocol


def test_v31_disambiguates_identical_physics_with_different_pending_commands():
    tr, batch = fixture(2)
    state = batch['state'].clone()
    state[1] = state[0]
    state[:,:21] = torch.tensor(tr.cfg.action_home)
    state[:,55:76] = 0
    state[:,83] = torch.tensor([0.,1.])
    target = state[:,:21]-torch.tensor(tr.cfg.encoder_bias_rad)
    lower,upper = tr.limits[...,0],tr.limits[...,1]
    action = 2*(target-lower)/(upper-lower)-1
    raw, _ = violation_costs(state,state[:,:55],action,tr.limits,tr.cfg)
    assert raw[0,1] == 0 and raw[1,1] > 60
    old_v = tr.critic.inputs(state,tr.context)
    old_w = tr.wm.inputs(state,target,tr.context)
    torch.testing.assert_close(old_v[0],old_v[1],atol=0,rtol=0)
    torch.testing.assert_close(old_w[0],old_w[1],atol=0,rtol=0)
    cfg = replace(tr.cfg,include_delay_ready=True)
    wm,critic = Dynamics(65,cfg),Critic(65,cfg)
    new_w,new_v = wm.inputs(state,target,tr.context),critic.inputs(state,tr.context)
    assert torch.equal(new_w[:,99],state[:,83])
    assert torch.equal(new_v[:,87],state[:,83])
    torch.testing.assert_close(torch.cat((new_w[:,:99],new_w[:,100:]),-1),old_w)
    torch.testing.assert_close(torch.cat((new_v[:,:87],new_v[:,88:]),-1),old_v)
    assert checkpoint_version(tr.cfg)==30 and checkpoint_version(cfg)==31
    assert imagination_protocol(cfg).endswith('_delay_ready_v31')
    # Deployment ABI, initial policy and all 81 observation features unchanged.
    actor = Actor(cfg)
    actor.load_state_dict(tr.actor.state_dict(),strict=True)
    torch.testing.assert_close(actor(batch['history']),tr.actor(batch['history']),atol=0,rtol=0)
