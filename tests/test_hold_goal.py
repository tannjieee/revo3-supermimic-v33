"""Protocol boundaries, task state, reward pulses, gradients and replay retention."""
import torch
import pytest
from test_supermimic import fixture, config, make_replay
from viserdex.supermimic.core import *
from viserdex.supermimic.goal_hold import hold_transition, required_steps
from viserdex.supermimic.networks import Actor, Critic, Dynamics
from viserdex.supermimic.trainer import Trainer


def test_ten_samples_not_nine_and_single_completion_pulse():
    ticks=torch.zeros(3);angle=torch.tensor([.05,.05,.05]);speed=torch.tensor([.1,.1,.1])
    for step in range(1,11):
        ticks,event=hold_transition(ticks,angle,speed,.05)
        assert event.tolist()==[step==10]*3
        assert ticks.tolist()==[0 if step==10 else step]*3
    ticks,event=hold_transition(ticks,angle,speed,.05)
    assert not event.any() and ticks.tolist()==[1]*3


def test_strict_thresholds_breaks_and_partial_reset():
    ticks=torch.full((6,),9.)
    angle=torch.tensor([.099,.1,.099,.099,float('nan'),.099])
    speed=torch.tensor([.499,.1,.5,.1,.1,float('inf')])
    blocked=torch.tensor([False,False,False,True,False,False])
    out,event=hold_transition(ticks,angle,speed,.05,blocked=blocked)
    assert event.tolist()==[True,False,False,False,False,False]
    assert out.tolist()==[0]*6
    ticks=torch.tensor([3.,8.]);out,event=hold_transition(ticks,torch.zeros(2),torch.zeros(2),.05,blocked=torch.tensor([True,False]))
    assert out.tolist()==[0.,9.] and not event.any()


def test_interrupt_requires_ten_new_consecutive_samples():
    ticks=torch.tensor([8.])
    ticks,event=hold_transition(ticks,torch.tensor([.2]),torch.zeros(1),.05)
    assert ticks.item()==0 and not event.item()
    for i in range(9):ticks,event=hold_transition(ticks,torch.zeros(1),torch.zeros(1),.05)
    assert ticks.item()==9 and not event.item()


def test_reward_and_goal_switch_share_completion_event():
    s,limits=fixture(1);cfg=config20();s[:,:21]=torch.tensor(cfg.action_home)
    physical=s[:,:55].clone();a=torch.zeros(1,21)
    newgoal=rotation_vector_quat(torch.tensor([[.8,0,0]]))
    for step in range(10):
        r=reward(s,physical,a,.05,cfg.success_bonus,hold_cfg=cfg)
        expected=.5+(cfg.success_bonus*.05 if step==9 else 0)
        torch.testing.assert_close(r,torch.tensor([expected]))
        nxt=advance(s,physical,a,ema_target(s,a,limits),newgoal,400,hold_cfg=cfg)
        assert nxt[:,COUNT].item()==(1 if step==9 else 0)
        assert nxt[:,HOLD].item()==(0 if step==9 else step+1)
        torch.testing.assert_close(nxt[:,GOAL_ROT],newgoal if step==9 else s[:,GOAL_ROT])
        s=nxt
    # The new goal is not close: no repeated bonus on the following step.
    assert reward(s,physical,a,.05,cfg.success_bonus,hold_cfg=cfg).item()<.5


@pytest.mark.parametrize('reason',['timeout','drop','count_limit'])
def test_ending_transition_cannot_complete_hold(reason):
    s,_=fixture(1);s[:,HOLD]=9
    if reason=='timeout':s[:,TIME]=1/400
    if reason=='count_limit':s[:,COUNT]=50
    physical=s[:,:55].clone()
    if reason=='drop':physical[:,POS]=torch.tensor([.31,0,0])
    ticks,event=goal_event(s,physical,.05)
    assert ticks.item()==0 and not event.item()
    assert abs(reward(s,physical,torch.zeros(1,21),.05).item()-.5)<1e-6


def test_counter_affects_critic_but_not_actor_or_physical_model_inputs():
    s,limits=fixture(2);s[1,HOLD]=9;cfg=config();context=torch.ones(2,2)
    critic=Critic(2,cfg);wm=Dynamics(2,cfg)
    assert critic.inputs(s,context).shape[-1]==179
    assert not torch.equal(critic.inputs(s,context)[0],critic.inputs(s,context)[1])
    torch.testing.assert_close(wm.inputs(s,s[:,EMA],context)[0],wm.inputs(s,s[:,EMA],context)[1])
    actor=Actor(cfg);assert actor(torch.zeros(2,10,124)).shape==(2,21)
    assert STATE_DIM==173 and full_features(s).shape[-1]==177


def test_new_state_is_retained_in_replay_and_critic():
    cfg=config();replay,s,limits,noise=make_replay(cfg)
    trainer=Trainer(cfg,noise,torch.ones(6,2),limits,torch.tensor([[0.,0.,0.,1.]]).expand(6,-1),'cpu',600)
    # Checkpoint tests in the parent suite also cover complete optimizer restore.
    assert trainer.critic.input_norm.mean.numel()==179
    assert replay.sample(2)['state'].shape[-1]==173


def test_duration_requires_exact_control_ticks():
    assert required_steps(.05,.5)==10 and required_steps(1/30,.5)==15
    with pytest.raises(ValueError):required_steps(.05,.51)


def test_imagination_propagates_hold_across_rollout_steps(monkeypatch):
    import viserdex.supermimic.trainer as module
    cfg=config(horizon=12);cfg.control_dt=.05
    replay,s,limits,noise=make_replay(cfg)
    trainer=Trainer(cfg,noise,torch.ones(6,2),limits,torch.tensor([[0.,0.,0.,1.]]).expand(6,-1),'cpu',400)
    class FixedPhysical(torch.nn.Module):
        def forward(self,s,target,context):return s[:,:55]
    trainer.wm=FixedPhysical()
    monkeypatch.setattr(module,'sample_xy_goal',lambda ref:rotation_vector_quat(ref.new_tensor([[.8,0.,0.]]).expand(len(ref),-1)))
    batch=replay.sample(6)
    batch['state'][:,HOLD]=0;batch['state'][:,TIME]=1;batch['state'][:,COUNT]=0
    batch['state'][:,ROT]=batch['state'][:,GOAL_ROT]=torch.tensor([0.,0.,0.,1.])
    batch['state'][:,OMEGA]=0;batch['state'][:,POS]=torch.tensor([.02,.03,.04])
    loss,states,targets,context,valid=trainer.imagine(batch)
    assert states[:,0,HOLD].squeeze(-1).tolist()==[0,1,2,3,4,5,6,7,8,9,0,0]
    assert states[:10,:,COUNT].sum()==0
    assert (states[10:,:,COUNT]==1).all()
    assert valid.all() and torch.isfinite(targets).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in trainer.actor.parameters() if p.grad is not None)
