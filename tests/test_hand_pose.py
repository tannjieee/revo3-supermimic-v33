"""Independent reward delta/gradient and actual-joint dependence checks."""
from dataclasses import replace
from types import SimpleNamespace
import torch
import pytest
from viserdex.supermimic.core import config20,Config,reward_from_orientation_error,Q
from viserdex.supermimic.hand_pose import hand_pose_penalty

def penalty_delta(state,nxt,action,cfg):
    error=torch.full(nxt.shape[:-1],.3,dtype=nxt.dtype)
    event=torch.zeros(nxt.shape[:-1],dtype=torch.bool)
    def evaluate(c):return reward_from_orientation_error(state,nxt,action,error,dt=c.control_dt,hold_cfg=c,success_event=event)
    return evaluate(cfg)-evaluate(replace(cfg,hand_pose_weight=0.))

def test_pose_penalty_is_actual_post_step_squared_sum_with_dt_once():
    cfg=config20();assert cfg.hand_pose_weight==.2
    home=torch.tensor(cfg.action_home,dtype=torch.float64)
    state=torch.zeros(3,173,dtype=torch.float64);state[:,Q]=home
    nxt=torch.zeros(3,55,dtype=torch.float64);nxt[:,Q]=home
    nxt[1,Q]+=.1;nxt[2,0]+=.5
    delta=penalty_delta(state,nxt,torch.zeros(3,21,dtype=torch.float64),cfg)
    torch.testing.assert_close(delta,torch.tensor([0.,-.0021,-.0025],dtype=torch.float64),atol=1e-14,rtol=0)
    # Penalize the current physical pose even with zero action. Conversely an
    # arbitrary action/EMA target must not change this added reward component.
    state[:,55:160]=9
    torch.testing.assert_close(penalty_delta(state,nxt,torch.full((3,21),.9,dtype=torch.float64),cfg),delta)

def test_pose_gradient_points_toward_home_and_survives_terminal_state():
    cfg=config20();home=torch.tensor(cfg.action_home,dtype=torch.float64)
    state=torch.zeros(2,173,dtype=torch.float64);state[:,Q]=home;state[:,167]=0
    nxt=torch.zeros(2,55,dtype=torch.float64);nxt[:,Q]=home+.2;nxt.requires_grad_()
    penalty_delta(state,nxt,torch.zeros(2,21,dtype=torch.float64),cfg).sum().backward()
    torch.testing.assert_close(nxt.grad[:,Q],torch.full((2,21),-.004,dtype=torch.float64))
    torch.testing.assert_close(nxt.grad[:,21:],torch.zeros_like(nxt.grad[:,21:]))

def test_real_term_uses_per_environment_default_and_actual_joints():
    # Simulate a partial reset: one environment has returned home, the other
    # retains its pre-reset physical deviation. Action target isn't an input.
    home=torch.randn(2,21);q=home.clone();q[1,3]+=.5
    wrap=lambda t:SimpleNamespace(torch=t)
    env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos=wrap(q),default_joint_pos=wrap(home)))})
    torch.testing.assert_close(hand_pose_penalty(env),torch.tensor([0.,.25]))

def test_hand_pose_rejects_missing_reference_and_invalid_weights():
    for weight in (-.2,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='hand_pose_weight'):Config(hand_pose_weight=weight)
    with pytest.raises(ValueError,match='21 finite'):Config(hand_pose_weight=.2)
