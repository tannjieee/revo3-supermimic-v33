"""Control contract: independent scalar oracle, reset timing, limits and gradients."""
import math
import torch
from viserdex.supermimic.core import config20,action_target,EMA,TIME,Config

def test_offset_ema_matches_scalar_oracle_with_warmup_and_joint_limits():
 cfg=config20();torch.manual_seed(9)
 limits=torch.tensor([-.2,1.6]).expand(37,21,2).clone()
 # Home must lie inside each joint's own range.
 home=torch.tensor(cfg.action_home);limits[:,:,0]=torch.minimum(limits[:,:,0],home-.1);limits[:,:,1]=torch.maximum(limits[:,:,1],home+.1)
 states=torch.zeros(37,173);states[:,EMA]=home+torch.randn(37,21)*.1
 ticks=torch.arange(37);states[:,TIME]=(1-ticks.float()/400)[:,None]
 a=torch.randn(37,21)*2;actual=action_target(states,a,limits,cfg)
 expected=torch.empty_like(a)
 for i in range(37):
  for j in range(21):
   lo,hi=map(float,limits[i,j]);raw=max(lo,min(hi,float(home[j])+.5*max(-1,min(1,float(a[i,j])))))
   value=float(home[j]) if i<8 else .5*raw+.5*float(states[i,55+j])
   expected[i,j]=max(lo,min(hi,value))
 torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)

def test_warmup_blocks_policy_gradients_then_releases_at_step_eight():
 cfg=config20();s=torch.zeros(2,173);s[:,EMA]=torch.tensor(cfg.action_home)
 s[:,TIME]=torch.tensor([[1-7/400],[1-8/400]])
 a=torch.full((2,21),.1,requires_grad=True);lim=torch.tensor([-3.,3.]).expand(2,21,2)
 action_target(s,a,lim,cfg).sum().backward()
 torch.testing.assert_close(a.grad[0],torch.zeros(21));torch.testing.assert_close(a.grad[1],torch.full((21,),.25))

def test_target_memory_reaches_home_and_preserves_full_offset_range():
 cfg=config20();s=torch.zeros(1,173);home=torch.tensor(cfg.action_home);s[:,EMA]=home;s[:,TIME]=.8
 lim=torch.stack((home-1,home+1),-1)[None]
 for _ in range(30):s[:,EMA]=action_target(s,torch.ones(1,21),lim,cfg)
 torch.testing.assert_close(s[:,EMA],(home+.5)[None],atol=3e-7,rtol=1e-6)
 s[:,TIME]=1;s[:,EMA]=home
 torch.testing.assert_close(action_target(s,-torch.ones(1,21),lim,cfg),home[None])

def test_invalid_home_contract_is_rejected():
 import pytest
 with pytest.raises(ValueError,match='21 finite'):Config(action_mode='home_offset',action_home=(1.,))
