import math
from dataclasses import replace
import pytest
import torch
from viserdex.supermimic.core import hand_center_config
from viserdex.supermimic.warmup_excitation import HomeExcitation


def fixture():
    cfg=hand_center_config(warmup_joint_amplitude_rad=math.radians(2),wm_only_iters=1000,
        absolute_policy_optimization=True,stable_value_learning=True,wm_task_losses=True)
    limits=torch.tensor(cfg.network_joint_limits).expand(32,-1,-1).clone()
    return cfg,limits,HomeExcitation(limits,cfg)


def test_bounded_physical_excitation_speed_and_partial_reset():
    torch.manual_seed(71);cfg,limits,exc=fixture();targets=[]
    for step in range(400):
        action=exc.action(step//3)
        target=(limits[...,0]+limits[...,1])/2+action*(limits[...,1]-limits[...,0])/2
        targets.append(target)
        assert action.abs().max()<=1
    targets=torch.stack(targets)
    torch.testing.assert_close(targets[0],exc.home.expand(32,-1),atol=2e-7,rtol=0)
    assert (targets-exc.home).abs().max()<=math.radians(2)+2e-7
    # Quintic derivative max=1.875; end-to-end distance <=2*amplitude.
    max_speed=2*math.radians(2)*1.875/(30*cfg.control_dt)
    assert (targets[1:]-targets[:-1]).abs().max()/cfg.control_dt<=max_speed+1e-5
    assert targets.std(dim=0).mean()>math.radians(.2)
    old_tick=exc.tick.clone();old_start=exc.start.clone();old_end=exc.end.clone()
    done=torch.arange(32)<16;exc.reset(done)
    assert not exc.tick[done].any() and not exc.start[done].any() and not exc.end[done].any()
    assert torch.equal(exc.tick[~done],old_tick[~done]) and torch.equal(exc.start[~done],old_start[~done]) and torch.equal(exc.end[~done],old_end[~done])
    action=exc.action(150)
    target=(limits[...,0]+limits[...,1])/2+action*(limits[...,1]-limits[...,0])/2
    torch.testing.assert_close(target[done],exc.home.expand(16,-1),atol=2e-7,rtol=0)


def test_end_of_warmup_returns_home_and_refuses_policy_phase():
    cfg,limits,exc=fixture();exc.action(0)
    a=exc.action(cfg.wm_only_iters-1)
    target=(limits[...,0]+limits[...,1])/2+a*(limits[...,1]-limits[...,0])/2
    torch.testing.assert_close(target,exc.home.expand(32,-1),atol=2e-7,rtol=0)
    with pytest.raises(ValueError,match='warmup phase'):exc.action(cfg.wm_only_iters)


@pytest.mark.parametrize('changes',[{'warmup_joint_amplitude_rad':float('nan')},{'warmup_joint_amplitude_rad':-.1},{'warmup_joint_segment_steps':1},{'warmup_joint_segment_steps':2.5},{'wm_only_iters':0}])
def test_invalid_warmup_contract_rejected(changes):
    cfg,_,_=fixture()
    with pytest.raises(ValueError):replace(cfg,**changes)
