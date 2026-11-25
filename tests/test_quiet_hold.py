"""Independent reward oracles, EMA semantics, gradients and saved-run compatibility."""
import copy
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from viserdex.supermimic.core import hand_center_config, reward_from_orientation_error
from viserdex.supermimic.orientation_reward import orientation_reward
from viserdex.supermimic.stability_reward import stability_reward
from viserdex.supermimic.quiet_hold import (
    LEGACY_DEFAULTS, fresh_options, motion_penalty, imagined_motion_penalty,
    configure_fresh_environment, validate_episode_timing,
)


def config():
    return hand_center_config(**fresh_options(), stability_reward_scale=10.,
        stability_angular_speed_scale=.3, stability_linear_speed_scale=.02)


def test_tolerance_plateau_and_far_field_arrival_gradient():
    c = config()
    assert c.orientation_tolerance_rad == math.radians(10.)
    assert c.orientation_coarse_fraction == 1.
    angle = torch.tensor([0., math.radians(5), math.radians(10), math.radians(10.1), .3, 1., 2.], dtype=torch.float64, requires_grad=True)
    value = orientation_reward(angle, c)
    torch.testing.assert_close(value[:3], torch.full((3,), 10., dtype=torch.float64))
    assert (value[3:-1] > value[4:]).all()
    value.sum().backward()
    assert (angle.grad[:3] == 0).all()
    assert (angle.grad[3:] < 0).all() and torch.isfinite(angle.grad).all()


@pytest.mark.parametrize('degrees', [0., 9.9, 10., 10.1, 30., 90., 180.])
def test_single_exponential_matches_scalar_oracle_and_saved_v1_is_unchanged(degrees):
    from viserdex.supermimic.quiet_hold import reward_protocol, PROTOCOL, SINGLE_EXP_PROTOCOL
    c = config()
    angle = torch.tensor(math.radians(degrees), dtype=torch.float64, requires_grad=True)
    value = orientation_reward(angle, c)
    excess = math.radians(max(degrees-10., 0.))
    expected = 10*math.exp(-(excess/(math.pi/2))**2)
    assert value.item() == pytest.approx(expected, abs=1e-12)
    value.backward()
    assert angle.grad.item() == pytest.approx(-2*excess/(math.pi/2)**2*expected, abs=1e-12)
    assert reward_protocol(c) == SINGLE_EXP_PROTOCOL
    old = replace(c, orientation_tolerance_rad=.1, orientation_coarse_fraction=.3)
    e = max(angle.item()-.1, 0.)
    expected_old = 10*(.3*math.exp(-(e/(math.pi/2))**2)+.7*math.exp(-(e/(math.pi/18))**2))
    assert orientation_reward(angle, old).item() == pytest.approx(expected_old, abs=1e-12)
    assert reward_protocol(old) == PROTOCOL


@pytest.mark.parametrize('angle,omega,speed', [(0., 0., 0.), (.05, .3, .02), (.3, .1, .01), (1.5, 0., 0.)])
def test_stability_scalar_oracle_and_velocity_gradients(angle, omega, speed):
    c = config()
    values = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (angle, omega, speed)]
    actual = stability_reward(values[0], torch.tensor(0.), values[1], values[2], c)
    expected = 10*math.exp(-(max(angle-math.radians(10), 0)/(math.pi/12))**2-(omega/.3)**2-(speed/.02)**2)
    assert actual.item() == pytest.approx(expected, rel=1e-12)
    actual.backward()
    for value in values:
        assert torch.isfinite(value.grad) and value.grad <= 0


def test_near_goal_motion_penalty_units_gate_and_gradient():
    c = config()
    speed = c.hold_joint_speed_scale_rad_s
    qd = torch.full((2, 21), speed, requires_grad=True)
    previous = torch.ones(2, 21)
    target = (previous + c.control_dt*speed).requires_grad_()
    angle = torch.tensor([.05, 1.5])
    value = motion_penalty(angle, qd, target, previous, c.control_dt, c)
    assert value[0].item() == pytest.approx(2., rel=1e-5)
    assert 0 <= value[1].item() < 1e-10
    value.sum().backward()
    assert (qd.grad[0] > 0).all() and (target.grad[0] > 0).all()
    assert motion_penalty(angle, qd.detach()*0, previous, previous, c.control_dt, c).count_nonzero() == 0


def test_v33_uses_executed_ema_target_and_geometric_velocity():
    from test_dex_reference_v33 import fixture
    from viserdex.supermimic import dex_reference as dex
    t, s = fixture()
    c = replace(t.cfg, **fresh_options(), stability_reward_scale=10.)
    action = torch.full((4, 21), .1, requires_grad=True)
    physical = s[:, :55].clone()
    physical[:, 21:42] = .02
    angle = torch.full((4,), .05)
    raw_target = t.limits.mean(-1)+action*(t.limits[..., 1]-t.limits[..., 0])/2
    delta = .51*(raw_target-s[:, dex.EMA])
    expected = (.02/c.hold_joint_speed_scale_rad_s)**2 + (delta/(.04*c.hold_target_speed_scale_rad_s)).square().mean(-1)
    actual = imagined_motion_penalty(s, physical, action, angle, c)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    # Arbitrary previous raw policy outputs do not change executed-target history.
    changed = s.clone(); changed[:, 55:76] = .9
    torch.testing.assert_close(imagined_motion_penalty(changed, physical, action, angle, c), actual)


def test_reward_is_continuous_hold_each_step_and_has_correct_dt():
    from test_dex_reference_v33 import fixture
    t, s = fixture()
    c = replace(t.cfg, **fresh_options(), stability_reward_scale=10.)
    # Decode the previous EMA exactly: no change of commanded target.
    from viserdex.supermimic.dex_reference import EMA
    action = (s[:, EMA]-t.limits.mean(-1))/((t.limits[..., 1]-t.limits[..., 0])/2)
    angle = torch.full((4,), .05)
    for _ in range(3):
        actual = reward_from_orientation_error(s, s[:, :55], action, angle, dt=.04,
            joint_vel_weight=0., action_rate_weight=0., hold_cfg=c)
        torch.testing.assert_close(actual, torch.full((4,), .8))


@pytest.mark.parametrize('hz', [20, 25, 30])
def test_ten_second_episode_and_legacy_duration(hz):
    c = SimpleNamespace(commands=SimpleNamespace(object_pose=SimpleNamespace()), episode_length_s=20.)
    configure_fresh_environment(c)
    assert c.episode_length_s == 10. and c.commands.object_pose.stability_reward_scale == 10.
    assert c.commands.object_pose.orientation_tolerance_rad == math.radians(10.)
    assert c.commands.object_pose.orientation_coarse_fraction == 1.
    validate_episode_timing(1/hz, 10*hz, compact=True)
    validate_episode_timing(1/hz, 20*hz, compact=True)
    with pytest.raises(ValueError): validate_episode_timing(1/hz, 20*hz-1, compact=True)
    with pytest.raises(ValueError): validate_episode_timing(1/hz, 10*hz, compact=False)


def test_legacy_config_and_environment_hash_remain_compatible():
    from viserdex.distillation.environment import contract_sha256
    from test_distillation_binding import _environment
    old = asdict(hand_center_config())
    for name in LEGACY_DEFAULTS: old.pop(name)
    assert type(hand_center_config())(**old) == hand_center_config()
    before = _environment(); after = copy.deepcopy(before)
    after['commands']['object_pose'].update(LEGACY_DEFAULTS)
    assert contract_sha256(before) == contract_sha256(after)
    after['commands']['object_pose'].update(fresh_options())
    assert contract_sha256(before) != contract_sha256(after)


def test_signed_quiet_reward_rejects_bounded_positive_critic():
    with pytest.raises(ValueError, match='signed rewards'):
        hand_center_config(**fresh_options(), wm_task_losses=True, stable_value_learning=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_quiet_reward_world_model_cuda_capture_matches_eager():
    from test_dex_reference_v33 import fixture, window
    from viserdex.supermimic.trainer import Trainer
    cpu, state = fixture()
    cpu.cfg = replace(cpu.cfg, **fresh_options(), stability_reward_scale=10.)
    eager = Trainer(cpu.cfg, cpu.noise_cfg, cpu.context.cuda(), cpu.limits.cuda(), cpu.base_inverse.cuda(), 'cuda')
    captured = copy.deepcopy(eager); captured.wm_cuda_graph = True
    w = {k: v.cuda() for k, v in window(cpu, state).items()}
    class Buffer:
        def sample(self, *args, **kw): return w
        def update_priorities(self, *args): pass
    for step in range(2):
        w['next_physical'][:, :, 21:42] = step*.01
        a, b = eager.wm_update(Buffer()), captured.wm_update(Buffer())
        for name in a: assert b[name] == pytest.approx(a[name], rel=3e-5, abs=3e-6), name
        for name, x in eager.wm.state_dict().items():
            torch.testing.assert_close(x, captured.wm.state_dict()[name], rtol=3e-5, atol=3e-6)
