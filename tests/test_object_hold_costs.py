"""Independent cost oracles, full gradients, and historical Critic compatibility."""
from dataclasses import asdict, replace
import math
import pytest
import torch

from viserdex.supermimic.core import Config, POS, ROT, VEL, OMEGA, rotation_vector_quat
from viserdex.supermimic.constraints import cost_names, violation_costs, object_hold_gate
from viserdex.supermimic.networks import Critic
from viserdex.supermimic.stability_reward import task_reward_bound
from viserdex.supermimic.state_layout import COMPACT
from test_hand_center_contract import fixture
from test_single_goal import evaluate


@pytest.mark.parametrize('angle_deg', [0., 5., 15., 30., 90.])
def test_scalar_oracle_for_gated_motion_and_ungated_drop(angle_deg):
    cfg, state, limits = fixture(1)
    nxt = state[:, :55].clone()
    nxt[:, ROT] = rotation_vector_quat(torch.tensor([[math.radians(angle_deg), 0., 0.]]))
    nxt[:, OMEGA.start] = 1.  # Twice 0.5 rad/s.
    nxt[:, VEL.start] = .15  # Three times 0.05 m/s.
    nxt[:, POS.start] += .6  # Twice the hand-center reach radius.
    costs, ratios = violation_costs(state, nxt, torch.zeros(1, 21), limits, cfg)
    gate = math.exp(-(angle_deg / 15.)**2)
    torch.testing.assert_close(costs, torch.tensor([[0., 0., gate, 4*gate, 1.]]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(ratios[:, 2:], torch.tensor([[2., 3., 2.]]))
    assert cost_names(cfg) == ('joint_speed', 'target_rate', 'object_angular_speed', 'object_linear_speed', 'drop')


@pytest.mark.parametrize('ratio', [0., .5, 1., 1.5, 2.])
def test_speed_cost_has_deadband_and_finite_directional_gradient(ratio):
    cfg, state, limits = fixture(1)
    nxt = state[:, :55].clone()
    nxt[:, OMEGA.start] = ratio * .5
    nxt[:, VEL.start] = ratio * .05
    nxt.requires_grad_()
    costs, _ = violation_costs(state, nxt, torch.zeros(1, 21), limits, cfg)
    expected = max(ratio - 1., 0.)**2
    torch.testing.assert_close(costs[:, 2:4], torch.full((1, 2), expected))
    costs[:, 2:].sum().backward()
    assert torch.isfinite(nxt.grad).all()
    for slot, threshold in ((OMEGA.start, .5), (VEL.start, .05)):
        assert nxt.grad[0, slot].item() == pytest.approx(2 * max(ratio - 1., 0.) / threshold, abs=2e-5)


def test_hand_translation_inside_reach_and_goal_position_cannot_change_cost():
    cfg, state, limits = fixture(3)
    state[:, COMPACT.goal_pos] = torch.tensor([[0., 0., 0.], [3., 2., 1.], [-1., -.1, .1]])
    nxt = state[:, :55].clone()
    nxt[:, POS] += torch.tensor([[0., 0., 0.], [.05, -.03, .02], [-.1, .1, -.1]])
    costs, _ = violation_costs(state, nxt, torch.zeros(3, 21), limits, cfg)
    assert torch.equal(costs, torch.zeros(3, 5))
    nxt.requires_grad_()
    values = evaluate(state, nxt, torch.zeros(3, 21), cfg)
    torch.testing.assert_close(values, torch.full((3,), .4))
    assert task_reward_bound(cfg) == pytest.approx(.4)


def test_drop_gradient_points_back_toward_hand_center_and_has_no_angle_gate():
    cfg, state, limits = fixture(1)
    nxt = state[:, :55].clone()
    nxt[:, POS.start] += .6
    nxt[:, ROT] = torch.tensor([[0., 0., 1., 0.]])
    nxt.requires_grad_()
    costs, _ = violation_costs(state, nxt, torch.zeros(1, 21), limits, cfg)
    assert costs[0, 4].item() == pytest.approx(1.)
    costs[0, 4].backward()
    assert nxt.grad[0, POS.start].item() == pytest.approx(2/.3)
    assert torch.isfinite(nxt.grad).all()


def test_gate_is_smooth_and_success_threshold_is_not_a_switch():
    cfg, state, _ = fixture(1)
    angle = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    nxt = state[:, :55].double().clone()
    nxt[:, ROT] = torch.stack((torch.sin(angle/2), angle*0, angle*0, torch.cos(angle/2)))[None]
    gate = object_hold_gate(state.double(), nxt, cfg)
    derivative, = torch.autograd.grad(gate.sum(), angle)
    sigma = math.pi/12
    assert gate.item() == pytest.approx(math.exp(-(.1/sigma)**2))
    assert derivative.item() == pytest.approx(-2*.1/sigma**2*gate.item())


def test_new_reward_does_not_duplicate_object_speed_costs():
    cfg, state, limits = fixture(2)
    nxt = state[:, :55].clone()
    nxt[1, OMEGA.start] = 1.
    nxt[1, VEL.start] = .1
    action = torch.zeros(2, 21)
    torch.testing.assert_close(evaluate(state, nxt, action, cfg), torch.full((2,), .4))
    costs, _ = violation_costs(state, nxt, action, limits, cfg)
    torch.testing.assert_close(costs[:, 2:4], torch.tensor([[0., 0.], [1., 1.]]))


def test_historical_two_cost_config_loads_original_three_head_weights():
    cfg, state, _ = fixture(2)
    old = replace(cfg, object_hold_costs_enabled=False, stability_reward_scale=3.)
    original = Critic(65, old)
    saved = asdict(old)
    saved.pop('object_hold_costs_enabled')
    saved.pop('cost_object_hold_angle_sigma_rad')
    restored_cfg = Config(**saved)
    restored = Critic(65, restored_cfg)
    restored.load_state_dict(original.state_dict(), strict=True)
    context = torch.ones(2, 65)
    torch.testing.assert_close(restored(state, context), original(state, context), atol=0, rtol=0)
    assert restored(state, context).shape == (2, 3)
    assert Critic(65, cfg)(state, context).shape == (2, 6)
    with pytest.raises(RuntimeError, match='size mismatch'):
        Critic(65, cfg).load_state_dict(original.state_dict(), strict=True)


@pytest.mark.parametrize('kwargs', [
    dict(stability_reward_scale=3.), dict(track_object_position=True),
    dict(constraints_enabled=False), dict(object_hold_costs_enabled=1),
    dict(cost_object_hold_angle_sigma_rad=0.), dict(cost_object_hold_angle_sigma_rad=float('nan')),
])
def test_inconsistent_cost_profile_is_rejected(kwargs):
    cfg, _, _ = fixture(1)
    with pytest.raises(ValueError):
        replace(cfg, **kwargs)
