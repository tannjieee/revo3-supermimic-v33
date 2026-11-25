"""Position goals: gradient behavior, joint success and inference command lifecycle."""
import math
import pytest
import torch
from viserdex.supermimic.core import POS, VEL, OMEGA, persistent_hold_transition
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.position_reward import position_hold_valid
GOAL_POS = COMPACT.goal_pos
from viserdex.supermimic.position_reward import position_reward
from viserdex.supermimic.core import manual_convex_config
from viserdex.supermimic.inference import ManualConvexPolicy
from test_manual_convex_policy import make_checkpoint
from test_single_goal import fixture, evaluate


@pytest.mark.parametrize('distance', [0., .005, .01, .02, .06, .11])
def test_position_kernel_scalar_oracle_and_finite_gradient(distance):
    cfg = manual_convex_config()
    p = torch.tensor([distance, 0., 0.], dtype=torch.float64, requires_grad=True)
    value = position_reward(p.norm(), cfg)
    value.backward()
    excess = max(distance - .01, 0.)
    expected = math.exp(-(excess/.05)**2)
    assert value.item() == pytest.approx(expected, abs=1e-12)
    assert p.grad[0].item() == pytest.approx(-2*excess/.05**2*expected, abs=1e-12)
    assert torch.isfinite(p.grad).all()


def test_position_and_stability_use_post_physics_error_from_first_step():
    cfg, s, a = fixture(3)
    physical = s[:, :55].clone()
    physical[:, POS] = s[:, GOAL_POS]
    physical[1, POS] += torch.tensor([.06, 0., 0.])
    physical[2, VEL] = torch.tensor([.1, 0., 0.])
    torch.testing.assert_close(evaluate(s, physical, a, cfg),
        torch.tensor([.7, .05*(10+1/math.e+3*math.exp(-6.25)), .05*(11+3*math.exp(-4))]))
    # Changing only the task command changes reward, not physical state.
    s[1, GOAL_POS] = physical[1, POS]
    assert evaluate(s, physical, a, cfg)[1].item() == pytest.approx(.7)


def test_pose_success_requires_position_and_linear_speed_and_can_recover():
    cfg, s, a = fixture(3)
    ticks = torch.full((len(s),), 9.)
    physical = s[:, :55].clone()
    physical[1, POS] += torch.tensor([.03, 0., 0.])
    physical[2, VEL] = torch.tensor([.06, 0., 0.])
    ticks, event = persistent_hold_transition(ticks, torch.zeros(len(s)), physical[:, OMEGA].norm(dim=-1), .05,
        already_completed=torch.zeros(len(s), dtype=torch.bool),
        blocked=~position_hold_valid((physical[:, POS]-s[:, GOAL_POS]).norm(dim=-1), physical[:, VEL].norm(dim=-1), cfg))
    assert event.tolist() == [True, False, False]
    assert ticks.tolist() == [10, 0, 0]
    physical[:, POS] = s[:, GOAL_POS]
    physical[:, VEL] = 0
    completed = event.clone()
    recovered = torch.zeros_like(completed)
    for _ in range(10):
        ticks, event = persistent_hold_transition(ticks, torch.zeros(len(s)), torch.zeros(len(s)), .05,
            already_completed=completed,
            blocked=~position_hold_valid((physical[:, POS]-s[:, GOAL_POS]).norm(dim=-1), physical[:, VEL].norm(dim=-1), cfg))
        recovered |= event
        completed |= event
    assert recovered.tolist() == [False, True, True]
    assert completed.all()


def test_inference_goal_position_is_absolute_and_new_position_preserves_last_action(tmp_path):
    path = tmp_path/'actor.pt'
    make_checkpoint(path)
    policy = ManualConvexPolicy(path)
    q = [0., 0., 0., 1.]
    p = [.03, .02, .11]
    policy.set_goal(q, p)
    for step in range(3):
        _, _, obs = policy.propose([0.]*21, [0.]*21, [.06, -.01, .1], q,
                                  [0., 0., 0., -1.], goal_position_base_m=p)
        torch.testing.assert_close(torch.tensor(obs[66:69]), torch.tensor(p))
        policy.commit()
    last_action = policy.state[:, COMPACT.last_action].clone()
    p[0] += .01
    _, _, obs = policy.propose([0.]*21, [0.]*21, [.06, -.01, .1], q, q, goal_position_base_m=p)
    torch.testing.assert_close(torch.tensor(obs[42:63]), last_action[0])
    torch.testing.assert_close(torch.tensor(obs[66:69]), torch.tensor(p))
    policy.commit()
    old = policy.state.clone()
    with pytest.raises(ValueError):
        policy.set_goal(q, [float('nan'), 0, 0])
    torch.testing.assert_close(policy.state, old)
    with pytest.raises(ValueError, match='both'):
        policy.propose([0.]*21, [0.]*21, [.06, -.01, .1], q, q)


@pytest.mark.parametrize('kwargs', [
    {'position_sigma_m': 0}, {'position_reward_scale': float('nan')},
    {'position_tolerance_m': -.01}, {'hold_position_threshold_m': .005},
    {'hold_linear_speed_threshold_m_s': 0}, {'stability_linear_speed_scale': 0},
])
def test_invalid_position_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        manual_convex_config(**kwargs)
