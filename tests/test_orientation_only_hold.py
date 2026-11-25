"""Behavioral oracles for free-position orientation holding and legacy replay."""
from dataclasses import asdict, replace
import math

import pytest
import torch

from viserdex.distillation.environment import contract_sha256
from viserdex.supermimic.core import Config, hand_center_config, POS, ROT, VEL, OMEGA
from viserdex.supermimic.position_reward import position_reward, position_hold_valid
from viserdex.supermimic.single_goal import dense_orientation_reward, dense_position_reward, dense_stability_reward
from viserdex.supermimic.stability_reward import task_reward_bound
from viserdex.supermimic.spatial_contract import drop_distance
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.trainer import imagined_continuation
from test_distillation_binding import _environment
from test_single_goal import fixture, evaluate
from test_single_goal_command import fixture as command_fixture


def test_current_profile_removes_position_objective_and_its_gradient():
    cfg = hand_center_config()
    assert cfg.track_object_position is False
    _, state, action = fixture(3)
    state[:, COMPACT.goal_pos] = torch.tensor([[.02, .01, .1], [.20, -.1, .03], [-.3, .2, .2]])
    physical = state[:, :55].clone()
    physical[:, POS] += torch.tensor([[0., 0., 0.], [.05, 0., 0.], [0., -.06, 0.]])
    physical.requires_grad_()
    values = evaluate(state, physical, action, cfg)
    torch.testing.assert_close(values, torch.full((3,), .4))
    values.sum().backward()
    assert torch.equal(physical.grad[:, POS], torch.zeros(3, 3))
    assert task_reward_bound(cfg) == pytest.approx(.4)
    assert torch.equal(position_reward(torch.tensor([0., .1, 10.]), cfg), torch.zeros(3))
    legacy = replace(cfg, track_object_position=True, object_hold_costs_enabled=False, stability_reward_scale=3.)
    assert task_reward_bound(legacy) == pytest.approx(.56)
    assert not torch.allclose(evaluate(state, physical, action, legacy), values)


@pytest.mark.parametrize('field,speed', [(VEL, .05), (OMEGA, .5)])
def test_historical_settling_reward_and_gradients_still_penalize_object_motion(field, speed):
    cfg = hand_center_config(object_hold_costs_enabled=False, stability_reward_scale=3.)
    _, state, action = fixture()
    physical = state[:, :55].clone()
    physical[:, field.start] = speed
    physical.requires_grad_()
    value = evaluate(state, physical, action, cfg)
    assert value.item() == pytest.approx(.04 * (10 + 3 / math.e))
    value.sum().backward()
    assert physical.grad[0, field.start] < 0


def orientation_command():
    c = command_fixture()
    c.cfg.track_object_position = False
    c.cfg.stability_reward_scale = 0.
    c.cfg.hold_duration = .52
    c._env.step_dt = .04
    # Substantial displacement from the old target, while still in hand reach.
    c.object.data.root_pos_w.torch[:] = torch.tensor([.05, -.03, .1])
    return c


def test_real_command_requires_13_stable_samples_ignores_position_and_holds_forever():
    c = orientation_command()
    cfg = hand_center_config()
    _, state, action = fixture(2)
    state[:, COMPACT.goal_pos] = c.pos_command_w
    physical = state[:, :55].clone()
    physical[:, POS] = c.object.data.root_pos_w.torch
    physical[:, ROT] = c.object.data.root_quat_w.torch
    goal = c.quat_command_w.clone()
    for step in range(1, 31):
        c._env.common_step_counter = step
        events = c.evaluate_step()
        assert events.tolist() == [step == 13, step == 13]
        assert c.hold_ticks.tolist() == [min(step, 13)] * 2
        assert torch.equal(c.quat_command_w, goal)
        real = .04 * sum(func(c._env) for func in
            (dense_orientation_reward, dense_position_reward, dense_stability_reward))
        torch.testing.assert_close(real, evaluate(state, physical, action, cfg))


@pytest.mark.parametrize('failure', ['linear_speed', 'angular_speed', 'angle', 'drop', 'timeout'])
def test_success_rejects_unstable_or_terminal_sample(failure):
    c = orientation_command()
    c.hold_ticks[:] = 12
    if failure == 'linear_speed':
        c.object.data.root_lin_vel_w.torch[:, 0] = .06
    elif failure == 'angular_speed':
        c.object.data.root_ang_vel_w.torch[:, 0] = .6
    elif failure == 'angle':
        c.object.data.root_quat_w.torch[:] = torch.tensor([math.sin(.1), 0., 0., math.cos(.1)])
    else:
        # Official lifecycle fills reset_buf for termination or collection timeout.
        c._env.reset_buf[:] = True
    c._env.common_step_counter = 1
    assert not c.evaluate_step().any()
    assert c.hold_ticks.tolist() == [0, 0]


def test_drop_boundary_and_imagined_survival_remain_independent_of_position_target():
    cfg = hand_center_config()
    _, state, _ = fixture(2)
    state[:, COMPACT.goal_pos] = torch.tensor([1., 1., 1.])
    physical = state[:, :55].clone()
    physical[:, POS] = torch.tensor(cfg.hand_center_base_m)
    physical[1, POS.start] += .4
    assert (drop_distance(physical, cfg) > cfg.drop_radius_m).tolist() == [False, True]
    alive = imagined_continuation(state, physical, cfg.survival_width, single_goal=True, cfg=cfg)
    assert alive[0] > .999 and alive[1] < .0001
    assert position_hold_valid(torch.tensor([1., 10.]), torch.tensor([0., .06]), cfg).tolist() == [True, False]


def test_historical_checkpoint_and_environment_hash_keep_old_tracking_semantics():
    saved = asdict(hand_center_config())
    saved.pop('track_object_position')
    saved.pop('object_hold_costs_enabled')
    saved.pop('cost_object_hold_angle_sigma_rad')
    assert Config(**saved).track_object_position is True
    env = _environment()
    old_hash = contract_sha256(env)
    env['commands']['object_pose']['track_object_position'] = True
    assert contract_sha256(env) == old_hash
    env['commands']['object_pose']['track_object_position'] = False
    assert contract_sha256(env) != old_hash
