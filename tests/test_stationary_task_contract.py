"""Intervention checks for the no-clock single-pose training contract."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from test_single_goal import fixture, evaluate
from test_single_goal_command import fixture as command_fixture
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import POS, ROT, VEL, OMEGA, rotation_vector_quat
from viserdex.supermimic.sensors import initial_sensors
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.single_goal import (
    dense_orientation_reward, dense_position_reward, dense_stability_reward,
)
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.trainer import Trainer


def test_compact_state_does_not_access_episode_clock_or_success_tracker():
    cfg, state, action = fixture(2)
    adapter = Adapter.__new__(Adapter)
    adapter.hold_cfg = cfg
    adapter.raw_history = action
    adapter.delay_ready = torch.ones(2, 1)
    adapter.base_inverse = torch.tensor([[0., 0., 0., 1.]]).expand(2, -1)
    adapter.physical = lambda: state[:, :55]
    origins = torch.tensor([[0., 0., 0.], [2., 3., 0.]])
    adapter.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=origins))
    command = torch.cat((state[:, COMPACT.goal_pos], state[:, COMPACT.goal_rot]), -1)
    # Deliberately no episode_length_buf, max_episode_length, common_step_counter,
    # goal clock or get_term(): accessing any of them would fail this contract.
    adapter.env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins),
        command_manager=SimpleNamespace(get_command=lambda name: command))
    actual = adapter.read_state()
    expected = state.clone()
    expected[:, COMPACT.delay_ready] = 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_reward_callbacks_ignore_elapsed_time_and_success_history():
    command = command_fixture()
    command.object.data.root_quat_w.torch[:] = rotation_vector_quat(torch.tensor([.08, -.04, .03]))
    command.object.data.root_pos_w.torch[:] = torch.tensor([.02, 0., 0.])
    command.object.data.root_ang_vel_w.torch[:] = torch.tensor([.1, .2, -.1])
    command.object.data.root_lin_vel_w.torch[:] = torch.tensor([.01, -.02, 0.])
    callbacks = (dense_orientation_reward, dense_position_reward, dense_stability_reward)
    values = []
    for step, start, marker, completed, ticks in (
        (1, 0, 10., 0, 0), (180, 0, .1, 1, 10),
        (200, 199, 1000., 0, 9), (1_000_000, 0, 10., 1, 0),
    ):
        command._env.common_step_counter = step
        command.goal_start_step[:] = start
        command.cfg.goal_evaluation_s = marker
        command.metrics['consecutive_success'][:] = completed
        command.hold_ticks[:] = ticks
        values.append(torch.stack([func(command._env) for func in callbacks]))
    assert (values[0] > 0).all()
    for actual in values[1:]:
        torch.testing.assert_close(actual, values[0], atol=0, rtol=0)


@pytest.mark.parametrize('evaluation_seconds,episode_steps', [(.05, 1), (3.37, 20), (10000., 400000)])
def test_rollout_values_and_actor_gradients_ignore_time_metadata(evaluation_seconds, episode_steps):
    torch.manual_seed(314)
    cfg, state, _ = fixture(2)
    cfg = replace(cfg, wm_hidden=(16,), actor_hidden=(16,), critic_hidden=(16,),
                  actor_batch=2, critic_minibatches=1)
    trainer = Trainer(cfg, PerceptionNoiseCfg(), torch.ones(2, 65),
        torch.tensor([[-1., 1.]]).expand(2, 21, 2),
        torch.tensor([[0., 0., 0., 1.]]).expand(2, -1), 'cpu', 400)
    # Exercise a nonzero action -> physical-state path, not just action costs.
    torch.nn.init.normal_(trainer.wm.net[-1].weight, std=.001)
    trainer.wm.delta_norm.variance.fill_(1e-4)
    batch = dict(state=state, history=torch.zeros(2, 10, 81), env_id=torch.arange(2),
                 goal_elapsed_s=torch.zeros(2))
    batch.update({'sensor/'+k: v for k, v in initial_sensors(2, 'cpu', trainer.noise_cfg).items()})
    def run():
        torch.manual_seed(1729)  # identical sensor noise realizations
        loss, states, targets, context, valid = trainer.imagine(batch)
        grads = torch.autograd.grad(loss, tuple(trainer.actor.parameters()))
        assert valid.all() and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
        return (loss, states, targets, context, valid, *grads)
    baseline = run()
    trainer.cfg = replace(cfg, goal_evaluation_s=evaluation_seconds)
    trainer.episode_steps = episode_steps
    batch['goal_elapsed_s'][:] = float('nan')  # cannot affect a learned computation
    for actual, expected in zip(run(), baseline):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('rotvec,distance,speed', [
    ([0., 0., 0.], 0., 0.), ([.15, -.08, .02], .025, .02),
    ([3.14158, 0., 0.], .01, 0.),
])
def test_complete_reward_has_finite_correct_physical_gradients(rotvec, distance, speed):
    cfg, state, action = fixture()
    state, action = state.double(), action.double()
    variables = torch.tensor(rotvec+[distance, 0., 0.]+[speed]*6,
                             dtype=torch.float64, requires_grad=True)
    def function(x):
        physical = state[:, :55].clone()
        physical[:, ROT] = rotation_vector_quat(x[:3])
        physical[:, POS] = state[:, COMPACT.goal_pos] + x[3:6]
        physical[:, OMEGA] = x[6:9]
        physical[:, VEL] = x[9:12]
        return evaluate(state, physical, action, cfg)
    assert torch.autograd.gradcheck(function, (variables,), eps=1e-7, atol=1e-5, rtol=1e-4)
