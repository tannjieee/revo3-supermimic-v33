"""Official command callbacks with synthetic measured poses; no physics or motors."""
from types import SimpleNamespace
import torch
from test_hold_command import command_fixture
from viserdex.supermimic.single_goal import (
    SingleGoalCommand, dense_orientation_reward, dense_position_reward, dense_stability_reward,
)
from viserdex.supermimic.core import manual_convex_config
from viserdex.supermimic.position_reward import POSITION_CONFIG_FIELDS
from viserdex.supermimic.stability_reward import STABILITY_CONFIG_FIELDS


def fixture():
    original = command_fixture()
    command = SingleGoalCommand.__new__(SingleGoalCommand)
    command.__dict__.update(original.__dict__)
    command._env.num_envs = 2
    command._env.device = 'cpu'
    command._env.extras = {}
    command.cfg.goal_evaluation_s = 10.
    command.cfg.resampling_time_range = (1e6, 1e6)
    cfg = manual_convex_config()
    for name in ('orientation_coarse_sigma_rad', 'orientation_fine_sigma_rad',
                 'orientation_coarse_fraction', 'orientation_reward_scale', *POSITION_CONFIG_FIELDS, *STABILITY_CONFIG_FIELDS):
        setattr(command.cfg, name, getattr(cfg, name))
    command.goal_start_step = torch.zeros(2, dtype=torch.long)
    command.time_left = torch.zeros(2)
    command.command_counter = torch.zeros(2, dtype=torch.long)
    command.metrics['success_rate'] = torch.zeros(2)
    command.last_position_error = torch.zeros(2)
    command.last_linear_speed = torch.zeros(2)
    command.object.data.root_lin_vel_w = SimpleNamespace(torch=torch.zeros(2, 3))
    command.pos_command_e = command.pos_command_w.clone()
    command.default_pos_command_e = command.pos_command_e.clone()
    command._env.scene = SimpleNamespace(env_origins=torch.zeros(2, 3))
    command.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        root_quat_w=SimpleNamespace(torch=command.quat_command_w.clone())))
    command._env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command


def test_success_and_clock_never_replace_goal_or_repeat_event():
    c = fixture()
    goal = c.quat_command_w.clone()
    events = torch.zeros(2)
    for step in range(1, 501):
        c._env.common_step_counter = step
        event = c.evaluate_step().clone()
        events += event
        c.compute(.05)
        c._update_command()
        torch.testing.assert_close(c.quat_command_w, goal)
        torch.testing.assert_close(c.evaluate_step(), event)
    assert events.tolist() == [1, 1]
    assert c.elapsed_seconds.tolist() == [25, 25]
    assert c.hold_ticks.tolist() == [10, 10]
    c._env.common_step_counter = 1_000_000_000
    c.compute(.05)
    torch.testing.assert_close(c.quat_command_w, goal)


def test_dense_rewards_are_active_from_first_step_and_sampled_after_physics():
    c = fixture()
    for step in (1, 179, 180, 199, 200, 500):
        c._env.common_step_counter = step
        c.object.data.root_ang_vel_w.torch[:] = torch.tensor([0., 0., .5])
        torch.testing.assert_close(dense_orientation_reward(c._env), torch.full((2,), 10.))
        torch.testing.assert_close(dense_stability_reward(c._env), torch.full((2,), 3/2.718281828459045))
        torch.testing.assert_close(dense_position_reward(c._env), torch.full((2,), 1.))


def test_partial_reset_or_explicit_command_restarts_only_selected_clock():
    c = fixture()
    c._env.common_step_counter = 250
    c.metrics['consecutive_success'][:] = 1
    c.hold_ticks[:] = 10
    goal = c.quat_command_w.clone()
    c.reset([0])
    assert c.elapsed_seconds.tolist() == [0, 12.5]
    assert c.metrics['consecutive_success'].tolist() == [0, 1]
    assert c.hold_ticks.tolist() == [0, 10]
    assert c._env.extras['log']['Metrics/success_rate'] == 1
    torch.testing.assert_close(c.quat_command_w[1], goal[1])
    c.set_goal([1], goal[1:], torch.zeros(1, 3))
    assert c.elapsed_seconds.tolist() == [0, 0]
    assert c.metrics['consecutive_success'].tolist() == [0, 0]


def test_base_frame_pose_command_updates_both_position_buffers_and_reset_restores_default():
    c = fixture()
    c._env.scene.env_origins[1] = torch.tensor([2., 3., 0.])
    c.robot.data.root_pos_w.torch[1] = torch.tensor([2., 3., .5])
    c.robot.data.root_quat_w.torch[1] = torch.tensor([0., 0., 2**-.5, 2**-.5])
    c._env.common_step_counter = 250
    c.set_goal([1], torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[.1, 0., .2]]))
    torch.testing.assert_close(c.pos_command_w[1], torch.tensor([2., 3.1, .7]))
    torch.testing.assert_close(c.pos_command_e[1], torch.tensor([0., .1, .7]))
    torch.testing.assert_close(c.quat_command_w[1], c.robot.data.root_quat_w.torch[1])
    assert c.elapsed_seconds.tolist() == [12.5, 0]
    c.reset([1])
    torch.testing.assert_close(c.pos_command_e[1], c.default_pos_command_e[1])
    torch.testing.assert_close(c.pos_command_w[1], c.default_pos_command_e[1]+c._env.scene.env_origins[1])


def test_physics_success_checks_position_and_linear_speed():
    c = fixture()
    c.hold_ticks[:] = 9
    c.object.data.root_pos_w.torch[0, 0] = .03
    c.object.data.root_lin_vel_w.torch[1, 0] = .06
    assert not c.evaluate_step().any()
    assert c.hold_ticks.tolist() == [0, 0]
    c._env.common_step_counter = 180
    c.evaluate_step()
    torch.testing.assert_close(c.last_linear_speed, torch.tensor([0., .06]))


def test_new_command_updates_compact_state_and_latest_observation_without_new_sensors():
    from viserdex.supermimic.simulation import Adapter
    from viserdex.supermimic.state_layout import COMPACT
    from viserdex.supermimic.core import rotation_features
    c = fixture()
    adapter = Adapter.__new__(Adapter)
    adapter.hold_cfg = manual_convex_config()
    adapter.sl = COMPACT
    adapter.state = torch.zeros(2, 84)
    adapter.state[:, COMPACT.last_action] = .123
    adapter.state[:, COMPACT.delay_ready] = 1
    adapter.state[:, COMPACT.goal_rot] = torch.tensor([0.,0.,0.,1.])
    adapter.history = torch.zeros(2, 10, 81)
    obj_q = torch.tensor([[.1, .2, .3, .9]]).repeat(2, 1)
    obj_q /= obj_q.norm(dim=-1, keepdim=True)
    adapter.history[:, :, 69:75] = rotation_features(obj_q)[:, None]
    def read():
        result = adapter.state.clone()
        result[:, COMPACT.goal_rot] = c.quat_command_w
        result[:, COMPACT.goal_pos] = c.pos_command_w
        return result
    adapter.read_state = read
    c._env.supermimic_adapter = adapter
    old_state, old_history = adapter.state.clone(), adapter.history.clone()
    c.set_goal([0], [[0., 0., 0., 1.]], [[.02, .03, .1]])
    torch.testing.assert_close(adapter.state[1], old_state[1])
    torch.testing.assert_close(adapter.state[:, COMPACT.last_action], old_state[:, COMPACT.last_action])
    torch.testing.assert_close(adapter.state[:, COMPACT.delay_ready], old_state[:, COMPACT.delay_ready])
    torch.testing.assert_close(adapter.history[:, :-1], old_history[:, :-1])
    torch.testing.assert_close(adapter.history[1], old_history[1])
    torch.testing.assert_close(adapter.history[0, -1, :63], old_history[0, -1, :63])
    torch.testing.assert_close(adapter.history[0, -1, 66:69], torch.tensor([.02,.03,.1]))
    torch.testing.assert_close(adapter.history[0, -1, -6:], rotation_features(obj_q)[0])
