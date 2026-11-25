"""Exercise the real callback class without starting a simulator or optimizer."""
from types import SimpleNamespace
import torch
from viserdex.supermimic.hold_command import HoldReorientCommand


def command_fixture():
    command = HoldReorientCommand.__new__(HoldReorientCommand)
    command._env = SimpleNamespace(common_step_counter=0, step_dt=.05,
                                   reset_buf=torch.tensor([False, False]))
    command.cfg = SimpleNamespace(orientation_success_threshold=.1,
                                  hold_speed_threshold=.5, hold_duration=.5)
    quat = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
    command.object = SimpleNamespace(data=SimpleNamespace(
        root_quat_w=SimpleNamespace(torch=quat.clone()),
        root_ang_vel_w=SimpleNamespace(torch=torch.zeros(2, 3)),
        root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3))))
    command.quat_command_w = quat
    command.pos_command_w = torch.zeros(2, 3)
    command.hold_ticks = torch.zeros(2)
    command.just_completed = torch.zeros(2, dtype=torch.bool)
    command.last_angle = torch.zeros(2)
    command.last_speed = torch.zeros(2)
    command.metrics = {'consecutive_success': torch.zeros(2)}
    command._evaluated_step = -1
    return command


def test_reward_callback_counts_once_per_physics_control_step():
    command = command_fixture()
    for step in range(10):
        command._env.common_step_counter = step
        pulse = command.evaluate_step().clone()
        ticks = command.hold_ticks.clone()
        count = command.metrics['consecutive_success'].clone()
        for _ in range(3):
            torch.testing.assert_close(command.evaluate_step(), pulse)
            torch.testing.assert_close(command.hold_ticks, ticks)
            torch.testing.assert_close(command.metrics['consecutive_success'], count)
        assert pulse.tolist() == [step == 9, step == 9]
    assert command.metrics['consecutive_success'].tolist() == [1, 1]


def test_metric_refresh_cannot_earn_success_and_reset_blocks_only_selected_env():
    command = command_fixture()
    command.hold_ticks[:] = 9
    command._env.reset_buf[0] = True
    for _ in range(3):
        command._update_metrics()
    assert not command.metrics['consecutive_success'].any()
    command.evaluate_step()
    assert command.just_completed.tolist() == [False, True]
    assert command.metrics['consecutive_success'].tolist() == [0, 1]
    assert command.hold_ticks.tolist() == [0, 0]
