"""Measured reward and controller-reset checks using official Isaac Lab callbacks."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from viserdex.supermimic.direct_actions import DirectJointPositionAction
from viserdex.supermimic.quiet_hold import fresh_options, imagined_motion_penalty
from viserdex.supermimic.single_goal import dense_stability_reward


@pytest.mark.parametrize('v33', [False, True])
def test_target_history_partial_reset_matches_imagined_reward(v33):
    from viserdex.supermimic.core import hand_center_config
    from test_dex_reference_v33 import config
    cfg = replace(config() if v33 else hand_center_config(), **fresh_options())
    action = DirectJointPositionAction.__new__(DirectJointPositionAction)
    action.cfg = SimpleNamespace(dex_reference_profile=cfg.control_profile if v33 else None, alpha=cfg.ema_alpha)
    action._raw_actions = torch.zeros(2, 21)
    action._processed_actions = torch.zeros(2, 21)
    action.previous_processed_actions = torch.zeros(2, 21)
    action._quiet_target_ready = torch.zeros(2, 1, dtype=torch.bool)
    action.encoder_bias = torch.tensor(cfg.encoder_bias_rad)
    action._joint_ids = torch.arange(21)
    home = torch.tensor(cfg.action_home).repeat(2, 1)
    action._asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=home, joint_pos=home.clone()))
    action.position_limits = torch.tensor(cfg.network_joint_limits).repeat(2, 1, 1)
    action.zero_bias = torch.zeros(2, 21)
    action.ema_encoder = home.clone()
    action.reset()
    for step in range(4):
        if step == 2:
            unchanged = action.processed_actions[1].clone()
            action.reset([0])
            torch.testing.assert_close(action.processed_actions[1], unchanged)
        state = torch.zeros(2, cfg.state_dim)
        state[:, :21] = home
        state[:, 55:76] = action.raw_actions
        if v33:
            state[:, 83:104] = action.ema_encoder
            state[:, 104:125] = action.zero_bias
        else:
            state[:, 83:84] = action._quiet_target_ready
        raw = torch.full((2, 21), .1*step)
        action.process_actions(raw)
        expected = ((action.processed_actions-action.previous_processed_actions)/(.04*cfg.hold_target_speed_scale_rad_s)).square().mean(-1)
        torch.testing.assert_close(imagined_motion_penalty(state, state[:, :55], raw, torch.zeros(2), cfg), expected)


@pytest.mark.parametrize('wrapped', [False, True])
def test_measured_callback_penalizes_quiet_motion_and_keeps_reward_after_success(wrapped):
    from test_single_goal_command import fixture
    c = fixture()
    for name, value in fresh_options().items(): setattr(c.cfg, name, value)
    c.cfg.stability_reward_scale = 10.
    c.cfg.stability_angular_speed_scale = .3
    c.cfg.stability_linear_speed_scale = .02
    qd = torch.full((2, 21), math.pi/9)
    c._env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(
        _asset=SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=qd) if wrapped else qd)),
        _joint_ids=torch.arange(21), processed_actions=torch.full((2, 21), .05*math.pi/9),
        previous_processed_actions=torch.zeros(2, 21)))
    for step in (1, 20, 100, 250):
        c._env.common_step_counter = step
        # Object stationary at the goal: +10 stability, -1 joint motion, -1 command rate.
        torch.testing.assert_close(dense_stability_reward(c._env), torch.full((2,), 8.))
