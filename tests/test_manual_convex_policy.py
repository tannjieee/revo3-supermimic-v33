"""Contract tests with independent rotation/scalar oracles; no hardware sessions."""
from dataclasses import asdict, replace
import math
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from viserdex.supermimic.core import (
    Config, manual_convex_config, action_target, Q, QD, POS, ROT, GOAL_POS, GOAL_ROT, TIME, BASE_POS, EMA,
)
from viserdex.supermimic.policy_contract import (
    DIRECT_OBSERVATION_PROTOCOL, absolute_joint_target, rotation_6d_xyzw, direct_observation, observation_fields,
)
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.sensors import initial_sensors, observe
from viserdex.supermimic.inference import ManualConvexPolicy, export_actor
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.assets.prepare_manual_convex import default_manual_source, inspect_manual_asset

from viserdex.supermimic.state_layout import COMPACT
GOAL_POS, GOAL_ROT = COMPACT.goal_pos, COMPACT.goal_rot


def limits():
    return torch.tensor(inspect_manual_asset(default_manual_source())['limits_rad'], dtype=torch.float32)[None]


def state():
    s = torch.zeros(2, COMPACT.dim)
    lim = limits()
    s[:, Q] = lim.mean(-1) + torch.tensor(manual_convex_config().encoder_bias_rad)
    s[:, QD] = torch.linspace(-10., 10., 21)
    s[:, ROT] = torch.tensor(Rotation.from_euler('xyz', [[70, -20, 179], [-30, 60, -90]], degrees=True).as_quat())
    s[:, GOAL_ROT] = torch.tensor(Rotation.from_euler('xyz', [[-40, 80, 15], [15, -10, 175]], degrees=True).as_quat())
    s[:, POS] = torch.tensor([.05, -.01, .10])
    s[:, GOAL_POS] = torch.tensor([.03, .02, .11])
    s[:, COMPACT.last_action] = torch.arange(21).float() / 100
    return s


def test_manual_asset_full_range_and_collision_count():
    report = inspect_manual_asset(default_manual_source())
    assert report['collision_instances'] == 144 and report['rigid_bodies'] == 22
    assert report['joint_names'] == list(REVO3_JOINT_NAMES)
    # Upstream 2a98621: mechanical bounds are independent of SDK/training limits.
    expected_deg = [[0, 110], [0, 106], [0, 75], [-10, 90], [-20, 90]]
    for mpr in ([-10, 30], [-20, 25], [-25, 20], [-30, 10]):
        expected_deg.extend([mpr, [-5, 90], [-10, 90], [-20, 90]])
    np.testing.assert_allclose(report['limits_rad'], np.deg2rad(expected_deg), atol=1e-10, rtol=0)


def test_absolute_decode_endpoints_midpoint_and_no_state_dependence():
    cfg, lim = manual_convex_config(), limits()
    for a in (-1., 0., 1.):
        action = torch.full((2, 21), a)
        expected = lim[..., 0] + (a + 1) / 2 * (lim[..., 1] - lim[..., 0])
        s = state()
        torch.testing.assert_close(action_target(s, action, lim, cfg), expected.expand(2, -1))
        s[:, COMPACT.last_action] = -1000
        s[:, COMPACT.delay_ready] = 1  # no initialization-dependent command override
        torch.testing.assert_close(action_target(s, action, lim, cfg), expected.expand(2, -1))


def test_decoder_has_no_hidden_clamp_or_gradient_attenuation():
    a = torch.full((1, 21), 1.1, requires_grad=True)
    lim = limits()
    target = absolute_joint_target(a, lim)
    assert (target > lim[..., 1]).all()  # production Actor bounds a; decoder is strictly affine
    target.sum().backward()
    torch.testing.assert_close(a.grad, (lim[..., 1] - lim[..., 0]) / 2)


def test_6d_matches_scipy_matrix_columns_and_quaternion_sign():
    q = Rotation.random(128, random_state=17).as_quat()
    expected = Rotation.from_quat(q).as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    actual = rotation_6d_xyzw(torch.tensor(q))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    torch.testing.assert_close(actual, rotation_6d_xyzw(torch.tensor(-q)))
    near_pi = Rotation.from_rotvec([[0, 0, math.pi - 1e-6], [0, 0, math.pi + 1e-6]]).as_quat()
    six = rotation_6d_xyzw(torch.tensor(near_pi))
    assert (six[0] - six[1]).abs().max() < 3e-6


def test_new_observation_layout_velocity_scale_last_action_and_no_base():
    cfg, s, lim = manual_convex_config(), state(), limits()
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    sensor = initial_sensors(2, 'cpu', noise)
    obs, _ = observe(s, sensor, lim, noise, cfg)
    assert obs.shape == (2, 81) and observation_fields()[-1]['stop'] == 81
    torch.testing.assert_close(obs[:, :21], torch.zeros(2, 21), atol=2e-7, rtol=0)
    torch.testing.assert_close(obs[:, 21:42], s[:, QD] * .1)
    torch.testing.assert_close(obs[:, 42:63], s[:, COMPACT.last_action])
    torch.testing.assert_close(obs[:, 63:66], s[:, POS])
    torch.testing.assert_close(obs[:, 66:69], s[:, GOAL_POS])
    expected_relative = (Rotation.from_quat(s[:, ROT]) * Rotation.from_quat(s[:, GOAL_ROT]).inv()).as_matrix()
    np.testing.assert_allclose(obs[:, 75:81], expected_relative[:, :, :2].transpose(0, 2, 1).reshape(2, 6), atol=3e-7)
    assert s.shape[-1] == 84  # base position and older actions are absent
    s[:, COMPACT.delay_ready] = 1  # actuator-only memory does not enter Actor
    other, _ = observe(s, sensor, lim, noise, cfg)
    torch.testing.assert_close(obs, other)


def test_position_ablation_and_scaled_velocity_config():
    cfg, s = replace(manual_convex_config(), include_object_position=False, joint_velocity_scale=.2), state()
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    sensor = initial_sensors(2, 'cpu', noise)
    obs, _ = observe(s, sensor, limits(), noise, cfg)
    assert obs.shape[-1] == cfg.observation_dim == 78
    torch.testing.assert_close(obs[:, 21:42], s[:, QD] * .2)
    s[:, POS] = -100
    changed, _ = observe(s, sensor, limits(), noise, cfg)
    torch.testing.assert_close(changed, obs)


def test_joint_velocity_uses_same_encoder_delay_and_reset_as_position():
    cfg, s = manual_convex_config(), state()
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=True, joint_noise_std=0., joint_bias_std=0., joint_delay_max_steps=1)
    sensor = initial_sensors(2, 'cpu', noise)
    sensor['delay'][:] = 1
    first, sensor = observe(s, sensor, limits(), noise, cfg)
    s[:, QD] = 123
    second, sensor = observe(s, sensor, limits(), noise, cfg)
    torch.testing.assert_close(first[:, 21:42], second[:, 21:42])
    third, _ = observe(s, sensor, limits(), noise, cfg)
    torch.testing.assert_close(third[:, 21:42], torch.full((2, 21), 12.3))


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_reject_invalid_velocity_scale(bad):
    with pytest.raises(ValueError, match='joint_velocity_scale'):
        manual_convex_config(joint_velocity_scale=bad)


def make_checkpoint(path):
    torch.manual_seed(9)
    cfg = manual_convex_config(actor_hidden=(32, 16))
    actor = Actor(cfg)
    meta = dict(state_protocol=cfg.state_protocol, observation_protocol=cfg.observation_protocol, observation_dim=cfg.observation_dim,
                quaternion_order='XYZW', joint_names=list(REVO3_JOINT_NAMES), joint_limits_rad=limits()[0].tolist())
    payload = dict(version=25, iteration=2, config=asdict(cfg), noise_config=asdict(PerceptionNoiseCfg()), actor=actor.state_dict(), metadata=meta)
    torch.save(payload, path)
    return payload, actor


def test_export_inference_matches_training_with_history_and_last_action(tmp_path):
    source, exported = tmp_path/'full.pt', tmp_path/'actor.pt'
    payload, actor = make_checkpoint(source)
    export_actor(source, exported)
    policy = ManualConvexPolicy(exported)
    assert torch.load(exported, weights_only=False)['metadata']['actor_only_export']
    s = state()[0]
    previous = torch.zeros(21)
    for i in range(12):
        target, raw, obs = policy.propose(s[Q] - torch.tensor(policy.cfg.encoder_bias_rad), s[QD] + i, s[POS], s[ROT], s[GOAL_ROT], goal_position_base_m=s[GOAL_POS])
        np.testing.assert_allclose(obs[42:63], previous, atol=0)
        with torch.no_grad():
            expected = actor(policy.history)
        np.testing.assert_allclose(raw, expected[0], atol=0)
        np.testing.assert_allclose(target, absolute_joint_target(expected, limits())[0], atol=0)
        previous = torch.tensor(raw)
        policy.commit()
    policy.reset()
    _, _, obs = policy.propose(s[Q] - torch.tensor(policy.cfg.encoder_bias_rad), s[QD], s[POS], s[ROT], s[GOAL_ROT], goal_position_base_m=s[GOAL_POS])
    assert not obs[42:63].any()
    with pytest.raises(RuntimeError, match='committed'):
        policy.propose(s[Q] - torch.tensor(policy.cfg.encoder_bias_rad), s[QD], s[POS], s[ROT], s[GOAL_ROT])


@pytest.mark.parametrize('old_version', [17, 18, 19, 20, 21, 22, 23, 24])
def test_reject_legacy_checkpoint_and_wrong_quaternion_order(tmp_path, old_version):
    path = tmp_path/'actor.pt'
    payload, _ = make_checkpoint(path)
    policy = ManualConvexPolicy(path)
    s = state()[0]
    with pytest.raises(ValueError, match='WXYZ'):
        policy.propose(s[Q] - torch.tensor(policy.cfg.encoder_bias_rad), s[QD], s[POS], s[ROT], s[GOAL_ROT], quaternion_order='WXYZ')
    payload['version'] = old_version
    torch.save(payload, path)
    with pytest.raises(ValueError, match='version-25'):
        ManualConvexPolicy(path)


def test_legacy_observation_and_actor_dimensions_are_unchanged():
    cfg = Config(actor_hidden=(32, 16))
    assert cfg.observation_dim == 124
    assert Actor(cfg)(torch.zeros(2, 10, 124)).shape == (2, 21)
