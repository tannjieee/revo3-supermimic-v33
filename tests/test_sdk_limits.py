import copy
import json
import numpy as np
import pytest
import torch
from viserdex.supermimic.sdk_limits import (DEFAULT_LIMITS, load_sdk_limits, policy_limits,
                                          sampling_geometry_limits)
from viserdex.supermimic.grasp_bank import validate_record, GraspBank, SCHEMA
from viserdex.supermimic.policy_contract import absolute_joint_target
from viserdex.supermimic.core import manual_convex_config, Q, ROT
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.sensors import observe, initial_sensors
from viserdex.distillation.config import PerceptionNoiseCfg


def test_current_readback_anatomical_order_and_exact_degrees():
    limits, source = load_sdk_limits()
    expected = [[0, 110], [0, 105], [0, 75], [-10, 90], [-20, 90],
                [-10, 30], [-5, 90], [-10, 90], [-20, 90],
                [-20, 25], [-5, 90], [-10, 90], [-20, 90],
                [-25, 20], [-5, 90], [-10, 90], [-20, 90],
                [-30, 10], [-5, 90], [-10, 90], [-20, 90]]
    np.testing.assert_allclose(np.rad2deg(limits), expected, atol=1e-12)
    assert source['serial'] == 'BCUTR2291J2600001'


def test_nonzero_sdk_offsets_are_applied_before_reordering():
    data = json.loads(DEFAULT_LIMITS.read_text())
    offsets = np.arange(21) / 100
    actual = policy_limits(data['sdk_joint_names'], data['sdk_limits_deg'], offsets)
    base, _ = load_sdk_limits()
    np.testing.assert_allclose(actual[0], base[0] - .19)  # thumb CMP is SDK slot 19
    np.testing.assert_allclose(actual[17], base[17])  # little MPR is SDK slot 0


@pytest.mark.parametrize('field', ['nan', 'reversed', 'names', 'converted', 'layout', 'sign'])
def test_corrupt_or_unsupported_limit_file_is_rejected(tmp_path, field):
    data = json.loads(DEFAULT_LIMITS.read_text())
    if field == 'nan': data['sdk_limits_deg'][0][0] = float('nan')
    if field == 'reversed': data['joint_limits_rad'][0] = [1, 0]
    if field == 'names': data['sdk_joint_names'][0] = data['sdk_joint_names'][1]
    if field == 'converted': data['joint_limits_rad'][0][0] += .01
    if field == 'layout': data['layout']['version'] = 2
    if field == 'sign': data['configured_signs'][0] = -1
    path = tmp_path/'limits.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_sdk_limits(path)


def test_software_encoder_range_and_geometric_sampling_are_distinct():
    enc = np.tile([0., 1.], (21, 1))
    mechanical = np.tile([0., 1.05], (21, 1))
    bias = np.full(21, .1)
    geom = sampling_geometry_limits(enc, bias, mechanical)
    np.testing.assert_allclose(geom, np.tile([.1, 1.05], (21, 1)))
    r = dict(id='seed', source_group='seed', q_geometry=[1.04]*21,
             hold_target_encoder=[1.]*21, object_pose_xyzw=[.05, 0., .12, 0., 0., 0., 1.])
    contract = {'assets': {'geometry_sampling_limits_rad': geom.tolist()}}
    validate_record(r, enc, certified=False, contract=contract)
    bad = copy.deepcopy(r); bad['hold_target_encoder'][0] = 1.04
    with pytest.raises(ValueError, match='hold_target_encoder'):
        validate_record(bad, enc, certified=False, contract=contract)
    bad = copy.deepcopy(r); bad['q_geometry'][0] = .05
    with pytest.raises(ValueError, match='q_geometry'):
        validate_record(bad, enc, certified=False, contract=contract)
    with pytest.raises(ValueError): sampling_geometry_limits(enc, np.full(21, 2.), mechanical)


def test_action_endpoints_and_encoder_normalization_share_sdk_limits():
    limits = torch.tensor(load_sdk_limits()[0], dtype=torch.float32)[None].expand(3, -1, -1)
    action = torch.tensor([-1., 0., 1.])[:, None].expand(-1, 21)
    target = absolute_joint_target(action, limits)
    torch.testing.assert_close(target[0], limits[0, :, 0])
    torch.testing.assert_close(target[2], limits[0, :, 1])
    cfg = manual_convex_config()
    state = torch.zeros(3, cfg.state_dim)
    state[:, Q] = target + torch.tensor(cfg.encoder_bias_rad)
    state[:, ROT.stop-1] = state[:, COMPACT.goal_rot.stop-1] = 1
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    obs, _ = observe(state, initial_sensors(3, 'cpu', noise), limits, noise, cfg)
    torch.testing.assert_close(obs[:, :21], action, atol=2e-7, rtol=0)


def test_previous_bank_cannot_be_relabelled_as_sdk_certified(tmp_path):
    old = {'assets': {'joint_limits_rad': [[-2., 2.]]*21}}
    path = tmp_path/'bank.json'
    path.write_text(json.dumps(dict(schema=SCHEMA, contract=old, records=[])))
    new = {'assets': {'joint_limits_rad': load_sdk_limits()[0].tolist()}}
    with pytest.raises(ValueError, match='recertify'):
        GraspBank(path, new)
