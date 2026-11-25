import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from viserdex.supermimic.training_limits import DEFAULT_TRAINING_LIMITS, load_training_limits
from viserdex.supermimic.sdk_limits import load_sdk_limits, sampling_geometry_limits
from viserdex.supermimic.identified_hand import load_identified_hand
from viserdex.supermimic.policy_contract import absolute_joint_target
from viserdex.supermimic.core import manual_convex_config, Q, ROT
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.sensors import observe, initial_sensors
from viserdex.distillation.config import PerceptionNoiseCfg


def test_exact_user_ranges_and_sdk_containment():
    limits, source = load_training_limits()
    expected = [[0,107],[0,102],[0,74],[0,85],[0,87],
                [-9.5,29.5],[0,85],[0,85],[0,85],
                [-19.5,24.5],[0,85],[0,85],[0,85],
                [-24.5,19.5],[0,85],[0,85],[0,85],
                [-29.5,9.5],[0,85],[0,85],[0,85]]
    np.testing.assert_allclose(np.rad2deg(limits), expected, atol=1e-12, rtol=0)
    sdk, sdk_source = load_sdk_limits()
    assert (limits[:,0] >= sdk[:,0]).all() and (limits[:,1] <= sdk[:,1]).all()
    assert source['sdk_software_limits'] == sdk_source
    assert source['identified_parameters']['sha256'] == load_identified_hand()[1]['sha256']['best_parameters.json']


def test_actual_urdf_contains_training_after_identified_coordinate_conversion():
    relative = 'assets/revo3_d405/vendor/brainco-description/revo3_system/urdf/revo3_right.urdf'
    path = next(p/relative for p in DEFAULT_TRAINING_LIMITS.parents if (p/relative).exists())
    joints = {j.get('name'):j for j in ET.parse(path).getroot().findall('joint')}
    names = json.loads(DEFAULT_TRAINING_LIMITS.read_text())['joint_names']
    mechanical = np.array([[float(joints[n].find('limit').get(k)) for k in ('lower','upper')] for n in names])
    limits, _ = load_training_limits()
    bias = np.array(load_identified_hand()[0]['encoder_bias_rad'])
    shifted = limits + bias[:,None]
    np.testing.assert_allclose(sampling_geometry_limits(limits, bias, mechanical), shifted, atol=1e-12)


@pytest.mark.parametrize('mutation', ['nan','reversed','conversion','sdk_overrun','wrong_sdk','wrong_fit','wrong_order'])
def test_invalid_or_mismatched_training_configuration_fails(tmp_path, mutation):
    data = json.loads(DEFAULT_TRAINING_LIMITS.read_text())
    if mutation == 'nan': data['joint_limits_deg'][0][1] = float('nan')
    if mutation == 'reversed': data['joint_limits_deg'][0] = [107,0]
    if mutation == 'conversion': data['joint_limits_rad'][0][1] += .01
    if mutation == 'sdk_overrun':
        data['joint_limits_deg'][0][1] = 111
        data['joint_limits_rad'][0][1] = float(np.deg2rad(111))
    if mutation == 'wrong_sdk': data['sdk_limits_sha256'] = 'wrong'
    if mutation == 'wrong_fit': data['identified_parameters']['sha256'] = 'wrong'
    if mutation == 'wrong_order': data['joint_names'] = data['joint_names'][::-1]
    path = tmp_path/'training_limits.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_training_limits(path)


def test_training_endpoints_and_normalization_use_same_independent_range():
    limits = torch.tensor(load_training_limits()[0],dtype=torch.float32)[None].expand(3,-1,-1)
    action = torch.tensor([-1.,0.,1.])[:,None].expand(-1,21)
    target = absolute_joint_target(action,limits)
    torch.testing.assert_close(target[0],limits[0,:,0])
    torch.testing.assert_close(target[2],limits[0,:,1])
    cfg = manual_convex_config();state = torch.zeros(3,cfg.state_dim)
    state[:,Q] = target + torch.tensor(cfg.encoder_bias_rad)
    state[:,ROT.stop-1] = state[:,COMPACT.goal_rot.stop-1] = 1
    noise = PerceptionNoiseCfg(enabled=False,joint_enabled=False)
    obs,_ = observe(state,initial_sensors(3,'cpu',noise),limits,noise,cfg)
    torch.testing.assert_close(obs[:,:21],action,atol=2e-7,rtol=0)
