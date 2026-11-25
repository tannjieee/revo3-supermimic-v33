"""Default pose-command metadata and explicit saved-command requests."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from viserdex.supermimic.goal_contract import goal_position_metadata, verify_saved_goal_position


def config(explicit=None):
    rotation=Rotation.from_euler('xyz',[.3,-.7,1.2])
    base=np.array([1.3,-2.5,.51])
    local=np.array([.03425,-.00856,.10169])
    return NS(commands=NS(object_pose=NS(goal_position_base_m=explicit)),scene=NS(
        robot=NS(init_state=NS(pos=tuple(base),rot=tuple(rotation.as_quat()))),
        object=NS(init_state=NS(pos=tuple(base+rotation.apply(local))))))


def test_nominal_command_uses_object_default_pose_in_base_frame():
    cfg=config()
    result=goal_position_metadata(cfg)
    np.testing.assert_allclose(result['default_goal_position_base_m'],[.03425,-.00856,.10169],atol=1e-15)
    assert result['default_goal_position_source']=='object_default_root_pose'
    assert all(type(v) is float for v in result['default_goal_position_base_m'])


def test_explicit_command_is_not_the_object_spawn_or_network_origin():
    cfg=config((.2,.1,.05))
    assert goal_position_metadata(cfg)==dict(default_goal_position_base_m=[.2,.1,.05],default_goal_position_source='explicit_config')


def test_inherited_command_offset_is_applied_in_world_axes_before_base_conversion():
    cfg=config()
    cfg.commands.object_pose.init_pos_offset=(.01,-.02,.03)
    expected=np.array([.03425,-.00856,.10169])+Rotation.from_quat(cfg.scene.robot.init_state.rot).inv().apply([.01,-.02,.03])
    result=goal_position_metadata(cfg)
    np.testing.assert_allclose(result['default_goal_position_base_m'],expected,atol=1e-15)
    assert result['default_goal_position_source']=='object_default_root_pose_plus_command_offset'


def test_explicit_base_command_replaces_inherited_offset():
    cfg=config((.2,.1,.05))
    cfg.commands.object_pose.init_pos_offset=(.01,-.02,.03)
    assert goal_position_metadata(cfg)['default_goal_position_base_m']==[.2,.1,.05]


@pytest.mark.parametrize('explicit',[None,(.2,.1,.05)])
def test_matching_resume_request_does_not_rewrite_saved_none_or_tuple(explicit):
    cfg=config(explicit)
    before=deepcopy(cfg)
    verify_saved_goal_position(cfg,goal_position_metadata(cfg)['default_goal_position_base_m'])
    assert cfg==before
    with pytest.raises(ValueError,match='cannot override'):
        verify_saved_goal_position(cfg,[.055,-.01,.1])
    assert cfg==before


@pytest.mark.parametrize('bad',[[float('nan'),0,0],[0,float('inf'),0],[0,0]])
def test_bad_requests_and_configurations_are_rejected(bad):
    with pytest.raises(ValueError,match='three finite'):
        verify_saved_goal_position(config(),bad)
    with pytest.raises(ValueError,match='three finite'):
        goal_position_metadata(config(bad))
