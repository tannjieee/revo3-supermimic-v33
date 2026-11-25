"""Convex contact placement against the authored hand and 50 mm die."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('mujoco')
from viserdex.assets.prepare_manual_convex import MANUAL_USD
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.supermimic.object_variant import VARIANTS
from viserdex.supermimic.support_placement import SupportPlacement
from viserdex.supermimic.wuji_pose import HOME_POSITION


@pytest.fixture(scope='module')
def query():
    robot = MANUAL_USD.resolve()
    return SupportPlacement(robot, build_geometry_spec(robot, VARIANTS['die50']))


def test_randomized_resets_are_touching_without_penetration(query):
    rng = np.random.default_rng(43)
    for _ in range(256):
        q = np.asarray(HOME_POSITION)+rng.uniform(-.03,.03,21)
        p = np.array([.055,-.01,.10])+rng.uniform([0,-.02,-.02],[0,.02,.02])
        quat = Rotation.from_euler('xyz',rng.uniform(-3.14,3.14,3)).as_quat()
        placed, audit = query.place(q,p,quat,[1,0,0])
        np.testing.assert_allclose(placed[1:],p[1:],atol=1e-12)
        query.set_pose(q,placed,quat)
        assert -2e-7 <= query.all_gaps().min() <= 2.21e-6
        assert audit['normal_up_dot'] > 0
        # A small upward displacement must open the contact, not deepen it.
        query.set_pose(q,placed+np.array([1e-5,0,0]),quat)
        assert query.all_gaps().min() > audit['gap_m']


def test_rejects_unsupported_lateral_position(query):
    with pytest.raises(ValueError,match='No hand support'):
        query.place(HOME_POSITION,[.055,10,10],[0,0,0,1],[1,0,0])


def test_rejects_invalid_gravity(query):
    with pytest.raises(ValueError,match='upward direction'):
        query.place(HOME_POSITION,[.055,-.01,.1],[0,0,0,1],[0,0,0])


def test_world_xyzw_roundtrip_including_rotated_hand_and_translated_worlds(query):
    import torch
    from viserdex.distillation.noise import quat_apply, quat_conjugate, quat_mul
    rng = np.random.default_rng(43)
    base_xyzw = np.array([0., -2**-.5, 0., 2**-.5],dtype=np.float32)
    for i in range(256):
        base_pos = np.array([.12 + 2*(i % 32), 2*(i // 32), .51],dtype=np.float32)
        q = (np.asarray(HOME_POSITION)+rng.uniform(-.03,.03,21)).astype(np.float32)
        offset = rng.uniform([-.02,-.02,0],[.02,.02,0])
        world_position = base_pos+np.array([-.10,-.01,.055])+offset
        world_quat = Rotation.from_euler('xyz',rng.uniform(-3.14,3.14,3)).as_quat()
        pose = np.r_[world_position, world_quat].astype(np.float32)
        placed, _ = query.place_world(q, pose, base_pos, base_xyzw, [0,0,1])
        np.testing.assert_array_equal(placed[3:],pose[3:])
        np.testing.assert_allclose(placed[:2],pose[:2],atol=1e-12)
        # Independent XYZW tensor transform after float32 simulator storage.
        inverse = quat_conjugate(torch.tensor(base_xyzw,dtype=torch.float32))
        actual = torch.tensor(placed,dtype=torch.float32)
        local_pos = quat_apply(inverse,actual[:3]-torch.tensor(base_pos,dtype=torch.float32))
        local_quat = quat_mul(inverse,actual[3:])
        query.set_pose(q,local_pos.numpy(),local_quat.numpy())
        assert -5e-6 <= query.minimum_gap() <= 1e-5


def test_affected_saved_reset_is_rejected_instead_of_silently_reinterpreted():
    from viserdex.supermimic.support_placement import reset_joints_with_supported_object
    with pytest.raises(ValueError,match='wrong quaternion order'):
        reset_joints_with_supported_object(None,None,None,None,None)
