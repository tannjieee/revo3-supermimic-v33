"""Calibrated camera, URDF palm and unchanged student-root protocol regression."""

import json
from pathlib import Path

import pytest
import torch
from scipy.spatial.transform import Rotation

from viserdex.distillation.config import PerceptionNoiseCfg, StudentCfg
from viserdex.distillation.frames import (
    CAMERA_FRAME, PALM_FRAME, STUDENT_POSE_FRAME, CalibratedPoseTransform,
    canonical_pose, pose_frame_metadata, validate_pose_metadata,
)
from viserdex.distillation.observations import StudentObservations
from test_distillation import fake_env, identity

CALIBRATION = Path(__file__).parents[1] / "config/cameras/hikrobot_DB0447679_1280x1024_extrinsics_revo3_20261022.json"


def test_actual_palm_origin_is_offset_in_student_frame():
    transform = CalibratedPoseTransform(CALIBRATION)
    position, quaternion = transform.to_student(torch.zeros(2, 3), identity(2), source_frame=PALM_FRAME)
    torch.testing.assert_close(position, torch.tensor([[0.0155, 0, 0.055]]).expand(2, 3))
    torch.testing.assert_close(quaternion, identity(2))
    assert transform.metadata["pose_frame"] == STUDENT_POSE_FRAME
    assert len(transform.metadata["calibration_sha256"]) == 64


@pytest.mark.parametrize("source_frame", [CAMERA_FRAME, PALM_FRAME])
@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_calibrated_pose_matches_independent_matrix_chain(source_frame, order):
    transform = CalibratedPoseTransform(CALIBRATION)
    matrix = transform.transforms[source_frame][0]
    expected = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    expected[:, :3, :3] = torch.tensor(Rotation.from_euler("xyz", [[20, -35, 110], [-70, 80, 12]], degrees=True).as_matrix())
    expected[:, :3, 3] = torch.tensor([[0.03, -0.06, 0.08], [-0.01, 0.09, 0.02]])
    measured = torch.linalg.inv(matrix) @ expected
    quat = torch.tensor(Rotation.from_matrix(measured[:, :3, :3].numpy()).as_quat())
    if order == "wxyz":
        quat = quat[:, [3, 0, 1, 2]]
    pos, orientation = transform.to_student(
        measured[:, :3, 3], quat, source_frame=source_frame, quaternion_order=order,
    )
    torch.testing.assert_close(pos, expected[:, :3, 3], atol=1e-12, rtol=0)
    actual_rotation = torch.tensor(Rotation.from_quat(orientation.numpy()).as_matrix())
    torch.testing.assert_close(actual_rotation, expected[:, :3, :3], atol=1e-12, rtol=0)
    # Already-converted input must not receive the base/palm offset a second time.
    second_pos, second_quat = transform.to_student(pos, orientation, source_frame=STUDENT_POSE_FRAME)
    torch.testing.assert_close(second_pos, pos, atol=0, rtol=0)
    torch.testing.assert_close(second_quat, orientation)


def test_adapter_converts_external_palm_and_camera_without_object_truth():
    transform = CalibratedPoseTransform(CALIBRATION)
    env = fake_env()
    adapter = StudentObservations(
        env, StudentCfg(), PerceptionNoiseCfg(enabled=False, joint_enabled=False), pose_transform=transform,
    )
    obs = {"policy": torch.zeros(2, 49), "privileged": torch.randn(2, 38)}
    # At the actual URDF palm origin, the student must see the fixed base offset.
    prop, ext, target = adapter(obs, (torch.zeros(2, 3), identity(2)), estimated_pose_frame=PALM_FRAME)
    torch.testing.assert_close(ext[:, :3], torch.tensor([[0.0155, 0, 0.055]]).expand(2, 3))
    camera_base = torch.linalg.inv(transform.transforms[CAMERA_FRAME][0])
    palm_base = transform.transforms[PALM_FRAME][0]
    camera_object = camera_base @ palm_base
    measured_pos = camera_object[:3, 3].float().repeat(2, 1)
    measured_quat = torch.tensor(Rotation.from_matrix(camera_object[:3, :3].numpy()).as_quat()).float().repeat(2, 1)
    _, camera_ext, _ = adapter(obs, (measured_pos, measured_quat), estimated_pose_frame=CAMERA_FRAME)
    torch.testing.assert_close(camera_ext, ext, atol=1e-6, rtol=1e-6)
    env.scene["object"].data.root_pos_w.torch += 10
    env.scene["object"].data.root_quat_w.torch[:] = torch.tensor([0, 0, 1, 0])
    obs["privileged"].fill_(999)
    prop2, ext2, target2 = adapter(obs, (measured_pos, measured_quat), estimated_pose_frame=CAMERA_FRAME)
    torch.testing.assert_close(prop2, prop)
    torch.testing.assert_close(ext2, camera_ext)
    assert not torch.equal(target, target2)


def test_adapter_refuses_unconverted_palm_without_calibration():
    adapter = StudentObservations(fake_env(), StudentCfg(), PerceptionNoiseCfg())
    with pytest.raises(ValueError, match="calibrated pose transform"):
        adapter({"policy": torch.zeros(2, 49), "privileged": torch.zeros(2, 38)},
                (torch.zeros(2, 3), identity(2)), estimated_pose_frame=PALM_FRAME)


@pytest.mark.parametrize("field,value", [
    ("pose_frame", PALM_FRAME), ("pose_frame", "palm"),
    ("quaternion_order", "wxyz"), ("position_unit", "mm"), ("pose_protocol", "unknown"),
])
def test_metadata_rejects_mixed_protocols(field, value):
    with pytest.raises(ValueError):
        validate_pose_metadata({**pose_frame_metadata(), field: value})


def test_legacy_palm_label_is_resolved_as_historical_root():
    actual = validate_pose_metadata({"pose_frame": "palm", "quaternion_order": "xyzw"})
    assert actual["pose_frame"] == STUDENT_POSE_FRAME
    assert "legacy_pose_frame_label" in actual
    with pytest.raises(ValueError):
        validate_pose_metadata({})


def test_reject_unknown_measurement_frame_and_invalid_quaternion():
    transform = CalibratedPoseTransform(CALIBRATION)
    with pytest.raises(ValueError, match="Unknown pose source frame"):
        transform.to_student(torch.zeros(2, 3), identity(2), source_frame="palm")
    with pytest.raises(ValueError, match="zero norm"):
        canonical_pose(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="non-finite"):
        canonical_pose(torch.full((2, 3), float("nan")), identity(2))


def test_reject_calibration_with_inconsistent_chain(tmp_path):
    data = json.loads(CALIBRATION.read_text())
    data["transforms"]["T_palm_camera"]["matrix"][0][3] += 0.01
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        CalibratedPoseTransform(path)
