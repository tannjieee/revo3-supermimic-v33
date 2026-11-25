"""The original GS and ray cameras must share calibrated rectified pixel rays."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viserdex.distillation.hikrobot_images import HikrobotImageProcessor
from viserdex.renderer.hikrobot_calibration import load_calibration
from viserdex.renderer.hikrobot_occlusion import calibrated_geometry, audit_extrinsic_precision


CALIBRATION = Path(__file__).parents[1] / "config/cameras"


def test_gs_and_hand_pattern_parameters_recover_exact_rectified_intrinsics():
    k, _, p, _ = calibrated_geometry(CALIBRATION, [0, 0, 0], [0, 0, 0, 1])
    assert all(type(v) is float for n, v in p.items() if n not in ("width", "height"))
    fx = p["width"] * p["focal_length"] / p["horizontal_aperture"]
    fy = p["height"] * p["focal_length"] / p["vertical_aperture"]
    recovered = np.array([[fx, 0, p["width"] / 2 + p["horizontal_aperture_offset"] * fx],
                          [0, fy, p["height"] / 2 + p["vertical_aperture_offset"] * fy], [0, 0, 1]])
    np.testing.assert_allclose(recovered, k, atol=1e-12, rtol=0)
    intrinsics, _, _ = load_calibration(CALIBRATION)
    np.testing.assert_allclose(k, HikrobotImageProcessor(intrinsics).intrinsics.numpy(), atol=0, rtol=0)


def test_camera_pose_composes_the_delivered_base_extrinsics_once():
    position = np.array([0.1, 0.2, 0.5])
    rotation = Rotation.from_euler("xyz", [15, -40, 80], degrees=True)
    _, world, _, _ = calibrated_geometry(CALIBRATION, position, rotation.as_quat())
    _, _, transforms = load_calibration(CALIBRATION)
    expected = np.eye(4)
    expected[:3, :3] = rotation.as_matrix()
    expected[:3, 3] = position
    np.testing.assert_allclose(world, expected @ transforms["T_base_camera"], atol=1e-12, rtol=0)


def test_large_environment_origins_accept_float32_rounding_but_reject_pose_errors():
    target = np.eye(4)
    target[:3, 3] = [15.364744466322055, -30.935058, 0.800568]
    actual = target.astype(np.float32).astype(np.float64)
    actual[0, 3] = 15.36474323272705
    tolerance = audit_extrinsic_precision(actual, target)
    assert np.max(tolerance) < 5e-6
    shifted = actual.copy()
    shifted[0, 3] += 0.001
    with pytest.raises(AssertionError):
        audit_extrinsic_precision(shifted, target)
    rotated = actual.copy()
    rotated[:3, :3] = Rotation.from_euler("x", 0.01).as_matrix()
    with pytest.raises(AssertionError):
        audit_extrinsic_precision(rotated, target)
