"""Independent camera-model checks for the RGB actually supplied to the student CNN."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

from viserdex.distillation.hikrobot_images import HikrobotImageProcessor, diagnostic_keypoints_uvd
from viserdex.distillation.vision import keypoints_to_frame_pose


@pytest.fixture
def intrinsics():
    return json.loads((Path(__file__).parents[1] / "config/cameras/hikrobot_DB0447679_1280x1024_intrinsics.json").read_text())


def test_rectification_grid_matches_opencv_calibration(intrinsics):
    processor = HikrobotImageProcessor(intrinsics)
    map_x, map_y = cv2.initUndistortRectifyMap(
        np.asarray(intrinsics["camera_matrix"]), np.asarray(intrinsics["dist_coeffs"]),
        None, processor.intrinsics.numpy(), processor.image_size, cv2.CV_32FC1,
    )
    grid = processor.grid[0].double()
    actual_x = (grid[..., 0] + 1) * intrinsics["width"] / 2 - 0.5
    actual_y = (grid[..., 1] + 1) * intrinsics["height"] / 2 - 0.5
    np.testing.assert_allclose(actual_x, map_x, atol=8e-5, rtol=0)
    np.testing.assert_allclose(actual_y, map_y, atol=8e-5, rtol=0)
    assert processor.metadata["crop"] is None
    assert not processor.metadata["object_state_used"]


def test_resize_uses_full_frame_pixel_centers_and_preserves_rgb(intrinsics):
    calibration = copy.deepcopy(intrinsics)
    calibration["dist_coeffs"] = [0] * 5
    processor = HikrobotImageProcessor(calibration)
    y, x = torch.meshgrid(torch.arange(1024), torch.arange(1280), indexing="ij")
    rgb = torch.stack((x // 5, y // 4, torch.full_like(x, 37)), -1).byte()[None]
    actual = processor(rgb)
    expected = F.interpolate(rgb.permute(0, 3, 1, 2).float() / 255, size=(120, 120), mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    assert actual.shape == (1, 3, 120, 120)
    torch.testing.assert_close(actual[:, 2], torch.full((1, 120, 120), 37 / 255))
    matrix = processor.intrinsics
    assert matrix[0, 2] == pytest.approx((intrinsics["cx"] + 0.5) * 120 / 1280 - 0.5)
    assert matrix[1, 2] == pytest.approx((intrinsics["cy"] + 0.5) * 120 / 1024 - 0.5)


def test_rectified_uvd_pose_round_trip_with_calibrated_camera(intrinsics):
    processor = HikrobotImageProcessor(intrinsics)
    extrinsics = json.loads((Path(__file__).parents[1] / "config/cameras/hikrobot_DB0447679_1280x1024_extrinsics_revo3_20261022.json").read_text())
    camera = torch.tensor(extrinsics["transforms"]["T_base_camera"]["matrix"], dtype=torch.float64)
    camera_rotation, camera_position = camera[:3, :3][None], camera[:3, 3][None]
    object_rotation = torch.tensor(Rotation.from_euler("xyz", [15, 30, -20], degrees=True).as_matrix())[None]
    object_position = torch.tensor([[0.03, -0.02, 0.01]], dtype=torch.float64)
    reference = torch.cat((torch.zeros(1, 3), torch.cartesian_prod(*[torch.tensor([-0.02925, 0.02925])] * 3))).double()
    uvd = diagnostic_keypoints_uvd(reference, object_position, object_rotation, camera_position, camera_rotation,
                                 processor.intrinsics, processor.image_size).double()
    rotation, position = keypoints_to_frame_pose(uvd, processor.intrinsics, reference, camera_rotation, camera_position,
                                                torch.eye(3).double()[None], torch.zeros(1, 3).double(), processor.image_size)
    torch.testing.assert_close(position, object_position, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(rotation, object_rotation, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bad", ["size", "dtype"])
def test_rejects_wrong_native_image_protocol(intrinsics, bad):
    processor = HikrobotImageProcessor(intrinsics)
    image = torch.zeros(1, 120, 120, 3, dtype=torch.uint8) if bad == "size" else torch.zeros(1, 1024, 1280, 3)
    with pytest.raises(ValueError):
        processor(image)
