"""Load the delivered Hikrobot calibration through Isaac Lab's public camera API."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_CALIBRATION_DIRECTORY = Path(__file__).resolve().parents[4] / "config" / "cameras"


def load_calibration(directory):
    directory = Path(directory)
    intrinsics = json.loads((directory / "hikrobot_DB0447679_1280x1024_intrinsics.json").read_text())
    extrinsics = json.loads(
        (directory / "hikrobot_DB0447679_1280x1024_extrinsics_revo3_20261022.json").read_text()
    )
    assert intrinsics["source"]["serial"] == extrinsics["source"]["serial"] == "DB0447679"
    assert (intrinsics["width"], intrinsics["height"]) == (1280, 1024)
    assert extrinsics["frames"]["base"] == "right_hand_base_link"
    assert extrinsics["frames"]["palm"] == "right_palm"
    assert extrinsics["camera_geometry"] == {
        "Width": 1280, "Height": 1024, "OffsetX": 0, "OffsetY": 0,
        "ReverseX": False, "ReverseY": False,
    }
    assert intrinsics["distortion_order"] == ["k1", "k2", "p1", "p2", "k3"]
    transforms = {name: np.asarray(record["matrix"], dtype=np.float64)
                  for name, record in extrinsics["transforms"].items()}
    for matrix in transforms.values():
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-10
    np.testing.assert_allclose(
        transforms["T_base_palm"] @ transforms["T_palm_camera"],
        transforms["T_base_camera"], atol=1e-10,
    )
    return intrinsics, extrinsics, transforms


def world_camera_transform(base_position, base_quaternion_xyzw, transforms):
    """The task root is the hand base, whereas the delivered palm has an offset."""
    world_base = np.eye(4)
    world_base[:3, :3] = Rotation.from_quat(base_quaternion_xyzw).as_matrix()
    world_base[:3, 3] = base_position
    return world_base @ transforms["T_base_palm"] @ transforms["T_palm_camera"]


def audit_calibrated_camera(camera, robot, directory=DEFAULT_CALIBRATION_DIRECTORY):
    """Check live sensor poses, native intrinsics and distortion against the bundle."""
    import hashlib

    intrinsics, _, transforms = load_calibration(directory)
    positions = camera.data.pos_w.torch.detach().cpu().numpy()
    quaternions = camera.data.quat_w_ros.torch.detach().cpu().numpy()
    base_positions = robot.data.root_pos_w.torch.detach().cpu().numpy()
    base_quaternions = robot.data.root_quat_w.torch.detach().cpu().numpy()
    actual, expected = [], []
    for position, quaternion, base_position, base_quaternion in zip(
        positions, quaternions, base_positions, base_quaternions, strict=True,
    ):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        matrix[:3, 3] = position
        target = world_camera_transform(base_position, base_quaternion, transforms)
        np.testing.assert_allclose(matrix, target, atol=1e-6, rtol=0)
        actual.append(matrix)
        expected.append(target)
    live_k = camera.data.intrinsic_matrices.torch.detach().cpu().numpy()
    np.testing.assert_allclose(live_k, np.broadcast_to(intrinsics["camera_matrix"], live_k.shape), atol=2e-4, rtol=0)
    distortion = camera.cfg.spawn.distortion
    assert distortion.apply_lens_distortion
    expected_distortion = dict(zip(intrinsics["distortion_order"], intrinsics["dist_coeffs"], strict=True))
    for name, value in expected_distortion.items():
        assert abs(getattr(distortion, name) - value) < 1e-12
    assert (camera.cfg.width, camera.cfg.height) == (1280, 1024)
    files = sorted(Path(directory).glob("hikrobot_DB0447679_*.json"))
    return {
        "renderer": "NewtonWarpRenderer", "native_image_size": [1280, 1024],
        "intrinsics": intrinsics["camera_matrix"], "distortion": intrinsics["dist_coeffs"],
        "distortion_order": intrinsics["distortion_order"], "native_opencv_distortion": True,
        "T_world_camera_env0": actual[0].tolist(),
        "T_base_camera": transforms["T_base_camera"].tolist(),
        "maximum_extrinsic_abs_error": float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))),
        "num_envs_checked": len(actual),
        "calibration_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def make_calibrated_camera(directory, base_position, base_quaternion_xyzw):
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        OpenCvPinholeDistortionCfg, PinholeCameraCfg,
    )
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    intrinsics, _, transforms = load_calibration(directory)
    world_camera = world_camera_transform(base_position, base_quaternion_xyzw, transforms)
    spawn = PinholeCameraCfg.from_intrinsic_matrix(
        np.asarray(intrinsics["camera_matrix"]).reshape(-1).tolist(),
        width=1280, height=1024, focal_length=24.0, clipping_range=(0.001, 3.0),
    )
    k1, k2, p1, p2, k3 = intrinsics["dist_coeffs"]
    spawn.distortion = OpenCvPinholeDistortionCfg(
        image_size=(1280, 1024), apply_lens_distortion=True,
        fx=intrinsics["fx"], fy=intrinsics["fy"],
        cx=intrinsics["cx"], cy=intrinsics["cy"],
        k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
    )
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/HikrobotCamera", update_period=0.0,
        width=1280, height=1024, data_types=["rgb", "distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(
            pos=tuple(world_camera[:3, 3]),
            rot=tuple(Rotation.from_matrix(world_camera[:3, :3]).as_quat()),
            convention="ros",
        ),
        spawn=spawn, update_latest_camera_pose=True,
        background_color=(0.92, 0.94, 0.97),
        renderer_cfg=NewtonWarpRendererCfg(
            use_cuda_graph=False, enable_textures=True,
            enable_shadows=False, enable_ambient_lighting=True,
        ),
    )
