"""Calibrate the original object-only GS + live hand-depth capture pipeline."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from viserdex.distillation.hikrobot_images import HikrobotImageProcessor
from .hikrobot_calibration import load_calibration, world_camera_transform


def calibrated_geometry(directory, base_position, base_quaternion):
    intrinsics, _, transforms = load_calibration(directory)
    processor = HikrobotImageProcessor(intrinsics)
    k = processor.intrinsics.numpy()
    world = world_camera_transform(base_position, base_quaternion, transforms)
    width, height = processor.image_size
    focal = 24.0
    pattern = dict(
        width=width, height=height, focal_length=focal,
        horizontal_aperture=float(width * focal / k[0, 0]),
        vertical_aperture=float(height * focal / k[1, 1]),
        horizontal_aperture_offset=float((k[0, 2] - width / 2) / k[0, 0]),
        vertical_aperture_offset=float((k[1, 2] - height / 2) / k[1, 1]),
    )
    return k, world, pattern, processor.metadata


def configure_original_capture(cfg, directory):
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import MultiMeshRayCasterCameraCfg
    from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
    from viserdex.tasks.manipulation.inhand.inhand_env_cfg import pose_estimator_post_init
    from .gaussian_splat_camera_cfg import GaussianSplatCameraCfg

    initial = cfg.scene.robot.init_state
    _, world, pattern, _ = calibrated_geometry(directory, initial.pos, initial.rot)
    # Existing project setup retains its GS asset, augmentations, pose labels,
    # collect-only path and original black-pixel depth compositor.
    pose_estimator_post_init(cfg)
    gs = cfg.scene.gsplat_camera
    gs.offset = GaussianSplatCameraCfg.OffsetCfg(
        pos=tuple(map(float, world[:3, 3])),
        rot=tuple(map(float, Rotation.from_matrix(world[:3, :3]).as_quat())), convention="ros",
    )
    gs.camera_model = GaussianSplatCameraCfg.PinholeCfg(**pattern)
    cfg.scene.hand_depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}",
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=gs.offset.pos, rot=gs.offset.rot, convention="ros"),
        class_type="viserdex.renderer.revo3_hand_depth_camera:create_revo3_hand_depth_camera",
        mesh_prim_paths=[], pattern_cfg=PinholeCameraPatternCfg(**pattern),
        data_types=["distance_to_image_plane"], update_period=0.0,
        max_distance=1.0, depth_clipping_behavior="none",
    )
    params = cfg.observations.pose.image_features.params
    params["masking_data"] = None
    params["hand_camera_cfg"] = SceneEntityCfg("hand_depth_camera")
    params["pose_estimator_cfg"].collect_only = True
    params["pose_estimator_cfg"].train = False


def audit_extrinsic_precision(actual, target):
    """Allow float32 world-position rounding while keeping rotation checks fixed."""
    np.testing.assert_allclose(actual[:3, :3], target[:3, :3], atol=1e-6, rtol=0)
    magnitude = np.maximum(np.abs(actual[:3, 3]), np.abs(target[:3, 3])).astype(np.float32)
    # Large environment-grid origins have coarser float32 spacing. Two rounded
    # frame compositions need a coordinate-wise ULP budget, not relative error.
    position_tolerance = 1e-6 + 2 * np.spacing(magnitude).astype(np.float64)
    np.testing.assert_array_less(np.abs(actual[:3, 3] - target[:3, 3]), position_tolerance)
    np.testing.assert_array_equal(actual[3], target[3])
    return position_tolerance


def audit_original_capture(term, hand, robot, directory):
    intrinsics, _, transforms = load_calibration(directory)
    k = HikrobotImageProcessor(intrinsics).intrinsics.numpy()
    np.testing.assert_allclose(term.camera.data.intrinsic_matrices.torch.cpu().numpy(),
                               np.broadcast_to(k, (len(term.camera_pos), 3, 3)), atol=2e-4, rtol=0)
    np.testing.assert_allclose(hand.data.intrinsic_matrices.torch.cpu().numpy(),
                               term.camera.data.intrinsic_matrices.torch.cpu().numpy(), atol=2e-4, rtol=0)
    positions = term.camera_pos.cpu().numpy()
    rotations = Rotation.from_quat(term.camera_quat_ros.cpu().numpy()).as_matrix()
    errors, position_tolerances = [], []
    for i in range(len(positions)):
        target = world_camera_transform(robot.data.root_pos_w.torch[i].cpu().numpy(),
                                        robot.data.root_quat_w.torch[i].cpu().numpy(), transforms)
        actual = np.eye(4)
        actual[:3, :3], actual[:3, 3] = rotations[i], positions[i]
        position_tolerances.append(audit_extrinsic_precision(actual, target))
        errors.append(float(np.abs(actual - target).max()))
    return {
        "image_protocol": "original_gsplat_object_plus_depth_ordered_black_hand_occlusion",
        "raw_calibration_size": [1280, 1024], "cnn_image_size": [120, 120],
        "camera_intrinsics": k.tolist(), "T_base_camera": transforms["T_base_camera"].tolist(),
        "maximum_extrinsic_abs_error": max(errors), "num_envs_checked": len(errors),
        "extrinsic_position_tolerance_m_per_axis_max": np.max(position_tolerances, axis=0).tolist(),
        "extrinsic_rotation_abs_tolerance": 1e-6,
        "distortion": "GS and ray camera use rectified pinhole rays; real RGB must use the same undistortion",
        "full_hand_rgb_in_network_input": False,
        "calibration_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(directory).glob("hikrobot_DB0447679_*.json"))},
    }
