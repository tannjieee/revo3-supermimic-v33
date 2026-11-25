"""Check stored Revo3 images, depth ordering, labels, and trajectory diversity."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def quaternion_matrix(quaternions):
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)


def audit(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["completed"]
    reference = np.array(manifest["object_keypoints"])
    intrinsics = np.array(manifest["camera_intrinsics"])
    camera_position = np.array(manifest["camera_position_env"])
    camera_rotation = quaternion_matrix(np.array(manifest["camera_quaternion_xyzw_ros"]))
    with h5py.File(root / "samples.h5", "r") as data:
        size = len(data["rgb"])
        assert size == manifest["samples"]
        fractions = []
        visible = []
        for start in range(0, size, 256):
            end = min(start + 256, size)
            rgb, original = data["rgb"][start:end], data["rgb_unoccluded"][start:end]
            depth = data["object_depth"][start:end].astype(np.float32)
            hand = data["hand_depth"][start:end].astype(np.float32)
            assert not np.isnan(hand).any(), "Hand misses should be +inf"
            assert np.isfinite(depth).all()
            foreground = np.any(original != 0, axis=-1) & (depth > 0) & (depth < 1)
            changed = np.any(rgb != original, axis=-1)
            assert not np.any(changed & ~foreground)
            assert np.all(rgb[changed] == 0), "Occlusion must only remove object pixels"
            # Depth is stored in float16; allow one millimetre for quantization.
            assert np.all(hand[changed] < depth[changed] + 0.001)
            fraction = changed.sum(axis=(1, 2)) / foreground.sum(axis=(1, 2)).clip(1)
            np.testing.assert_allclose(fraction, data["occlusion_fraction"][start:end], atol=1e-6)
            assert np.all(fraction >= manifest["arguments"]["min_occlusion"] - 1e-6)
            count = foreground.sum(axis=(1, 2)) - changed.sum(axis=(1, 2))
            assert np.all(count >= manifest["arguments"]["min_visible_pixels"])
            np.testing.assert_array_equal(count, data["visible_pixels"][start:end])
            labels = data["keypoints_uvd"][start:end]
            assert labels.shape[1] == 27 and np.isfinite(labels).all()
            assert (labels.reshape(-1, 9, 3)[..., 2] > 0).all()
            poses = data["object_pose"][start:end]
            world_points = reference @ quaternion_matrix(poses[:, 3:]).transpose(0, 2, 1) + poses[:, None, :3]
            camera_points = (world_points - camera_position) @ camera_rotation
            homogeneous = camera_points @ intrinsics.T
            expected = np.concatenate(
                (homogeneous[..., :2] / homogeneous[..., 2:] / 120, camera_points[..., 2:]), axis=-1
            )
            np.testing.assert_allclose(labels.reshape(-1, 9, 3), expected, atol=1e-4, rtol=1e-4)
            fractions.extend(fraction.tolist())
            visible.extend(count.tolist())
        joints = data["joint_pos"][:]
        poses = data["object_pose"][:]
        assert np.isfinite(joints).all() and np.isfinite(poses).all()
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=-1), 1, atol=1e-4)
        assert np.max(joints.std(axis=0)) > 0.01, "No articulated hand motion"
        assert np.max(poses[:, 3:].std(axis=0)) > 0.05, "No object rotation diversity"
        result = {
            "passed": True,
            "samples": size,
            "seed": manifest["seed"],
            "teacher_sha256": manifest["teacher_sha256"],
            "occlusion_quantiles": dict(
                zip(
                    ("min", "p25", "median", "p75", "p95", "max"),
                    np.quantile(fractions, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
                )
            ),
            "frames_over_10pct": int(np.sum(np.array(fractions) >= 0.1)),
            "frames_over_25pct": int(np.sum(np.array(fractions) >= 0.25)),
            "frames_over_50pct": int(np.sum(np.array(fractions) >= 0.5)),
            "visible_pixels_min": int(min(visible)),
            "joint_std_radians": joints.std(axis=0).tolist(),
            "object_position_std_m": poses[:, :3].std(axis=0).tolist(),
            "environments": int(len(np.unique(data["env_id"][:]))),
            "sim_steps": int(len(np.unique(data["sim_step"][:]))),
        }
    (root / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    audit(parser.parse_args().dataset)
