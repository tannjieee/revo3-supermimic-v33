"""Shared geometry and on-disk contracts (metres, OpenCV cameras)."""

import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def intrinsics_matrix(intrinsics):
    i = intrinsics
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], dtype=np.float64)


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def unproject(depth_m, intrinsics, stride=1):
    """Return metric optical XYZ and the corresponding integer (v,u) pixels."""
    v, u = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[v, u]
    valid = np.isfinite(z) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]
    xyz = np.column_stack(
        ((u - intrinsics["cx"]) * z / intrinsics["fx"], (v - intrinsics["cy"]) * z / intrinsics["fy"], z)
    )
    return xyz, v, u


def metric_scale(measured_depth, sfm_depth, min_samples=100, max_relative_mad=0.08):
    """Robust zero-intercept scale: RealSense optical Z / COLMAP optical Z.

    Both depths must be sampled at the SAME rectified pixels, not Euclidean range.
    Reject inconsistent evidence instead of silently inventing a metric scale.
    """
    measured, predicted = np.asarray(measured_depth), np.asarray(sfm_depth)
    valid = np.isfinite(measured) & np.isfinite(predicted) & (measured > 0) & (predicted > 0)
    ratios = measured[valid] / predicted[valid]
    if len(ratios) < min_samples:
        raise ValueError(f"Only {len(ratios)} valid depth/SfM correspondences; need {min_samples}")
    center = float(np.median(ratios))
    relative_mad = float(np.median(np.abs(ratios - center)) / center)
    if relative_mad > max_relative_mad:
        raise ValueError(f"Inconsistent depth/SfM scale: relative MAD={relative_mad:.3f}")
    inliers = np.abs(ratios - center) <= max(3 * 1.4826 * relative_mad, 0.02) * center
    if inliers.sum() < min_samples or inliers.mean() < 0.6:
        raise ValueError("Insufficient consistent metric scale evidence")
    return float(np.median(ratios[inliers])), {
        "correspondences": len(ratios),
        "inliers": int(inliers.sum()),
        "relative_mad": relative_mad,
        "method": "median_aligned_depth_z_over_sfm_z",
    }


def validate_bounds(lower, upper):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
        raise ValueError("Bounds must be finite XYZ triples in metres")
    if (upper <= lower).any():
        raise ValueError("Every upper bound must exceed its lower bound")
    return lower, upper


def inside_bounds(points, lower, upper):
    return ((points >= lower) & (points <= upper)).all(axis=-1)


def load_scan(path):
    root = Path(path).resolve()
    manifest = read_json(root / "scan.json")
    if manifest.get("schema_version") != 1 or not manifest.get("completed"):
        raise ValueError("Expected a completed version-1 scan.json")
    if not manifest.get("rectified") or manifest["intrinsics"]["model"] != "PINHOLE":
        raise ValueError("Images and aligned optical-Z depth must be rectified to PINHOLE")
    i = manifest["intrinsics"]
    if not np.isfinite([i[k] for k in ("width", "height", "fx", "fy", "cx", "cy")]).all():
        raise ValueError("Nonfinite camera intrinsics")
    if min(i[k] for k in ("width", "height", "fx", "fy")) <= 0:
        raise ValueError("Image size and focal lengths must be positive")
    if not np.isfinite(manifest["depth_scale_m"]) or manifest["depth_scale_m"] <= 0:
        raise ValueError("Depth scale must be a finite positive metres-per-unit value")
    if len(manifest["frames"]) < 3:
        raise ValueError("At least three frames are required")
    image_names = [frame["image"] for frame in manifest["frames"]]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate capture frames")
    for frame in manifest["frames"]:
        for key in ("image", "depth"):
            file = (root / frame[key]).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                raise ValueError(f"Invalid {key} path: {frame[key]}")
    return root, manifest
