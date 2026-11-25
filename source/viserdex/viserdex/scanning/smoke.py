"""Repeatable full synthetic reconstruction/training/import validation."""

import itertools
from pathlib import Path

import numpy as np

from .common import new_directory, read_json, transform_points, write_json


def smoke(args):
    from .cli import build_parser

    root = new_directory(args.output)

    def run(argv):
        print("SCAN SMOKE:", " ".join(map(str, argv)), flush=True)
        parsed = build_parser().parse_args([str(value) for value in argv])
        parsed.func(parsed)

    run(["synthetic", "--output", root / "capture", "--views", 48])
    run(["reconstruct", "--scan", root / "capture", "--output", root / "reconstruction", "--depth-max", 0.8])
    gt = read_json(root / "capture/ground_truth.json")
    rec = read_json(root / "reconstruction/reconstruction.json")
    anchor_index = int(Path(rec["anchor_image"]).stem)
    anchor = np.linalg.inv(gt["camera_to_world"][anchor_index])
    half_extent = np.array(gt["physical_box_extent_m"]) / 2
    corners = transform_points(np.array(list(itertools.product(*zip(-half_extent, half_extent)))), anchor)
    lower, upper = corners.min(0) - 0.004, corners.max(0) + 0.004
    normal = anchor[:3, :3] @ np.array([0, 0, 1.0])
    plane_point = transform_points(np.array([[0, 0, -0.06]]), anchor)[0]
    plane = np.r_[normal, -normal @ plane_point]
    run(
        [
            "prepare",
            "--reconstruction",
            root / "reconstruction",
            "--output",
            root / "prepared",
            "--bounds",
            *lower,
            *upper,
            "--plane",
            *plane,
        ]
    )
    # Exercise the same config-driven entry point users can use after inspecting their crop.
    config = {
        "train": {
            "dataset": str(root / "prepared"),
            "output": str(root / "training"),
            "iterations": args.iterations,
            "max_gaussians": 15000,
        },
        "import": {
            "dataset": str(root / "prepared"),
            "training": str(root / "training"),
            "name": "ScanPipelineSmoke",
            "mass_kg": 0.05,
            "assets_dir": str(root / "assets"),
        },
    }
    write_json(root / "pipeline.json", config)
    run(["process", "--config", root / "pipeline.json"])
    from viserdex.assets.scanned_objects import load_scanned_objects
    from viserdex.utils.load_utils import load_splat_file
    import torch

    entry = load_scanned_objects(root / "assets")["ScanPipelineSmoke"]
    splats = load_splat_file(entry["splat_path"])
    points = torch.load(entry["points_path"], weights_only=True)
    assert points.shape == (8, 3)
    for key in ("means3d", "sh0", "scales", "quats"):
        labels = torch.load(
            Path(entry["splat_path"]).parent / f"clusters/indices/splat_clustered_{key}_indices.pt", weights_only=True
        )
        assert labels.shape == (len(splats["means3d"]),)
    errors = []
    for frame in rec["frames"]:
        idx = int(Path(frame["image"]).stem)
        truth = anchor @ np.asarray(gt["camera_to_world"][idx])
        errors.append(np.linalg.norm(np.asarray(frame["camera_to_world"])[:3, 3] - truth[:3, 3]))
    evaluation = read_json(root / "training/evaluation.json")
    report = {
        "completed": True,
        "synthetic": True,
        "registered_views": len(rec["frames"]),
        "mean_camera_position_error_m": float(np.mean(errors)),
        "max_camera_position_error_m": float(max(errors)),
        "evaluation": evaluation,
        "gaussians_loaded_by_viserdex": len(splats["means3d"]),
        "keypoints": list(points.shape),
        "real_camera_capture_tested": False,
        "isaac_sim_spawn_tested": False,
    }
    if len(rec["frames"]) < 44 or max(errors) > 0.003:
        raise AssertionError("Synthetic metric camera recovery failed")
    if evaluation["exported"]["foreground_psnr"] < evaluation["initial"]["foreground_psnr"] + 1:
        raise AssertionError("Gaussian training failed to improve held-out foreground appearance")
    if evaluation["exported"]["mask_iou"] < 0.85:
        raise AssertionError("Exported model lost the object silhouette")
    write_json(root / "validation.json", report)
    print(f"Full synthetic scan pipeline passed. Report: {root / 'validation.json'}")
