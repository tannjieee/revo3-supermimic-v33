# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

from .scanned_objects import load_scanned_objects


VISERDEX_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


_OBJECT_CFGS = {
    "RSLCube": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/RSLCube/RSLCube_0p9.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/RSLCube/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/RSLCube/cube.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/RSLCube/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "test_release_ppo_docker_4",
        "revo3_rollout_policy_run_name": None,
    },
    "3DPrintedToy": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/3DPrintedToy/3DPrintedToy.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/3DPrintedToy/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/3DPrintedToy/bulbasaur_resized_1p25x_pm_triangulated_centered.obj",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/3DPrintedToy/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "biotac_3d_printed_toy_1p25",
        "revo3_rollout_policy_run_name": None,
    },
    "Duck": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/Duck/Duck.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/Duck/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/Duck/Duck.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/Duck/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "priveleged_duck",
        "revo3_rollout_policy_run_name": None,
    },
    "DuckRevo3Scaled": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/DuckRevo3Scaled/DuckRevo3Scaled.usd",
        # Separate paths prevent accidentally using unscaled visual-learning data.
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/DuckRevo3Scaled/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/DuckRevo3Scaled/DuckRevo3Scaled.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/DuckRevo3Scaled/icosphere/exports/splat.ply",
        "supports_pose_estimation": False,
        "rollout_policy_run_name": None,
        "revo3_rollout_policy_run_name": None,
    },
    "TabletBottle": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/TabletBottle/TabletBottle.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/TabletBottle/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/TabletBottle/Mentos.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/TabletBottle/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "priveleged_tablet_bottle",
        "revo3_rollout_policy_run_name": None,
    },
    "Globe": {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/Globe/Globe.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/Globe/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/Globe/Globe_o3d.obj",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/Globe/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "privileged_globe",
        "revo3_rollout_policy_run_name": None,
    },
}

_sphere_dir = Path(__file__).resolve().parents[4] / "assets/rslsphere65"
_OBJECT_CFGS["RSLSphere65"] = {
    "usd_path": str(_sphere_dir / "RSLSphere65/RSLSphere65.usda"),
    "mesh_path": str(_sphere_dir / "RSLSphere65/RSLSphere65.glb"),
    "splat_path": str(_sphere_dir / "rslsphere65_gaussians.ply"),
    "points_path": None, "supports_pose_estimation": False,
    "rollout_policy_run_name": None, "revo3_rollout_policy_run_name": None,
}

_die_dir = Path(__file__).resolve().parents[4] / "assets/rounded_die"
_OBJECT_CFGS["RoundedDie"] = {
    "usd_path": str(_die_dir / "object_newton.usda"),
    "mesh_path": str(_die_dir / "visual.glb"),
    "splat_path": str(_die_dir / "gaussians.ply"),
    "points_path": None, "supports_pose_estimation": False,
    "rollout_policy_run_name": None, "revo3_rollout_policy_run_name": None,
}

# Generated objects carry portable manifests next to their mesh, USD, and splat.
_OBJECT_CFGS.update(
    load_scanned_objects(
        os.environ.get("VISERDEX_SCANNED_ASSETS_DIR", os.path.join(VISERDEX_ASSETS_DIR, "scanned")),
        reserved_names=_OBJECT_CFGS,
    )
)

# "3DPrintedToy"  # "RSLCube"  # "Toytruck"  # "Duck"  # "TabletBottle"  # "Globe"
object_name = os.environ.get("VISERDEX_OBJECT", "RSLCube")
if object_name not in _OBJECT_CFGS:
    raise ValueError(f"Unknown VISERDEX_OBJECT {object_name!r}; choose from {', '.join(_OBJECT_CFGS)}")
TARGET_OBJECT_CFG = _OBJECT_CFGS[object_name]
TARGET_OBJECT_CFG["name"] = object_name
TARGET_OBJECT_CFG["source_usd_path"] = TARGET_OBJECT_CFG["usd_path"]
TARGET_OBJECT_CFG["usd_path"] = str(
    Path(__file__).resolve().parent
    / "generated"
    / "usd"
    / object_name
    / Path(TARGET_OBJECT_CFG["source_usd_path"]).with_suffix(".usda").name
)
# Version the repaired cube so saved runs retain their original collision asset.
if object_name == "RSLCube":
    _cube_target = Path(TARGET_OBJECT_CFG["usd_path"])
    TARGET_OBJECT_CFG["usd_path"] = str(_cube_target.with_name(_cube_target.stem + "_exact_box_v1.usda"))

if object_name in ("RSLSphere65", "RoundedDie"):
    TARGET_OBJECT_CFG["usd_path"] = TARGET_OBJECT_CFG["source_usd_path"]


def get_rollout_policy_run(task_name: str, override: str | None = None) -> str:
    """Choose a checkpoint run for the task's hand, without an Allegro fallback for Revo3."""
    if override:
        return override
    key = "revo3_rollout_policy_run_name" if "-Revo3" in task_name else "rollout_policy_run_name"
    run = TARGET_OBJECT_CFG.get(key)
    if not run:
        raise ValueError(
            f"No rollout policy configured for {task_name} / {object_name}. "
            "Train Isaac-ViserDex-Repose-Revo3 first, then pass --policy_run <run_name> "
            f"or set TARGET_OBJECT_CFG[{key!r}] in viserdex/assets/objects.py. "
            "Allegro checkpoints are incompatible with the Revo3 observation/action dimensions."
        )
    return run
