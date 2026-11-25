"""Validate and publish a portable scanned-object bundle for ViserDex."""

import itertools
import re
import shutil
from pathlib import Path

import numpy as np

from .common import fingerprint, read_json, write_json


def validate_splat(path):
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    required = ["x", "y", "z", "opacity"] + [f"f_dc_{j}" for j in range(3)]
    required += [f"f_rest_{j}" for j in range(45)]
    required += [f"scale_{j}" for j in range(3)] + [f"rot_{j}" for j in range(4)]
    missing = set(required) - set(vertex.data.dtype.names)
    if missing:
        raise ValueError(f"Not a ViserDex degree-3 Gaussian PLY; missing {sorted(missing)}")
    array = np.column_stack([vertex[name] for name in required])
    if len(array) < 10 or not np.isfinite(array).all():
        raise ValueError("Empty or nonfinite Gaussian PLY")
    quats = np.column_stack([vertex[f"rot_{j}"] for j in range(4)])
    if not np.allclose(np.linalg.norm(quats, axis=1), 1, atol=1e-3):
        raise ValueError("Gaussian WXYZ rotations are not normalized")
    log_scales = np.column_stack([vertex[f"scale_{j}"] for j in range(3)])
    if (log_scales < -20).any() or (log_scales > 2).any():
        raise ValueError("Implausible metric Gaussian scales")
    return vertex


def create_usd(mesh, path, mass_kg):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass_kg))
    shape = UsdGeom.Mesh.Define(stage, "/Object/geometry")
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    shape.CreateFaceVertexCountsAttr([3] * len(triangles))
    shape.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
    shape.CreateSubdivisionSchemeAttr("none")
    shape.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
    if mesh.has_vertex_colors():
        color = shape.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        color.Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_colors).astype(np.float32)))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()
    # Validate saved file, including units and default prim, not just the in-memory stage.
    saved = Usd.Stage.Open(str(path))
    if not saved or not saved.GetDefaultPrim() or UsdGeom.GetStageMetersPerUnit(saved) != 1:
        raise ValueError("Exported USD validation failed")


def import_asset(args):
    import open3d as o3d
    import torch
    from sklearn.cluster import MiniBatchKMeans

    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", args.name):
        raise ValueError("Object name must start with a letter and contain only letters, digits or underscores")
    dataset, training = Path(args.dataset).resolve(), Path(args.training).resolve()
    data, completion = read_json(dataset / "dataset.json"), read_json(training / "completion.json")
    if not completion.get("completed") or not data.get("completed"):
        raise ValueError("Prepare and training must complete before import")
    if completion.get("dataset_manifest_sha256") != fingerprint(dataset / "dataset.json"):
        raise ValueError("Training and geometry are from different prepared datasets")
    if completion.get("splat_sha256") != fingerprint(training / "splat.ply"):
        raise ValueError("Exported splat changed after validation")
    if data.get("mesh_sha256") != fingerprint(dataset / "mesh.ply"):
        raise ValueError("Mesh changed after preparation")
    from viserdex.assets.objects import _OBJECT_CFGS

    if args.name in _OBJECT_CFGS:
        raise ValueError(f"Object name {args.name!r} already exists in ViserDex")
    vertex = validate_splat(training / "splat.ply")
    means = np.column_stack([vertex[a] for a in "xyz"])
    lower, upper = np.asarray(data["bounds_m"])
    if (means.min(0) < lower - 0.02).any() or (means.max(0) > upper + 0.02).any():
        raise ValueError("Splat and metric mesh bounds disagree")
    destination = Path(args.assets_dir).resolve() / args.name
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ("meshes", "usd", "splats/clusters/indices", "keypoints"):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(dataset / "mesh.ply"))
    if len(mesh.triangles) == 0:
        raise ValueError("Empty mesh")
    if not np.allclose([mesh.get_min_bound(), mesh.get_max_bound()], data["bounds_m"], atol=1e-5):
        raise ValueError("Mesh bounds differ from prepared dataset")
    shutil.copy2(dataset / "mesh.ply", destination / "meshes/object.ply")
    shutil.copy2(training / "splat.ply", destination / "splats/splat.ply")
    create_usd(mesh, destination / "usd/object.usda", args.mass_kg)
    # Eight virtual bbox anchors, plus the origin appended by image_to_pose = 9 keypoints.
    corners = np.array(list(itertools.product(*zip(lower, upper))), dtype=np.float32)
    torch.save(torch.from_numpy(corners), destination / "keypoints/points_8.pt")
    for key, values, clusters in (
        ("means3d", means, 128),
        ("sh0", np.column_stack([vertex[f"f_dc_{j}"] for j in range(3)]), 32),
        ("scales", np.column_stack([vertex[f"scale_{j}"] for j in range(3)]), 64),
        ("quats", np.column_stack([vertex[f"rot_{j}"] for j in range(4)]), 128),
    ):
        labels = MiniBatchKMeans(
            n_clusters=min(clusters, len(values)), random_state=42, n_init=3, batch_size=1024
        ).fit_predict(values.copy())
        torch.save(
            torch.tensor(labels, dtype=torch.int64),
            destination / f"splats/clusters/indices/splat_clustered_{key}_indices.pt",
        )
    provenance = {
        "dataset_manifest_sha256": fingerprint(dataset / "dataset.json"),
        "splat_sha256": fingerprint(training / "splat.ply"),
        "training": completion,
        "units": "metres",
        "extent_m": data["extent_m"],
        "mass_kg": args.mass_kg,
        "collision": "convexHull",
        "mesh_watertight": data["mesh_watertight"],
        "object_from_reconstruction": data["object_from_reconstruction"],
        "synthetic": data.get("synthetic", False),
        "notes": [
            "Unobserved surfaces are not reconstructed.",
            "Convex hull is an approximation for concave objects.",
            "New geometry needs its own policy and pose-estimator training.",
        ],
    }
    write_json(destination / "provenance.json", provenance)
    # Publication marker written LAST. objects.py only discovers directories with this manifest.
    write_json(
        destination / "object.json",
        {
            "schema_version": 1,
            "name": args.name,
            "usd_path": "usd/object.usda",
            "mesh_path": "meshes/object.ply",
            "splat_path": "splats/splat.ply",
            "points_path": "keypoints/points_8.pt",
            "supports_pose_estimation": True,
            "rollout_policy_run_name": None,
            "revo3_rollout_policy_run_name": None,
        },
    )
    print(f"Imported {args.name} into {destination}. Select with VISERDEX_OBJECT={args.name}")
