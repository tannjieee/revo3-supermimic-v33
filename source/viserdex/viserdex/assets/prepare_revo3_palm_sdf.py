"""Build a closed, low-poly fixed palm and author official Newton SDF schemas.

The six visual surfaces are read from the official multi-physics conversion.
Only colliders attached to the fixed palm body are replaced. Geometry is
regularized in a fine voxel grid, never with a convex hull. Original visuals,
finger colliders, joints, mass properties and vendored inputs are preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

PALM_BODY = "/revo3_single/Geometry/world"
PALM_COLLIDER = PALM_BODY + "/palm_sdf_collision"


def _owner(prim):
    from pxr import UsdPhysics

    parent = prim.GetParent()
    while parent and not parent.IsPseudoRoot():
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(parent.GetPath())
        parent = parent.GetParent()
    return None


def extract_palm(stage):
    """Return the palm's visual meshes in the fixed body's local frame."""
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

    body = stage.GetPrimAtPath(PALM_BODY)
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Missing fixed palm rigid body: {PALM_BODY}")
    cache = UsdGeom.XformCache()
    inv = cache.GetLocalToWorldTransform(body).GetInverse()
    meshes, records, colliders = [], [], []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if _owner(prim) != PALM_BODY:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(prim.GetPath()))
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputePurpose() == "guide" or imageable.ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f"Expected triangulated visual mesh: {prim.GetPath()}")
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        relative = np.asarray(cache.GetLocalToWorldTransform(prim) * inv)
        vertices = np.c_[vertices, np.ones(len(vertices))].dot(relative)[:, :3]
        meshes.append(trimesh.Trimesh(vertices, faces, process=True))
        records.append({"prim": str(prim.GetPath()), "vertices": len(vertices), "faces": len(faces)})
    if len(meshes) != 6 or len(colliders) != 5:
        raise ValueError(f"Unexpected source palm layout: {len(meshes)} visuals, {len(colliders)} colliders")
    return meshes, records, colliders


def build_surface(meshes, *, voxel_size=0.0003, face_count=1200):
    """Close submillimeter seams, fill enclosed voids, smooth and decimate."""
    import trimesh
    from scipy import ndimage
    from skimage import measure

    whole = trimesh.util.concatenate(meshes)
    voxels = whole.voxelized(pitch=voxel_size)
    padding = 4
    occupied = np.pad(voxels.matrix, padding)
    occupied = ndimage.binary_closing(occupied, iterations=2)
    occupied = ndimage.binary_fill_holes(occupied)
    labels, _ = ndimage.label(occupied)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    occupied = labels == counts.argmax()
    del labels
    vertices, faces, _, _ = measure.marching_cubes(
        occupied.astype(np.uint8), level=0.5, spacing=(voxel_size,) * 3, allow_degenerate=False
    )
    vertices += voxels.transform[:3, 3] - padding * voxel_size
    surface = trimesh.Trimesh(vertices, faces, process=True)
    surface.fix_normals()
    # Drop disconnected interior surfaces; retain the complete outer surface.
    surface = max(surface.split(only_watertight=False), key=lambda mesh: abs(mesh.volume))
    surface.fix_normals()
    high_faces = len(surface.faces)
    trimesh.smoothing.filter_taubin(surface, iterations=20)
    low = surface.simplify_quadric_decimation(face_count=face_count, aggression=5)
    low.fix_normals()
    if not (low.is_watertight and low.is_winding_consistent and low.body_count == 1 and low.volume > 0):
        raise ValueError("SDF input must be one closed, consistently oriented positive-volume surface")
    if len(low.faces) != face_count or low.euler_number != 2:
        raise ValueError(f"Unexpected topology: faces={len(low.faces)}, Euler={low.euler_number}")
    return low, {
        "regularized_faces": high_faces,
        "voxel_size_m": voxel_size,
        "seam_closing_iterations": 2,
        "smoothing_iterations": 20,
        "vertices": len(low.vertices),
        "faces": len(low.faces),
        "watertight": bool(low.is_watertight),
        "euler_number": int(low.euler_number),
        "connected_surfaces": int(low.body_count),
        "volume_m3": float(low.volume),
        "bounds_m": low.bounds.tolist(),
    }


def mechanical_signature(stage):
    from pxr import UsdPhysics

    signature = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
            signature[str(prim.GetPath())] = {
                attr.GetName(): str(attr.Get())
                for attr in prim.GetAttributes()
                if attr.GetName().startswith(("physics:", "drive:", "mjc:", "newton:"))
            }
    return signature


def author_asset(source, output, mesh, old_colliders, *, sdf_resolution=512):
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, output.parent)]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/revo3_single"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for path in old_colliders:
        prim = stage.GetPrimAtPath(path)
        ancestor = prim
        while ancestor.IsInstanceProxy():
            ancestor = ancestor.GetParent()
        if ancestor.IsInstance():
            stage.OverridePrim(ancestor.GetPath()).SetInstanceable(False)
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    geom = UsdGeom.Mesh.Define(stage, PALM_COLLIDER)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces, dtype=np.int32).ravel().tolist())
    geom.CreateSubdivisionSchemeAttr("none")
    geom.CreateDoubleSidedAttr(True)
    geom.CreatePurposeAttr("guide")
    geom.CreateDisplayColorAttr([(0.1, 0.65, 0.9)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
    # Newton's importer uses NewtonSDFCollisionAPI; approximation stays 'none'
    # so a convex remeshing request cannot override the intended concave mesh.
    ok = apply_mesh_collision_properties(
        PALM_COLLIDER,
        [
            UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
            NewtonSDFCollisionCfg(
                sdf_max_resolution=sdf_resolution,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
                sdf_texture_format="uint16",
                sdf_padding=0.005,
                hydroelastic_enabled=False,
            ),
        ],
        stage=stage,
    )
    if not ok:
        raise RuntimeError("Official Newton SDF schema application failed")
    stage.GetRootLayer().Save()
    return stage


def prepare_palm_sdf(source, output_dir, *, face_count=1200, sdf_resolution=512, voxel_size=0.0003, prepared_mesh=None):
    """Create a portable USD overlay and record source, geometry and mechanics."""
    import trimesh
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_stage = Usd.Stage.Open(str(source))
    before = mechanical_signature(source_stage)
    meshes, source_records, colliders = extract_palm(source_stage)
    if prepared_mesh is None:
        mesh, geometry = build_surface(meshes, voxel_size=voxel_size, face_count=face_count)
    else:
        mesh = trimesh.load(prepared_mesh, force="mesh")
        if not (
            mesh.is_watertight
            and mesh.is_winding_consistent
            and mesh.body_count == 1
            and mesh.euler_number == 2
            and len(mesh.faces) == face_count
        ):
            raise ValueError("Invalid prepared SDF input")
        geometry = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": True,
            "euler_number": int(mesh.euler_number),
            "connected_surfaces": 1,
            "volume_m3": float(mesh.volume),
            "bounds_m": mesh.bounds.tolist(),
            "prepared_mesh": str(prepared_mesh),
            "voxel_size_m": voxel_size,
        }
    mesh_path = output_dir / "palm_surface_1200.ply"
    mesh.export(mesh_path)
    output = output_dir / "revo3_right_palm_sdf.usda"
    if output.exists():
        output.unlink()
    stage = author_asset(source, output, mesh, colliders, sdf_resolution=sdf_resolution)
    standalone_path = output_dir / "palm_collision_sdf.usda"
    if standalone_path.exists():
        standalone_path.unlink()
    standalone = Usd.Stage.CreateNew(str(standalone_path))
    Sdf.CopySpec(stage.GetRootLayer(), PALM_COLLIDER, standalone.GetRootLayer(), "/PalmCollision")
    standalone.SetDefaultPrim(standalone.GetPrimAtPath("/PalmCollision"))
    UsdGeom.SetStageMetersPerUnit(standalone, 1.0)
    UsdGeom.SetStageUpAxis(standalone, UsdGeom.Tokens.z)
    standalone.GetRootLayer().Save()
    if mechanical_signature(stage) != before:
        raise AssertionError("Palm replacement changed a rigid body's or joint's mechanical properties")
    active = [
        str(p.GetPath())
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    ]
    if len(active) != 78 or PALM_COLLIDER not in active:
        raise AssertionError(f"Expected 77 finger + 1 palm colliders, got {len(active)}")
    missing = UsdUtils.ComputeAllDependencies(str(output))[2]
    if missing:
        raise ValueError(f"Missing USD dependencies: {missing}")
    report = {
        "source_usd": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_visuals": source_records,
        "source_visual_faces": sum(r["faces"] for r in source_records),
        "replaced_colliders": colliders,
        "active_hand_colliders": len(active),
        "unchanged_mechanical_signature_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
        "geometry": geometry,
        "sdf_resolution": sdf_resolution,
        "usd": str(output),
        "standalone_collision_usd": str(standalone_path),
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "requires": "Newton contact pipeline: MJWarpSolverCfg.use_mujoco_contacts=False",
    }
    (output_dir / "palm_sdf_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
