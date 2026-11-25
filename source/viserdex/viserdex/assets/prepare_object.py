"""Repair legacy object collision authoring on a generated USD copy."""

from pathlib import Path
import hashlib


def _repair_rsl_cube_winding(stage):
    """Correct the source cube's inward H face, preserving vertex/UV associations."""
    import numpy as np
    from pxr import UsdGeom

    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError('RSLCube must have exactly one authored visual mesh')
    mesh = UsdGeom.Mesh(meshes[0])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    assert len(points) == 24 and counts.tolist() == [4]*6
    assert len(np.unique(points, axis=0)) == 8
    assert mesh.GetNormalsAttr().Get() is None, 'Review authored normals before repair'
    center = (points.max(axis=0)+points.min(axis=0))/2
    perm = np.arange(len(indices))
    changed = []
    for face in range(6):
        start = face*4
        vertices = points[indices[start:start+4]]
        normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
        if np.dot(normal, vertices.mean(axis=0)-center) < 0:
            perm[start:start+4] = start+np.array([0, 3, 2, 1])
            changed.append(face)
    mesh.GetFaceVertexIndicesAttr().Set(indices[perm].tolist())
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if pv.GetInterpolation() != UsdGeom.Tokens.faceVarying:
            continue
        if pv.IsIndexed():
            values = pv.GetIndices()
            pv.SetIndices(type(values)([values[int(i)] for i in perm]))
        else:
            values = pv.Get()
            if values is not None:
                assert len(values) == len(perm), pv.GetPrimvarName()
                pv.Set(type(values)([values[int(i)] for i in perm]))
    # UVs were permuted with their corners; face identity and texture are unchanged.
    return [{'mesh': str(mesh.GetPath()), 'faces': changed}] if changed else []


def _materialize_bounding_boxes(stage):
    """Keep USD boundingCube semantics without a solver's oriented-box refit.

    Separate collision primitives also let the USD importer retain the original
    visual mesh's per-face material subsets.
    """
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics

    records = []
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if str(approximation).lower() != 'boundingcube':
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f'Invalid boundingCube vertices: {prim.GetPath()}')
        lo, hi = points.min(axis=0), points.max(axis=0)
        half, center = (hi-lo)/2, (hi+lo)/2
        if np.any(half <= 0):
            raise ValueError(f'Degenerate boundingCube: {prim.GetPath()}')
        box_path = prim.GetPath().GetParentPath().AppendChild(prim.GetName()+'_collision_box')
        if stage.GetPrimAtPath(box_path):
            raise ValueError(f'Collision primitive already exists: {box_path}')
        box = UsdGeom.Cube.Define(stage, box_path)
        box.CreateSizeAttr(2.)
        box.CreatePurposeAttr(UsdGeom.Tokens.guide)
        mesh_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
        box_xform = (Gf.Matrix4d(1.).SetScale(Gf.Vec3d(*half))
                     * Gf.Matrix4d(1.).SetTranslate(Gf.Vec3d(*center)) * mesh_xform)
        box.AddTransformOp().Set(box_xform)
        collision = UsdPhysics.CollisionAPI(prim)
        enabled = collision.GetCollisionEnabledAttr().Get() is not False
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(enabled)
        # A direct physics material binding belongs on the new collider too.
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding:physics'):
                box.GetPrim().CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        records.append({'mesh': str(prim.GetPath()), 'box': str(box_path),
                        'local_size': (hi-lo).tolist(), 'local_center': center.tolist()})
    return records


def prepare_object(cfg):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    source_path = Path(cfg["source_usd_path"])
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = Usd.Stage.Open(str(source_path))
    target = Path(cfg["usd_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    source.Flatten().Export(str(target))
    stage = Usd.Stage.Open(str(target))
    # Author on real mesh prims, not instance proxies or their Xform ancestors.
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    repaired = 0
    for prim in list(stage.Traverse()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
            continue
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "convexHull"
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ValueError(f"Collision Xform contains no meshes: {prim.GetPath()}")
        for mesh in meshes:
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(enabled is not False)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
            repaired += 1
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    winding_repairs = _repair_rsl_cube_winding(stage) if cfg.get('name') == 'RSLCube' else []
    boxes = _materialize_bounding_boxes(stage)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders and all(p.IsA(UsdGeom.Gprim) for p in colliders), "Object has no geometric colliders"
    mass_models = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.MassAPI.Apply(prim)
        mass = api.GetMassAttr().Get()
        if mass is not None and mass > 0:
            api.GetDensityAttr().Clear()
            mass_models.append({"body": str(prim.GetPath()), "mass": mass})
        else:
            api.CreateDensityAttr(500.0)
            mass_models.append({"body": str(prim.GetPath()), "density_fallback": 500.0})
    stage.GetRootLayer().Save()
    # OmniPBR is an Isaac Sim built-in MDL module, not a missing object file.
    missing = set(UsdUtils.ComputeAllDependencies(str(target))[2])
    assert missing <= {"OmniPBR.mdl"}, missing
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha256
    return {
        "source": cfg["source_usd_path"],
        "usd": str(target),
        "repaired_meshes": repaired,
        "colliders": len(colliders),
        "builtin_mdl": sorted(missing),
        "mass_models": mass_models,
        "source_sha256": source_sha256,
        "exact_bounding_boxes": boxes,
        "visual_winding_repairs": winding_repairs,
    }
