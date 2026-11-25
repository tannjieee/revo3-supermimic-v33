"""Reference the user's already decomposed USD without re-cooking its convex pieces."""

import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
from .revo3_spec import REVO3_JOINT_NAMES

MANUAL_SOURCE_COMMIT = '2a98621de70d59d182b8075214cebb96fd5375d3'
MANUAL_SOURCE_SHA256 = '737c42704c4e464b9da372142683f206ff43cbc1d9aacb5e25d05a4c7ce1914f'
REPOSITORY = Path(__file__).resolve().parents[4]
MANUAL_USD = REPOSITORY / 'generated/revo3_manual_convex/revo3_right.usda'


def default_manual_source():
    relative = Path('assets/revo3_d405/vendor/brainco-description/revo3_system/optimized/right_20261027/revo3_right.usd')
    for base in (REPOSITORY, *REPOSITORY.parents):
        if (base / relative).is_file():
            return base / relative
    raise FileNotFoundError('Manual convex asset is missing; specify --manual_convex_usd')


def inspect_manual_asset(path):
    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected a meter-scale Z-up manual convex USD')
    joints = {p.GetName(): UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
              if p.IsA(UsdPhysics.RevoluteJoint)}
    if set(joints) != set(REVO3_JOINT_NAMES):
        raise ValueError('Manual asset joint names differ from the 21-joint policy order')
    limits = [[math.radians(joints[n].GetLowerLimitAttr().Get()),
               math.radians(joints[n].GetUpperLimitAttr().Get())] for n in REVO3_JOINT_NAMES]
    limits = np.asarray(limits)
    if not np.isfinite(limits).all() or not (limits[:, 1] > limits[:, 0]).all():
        raise ValueError('Invalid asset joint limits')
    colliders = [p for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if len(colliders) != 144:
        raise ValueError(f'Expected 144 authored convex instances, got {len(colliders)}')
    for p in colliders:
        if not p.IsA(UsdGeom.Mesh) or UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() != 'convexHull':
            raise ValueError(f'Non-convex collision primitive: {p.GetPath()}')
        if p.GetAttribute('newton:maxHullVertices').Get() != -1:
            raise ValueError(f'Explicit hull vertices must be preserved: {p.GetPath()}')
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 22:
        raise ValueError('Expected 22 rigid bodies')
    for p in bodies:
        mass = UsdPhysics.MassAPI(p)
        inertia = np.array(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
        if not mass.GetMassAttr().Get() > 0 or not np.isfinite(inertia).all() or not (inertia > 0).all():
            raise ValueError(f'Invalid mass/inertia: {p.GetPath()}')
    # OmniPBR is an Isaac Sim MDL search-path module, not a missing USD layer or mesh.
    # The standalone USD resolver cannot resolve MDL search-path identifiers.
    missing = [item for item in UsdUtils.ComputeAllDependencies(str(path))[2] if item != 'OmniPBR.mdl']
    if missing:
        raise ValueError(f'Missing USD dependencies: {missing}')
    return dict(joint_names=list(REVO3_JOINT_NAMES), limits_rad=limits.tolist(),
                collision_instances=len(colliders), rigid_bodies=len(bodies),
                joint_frame_quaternion_conversion='USD Gf: imaginary XYZ then real W; runtime XYZW',
                root_prim=str(stage.GetDefaultPrim().GetPath()))


def prepare_manual_convex(source, output_dir):
    source, output_dir = Path(source).resolve(strict=True), Path(output_dir).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != MANUAL_SOURCE_SHA256:
        raise ValueError('Manual convex source differs from the pinned right-hand asset revision')
    original = inspect_manual_asset(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'revo3_right.usda'
    # A reference layer preserves the user's vertices, transforms, joint frames and limits.
    stage = Usd.Stage.Open(str(target)) if target.exists() else Usd.Stage.CreateNew(str(target))
    stage.GetRootLayer().Clear()
    prim = stage.DefinePrim(original['root_prim'], 'Xform')
    prim.GetReferences().AddReference(os.path.relpath(source, output_dir))
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetRootLayer().Save()
    actual = inspect_manual_asset(target)
    if actual != original:
        raise ValueError('Reference layer changed the asset contract')
    report = dict(actual, source=str(source), source_commit=MANUAL_SOURCE_COMMIT, source_sha256=digest, usd=str(target),
                  usd_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  convex_decomposition_performed=False, source_modified=False)
    (output_dir / 'mechanical-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
