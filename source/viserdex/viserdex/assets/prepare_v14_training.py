"""Compose V14 collision geometry with the existing teacher's rigid-body data."""

import json
import os
from pathlib import Path


def prepare_v14_training_asset(v14_usd: Path, baseline_usd: Path, output_usd: Path):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    source = Usd.Stage.Open(str(v14_usd))
    baseline = Usd.Stage.Open(str(baseline_usd))
    assert source and baseline
    def bodies(s):
        return {p.GetName(): p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}

    def joints(s):
        return {p.GetName(): p for p in s.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}

    a, b = bodies(source), bodies(baseline)
    assert a.keys() == b.keys() and len(a) == 22
    ja, jb = joints(source), joints(baseline)
    assert ja.keys() == jb.keys() and len(ja) == 21
    for name in a:
        np.testing.assert_array_equal(
            np.asarray(UsdGeom.Xformable(a[name]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
            np.asarray(UsdGeom.Xformable(b[name]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
        )
    for name in ja:
        for attr in jb[name].GetAttributes():
            if attr.GetName() in ("physics:axis", "physics:localPos0", "physics:localPos1",
                                  "physics:localRot0", "physics:localRot1",
                                  "physics:lowerLimit", "physics:upperLimit"):
                assert str(attr.Get()) == str(ja[name].GetAttribute(attr.GetName()).Get()), (name, attr.GetName())

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim(source.GetDefaultPrim().GetPath(), "Xform")
    root.GetReferences().AddReference(os.path.relpath(v14_usd, output_usd.parent), source.GetDefaultPrim().GetPath())
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    # These frames are identical. Preserve the old teacher's mass/COM/inertia,
    # including Newton's explicit full tensor, instead of mesh-derived values.
    keys = ("physics:mass", "physics:density", "physics:centerOfMass", "physics:diagonalInertia",
            "physics:principalAxes", "newton:inertia")
    copied = []
    for name, old in b.items():
        target = stage.GetPrimAtPath(a[name].GetPath())
        for key in keys:
            attr = old.GetAttribute(key)
            if attr and attr.Get() is not None:
                target.CreateAttribute(key, attr.GetTypeName(), custom=attr.IsCustom()).Set(attr.Get())
                copied.append((name, key))
    # URDF-derived Newton passive joint attributes can carry source SysID
    # values even when the USD kinematics match. Keep the teacher's values too.
    joint_keys = {}
    for name, old in jb.items():
        target = stage.GetPrimAtPath(ja[name].GetPath())
        joint_keys[name] = []
        for attr in old.GetAttributes():
            key = attr.GetName()
            if key.startswith(("physics:", "newton:", "mjc:", "drive:")) and attr.Get() is not None:
                target.CreateAttribute(key, attr.GetTypeName(), custom=attr.IsCustom()).Set(attr.Get())
                joint_keys[name].append(key)
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output_usd))
    out_bodies = bodies(stage)
    for name in b:
        for key in keys:
            assert str(out_bodies[name].GetAttribute(key).Get()) == str(b[name].GetAttribute(key).Get()), (name, key)
    out_joints = joints(stage)
    for name, keys_for_joint in joint_keys.items():
        for key in keys_for_joint:
            assert str(out_joints[name].GetAttribute(key).Get()) == str(jb[name].GetAttribute(key).Get()), (name, key)
    colliders = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    assert len(colliders) == 93
    for p in colliders:
        original = source.GetPrimAtPath(p.GetPath())
        assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull"
        assert p.GetAttribute("newton:maxHullVertices").Get() == -1
        assert not any("sdf" in x.GetName().lower() for x in p.GetAttributes())
        np.testing.assert_array_equal(UsdGeom.Mesh(p).GetPointsAttr().Get(), UsdGeom.Mesh(original).GetPointsAttr().Get())
    missing = UsdUtils.ComputeAllDependencies(str(output_usd))[2]
    assert not missing, missing
    result = {"usd": str(output_usd), "v14_geometry": str(v14_usd), "teacher_mechanics": str(baseline_usd),
              "colliders": 93, "active_joints": 21, "rigid_bodies": 22, "sdf": False,
              "body_frames_exact": True, "joint_definitions_exact": True,
              "teacher_mass_com_inertia_exact": True, "v14_mesh_points_exact": True,
              "copied_attributes": copied, "teacher_joint_attributes_exact": True, "copied_joint_attributes": joint_keys, "missing_dependencies": []}
    output_usd.with_suffix(".audit.json").write_text(json.dumps(result, indent=2))
    print("V14_TRAINING_ASSET_READY", json.dumps(result), flush=True)
    return result
