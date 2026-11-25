"""Asset/interface checks that can run without launching Isaac Sim."""

import hashlib
import json
import math
import xml.etree.ElementTree as ET

import pytest

from viserdex.assets.objects import get_rollout_policy_run
from viserdex.assets.revo3_spec import (
    REVO3_ASSET_DIR,
    REVO3_DEFAULT_JOINT_POS,
    REVO3_JOINT_NAMES,
    REVO3_RUNTIME_USD,
)


def test_vendored_files_match_upstream_manifest():
    root = REVO3_ASSET_DIR.parent
    manifest = json.loads((root / "UPSTREAM.json").read_text())
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name


@pytest.mark.parametrize("path", sorted((REVO3_ASSET_DIR / "urdf").glob("*.urdf")))
def test_urdf_meshes_and_joint_interface(path):
    model = ET.parse(path)
    for mesh in model.findall(".//mesh"):
        assert (path.parent / mesh.get("filename")).is_file()
    joints = {j.get("name"): j for j in model.findall("joint") if j.get("type") == "revolute"}
    expected = [n.replace("right_", "left_") if "left" in path.name else n for n in REVO3_JOINT_NAMES]
    assert len(expected) == 21
    assert set(joints) == set(expected)
    # Decorative/tactile fixed links carry mimic metadata in upstream URDFs;
    # none of the 21 revolute joints is coupled to another revolute joint.
    assert all(j.find("mimic") is None for j in joints.values())
    for name, joint in joints.items():
        limits = joint.find("limit")
        default = REVO3_DEFAULT_JOINT_POS[name.replace("left_", "right_")]
        assert float(limits.get("lower")) <= default <= float(limits.get("upper"))


@pytest.mark.parametrize("path", sorted((REVO3_ASSET_DIR / "mjcf").glob("*.xml")))
def test_mjcf_meshes_resolve(path):
    for mesh in ET.parse(path).findall("asset/mesh"):
        assert (path.parent / mesh.get("file")).is_file()


@pytest.mark.parametrize("path", sorted((REVO3_ASSET_DIR / "usd").glob("*.usd")))
def test_usd_matches_urdf(path):
    pytest.importorskip("pxr.Usd", reason="Install usd-core or run in an initialized Isaac Sim interpreter")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim()
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    # OmniPBR is a built-in Isaac Sim material, resolved by Kit's MDL search path.
    assert set(UsdUtils.ComputeAllDependencies(str(path))[2]) <= {"OmniPBR.mdl"}
    urdf_joints = {
        j.get("name"): j.find("limit").attrib
        for j in ET.parse(REVO3_ASSET_DIR / "urdf" / f"{path.stem}.urdf").findall("joint")
        if j.get("type") == "revolute"
    }
    usd_joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert {j.GetPrim().GetName() for j in usd_joints} == set(urdf_joints)
    assert sum(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()) == 1
    for joint in usd_joints:
        limits = urdf_joints[joint.GetPrim().GetName()]
        assert math.radians(joint.GetLowerLimitAttr().Get()) == pytest.approx(float(limits["lower"]), abs=1e-6)
        assert math.radians(joint.GetUpperLimitAttr().Get()) == pytest.approx(float(limits["upper"]), abs=1e-6)


def test_revo3_does_not_reuse_allegro_checkpoint():
    with pytest.raises(ValueError, match="Allegro checkpoints are incompatible"):
        get_rollout_policy_run("Isaac-ViserDex-Repose-Revo3-Estimator-v0")
    assert get_rollout_policy_run("Isaac-ViserDex-Repose-Revo3-Estimator-v0", "revo3_test") == "revo3_test"
    assert get_rollout_policy_run("Isaac-ViserDex-Repose-Allegro-Estimator-v0")


def test_runtime_usd_collision_cooking_limits():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(str(REVO3_RUNTIME_USD))
    assert stage.GetDefaultPrim().GetName() == "revo3_single"
    assert set(UsdUtils.ComputeAllDependencies(str(REVO3_RUNTIME_USD))[2]) <= {"OmniPBR.mdl"}
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()) == 21
    palm = stage.GetPrimAtPath("/revo3_single/world/collisions/right_plam_backtrim_v2/World")
    hull_limit = palm.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit")
    # PhysX reads the schema's signed int; an unsigned override is silently ignored.
    assert str(hull_limit.GetTypeName()) == "int"
    assert 8 <= hull_limit.Get() <= 64
    assert 0.01 <= palm.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get() <= 25


def test_instancing_preserves_geometry_physics_and_materials():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdShade

    original = Usd.Stage.Open(str(REVO3_RUNTIME_USD.with_name("revo3_right_isaaclab.usda")))
    shared = Usd.Stage.Open(str(REVO3_RUNTIME_USD))
    original_prims = {p.GetPath(): p for p in original.Traverse()}
    shared_prims = {p.GetPath(): p for p in Usd.PrimRange(shared.GetDefaultPrim(), Usd.TraverseInstanceProxies())}
    assert original_prims.keys() <= shared_prims.keys()
    assert all("/__materials" in str(path) for path in shared_prims.keys() - original_prims.keys())
    for path, before in original_prims.items():
        after = shared_prims[path]
        assert before.GetTypeName() == after.GetTypeName(), path
        assert before.GetAppliedSchemas() == after.GetAppliedSchemas(), path
        # Includes vertex/topology arrays, transforms, mass/inertia, collision
        # cooking parameters, joint limits and drive settings.
        for attr in before.GetAttributes():
            assert attr.Get() == after.GetAttribute(attr.GetName()).Get(), attr.GetPath()
        for rel in before.GetRelationships():
            after_targets = after.GetRelationship(rel.GetName()).GetTargets()
            if rel.GetName() == "material:binding":
                # External materials are moved inside the shared reference.
                assert [p.name for p in rel.GetTargets()] == [p.name for p in after_targets], rel.GetPath()
            else:
                assert rel.GetTargets() == after_targets, rel.GetPath()
        if before.IsA(UsdGeom.Mesh):
            before_material = UsdShade.MaterialBindingAPI(before).ComputeBoundMaterial()[0]
            after_material = UsdShade.MaterialBindingAPI(after).ComputeBoundMaterial()[0]
            assert bool(before_material) == bool(after_material), path
            if before_material:
                assert before_material.GetPrim().GetName() == after_material.GetPrim().GetName(), path
                for material_prim in Usd.PrimRange(before_material.GetPrim()):
                    other_path = material_prim.GetPath().ReplacePrefix(
                        before_material.GetPath(), after_material.GetPath()
                    )
                    other_prim = shared.GetPrimAtPath(other_path)
                    assert other_prim
                    for attr in material_prim.GetAttributes():
                        other_attr = other_prim.GetAttribute(attr.GetName())
                        assert attr.Get() == other_attr.Get(), attr.GetPath()
                        expected_connections = [
                            p.ReplacePrefix(before_material.GetPath(), after_material.GetPath())
                            for p in attr.GetConnections()
                        ]
                        assert expected_connections == other_attr.GetConnections()


def test_two_robots_share_meshes_but_have_independent_bodies_and_joints():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name in ("A", "B"):
        stage.DefinePrim(f"/{name}").GetReferences().AddReference(str(REVO3_RUNTIME_USD))
    instances = [p for p in stage.Traverse() if p.IsInstance()]
    assert len(instances) == 88
    assert len(stage.GetPrototypes()) == 44
    for prim in instances:
        if prim.GetPath().HasPrefix("/A"):
            other = stage.GetPrimAtPath(prim.GetPath().ReplacePrefix("/A", "/B"))
            assert prim.GetPrototype() == other.GetPrototype()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
            assert not prim.IsInstance() and not prim.IsInstanceProxy()
    a = UsdGeom.Xformable(stage.GetPrimAtPath("/A/right_index_MPR_link"))
    b = UsdGeom.Xformable(stage.GetPrimAtPath("/B/right_index_MPR_link"))
    before_b = b.GetLocalTransformation()
    a.AddTranslateOp(opSuffix="independence_test").Set(Gf.Vec3d(1, 2, 3))
    assert b.GetLocalTransformation() == before_b
    assert a.GetLocalTransformation() != before_b
