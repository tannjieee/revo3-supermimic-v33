"""Build a separate, self-contained Duck USD matched to Revo3 finger reach.

Run with an environment providing pxr (e.g. .venv-sim6/bin/python).
The input Duck directory is never modified. Existing output is refused.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


ALLEGRO_REACH_M = 0.15840233628281694
REVO3_REACH_M = 0.10669390306832129
SCALE = REVO3_REACH_M / ALLEGRO_REACH_M
SOURCE_SHA256 = "885fcbe52da9a226b9ffbad3322b54cd90ba1f7e49ca49e8b9e9663a6cd701ed"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(stage, prim):
    return list(
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
        .GetSize()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd-dir", type=Path, default=Path(__file__).resolve().parents[2] / "source/viserdex/viserdex/assets/data/usd"
    )
    args = parser.parse_args()
    source = args.usd_dir / "Duck" / "Duck.usd"
    destination = args.usd_dir / "DuckRevo3Scaled"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Duck source differs from the measured asset; re-audit before scaling.")
    original = Usd.Stage.Open(str(source))
    original_size = bounds(original, original.GetDefaultPrim())
    original_mass = UsdPhysics.MassAPI(original.GetDefaultPrim()).GetMassAttr().Get()
    # Copy textures and instanceable geometry as well, so this asset has no dependency on Duck/.
    shutil.copytree(source.parent, destination)
    target = destination / "DuckRevo3Scaled.usd"
    (destination / "Duck.usd").rename(target)
    stage = Usd.Stage.Open(str(target))
    root = stage.GetDefaultPrim()
    assert not UsdGeom.Xformable(root).GetOrderedXformOps()
    # Keep the rigid-body root at unit scale. The goal marker explicitly sets root
    # scale=(1,1,1), so root scaling would otherwise disappear for that marker.
    geometry_root = root.GetChild("geometry")
    assert geometry_root.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(0)
    assert geometry_root.GetAttribute("xformOp:rotateXYZ").Get() == Gf.Vec3f(0)
    scale_attr = geometry_root.GetAttribute("xformOp:scale")
    assert scale_attr.Get() == Gf.Vec3f(1)
    scale_attr.Set(Gf.Vec3f(SCALE))
    # The source has an explicit 40 g mass, which takes precedence over spawn density.
    # Preserve density: mass ~ length^3. No source COM/inertia overrides are authored;
    # PhysX computes them from the scaled collision geometry and the new mass.
    mass = UsdPhysics.MassAPI(root)
    assert not mass.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
    assert not mass.GetCenterOfMassAttr().HasAuthoredValueOpinion()
    mass.GetMassAttr().Set(original_mass * SCALE**3)
    stage.GetRootLayer().Save()

    # Contact offsets are distances in meters, independent of the mesh's xform scale.
    geometry = Usd.Stage.Open(str(destination / "Props/instanceable_meshes.usd"))
    for prim in geometry.Traverse():
        for name in ("physxCollision:contactOffset", "physxCollision:restOffset"):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                attr.Set(attr.Get() * SCALE)
    geometry.GetRootLayer().Save()
    stage.Reload()
    new_size = bounds(stage, root)
    collisions = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            old_prim = original.GetPrimAtPath(prim.GetPath())
            before, after = bounds(original, old_prim), bounds(stage, prim)
            assert all(abs(b * SCALE - a) < 1e-7 for b, a in zip(before, after))
            collisions.append({"prim": str(prim.GetPath()), "before_m": before, "after_m": after})
    assert collisions
    assert all(abs(b * SCALE - a) < 1e-7 for b, a in zip(original_size, new_size))
    assert digest(source) == SOURCE_SHA256
    report = {
        "object": "DuckRevo3Scaled",
        "scale": SCALE,
        "metric": "authored straight middle MCP-to-tip reach, including BioTac on Allegro",
        "allegro_reach_m": ALLEGRO_REACH_M,
        "revo3_reach_m": REVO3_REACH_M,
        "measurement_source": "logs/size_audit/20261015_1608/comparison_landmarks.json",
        "source_sha256": SOURCE_SHA256,
        "original_dimensions_m": original_size,
        "scaled_dimensions_m": new_size,
        "original_mass_kg": original_mass,
        "scaled_mass_kg": mass.GetMassAttr().Get(),
        "collision_bounds": collisions,
        "notes": [
            "Hands are not geometrically similar; this matches finger reach, not palm width.",
            "Initial pose, reset distribution, reward and training configuration are unchanged.",
            "Teacher USD asset only; scaled splats/keypoints/mesh exports are not supplied.",
        ],
        "files": {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()},
    }
    (destination / "scale_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
