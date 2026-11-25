"""Generate and audit Revo3 with Isaac Lab's official multi-physics importer.

Run in a fresh interpreter before launching either physics backend. The importer
extra must be installed from the official v3.0.0 lock and dependency overrides.
Vendored inputs and the historical PhysX USD remain unchanged.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.utils.version import standalone_importers_available
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from viserdex.assets.revo3_spec import REVO3_ASSET_DIR, REVO3_JOINT_NAMES, REVO3_MULTIPHYSICS_USD
from viserdex.assets.objects import TARGET_OBJECT_CFG
from viserdex.assets.prepare_object import prepare_object


def main(build_palm_sdf=False):
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(REVO3_ASSET_DIR / "urdf/revo3_right.urdf"),
            usd_dir=str(REVO3_MULTIPHYSICS_USD.parent.parent),
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=True,
            collision_type="Convex Decomposition",
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=7.0, damping=0.22)
            ),
        )
    )
    import numpy as np
    from pxr import Usd, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(converter.usd_path)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert set(joints) == set(REVO3_JOINT_NAMES), joints
    assert len(bodies) == 22, len(bodies)
    records = []
    for body in bodies:
        mass = UsdPhysics.MassAPI(body)
        m = mass.GetMassAttr().Get()
        inertia = np.array(mass.GetDiagonalInertiaAttr().Get())
        assert m is not None and np.isfinite(m) and m > 0, body.GetPath()
        assert np.isfinite(inertia).all() and (inertia > 0).all(), body.GetPath()
        assert inertia.max() <= inertia.sum() - inertia.max() + 1e-8, body.GetPath()
        records.append({"body": str(body.GetPath()), "mass": m, "diagonal_inertia": inertia.tolist()})
    colliders = [
        p
        for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert colliders, "No colliders in converted asset"
    missing = UsdUtils.ComputeAllDependencies(converter.usd_path)[2]
    assert not missing, missing
    result = {"usd": converter.usd_path, "joint_names": joints, "bodies": records, "colliders": len(colliders)}
    Path(converter.usd_path).with_name("mechanical-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("REVO3_MULTIPHYSICS_READY:", converter.usd_path, f"bodies={len(bodies)}, colliders={len(colliders)}")
    print("OBJECT_MULTIPHYSICS_READY:", json.dumps(prepare_object(TARGET_OBJECT_CFG)))
    if build_palm_sdf:
        from viserdex.assets.prepare_revo3_palm_sdf import prepare_palm_sdf

        report = prepare_palm_sdf(converter.usd_path, REVO3_MULTIPHYSICS_USD.parents[2] / "revo3_palm_sdf")
        print("PALM_SDF_ASSET_READY:", json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palm_sdf", action="store_true", help="Also build the 1200-face fixed-palm Newton SDF.")
    parser.add_argument("--convex_urdf", type=Path, help="Prepare an already partitioned convex URDF separately.")
    parser.add_argument("--usd_dir", type=Path, help="Separate output directory for --convex_urdf.")
    parser.add_argument("--manual_convex_usd", type=Path, help="Reference the authored 144-piece hand without new decomposition.")
    parser.add_argument("--v14_training", action="store_true", help="Compose V14 convex geometry with the existing teacher mechanics.")
    parser.add_argument("--exact_hull_prefix", action="append", help="Keep all vertices only on these collision path fragments; other hulls keep the original 64-vertex cooking.")
    parser.add_argument("--convex_joint_stiffness", type=float, default=0.0)
    parser.add_argument("--convex_joint_damping", type=float, default=0.0)
    add_launcher_args(parser)
    args = parser.parse_args()
    # Same importer launch decision as official scripts/tools/convert_urdf.py.
    args.require_kit = not standalone_importers_available()
    args.physics = "isaacsim_physx" if args.require_kit else "newton_mjwarp"
    with launch_simulation(PhysicsCfg(), args):
        if args.manual_convex_usd is not None:
            if args.usd_dir is None or args.convex_urdf is not None or args.v14_training or args.palm_sdf:
                parser.error("--manual_convex_usd requires --usd_dir and excludes other conversion modes")
            from viserdex.assets.prepare_manual_convex import prepare_manual_convex
            print(json.dumps(prepare_manual_convex(args.manual_convex_usd, args.usd_dir), indent=2))
        elif args.v14_training:
            if args.convex_urdf is not None or args.palm_sdf:
                parser.error("--v14_training cannot be combined with another conversion mode")
            from viserdex.assets.prepare_v14_training import prepare_v14_training_asset
            from viserdex.assets.revo3_spec import REVO3_V14_CONVEX_USD, REVO3_PALM_SDF_USD, REVO3_V14_TRAINING_USD

            prepare_v14_training_asset(REVO3_V14_CONVEX_USD, REVO3_PALM_SDF_USD, REVO3_V14_TRAINING_USD)
        elif args.convex_urdf is not None:
            if args.usd_dir is None or args.palm_sdf:
                parser.error("--convex_urdf requires --usd_dir and cannot use --palm_sdf")
            from viserdex.assets.prepare_convex_newton import prepare_convex_newton

            prepare_convex_newton(args.convex_urdf, args.usd_dir, exact_hull_prefixes=args.exact_hull_prefix, stiffness=args.convex_joint_stiffness, damping=args.convex_joint_damping)
        else:
            main(build_palm_sdf=args.palm_sdf)
