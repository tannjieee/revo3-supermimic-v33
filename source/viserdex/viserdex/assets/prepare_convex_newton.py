"""Prepare an explicitly partitioned convex URDF through the official importer."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare_convex_newton(source_urdf: Path, usd_dir: Path, *, exact_hull_prefixes=None, stiffness=0.0, damping=0.0):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties
    from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg

    tree = ET.parse(source_urdf).getroot()
    expected = len(tree.findall('.//collision'))
    names = {j.get('name') for j in tree.findall('joint') if j.get('type') != 'fixed'}
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(source_urdf.resolve()), usd_dir=str(usd_dir.resolve()),
        fix_base=True, merge_fixed_joints=True, self_collision=True,
        collision_type='Convex Hull', merge_mesh=False,
        run_asset_transformer=True, run_multi_physics_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping)),
    ))
    stage = Usd.Stage.Open(converter.usd_path)
    collision_paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                       if p.HasAPI(UsdPhysics.CollisionAPI)
                       and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    assert expected > 0 and len(collision_paths) == expected, (len(collision_paths), expected)
    # Author only into the generated interface layer. Keeping every hull vertex
    # avoids the generic 64-vertex approximation changing this selected asset.
    for path in collision_paths:
        ancestor = stage.GetPrimAtPath(path)
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.IsInstance():
                ancestor.SetInstanceable(False)
            ancestor = ancestor.GetParent()
        exact = exact_hull_prefixes is None or any(prefix in path for prefix in exact_hull_prefixes)
        assert apply_mesh_collision_properties(path, [
            UsdPhysicsMeshCollisionCfg(mesh_approximation_name='convexHull'),
            NewtonMeshCollisionCfg(max_hull_vertices=-1 if exact else 64),
        ], stage=stage)
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(converter.usd_path)
    colliders = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    assert len(colliders) == expected
    for p in colliders:
        assert p.IsA(UsdGeom.Mesh), str(p.GetPath())
        assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == 'convexHull'
        exact = exact_hull_prefixes is None or any(prefix in str(p.GetPath()) for prefix in exact_hull_prefixes)
        assert p.GetAttribute('newton:maxHullVertices').Get() == (-1 if exact else 64)
        assert not any('sdf' in a.GetName().lower() for a in p.GetAttributes())
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert {p.GetName() for p in joints} == names
    bodies = []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(p)
            mass = m.GetMassAttr().Get()
            inertia = np.array(m.GetDiagonalInertiaAttr().Get())
            assert mass is not None and mass > 0 and np.isfinite(mass), str(p.GetPath())
            assert np.isfinite(inertia).all() and (inertia > 0).all()
            assert inertia.max() <= inertia.sum() - inertia.max() + 1e-8
            bodies.append({'name': p.GetName(), 'path': str(p.GetPath()), 'mass': mass,
                           'com': list(m.GetCenterOfMassAttr().Get()),
                           'diagonal_inertia': inertia.tolist()})
    assert len(bodies) == 22, len(bodies)
    missing = [str(x) for x in UsdUtils.ComputeAllDependencies(converter.usd_path)[2]]
    assert not missing, missing
    result = {'usd': converter.usd_path, 'source_urdf': str(source_urdf), 'colliders': len(colliders),
              'collision_approximation': 'convexHull', 'max_hull_vertices': -1 if exact_hull_prefixes is None else 'selected=-1; others=64', 'exact_hull_prefixes': exact_hull_prefixes, 'sdf': False,
              'joint_names': sorted(names), 'bodies': bodies, 'missing_dependencies': missing}
    Path(converter.usd_path).with_name('convex-audit.json').write_text(json.dumps(result, indent=2))
    print('CONVEX_NEWTON_READY', json.dumps(result), flush=True)
    return result
