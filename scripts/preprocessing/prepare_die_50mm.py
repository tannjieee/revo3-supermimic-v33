"""Bake the accepted original die's 5/7 scale into an independent SI USD.

The print project's bottom-centered frame is deliberately not used for physics:
retain the original body frame and face numbering. The authored 200-vertex outer
convex collider scales with the visual, with no hull rebuild or engine changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[2]


def generate(output, accepted_source, mass_kg):
    source = ROOT/'assets/rounded_die'
    accepted = json.loads((accepted_source/'validation.json').read_text())
    assert accepted['shape_rebuilt'] is False and accepted['uniform_scale'] == 5/7
    for name, sha in accepted['source_sha256'].items():
        assert hashlib.sha256((source/'meshes'/name).read_bytes()).hexdigest() == sha
    output.mkdir(parents=True, exist_ok=False)
    scale = 5/7
    original = Usd.Stage.Open(str(source/'object_newton.usda'))
    stage = Usd.Stage.Open(original.Flatten())
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            assert not UsdGeom.Xformable(prim).GetOrderedXformOps()
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)*scale
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0),points.max(0)]).astype(np.float32)))
    mass = UsdPhysics.MassAPI(stage.GetDefaultPrim())
    old_mass = float(mass.GetMassAttr().Get())
    new_mass = mass_kg
    if not np.isfinite(new_mass) or new_mass <= 0:
        raise ValueError('Mass must be positive kg')
    inertia_factor = (new_mass/old_mass)*scale**2
    mass.GetMassAttr().Set(new_mass)
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(np.asarray(mass.GetDiagonalInertiaAttr().Get())*inertia_factor).tolist()))
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*(np.asarray(mass.GetCenterOfMassAttr().Get())*scale).tolist()))
    # Approved print variant: yellow body and matte midnight-green pips.
    colors = {'yellow_body': ('F4EE2A', .55), 'black_pips': ('123E35', .85)}
    for name, (rgb, roughness) in colors.items():
        srgb = np.array([int(rgb[i:i+2],16)/255 for i in (0,2,4)])
        linear = np.where(srgb<=.04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
        shader = UsdShade.Shader(stage.GetPrimAtPath(f'/Object/Materials/{name}/Surface'))
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*linear.tolist()))
        shader.GetInput('roughness').Set(roughness)
    stage.GetRootLayer().Export(str(output/'object_newton.usda'))
    shutil.copyfile(source/'ATTRIBUTION.txt',output/'ATTRIBUTION.txt')
    visual = UsdGeom.Mesh(stage.GetPrimAtPath('/Object/Visual/Part_0'))
    collision = UsdGeom.Mesh(stage.GetPrimAtPath('/Object/Collision'))
    manifest = dict(variant='rounded_die_50mm_original_v1', uniform_scale=scale,
        accepted_source=str(accepted_source), accepted_source_validation=accepted,
        source_usd_sha256=hashlib.sha256((source/'object.usda').read_bytes()).hexdigest(),
        source_wrapper_sha256=hashlib.sha256((source/'object_newton.usda').read_bytes()).hexdigest(),
        usd_sha256=hashlib.sha256((output/'object_newton.usda').read_bytes()).hexdigest(),
        dimensions_m=np.ptp(np.asarray(visual.GetPointsAttr().Get()),axis=0).tolist(),
        collision_dimensions_m=np.ptp(np.asarray(collision.GetPointsAttr().Get()),axis=0).tolist(),
        visual_faces=len(visual.GetFaceVertexCountsAttr().Get()), collision_vertices=len(collision.GetPointsAttr().Get()),
        mass_kg=new_mass, mass_source='user_supplied',
        inertia_scale=inertia_factor, diagonal_inertia_kg_m2=list(mass.GetDiagonalInertiaAttr().Get()),
        body_frame='original body frame uniformly scaled; not print-bed coordinates',
        material_srgb={name:'#'+rgb for name,(rgb,_) in colors.items()},
        collision_method='original 200-vertex outer convex hull uniformly scaled; pips remain visual concavities',
        source_assets_modified=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accepted_source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'assets/rounded_die_50mm')
    p.add_argument('--mass_kg',type=float,required=True,help='Explicit physical mass in kg; never inferred from dimensions')
    args=p.parse_args()
    print(json.dumps(generate(args.output,args.accepted_source,args.mass_kg),indent=2))
