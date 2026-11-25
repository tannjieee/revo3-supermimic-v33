"""Bind the existing 58 mm package to the same Newton convex-hull contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]


def generate(source, output):
    metadata = json.loads((source/'asset_manifest.json').read_text())
    original = source/'dice_58mm.usda'
    stage = Usd.Stage.Open(str(original))
    stage = Usd.Stage.Open(stage.Flatten())
    collision = stage.GetPrimAtPath('/Object/Collision')
    schemas = list(collision.GetMetadata('apiSchemas').GetAppliedItems())
    if 'NewtonMeshCollisionAPI' not in schemas:
        schemas.append('NewtonMeshCollisionAPI')
    collision.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(schemas))
    collision.CreateAttribute('newton:maxHullVertices', Sdf.ValueTypeNames.Int).Set(200)
    visual = UsdGeom.Mesh(stage.GetPrimAtPath('/Object/Visual/Part_0'))
    hull = UsdGeom.Mesh(collision)
    dims = np.ptp(np.asarray(visual.GetPointsAttr().Get()), axis=0)
    collision_dims = np.ptp(np.asarray(hull.GetPointsAttr().Get()), axis=0)
    np.testing.assert_allclose(dims, .058, atol=1e-7)
    np.testing.assert_allclose(collision_dims, dims, atol=1e-9)
    assert len(hull.GetPointsAttr().Get()) == 200
    mass = UsdPhysics.MassAPI(stage.GetDefaultPrim())
    assert np.isclose(mass.GetMassAttr().Get(), metadata['mass_kg'], rtol=1e-6)
    output.mkdir(parents=True, exist_ok=False)
    target = output/'object_newton.usda'
    stage.GetRootLayer().Export(str(target))
    shutil.copy2(source/'ATTRIBUTION.txt', output/'ATTRIBUTION.txt')
    manifest = dict(variant='rounded_die_58mm_packaged_v1',
        source=str(original.resolve()), source_usd_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        source_manifest=metadata, usd_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        dimensions_m=dims.tolist(), collision_dimensions_m=collision_dims.tolist(),
        mass_kg=metadata['mass_kg'], mass_source=metadata['mass_source'], mass_is_estimated=True,
        diagonal_inertia_kg_m2=list(mass.GetDiagonalInertiaAttr().Get()),
        material_srgb={'yellow_body':'#F4EE2A', 'black_pips':'#123E35'},
        visual_faces=len(visual.GetFaceVertexCountsAttr().Get()), collision_vertices=200,
        source_assets_modified=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/rounded_die_58mm')
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output), indent=2))
