"""Audit the appearance-only variant and export a provenance-marked actor preview.

No optimizer state is exported, and no training process is launched or modified.
All compatibility checks must pass before a parent actor can be previewed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/viserdex'))
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from viserdex.distillation.environment import SavedEnvironmentBinding, environment_contract
from viserdex.supermimic.geometry_asset import build_geometry_spec, active_colliders
from viserdex.supermimic.geometry import geometry_digest

TASK = 'Isaac-ViserDex-Repose-Revo3-SuperMimic-YellowDie-WujiOffset-HandPose-Hold-v1'
PARENT_TASK = 'Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-WujiOffset-HandPose-Hold-v1'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent_root', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output_dir', type=Path, required=True)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
parent = a.parent_root.resolve()
old = SavedEnvironmentBinding(parent / 'configs/rounded_die_wuji_handpose/env.yaml')
new = SavedEnvironmentBinding(ROOT / 'configs/yellow_die_handpose/env.yaml')

def relocate(v):
    if isinstance(v, str): return v.replace(str(ROOT), str(parent))
    if isinstance(v, dict): return {k: relocate(x) for k, x in v.items()}
    if isinstance(v, list): return [relocate(x) for x in v]
    if isinstance(v, tuple): return tuple(relocate(x) for x in v)
    return v

assert environment_contract(relocate(new.saved)) == old.contract, 'Non-path environment change'
assert old.assets['robot']['sha256'] == new.assets['robot']['sha256']
old_asset, new_asset = parent / 'assets/rounded_die', ROOT / 'assets/rounded_die'
assert (old_asset/'collision.obj').read_bytes() == (new_asset/'collision.obj').read_bytes()

def physical_signature(path):
    stage = Usd.Stage.Open(str(path))
    assert stage and UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == 'Z'
    signature = {}
    for prim in stage.Traverse():
        attrs = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(('physics:', 'physx', 'newton:')) or (
                str(prim.GetPath()) in ('/Object', '/Object/Collision') and
                (name.startswith('xformOp') or name in ('points', 'faceVertexCounts', 'faceVertexIndices'))
            ):
                attrs[name] = str(attr.Get())
        if attrs: signature[str(prim.GetPath())] = attrs
    colliders = active_colliders(stage)
    assert len(colliders) == 1
    mesh = UsdGeom.Mesh(colliders[0])
    points = np.asarray(mesh.GetPointsAttr().Get())
    return signature, dict(vertices=len(points), triangles=len(mesh.GetFaceVertexCountsAttr().Get()),
                          dimensions_mm=(np.ptp(points, axis=0)*1000).tolist())

sig_old, info_old = physical_signature(old_asset/'object_newton.usda')
sig_new, info_new = physical_signature(new_asset/'object_newton.usda')
assert sig_old == sig_new, 'USD physics, collision or transform changed'
spec_old = build_geometry_spec(old.assets['robot']['path'], old.assets['object']['path'])
spec_new = build_geometry_spec(new.assets['robot']['path'], new.assets['object']['path'])
assert spec_old == spec_new, 'Policy geometry changed'

# Freeze the source bytes first, since latest.pt advances during training.
source_bytes = a.checkpoint.read_bytes()
snapshot = a.output_dir/'parent_checkpoint.pt'
snapshot.write_bytes(source_bytes)
payload = torch.load(snapshot, map_location='cpu', weights_only=False)
assert payload['version'] == 17
assert payload['metadata']['task_id'] == PARENT_TASK
assert payload['metadata']['environment_contract'] == old.contract_sha
assert payload['metadata']['geometry_digest'] == geometry_digest(spec_new)
assert not payload['metadata'].get('smoke') and not payload['metadata'].get('check_joint_update')
report = dict(
    parent_task=PARENT_TASK, task_id=TASK, parent_checkpoint=str(a.checkpoint),
    parent_checkpoint_sha256=hashlib.sha256(source_bytes).hexdigest(), iteration=payload['iteration'],
    old_contract=old.contract_sha, new_contract=new.contract_sha,
    geometry_digest=geometry_digest(spec_new), collision=info_new,
    collision_obj_identical=True, usd_physics_identical=True, geometry_spec_identical=True,
    environment_equal_except_relocated_paths=True,
    change='Yellow/black visual mesh, materials and matching Gaussian assets',
    observation='Original noisy 124 x 10 state observations; no rendered RGB input',
    training_started=False, export_purpose='Actor-only preview, not a training-resume checkpoint',
)
preview = {k: payload[k] for k in ('version', 'iteration', 'config', 'noise_config', 'actor')}
preview['metadata'] = copy.deepcopy(payload['metadata'])
preview['metadata'].update(task_id=TASK, environment_contract=new.contract_sha,
                           actor_only_preview=True, appearance_transfer=report)
torch.save(preview, a.output_dir/'actor_preview.pt')
(a.output_dir/'appearance_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
