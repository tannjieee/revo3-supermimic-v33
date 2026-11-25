"""Bind an independent rounded-die environment; no simulation or training starts."""
from pathlib import Path
import sys, json, hashlib, copy
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.distillation.environment import load_environment, environment_contract
source=ROOT/'configs/rslsphere65/env_smooth.yaml'
out=ROOT/'configs/rounded_die/env_smooth.yaml'
if out.exists(): raise FileExistsError(out)
base=load_environment(source)
c=copy.deepcopy(base)
# Bind independent assets/output and the user-selected lateral initial position.
robot=ROOT/'source/viserdex/viserdex/assets/generated/revo3_compact_visual_palm/revo3_right_compact_v4/revo3_right_compact_v4.usda'
obj=ROOT/'assets/rounded_die/object_newton.usda'
obj.write_text('''#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@object.usda@]
)
over "Object"
{
    over "Collision" (
        prepend apiSchemas = ["NewtonMeshCollisionAPI"]
    )
    {
        int newton:maxHullVertices = 200
    }
}
''')
for p in (robot,obj): assert p.is_file(),p
c['scene']['robot']['spawn']['usd_path']=str(robot)
c['scene']['object']['spawn']['usd_path']=str(obj)
c['commands']['object_pose']['goal_pose_visualizer_cfg']['markers']['goal']['usd_path']=str(obj)
c['commands']['object_pose']['goal_pose_visualizer_cfg']['markers']['goal']['scale']=(1.,1.,1.)
c['log_dir']=str(ROOT/'outputs/rounded_die_smooth')
# Shift toward the little-finger side (world -Y), preserving height and finger-axis position.
c['scene']['object']['init_state']['pos']=(0.02,-0.01,0.565)
undo=copy.deepcopy(c)
undo['scene']['object']['init_state']['pos']=base['scene']['object']['init_state']['pos']
for name in ('robot','object'): undo['scene'][name]['spawn']['usd_path']=base['scene'][name]['spawn']['usd_path']
undo['commands']['object_pose']['goal_pose_visualizer_cfg']=base['commands']['object_pose']['goal_pose_visualizer_cfg']
assert environment_contract(undo)==environment_contract(base)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(yaml.dump(c,sort_keys=False))
manifest=dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Smooth-v1',
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_started=False,
 initialization='scratch_when_authorized',checkpoint_version=14,control_hz=20,ema_alpha=.5,
 total_num_envs=4096,envs_per_gpu=2048,observations_changed=False,randomization_changed=False,
 initial_object_position_m=[0.02,-0.01,0.565],placement_offset_world_m=[0.,-0.01,0.],
 original_asset_unchanged=True,importer_override='NewtonMeshCollisionAPI newton:maxHullVertices=200; wrapper only',
 gs_role='asset supplied; current actor still uses simulated noisy pose, not GS image pixels')
(out.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
