"""Create the independent Hold-v1 binding; never launch or overwrite a run."""
from pathlib import Path
import sys, json, copy, hashlib
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/viserdex'))
from viserdex.distillation.environment import load_environment, environment_contract
source = ROOT/'configs/rounded_die/env_smooth.yaml'
out = ROOT/'configs/rounded_die_hold/env.yaml'
if out.exists(): raise FileExistsError(out)
base = load_environment(source)
c = copy.deepcopy(base)
old_root = str(Path(base['scene']['object']['spawn']['usd_path']).parents[2])
def relocate(v):
    if isinstance(v, str): return v.replace(old_root, str(ROOT))
    if isinstance(v, dict): return {k:relocate(x) for k,x in v.items()}
    if isinstance(v, tuple): return tuple(relocate(x) for x in v)
    if isinstance(v, list): return [relocate(x) for x in v]
    return v
c = relocate(c)
c['commands']['object_pose'].update(class_type='viserdex.supermimic.hold_command:HoldReorientCommand',
    orientation_success_threshold=0.1, hold_speed_threshold=0.5, hold_duration=0.5, update_goal_on_success=True)
c['rewards']['success_bonus'].update(func='viserdex.supermimic.hold_command:hold_success_bonus', params={'command_name':'object_pose'})
c['log_dir'] = str(ROOT/'outputs/rounded_die_hold')
undo = copy.deepcopy(c)
undo['commands'] = relocate(base['commands'])
undo['rewards'] = relocate(base['rewards'])
assert environment_contract(undo) == environment_contract(relocate(base))
assert tuple(c['scene']['object']['init_state']['pos']) == (0.02,-0.01,0.565)
for name in ('robot','object'): assert Path(c['scene'][name]['spawn']['usd_path']).is_file()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(c,sort_keys=False))
manifest = dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Hold-v1',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_version=15,
    initialization='scratch',training_started=False,angle_threshold_rad=0.1,
    angular_speed_threshold_rad_s=0.5,hold_duration_seconds=0.5,hold_control_steps=10,
    control_hz=20,state_dim=173,actor_input=[10,124],physics_prediction_dim=55,
    randomization_changed=False,penalty_weights_changed=False,initial_position_m=[.02,-.01,.565])
(out.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
