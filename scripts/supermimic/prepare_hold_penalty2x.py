"""Generate an isolated Hold binding with four doubled penalties and Kp=3."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/viserdex'))
from viserdex.distillation.environment import load_environment, environment_contract
from viserdex.supermimic.core import config20

source = ROOT / 'configs/rounded_die_hold/env.yaml'
out = ROOT / 'configs/rounded_die_hold_penalty2x/env.yaml'
if out.exists():
    raise FileExistsError(out)
base = load_environment(source)
old_root = str(Path(base['scene']['object']['spawn']['usd_path']).parents[2])

def relocate(value):
    if isinstance(value, str):
        return value.replace(old_root, str(ROOT))
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(relocate(v) for v in value)
    if isinstance(value, list):
        return [relocate(v) for v in value]
    return value

base = relocate(base)
cfg = copy.deepcopy(base)
assert cfg['scene']['robot']['actuators']['fingers']['stiffness'] == 7.0
cfg['scene']['robot']['actuators']['fingers']['stiffness'] = 3.0
settings = config20()
terms = {'joint_vel_l2': 'joint_vel_weight', 'action_l2': 'action_l2_weight',
         'action_rate_l2': 'action_rate_weight', 'action_second_diff_l2': 'action_second_diff_weight'}
changes = {}
for name, field in terms.items():
    before = base['rewards'][name]['weight']
    after = before * 2
    assert abs(after + getattr(settings, field)) < 1e-12
    cfg['rewards'][name]['weight'] = after
    changes[name] = dict(before=before, after=after)
cfg['log_dir'] = str(ROOT / 'outputs/rounded_die_hold_penalty2x')
undo = copy.deepcopy(cfg)
undo['rewards'] = copy.deepcopy(base['rewards'])
undo['scene']['robot']['actuators']['fingers']['stiffness'] = 7.0
assert environment_contract(undo) == environment_contract(base)
for name in ('robot', 'object'):
    assert Path(cfg['scene'][name]['spawn']['usd_path']).is_file()
out.parent.mkdir(parents=True, exist_ok=False)
out.write_text(yaml.dump(cfg, sort_keys=False))
manifest = dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Hold-Penalty2x-v1',
                parent_task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Hold-v1',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                penalty_multiplier=2.0, changes=changes, initialization='scratch',
                parent_checkpoint=None, training_started=False,
                nominal_kp=3.0, previous_nominal_kp=7.0, kp_randomization_range=[1.8, 4.5],
                ema_alpha=settings.ema_alpha, control_hz=1/settings.control_dt,
                hold_duration=settings.hold_duration, randomization_changed=False)
(out.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
