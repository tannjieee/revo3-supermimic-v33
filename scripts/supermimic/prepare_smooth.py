"""Create Smooth-v1's independent environment binding without loading weights."""
from pathlib import Path
import sys
import json
import hashlib
import copy
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/viserdex'))
from viserdex.distillation.environment import load_environment, environment_contract
from viserdex.supermimic.core import config20

out = ROOT / 'configs/rslsphere65/env_smooth.yaml'
if out.exists():
    raise FileExistsError(out)
source = ROOT / 'configs/rslsphere65/env_training_original.yaml'
original = load_environment(source)
def relocate(value):
    if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list): return [relocate(v) for v in value]
    if isinstance(value, tuple): return tuple(relocate(v) for v in value)
    if isinstance(value, str) and value.startswith('/'):
        for marker in ('/source/viserdex/', '/assets/rslsphere65/'):
            if marker in value:
                path = ROOT / (marker.lstrip('/') + value.split(marker, 1)[1])
                if not path.is_file(): raise FileNotFoundError(path)
                return str(path.resolve())
    return value
baseline = relocate(original)
smooth = copy.deepcopy(baseline)
cfg = config20()
smooth['actions']['joint_pos']['alpha'] = cfg.ema_alpha
for name, weight in [('joint_vel_l2', cfg.joint_vel_weight), ('action_l2', cfg.action_l2_weight), ('action_rate_l2', cfg.action_rate_weight)]:
    smooth['rewards'][name]['weight'] = -weight
smooth['rewards']['action_second_diff_l2'] = dict(func='viserdex.supermimic.action_smoothness:ActionSecondDifference', params={}, weight=-cfg.action_second_diff_weight)
# Assert that only the five requested semantic edits occurred.
undo = copy.deepcopy(smooth)
undo['actions']['joint_pos']['alpha'] = baseline['actions']['joint_pos']['alpha']
undo['rewards'] = baseline['rewards']
assert environment_contract(undo) == environment_contract(baseline)
with out.open('w') as stream: yaml.dump(smooth, stream, sort_keys=False)
manifest = dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RSLSphere65-Smooth-v1', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), environment=str(out), ema_alpha=cfg.ema_alpha, penalty_increase=1.25, second_diff_weight=cfg.action_second_diff_weight, initialization='scratch', original_weights_loaded=False, observations_changed=False, randomization_changed=False)
(ROOT/'configs/rslsphere65/smooth_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
