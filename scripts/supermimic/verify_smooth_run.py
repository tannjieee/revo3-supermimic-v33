"""Verify the actual saved production settings against the stopped original run."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
run = ROOT/'outputs/rslsphere65_smooth_ema05_2x2048_scratch_seed42'
parent = Path('/home/zhangke/viserDex_supermimic_rslsphere65_20261029/outputs/revo3_rslsphere65_2x2048_scratch_seed42')
original = json.loads((parent/'config.json').read_text())
actual = json.loads((run/'config.json').read_text())
changes = {k: dict(original=v, smooth=actual['config'].get(k)) for k,v in original['config'].items() if actual['config'].get(k) != v}
assert set(changes) == {'ema_alpha', 'action_rate_weight'}, changes
cfg = actual['config']
assert cfg['ema_alpha'] == .5
assert cfg['joint_vel_weight'] == 3.125e-5
assert cfg['action_l2_weight'] == .000125
assert cfg['action_second_diff_weight'] == cfg['action_rate_weight']
assert cfg['wm_only_iters'] == 500 and cfg['actor_ramp_iters'] == 1000
assert actual['noise_config'] == original['noise_config']
meta = actual['metadata']
assert meta['task_id'] == 'Isaac-ViserDex-Repose-Revo3-SuperMimic-RSLSphere65-Smooth-v1'
assert meta['world_size'] == 2
assert not meta['smoke'] and not meta['check_joint_update'] and not meta['forced_reset_audit']
assert meta['parent_checkpoint'] is None
assert meta['source_root'] == str(ROOT)
for key in ('geometry_digest', 'joint_names', 'observation_protocol'):
    assert meta[key] == original['metadata'][key], key
pids = []
for rank in range(2):
    path = run/(f'rank_{rank}/config.json' if rank else 'config.json')
    data = json.loads(path.read_text())
    assert data['config'] == cfg and data['noise_config'] == actual['noise_config']
    pid = data['metadata']['pid']; os.kill(pid, 0); pids.append(pid)
metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
last = metrics[-1]
assert last['world_size'] == 2 and last['total_num_envs'] == 4096
assert last['parity_max_errors']['action_second_diff'] < 2e-6
report = dict(task_id=meta['task_id'], training_pids=pids, iteration=last['iteration'], phase=last['phase'], initialization='scratch', total_num_envs=4096, changed_original_config_fields=changes, new_config_fields={k:v for k,v in cfg.items() if k not in original['config']}, noise_unchanged=True, observation_protocol_unchanged=True, geometry_and_joint_order_unchanged=True, original_checkpoint_loaded=False, diagnostic_flags_disabled=True, parity_max_errors=last['parity_max_errors'])
print(json.dumps(report, indent=2))
