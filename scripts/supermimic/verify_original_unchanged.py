"""Read-only verification of the original source and its frozen latest checkpoint."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
manifest = Path('/home/zhangke/viserDex_student_ppo_rslsphere65_20261029/evidence/parent_protection_before.json')
data = json.loads(manifest.read_text())
parent = Path(data['parent'])
changed = [name for name, digest in data['source_hashes'].items() if hashlib.sha256((parent/name).read_bytes()).hexdigest() != digest]
assert not changed, changed
checkpoint = Path('/home/zhangke/rslsphere65_preserved/supermimic_stopped_20261029_iter19234/checkpoint.pt')
expected = '81a1d6bfe9b5128489d12b3f648a7b4a6a7ba6bc564953e4a6e70a4503a94db3'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
run = parent/'outputs/revo3_rslsphere65_2x2048_scratch_seed42'
assert hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest() == expected
assert hashlib.sha256((run/'config.json').read_bytes()).hexdigest() == data['parent_config_sha256']
state = json.loads((parent/'outputs/active_rslsphere65_scratch.json').read_text())
assert state['status'] == 'stopped', state['status']
report = dict(source_files_verified=len(data['source_hashes']), source_changed=False, original_training_status=state['status'], original_iteration=19234, frozen_checkpoint_sha256=expected, live_original_checkpoint_sha256=expected, original_config_changed=False)
print(json.dumps(report, indent=2))
