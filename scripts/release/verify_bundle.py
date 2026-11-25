"""Verify all published payload files, including materialized Git LFS objects."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'release/files.sha256.json').read_text())
failures=[]
for name,record in manifest.items():
    p=ROOT/name
    if not p.is_file():
        failures.append(name+': missing'); continue
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if p.stat().st_size != record['bytes'] or digest != record['sha256']:
        failures.append(name+': content mismatch (run git lfs pull for model pointers)')
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: {len(manifest)} published files match sizes and SHA-256.')
