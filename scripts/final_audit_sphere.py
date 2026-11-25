from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('/home/zhangke/viserDex_supermimic_rslsphere65_20261028');old=Path('/home/zhangke/viserDex_supermimic_horizon20_20261025')
before=json.loads((r/'evidence/protected_before.json').read_text())
after={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if '.git' not in p.parts and '__pycache__' not in p.parts and p.is_file() and not p.is_symlink()}
changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
assets={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'assets/rslsphere65').rglob('*') if p.is_file()}
assert assets['assets/rslsphere65/rslsphere65_gaussians.ply']=='39c1e2c801bfa38eda4d2cba5027709a56acf6e01f47a1cab8cda3a36a93b6da'
assert assets['assets/rslsphere65/RSLSphere65/RSLSphere65.glb']=='32318829419a97770424eb1327607d40cfbd1d40d2b99a2674fd12bb913b9377'
(r/'evidence/asset_hashes.json').write_text(json.dumps(assets,indent=2,sort_keys=True))
result=dict(protected_root=str(old),files_checked=len(before),changed_files=changed,unchanged=not changed,
 final_weight='outputs/revo3_horizon20_2x2048_scratch_seed42/latest.pt',
 final_weight_sha256=after['outputs/revo3_horizon20_2x2048_scratch_seed42/latest.pt'],
 viewer=subprocess.check_output(['ps','-p','1900441','-o','pid,lstart,args'],text=True),
 gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
 shared_environment_modified=False,formal_training_started=False,server21_accessed=False)
(r/'evidence/protection_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert not changed
