from pathlib import Path
import hashlib,json,os,shutil
r=Path('/home/zhangke/viserDex_supermimic_rslsphere65_20261028')
old=Path('/home/zhangke/viserDex_supermimic_horizon20_20261025')
if (r / "evidence/protected_before.json").exists():
 raise SystemExit("One-time setup already recorded; do not overwrite protection baseline")
# Hash protected regular files, including source, logs and weights. Do not modify the source index.
def manifest(root):
 out={}
 for p in root.rglob('*'):
  if '.git' in p.parts or '__pycache__' in p.parts or p.is_symlink() or not p.is_file(): continue
  out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out
(r/'evidence/protected_before.json').write_text(json.dumps(manifest(old),sort_keys=True))
# Keep a private copy of the generated robot tree. Never write through the inherited symlink.
p=r/'source/viserdex/viserdex/assets/generated'
target=p.resolve();p.unlink();shutil.copytree(target,p,symlinks=True)
p=r/'source/viserdex/viserdex/assets/objects.py';s=p.read_text();a=s.index('# Generated objects carry')
s=s[:a]+'''_sphere_dir = Path(__file__).resolve().parents[4] / "assets/rslsphere65"
_OBJECT_CFGS["RSLSphere65"] = {
    "usd_path": str(_sphere_dir / "RSLSphere65/RSLSphere65.usda"),
    "mesh_path": str(_sphere_dir / "RSLSphere65/RSLSphere65.glb"),
    "splat_path": str(_sphere_dir / "rslsphere65_gaussians.ply"),
    "points_path": None, "supports_pose_estimation": False,
    "rollout_policy_run_name": None, "revo3_rollout_policy_run_name": None,
}

'''+s[a:]
a=s.index('\n\ndef get_rollout_policy_run');s=s[:a]+'''\nif object_name == "RSLSphere65":
    TARGET_OBJECT_CFG["usd_path"] = TARGET_OBJECT_CFG["source_usd_path"]
'''+s[a:];p.write_text(s)
# Preserve the original saved environment except object and private asset paths/capacity.
import sys
sys.path.insert(0,str(r/'source/viserdex'))
from viserdex.distillation.environment import load_environment
import yaml
cfg=load_environment(r/'evidence/env20_baseline.yaml')
for asset in ('robot','object'):
 cfg['scene'][asset]['spawn']['usd_path']=cfg['scene'][asset]['spawn']['usd_path'].replace('/home/zhangke/viserDex_revo3',str(r))
cfg['scene']['object']['spawn']['usd_path']=str(r/'assets/rslsphere65/RSLSphere65/RSLSphere65.usda')
cfg['sim']['physics']['solver_cfg']['nconmax']=1536
(r/'evidence/env_sphere20.yaml').write_text(yaml.dump(cfg,sort_keys=False))
# Shared Python is read-only; all project imports are selected by entry point/PYTHONPATH.
(r/'evidence/runtime.json').write_text(json.dumps({'python':'/home/zhangke/viserDex_revo3/.venv-newton/bin/python','install_modified':False}))
print('SETUP_COMPLETE',len(manifest(old)))
