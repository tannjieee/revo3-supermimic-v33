from pathlib import Path
import json,hashlib,sys,subprocess
r=Path('/home/zhangke/viserDex_supermimic_rslsphere65_20261029')
source='/home/zhangke/viserDex_supermimic_rslsphere65_20261028'
(r/'evidence/origin_revision.txt').write_text('31: '+source+' @ 550d5322fdc0b684eb746092fb2c1e3e13c44690\n')
p=r/'evidence/env_sphere20.yaml';p.write_text(p.read_text().replace(source,str(r)))
# Record shared dependency installation files; never install/edit anything here.
env=Path('/home/zhangke/viserDex_revo3/.venv-newton/lib/python3.12/site-packages')
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in env.glob('*.pth')}
protected.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in env.glob('*viserdex*.py')})
(r/'evidence/shared_install_before.json').write_text(json.dumps(protected,indent=2))
sys.path.insert(0,str(r/'source/viserdex'))
from viserdex.distillation.environment import SavedEnvironmentBinding
import viserdex
binding=SavedEnvironmentBinding(r/'evidence/env_sphere20.yaml')
for name,x in binding.assets.items():assert Path(x['path']).is_relative_to(r),(name,x)
asset=r/'assets/rslsphere65/rslsphere65_gaussians.ply'
assert hashlib.sha256(asset.read_bytes()).hexdigest()=='39c1e2c801bfa38eda4d2cba5027709a56acf6e01f47a1cab8cda3a36a93b6da'
assert Path(viserdex.__file__).is_relative_to(r)
(r/'evidence/deployment.json').write_text(json.dumps(dict(source_module=viserdex.__file__,assets=binding.assets,
 environment_contract=binding.contract_sha,shared_install_modified=False,student_pid=2483453),indent=2))
print('DEPLOYMENT_VERIFIED',viserdex.__file__,binding.contract_sha)
