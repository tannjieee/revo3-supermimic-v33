from pathlib import Path
import json,sys
r=Path('/home/zhangke/viserDex_supermimic_rslsphere65_20261028');sys.path.insert(0,str(r/'source/viserdex'))
from viserdex.distillation.environment import load_environment,environment_contract
old=environment_contract(load_environment(r/'evidence/env20_baseline.yaml'))
new=environment_contract(load_environment(r/'artifacts/final_preview_v2/env.yaml'))
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  return [v for k in sorted(set(a)|set(b)) for v in diff(a.get(k),b.get(k),path+'.'+k)]
 return [] if a==b else [dict(field=path,old=a,new=b)]
changes=diff(old,new)
assert {d['field'] for d in changes}=={'.scene.robot.spawn.usd_path','.scene.object.spawn.usd_path'},changes
(r/'evidence/semantic_diff.json').write_text(json.dumps(dict(passed=True,changes=changes,meaning='Only private robot path and sphere asset path differ in the normalized environment contract'),indent=2))
print('SEMANTIC_DIFF_PASSED',len(changes))
