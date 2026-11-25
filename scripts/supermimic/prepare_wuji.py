"""Bind a new task to its own assets and validated Revo3 grasp center."""
from pathlib import Path
import sys,json,copy,hashlib,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.distillation.environment import load_environment,environment_contract
from viserdex.supermimic.wuji_pose import HOME_POSITION,OFFSET_SCALE,WARMUP_SECONDS
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
src=ROOT/'configs/rounded_die_hold_penalty2x/env.yaml';dst=ROOT/'configs/rounded_die_wuji/env.yaml'
assert not dst.exists(),dst
cfg=load_environment(src);old=str(Path(cfg['scene']['object']['spawn']['usd_path']).parents[2])
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(ROOT))
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 if isinstance(x,tuple):return tuple(relocate(v) for v in x)
 if isinstance(x,list):return [relocate(v) for v in x]
 return x
cfg=relocate(cfg);base=copy.deepcopy(cfg)
cfg['actions']['joint_pos'].update(class_type='viserdex.supermimic.wuji_actions:WujiHomeOffsetAction',rescale_to_limits=False,offset_scale=OFFSET_SCALE,warmup_time_s=WARMUP_SECONDS)
cfg['scene']['robot']['init_state']['joint_pos']=dict(zip(REVO3_JOINT_NAMES,HOME_POSITION))
cfg['log_dir']=str(ROOT/'outputs/rounded_die_wuji')
undo=copy.deepcopy(cfg);undo['actions']=base['actions'];undo['scene']['robot']['init_state']=base['scene']['robot']['init_state'];assert environment_contract(undo)==environment_contract(base)
for asset in ('robot','object'):assert Path(cfg['scene'][asset]['spawn']['usd_path']).is_file()
dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(yaml.dump(cfg,sort_keys=False))
manifest=dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-WujiOffset-Hold-v1',checkpoint_version=16,action_home=dict(zip(REVO3_JOINT_NAMES,HOME_POSITION)),offset_scale=OFFSET_SCALE,ema_alpha=.5,warmup_seconds=WARMUP_SECONDS,control_hz=20,nominal_kp=3.,nominal_kd=.22,rewards_unchanged=True,randomization_unchanged=True,object_spawn_unchanged=True,parent_checkpoint=None,initialization='scratch',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
(dst.parent/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
