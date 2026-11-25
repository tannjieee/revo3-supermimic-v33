"""Create a separate environment whose only task-contract change is hand_pose."""
from pathlib import Path
import copy,hashlib,json,sys,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.distillation.environment import load_environment,environment_contract
from viserdex.supermimic.core import config20
src=ROOT/'configs/rounded_die_wuji/env.yaml'
dst=ROOT/'configs/rounded_die_wuji_handpose/env.yaml'
assert not dst.exists(),dst
cfg=load_environment(src)
old=str(Path(cfg['scene']['object']['spawn']['usd_path']).parents[2])
def relocate(x):
    if isinstance(x,str):return x.replace(old,str(ROOT))
    if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
    if isinstance(x,tuple):return tuple(relocate(v) for v in x)
    if isinstance(x,list):return [relocate(v) for v in x]
    return x
cfg=relocate(cfg);base=copy.deepcopy(cfg);settings=config20()
assert 'hand_pose' not in cfg['rewards']
cfg['rewards']['hand_pose']={'func':'viserdex.supermimic.hand_pose:hand_pose_penalty','params':{},'weight':-settings.hand_pose_weight}
cfg['log_dir']=str(ROOT/'outputs/rounded_die_wuji_handpose')
undo=copy.deepcopy(cfg);del undo['rewards']['hand_pose']
assert environment_contract(undo)==environment_contract(base)
for name in ('robot','object'):assert Path(cfg['scene'][name]['spawn']['usd_path']).is_file()
dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(yaml.dump(cfg,sort_keys=False))
manifest=dict(task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-WujiOffset-HandPose-Hold-v1',checkpoint_version=17,hand_pose_weight=-settings.hand_pose_weight,formula='-0.2 * control_dt * sum((q_actual_post_physics - q_default)**2)',reference=list(settings.action_home),control_dt=settings.control_dt,reward_curriculum=False,initialization='scratch',parent_checkpoint=None,all_other_environment_contract_fields_unchanged=True,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),wuji_reference='https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/src/wuji_mjlab/tasks/reorient/mdp/rewards.py#L81-L92')
(dst.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
