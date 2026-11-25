"""Launch one isolated single-GPU sphere job behind the existing occupancy guard."""
import argparse,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase',choices=['contacts','capacity','train'],required=True)
p.add_argument('--gpu',required=True)
a=p.parse_args()
names={'contacts':'rslsphere65_contact2048_20261029','capacity':'rslsphere65_capacity2048_20261029','train':'rslsphere65_single2048_scratch_seed42_20261029'}
run=root/'outputs'/names[a.phase]
state=root/'outputs'/('active_'+names[a.phase]+'.json')
if run.exists() or state.exists():raise FileExistsError(run)
if a.phase=='train':
 contact=json.loads((root/'outputs'/names['contacts']/'stress_rank_0.json').read_text())
 capacity=json.loads((root/'outputs'/names['capacity']/'validation.json').read_text())
 assert contact['passed'] and capacity['optimizer_steps']=={'wm':20,'actor':2,'critic':40}
 assert capacity['max_errors']['success_boundary_disagreements']==0
 assert capacity['max_errors']['object_collision_scale_m']==0
common=['--num_envs','2048','--viz','none','--device','cuda:0','--seed','42','--env_config',str(root/'evidence/env_sphere20.yaml'),'--output_dir',str(run)]
if a.phase=='contacts':
 command=['bash','scripts/run_sphere.sh','scripts/supermimic/stress_contacts.py',*common,'--iterations','1']
elif a.phase=='capacity':
 command=['bash','scripts/run_sphere.sh','scripts/supermimic/profile_in_sim.py',*common,'--iterations','2','--check_joint_update','--audit','--holdout','--profile_wm_chunk','2048','--profile_actor_chunk','2048','--profile_critic_chunk','32768']
else:
 command=['bash','scripts/run_sphere.sh','scripts/supermimic/train.py',*common,'--iterations','320000','--save_interval','2000','--latest_interval','100','--amp','bf16','--wm_microbatch','2048','--actor_microbatch','2048','--critic_microbatch','32768','--audit_interval','200']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',VISERDEX_SOURCE_REVISION=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
guard=['/home/zhangke/viserDex_revo3/.venv-newton/bin/python','scripts/supermimic/run_guard.py','--gpu',a.gpu,'--allowed_idle_viewer_pid','1900441','--run_dir',str(run),'--state_file',str(state),'--console',str(run.with_suffix('.log')),'--',*command]
(root/'evidence').mkdir(exist_ok=True)
with (root/'evidence'/('launch_'+a.phase+'_20261029.log')).open('x') as log:
 proc=subprocess.Popen(guard,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(phase=a.phase,guard_pid=proc.pid,physical_gpu=a.gpu,run_dir=str(run),state_file=str(state),command=command,world_size=1,num_envs=2048,control_hz=20,initialization='fresh networks and optimizers; no checkpoint loaded',formal_training=a.phase=='train')
(root/'evidence'/('launch_'+a.phase+'_20261029.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
