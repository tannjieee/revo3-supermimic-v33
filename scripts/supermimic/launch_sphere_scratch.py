"""Start one independent two-GPU RSLSphere65 run with fresh networks/optimizers."""
import argparse,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',action='append',required=True)
a=p.parse_args()
if len(a.gpu)!=2 or len(set(a.gpu))!=2:p.error('Exactly two different idle physical GPU IDs/UUIDs required')
run=root/'outputs/revo3_rslsphere65_2x2048_scratch_seed42'
state=root/'outputs/active_rslsphere65_scratch.json'
if run.exists() or state.exists():raise FileExistsError('Fresh-run output/state already exists')
runtime='/home/zhangke/viserDex_revo3/.venv-newton/bin/python'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=','.join(a.gpu),OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
 VISERDEX_SOURCE_REVISION=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
command=['bash','scripts/run_sphere.sh','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
 'scripts/supermimic/train.py','--distributed','--num_envs','2048','--viz','none','--seed','42',
 '--env_config',str(root/'evidence/env_sphere20.yaml'),'--output_dir',str(run),
 '--iterations','320000','--save_interval','2000','--latest_interval','100',
 '--amp','bf16','--wm_microbatch','2048','--actor_microbatch','2048','--critic_microbatch','32768',
 '--audit_interval','200','--holdout','env.sim.physics.solver_cfg.nconmax=1536']
guard=[runtime,'scripts/supermimic/run_guard.py']
for gpu in a.gpu:guard+=['--gpu',gpu]
guard+=['--run_dir',str(run),'--state_file',str(state),'--console',str(run.with_suffix('.log')),'--',*command]
(root/'evidence').mkdir(exist_ok=True)
with (root/'evidence/sphere_scratch_guard.log').open('x') as log:
 proc=subprocess.Popen(guard,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record=dict(guard_pid=proc.pid,run_dir=str(run),state_file=str(state),command=command,
 initialization='fresh networks, normalizers, optimizers, replay and counters',parent_checkpoint=None,
 world_size=2,num_envs_per_gpu=2048,total_num_envs=4096,control_hz=20,wm_window=8,actor_horizon=8,
 wm_only_iters=500,actor_ramp_iters=1000,iterations=320000,nconmax=1536,njmax=512,gpus=a.gpu)
(root/'evidence/sphere_scratch_launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
