"""Start one two-GPU Ramp20 model with fresh networks and optimizers."""
import argparse
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', action='append', required=True)
parser.add_argument('--allowed_idle_viewer_pid', type=int, action='append', default=[])
args = parser.parse_args()
assert len(set(args.gpu)) == 2
run = root/'outputs/revo3_ramp20_2x2048_scratch_seed42'
state = root/'outputs/active_ramp20_scratch.json'
assert not run.exists() and not state.exists()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=','.join(args.gpu), OMP_NUM_THREADS='4',
           UV_BIN='/home/zhangke/isaaclab-ga-upgrade-20261020/bin/uv',
           VISERDEX_SOURCE_REVISION=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip())
command = ['bash','scripts/run_newton.sh','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
           'scripts/supermimic/train.py','--distributed','--num_envs','2048','--viz','none','--seed','42',
           '--env_config',str(root/'evidence/env20.yaml'),'--output_dir',str(run),
           '--iterations','320000','--save_interval','2000','--latest_interval','100',
           '--amp','bf16','--wm_microbatch','2048','--actor_microbatch','2048',
           '--critic_microbatch','32768','--audit_interval','200','--holdout',
           'env.sim.physics.solver_cfg.nconmax=1536']
guard = [str(root/'.venv-newton/bin/python'),'scripts/supermimic/run_guard.py']
for gpu in args.gpu:
    guard += ['--gpu',gpu]
for pid in args.allowed_idle_viewer_pid:
    guard += ['--allowed_idle_viewer_pid',str(pid)]
guard += ['--run_dir',str(run),'--state_file',str(state),'--console',str(run.with_suffix('.log')),'--',*command]
with (root/'evidence/ramp_scratch_guard.log').open('x') as log:
    proc = subprocess.Popen(guard, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
record = dict(guard_pid=proc.pid, run_dir=str(run), state_file=str(state), command=command,
              initialization='fresh networks, normalizers, optimizers, replay and counters',
              parent_checkpoint=None, world_size=2, total_num_envs=4096, control_hz=20)
(root/'evidence/ramp_scratch_launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
