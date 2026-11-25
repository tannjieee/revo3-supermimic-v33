"""Launch isolated WujiOffset-HandPose-Hold-v1 from scratch on two explicitly selected idle GPUs."""
import argparse
import os
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', action='append', required=True)
p.add_argument('--iterations', type=int, default=320000)
p.add_argument('--start-training', action='store_true', help='Explicitly launch this new production run')
a = p.parse_args()
if not a.start_training: p.error('Training requires explicit --start-training')
if len(a.gpu) != 2 or len(set(a.gpu)) != 2: p.error('Select two distinct physical GPU UUIDs')
if a.iterations < 1: p.error('iterations must be positive')
run = ROOT/'outputs/yellow_die_handpose_2x2048_scratch_seed42'
state = ROOT/'outputs/active_yellow_die_handpose.json'
console = ROOT/'outputs/yellow_die_handpose_console.log'
for path in (run, state, console):
    if path.exists(): raise FileExistsError(path)
if not (ROOT/'configs/yellow_die_handpose/env.yaml').is_file(): raise FileNotFoundError('Run prepare_yellow_die.py first')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=','.join(a.gpu), ISAACLAB_PATH='/home/zhangke/IsaacLab-v3.0.0', VISERDEX_PYTHON='/home/zhangke/viserDex_revo3/.venv-newton/bin/python', UV_BIN='/home/zhangke/isaaclab-ga-upgrade-20261020/bin/uv', PYTHONDONTWRITEBYTECODE='1')
env["VISERDEX_SOURCE_REVISION"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
# Guard launches and monitors only this new run; it refuses occupied GPUs.
command = [sys.executable, str(ROOT/'scripts/supermimic/run_guard.py')]
for gpu in a.gpu: command.extend(['--gpu', gpu])
command.extend(['--run_dir', str(run), '--state_file', str(state), '--console', str(console), '--', 'bash', str(ROOT/'scripts/run_yellow_die.sh'), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2', str(ROOT/'scripts/supermimic/train.py'), '--distributed', '--num_envs', '2048', '--viz', 'none', '--env_config', str(ROOT/'configs/yellow_die_handpose/env.yaml'), '--output_dir', str(run), '--iterations', str(a.iterations), '--seed', '42', '--amp', 'bf16', '--wm_microbatch', '2048', '--actor_microbatch', '2048', '--critic_microbatch', '32768', '--latest_interval', '100', '--save_interval', '2000', 'env.sim.physics.solver_cfg.nconmax=1536'])
(ROOT/'outputs').mkdir(exist_ok=True)
with (ROOT/'outputs/yellow_die_handpose_guard.log').open('x') as log:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(guard_pid=proc.pid, task_id='Isaac-ViserDex-Repose-Revo3-SuperMimic-YellowDie-WujiOffset-HandPose-Hold-v1', run_dir=str(run), state_file=str(state), initialization='scratch', original_weights_loaded=False), indent=2))
