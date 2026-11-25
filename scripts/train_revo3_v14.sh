#!/usr/bin/env bash
# V14 collision geometry with the existing teacher/control configuration.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/run_newton.sh" isaaclab train \
    --viz none --device cuda:0 --task Isaac-ViserDex-Repose-Revo3-V14 \
    --num_envs 4096 --logger tensorboard physics=newton_mjwarp \
    env.sim.physics.use_cuda_graph=false \
    env.commands.object_pose.debug_vis=false agent.save_interval=100 "$@"
