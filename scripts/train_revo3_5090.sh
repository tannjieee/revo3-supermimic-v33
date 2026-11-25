#!/usr/bin/env bash
# Official Newton/MJWarp training; use --num_envs to override the initial batch.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/run_newton.sh" isaaclab train \
    --viz none --device cuda:0 --task Isaac-ViserDex-Repose-Revo3 \
    --num_envs 2048 --logger tensorboard physics=newton_mjwarp \
    env.commands.object_pose.debug_vis=false agent.save_interval=100 "$@"
