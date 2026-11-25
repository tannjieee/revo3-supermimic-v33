#!/usr/bin/env bash
# 3DGS keypoint estimator; the rollout teacher stays frozen.
# Pass --policy_run <teacher_run> and --checkpoint <model_file> explicitly.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VISERDEX_OBJECT=RSLCube
exec bash "$project_dir/scripts/run_newton.sh" scripts/pose_estimator/train_pose_estimator.py \
    --viz none --device cuda:0 --num_envs 512 --num_batches 16 \
    --task Isaac-ViserDex-Repose-Revo3-Estimator \
    --num_steps 6200 --seed 42 --learning_rate 1e-4 \
    --warmup_steps 500 --decay_steps 5500 --min_lr_factor 0.2 \
    --save_interval 500 --log_interval 50 --logger tensorboard \
    "$@"
