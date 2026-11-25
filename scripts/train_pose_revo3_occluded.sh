#!/usr/bin/env bash
# Reproducible collection -> depth/label audit -> offline pose training.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash scripts/train_pose_revo3_occluded.sh <teacher_checkpoint> [run_name]"
    exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/train_pose_revo3_occluded.sh <teacher_checkpoint> [run_name]" >&2
    exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export VISERDEX_OBJECT=RSLCube
teacher="$1"
run_name="${2:-revo3_occluded_$(date +%Y%m%d_%H%M%S)}"
if [[ ! -f "$teacher" ]]; then
    echo "Teacher checkpoint not found: $teacher" >&2
    exit 2
fi
if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Run name must contain only letters, digits, underscores, dots and hyphens." >&2
    exit 2
fi
train_dir="logs/occlusion_datasets/${run_name}_train"
val_dir="logs/occlusion_datasets/${run_name}_val"
for split in train val; do
    if [[ "$split" == train ]]; then
        directory="$train_dir"; seed=42; samples=300000; environments=256
    else
        directory="$val_dir"; seed=43; samples=30000; environments=128
    fi
    bash scripts/run_newton.sh scripts/pose_estimator/collect_revo3_occlusion.py \
        --viz none --device cuda:0 --teacher_checkpoint "$teacher" --output_dir "$directory" \
        --num_envs "$environments" --samples "$samples" --seed "$seed" \
        --burn_in 120 --stride 4 --max_steps 20000 --min_occlusion .05
    bash scripts/run_newton.sh scripts/pose_estimator/audit_occlusion_dataset.py "$directory"
done
exec bash scripts/run_newton.sh scripts/pose_estimator/train_pose_dataset.py \
    --train_dir "$train_dir" --val_dir "$val_dir" \
    --output_dir "logs/rsl_rl/pose_estimator/$run_name" \
    --epochs 10 --batch_size 64 --learning_rate 1e-4 --warmup_steps 1000 --save_interval 500
