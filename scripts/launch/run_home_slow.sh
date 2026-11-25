#!/usr/bin/env bash
# Fresh V33 base training profile (die58, 25 Hz, quiet-hold, CUDA replay + WM CUDA Graph).
# Supply --output_dir and --iterations explicitly.
set -euo pipefail
has_output=false
has_iterations=false
for arg in "$@"; do
    case "$arg" in
        --output_dir|--output_dir=*) has_output=true ;;
        --iterations|--iterations=*) has_iterations=true ;;
    esac
done
if ! "$has_output" || ! "$has_iterations"; then
    echo 'Supply --output_dir PATH and --iterations COUNT for a fresh run.' >&2
    exit 2
fi
launch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$launch_dir/../.." && pwd)"
exec bash "$launch_dir/run.sh" train \
    --dex_reference_config "$repo_dir/configs/supermimic_v33/control_home_slow.yaml" \
    --object_variant die58 --quiet_hold --episode_length_s 10 --no-track_object_position \
    --joint_speed_limit_deg_s 90 --target_rate_limit_deg_s 60 \
    --joint_velocity_norm_floor_rad_s 1.5707963267948966 \
    --seed 42 --wm_only_iterations 0 --warmup_joint_amplitude_deg 0 \
    --adaptive_replay --replay_storage cuda --wm_cuda_graph --amp none \
    --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 8192 \
    --save_interval 100 --latest_interval 50 --audit_interval 200 \
    --video_interval 500 --video_steps 500 --video_seed 43 "$@"
