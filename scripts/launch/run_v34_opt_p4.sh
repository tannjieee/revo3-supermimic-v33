#!/usr/bin/env bash
# Fresh V36 Phase 4 (Latest revo3-sysid Fresh-50Hz + Stability & Sample-Efficiency) training for Revo3 (25 Hz, die58):
# Latest 2026-11-24 21-DOF closed-loop sysid (0.338 deg RMSE) + WM Huber/ResMLP/AdamW +
# Twin Conservative Critic + Polyak EMA Target Critic + Critic Huber +
# cosine exploration annealing (2.5 -> 0.5 deg) + scoped 30k cosine LR (floor 0.15) +
# all V35 Phase 3 fast-convergence upgrades.
# Supply --output_dir and --iterations explicitly.
set -euo pipefail
has_output=false
has_iterations=false
is_smoke=false
for arg in "$@"; do
    case "$arg" in
        --output_dir|--output_dir=*) has_output=true ;;
        --iterations|--iterations=*) has_iterations=true ;;
        --smoke) is_smoke=true ;;
    esac
done
if ! "$has_output" || ! "$has_iterations"; then
    echo 'Supply --output_dir PATH and --iterations COUNT for a fresh V36 Phase 4 run.' >&2
    exit 2
fi
launch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$launch_dir/../.." && pwd)"
scale_opts=(--num_envs 2048 --buffer_steps 600 --wm_only_iterations 150 --warmup_joint_amplitude_deg 6.0)
if "$is_smoke"; then
    scale_opts=(--num_envs 8 --wm_only_iterations 0 --warmup_joint_amplitude_deg 0 --wm_microbatch 32 --actor_microbatch 32 --critic_microbatch 32)
fi
exec bash "$launch_dir/run.sh" train \
    --dex_reference_config "$repo_dir/configs/supermimic_v34/control_opt_p4.yaml" \
    --object_variant die58 --quiet_hold --episode_length_s 20 --no-track_object_position \
    --hold_duration 0.40 \
    --gamma 0.98 --lam 0.9025 --horizon 8 \
    --actor_cost_weight 0.55 \
    --stability_reward_scale 4.5 --stability_angle_sigma_deg 9.0 \
    --joint_speed_limit_deg_s 90 --target_rate_limit_deg_s 160 \
    --joint_velocity_norm_floor_rad_s 1.5707963267948966 \
    --seed 42 --replay_storage cuda --wm_cuda_graph --amp bf16 \
    --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 8192 \
    --save_interval 100 --latest_interval 50 --audit_interval 200 \
    --video_interval 0 "${scale_opts[@]}" "$@"
