#!/usr/bin/env bash
# Fresh V33 training: separate Actor costs, six-head Critic, target100 profile.
# Supply --output_dir and --iterations explicitly.
set -euo pipefail
launch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$launch_dir/../.." && pwd)"
replay_options=(--buffer_steps 2000)
for arg in "$@"; do
    if [[ "$arg" == --smoke ]]; then replay_options=(); fi
done
exec bash "$launch_dir/run_home_slow.sh" \
    --dex_reference_config "$repo_dir/configs/supermimic_v33/control_target100.yaml" \
    --num_envs 2048 "${replay_options[@]}" \
    --target_rate_limit_deg_s 100 "$@"
