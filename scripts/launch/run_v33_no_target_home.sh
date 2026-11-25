#!/usr/bin/env bash
# Fresh V33 ablation baseline: hold_target_motion_weight=0, home_pose_weight=0, video disabled.
# Supply --output_dir and --iterations explicitly.
set -euo pipefail
launch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$launch_dir/../.." && pwd)"
exec bash "$launch_dir/run_v33.sh" \
    --dex_reference_config "$repo_dir/configs/supermimic_v33/control_no_target_home.yaml" \
    --video_interval 0 "$@"
