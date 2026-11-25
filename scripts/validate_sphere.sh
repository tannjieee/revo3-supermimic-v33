#!/usr/bin/env bash
# Exactly one diagnostic update of all three fresh networks; never a training launch.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-$root/artifacts/full_update_check}"
exec bash "$root/scripts/run_sphere.sh" scripts/supermimic/profile_in_sim.py \
 --profile_wm_chunk 512 --profile_actor_chunk 512 --profile_critic_chunk 4096 \
 --check_joint_update --iterations 1 --num_envs 16 \
 --env_config "$root/evidence/env_sphere20.yaml" --output_dir "$output_dir" \
 --viz none --device cuda:0 --eager_geometry --audit --holdout
