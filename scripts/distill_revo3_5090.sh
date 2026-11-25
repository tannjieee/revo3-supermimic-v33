#!/usr/bin/env bash
# Select a frozen teacher checkpoint explicitly; defaults preserve Appendix B-B settings.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/run_newton.sh" scripts/rsl_rl/distill.py \
    --viz none --device cuda:0 --num_envs 256 \
    "$@"
