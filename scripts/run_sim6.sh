#!/usr/bin/env bash
# Compatibility filename; the active runtime is the official Isaac Lab v3.0.0 stack.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/scripts/run_newton.sh" "$@"
