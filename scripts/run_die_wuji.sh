#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VISERDEX_OBJECT=RoundedDie
export VISERDEX_PYTHON="${VISERDEX_PYTHON:-$root/.venv-newton/bin/python}"
export UV_BIN="${UV_BIN:-$(command -v uv)}"
export WARP_CACHE_PATH="$root/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$root/.cache/torchinductor"
export PYTHONDONTWRITEBYTECODE=1
exec bash "$root/scripts/run_newton.sh" "$@"
