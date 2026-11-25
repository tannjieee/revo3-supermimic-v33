#!/usr/bin/env bash
# Run inside the isolated, officially locked Isaac Lab v3.0.0 environment.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lab_dir="${ISAACLAB_PATH:-$project_dir/../IsaacLab-v3.0.0}"
runtime_python="${VISERDEX_PYTHON:-$project_dir/.venv-newton/bin/python}"
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" || ! -x "$runtime_python" || $# -eq 0 ]]; then
    echo "Usage: UV_BIN=/path/to/uv bash scripts/run_newton.sh <script.py|isaaclab> [args...]" >&2
    echo "Prepare .venv-newton as described in docs/newton_migration.md." >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$runtime_python"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-6}"
export UV_PROJECT_ENVIRONMENT="$(dirname -- "$(dirname -- "$runtime_python")")"
export UV_PYTHON="$runtime_python"
export UV_PYTHON_PREFERENCE=only-system
export UV_PYTHON_DOWNLOADS=never
export PYTHONPATH="$project_dir/source/viserdex"
export PYTHONDONTWRITEBYTECODE=1
runtime_prefix="$(dirname -- "$(dirname -- "$runtime_python")")"
export VIRTUAL_ENV="$runtime_prefix"
cuda_prefix="$runtime_prefix/lib/python3.12/site-packages/nvidia/cu13"
if [[ -z "${CUDA_HOME:-}" && -x "$cuda_prefix/bin/nvcc" ]]; then
    export CUDA_HOME="$cuda_prefix"
fi
if [[ -d "$runtime_prefix/lib/cuda-build" ]]; then
    export LIBRARY_PATH="$runtime_prefix/lib/cuda-build${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -f "$runtime_prefix/include/python3.12/Python.h" ]]; then
    export CPATH="$runtime_prefix/include/python3.12:$runtime_prefix/include${CPATH:+:$CPATH}"
fi
"$runtime_python" -c 'from importlib.metadata import version; expected={"isaacsim":"6.2.0.0","isaaclab":"28.0.0","newton":"1.6.1","warp-lang":"1.17.0"}; actual={k:version(k) for k in expected}; assert actual==expected, actual'
cd -- "$project_dir"
if [[ "$1" == "isaaclab" ]]; then
    exec "$uv_bin" run --project "$lab_dir" --no-sync "$@"
fi
exec "$uv_bin" run --project "$lab_dir" --no-sync python -u "$@"
