#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scan_python="${VISERDEX_SCAN_PYTHON:-$project_dir/.venv-scan/bin/python}"
if [[ ! -x "$scan_python" ]]; then
    echo "Run bash scripts/setup_scan.sh, or set VISERDEX_SCAN_PYTHON." >&2
    exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PYTHONNOUSERSITE=1
export PATH="$(dirname -- "$scan_python"):$PATH"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
# Let PyTorch detect the active GPU unless the operator deliberately specifies an architecture.
exec "$scan_python" "$project_dir/scripts/scan_object.py" "$@"
