#!/usr/bin/env bash
# Reuse a CUDA PyTorch environment through a venv overlay, or install a standalone environment.
set -euo pipefail
export PYTHONNOUSERSITE=1
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${VISERDEX_SCAN_BASE_PYTHON:-python3}"
capture_only=false
if [[ "${1:-}" == "--capture-only" ]]; then
    capture_only=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: VISERDEX_SCAN_BASE_PYTHON=/path/to/python bash scripts/setup_scan.sh [--capture-only]" >&2
    exit 2
fi
if [[ ! -d "$project_dir/.venv-scan" ]]; then
    if [[ -n "${VISERDEX_SCAN_BASE_PYTHON:-}" ]]; then
        "$base_python" -m venv --system-site-packages "$project_dir/.venv-scan"
    else
        "$base_python" -m venv "$project_dir/.venv-scan"
    fi
fi
scan_python="$project_dir/.venv-scan/bin/python"
if $capture_only; then
    "$scan_python" -m pip install --no-cache-dir -r "$project_dir/requirements/scan-capture.txt"
else
    if ! "$scan_python" -c 'import torch; assert torch.version.cuda'; then
        "$scan_python" -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
    fi
    "$scan_python" -m pip install --no-cache-dir -r "$project_dir/requirements/scan.txt"
fi
mkdir -p "$project_dir/.runtime/scanning"
"$scan_python" -m pip freeze > "$project_dir/.runtime/scanning/environment.txt"
echo "Ready: bash scripts/scan_object.sh doctor --camera"
