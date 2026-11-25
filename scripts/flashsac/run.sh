#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source_root="$(cd -- "$task_dir/../.." && pwd)"
project_root="$(cd -- "$source_root/../../.." && pwd)"
runtime_prefix="$source_root/conda-forge/miniforge3/envs/viserdex-newton"
lab_dir="$source_root/xinyu/IsaacLab-v3.0.0"
upstream_dir="$project_root/experiments/flashsac_hand_20261105"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$task_dir/source/viserdex:$upstream_dir/deps:$upstream_dir/vendor/FlashSAC:$source_root/xinyu/revo3_walnut_rl"
for package_dir in "$lab_dir"/source/*; do
    [[ -d "$package_dir" ]] && export PYTHONPATH="$PYTHONPATH:$package_dir"
done
export VISERDEX_OBJECT=RoundedDie OMNI_KIT_ACCEPT_EULA=YES PYGLET_HEADLESS=true
export CUDA_HOME="$runtime_prefix/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$runtime_prefix/bin:$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$runtime_prefix/lib/cuda-build${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$runtime_prefix/include/python3.12:$runtime_prefix/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export WARP_CACHE_PATH="$project_root/runtime/cache/warp"
export FLASH_SAC_VENDOR="$upstream_dir/vendor/FlashSAC"
mode="${1:-help}"
[[ $# -eq 0 ]] || shift
cd -- "$task_dir"
"$runtime_prefix/bin/python" -c 'from importlib.metadata import version; expected={"isaacsim":"6.2.0.0","isaaclab":"28.0.0","newton":"1.6.1","warp-lang":"1.17.0"}; assert {k:version(k) for k in expected}==expected'
case "$mode" in
    probe) exec "$runtime_prefix/bin/python" "$task_dir/scripts/flashsac/probe.py" "$@" ;;
    test) exec "$runtime_prefix/bin/python" -m pytest -q tests/test_flashsac.py "$@" ;;
    train|play) exec "$runtime_prefix/bin/python" scripts/flashsac/cli.py "$mode" --rl_library flashsac --device cuda:0 --viz none physics=newton_mjwarp "$@" ;;
    *) echo 'Usage: scripts/flashsac/run.sh test | train --output_dir PATH | play --checkpoint PATH --output_dir PATH [--record]' ;;
esac
