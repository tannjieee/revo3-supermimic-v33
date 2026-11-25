#!/usr/bin/env bash
# Pinned Isaac Lab / Newton environment launcher for Revo3 SuperMimic V33.
# Override ISAACLAB_DIR and VISERDEX_ENV_PREFIX to match your local installation.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
parent_workspace="$(cd -- "$repo_dir/.." && pwd)"

default_source_root="$parent_workspace/source_home/brainco1/workspace"
lab_dir="${ISAACLAB_DIR:-$default_source_root/xinyu/IsaacLab-v3.0.0}"
runtime_prefix="${VISERDEX_ENV_PREFIX:-$default_source_root/conda-forge/miniforge3/envs/viserdex-newton}"
runtime_python="${VISERDEX_PYTHON:-$runtime_prefix/bin/python}"
uv_bin="${UV_BIN:-$(command -v uv)}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_dir/source/viserdex"
for package_dir in "$lab_dir"/source/*; do
    [[ -d "$package_dir" ]] && export PYTHONPATH="$PYTHONPATH:$package_dir"
done
export VISERDEX_OBJECT="${VISERDEX_OBJECT:-RoundedDie}"
export UV_PROJECT_ENVIRONMENT="$runtime_prefix" UV_PYTHON="$runtime_python"
export UV_PYTHON_PREFERENCE=only-system UV_PYTHON_DOWNLOADS=never VIRTUAL_ENV="$runtime_prefix"
if [[ -d "$runtime_prefix/lib/python3.12/site-packages/nvidia/cu13" ]]; then
    export CUDA_HOME="$runtime_prefix/lib/python3.12/site-packages/nvidia/cu13"
    export PATH="$runtime_prefix/bin:$CUDA_HOME/bin:$PATH"
else
    export PATH="$runtime_prefix/bin:$PATH"
fi
export LIBRARY_PATH="$runtime_prefix/lib/cuda-build${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$runtime_prefix/include/python3.12:$runtime_prefix/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MAX_JOBS="${MAX_JOBS:-4}"
export OMNI_KIT_ACCEPT_EULA=YES PYGLET_HEADLESS=true

mode="${1:-help}"
[[ $# -eq 0 ]] || shift
cd -- "$repo_dir"

manual_usd="$repo_dir/assets/revo3_d405/vendor/brainco-description/revo3_system/optimized/right_20261027/revo3_right.usd"

case "$mode" in
    prepare)
        exec "$uv_bin" run --project "$lab_dir" --no-sync python scripts/preprocessing/prepare_revo3_newton.py \
            --manual_convex_usd "$manual_usd" \
            --usd_dir "$repo_dir/generated/revo3_manual_convex" --viz none "$@" ;;
    train)
        exec "$uv_bin" run --project "$lab_dir" --no-sync python scripts/supermimic/train.py \
            --policy_variant manual_convex --network_revision hardware_center25_v3 \
            --output_dir "$repo_dir/runs/$(date +%Y%m%d_%H%M%S)" \
            --num_envs 1024 --device cuda:0 --viz none physics=newton_mjwarp "$@" ;;
    check_fixed_reset)
        exec "$uv_bin" run --project "$lab_dir" --no-sync python scripts/supermimic/check_fixed_palm_reset.py \
            --device cuda:0 --viz none physics=newton_mjwarp "$@" ;;
    play)
        exec "$uv_bin" run --project "$lab_dir" --no-sync python scripts/supermimic/record_manual.py \
            --num_envs 1 --device cuda:0 --viz newton_gl physics=newton_mjwarp "$@" ;;
    export)
        exec "$runtime_python" scripts/supermimic/export_direct_actor.py "$@" ;;
    test)
        exec "$runtime_python" -m pytest tests/test_dex_reference_v33.py tests/test_home_slow.py tests/test_quiet_hold.py tests/test_position_boundary.py tests/test_paper_critic.py "$@" ;;
    *)
        echo 'Usage: scripts/launch/run.sh prepare | train [args] | check_fixed_reset [args] | play [args] | export [args] | test'
        [[ "$mode" == help ]] ;;
esac
