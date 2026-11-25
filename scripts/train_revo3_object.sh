#!/usr/bin/env bash
# One independent object policy per GPU, using the 8192-environment settings.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/train_revo3_object.sh <object> <gpu-index> [training arguments...]" >&2
    exit 2
fi
object_name="$1"
gpu_index="$2"
shift 2
case "$object_name" in
    RSLCube|Duck|DuckRevo3Scaled|3DPrintedToy|TabletBottle|Globe) ;;
    *) echo "Unknown object: $object_name" >&2; exit 2 ;;
esac
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo "GPU index must be an integer" >&2; exit 2; }
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VISERDEX_OBJECT="$object_name"
exec bash "$project_dir/scripts/train_revo3_5090.sh" \
    --device "cuda:$gpu_index" \
    --run_name "${object_name}_gpu${gpu_index}_$(date +%Y%m%d_%H%M%S)" "$@"
