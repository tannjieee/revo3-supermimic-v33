"""Record the audited local MJWarp scratch-capacity patch in checkpoint contracts."""
import hashlib
import json
from pathlib import Path


def capacity_manifest():
    from mujoco_warp._src import collision_convex, types
    edges = types.MJ_MAX_EPAHORIZON
    if collision_convex.MJ_MAX_EPAHORIZON != edges:
        raise ValueError('MJWarp capacity changed after import; restart the process')
    if edges == 24:
        return None  # Original runtime, including historical checkpoints.
    path = Path(__file__).resolve().parents[4] / 'config/mjwarp_capacity_patch.json'
    profile = json.loads(path.read_text())
    actual_sha = hashlib.sha256(Path(types.__file__).read_bytes()).hexdigest()
    if edges != profile['current'] or actual_sha != profile['patched_sha256']:
        raise ValueError('Unrecorded MJWarp EPA horizon modification')
    return profile
