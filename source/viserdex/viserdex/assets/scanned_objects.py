"""Discover generated objects without editing the built-in object catalogue."""

import json
from pathlib import Path
import re


def load_scanned_objects(directory, reserved_names=()):
    result = {}
    for manifest in sorted(Path(directory).glob("*/object.json")):
        entry = json.loads(manifest.read_text())
        name = entry.get("name", "")
        if entry.get("schema_version") != 1 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"Invalid scanned object manifest: {manifest}")
        if name in reserved_names or name in result:
            raise ValueError(f"Scanned object {name!r} would shadow an existing object")
        resolved = {}
        for key in ("usd_path", "mesh_path", "splat_path", "points_path"):
            path = (manifest.parent / entry[key]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
                raise ValueError(f"Missing or external {key} in {manifest}")
            resolved[key] = str(path)
        resolved.update(
            supports_pose_estimation=bool(entry.get("supports_pose_estimation", False)),
            rollout_policy_run_name=entry.get("rollout_policy_run_name"),
            revo3_rollout_policy_run_name=entry.get("revo3_rollout_policy_run_name"),
        )
        result[name] = resolved
    return result
