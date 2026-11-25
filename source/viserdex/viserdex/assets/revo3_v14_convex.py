"""Opt-in selected V14 convex asset; keeps the training asset unchanged."""
import json
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg

from .revo3 import REVO3_HAND_CFG


def make_v14_convex_cfg(generated_dir: str | Path | None = None):
    """Load the official multi-physics USD and the selected source's SysID drives."""
    root = Path(generated_dir) if generated_dir else Path(__file__).parent / "generated"
    manifest = json.loads((root / "revo3_v14_touch_source/source_manifest.json").read_text())
    assert manifest["collisions"] == 93 and manifest["sdf"] is False
    assert manifest["selected_asset"]["adapter"] == "v14_mcp_touch_restored_20261013"
    usd = root / "revo3_v14_touch_newton/revo3_v14_touch_convex/revo3_v14_touch_convex.usda"
    if not usd.is_file():
        raise FileNotFoundError(usd)
    cfg = REVO3_HAND_CFG.copy()
    cfg.spawn.usd_path = str(usd)
    drives = manifest["drives"]
    cfg.actuators = {"fingers": ImplicitActuatorCfg(
        joint_names_expr=list(drives),
        # Source URDF speed limits are zero placeholders; inspection uses 10 rad/s.
        joint_velocity_limit=10.0,
        joint_effort_limit={n: d["effort_limit"] for n, d in drives.items()},
        stiffness={n: d["stiffness"] for n, d in drives.items()},
        damping={n: d["damping"] for n, d in drives.items()},
        armature={n: d["armature"] for n, d in drives.items()},
        friction={n: d["friction"] for n, d in drives.items()},
    )}
    cfg.init_state.joint_pos = {".*": 0.0}
    return cfg
