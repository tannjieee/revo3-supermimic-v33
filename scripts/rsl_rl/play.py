"""Compatibility script for the official Isaac Lab 3.0 play entrypoint.

Prefer ``uv run --no-sync isaaclab play --task Isaac-ViserDex-Repose-Revo3 physics=newton_mjwarp``.
"""

import sys
from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(arg == "--task" or arg.startswith("--task=") for arg in args):
        args += ["--task", "Isaac-ViserDex-Repose-Revo3"]
    raise SystemExit(run_play_cli(args))
