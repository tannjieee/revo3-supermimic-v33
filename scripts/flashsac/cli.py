"""Project-local FlashSAC backend for the pinned unified Isaac Lab dispatcher.

v3's backend table is closed; extend it in this process only, never on disk.
The official argument dispatcher, task registry and launch_simulation are used.
"""
import sys
from isaaclab_rl.entrypoints import dispatch

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'play'):
        raise SystemExit('Usage: cli.py train|play --rl_library flashsac [arguments]')
    action, argv = sys.argv[1], sys.argv[2:]
    dispatch._BACKEND_MODULES[action]['flashsac'] = 'viserdex.flashsac.runner'
    if action == 'play':
        argv += ['--evaluate']
    raise SystemExit(dispatch.run_cli(action, argv))
