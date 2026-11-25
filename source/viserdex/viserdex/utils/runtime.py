"""Official configuration-first runtime for ViserDex's custom rollout scripts.

Teacher training/playback use Isaac Lab's unified entrypoints directly. Custom
DAgger and image collection loops share the same preset resolution and launcher.
"""

import atexit
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import parse_env_cfg, setup_preset_cli


def add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3"):
    if "--task" not in parser._option_string_actions:
        parser.add_argument("--task", default=default_task)
    add_launcher_args(parser)


def parse_runtime_args(parser):
    import viserdex.tasks  # noqa: F401

    args, overrides = setup_preset_cli(parser)
    args.preset_overrides = list(overrides)
    sys.argv = [sys.argv[0], *overrides]
    return args


class SimulationRuntime:
    """Own the official launch context while preserving post-launch imports."""

    def __init__(self, args, configure=None):
        self.args = args
        self.cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=getattr(args, "num_envs", None),
            overrides=args.preset_overrides,
        )
        if configure is not None:
            configure(self.cfg)
        self._context = launch_simulation(self.cfg, args)
        self._context.__enter__()
        self._closed = False
        atexit.register(self.close)

    def env_cfg(self, task_name=None, **kwargs):
        if task_name is None or task_name == self.args.task:
            cfg = self.cfg
            if kwargs.get("device") is not None:
                cfg.sim.device = kwargs["device"]
            if kwargs.get("num_envs") is not None:
                cfg.scene.num_envs = kwargs["num_envs"]
            if kwargs.get("use_fabric") is not None:
                cfg.sim.use_fabric = kwargs["use_fabric"]
            return cfg
        return parse_env_cfg(task_name, overrides=self.args.preset_overrides, **kwargs)

    def is_running(self):
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        return sim is None or sim.is_running()

    def close(self):
        if not self._closed:
            self._closed = True
            atexit.unregister(self.close)
            self._context.__exit__(*sys.exc_info())
