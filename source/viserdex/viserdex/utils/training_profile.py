"""Optional wall-time measurements without changing the simulation or GPU synchronization."""

from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import resource
import time


@contextmanager
def profile_training(runner, log_dir):
    """Measure host call durations; nested timings are inclusive, not GPU kernel timings."""
    env = runner.env.unwrapped
    targets = {
        "environment_step": (env, "step"),
        "physics_step": (env.sim, "step"),
        "scene_write": (env.scene, "write_data_to_sim"),
        "scene_update": (env.scene, "update"),
        "reset": (env, "_reset_idx"),
        "observations": (env.observation_manager, "compute"),
        "ppo_update": (runner.alg, "update"),
    }
    timings = {name: {"calls": 0, "seconds": 0.0} for name in targets}
    originals = []

    def timed(function, stats):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stats["seconds"] += time.perf_counter() - start
                stats["calls"] += 1

        return wrapper

    start = time.perf_counter()
    try:
        for name, (obj, attr) in targets.items():
            original = getattr(obj, attr)
            originals.append((obj, attr, original, attr in vars(obj)))
            setattr(obj, attr, timed(original, timings[name]))
        yield
    finally:
        for obj, attr, original, was_instance_attr in reversed(originals):
            if was_instance_attr:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)
        output = {
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "num_envs": env.num_envs,
            "note": "Inclusive host wall times, without extra CUDA synchronizations; not GPU kernel timings.",
            "timings": timings,
            "settings": {
                key: env.sim.get_setting(key)
                for key in (
                    "/persistent/physics/numThreads",
                    "/physics/fabricUpdateTransformations",
                    "/physics/fabricUpdateVelocities",
                    "/physics/fabricUseGPUInterop",
                    "/physics/suppressReadback",
                )
            },
        }
        Path(log_dir, "profile.json").write_text(json.dumps(output, indent=2) + "\n")
