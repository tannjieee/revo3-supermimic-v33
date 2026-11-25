"""Handle checkpoint requests only after a complete RSL-RL 5 policy update."""

from contextlib import contextmanager
from pathlib import Path
import signal


class _TrainingStopped(Exception):
    pass


@contextmanager
def checkpoint_on_signal(runner, log_dir):
    """SIGUSR1 saves and continues; SIGINT/SIGTERM save and stop at an iteration boundary.

    The signal handler only sets flags. Saving inside it could interrupt an optimizer
    update or CUDA operation and produce an inconsistent checkpoint.
    """
    pending = {"save": False, "stop": None}
    previous_handlers = {}
    original_log = runner.logger.log
    had_log_override = "log" in vars(runner.logger)

    def request(signum, _frame):
        pending["save"] = True
        if signum != signal.SIGUSR1:
            pending["stop"] = signum

    def log_and_checkpoint(*args, **kwargs):
        result = original_log(*args, **kwargs)
        if pending["save"]:
            pending["save"] = False
            iteration = runner.current_learning_iteration
            # Keep the standard name so Isaac Lab's numeric checkpoint sorting works.
            path = Path(log_dir, f"model_{iteration}.pt")
            runner.save(str(path))
            print(f"[INFO] Checkpoint saved after completed iteration {iteration}: {path}", flush=True)
            if pending["stop"] is not None:
                raise _TrainingStopped
        return result

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, request)
        runner.logger.log = log_and_checkpoint
        print("[INFO] Checkpoint control ready: SIGUSR1 saves; SIGINT/SIGTERM save and stop.", flush=True)
        try:
            yield
        except _TrainingStopped:
            runner.logger.stop_logging_writer()
            print("[INFO] Training stopped with a completed checkpoint.", flush=True)
    finally:
        if had_log_override:
            runner.logger.log = original_log
        else:
            del runner.logger.log
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
