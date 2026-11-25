"""Signals must never snapshot a partially updated optimizer."""

import signal
from types import SimpleNamespace

import pytest

from viserdex.utils.training_control import checkpoint_on_signal


class Logger:
    def __init__(self):
        self.closed = False
        self.logged = []

    def log(self, **kwargs):
        self.logged.append(kwargs)

    def stop_logging_writer(self):
        self.closed = True


def make_runner():
    snapshots = []
    runner = SimpleNamespace(logger=Logger(), current_learning_iteration=0)
    runner.save = lambda path: snapshots.append((path, runner.current_learning_iteration))
    return runner, snapshots


def test_save_is_deferred_until_completed_update(tmp_path):
    runner, snapshots = make_runner()
    previous = signal.getsignal(signal.SIGUSR1)
    with checkpoint_on_signal(runner, tmp_path):
        signal.raise_signal(signal.SIGUSR1)
        assert snapshots == []
        runner.current_learning_iteration = 7
        runner.logger.log(it=7)
        runner.logger.log(it=8)
        assert snapshots == [(str(tmp_path / "model_7.pt"), 7)]
    assert signal.getsignal(signal.SIGUSR1) == previous
    assert "log" not in vars(runner.logger)
    assert not runner.logger.closed


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_stop_saves_latest_update_and_closes_writer(tmp_path, signum):
    runner, snapshots = make_runner()
    previous = signal.getsignal(signum)
    with checkpoint_on_signal(runner, tmp_path):
        signal.raise_signal(signum)
        assert snapshots == []
        runner.current_learning_iteration = 9
        runner.logger.log(it=9)
        pytest.fail("Training must leave the loop after the checkpoint")
    assert snapshots[0][1] == 9
    assert runner.logger.closed
    assert signal.getsignal(signum) == previous


def test_failed_save_propagates_and_restores_handlers(tmp_path):
    runner, _ = make_runner()
    previous = signal.getsignal(signal.SIGTERM)

    def failed_save(_path):
        raise OSError("disk full")

    runner.save = failed_save
    with pytest.raises(OSError, match="disk full"):
        with checkpoint_on_signal(runner, tmp_path):
            signal.raise_signal(signal.SIGTERM)
            runner.logger.log(it=0)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert "log" not in vars(runner.logger)
