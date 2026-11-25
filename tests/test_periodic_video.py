"""Periodic recording boundaries, artifact integrity, and child process isolation."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from viserdex.supermimic import periodic_video as video


@pytest.mark.parametrize('iteration,due', [(0, False), (1, False), (499, False),
    (500, True), (501, False), (999, False), (1000, True), (1500, True)])
def test_absolute_iteration_boundaries(iteration, due):
    assert video.VideoSettings().due(iteration) is due
    assert not video.VideoSettings(interval=0).due(iteration)


@pytest.mark.parametrize('kwargs', [{'interval': -1}, {'interval': .5}, {'steps': 0},
    {'steps': 1.5}, {'timeout_s': 0}, {'timeout_s': 241}, {'timeout_s': float('nan')}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        video.VideoSettings(**kwargs)


def test_child_keeps_runtime_but_not_distributed_identity(monkeypatch):
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'):
        monkeypatch.setenv(key, 'test')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    monkeypatch.setenv('PYTHONPATH', '/pinned/runtime')
    env = video.child_environment()
    assert not any(k in env for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'))
    assert env['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert env['PYTHONPATH'] == '/pinned/runtime'
    assert env['PYGLET_HEADLESS'] == 'true'


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    checkpoint, config = tmp_path/'model_500.pt', tmp_path/'env.yaml'
    checkpoint.write_bytes(b'frozen-checkpoint')
    config.write_text('saved_environment: true\n')
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index('--output_dir')+1])
        output.mkdir()
        (output/'rollout.mp4').write_bytes(b'fake-video-for-unit-test')
        (output/'summary.json').write_text(json.dumps(dict(iteration=500, frames=500)))
        (output/'manifest.json').write_text(json.dumps(dict(iteration=500, steps=500, seed=43,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            exploration_action_noise=False, target_shaping=None)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(video.subprocess, 'run', child)
    def run():
        return video.record_checkpoint(checkpoint, config, tmp_path, 500, video.VideoSettings(),
                                       device='cuda:0', repo=tmp_path)
    return SimpleNamespace(run=run, calls=calls, child=child, checkpoint=checkpoint, config=config,
                           root=tmp_path, output=tmp_path/'videos/iteration_000500',
                           status=tmp_path/'videos/iteration_000500.status.json')


def test_completed_recording_and_resume_are_idempotent(fixture):
    first = fixture.run()
    assert first['status'] == 'complete'
    assert fixture.run() == first
    assert len(fixture.calls) == 1
    command, kwargs = fixture.calls[0]
    assert command[-1] == 'physics=newton_mjwarp'
    assert command[command.index('--steps')+1] == '500'
    assert command[command.index('--num_envs')+1] == '1'
    assert command[command.index('--seed')+1] == '43'
    assert '--target_velocity_deg_s' not in command
    assert kwargs['timeout'] == 240
    assert len((fixture.root/'videos/recordings.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('artifact', ['rollout.mp4', 'summary.json', 'manifest.json'])
def test_resume_rechecks_completed_artifacts(fixture, artifact):
    fixture.run()
    (fixture.output/artifact).unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        fixture.run()
    assert len(fixture.calls) == 1


@pytest.mark.parametrize('changed', ['checkpoint', 'config'])
def test_resume_rejects_changed_input(fixture, changed):
    fixture.run()
    getattr(fixture, changed).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Existing video status'):
        fixture.run()
    assert len(fixture.calls) == 1


@pytest.mark.parametrize('failure', ['nonzero', 'timeout', 'missing', 'wrong_policy', 'shaped', 'interrupt'])
def test_failure_is_durable_and_never_overwrites_checkpoint(fixture, monkeypatch, failure):
    original = fixture.checkpoint.read_bytes()
    def failed(command, **kwargs):
        if failure == 'nonzero':
            return SimpleNamespace(returncode=2)
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        result = fixture.child(command, **kwargs)
        if failure == 'missing':
            (fixture.output/'rollout.mp4').unlink()
        else:
            manifest = json.loads((fixture.output/'manifest.json').read_text())
            manifest['checkpoint_sha256' if failure == 'wrong_policy' else 'target_shaping'] = 'unexpected'
            (fixture.output/'manifest.json').write_text(json.dumps(manifest))
        return result
    monkeypatch.setattr(video.subprocess, 'run', failed)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt)):
        fixture.run()
    status = json.loads(fixture.status.read_text())
    assert status['status'] == 'failed' and status['error']
    assert fixture.checkpoint.read_bytes() == original
    with pytest.raises(RuntimeError, match='Existing video status'):
        fixture.run()


def test_partial_untracked_video_is_not_overwritten(fixture):
    fixture.output.mkdir(parents=True)
    (fixture.output/'rollout.partial.mp4').write_bytes(b'partial')
    with pytest.raises(RuntimeError, match='partial video directory'):
        fixture.run()
    assert not fixture.calls


def test_resume_boundary_records_exact_source_before_any_new_update(fixture):
    source_bytes = fixture.checkpoint.read_bytes()
    restarted = fixture.root/'restart'
    def resume():
        return video.record_resume_boundary(fixture.checkpoint, fixture.config, restarted, 500,
            video.VideoSettings(), device='cuda:0', repo=fixture.root)
    result = resume()
    assert result['iteration'] == 500 and result['trigger'] == 'resume_boundary'
    assert Path(result['checkpoint']) == restarted/'model_500.pt'
    # Byte equality covers all weights, normalizer buffers and metadata; this
    # snapshot must not be produced after collection has changed those buffers.
    assert (restarted/'model_500.pt').read_bytes() == fixture.checkpoint.read_bytes() == source_bytes
    assert resume() == result and len(fixture.calls) == 1
    # Existing successful parent videos do not suppress the new run boundary.
    assert fixture.run()['status'] == 'complete'
    second_restart = fixture.root/'second_restart'
    result = video.record_resume_boundary(fixture.checkpoint, fixture.config, second_restart, 500,
        video.VideoSettings(), device='cuda:0', repo=fixture.root)
    assert result['status'] == 'complete' and len(fixture.calls) == 3


@pytest.mark.parametrize('iteration,interval', [(0, 500), (499, 500), (501, 500), (500, 0)])
def test_resume_nondue_boundaries_do_not_touch_files(tmp_path, iteration, interval):
    assert video.record_resume_boundary(tmp_path/'missing.pt', tmp_path/'missing.yaml',
        tmp_path/'unused', iteration, video.VideoSettings(interval=interval), device='cpu', repo=tmp_path) is None
    assert not (tmp_path/'unused').exists()


def test_resume_boundary_failure_preserves_frozen_checkpoint(fixture, monkeypatch):
    original = fixture.checkpoint.read_bytes()
    monkeypatch.setattr(video.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=2))
    restarted = fixture.root/'restart'
    with pytest.raises(RuntimeError, match='Recorder exited'):
        video.record_resume_boundary(fixture.checkpoint, fixture.config, restarted, 500,
            video.VideoSettings(), device='cpu', repo=fixture.root)
    assert (restarted/'model_500.pt').read_bytes() == fixture.checkpoint.read_bytes() == original
    status = json.loads((restarted/'videos/iteration_000500.status.json').read_text())
    assert status['status'] == 'failed' and status['trigger'] == 'resume_boundary'


def test_resume_boundary_refuses_conflicting_snapshot(fixture):
    restarted = fixture.root/'restart'
    restarted.mkdir()
    snapshot = restarted/'model_500.pt'
    snapshot.write_bytes(b'different-training-checkpoint')
    with pytest.raises(RuntimeError, match='conflicts'):
        video.record_resume_boundary(fixture.checkpoint, fixture.config, restarted, 500,
            video.VideoSettings(), device='cpu', repo=fixture.root)
    assert snapshot.read_bytes() == b'different-training-checkpoint'
    assert not fixture.calls


def test_resume_boundary_accepts_already_saved_current_run_snapshot(fixture):
    # The train entrypoint may first serialize restored model state with its
    # validated new environment metadata (e.g. an allowed solver override).
    original = fixture.checkpoint.read_bytes()
    def record():
        return video.record_resume_boundary(fixture.checkpoint, fixture.config, fixture.root, 500,
            video.VideoSettings(), device='cpu', repo=fixture.root)
    first = record()
    assert first['trigger'] == 'resume_boundary' and first['status'] == 'complete'
    assert fixture.checkpoint.read_bytes() == original
    assert not fixture.checkpoint.with_suffix('.resume.tmp').exists()
    assert record() == first and len(fixture.calls) == 1
