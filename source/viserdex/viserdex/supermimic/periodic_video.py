"""Frozen-checkpoint videos at completed training iteration boundaries.

No Torch imports: child rendering cannot consume the trainer's RNG or mutate
its simulator, Replay, networks or normalizers. Training waits for completion.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


@dataclass(frozen=True)
class VideoSettings:
    interval: int = 500
    steps: int = 500
    seed: int = 43
    timeout_s: float = 240.

    def __post_init__(self):
        if type(self.interval) is not int or self.interval < 0:
            raise ValueError('Video interval must be a nonnegative integer; zero disables recording')
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError('Video steps must be a positive integer')
        # Stay below the 300-second distributed collective timeout.
        if not 0 < self.timeout_s <= 240:
            raise ValueError('Video timeout must be in (0, 240] seconds')

    def due(self, iteration):
        return self.interval > 0 and iteration > 0 and iteration % self.interval == 0


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def child_environment():
    excluded = {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
    env = {k: v for k, v in os.environ.items()
           if k not in excluded and not k.startswith('TORCHELASTIC_')}
    env['PYGLET_HEADLESS'] = 'true'
    return env


def completed_artifacts(destination, iteration, settings, digest):
    summary = json.loads((destination/'summary.json').read_text())
    manifest = json.loads((destination/'manifest.json').read_text())
    if (summary['iteration'] != iteration or summary['frames'] != settings.steps
            or manifest['iteration'] != iteration or manifest['steps'] != settings.steps
            or manifest['checkpoint_sha256'] != digest or manifest['seed'] != settings.seed
            or manifest.get('exploration_action_noise') is not False
            or manifest.get('target_shaping') is not None):
        raise RuntimeError('Recorded video does not match the requested frozen policy')
    video = destination/'rollout.mp4'
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError('Recorder returned without a nonempty completed video')
    return summary


def record_checkpoint(checkpoint, env_config, run_dir, iteration, settings, *, device, repo,
                      trigger='completed_iteration'):
    """Render exactly one immutable checkpoint; fail visibly after saving it.

    A completed matching recording is idempotent. A stale/partial directory is
    never mistaken for success or silently overwritten. Logs survive failures.
    """
    if not settings.due(iteration):
        raise ValueError('Recording requested outside its configured boundary')
    checkpoint, env_config = Path(checkpoint).resolve(strict=True), Path(env_config).resolve(strict=True)
    repo = Path(repo).resolve(strict=True)
    videos = Path(run_dir).resolve() / 'videos'
    videos.mkdir(parents=True, exist_ok=True)
    name = f'iteration_{iteration:06d}'
    destination = videos / name
    status_path, log_path = videos / f'{name}.status.json', videos / f'{name}.log'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    identity = dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                    env_config_sha256=hashlib.sha256(env_config.read_bytes()).hexdigest(), settings=asdict(settings))
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('status') == 'complete' and all(previous.get(k) == v for k, v in identity.items()):
            completed_artifacts(destination, iteration, settings, digest)
            return previous
        raise RuntimeError(f'Existing video status requires inspection: {status_path}')
    if destination.exists():
        raise RuntimeError(f'Untracked/partial video directory requires inspection: {destination}')
    command = [sys.executable, str(repo/'scripts/supermimic/record_manual.py'),
               '--checkpoint', str(checkpoint), '--env_config', str(env_config),
               '--output_dir', str(destination), '--num_envs', '1', '--steps', str(settings.steps),
               '--seed', str(settings.seed), '--device', str(device), '--viz', 'newton_gl',
               'physics=newton_mjwarp']
    record = dict(identity, status='running', trigger=trigger, command=command, log=str(log_path),
                  video=str(destination/'rollout.mp4'), started_unix=time.time())
    atomic_json(status_path, record)
    start = time.monotonic()
    try:
        with log_path.open('w') as stream:
            process = subprocess.run(command, cwd=repo, env=child_environment(), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=settings.timeout_s, check=False)
        record['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'Recorder exited with code {process.returncode}; see {log_path}')
        summary = completed_artifacts(destination, iteration, settings, digest)
        record.update(status='complete', summary=summary)
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(wall_seconds=time.monotonic()-start, finished_unix=time.time())
        atomic_json(status_path, record)
        with (videos/'recordings.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
    return record


def record_resume_boundary(checkpoint, env_config, run_dir, iteration, settings, *, device, repo):
    """Record a due restored iteration in its new run before replay warmup.

    Always record that boundary in the new directory, even when the parent run
    already has a video. A byte-for-byte snapshot preserves the restored Actor
    and normalizers without modifying the source checkpoint or running an update.
    Disabled/non-due boundaries do nothing. The caller excludes evaluation and
    diagnostic-only modes before invoking this function.
    """
    if not settings.due(iteration):
        return None
    source = Path(checkpoint).resolve(strict=True)
    destination = Path(run_dir).resolve() / f'model_{iteration}.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Resume video snapshot conflicts with an existing checkpoint: {destination}')
    else:
        temporary = destination.with_suffix('.resume.tmp')
        if temporary.exists():
            raise RuntimeError(f'Partial resume video snapshot requires inspection: {temporary}')
        shutil.copyfile(source, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Resume checkpoint changed while copying its frozen video snapshot')
        temporary.replace(destination)
    return record_checkpoint(destination, env_config, run_dir, iteration, settings,
                             device=device, repo=repo, trigger='resume_boundary')
