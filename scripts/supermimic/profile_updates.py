"""Controlled diagnostic variants of the unchanged production update code.

Each variant starts from the same checkpoint and actual Newton replay. Simulator
memory/time is excluded. No task/hyperparameter change or training continuation.
Execution options are changed only in this short-lived profiling process.
"""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import weakref

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "viserdex"))
from viserdex.supermimic.core import Config
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.trainer import Trainer
import viserdex.supermimic.trainer as training

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--fixture", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--variants", nargs="+", default=["baseline", "critic_batch", "larger_chunks", "bf16"])
args = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda:0"
fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)


def synchronize_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def run_variant(name):
    settings = {
        "baseline": (128, 128, 128, False),
        "critic_batch": (128, 128, 32768, False),
        "larger_chunks": (512, 512, 32768, False),
        "bf16": (512, 512, 32768, True),
        "full_batch_bf16": (2048, 4096, 32768, True),
    }
    wm_chunk, actor_chunk, critic_chunk, amp = settings[name]
    torch.manual_seed(12345)
    cfg = Config(**payload["config"])
    trainer = Trainer(
        cfg,
        PerceptionNoiseCfg(**payload["noise_config"]),
        fixture["context"].to(device),
        fixture["limits"].to(device),
        fixture["base_inverse"].to(device),
        device,
    )
    # Adam's CPU step tensors can alias the loaded checkpoint even when moment
    # tensors move to CUDA. Isolate all state so variants start identically.
    trainer.restore(copy.deepcopy(payload), payload["metadata"], restore_rng=False)
    trainer.optimizer_steps = dict(trainer.optimizer_steps)
    initial_steps = dict(trainer.optimizer_steps)
    replay = Replay(fixture["capacity"], fixture["num_envs"], device)
    replay.data = fixture["data"]
    replay.size = fixture["size"]
    replay.cursor = fixture["cursor"]
    cfg.wm_microbatch, cfg.actor_microbatch, cfg.critic_microbatch = wm_chunk, actor_chunk, critic_chunk
    cfg.amp = "bf16" if amp else "none"
    for model in (trainer.wm, trainer.actor, trainer.critic):
        model.amp = cfg.amp
    stages = {}
    original_checked_step = training.checked_step
    actor_ref = weakref.ref(trainer.actor)
    sampling = {"seconds": 0.0}
    replay_ref = weakref.ref(replay)

    def instrumented_sample(*sample_args, **sample_kwargs):
        before = synchronize_time()
        result = Replay.sample(replay_ref(), *sample_args, **sample_kwargs)
        sampling["seconds"] += synchronize_time() - before
        return result

    replay.sample = instrumented_sample

    def instrumented_step(model, optimizer, value_clip=False):
        result = original_checked_step(model, optimizer, value_clip)
        if model is actor_ref():
            stages["actor_end"] = synchronize_time()
        return result

    training.checked_step = instrumented_step
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for repetition in range(args.repeats + 1):
        sampling["seconds"] = 0.0
        begin = synchronize_time()
        for _ in range(cfg.wm_updates):
            wm_metrics = trainer.wm_update(replay)
        after_wm = synchronize_time()
        metrics = trainer.policy_update(replay)
        end = synchronize_time()
        row = dict(
            warmup=repetition == 0,
            repetition=repetition,
            wm_seconds=after_wm - begin,
            actor_seconds=stages["actor_end"] - after_wm,
            critic_seconds=end - stages["actor_end"],
            total_seconds=end - begin,
            replay_sample_seconds=sampling["seconds"],
            torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            **wm_metrics,
            **metrics,
        )
        print(json.dumps(dict(variant=name, **row)), flush=True)
        rows.append(row)
    result = dict(
        settings=dict(wm_chunk=wm_chunk, actor_chunk=actor_chunk, critic_chunk=critic_chunk, bf16=amp),
        samples=rows,
        median_seconds={
            k: statistics.median(r[k] for r in rows[1:])
            for k in ("wm_seconds", "actor_seconds", "critic_seconds", "total_seconds", "replay_sample_seconds")
        },
        optimizer_steps={k: trainer.optimizer_steps[k] - initial_steps[k] for k in initial_steps},
    )
    expected = {"wm": cfg.wm_updates, "actor": 1, "critic": cfg.critic_epochs * cfg.critic_minibatches}
    assert result["optimizer_steps"] == {k: v * (args.repeats + 1) for k, v in expected.items()}
    training.checked_step = original_checked_step
    del trainer, replay
    gc.collect()
    torch.cuda.empty_cache()
    return result


report = dict(
    gpu=torch.cuda.get_device_name(),
    torch_version=torch.__version__,
    trainer_sha256=hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
    fixture_num_envs=fixture["num_envs"],
    fixture_steps=fixture["size"],
    logical_config=payload["config"],
    measurement="network-only, real Newton replay, k8/H32, 10/1/20 optimizer updates per sample",
    variants={},
)
args.output.parent.mkdir(parents=True, exist_ok=True)
for variant in args.variants:
    report["variants"][variant] = run_variant(variant)
    args.output.write_text(json.dumps(report, indent=2))
