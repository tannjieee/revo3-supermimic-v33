#!/usr/bin/env python3
"""Replay-only CPU/CUDA storage benchmark, using a production config/schema.

The data and priority errors are synthetic. No simulator, neural network or
production checkpoint is updated. Timings include actual Replay sampling,
priority updates and transfers to/from CUDA, unlike a fixed-batch WM benchmark.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import Config
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.sensors import initial_sensors


def summarize(values):
    ordered = sorted(values)
    return dict(median_ms=statistics.median(values) * 1000,
                mean_ms=statistics.mean(values) * 1000,
                p95_ms=ordered[int(.95 * (len(ordered) - 1))] * 1000,
                minimum_ms=ordered[0] * 1000, repeats=len(values))


def measure(fn, warmup, repeats):
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        before = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - before)
    del result
    return summarize(durations)


def record_from_config(cfg, noise, n, device):
    state = torch.randn(n, cfg.state_dim, device=device) * .01
    state[:, 45:49] = state.new_tensor([0, 0, 0, 1])
    state[:, 79:83] = state.new_tensor([0, 0, 0, 1])
    record = dict(state=state,
                  history=torch.randn(n, cfg.history, cfg.observation_dim, device=device),
                  action=torch.rand(n, 21, device=device) * 2 - 1,
                  next_physical=state[:, :55].clone(),
                  next_aux=state[:, 55:].clone(),
                  done=torch.zeros(n, dtype=torch.bool, device=device),
                  terminated=torch.zeros(n, dtype=torch.bool, device=device),
                  reward=torch.randn(n, device=device),
                  success=torch.zeros(n, dtype=torch.bool, device=device),
                  goal_elapsed_s=torch.ones(n, device=device))
    record.update({"sensor/" + k: v for k, v in initial_sensors(n, device, noise).items()})
    return record


@torch.no_grad()
def benchmark_storage(storage, cfg, noise, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    record = record_from_config(cfg, noise, args.num_envs, "cuda:0")
    replay = Replay(cfg.buffer_steps, args.num_envs, "cuda:0", cfg.event_capacity,
                    cfg.event_fraction, cfg.wm_window, adaptive=True, storage_device=storage)
    ids = torch.arange(args.num_envs, device="cuda:0")
    before = time.perf_counter()
    for t in range(cfg.buffer_steps):
        record["state"][:, 0] = t
        record["next_physical"][:, 0] = t + .01
        record["done"].copy_((ids + t) % 97 == 0)
        record["terminated"].copy_(record["done"])
        replay.add(record)
    torch.cuda.synchronize()
    fill_seconds = time.perf_counter() - before
    replay.priority.copy_(torch.linspace(.01, 100., replay.priority.numel(),
                                         device=replay.storage_device).reshape_as(replay.priority))
    errors = torch.linspace(.05, 5., cfg.wm_batch, device="cuda:0")

    # Check sampled probabilities/weights/masks against explicit formulas, not
    # cross-device RNG indices (CPU and CUDA generators intentionally differ).
    check = replay.sample(cfg.wm_batch, cfg.wm_window)
    starts = (torch.arange(replay.size - cfg.wm_window + 1, device=replay.storage_device)
              + replay.cursor) % replay.capacity
    probability = replay.priority[starts].flatten().pow(.6)
    probability = .2 / probability.numel() + .8 * probability / probability.sum()
    logical_start = (check["priority_slot"].to(replay.storage_device) - replay.cursor) % replay.capacity
    choice = logical_start * args.num_envs + check["env_id"].to(replay.storage_device)
    torch.testing.assert_close(check["sample_probability"], probability[choice].cuda())
    torch.testing.assert_close(check["importance_weight"], 1 / (probability.numel() * check["sample_probability"]))
    mask = torch.cat((torch.ones(cfg.wm_batch, 1, device="cuda:0"),
                      (~check["done"][:, :-1]).float().cumprod(1)), 1)
    torch.testing.assert_close(check["mask"], mask)
    torch.testing.assert_close(check["state"][:, 1:, 0] - check["state"][:, :-1, 0],
                               torch.ones(cfg.wm_batch, cfg.wm_window - 1, device="cuda:0"))
    del check
    sample_batch = replay.sample(cfg.wm_batch, cfg.wm_window)

    def wm_cycle():
        batch = replay.sample(cfg.wm_batch, cfg.wm_window)
        return replay.update_priorities(batch, errors)

    iteration_counter = [0]

    def iteration_replay_path():
        # Same average 2.5 real vector steps / 10 WM / 1 actor sampling as UTD .4.
        count = 2 + iteration_counter[0] % 2
        iteration_counter[0] += 1
        for _ in range(count):
            replay.add(record)
        for _ in range(cfg.wm_updates):
            wm_cycle()
        return replay.sample(cfg.actor_batch)

    timings = {}
    operations = (
        ("add_from_cuda", lambda: replay.add(record)),
        ("wm_sample_with_transfer", lambda: replay.sample(cfg.wm_batch, cfg.wm_window)),
        ("priority_update_from_cuda", lambda: replay.update_priorities(sample_batch, errors)),
        ("wm_sample_and_priority", wm_cycle),
        ("actor_sample_with_transfer", lambda: replay.sample(cfg.actor_batch)),
        ("iteration_replay_path", iteration_replay_path),
    )
    for name, fn in operations:
        if name == "priority_update_from_cuda":
            sample_batch = replay.sample(cfg.wm_batch, cfg.wm_window)
        timings[name] = measure(fn, args.warmup, args.repeats)
        print(json.dumps(dict(storage=storage, operation=name, **timings[name])), flush=True)
    free, total = torch.cuda.mem_get_info()
    rss_kib = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines()
                   if line.startswith("VmRSS:"))
    return dict(storage_device=str(replay.storage_device), output_device="cuda:0",
                fill_seconds=fill_seconds, timings=timings,
                allocated_replay_gib=replay.allocated_bytes / 2**30,
                cuda_process_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                cuda_process_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                cuda_process_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                cuda_device_used_gib=(total - free) / 2**30, process_rss_gib=rss_kib / 2**20,
                semantic_checks="sample probability, exact importance formula, terminal masks, ordered real windows passed",
                transition_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in record.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text())
    cfg, noise = Config(**payload["config"]), PerceptionNoiseCfg(**payload["noise_config"])
    if not cfg.compact_state or cfg.goal_mode != "single_dense_hold":
        parser.error("This schema benchmark currently targets compact single_dense_hold records")
    if args.repeats < 1 or args.warmup < 0 or args.num_envs < 1:
        parser.error("Invalid benchmark counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(config_path=str(args.config.resolve()), torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), synthetic_data=True,
                  synthetic_error="fixed positive linspace .05..5; not real model errors",
                  excludes="physics simulation, all network computation, checkpoints and production training",
                  sampling_rng="storage device generator; cross-device draws need not match bitwise",
                  num_envs=args.num_envs, buffer_steps=cfg.buffer_steps, wm_batch=cfg.wm_batch,
                  wm_window=cfg.wm_window, actor_batch=cfg.actor_batch, wm_updates=cfg.wm_updates,
                  observation_dim=cfg.observation_dim, history=cfg.history,
                  repeats=args.repeats, warmup=args.warmup, rows=[])
    for storage in ("cpu", "cuda:0"):
        result["rows"].append(benchmark_storage(storage, cfg, noise, args))
        args.output.write_text(json.dumps(result, indent=2))
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(dict(output=str(args.output), status="complete")), flush=True)


if __name__ == "__main__":
    main()
