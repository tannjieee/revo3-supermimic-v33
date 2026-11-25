"""Opt-in execution tuning using real replay and full production network sizes.

Diagnostic updates never become production checkpoints. Every candidate starts
from identical models/Adam/RNG and uses the same real sampled logical batches.
The live simulator and replay stay allocated, so memory measurements include
their residency. Compilation warmup is excluded from steady-state timings.
"""
from dataclasses import replace
import copy
import gc
import json
import math
import statistics
import time
import torch


def benchmark_execution(trainer, replay, metadata, output):
    baseline = output / "benchmark_start.pt"
    trainer.save(baseline, metadata)
    payload = torch.load(baseline, map_location="cpu", weights_only=False)
    original = trainer.cfg
    original_capture = trainer.wm_cuda_graph
    batches = {original.wm_window: replay.sample(original.wm_batch, original.wm_window),
               1: replay.sample(original.actor_batch)}

    class FixedReplay:
        def sample(self, batch, window=1, **kwargs):
            result = batches[window]
            assert len(result["state"]) == batch
            return result

        def update_priorities(self, batch, errors):
            replay.update_priorities(batch, errors)

    rows = []
    path = output / "execution_benchmark.json"
    def save():
        path.write_text(json.dumps(dict(source_iteration=payload["iteration"],
            amp=original.amp, wm_batch=original.wm_batch, actor_batch=original.actor_batch,
            horizon=original.horizon, rows=rows), indent=2))

    wm_candidates = [(2048,2048,8192)] if trainer.wm_cuda_graph else [(128,128,128), (512,128,128), (1024,128,128), (2048,128,128)]
    policy_candidates = [(2048,2048,8192), (2048,4096,8192)] if trainer.wm_cuda_graph else [(2048,128,128), (2048,512,8192), (2048,1024,8192), (2048,2048,8192)]
    phases = ([("wm_eager", [(2048,2048,8192)])] if original_capture else []) + [("wm", wm_candidates), ("policy", policy_candidates)]
    for phase, candidates in phases:
        for wm, actor, critic in candidates:
            trainer.wm_cuda_graph = original_capture and phase != "wm_eager"
            row = dict(phase=phase, wm_microbatch=wm, actor_microbatch=actor,
                       critic_microbatch=critic, wm_cuda_graph=trainer.wm_cuda_graph, started_unix=time.time())
            print(json.dumps(dict(benchmark_start=row)), flush=True)
            trainer.cfg = replace(original, wm_microbatch=wm, actor_microbatch=actor, critic_microbatch=critic)
            trainer.restore(copy.deepcopy(payload), metadata, allow_microbatch_override=True)
            # Exercise genuine policy optimizer work even during WM-only warmup.
            for group in trainer.opt_actor.param_groups:
                group['lr'] = original.learning_rate * original.actor_lr_scale
            for model in (trainer.wm, trainer.actor, trainer.critic):
                model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            update = trainer.wm_update if phase.startswith("wm") else trainer.policy_update
            try:
                before = time.perf_counter()
                first = update(FixedReplay())
                torch.cuda.synchronize()
                row.update(warmup_seconds=time.perf_counter()-before, first_metrics=first)
                durations = []
                for _ in range(5 if phase.startswith("wm") else 2):
                    before = time.perf_counter()
                    metrics = update(FixedReplay())
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter()-before)
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise ValueError("Nonfinite benchmark metrics")
                free, total = torch.cuda.mem_get_info()
                row.update(status="ok", seconds=durations, median_seconds=statistics.median(durations),
                           final_metrics=metrics, cuda_used_gib=(total-free)/2**30,
                           torch_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                if original.dex_reference:
                    # Profile separately: profiler overhead is excluded from timings.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        update(FixedReplay())
                        torch.cuda.synchronize()
                    label=f'{phase}_{actor}'
                    prof.export_chrome_trace(str(output/f'profile_{label}.json'))
                    averages=prof.key_averages()
                    (output/f'profile_{label}.txt').write_text(
                        averages.table(sort_by='self_cpu_time_total',row_limit=20)+'\n'+
                        averages.table(sort_by='self_device_time_total',row_limit=20))
                    row['profile_launches']={e.key:e.count for e in averages
                        if e.key in ('cudaLaunchKernel','cudaGraphLaunch','cudaStreamSynchronize','cudaDeviceSynchronize')}
            except torch.OutOfMemoryError as exc:
                row.update(status="oom", error=str(exc))
            rows.append(row)
            save()
            print(json.dumps(dict(benchmark_result=row)), flush=True)
    trainer.cfg = original
    trainer.wm_cuda_graph = original_capture
