"""Short capacity check of the fast execution variant alongside REAL simulation.

Runs the normal official-launcher training entrypoint, with process-local changes
to execution only. Requires --check_joint_update; accepts at most three updates.
The checkpoint remains explicitly diagnostic. No default Trainer files change.
"""

import argparse
import runpy
from pathlib import Path
import sys
import time
import weakref

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--profile_wm_chunk", type=int, default=2048)
p.add_argument("--profile_actor_chunk", type=int, default=4096)
p.add_argument("--profile_critic_chunk", type=int, default=32768)
p.add_argument("--profile_fp32", action="store_true")
profile, remaining = p.parse_known_args()
if min(profile.profile_wm_chunk, profile.profile_actor_chunk, profile.profile_critic_chunk) < 1:
    p.error("Profile chunks must be positive")
sys.argv = [
    str(Path(__file__).with_name("train.py")),
    *remaining,
    "--wm_microbatch",
    str(profile.profile_wm_chunk),
    "--actor_microbatch",
    str(profile.profile_actor_chunk),
    "--critic_microbatch",
    str(profile.profile_critic_chunk),
    "--amp",
    "none" if profile.profile_fp32 else "bf16",
]
ns = runpy.run_path(sys.argv[0], run_name="profile_training_entrypoint")
args = ns["args"]
if not args.check_joint_update or args.smoke or args.resume or args.evaluate or args.iterations > 3:
    ns["app"].close()
    raise ValueError("Capacity check requires --check_joint_update, iterations<=3, no smoke/resume/evaluate")

import torch
import viserdex.supermimic.trainer as training

BaseTrainer = training.Trainer
original_checked_step = training.checked_step


def synchronize_time():
    torch.cuda.synchronize()
    return time.perf_counter()


class ProfileTrainer(BaseTrainer):
    def __init__(self, *constructor_args, **constructor_kwargs):
        super().__init__(*constructor_args, **constructor_kwargs)
        self.wm_chunk = profile.profile_wm_chunk
        self.actor_chunk = profile.profile_actor_chunk
        self.critic_chunk = profile.profile_critic_chunk
        self.phases = {}
        actor_ref = weakref.ref(self.actor)
        phases = self.phases

        def instrumented_step(model, optimizer, value_clip=False, distributed=None):
            result = original_checked_step(model, optimizer, value_clip, distributed=distributed)
            if model is actor_ref():
                phases["actor_end"] = synchronize_time()
            return result

        training.checked_step = instrumented_step
        raw_wm = super().wm_update

        def measured_wm(replay):
            result = raw_wm(replay)
            phases["wm_end"] = synchronize_time()
            return result

        self.wm_update = measured_wm

    def update(self, replay):
        before = synchronize_time()
        result = super().update(replay)
        after = synchronize_time()
        result.update(
            wm_seconds=self.phases["wm_end"] - before,
            actor_seconds=self.phases["actor_end"] - self.phases["wm_end"],
            critic_seconds=after - self.phases["actor_end"],
            network_update_seconds=after - before,
        )
        assert self.optimizer_steps == {
            "wm": 10 * self.iteration,
            "actor": self.iteration,
            "critic": 20 * self.iteration,
        }
        return result

    def save(self, path, metadata):
        metadata = dict(metadata)
        metadata["performance_diagnostic"] = dict(
            wm_chunk=self.wm_chunk,
            actor_chunk=self.actor_chunk,
            critic_chunk=self.critic_chunk,
            mlp_bf16=not profile.profile_fp32,
            note="Short capacity test; execution settings are persisted in the version-3 config",
        )
        return super().save(path, metadata)


ns["main"].__globals__["Trainer"] = ProfileTrainer
try:
    ns["main"]()
finally:
    training.checked_step = original_checked_step
