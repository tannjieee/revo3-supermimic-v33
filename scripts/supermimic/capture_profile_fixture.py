"""Capture a small REAL Newton replay for offline update profiling; no optimization."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "viserdex"))
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID, TransitionRecorderCfg

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--env_config", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=64)
add_runtime_args(p, TASK_ID)
args = parse_runtime_args(p)
binding = SavedEnvironmentBinding(args.env_config)


def configure(cfg):
    binding.configure(cfg, args)
    configure_clean_state_observations(cfg)
    cfg.recorders = TransitionRecorderCfg()
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = 1729
    binding.verify(cfg)


app = SimulationRuntime(args, configure=configure)
import torch
import gymnasium as gym
from viserdex.supermimic.core import Config
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.replay import Replay

torch.manual_seed(1729)
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
assert payload["version"] == 13, "Legacy checkpoints need their original source"
cfg = Config(**payload["config"])
noise = PerceptionNoiseCfg(**payload["noise_config"])
env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
try:
    obs, _ = env.reset()
    adapter = Adapter(env, noise, cfg.history)
    adapter.audit_observation(obs)
    actor = Actor(cfg).to(env.device).eval()
    actor.load_state_dict(payload["actor"])
    replay = Replay(args.steps, args.num_envs, env.device)
    with torch.no_grad():
        for i in range(args.steps):
            action = actor.sample(adapter.history, cfg.exploration)
            row, _, _, _ = adapter.step(action)
            replay.add(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            data=replay.data,
            size=replay.size,
            cursor=replay.cursor,
            capacity=replay.capacity,
            num_envs=args.num_envs,
            context=adapter.context.cpu(),
            limits=adapter.limits.cpu(),
            base_inverse=adapter.base_inverse.cpu(),
            checkpoint=str(args.checkpoint.resolve()),
            environment_contract=binding.contract_sha,
        ),
        args.output,
    )
    print(f"Saved real replay: {args.output}; {args.steps} steps x {args.num_envs} environments", flush=True)
finally:
    env.close()
    app.close()
