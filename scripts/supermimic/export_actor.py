"""Export only the deployable 10x124 observation-history actor (no privileged input)."""

import argparse
from pathlib import Path
import sys
import json
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "viserdex"))
from viserdex.supermimic.core import Config
from viserdex.supermimic.networks import Actor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if payload.get("version") != 17:
    raise ValueError("Use a version-17 RoundedDie WujiOffset-HandPose-Hold-v1 checkpoint; legacy checkpoints need their original source")
if payload.get("metadata", {}).get("task_id") != "Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-WujiOffset-HandPose-Hold-v1":
    raise ValueError("Actor export requires the RoundedDie task checkpoint")
cfg = Config(**payload["config"])
# Preserve the existing portable FP32 deployment export. Training precision is
# recorded separately; this is not a bitwise reproduction of BF16 inference.
training_amp = cfg.amp
cfg.amp = "none"
actor = Actor(cfg).eval()
actor.load_state_dict(payload["actor"])
example = torch.randn(3, cfg.history, 124)
export = torch.jit.trace(actor, example)
args.output.parent.mkdir(parents=True, exist_ok=True)
export.save(str(args.output))
loaded = torch.jit.load(str(args.output))
for n in (1, 7):
    x = torch.randn(n, cfg.history, 124)
    torch.testing.assert_close(loaded(x), actor(x))
args.output.with_suffix(".json").write_text(
    json.dumps(
        dict(
            observation_protocol=payload["metadata"]["observation_protocol"],
            input="[batch,10,124], oldest to newest; same student normalization/XYZW/base frame",
            output="21 bounded [-1,1] offsets: clamp(home + offset_scale * action, joint limits), then EMA; hold home during reset warmup",
            control_hz=1/cfg.control_dt,
            ema_alpha=cfg.ema_alpha,
            action_mode=cfg.action_mode, action_home=list(cfg.action_home),
            action_offset_scale=cfg.action_offset_scale, action_warmup_s=cfg.action_warmup_s,
            reset="Repeat first measured observation ten times; clear four-action history; reset EMA target to action_home and restart warmup timer",
            joint_names=payload["metadata"]["joint_names"],
            noise_config=payload["noise_config"],
            training_amp=training_amp,
            export_precision="fp32",
        ),
        indent=2,
    )
)
print(f"Exported and checked: {args.output}")
