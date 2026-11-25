"""Export an explicit-state TorchScript student, without Isaac Sim or the teacher."""

import argparse
import copy
import json
from pathlib import Path

import torch

from viserdex.distillation.config import StudentCfg
from viserdex.distillation.frames import validate_pose_metadata
from viserdex.distillation.model import BeliefStudent, DeploymentPolicy
from viserdex.distillation.trainer import load_student_checkpoint


def export(checkpoint, output_dir):
    payload = load_student_checkpoint(checkpoint)
    frame_metadata = validate_pose_metadata(payload["metadata"])
    cfg = StudentCfg(**payload["student_cfg"])
    model = BeliefStudent(cfg).eval()
    model.load_state_dict(payload["student"], strict=True)
    deployment = DeploymentPolicy(copy.deepcopy(model)).eval()
    with torch.no_grad():
        scripted = torch.jit.script(deployment)
        # Verify nonzero recurrent state and a different batch size after serialization.
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "student_policy.pt"
        scripted.save(str(path))
        restored = torch.jit.load(str(path))
        probes = (
            torch.randn(3, cfg.proprio_dim),
            torch.randn(3, cfg.extero_dim),
            torch.randn(cfg.rnn_layers, 3, cfg.rnn_hidden),
        )
        expected = deployment(*probes)
        actual = restored(*probes)
        for ref, value in zip(expected, actual):
            torch.testing.assert_close(ref, value)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "iteration": payload["iteration"],
        "metadata": {**payload["metadata"], **frame_metadata},
        "proprio_dim": cfg.proprio_dim,
        "extero_dim": cfg.extero_dim,
        "hidden_shape": [cfg.rnn_layers, "batch", cfg.rnn_hidden],
        "action_history_order": "newest_first_raw_commands",
        "action_history_length": cfg.action_history,
        "proprio_layout": [
            "joint_positions_limit_normalized[21]",
            "last_four_actions[84]",
            "goal_quaternion_right_hand_base_link_xyzw[4]",
            "right_hand_base_link_position_environment[3]",
            "remaining_time[1]",
        ],
        "extero_layout": [
            "estimated_position_right_hand_base_link_m[3]",
            "estimated_quaternion_right_hand_base_link_xyzw[4]",
            "unique(estimated_quat * conjugate(goal_quat))[4]",
        ],
        "quaternion_sign": "canonical w >= 0",
        "reset": "zero hidden state and action history on episode reset",
        "action_mapping": "raw normalized position commands; use the same Revo3 EMA/limits as the teacher",
    }
    (output_dir / "student_policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported and verified {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
