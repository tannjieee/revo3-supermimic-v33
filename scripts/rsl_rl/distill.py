"""Train or evaluate the recurrent Revo3 student (ViserDex Appendix B-B).

Run through scripts/run_newton.sh. No image renderer is needed for pose-noise distillation.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from importlib.metadata import version
from pathlib import Path
import resource
import signal
import sys
import time

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--teacher_checkpoint", type=Path, help="Exact Revo3 Sim 6 PPO checkpoint; required for new training"
)
parser.add_argument("--student_checkpoint", type=Path, help="Resume/evaluate this student checkpoint")
parser.add_argument("--evaluate", action="store_true", help="Student-only rollout; no DAgger teacher actions")
parser.add_argument("--teacher_only", action="store_true", help="Evaluate the frozen teacher as a baseline")
parser.add_argument("--clean_pose", action="store_true", help="Evaluation ablation without simulated perception noise")
parser.add_argument(
    "--clean_observations", action="store_true", help="Evaluation ablation without pose or encoder noise"
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument(
    "--max_iterations", type=int, default=1000, help="Total completed updates, including resumed updates"
)
parser.add_argument("--epochs", type=int, default=None, help="Override 32 epochs for new training only")
parser.add_argument("--update_env_batch_size", type=int, default=None, help="Accumulate one full-batch gradient in environment chunks")
parser.add_argument("--optimizer_env_batch_size", type=int, default=None, help="Environment sequences per optimizer step, independent of rollout count")
parser.add_argument("--refresh_hidden_after_update", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--teacher_quaternion_sign", choices=("raw", "positive_world_w"), default=None)
parser.add_argument("--rollout_steps", type=int, default=None, help="Override 45-step BPTT for new training only")
parser.add_argument("--num_steps", type=int, default=1200, help="Evaluation duration")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--run_name", default=None)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument("--seed", type=int, default=None)
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3")
args = parse_runtime_args(parser)
if args.num_envs < 1 or args.max_iterations < 1 or args.num_steps < 1 or args.save_interval < 1:
    parser.error("Environment counts, iterations, steps and save interval must be positive")
if args.teacher_only and not args.evaluate:
    parser.error("--teacher_only requires --evaluate")
if (args.clean_pose or args.clean_observations) and not args.evaluate:
    parser.error("Clean observation options are evaluation ablations only")
if args.student_checkpoint and any(value is not None for value in (
    args.epochs, args.rollout_steps, args.update_env_batch_size, args.optimizer_env_batch_size,
    args.refresh_hidden_after_update, args.teacher_quaternion_sign,
)):
    parser.error("Resume/evaluation uses checkpoint hyperparameters; remove epochs/rollout_steps overrides")
if args.evaluate and not args.teacher_only and not args.student_checkpoint:
    parser.error("Student evaluation requires --student_checkpoint")
if not args.student_checkpoint and not args.teacher_checkpoint:
    parser.error("Provide --teacher_checkpoint for new training or teacher evaluation")
for checkpoint in (args.teacher_checkpoint, args.student_checkpoint):
    if checkpoint and not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

# Bind the persisted environment before launch_simulation chooses its runtime.
from viserdex.assets.objects import TARGET_OBJECT_CFG
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import teacher_metadata
from viserdex.distillation.trainer import load_student_checkpoint

_prepared_payload = load_student_checkpoint(args.student_checkpoint) if args.student_checkpoint else None
if not args.evaluate or args.teacher_only:
    _teacher_path = args.teacher_checkpoint or Path(_prepared_payload["metadata"]["teacher"]["path"])
    _prepared_teacher = teacher_metadata(_teacher_path, TARGET_OBJECT_CFG["name"])
else:
    _prepared_teacher = _prepared_payload["metadata"]["teacher"]
if _prepared_payload:
    _metadata = _prepared_payload["metadata"]
    _environment = args.student_checkpoint.parent / "env.yaml"
    if "rollout_environment_sha256" not in _metadata:
        raise ValueError("Legacy student lacks a saved environment fingerprint; start a bound student run")
    _binding = SavedEnvironmentBinding(_environment, _metadata["rollout_environment_sha256"])
    if _binding.contract_sha != _metadata["environment_binding"]["contract_sha256"]:
        raise ValueError("Saved student environment contract changed")
else:
    _binding = SavedEnvironmentBinding(_prepared_teacher["environment_path"], _prepared_teacher["environment_sha256"])
app = SimulationRuntime(args, configure=lambda cfg: _binding.configure(cfg, args))

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import viserdex.tasks  # noqa: F401
from viserdex.assets.objects import TARGET_OBJECT_CFG
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.distillation.config import DistillationCfg, PerceptionNoiseCfg, StudentCfg
from viserdex.distillation.frames import pose_frame_metadata, validate_pose_metadata
from viserdex.distillation.model import BeliefStudent
from viserdex.distillation.observations import StudentObservations
from viserdex.distillation.teacher import (
    audit_clean_state_observations,
    configure_clean_state_observations,
    load_teacher,
    teacher_metadata,
    audit_teacher_observations,
    teacher_observations,
    teacher_observation_protocol,
)
from viserdex.distillation.trainer import DAggerTrainer, load_student_checkpoint, mix_actions
from viserdex.tasks.manipulation.inhand.config.revo3_hand.revo3_env_cfg import ViserDexRevo3EnvCfg


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = args.device or "cuda:0"
    payload = _prepared_payload
    if payload:
        validate_pose_metadata(payload["metadata"])
    args.seed = args.seed if args.seed is not None else payload["metadata"]["seed"] if payload else 42
    torch.manual_seed(args.seed)
    student_cfg = StudentCfg(**payload["student_cfg"]) if payload else StudentCfg()
    noise_cfg = PerceptionNoiseCfg(**payload["noise_cfg"]) if payload else PerceptionNoiseCfg()
    if payload and "joint_enabled" not in payload["noise_cfg"]:
        raise ValueError(
            "Legacy checkpoint predates explicit sensor-noise configuration. Start a new distillation run."
        )
    if payload and "joint_delay_max_steps" not in payload["noise_cfg"]:
        noise_cfg.joint_delay_max_steps = 0  # Preserve old checkpoints' encoder timing in evaluation.
    train_cfg = (
        DistillationCfg(**payload["training_cfg"])
        if payload
        else DistillationCfg(
            epochs=args.epochs if args.epochs is not None else 32,
            rollout_steps=args.rollout_steps if args.rollout_steps is not None else 45,
            update_env_batch_size=args.update_env_batch_size,
            optimizer_env_batch_size=args.optimizer_env_batch_size,
            refresh_hidden_after_update=bool(args.refresh_hidden_after_update),
            teacher_quaternion_sign=args.teacher_quaternion_sign or "raw",
        )
    )
    observation_protocol = teacher_observation_protocol(train_cfg.teacher_quaternion_sign)
    if payload and not args.evaluate and payload["metadata"].get("teacher_observation_protocol") != observation_protocol:
        raise ValueError("Resume requires the saved teacher observation protocol")
    if args.clean_pose or args.clean_observations:
        noise_cfg.enabled = False
    if args.clean_observations:
        noise_cfg.joint_enabled = False
    object_name = TARGET_OBJECT_CFG["name"]
    if payload:
        if payload["metadata"]["object"] != object_name:
            raise ValueError("VISERDEX_OBJECT must match the saved student checkpoint")
        if payload["metadata"]["joint_names"] != list(REVO3_JOINT_NAMES):
            raise ValueError("Checkpoint joint order differs from REVO3_JOINT_NAMES")
        if not args.evaluate and args.max_iterations <= payload["iteration"]:
            raise ValueError("max_iterations is a total budget and must exceed the saved completed iterations")
    teacher_info = _prepared_teacher
    if not args.evaluate or args.teacher_only:
        if teacher_info["physics"] != str(app.env_cfg().sim.physics.class_type):
            raise ValueError("Teacher physics differs from the bound rollout")
        if payload and teacher_info["sha256"] != payload["metadata"]["teacher"]["sha256"]:
            raise ValueError("Resume requires the same frozen teacher checkpoint (SHA-256 mismatch)")

    mode = "teacher_eval" if args.teacher_only else "student_eval" if args.evaluate else "student"
    name = args.run_name or f"{mode}_{object_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    log_dir = args.output_dir or Path("logs/distillation/revo3_v3") / name
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    cfg = app.env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = device
    cfg.log_dir = str(log_dir)
    # Same measured performance options as train_revo3_5090.sh; dynamics unchanged.
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    configure_clean_state_observations(cfg)
    environment_binding = _binding.verify(cfg)
    env = None
    writer = None
    previous_handlers = {}
    requested = {"stop": False, "save": False}

    def handle_signal(signum, _frame):
        requested["save"] = True
        if signum != signal.SIGUSR1:
            requested["stop"] = True

    try:
        env = RslRlVecEnvWrapper(gym.make("Isaac-ViserDex-Repose-Revo3", cfg=cfg))
        obs = env.get_observations()
        source_state_audit = audit_clean_state_observations(env.unwrapped, obs)
        student = BeliefStudent(student_cfg).to(device).eval()
        trainer = DAggerTrainer(student, train_cfg)
        if payload:
            if args.evaluate:
                student.load_state_dict(payload["student"], strict=True)
            else:
                trainer.restore(payload)
        teacher = load_teacher(teacher_info, obs, device) if not args.evaluate or args.teacher_only else None
        adapter = StudentObservations(env.unwrapped, student_cfg, noise_cfg)
        hidden = student.initial_state(env.num_envs, device)
        reset_before = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        metadata = {
            "object": object_name,
            "physics": str(cfg.sim.physics.class_type),
            "teacher": teacher_info,
            "joint_names": list(REVO3_JOINT_NAMES),
            **pose_frame_metadata(),
            "step_dt": env.unwrapped.step_dt,
            "seed": args.seed,
            "task": "Isaac-ViserDex-Repose-Revo3",
            "environment_binding": environment_binding,
            "cameras_rendered": False,
            "sensor_timing": {
                "convention": "preserve step-based Appendix defaults at the actual control rate",
                "joint_delay_max_s": noise_cfg.joint_delay_max_steps * env.unwrapped.step_dt,
                "visual_update_max_s": noise_cfg.max_period * env.unwrapped.step_dt,
                "bptt_duration_s": train_cfg.rollout_steps * env.unwrapped.step_dt,
            },
            "teacher_observation_corruption": noise_cfg.joint_enabled,
            "teacher_observation_protocol": observation_protocol,
            "teacher_quaternion_sign": train_cfg.teacher_quaternion_sign,
            "recurrent_state_refresh": train_cfg.refresh_hidden_after_update,
            "optimizer_env_batch_size": train_cfg.optimizer_env_batch_size,
            "teacher_noisy_terms": ["normalized_joint_positions"] if noise_cfg.joint_enabled else [],
            "teacher_visual_noise": False,
            "teacher_privileged_noise": False,
            "clean_state_audit": source_state_audit,
            "student_noise_version": 3,
            "sensor_noise_hardware_calibrated": False,
            "noise_references": [
                "https://arxiv.org/html/2604.11138v1",
                "https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeManualDR.yaml",
            ],
            "versions": {
                name: version(name) for name in ("torch", "isaacsim", "isaaclab", "newton", "warp-lang", "rsl-rl-lib")
            },
        }
        if payload and not args.evaluate and payload["metadata"].get("physics") != metadata["physics"]:
            raise ValueError("Resume requires a checkpoint from the same physics backend; start a new Newton run")
        if payload and abs(payload["metadata"]["step_dt"] - metadata["step_dt"]) > 1e-9:
            raise ValueError("Student checkpoint control period differs from the environment")
        import hashlib
        dump_yaml(str(log_dir / "env.yaml"), cfg)
        metadata["rollout_environment_sha256"] = hashlib.sha256((log_dir / "env.yaml").read_bytes()).hexdigest()
        manifest = {
            "metadata": metadata,
            "student": asdict(student_cfg),
            "training": asdict(train_cfg),
            "noise": asdict(noise_cfg),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        (log_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"[DISTILL] log_dir={log_dir}; teacher={teacher_info['path']}; student_inputs=113+11; actions=21",
            flush=True,
        )
        writer = SummaryWriter(str(log_dir))
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, handle_signal)

        if args.evaluate:
            rewards, errors = [], []
            completed = 0
            successes = 0
            episode_successes = []
            successes_per_env = torch.zeros(env.num_envs, device=device)
            termination_counts = {}
            command = env.unwrapped.command_manager.get_term("object_pose")
            # Command counters are reset before post-step observations on completed episodes.
            previous_counter = command.command_counter.clone()
            with torch.no_grad():
                for step in range(args.num_steps):
                    if not app.is_running() or requested["stop"]:
                        break
                    prop, extero, _ = adapter(obs)
                    action, _, hidden = student(prop, extero, hidden)
                    if args.teacher_only:
                        teacher_input = teacher_observations(obs, prop, train_cfg.teacher_quaternion_sign)
                        if step % 100 == 0:
                            print(
                                f"[TEACHER_SENSOR_AUDIT] {json.dumps(audit_teacher_observations(teacher_input, obs, prop, train_cfg.teacher_quaternion_sign))}",
                                flush=True,
                            )
                        action = teacher(teacher_input)
                    obs, reward, dones, info = env.step(action)
                    dones = dones.bool()
                    counter = command.command_counter
                    episode_successes.extend(successes_per_env[dones].cpu().tolist())
                    successes_per_env[dones] = 0
                    newly_succeeded = torch.where(dones, 0, counter - previous_counter).clamp_min(0)
                    successes_per_env += newly_succeeded
                    successes += int(newly_succeeded.sum())
                    previous_counter = counter.clone()
                    completed += int(dones.sum())
                    rewards.append(float(reward.mean()))
                    errors.append(float(command.metrics["orientation_error"].mean()))
                    for term in env.unwrapped.termination_manager.active_terms:
                        count = env.unwrapped.termination_manager.get_term(term).sum()
                        termination_counts[term] = termination_counts.get(term, 0) + int(count)
                    hidden *= (~dones)[None, :, None]
                    adapter.record_action(action, dones)
                    if (step + 1) % 100 == 0:
                        eval_progress = {
                            "mode": mode,
                            "steps": step + 1,
                            "num_envs": env.num_envs,
                            "completed_episodes": completed,
                            "observed_success_increments": successes,
                            **audit_clean_state_observations(env.unwrapped, obs),
                        }
                        (log_dir / "progress.json").write_text(json.dumps(eval_progress, indent=2) + "\n")
                        print(f"[EVAL_PROGRESS] {json.dumps(eval_progress)}", flush=True)
            result = {
                "mode": mode,
                "steps": len(rewards),
                "num_envs": env.num_envs,
                "noise_enabled": noise_cfg.enabled,
                "completed_episodes": completed,
                "joint_noise_enabled": noise_cfg.joint_enabled,
                "teacher_observation_protocol": observation_protocol,
                "teacher_used": args.teacher_only,
                "teacher_proprio_noise_enabled": args.teacher_only and noise_cfg.joint_enabled,
                "teacher_visual_noise": False,
                "teacher_privileged_noise": False,
                "observed_success_increments": successes,
                "mean_completed_episode_successes": sum(episode_successes) / max(1, len(episode_successes)),
                "mean_step_reward": sum(rewards) / max(1, len(rewards)),
                "mean_orientation_error_rad": sum(errors) / max(1, len(errors)),
                "termination_counts": termination_counts,
                "checkpoint": str(args.teacher_checkpoint if args.teacher_only else args.student_checkpoint),
            }
            (log_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"[EVAL] {json.dumps(result)}", flush=True)
            return

        metrics_file = log_dir / "metrics.jsonl"
        command = env.unwrapped.command_manager.get_term("object_pose")
        previous_counter = command.command_counter.clone()
        while trainer.iteration < args.max_iterations and app.is_running():
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            beta = train_cfg.beta(trainer.iteration)
            batch = {name: [] for name in ("prop", "extero", "target", "teacher", "reset")}
            batch["hidden"] = hidden.detach().clone()
            reward_sum, selected_sum, done_sum = 0.0, 0.0, 0
            mixed_successes, mixed_out_of_reach = 0, 0
            joint_error_sq, pose_error_sq = 0.0, 0.0
            teacher_sensor_audit = {}
            env_logs = {}
            with torch.no_grad():
                for rollout_step in range(train_cfg.rollout_steps):
                    prop, extero, target = adapter(obs)
                    joint_error_sq += float((prop[:, :21] - obs["policy"][:, :21]).square().mean())
                    pose_error_sq += float((extero[:, :3] - target[:, :3]).square().mean())
                    teacher_input = teacher_observations(obs, prop, train_cfg.teacher_quaternion_sign)
                    if rollout_step == 0:
                        teacher_sensor_audit = audit_teacher_observations(
                            teacher_input, obs, prop, train_cfg.teacher_quaternion_sign
                        )
                    teacher_action = teacher(teacher_input)
                    student_action, _, hidden = student(prop, extero, hidden)
                    student_action += train_cfg.action_noise_std * torch.randn_like(student_action)
                    action, selected = mix_actions(teacher_action, student_action, beta)
                    for key, value in {
                        "prop": prop,
                        "extero": extero,
                        "target": target,
                        "teacher": teacher_action,
                        "reset": reset_before,
                    }.items():
                        batch[key].append(value.detach().clone())
                    obs, reward, dones, info = env.step(action)
                    dones = dones.bool()
                    counter = command.command_counter
                    mixed_successes += int(torch.where(dones, 0, counter - previous_counter).clamp_min(0).sum())
                    previous_counter = counter.clone()
                    if "object_out_of_reach" in env.unwrapped.termination_manager.active_terms:
                        mixed_out_of_reach += int(env.unwrapped.termination_manager.get_term("object_out_of_reach").sum())
                    hidden *= (~dones)[None, :, None]
                    adapter.record_action(action, dones)
                    reset_before = dones.clone()
                    reward_sum += float(reward.mean())
                    selected_sum += float(selected.float().mean())
                    done_sum += int(dones.sum())
                    for key, value in info.get("log", {}).items():
                        if isinstance(value, (float, int)) or isinstance(value, torch.Tensor) and value.numel() == 1:
                            env_logs[key] = float(value)
            for key in ("prop", "extero", "target", "teacher", "reset"):
                batch[key] = torch.stack(batch[key])
                if not torch.isfinite(batch[key]).all():
                    raise FloatingPointError(f"Non-finite rollout {key}")
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            rollout_seconds = time.monotonic() - started
            update_started = time.monotonic()
            metrics = trainer.update(batch)
            if train_cfg.refresh_hidden_after_update:
                rebuilt = trainer.replay_hidden(batch)
                rebuilt *= (~reset_before)[None, :, None]
                metrics["hidden_refresh_rms"] = float((rebuilt - hidden).square().mean().sqrt())
                hidden = rebuilt
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.monotonic() - update_started
            metrics.update(
                iteration=trainer.iteration,
                total_steps=trainer.total_steps,
                beta=beta,
                teacher_fraction=selected_sum / train_cfg.rollout_steps,
                mean_step_reward=reward_sum / train_cfg.rollout_steps,
                completed_episodes=done_sum,
                mixed_rollout_success_increments=mixed_successes,
                mixed_rollout_object_out_of_reach=mixed_out_of_reach,
                iteration_seconds=time.monotonic() - started,
                rollout_seconds=rollout_seconds,
                update_seconds=update_seconds,
                process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                student_joint_noise_rms=(joint_error_sq / train_cfg.rollout_steps) ** 0.5,
                student_position_error_rms_m=(pose_error_sq / train_cfg.rollout_steps) ** 0.5,
                **teacher_sensor_audit,
                **audit_clean_state_observations(env.unwrapped, obs),
            )
            if torch.device(device).type == "cuda":
                metrics.update(
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                    cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                )
            with metrics_file.open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            progress_tmp = log_dir / "progress.tmp"
            progress_tmp.write_text(json.dumps(metrics, indent=2) + "\n")
            progress_tmp.replace(log_dir / "progress.json")
            for key, value in metrics.items():
                writer.add_scalar(f"distillation/{key}", value, trainer.iteration)
            for key, value in env_logs.items():
                writer.add_scalar(key, value, trainer.iteration)
            writer.flush()
            print(f"[DISTILL] {json.dumps(metrics)}", flush=True)
            if trainer.iteration == 1 or trainer.iteration % args.save_interval == 0 or requested["save"]:
                trainer.save(log_dir / f"student_{trainer.iteration}.pt", metadata, noise_cfg)
                requested["save"] = False
            if requested["stop"]:
                break
        trainer.save(log_dir / f"student_{trainer.iteration}.pt", metadata, noise_cfg)
        completion = {
            "completed": trainer.iteration >= args.max_iterations,
            "stop_reason": "max_iterations" if trainer.iteration >= args.max_iterations else "interrupted",
            "iteration": trainer.iteration,
            "total_steps": trainer.total_steps,
            "checkpoint": str(log_dir / f"student_{trainer.iteration}.pt"),
        }
        (log_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
        print(f"[DISTILL] saved completed_updates={trainer.iteration}; total_steps={trainer.total_steps}", flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if writer:
            writer.close()
        if env:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
