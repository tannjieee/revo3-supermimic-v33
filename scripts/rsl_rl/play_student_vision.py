"""Deploy the exported Revo3 student in simulation, with RGB pose feedback and video.

The official Newton camera uses the delivered Hikrobot calibration and renders
the full live scene. RGB is rectified before the frozen CNN; object ground truth
is used only for diagnostic overlays and errors, never as a policy pose input.
"""

import argparse
from pathlib import Path
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", type=Path, required=True)
parser.add_argument("--pose_checkpoint", type=Path, required=True)
parser.add_argument("--policy", type=Path)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--mode", choices=("baseline", "vision"), default="vision")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=53)
parser.add_argument("--video_env", type=int, default=0)
parser.add_argument("--video_stride", type=int, default=2)
parser.add_argument("--no_video", action="store_true")
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3-Hikrobot")
args = parse_runtime_args(parser)
if min(args.num_envs, args.num_steps, args.video_stride) <= 0 or not 0 <= args.video_env < args.num_envs:
    parser.error("Invalid environment, step or video settings")
for path in (args.student_checkpoint, args.pose_checkpoint):
    if not path.is_file():
        parser.error(f"Missing checkpoint: {path}")
args.enable_cameras = True  # RGB inference still needs the sensor when video is disabled.
app = SimulationRuntime(args)


def main():
    from dataclasses import asdict
    import hashlib
    import json
    import time
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    import viserdex.tasks  # noqa: F401
    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from viserdex.assets.objects import TARGET_OBJECT_CFG
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    from viserdex.distillation.config import StudentCfg, PerceptionNoiseCfg
    from viserdex.distillation.frames import STUDENT_POSE_FRAME, validate_pose_metadata
    from viserdex.distillation.model import BeliefStudent
    from viserdex.distillation.hikrobot_images import HikrobotImageProcessor, diagnostic_keypoints_uvd
    from viserdex.renderer.hikrobot_calibration import (
        DEFAULT_CALIBRATION_DIRECTORY, audit_calibrated_camera, load_calibration,
    )
    from viserdex.distillation.noise import quat_unique
    from viserdex.distillation.observations import StudentObservations
    from viserdex.distillation.teacher import configure_clean_state_observations, audit_clean_state_observations
    from viserdex.distillation.trainer import load_student_checkpoint
    from viserdex.distillation.vision import keypoints_to_frame_pose
    from viserdex.tasks.manipulation.inhand.pose_estimator.networks import PoseEstimatorNetwork
    from viserdex.tasks.manipulation.inhand.pose_estimator.transforms import get_transforms
    from viserdex.utils.student_video import StudentDashboard

    device = args.device or "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = load_student_checkpoint(args.student_checkpoint, device)
    frame_metadata = validate_pose_metadata(payload["metadata"])
    if payload["metadata"]["object"] != TARGET_OBJECT_CFG["name"]:
        raise ValueError("Student and simulation objects differ")
    if payload["metadata"]["joint_names"] != list(REVO3_JOINT_NAMES):
        raise ValueError("Joint ordering differs from the exported student")
    student_cfg = StudentCfg(**payload["student_cfg"])
    noise_cfg = PerceptionNoiseCfg(**payload["noise_cfg"])
    policy_path = args.policy or args.student_checkpoint.parent / "exported/student_policy.pt"
    policy = torch.jit.load(str(policy_path), map_location=device).eval()
    policy_info = json.loads(policy_path.with_suffix(".json").read_text())
    validate_pose_metadata(policy_info["metadata"])
    if policy_info["iteration"] != payload["iteration"]:
        raise ValueError("Export and checkpoint iterations differ")
    # Verify the exported controller is precisely the supplied student, even if
    # a deployment file was copied from another run with the same iteration.
    reference_policy = BeliefStudent(student_cfg).to(device).eval()
    reference_policy.load_state_dict(payload["student"], strict=True)
    probe = (
        torch.randn(2, student_cfg.proprio_dim, device=device),
        torch.randn(2, student_cfg.extero_dim, device=device),
        reference_policy.initial_state(2, device),
    )
    with torch.inference_mode():
        expected = reference_policy(*probe)
        actual = policy(*probe)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[2])
    student_step_dt = payload["metadata"]["step_dt"]
    del reference_policy, payload
    cnn = PoseEstimatorNetwork(27).to(device)
    cnn.load_state_dict(torch.load(args.pose_checkpoint, map_location=device, weights_only=True), strict=True)
    cnn.eval().requires_grad_(False)
    # Keep BinaryOpening active as in pose training/evaluation (p=1); no image augmentation.
    preprocess = get_transforms(["binary_opening"], device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    cfg = app.env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = device
    cfg.log_dir = str(output)
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    configure_clean_state_observations(cfg)
    env = writer = native_writer = None
    started = time.monotonic()
    trace = []
    errors = []
    diagnostics = []
    try:
        env = RslRlVecEnvWrapper(
            gym.make(
                args.task, cfg=cfg
            )
        )
        raw = env.unwrapped
        obs = env.get_observations()
        source_audit = audit_clean_state_observations(raw, obs)
        if abs(raw.step_dt - student_step_dt) > 1e-9:
            raise ValueError("Student checkpoint control period differs from the calibrated camera task")
        camera = raw.scene["hikrobot_camera"]
        camera.update(0., force_recompute=True)
        camera_audit = audit_calibrated_camera(camera, raw.scene["robot"], DEFAULT_CALIBRATION_DIRECTORY)
        intrinsics, _, _ = load_calibration(DEFAULT_CALIBRATION_DIRECTORY)
        image_processor = HikrobotImageProcessor(intrinsics, device)
        camera_k = image_processor.intrinsics
        points = torch.load(TARGET_OBJECT_CFG["points_path"], map_location=device, weights_only=True)
        reference = torch.cat((torch.zeros(1, 3, device=device), points))
        adapter = StudentObservations(raw, student_cfg, noise_cfg)
        hidden = torch.zeros(student_cfg.rnn_layers, args.num_envs, student_cfg.rnn_hidden, device=device)
        command = raw.command_manager.get_term("object_pose")
        previous_counter = command.command_counter.clone()
        successes = torch.zeros(args.num_envs, device=device)
        per_episode = torch.zeros_like(successes)
        resets = torch.zeros_like(successes)
        episode_successes, rewards = [], []
        terminations = {}
        dashboard = StudentDashboard(args.mode, policy_info["iteration"])
        manifest = {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "student_iteration": policy_info["iteration"],
            "policy": str(policy_path.resolve()),
            "student_sha256": hashlib.sha256(args.student_checkpoint.read_bytes()).hexdigest(),
            "pose_sha256": hashlib.sha256(args.pose_checkpoint.read_bytes()).hexdigest(),
            "noise": asdict(noise_cfg),
            "actual_visual_source": "hikrobot_full_scene_rgb_cnn" if args.mode == "vision" else "simulated_pose_noise",
            "visual_noise_added_to_cnn_pose": False,
            "joint_noise_enabled": noise_cfg.joint_enabled,
            "teacher_used": False,
            "camera": camera_audit,
            "image_processing": image_processor.metadata,
            "pose_checkpoint_viewpoint_verified": False,
            **frame_metadata,
            "step_dt": raw.step_dt,
            "cnn_preprocessing": ["Brown-Conrady rectification and full-frame resize", "binary_opening"],
            "cnn_stochastic_augmentation": False,
            "frame_alignment": "scene, RGB, prediction and policy input captured before the same env.step",
            "source_audit": source_audit,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if not args.no_video:
            options = dict(fps=1 / raw.step_dt / args.video_stride, codec="libx264", quality=8,
                           macro_block_size=16, pixelformat="yuv420p")
            writer = imageio.get_writer(str(output / "student.mp4"), **options)
            native_writer = imageio.get_writer(str(output / "camera_native.mp4"), **options)
        with torch.inference_mode(), (output / "observations.jsonl").open("w", buffering=1) as stream:
            for step in range(args.num_steps):
                if not app.is_running():
                    raise RuntimeError("Simulation stopped before evaluation completed")
                # Capture RGB belongs to the current state, produced by the previous env.step.
                native_rgb = camera.data.output["rgb"].torch[..., :3]
                pixels = image_processor(native_rgb)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
                inference_start = time.monotonic()
                cnn_input = preprocess(pixels)
                prediction = cnn(cnn_input)
                robot = raw.scene["robot"]
                rotation, position = keypoints_to_frame_pose(
                    prediction,
                    camera_k,
                    reference,
                    matrix_from_quat(camera.data.quat_w_ros.torch),
                    camera.data.pos_w.torch,
                    matrix_from_quat(robot.data.root_quat_w.torch),
                    robot.data.root_pos_w.torch,
                )
                quaternion = quat_unique(quat_from_matrix(rotation))
                if not torch.isfinite(prediction).all() or not torch.isfinite(quaternion).all():
                    raise FloatingPointError("Non-finite RGB pose estimate; no ground-truth fallback is allowed")
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.monotonic() - inference_start) * 1000
                prop, extero, target = adapter(
                    obs, estimated_pose=(position, quaternion) if args.mode == "vision" else None,
                    estimated_pose_frame=STUDENT_POSE_FRAME,
                )
                if args.mode == "vision":
                    torch.testing.assert_close(extero[:, :3], position, atol=0, rtol=0)
                    torch.testing.assert_close(extero[:, 3:7], quat_unique(quaternion), atol=1e-6, rtol=1e-6)
                action, next_hidden = policy(prop, extero, hidden)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Non-finite student action")

                def angle(a, b):
                    a, b = a.double(), b.double()
                    cosine = ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).abs().clamp(0, 1)
                    return (2 * torch.acos(cosine) * (180 / torch.pi)).float()

                cnn_pos_error = (position - target[:, :3]).norm(dim=-1) * 1000
                cnn_rot_error = angle(quaternion, target[:, 3:7])
                input_pos_error = (extero[:, :3] - target[:, :3]).norm(dim=-1) * 1000
                input_rot_error = angle(extero[:, 3:7], target[:, 3:7])
                errors.append(
                    torch.stack((cnn_pos_error, cnn_rot_error, input_pos_error, input_rot_error), -1).cpu().numpy()
                )
                i = args.video_env
                record = {
                    "step": step,
                    "time_s": step * raw.step_dt,
                    "env_id": i,
                    "successes": int(successes[i]),
                    "resets": int(resets[i]),
                    "extero": extero[i].tolist(),
                    "cnn_translation_mm": float(cnn_pos_error[i]),
                    "cnn_rotation_deg": float(cnn_rot_error[i]),
                    "input_translation_mm": float(input_pos_error[i]),
                    "input_rotation_deg": float(input_rot_error[i]),
                    "occlusion_fraction": None,  # Native full-scene RGB includes actual hand occlusion; no oracle mask.
                    "inference_ms": inference_ms,
                    "joint_error_rms": float((prop[i, :21] - obs["policy"][i, :21]).square().mean().sqrt()),
                }
                stream.write(json.dumps(record) + "\n")
                trace.append(
                    {
                        key: value[i].cpu().numpy().copy()
                        for key, value in {
                            "proprio": prop,
                            "extero": extero,
                            "actions": action,
                            "cnn_keypoints_uvd": prediction,
                            "ground_truth_pose": target[:, :7],
                        }.items()
                    }
                )
                diagnostics.append(inference_ms)
                if writer and step % args.video_stride == 0:
                    scene = native_rgb[i].cpu().numpy().copy()
                    obj = raw.scene["object"]
                    truth = diagnostic_keypoints_uvd(
                        reference, obj.data.root_pos_w.torch, matrix_from_quat(obj.data.root_quat_w.torch),
                        camera.data.pos_w.torch, matrix_from_quat(camera.data.quat_w_ros.torch),
                        camera_k, image_processor.image_size,
                    )
                    rectified = (pixels[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    processed = (cnn_input[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    frame = dashboard.frame(scene, rectified, processed, prediction[i].cpu().numpy(), truth[i].cpu().numpy(), record)
                    writer.append_data(frame)
                    native_writer.append_data(scene)
                    if step in (0, args.num_steps // 2 // args.video_stride * args.video_stride):
                        Image.fromarray(frame).save(output / f"preview_{step:04d}.png")
                        Image.fromarray(scene).save(output / f"camera_native_{step:04d}.png")
                        Image.fromarray(processed).save(output / f"cnn_input_{step:04d}.png")
                obs, reward, dones, _ = env.step(action)
                dones = dones.bool()
                counter = command.command_counter
                episode_successes.extend(per_episode[dones].cpu().tolist())
                per_episode[dones] = 0
                increments = torch.where(dones, 0, counter - previous_counter).clamp_min(0)
                per_episode += increments
                successes += increments
                resets += dones
                previous_counter = counter.clone()
                rewards.append(float(reward.mean()))
                for name in raw.termination_manager.active_terms:
                    terminations[name] = terminations.get(name, 0) + int(raw.termination_manager.get_term(name).sum())
                hidden = next_hidden * (~dones)[None, :, None]
                adapter.record_action(action, dones)
                if (step + 1) % 100 == 0 or step + 1 == args.num_steps:
                    progress = {
                        "step": step + 1,
                        "num_steps": args.num_steps,
                        "mode": args.mode,
                        "successes": int(successes.sum()),
                        "completed_episodes": int(resets.sum()),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    print("[STUDENT_VISION] " + json.dumps(progress), flush=True)
        values = np.concatenate(errors)
        metric_names = (
            "cnn_translation_mm",
            "cnn_rotation_deg",
            "input_translation_mm",
            "input_rotation_deg",
        )
        result = {
            "completed": True,
            "mode": args.mode,
            "num_envs": args.num_envs,
            "steps": args.num_steps,
            "seed": args.seed,
            "observed_success_increments": int(successes.sum()),
            "completed_episodes": int(resets.sum()),
            "mean_completed_episode_successes": float(np.mean(episode_successes)) if episode_successes else None,
            "termination_counts": terminations,
            "mean_step_reward": float(np.mean(rewards)),
            "cnn_inference_mean_ms": float(np.mean(diagnostics)),
            "elapsed_seconds": time.monotonic() - started,
            "mean": dict(zip(metric_names, values.mean(0).tolist())),
            "p95": dict(zip(metric_names, np.quantile(values, 0.95, axis=0).tolist())),
            "video_env_successes": int(successes[args.video_env]),
            "video_env_resets": int(resets[args.video_env]),
            "teacher_used": False,
            "ground_truth_fallback_count": 0,
            "external_pose_matches_student_input": args.mode == "vision",
        }
        np.savez_compressed(output / "observations.npz", **{k: np.stack([r[k] for r in trace]) for k in trace[0]})
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[STUDENT_VISION_COMPLETE] " + json.dumps(result), flush=True)
    finally:
        if writer:
            writer.close()
        if native_writer:
            native_writer.close()
        if env:
            env.close()


if __name__ == "__main__":
    import sys

    code = 0
    try:
        main()
    except BaseException:
        code = 1
        import traceback

        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
