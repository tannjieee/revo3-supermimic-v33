"""Collect teacher-driven 3DGS images with physically aligned Revo3 hand occlusion."""

import argparse
from pathlib import Path
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.utils import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-ViserDex-Repose-Revo3-Estimator")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--hikrobot_calibration", type=Path, help="Use original GS+hand-depth images at the calibrated rectified Hikrobot viewpoint; pass the base Revo3 task.")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--samples", type=int, default=300000)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--burn_in", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_occlusion", type=float, default=0.02)
parser.add_argument("--max_occlusion", type=float, default=0.95)
parser.add_argument("--min_visible_pixels", type=int, default=32)
cli_args.add_rsl_rl_args(parser)
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3")
args = parse_runtime_args(parser)
if min(args.samples, args.num_envs, args.max_steps, args.stride) <= 0:
    parser.error("Counts and stride must be positive")
prepared_teacher = binding = None
binding_info = {}
if args.hikrobot_calibration:
    from viserdex.assets.objects import TARGET_OBJECT_CFG
    from viserdex.distillation.environment import SavedEnvironmentBinding
    from viserdex.distillation.teacher import teacher_metadata
    from viserdex.renderer.hikrobot_occlusion import configure_original_capture

    if args.task != "Isaac-ViserDex-Repose-Revo3":
        parser.error("Calibrated capture requires --task Isaac-ViserDex-Repose-Revo3; pose sensors are attached after saved environment binding")
    prepared_teacher = teacher_metadata(args.teacher_checkpoint, TARGET_OBJECT_CFG["name"])
    binding = SavedEnvironmentBinding(prepared_teacher["environment_path"], prepared_teacher["environment_sha256"])


def configure(cfg):
    if binding:
        binding.configure(cfg, args)
        binding_info.update(binding.verify(cfg))
        configure_original_capture(cfg, args.hikrobot_calibration)


app = SimulationRuntime(args, configure=configure)


def main():
    import hashlib
    import json
    import signal
    import time
    from pathlib import Path
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import create_rsl_rl_runner
    import viserdex.tasks  # noqa: F401
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = Path(args.teacher_checkpoint).resolve()
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    cfg = app.env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.log_dir = str(output)
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    pose_cfg = cfg.observations.pose.image_features.params["pose_estimator_cfg"]
    pose_cfg.collect_only = True
    pose_cfg.train = False
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
    raw = env.unwrapped
    robot = raw.scene["robot"]
    obj = raw.scene["object"]
    term = _get_observation_term_func(raw.observation_manager, "pose", "image_features")
    assert term._hand_depth_camera is not None, "Collection requires live Revo3 hand depth"
    hand = term._hand_depth_camera
    torch.testing.assert_close(hand.data.intrinsic_matrices.torch, term.camera.data.intrinsic_matrices.torch)
    torch.testing.assert_close(hand.data.pos_w.torch, term.camera_pos, atol=1e-5, rtol=1e-5)
    # q and -q encode the same camera orientation.
    orientation_dot = (hand.data.quat_w_ros.torch * term.camera_quat_ros).sum(-1).abs()
    torch.testing.assert_close(orientation_dot, torch.ones_like(orientation_dot), atol=1e-5, rtol=1e-5)
    joint_ids, names = robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
    assert names == list(REVO3_JOINT_NAMES)
    if prepared_teacher:
        from viserdex.distillation.teacher import load_teacher
        policy = load_teacher(prepared_teacher, env.get_observations(), raw.device)
    else:
        agent = cli_args.parse_rsl_rl_cfg(args.task, args)
        runner = create_rsl_rl_runner(env, agent, log_dir=None)
        runner.load(str(teacher))
        policy = runner.get_inference_policy(device=raw.device)
    dump_yaml(str(output / "env.yaml"), cfg)
    manifest = {
        "format_version": 1,
        "object": "RSLCube",
        "hand": "Revo3",
        "seed": args.seed,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
        "method": "ViserDex III-C2: live articulated hand ray depth < 3DGS expected metric depth",
        "paper": "https://arxiv.org/html/2604.11138v1#S3.SS3.SSS2",
        "joint_names": names,
        "camera_intrinsics": hand.data.intrinsic_matrices.torch[0].cpu().tolist(),
        "camera_position_env": (hand.data.pos_w.torch[0] - raw.scene.env_origins[0]).cpu().tolist(),
        "camera_quaternion_xyzw_ros": hand.data.quat_w_ros.torch[0].cpu().tolist(),
        "object_keypoints": torch.cat((torch.zeros(1, 3, device=raw.device), term.object_frame_pts[0])).cpu().tolist(),
        "hand_meshes_per_env": hand._num_meshes_per_env,
        "completed": False,
    }
    if args.hikrobot_calibration:
        from viserdex.renderer.hikrobot_occlusion import audit_original_capture
        manifest["hikrobot_camera"] = audit_original_capture(term, hand, robot, args.hikrobot_calibration)
        manifest["environment_binding"] = binding_info
        manifest["arguments"]["hikrobot_calibration"] = str(args.hikrobot_calibration.resolve())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    obs = env.get_observations()
    episode = torch.zeros(args.num_envs, dtype=torch.int64, device=raw.device)
    env_ids = torch.arange(args.num_envs, device=raw.device)
    count = 0
    rejected = 0
    histogram = np.zeros(6, dtype=np.int64)
    bins = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.00001])
    preview_rows = []
    stop = {"requested": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    started = time.monotonic()
    with h5py.File(output / "samples.h5", "w") as store:
        datasets = {}
        for step in range(args.max_steps):
            # The privileged teacher does not consume image features. Render
            # only frames that will be considered for storage.
            term.capture_enabled = step >= args.burn_in and step % args.stride == 0
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                episode += dones.long()
            if step >= args.burn_in and step % args.stride == 0:
                fraction = term.occlusion_fraction
                accept = (
                    (fraction >= args.min_occlusion)
                    & (fraction <= args.max_occlusion)
                    & (term.visible_pixels >= args.min_visible_pixels)
                    & ~dones.bool()
                )
                ids = accept.nonzero().flatten()[: args.samples - count]
                rejected += int((~accept).sum())
                if len(ids):
                    arrays = {
                        "rgb": term.masked_rgb[ids].permute(0, 2, 3, 1).cpu().numpy(),
                        "rgb_unoccluded": term.unoccluded_rgb[ids].permute(0, 2, 3, 1).cpu().numpy(),
                        "hand_depth": term.hand_depth[ids, 0].cpu().numpy().astype(np.float16),
                        "object_depth": term.object_depth[ids, 0].cpu().numpy().astype(np.float16),
                        "keypoints_uvd": term.gt_input[ids].cpu().numpy(),
                        "object_pose": torch.cat(
                            (
                                obj.data.root_pos_w.torch[ids] - raw.scene.env_origins[ids],
                                obj.data.root_quat_w.torch[ids],
                            ),
                            dim=-1,
                        )
                        .cpu()
                        .numpy(),
                        "joint_pos": robot.data.joint_pos.torch[ids][:, joint_ids].cpu().numpy(),
                        "occlusion_fraction": fraction[ids].cpu().numpy(),
                        "visible_pixels": term.visible_pixels[ids].cpu().numpy(),
                        "env_id": env_ids[ids].cpu().numpy(),
                        "episode_id": episode[ids].cpu().numpy(),
                        "sim_step": np.full(len(ids), step, dtype=np.int64),
                    }
                    if not datasets:
                        for key, value in arrays.items():
                            datasets[key] = store.create_dataset(
                                key,
                                shape=(0, *value.shape[1:]),
                                maxshape=(None, *value.shape[1:]),
                                chunks=(1 if value.ndim >= 3 else 256, *value.shape[1:]),
                                compression="lzf",
                                dtype=value.dtype,
                            )
                    for key, value in arrays.items():
                        datasets[key].resize(count + len(ids), axis=0)
                        datasets[key][count : count + len(ids)] = value
                    count += len(ids)
                    histogram += np.histogram(arrays["occlusion_fraction"], bins)[0]
                    # Spread the preview across different trajectories and simulation times.
                    if len(preview_rows) < 12 and step >= args.burn_in + len(preview_rows) * 8:
                        j = int(np.argmax(arrays["occlusion_fraction"]))
                        preview_rows.append({key: value[j].copy() for key, value in arrays.items()})
            if step % 100 == 0 or count >= args.samples or stop["requested"]:
                progress = {
                    "step": step,
                    "samples": count,
                    "target": args.samples,
                    "occlusion_histogram": histogram.tolist(),
                    "elapsed_s": time.monotonic() - started,
                }
                (output / "progress.json").write_text(json.dumps(progress, indent=2))
                print("COLLECTION", json.dumps(progress), flush=True)
                store.flush()
            if count >= args.samples or stop["requested"]:
                break
        manifest.update(
            completed=count >= args.samples,
            samples=count,
            rejected_frames=rejected,
            steps=step + 1,
            elapsed_s=time.monotonic() - started,
            occlusion_histogram=histogram.tolist(),
            occlusion_bins=bins.tolist(),
        )
        store.attrs["manifest"] = json.dumps(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if preview_rows:
        canvas = Image.new("RGB", (480, len(preview_rows) * 142), "white")
        draw = ImageDraw.Draw(canvas)
        for row, sample in enumerate(preview_rows):
            hand_img = np.clip((0.8 - sample["hand_depth"].astype(np.float32)) / 0.6, 0, 1)
            hand_rgb = np.repeat((hand_img * 255).astype(np.uint8)[..., None], 3, axis=-1)
            mask = np.any(sample["rgb"] != sample["rgb_unoccluded"], axis=-1)
            mask_rgb = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=-1)
            for col, (title, pixels) in enumerate(
                zip(
                    ("3DGS", "Revo3 depth", "occluded pixels", "training RGB"),
                    (sample["rgb_unoccluded"], hand_rgb, mask_rgb, sample["rgb"]),
                )
            ):
                canvas.paste(Image.fromarray(pixels), (col * 120, row * 142 + 22))
                draw.text((col * 120 + 2, row * 142 + 2), title, fill="black")
        canvas.save(output / "preview.png")
    env.close()
    if not manifest["completed"]:
        raise RuntimeError(f"Collection incomplete: {count}/{args.samples}; see {output}/manifest.json")
    print("COLLECTION_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    import sys
    import traceback

    code = 0
    try:
        main()
    except BaseException:
        code = 1
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
