"""Exercise camera buffers, projection and optional trained pose inference with real assets."""

import argparse
import json
from pathlib import Path
import traceback

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-ViserDex-Repose-Revo3-Estimator")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--pose_run", default=None, help="Pose run directory containing pose_estimator/cnn_*.pth.")
parser.add_argument("--output_dir", required=True)
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3-Estimator")
args = parse_runtime_args(parser)
args.enable_cameras = True
app = SimulationRuntime(args)


def main():
    import gymnasium as gym
    import torch
    import warp as wp
    from PIL import Image
    from isaaclab_tasks.utils import parse_env_cfg
    import viserdex.tasks  # noqa: F401
    from viserdex.renderer import GaussianSplatCamera
    from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func

    cfg = app.env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.log_dir = str(Path(args.pose_run or args.output_dir).resolve())
    if args.pose_run:
        assert list(Path(cfg.log_dir, "pose_estimator").glob("cnn_*.pth")), "Missing estimator checkpoint"
    for name in ("pose", "gaussian_features"):
        group = getattr(cfg.observations, name, None)
        if group is not None:
            group.image_features.params["pose_estimator_cfg"].train = False
            group.image_features.params["pose_estimator_cfg"].collect_only = args.pose_run is None
            group.image_features.params["debug_vis"] = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg.commands.object_pose.debug_vis = False
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.visualizer else None)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            for _ in range(args.steps):
                obs, rewards, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            result = {
                "task": args.task,
                "steps": args.steps,
                "num_envs": args.num_envs,
                "cameras": {},
                "trained_pose_inference": args.pose_run is not None,
            }
            for name, camera in env.unwrapped.scene.sensors.items():
                depth = camera.data.output["distance_to_image_plane"].torch
                if name == "hand_depth_camera":
                    assert depth.shape == (args.num_envs, 120, 120, 1)
                    assert (torch.isfinite(depth) & (depth > 0)).any()
                    result["cameras"][name] = {"shape": list(depth.shape), "dynamic_hand_depth": True}
                    continue
                rgb = camera.data.output["rgb"].torch
                assert rgb.dtype == torch.uint8 and rgb.shape[0] == args.num_envs
                assert rgb.float().std() > 1, f"Blank image from {name}"
                # RTX uses +inf for background; valid foreground depths must exist.
                assert not torch.isnan(depth).any() and not torch.isneginf(depth).any()
                assert (torch.isfinite(depth) & (depth > 0)).any()
                Image.fromarray(rgb[0, ..., :3].cpu().numpy()).save(output / f"{name}.png")
                result["cameras"][name] = {"shape": list(rgb.shape), "std": rgb.float().std().item()}
                if isinstance(camera, GaussianSplatCamera) and args.num_envs > 1:
                    before_rgb = rgb[1:].clone()
                    before_frames = camera._frame[1:].clone()
                    mask = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.bool)
                    mask[0] = True
                    camera.reset(env_mask=wp.from_torch(mask))
                    camera.data  # update only the invalidated environment
                    torch.testing.assert_close(camera.data.output["rgb"].torch[1:], before_rgb)
                    torch.testing.assert_close(camera._frame[1:], before_frames)
                    result["partial_sensor_reset"] = True
            term = _get_observation_term_func(env.unwrapped.observation_manager, "pose", "image_features")
            hand = term._hand_depth_camera
            assert hand is not None
            torch.testing.assert_close(hand.data.intrinsic_matrices.torch, term.camera.data.intrinsic_matrices.torch)
            torch.testing.assert_close(hand.data.pos_w.torch, term.camera_pos, atol=1e-5, rtol=1e-5)
            dots = (hand.data.quat_w_ros.torch * term.camera_quat_ros).sum(-1).abs()
            torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=1e-5)
            assert ((term.occlusion_fraction >= 0) & (term.occlusion_fraction <= 1)).all()
            result["camera_alignment"] = True
            result["occlusion_fraction"] = term.occlusion_fraction.tolist()
            Image.fromarray(term.masked_rgb[0].permute(1, 2, 0).cpu().numpy()).save(output / "masked_rgb.png")
            if args.visualizer:
                visualizer = next(v for v in env.unwrapped.sim.visualizers if hasattr(v, "render_rgb_array"))
                if visualizer.cfg.visualizer_type == "kit":
                    assert visualizer.cfg.origin_type == "env" and visualizer.cfg.origin_env_index == 0
                origin = env.unwrapped.scene.env_origins[0].tolist()
                eye = tuple(a + b for a, b in zip(origin, cfg.sim.default_visualizer_cfg.eye))
                lookat = tuple(a + b for a, b in zip(origin, cfg.sim.default_visualizer_cfg.lookat))
                visualizer.set_camera_view(eye, lookat)
                result["viewer_env0_origin"] = origin
                for _ in range(10):
                    env.unwrapped.sim.render()
                    frame = visualizer.render_rgb_array()
                assert frame is not None and frame.ndim == 3
                assert frame.reshape(-1, frame.shape[-1]).std(axis=0).max() > 1
                Image.fromarray(frame[..., :3]).save(output / "scene.png")
                result["visualizer_capture"] = list(frame.shape)
            points = term.gt_keypoints.reshape(args.num_envs, -1, 3).clone()
            projected = term._project_points_to_image(points)
            recovered = term._unproject_points_to_camera_frame(torch.cat((projected, points[..., 2:3]), dim=-1))
            torch.testing.assert_close(recovered, points, atol=1e-5, rtol=1e-4)
            result["projection_round_trip"] = True
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("VISION_SMOKE_OK:", json.dumps(result))
    finally:
        env.close()


if __name__ == "__main__":
    import sys

    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
