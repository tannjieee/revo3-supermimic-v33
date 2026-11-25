"""Record a continuous actor-only Newton rollout using the saved training contract."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import copy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "viserdex"))
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID, TransitionRecorderCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--env_config", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--seed", type=int, default=42)
add_runtime_args(parser, TASK_ID)
args = parse_runtime_args(parser)
if args.num_envs != 1 or args.steps < 1:
    parser.error("This recorder displays one continuous environment; steps must be positive")
binding = SavedEnvironmentBinding(args.env_config)


def configure(cfg):
    binding.configure(cfg, args)
    configure_clean_state_observations(cfg)
    cfg.recorders = TransitionRecorderCfg()
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed
    # Operational contact capacity used by the previously verified single-world playbacks.
    cfg.sim.physics.solver_cfg.nconmax = max(2048, cfg.sim.physics.solver_cfg.nconmax)
    view = NewtonGLVisualizerCfg(
        headless=True, window_width=1280, window_height=960,
        eye=(-0.22, -0.24, 0.78), lookat=(0.02, 0.0, 0.565),
        streaming_view=False, enable_picking=False,
        background_color=(0.08, 0.11, 0.16),
    )
    cfg.sim.visualizer_cfgs = [view]
    cfg.sim.default_visualizer_cfg = view
    binding.verify(cfg)


app = SimulationRuntime(args, configure=configure)
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.utils.math import quat_error_magnitude
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import Config
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.simulation import Adapter


def main():
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert payload["version"] == 13, "Legacy checkpoints need their original source"
    assert payload["metadata"]["environment_contract"] == binding.contract_sha
    assert not payload["metadata"].get("smoke", False)
    assert not payload["metadata"].get("check_joint_update", False)
    assert payload["metadata"]["task_id"] == TASK_ID
    cfg = Config(**payload["config"])
    noise = PerceptionNoiseCfg(**payload["noise_config"])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    writer = None
    try:
        clean, _ = env.reset()
        adapter = Adapter(env, noise, cfg.history)
        adapter.audit_observation(clean)
        assert adapter.geometry.digest == payload["metadata"]["geometry_digest"]
        expected_schema = {k: tuple(v) for k, v in payload["metadata"]["context_schema"].items()}
        assert {k: tuple(v) for k, v in adapter.context_schema.items()} == expected_schema
        actor = Actor(cfg).to(env.device).eval()
        actor.load_state_dict(payload["actor"], strict=True)
        # Visual-only prototypes never enter the physics scene or observation state.
        # Newton's USD marker loader reads one mesh per USD, so retain all six faces
        # in six separate prototypes, at the same pose, with the original textures.
        from pxr import Usd, UsdGeom, UsdUtils, Gf, Sdf
        asset_path = Path(env.cfg.scene.object.spawn.usd_path)
        source_stage = Usd.Stage.Open(str(asset_path))
        mesh_paths = [str(p.GetPath()) for p in source_stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(mesh_paths) == 6
        markers = {}
        for i, mesh_path in enumerate(mesh_paths):
            layer = source_stage.Flatten()
            UsdUtils.ModifyAssetPaths(layer, lambda p: str((asset_path.parent / p).resolve()) if p and not p.startswith('/') else p)
            stage = Usd.Stage.Open(layer)
            # Keep the texture untinted in the marker renderer, as in the
            # physics object's renderer. These are recording-only USD copies.
            for prim in stage.Traverse():
                color = prim.GetAttribute('inputs:diffuseColor')
                if color:
                    color.Set(Gf.Vec3f(1.0))
                if prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(prim).CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(1.0)])
            for other in mesh_paths:
                if other != mesh_path:
                    stage.RemovePrim(other)
            stage.RemovePrim('/RSLSphere65/collision')
            face_path = args.output_dir / f'target_face_{i}.usda'
            stage.GetRootLayer().Export(str(face_path.resolve()))
            prototype = copy.deepcopy(env.cfg.commands.object_pose.goal_pose_visualizer_cfg.markers['goal'])
            prototype.usd_path = str(face_path.resolve())
            prototype.scale = (1.0, 1.0, 1.0)
            prototype.visual_material = PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
            markers[f'face_{i}'] = prototype
        target_marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path='/Visuals/RecordingTarget', markers=markers))
        command = env.command_manager.get_term('object_pose')
        marker_pos = command.pos_command_w[:1] + torch.tensor([[0.14, -0.14, 0.05]], device=env.device)
        def update_target():
            target_marker.visualize(translations=marker_pos.expand(6, -1),
                                    orientations=command.quat_command_w[:1].expand(6, -1),
                                    marker_indices=list(range(6)))
        update_target()
        viz = next(v for v in env.sim.visualizers if v.cfg.visualizer_type == "newton_gl")
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        eye = tuple(origin + np.array([-0.22, -0.24, 0.78]))
        target = tuple(origin + np.array([0.02, 0.0, 0.565]))
        env.sim.set_camera_view(eye, target)
        for _ in range(3):
            env.sim.render()
            pixels = viz.render_rgb_array()
        Image.fromarray(pixels[..., :3]).save(args.output_dir / "initial_view.png")
        dump_yaml(str(args.output_dir / "env.yaml"), env.cfg)
        iteration = payload["iteration"]
        manifest = dict(
            checkpoint=str(args.checkpoint), iteration=iteration,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            environment=binding.verify(env.cfg), geometry_digest=adapter.geometry.digest,
            noise_config=asdict(noise), seed=args.seed, steps=args.steps,
            fps=1 / env.step_dt, policy_hz=1 / env.step_dt,
            policy_input="10 x 124 noisy observations", actor_amp=cfg.amp,
            exploration_action_noise=False, camera_eye=eye, camera_target=target,
            continuous=True, selected_environment=0,
            target_visualization='six original textured faces, world goal quaternion, visual-only, scale 1',
            target_marker_position=marker_pos[0].cpu().tolist(),
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        writer = imageio.get_writer(
            str(args.output_dir / "rollout.mp4"), fps=1 / env.step_dt,
            codec="libx264", quality=None, macro_block_size=16,
            ffmpeg_params=["-crf", "23", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"],
        )
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 24)
        small = ImageFont.truetype(font_path, 19)
        successes = drops = resets = timeouts = episode_successes = 0
        episodes, rows, actions, physical, target_quats, current_quats = [], [], [], [], [], []
        with torch.no_grad():
            for step in range(args.steps):
                action = actor(adapter.history)
                row, _, _, _ = adapter.step(action)
                assert torch.isfinite(row["next_physical"]).all()
                reached = bool(((adapter.captured_orientation_error < 0.1) & ~row["done"])[0])
                dropped = bool(((row["next_physical"][:, 42:45].norm(dim=-1) > 0.3) & row["terminated"])[0])
                done = bool(row["done"][0])
                timeout = done and not bool(row["terminated"][0])
                successes += int(reached)
                episode_successes += int(reached)
                drops += int(dropped)
                timeouts += int(timeout)
                if done:
                    resets += 1
                    episodes.append(dict(end_step=step + 1, goals=episode_successes, drop=dropped, timeout=timeout))
                    episode_successes = 0
                error_deg = float(adapter.captured_orientation_error[0]) * 180 / np.pi
                # Goal resampling has already occurred; display the NEW live goal
                # and its error against the visible post-step object orientation.
                current_quat = env.scene['object'].data.root_quat_w.torch[:1]
                live_error_deg = float(quat_error_magnitude(current_quat, command.quat_command_w[:1])[0]) * 180 / np.pi
                update_target()
                env.sim.render()
                record = dict(step=step + 1, time_s=(step + 1) * env.step_dt,
                              reached=reached, dropped=dropped, reset=done,
                              successes=successes, drops=drops, resets=resets,
                              orientation_error_deg=error_deg, displayed_goal_error_deg=live_error_deg)
                rows.append(record)
                actions.append(action[0].cpu().numpy())
                physical.append(row["next_physical"][0].cpu().numpy())
                target_quats.append(command.quat_command_w[0].cpu().numpy().copy())
                current_quats.append(current_quat[0].cpu().numpy().copy())
                pixels = viz.render_rgb_array()[..., :3]
                frame = Image.fromarray(pixels)
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 116), fill="#101820")
                draw.text((20, 12), f"Revo3 | RSLSphere65 | SuperMimic | checkpoint {iteration}", font=font, fill="white")
                draw.text((20, 48), f"t={record['time_s']:05.2f}s   Goals: {successes}   Drops: {drops}   Resets: {resets}", font=font, fill="#6ae0d0")
                draw.text((20, 84), f"{1/env.step_dt:g} Hz | 1x speed | seed {args.seed} | noisy observations | continuous rollout", font=small, fill="#c5d4df")
                draw.text((470, 690), 'CURRENT / in-hand', font=font, fill='white')
                draw.text((880, 260), 'TARGET orientation', font=font, fill='#6ae0d0')
                draw.text((880, 295), f'Angle error: {live_error_deg:5.1f} deg', font=small, fill='white')
                draw.text((880, 535), 'Same world axes / scale 1:1', font=small, fill='#c5d4df')
                draw.text((880, 565), 'New goal after < 5.7 deg', font=small, fill='#c5d4df')
                writer.append_data(np.asarray(frame))
                if step in (0, 299, 599, 899, 1199, 1499, args.steps - 1):
                    frame.save(args.output_dir / f"frame_{step+1:04d}.jpg")
                if (step + 1) % 300 == 0:
                    print(json.dumps(record), flush=True)
        writer.close()
        writer = None
        np.savez_compressed(args.output_dir / "trajectory.npz", actions=actions, physical=physical,
                            target_quaternions_xyzw=target_quats, current_quaternions_xyzw=current_quats)
        (args.output_dir / "events.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        summary = dict(iteration=iteration, seconds=args.steps * env.step_dt, frames=args.steps,
                       successes=successes, drops=drops, resets=resets, timeouts=timeouts,
                       completed_episodes=episodes, unfinished_episode_goals=episode_successes,
                       parity_max_errors=adapter.max_errors)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("RECORDING_COMPLETE", json.dumps(summary), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
