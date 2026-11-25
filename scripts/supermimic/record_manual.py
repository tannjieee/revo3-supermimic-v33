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
from viserdex.supermimic.simulation import TransitionRecorderCfg
from viserdex.supermimic.manual_convex import TASK_ID
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--env_config", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_velocity_deg_s", type=float, help="Explicit playback experiment; bounds commanded target speed only")
parser.add_argument("--target_filter_tau_s", type=float, default=0.)
add_runtime_args(parser, TASK_ID)
args = parse_runtime_args(parser)
if args.num_envs != 1 or args.steps < 1:
    parser.error("This recorder displays one continuous environment; steps must be positive")
if args.target_filter_tau_s and args.target_velocity_deg_s is None:
    parser.error('Target smoothing requires an explicit velocity limit')
binding = SavedEnvironmentBinding(args.env_config)


def configure(cfg):
    revision = binding.saved['actions']['joint_pos'].get('limit_contract', 'legacy')
    if revision in ('urdf_little_v2', 'hardware_center25_v3'):
        from viserdex.supermimic.revised_contract import configure_environment
        configure_environment(cfg, revision)
    from viserdex.supermimic.position_boundary import prepare_saved_boundary
    prepare_saved_boundary(cfg, binding.saved)
    binding.configure(cfg, args)
    configure_clean_state_observations(cfg)
    cfg.recorders = TransitionRecorderCfg()
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed
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
from viserdex.supermimic.spatial_contract import drop_distance
from viserdex.supermimic.networks import Actor
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.object_variant import object_metadata
from viserdex.supermimic.recording_trajectory import trajectory_contract, trajectory_snapshot


def main():
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from viserdex.supermimic.trainer import checkpoint_version
    assert payload["version"] == checkpoint_version(Config(**payload["config"])), "Checkpoint version mismatch"
    assert payload["metadata"]["environment_contract"] == binding.contract_sha
    assert not payload["metadata"].get("smoke", False)
    assert not payload["metadata"].get("check_joint_update", False)
    assert payload["metadata"]["task_id"] == TASK_ID
    cfg = Config(**payload["config"])
    from viserdex.supermimic.position_boundary import verify_policy_boundary
    verify_policy_boundary(cfg, app.cfg)
    noise = PerceptionNoiseCfg(**payload["noise_config"])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    writer = None
    try:
        # Always evaluate recorded videos on full 180 deg SO(3) goals, never curriculum-bounded goals.
        eval_command = env.command_manager.get_term('object_pose')
        if hasattr(eval_command, 'set_curriculum_max_angle'):
            eval_command.set_curriculum_max_angle(float(np.pi))
        clean, _ = env.reset()
        adapter = Adapter(env, noise, cfg.history, policy_cfg=cfg)
        assert adapter.identified_hand_audit == payload["metadata"]["identified_hand"]
        assert adapter.context_manifest == payload["metadata"]["context_manifest"]
        adapter.audit_observation(clean)
        assert adapter.geometry.digest == payload["metadata"]["geometry_digest"]
        expected_schema = {k: tuple(v) for k, v in payload["metadata"]["context_schema"].items()}
        assert {k: tuple(v) for k, v in adapter.context_schema.items()} == expected_schema
        actor = Actor(cfg).to(env.device).eval()
        actor.load_state_dict(payload["actor"], strict=True)
        asset = object_metadata(env.cfg.scene.object.spawn.usd_path)
        if 'object_asset' in payload['metadata']:
            assert asset == payload['metadata']['object_asset'], 'Checkpoint object asset differs'
        dimensions_mm = np.ptp(np.asarray(adapter.geometry_spec['convex_vertices']), axis=0)*1000
        shaper = None
        if args.target_velocity_deg_s is not None:
            from viserdex.supermimic.target_shaping import TargetShaper
            shaper = TargetShaper(adapter.limits, env.step_dt, args.target_velocity_deg_s, args.target_filter_tau_s)
            shaper.reset(adapter.state[:, :21] - adapter.state.new_tensor(cfg.encoder_bias_rad))
        # Independent visual-only target: retain the original textured mesh.
        from pxr import Usd, UsdUtils, Sdf
        from isaaclab_newton.physics import NewtonManager
        asset_path = Path(env.cfg.scene.object.spawn.usd_path)
        stage = Usd.Stage.Open(str(asset_path))
        layer = stage.Flatten()
        UsdUtils.ModifyAssetPaths(layer, lambda p: str((asset_path.parent / p).resolve()) if p and not p.startswith('/') else p)
        stage = Usd.Stage.Open(layer)
        stage.RemovePrim('/Object/Collision')
        root = stage.GetDefaultPrim()
        root.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([]))
        for name in list(root.GetPropertyNames()):
            if name.startswith(('physics:', 'physxRigidBody:')):
                root.RemoveProperty(name)
        # Newton's marker loader reads one mesh/material and ignores material
        # subsets. Render each authored subset as its own visual-only marker.
        from pxr import UsdGeom, UsdShade
        mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
        mesh = UsdGeom.Mesh(mesh_prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        assert np.all(counts == 3)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        subsets = [UsdGeom.Subset(p) for p in mesh_prim.GetChildren() if p.IsA(UsdGeom.Subset)]
        subset_names = {p.GetPrim().GetName() for p in subsets}
        assert subset_names in (
            {'yellow_body', 'black_pips'},
            {'yellow_body', 'midnight_green_11501'},
        ), f"Unsupported object material subsets: {subset_names}"
        partition = np.concatenate([np.asarray(p.GetIndicesAttr().Get()) for p in subsets])
        assert np.array_equal(np.sort(partition), np.arange(len(faces)))
        target_markers = []
        for subset in subsets:
            name = subset.GetPrim().GetName()
            indices = np.asarray(subset.GetIndicesAttr().Get())
            material_path = subset.GetPrim().GetRelationship('material:binding').GetTargets()[0]
            part_layer = Sdf.Layer.CreateAnonymous()
            part_layer.TransferContent(stage.GetRootLayer())
            part_stage = Usd.Stage.Open(part_layer)
            part_mesh = UsdGeom.Mesh(part_stage.GetPrimAtPath(mesh_prim.GetPath()))
            for child in list(part_mesh.GetPrim().GetChildren()):
                part_stage.RemovePrim(child.GetPath())
            part_mesh.GetFaceVertexCountsAttr().Set([3] * len(indices))
            part_mesh.GetFaceVertexIndicesAttr().Set(faces[indices].ravel().tolist())
            UsdShade.MaterialBindingAPI(part_mesh.GetPrim()).Bind(UsdShade.Material(part_stage.GetPrimAtPath(material_path)))
            target_path = args.output_dir / f'target_{name}.usda'
            part_stage.GetRootLayer().Export(str(target_path.resolve()))
            prototype = copy.deepcopy(env.cfg.commands.object_pose.goal_pose_visualizer_cfg.markers['goal'])
            prototype.usd_path = str(target_path.resolve())
            prototype.scale = (1., 1., 1.)
            prototype.visual_material = None
            target_markers.append(VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=f'/Visuals/RecordingTarget/{name}', markers={name: prototype})))
        command = env.command_manager.get_term('object_pose')
        marker_pos = command.pos_command_w[:1] + torch.tensor([[0.14, -0.14, 0.05]], device=env.device)
        def update_target():
            for marker in target_markers:
                marker.visualize(translations=marker_pos, orientations=command.quat_command_w[:1])
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
            object_asset=asset, collision_dimensions_mm=dimensions_mm.tolist(),
            noise_config=asdict(noise), seed=args.seed, steps=args.steps,
            fps=1 / env.step_dt, policy_hz=1 / env.step_dt,
            policy_input=f"{cfg.history} x {cfg.observation_dim} noisy observations", actor_amp=cfg.amp,
            exploration_action_noise=False, camera_eye=eye, camera_target=target,
            continuous=True, selected_environment=0,
            target_shaping=(dict(velocity_deg_s=args.target_velocity_deg_s, tau_s=args.target_filter_tau_s,
                                scope='playback experiment only; actual joint speed is not hard bounded',
                                previous_action_observation='applied shaped command', reset='measured encoder clipped to existing position limits')
                            if shaper else None),
            hold_protocol=dict(angle_rad=cfg.hold_angle_threshold, angular_speed_rad_s=cfg.hold_speed_threshold, duration_s=cfg.hold_duration),
            target_visualization='authored yellow-body and black-pips subsets, two coincident visual-only markers, world goal quaternion, scale 1',
            target_marker_position=marker_pos[0].cpu().tolist(),
            trajectory=trajectory_contract(),
            event_metrics=dict(orientation_error_deg='completed transition, pre-reset goal and object',
                pre_reset_angular_speed_rad_s='completed transition, pre-reset object',
                displayed_goal_error_deg='displayed post-step/post-reset object and goal',
                displayed_angular_speed_rad_s='displayed post-step/post-reset object',
                hold_ticks='displayed post-step/post-reset goal',
                outcome_counts='cumulative completed-transition events'),
        )
        manifest['object_support_reset'] = getattr(env, 'object_support_audit', None)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        writer = imageio.get_writer(
            str(args.output_dir / "rollout.partial.mp4"), fps=1 / env.step_dt,
            codec="libx264", quality=None, macro_block_size=16,
            ffmpeg_params=["-crf", "23", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"],
        )
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 24)
        small = ImageFont.truetype(font_path, 19)
        successes = drops = resets = timeouts = episode_successes = 0
        episodes, rows = [], []
        trajectory = {name: [] for name in trajectory_contract()['fields']}
        data = NewtonManager._solver.mjw_data
        peaks = dict(candidates=0, contacts=0, constraints=0)
        with torch.no_grad():
            for step in range(args.steps):
                action = actor(adapter.history)
                raw_action = action
                if shaper is not None:
                    action = shaper(action)
                try:
                    row, _, _, _ = adapter.step(action)
                except RuntimeError:
                    if hasattr(adapter, 'physics_failure'):
                        (args.output_dir / 'physics_failure.json').write_text(json.dumps(adapter.physics_failure, indent=2))
                        torch.save(dict(state=adapter.state.detach().cpu(), action=action.detach().cpu(),
                                        diagnostics=adapter.physics_failure_snapshot()),
                                   args.output_dir / 'physics_failure_state.pt')
                    raise
                if shaper is not None and bool(row['done'][0]):
                    shaper.reset(adapter.state[:, :21] - adapter.state.new_tensor(cfg.encoder_bias_rad))
                assert torch.isfinite(row["next_physical"]).all()
                assert torch.isfinite(row["reward"]).all()
                assert int(np.bitwise_or.reduce(data.overflow.numpy().ravel())) == 0
                peaks['candidates'] = max(peaks['candidates'], int(data.ncollision.numpy().max()))
                peaks['contacts'] = max(peaks['contacts'], int(data.nacon.numpy().max()))
                peaks['constraints'] = max(peaks['constraints'], int(data.nefc.numpy().max()))
                reached = bool(row["success"][0])
                dropped = bool(((drop_distance(row["next_physical"], cfg) > cfg.drop_radius_m) & row["terminated"])[0])
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
                              orientation_error_deg=error_deg, displayed_goal_error_deg=live_error_deg,
                              pre_reset_angular_speed_rad_s=float(adapter.captured_angular_speed[0]),
                              hold_ticks=float(command.hold_ticks[0]),
                              displayed_angular_speed_rad_s=float(env.scene['object'].data.root_ang_vel_w.torch[0].norm()))
                rows.append(record)
                snapshot = trajectory_snapshot(row, raw_action, adapter.state, current_quat,
                    command.quat_command_w[:1], env.scene['robot'].data.root_quat_w.torch[:1])
                for name, value in snapshot.items():
                    trajectory[name].append(value)
                pixels = viz.render_rgb_array()[..., :3]
                frame = Image.fromarray(pixels)
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 116), fill="#101820")
                draw.text((20, 12), f"Revo3 | Absolute-position policy | checkpoint {iteration} | die {dimensions_mm.max():.0f} mm", font=font, fill="white")
                draw.text((20, 48), f"t={record['time_s']:05.2f}s   Goals: {successes}   Drops: {drops}   Resets: {resets}", font=font, fill="#6ae0d0")
                draw.text((20, 84), f"{1/env.step_dt:g} Hz | 1x speed | seed {args.seed} | noisy observations | continuous rollout", font=small, fill="#c5d4df")
                if shaper is not None:
                    draw.text((20, 124), f"PLAYBACK LIMIT: {args.target_velocity_deg_s:g} deg/s target | smoothing tau={args.target_filter_tau_s:g}s", font=small, fill="#ffcc66")
                draw.text((470, 690), 'CURRENT / in-hand', font=font, fill='white')
                draw.text((880, 260), 'TARGET orientation', font=font, fill='#6ae0d0')
                draw.text((880, 295), f'Angle error: {live_error_deg:5.1f} deg', font=small, fill='white')
                draw.text((880, 535), 'Same world axes / scale 1:1', font=small, fill='#c5d4df')
                draw.text((880, 565), f"Hold <{np.degrees(cfg.hold_angle_threshold):.1f} deg, <{cfg.hold_speed_threshold:g} rad/s", font=small, fill='#c5d4df')
                draw.text((880, 595), f"Speed: {record['displayed_angular_speed_rad_s']:.2f} rad/s", font=small, fill='white')
                distance = (float(adapter.captured_position_error[0]) if cfg.track_object_position
                            else float(drop_distance(adapter.state, cfg)[0]))
                if not cfg.track_object_position and cfg.drop_boundary_mode == 'com_ellipsoid':
                    distance_text = f"COM boundary ratio: {distance/cfg.drop_radius_m:.3f} / 1.000"
                else:
                    distance_label = 'Position error' if cfg.track_object_position else 'Hand-center distance'
                    distance_text = f"{distance_label}: {distance*100:.2f} cm"
                draw.text((880, 655), distance_text, font=small, fill="white")
                draw.text((880, 625), f"Hold: {record['hold_ticks']*env.step_dt:.2f} / {cfg.hold_duration:.2f} s", font=small, fill='#6ae0d0')
                writer.append_data(np.asarray(frame))
                if step in (0, 299, 599, 899, 1199, 1499, args.steps - 1):
                    frame.save(args.output_dir / f"frame_{step+1:04d}.jpg")
                if (step + 1) % 300 == 0:
                    print(json.dumps(record), flush=True)
        writer.close()
        writer = None
        (args.output_dir / "rollout.partial.mp4").replace(args.output_dir / "rollout.mp4")
        np.savez_compressed(args.output_dir / "trajectory.npz", **trajectory)
        (args.output_dir / "events.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        summary = dict(iteration=iteration, seconds=args.steps * env.step_dt, frames=args.steps,
                       successes=successes, drops=drops, resets=resets, timeouts=timeouts,
                       completed_episodes=episodes, unfinished_episode_goals=episode_successes,
                       parity_max_errors=adapter.max_errors, contact_peaks=peaks, live_convex_audit=adapter.manual_asset_audit)
        summary['object_support_reset'] = getattr(env, 'object_support_audit', None)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("RECORDING_COMPLETE", json.dumps(summary), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
