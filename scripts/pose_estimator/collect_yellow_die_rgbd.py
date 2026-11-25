"""Collect synchronized RGB-D during actor-only policy play; no training updates."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/viserdex"))
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding
from viserdex.distillation.teacher import configure_clean_state_observations
# Reuse the unchanged environment configuration only, never its training loop.
from viserdex.supermimic.simulation import SuperMimicEnvCfg, TransitionRecorderCfg, TASK_ID as POLICY_TASK_ID
import gymnasium as gym

TASK_ID = "Isaac-ViserDex-Repose-Revo3-YellowDie-WujiOffset-HandPose-Hold-RGBD-Capture-v1"
gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": SuperMimicEnvCfg})
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--env_config", type=Path, required=True)
parser.add_argument("--calibration", type=Path,
                    default=ROOT / "config/cameras/realsense_261722074750_1280x720_extrinsics_revo3_20261101.json",
                    help="RGB intrinsics and base-to-camera extrinsics; defaults to the 2026-11-01 calibration")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--fixture", type=Path,
                    default=ROOT / "assets/calibration_fixture/calibration_fixture_visual.usda")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--samples", type=int, default=120)
parser.add_argument("--policy_checkpoint", type=Path)
parser.add_argument("--policy_env_config", type=Path)
parser.add_argument("--zero_action_preview", action="store_true", help="Explicit diagnostic mode only")
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--frames_per_episode", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--splat_augmentation", type=Path,
                    default=ROOT / "configs/rgbd_capture/viserdex_default_v2.json")
parser.add_argument("--no_splat_augmentation", action="store_true", help="Save unperturbed Gaussian RGB")
parser.add_argument("--augmentation_seed", type=int, default=20261102)
parser.add_argument("--point_noise", type=Path, default=ROOT / "configs/rgbd_capture/point_noise_v1.json")
parser.add_argument("--no_point_noise", action="store_true", help="Keep point inputs equal to clean observed points")
parser.add_argument("--formal", action="store_true", help="Audit occlusion, reject invalid pairs and replenish")
parser.add_argument("--burn_in", type=int, default=40, help="Formal collection: policy steps after every reset")
parser.add_argument("--split", choices=("train", "validation"), default="train")
parser.add_argument("--render_batch_size", type=int, default=8)
parser.add_argument("--writer_threads", type=int, default=4)
add_runtime_args(parser, TASK_ID)
args = parse_runtime_args(parser)
if min(args.num_envs, args.samples, args.stride, args.frames_per_episode, args.render_batch_size, args.writer_threads) < 1:
    parser.error("All environment, sample, batch and worker counts must be positive")
if args.num_envs > 1 and not args.formal:
    parser.error("Parallel capture requires --formal")
if not args.zero_action_preview and (args.policy_checkpoint is None or args.policy_env_config is None):
    parser.error("Policy collection requires --policy_checkpoint and --policy_env_config")
if args.zero_action_preview and args.policy_checkpoint is not None:
    parser.error("Choose policy rollout or explicit zero-action diagnostic, not both")
if args.output_dir.exists():
    parser.error("Refusing to overwrite an existing dataset")
if args.formal and (args.zero_action_preview or args.burn_in < 8):
    parser.error("Formal capture requires the policy and at least its 8-step warmup")
if args.formal and args.split == "validation" and not (args.no_point_noise and args.no_splat_augmentation):
    parser.error("Validation must explicitly use clean RGB and point inputs")
from viserdex.renderer.splat_randomization import SplatRandomizationCfg
from viserdex.collectors.point_noise import PointNoiseCfg
point_noise_cfg = None if args.no_point_noise else PointNoiseCfg(**json.loads(args.point_noise.read_text()))
splat_cfg = (None if args.no_splat_augmentation else
             SplatRandomizationCfg(**json.loads(args.splat_augmentation.read_text())))
binding = SavedEnvironmentBinding(args.env_config)
base_contract = {}
policy_binding = None
if not args.zero_action_preview:
    policy_binding = SavedEnvironmentBinding(args.policy_env_config)
    # Only relocate identical assets into this isolated checkout; preserve all
    # dynamics, actions, observations, rewards and randomization settings.
    expected = copy.deepcopy(policy_binding.contract)
    for name in ("robot", "object"):
        if binding.assets[name]["sha256"] != policy_binding.assets[name]["sha256"]:
            raise ValueError("Policy/capture asset differs: " + name)
        expected["scene"][name]["spawn"]["usd_path"] = binding.contract["scene"][name]["spawn"]["usd_path"]
    expected["commands"]["object_pose"]["goal_pose_visualizer_cfg"]["markers"]["goal"]["usd_path"] = (
        binding.contract["commands"]["object_pose"]["goal_pose_visualizer_cfg"]["markers"]["goal"]["usd_path"])
    if expected != binding.contract:
        raise ValueError("Capture environment differs from policy contract beyond identical asset relocation")


def configure(cfg):
    binding.configure(cfg, args)
    configure_clean_state_observations(cfg)
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed
    cfg.recorders = TransitionRecorderCfg()
    cfg.sim.physics.solver_cfg.nconmax = max(2048, cfg.sim.physics.solver_cfg.nconmax)
    base_contract.update(binding.verify(cfg))
    from viserdex.collectors.yellow_die_rgbd import configure_cameras
    configure_cameras(cfg, args.calibration, args.fixture)


app = SimulationRuntime(args, configure=configure)
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from isaaclab.utils.io import dump_yaml
from viserdex.collectors.yellow_die_rgbd import YellowDieRGBDCapture, tensor, digest


def transform(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def array(value):
    return value.detach().cpu().numpy()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    payload = None
    if policy_binding is not None:
        payload = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
        identity = json.loads((ROOT / "configs/rgbd_capture/yellow_policy_identity.json").read_text())
        assert digest(args.policy_checkpoint) == identity["checkpoint_sha256"], "Wrong yellow policy checkpoint"
        assert payload["version"] == identity["checkpoint_version"] == 17
        assert payload["iteration"] == identity["iteration"] == 32100
        assert POLICY_TASK_ID == identity["task_id"]
        assert policy_binding.contract_sha == identity["environment_contract"]
        assert payload["metadata"].get("actor_only_preview") is True
        assert digest(ROOT / "assets/rounded_die/gaussians.ply") == identity["splat_sha256"]
        assert payload["metadata"]["environment_contract"] == policy_binding.contract_sha
        assert payload["metadata"]["task_id"] == POLICY_TASK_ID
        assert not payload["metadata"].get("smoke", False)
        assert not payload["metadata"].get("check_joint_update", False)
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = dict(completed=False, samples=0, mode="preview_zero_action_short_sequences" if args.zero_action_preview else "policy_rollout_rgbd",
                    training_ready=False, task_id=TASK_ID, seed=args.seed, environment=base_contract,
                    coordinate_frame="optical x-right y-down z-forward; metres; quaternion XYZW",
                    label_convention="T_A_B maps coordinates B into A; labels are simulator ground truth",
                    collector_sha256=digest(__file__), control="zero-action diagnostic" if args.zero_action_preview else "checkpoint actor at original policy rate, no exploration action noise",
                    limitation="Short collection preview; coverage not validated as a full training dataset")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        clean, _ = env.reset()
        actor = adapter = None
        if payload is not None:
            from dataclasses import asdict
            from viserdex.distillation.config import PerceptionNoiseCfg
            from viserdex.supermimic.core import Config
            from viserdex.supermimic.networks import Actor
            from viserdex.supermimic.simulation import Adapter
            cfg = Config(**payload["config"])
            noise = PerceptionNoiseCfg(**payload["noise_config"])
            adapter = Adapter(env, noise, cfg.history)
            adapter.audit_observation(clean)
            assert cfg.action_mode == "home_offset" and cfg.ema_alpha == .5
            assert cfg.action_offset_scale == .5 and cfg.action_warmup_s == .4
            assert cfg.hold_duration == .5 and cfg.hold_speed_threshold == .5 and cfg.hold_angle_threshold == .1
            assert cfg.hand_pose_weight == .2
            assert adapter.action_alpha == cfg.ema_alpha
            assert adapter.geometry.digest == payload["metadata"]["geometry_digest"]
            assert {k:tuple(v) for k,v in adapter.context_schema.items()} == {k:tuple(v) for k,v in payload["metadata"]["context_schema"].items()}
            actor = Actor(cfg).to(env.device).eval()
            actor.load_state_dict(payload["actor"], strict=True)
            manifest["policy"] = dict(checkpoint=str(args.policy_checkpoint), checkpoint_sha256=digest(args.policy_checkpoint),
                iteration=payload["iteration"], task_id=POLICY_TASK_ID, history=cfg.history,
                input="original noisy state observations; RGB-D is recorded for predictor supervision",
                noise_config=asdict(noise), control_hz=1/env.step_dt,
                action_mode=cfg.action_mode, ema_alpha=cfg.ema_alpha,
                action_offset_scale=cfg.action_offset_scale, action_warmup_s=cfg.action_warmup_s,
                hand_pose_weight=cfg.hand_pose_weight,
                hold_protocol=dict(angle_rad=cfg.hold_angle_threshold, speed_rad_s=cfg.hold_speed_threshold, duration_s=cfg.hold_duration),
                original_environment_contract=policy_binding.contract_sha,
                isolated_asset_relocation_verified=True, optimizer_loaded=False)
            del payload
        capture = YellowDieRGBDCapture(env, args.calibration, splat_cfg, args.augmentation_seed, point_noise_cfg,
                                      occlusion_audit=args.formal)
        from viserdex.supermimic.geometry_asset import build_geometry_spec
        from viserdex.supermimic.convex_audit import audit_live_convex
        spec = build_geometry_spec(env.cfg.scene.robot.spawn.usd_path, env.cfg.scene.object.spawn.usd_path,
                                   env.cfg.scene.robot.spawn.scale, env.cfg.scene.object.spawn.scale)
        manifest["live_collision_audit"] = audit_live_convex(env.scene["object"], env, spec)
        if args.num_envs > 1:
            from viserdex.collectors.yellow_parallel import collect_parallel
            collect_parallel(env, args, actor, adapter, capture, manifest, out)
            return
        episode, total_steps, episode_steps = 0, 0, 0
        transitions = []
        transition_stream = (out / "policy_transitions.jsonl").open("w")
        rejected_stream = (out / "rejected.jsonl").open("w")
        from isaaclab_newton.physics import NewtonManager
        solver_data = NewtonManager._solver.mjw_data
        manifest["capture_fps"] = 1 / (env.step_dt * args.stride)
        manifest["capture"] = capture.contract
        manifest.update(formal=args.formal, split=args.split, requested_samples=args.samples,
                        burn_in_steps=args.burn_in if args.formal else 0,
                        stride=args.stride, attempts=0, rejected=0, rejection_reasons={},
                        started_unix=time.time())
        if args.formal:
            manifest["limitation"] = "Synthetic yellow-die policy trajectories; virtual co-located depth; no real-sensor evaluation"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

        def progress():
            manifest.update(samples=len(rows), physics_steps=total_steps, episodes=episode+1,
                            updated_unix=time.time(), capture=capture.contract)
            tmp = out / "manifest.json.tmp"
            tmp.write_text(json.dumps(manifest, indent=2))
            tmp.replace(out / "manifest.json")

        def advance(steps, action=None):
            nonlocal episode, episode_steps, total_steps
            for substep in range(steps):
                if action is None or substep:
                    action = actor(adapter.history) if actor is not None else torch.zeros(1,21,device=env.device)
                assert torch.isfinite(action).all()
                if actor is not None:
                    transition, _, _, _ = adapter.step(action)
                    assert torch.isfinite(transition["next_physical"]).all()
                    assert torch.isfinite(transition["reward"]).all()
                    done = bool(transition["done"][0])
                    row = dict(step=total_steps+1, sequence=episode, action=array(action[0]).tolist(),
                        reward=float(transition["reward"][0]), done=done,
                        reached=bool(transition["success"][0]),
                        orientation_error_deg=float(adapter.captured_orientation_error[0])*180/np.pi,
                        angular_speed_rad_s=float(adapter.captured_angular_speed[0]),
                        hold_ticks=float(env.command_manager.get_term("object_pose").hold_ticks[0]))
                    transitions.append(row)
                    transition_stream.write(json.dumps(row)+"\n")
                else:
                    _, _, terminated, truncated, _ = env.step(action)
                    done = bool((terminated | truncated)[0])
                assert int(np.bitwise_or.reduce(solver_data.overflow.numpy().ravel())) == 0, "Newton capacity overflow"
                total_steps += 1
                episode_steps += 1
                if done:
                    episode += 1
                    episode_steps = 0
            transition_stream.flush()

        index = 0
        while index < args.samples:
            if args.formal:
                while episode_steps < args.burn_in:
                    advance(1)
                    if total_steps > args.samples * args.stride * 100:
                        raise RuntimeError("Policy repeatedly resets before burn-in; stopping")
                if manifest['attempts'] >= 10*args.samples:
                    raise RuntimeError("Rejection limit reached; refusing incomplete training dataset")
            if args.zero_action_preview and index and index % args.frames_per_episode == 0:
                env.reset()
                episode += 1
                episode_steps = 0
            action = actor(adapter.history) if actor is not None else torch.zeros(1,21,device=env.device)
            assert torch.isfinite(action).all()
            rgb, points, uv, valid, counts = capture.capture()
            f = capture.last_frame
            manifest['attempts'] += 1
            if args.formal and not f['occlusion_audit']['passed']:
                audit = f['occlusion_audit']
                manifest['rejected'] += 1
                for reason in audit['reasons']:
                    manifest['rejection_reasons'][reason] = manifest['rejection_reasons'].get(reason, 0)+1
                rejected_stream.write(json.dumps(dict(attempt=manifest['attempts'], step=total_steps,
                    sequence=episode, audit=audit))+"\n")
                rejected_stream.flush()
                if manifest['attempts'] % 25 == 0:
                    progress()
                advance(args.stride, action)
                continue
            rgb_image = array(rgb[0].permute(1, 2, 0).mul(255).byte())
            scene = array(f["scene_rgb"][0]).astype("uint8")
            twc = transform(array(f["camera_pos"][0]), array(f["camera_quat"][0]))
            two = transform(array(f["object_pos"][0]), array(f["object_quat"][0]))
            tco = np.linalg.inv(twc) @ two
            keypoints = capture.reference @ tco[:3, :3].T + tco[:3, 3]
            robot = env.scene["robot"].data
            twb = transform(array(tensor(robot.root_pos_w)[0]), array(tensor(robot.root_quat_w)[0]))
            folder = out / f"{index:05d}"
            folder.mkdir()
            Image.fromarray(rgb_image).save(folder / "rgb.png")
            clean_rgb = array(f["rgb_clean"][0])
            Image.fromarray(clean_rgb).save(folder / "rgb_clean.png")
            Image.fromarray(scene).save(folder / "scene.png")
            play_raw = array(f["play_raw_rgb"][0]).astype("uint8")
            Image.fromarray(play_raw).save(folder / "play_raw.png")
            extra = dict(audit_isolated_depth_m=array(f['isolated_depth_m'][0])) if args.formal else {}
            np.savez_compressed(folder / "pair.npz", **extra, rgb=rgb_image, rgb_clean=clean_rgb,
                play_raw_rgb=play_raw,
                policy_action_to_next=array(action[0]),
                depth_m=array(f["depth_m"][0]), rgb_visible=array(f["rgb_visible"][0]),
                depth_visible=array(f["depth_visible"][0]), points_camera_m=array(points[0]),
                points_camera_clean_m=array(f["points_camera_clean_m"][0]),
                point_depth_noise_m=array(f["point_depth_noise_m"][0]),
                point_uv_rgb=array(uv[0]), point_valid=array(valid[0]),
                K_rgb=array(f["K_rgb"][0]), K_depth=array(f["K_depth"][0]),
                T_world_camera=twc, T_world_object=two, T_camera_object=tco, T_base_object=np.linalg.inv(twb) @ two,
                keypoints_object_m=capture.reference, keypoints_camera_m=keypoints.astype("float32"),
                joint_pos=array(tensor(robot.joint_pos)[0]),
                sequence_id=np.int64(episode), frame_id=np.int64(index),
                policy_step=np.int64(total_steps),
                sim_time_s=np.float64(total_steps*env.step_dt), episode_time_s=np.float64(episode_steps*env.step_dt))
            row = dict(frame=index, sequence=episode, splat_augmentation=f["splat_augmentation"], visible_depth_pixels=int(counts[0]),
                       policy_step=total_steps, occlusion_audit=f['occlusion_audit'],
                       point_noise=f["point_noise"],
                       visible_fixture_rgb_pixels=f["visible_fixture_rgb_pixels"],
                       visible_fixture_depth_pixels=f["visible_fixture_depth_pixels"],
                       valid_points=int(valid[0].sum()), visible_rgb_pixels=int(f["rgb_visible"][0].sum()),
                       pair_sha256=digest(folder / "pair.npz"))
            rows.append(row)
            (folder / "metadata.json").write_text(json.dumps(row, indent=2))
            if not args.formal or index % 25 == 0:
                print(json.dumps(row), flush=True)
                progress()
            if index+1 < args.samples and (actor is not None or (index+1) % args.frames_per_episode):
                advance(args.stride, action)
            index += 1
        transition_stream.close()
        rejected_stream.close()
        if adapter is not None:
            manifest["policy_audit"] = dict(parity_max_errors=adapter.max_errors, resets=adapter.reset_count,
                successes=sum(row["reached"] for row in transitions),
                nonzero_action_steps=sum(any(abs(x)>1e-6 for x in row["action"]) for row in transitions),
                steps=len(transitions), contact_overflow=False)
        manifest.update(completed=True, samples=len(rows), frames=rows, capture=capture.contract,
                        physics_steps=total_steps, episodes=episode+1, finished_unix=time.time())
        dump_yaml(str(out / "env.yaml"), env.cfg)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
