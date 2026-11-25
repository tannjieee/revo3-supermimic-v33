"""Exercise the Revo3 task in Isaac Sim using a procedural cube (no object dataset required).

Run from the repository root with the Isaac Lab interpreter:
    python scripts/tests/smoke_revo3.py --viz none --device cpu env.sim.physics.solver_cfg.use_mujoco_cpu=true
"""

import argparse
import json
import time
import traceback

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--action_mode", choices=("hold", "zero", "random", "mixed"), default="mixed")
parser.add_argument(
    "--dataset_object", action="store_true", help="Use the active dataset object instead of a test cube."
)
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3")
args = parse_runtime_args(parser)
app = SimulationRuntime(args)

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab_tasks.utils import parse_env_cfg

import viserdex.tasks  # noqa: F401
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES


def main():
    task = "Isaac-ViserDex-Repose-Revo3"
    # Validate every registration/config, including camera variants, before simulation.
    for suffix in ("", "-Play", "-Estimator", "-Estimator-Eval", "-Estimator-Camera", "-Estimator-NoAug"):
        cfg = app.env_cfg(f"Isaac-ViserDex-Repose-Revo3{suffix}", device=args.device, num_envs=args.num_envs)
        assert cfg.actions.joint_pos.joint_names == list(REVO3_JOINT_NAMES)
        assert cfg.actions.joint_pos.preserve_order
        for group_name in ("pose", "gaussian_features"):
            group = getattr(cfg.observations, group_name, None)
            if group is not None:
                assert group.image_features.params["masking_data"] is None
                assert group.image_features.params["hand_camera_cfg"].name == "hand_depth_camera"
                assert all(target.track_mesh_transforms for target in cfg.scene.hand_depth_camera.mesh_prim_paths)

    cfg = app.env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    if not args.dataset_object:
        cfg.scene.object.spawn = sim_utils.CuboidCfg(
            size=(0.045, 0.045, 0.045),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.05),
        )
    cfg.commands.object_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.observations.privileged.enable_corruption = False
    env = gym.make(task, cfg=cfg)
    try:
        raw = env.unwrapped
        # Native physics clones need not exist as separate USD prims. Verify
        # the live articulation contract rather than a renderer's clone count.
        obs, _ = env.reset()
        assert raw.action_manager.total_action_dim == 21
        assert obs["policy"].shape == (args.num_envs, 49)
        assert obs["privileged"].shape == (args.num_envs, 38)
        object_mass = raw.scene["object"].data.default_mass.torch
        assert torch.isfinite(object_mass).all() and (object_mass > 0).all()
        print("OBJECT_DEFAULT_MASS_KG:", object_mass[0].tolist())
        robot = raw.scene["robot"]
        ids, names = robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
        assert names == list(REVO3_JOINT_NAMES)
        term = raw.action_manager.get_term("joint_pos")
        assert term._joint_names == names
        limits = robot.data.soft_joint_pos_limits.torch[:, ids]
        default = robot.data.default_joint_pos.torch[:, ids]
        hold = 2 * (default - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1

        # Test limit mapping, EMA and target writes with deliberately out-of-range inputs.
        for value in (-2.0, 2.0):
            term.reset()
            previous = robot.data.joint_pos.torch[:, ids].clone()
            term.process_actions(torch.full_like(hold, value))
            expected = cfg.actions.joint_pos.alpha * limits[..., int(value > 0)]
            expected += (1 - cfg.actions.joint_pos.alpha) * previous
            expected = expected.clamp(limits[..., 0], limits[..., 1])
            torch.testing.assert_close(term.processed_actions, expected)
            term.apply_actions()
            torch.testing.assert_close(robot.data.joint_pos_target.torch[:, ids], expected)
        term.reset()

        started = time.monotonic()
        max_joint_speed = 0.0
        resets = 0
        for step in range(args.steps):
            mode = args.action_mode
            if mode == "mixed":
                mode = ("hold", "zero", "random")[min(2, step * 3 // args.steps)]
            actions = (hold + 0.1 * torch.sin(torch.tensor(step / 10, device=raw.device))).clamp(-1, 1)
            if mode == "zero":
                actions = torch.zeros_like(hold)
            elif mode == "random":
                actions = 2 * torch.rand_like(hold) - 1

            obs, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(robot.data.joint_pos.torch).all()
            max_joint_speed = max(max_joint_speed, robot.data.joint_vel.torch.abs().max().item())
            # Position/velocity observations must follow the declared policy order.
            expected_pos = (
                2 * (robot.data.joint_pos.torch[:, ids] - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1
            )
            torch.testing.assert_close(obs["policy"][:, :21], expected_pos)
            torch.testing.assert_close(obs["policy"][:, -21:], raw.action_manager.action)
            torch.testing.assert_close(obs["privileged"][:, :21], robot.data.joint_vel.torch[:, ids] * 0.2)
            assert (term.processed_actions >= limits[..., 0] - 1e-6).all()
            assert (term.processed_actions <= limits[..., 1] + 1e-6).all()
            if step and step % max(1, args.steps // 3) == 0:
                env.reset()
                resets += 1
            if step == args.steps // 2:
                # Exercise selective reset with a permuted (non-PhysX-order) joint list.
                reset_ids = torch.tensor([0], device=raw.device)
                raw._reset_idx(reset_ids)
                torch.testing.assert_close(term._prev_applied_actions[0], robot.data.joint_pos.torch[0, ids])

        print(
            "PHYSICS_DIAGNOSTICS:",
            json.dumps(
                {
                    "mode": args.action_mode,
                    "resets": resets,
                    "max_joint_speed_rad_s": max_joint_speed,
                    "seconds": time.monotonic() - started,
                    "torch_peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available()
                    else 0,
                }
            ),
        )
        print(f"REVO3_SMOKE_OK: {args.num_envs} envs, {args.steps} steps, actions=21, policy=49, privileged=38")
    finally:
        env.close()


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    # Kit's default fast shutdown can exit the interpreter before Python emits
    # a pending exception. Print it explicitly so a failed smoke test is visible.
    traceback.print_exc()
    raise
finally:
    app.close()
