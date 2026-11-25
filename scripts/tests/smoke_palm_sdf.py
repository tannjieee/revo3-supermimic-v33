"""Check actual packed Newton SDF distances, task stepping and contact capacities."""

# ruff: noqa: E402 -- Simulation launch must precede backend-dependent imports.
import argparse
import json
from pathlib import Path
import time
import traceback

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--output", required=True)
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3-SDF")
args = parse_runtime_args(parser)
app = SimulationRuntime(args)

import gymnasium as gym
import newton
import numpy as np
import torch
import warp as wp
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom
from isaaclab_newton.physics import NewtonManager
from newton._src.geometry.soft_contacts_sdf import TextureSDFData, eval_shape_sdf
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES, REVO3_MULTIPHYSICS_USD


@wp.kernel
def query_palm(
    table: wp.array(dtype=TextureSDFData),
    sdf_index: int,
    scale: wp.vec3,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    i = wp.tid()
    lower, value, gradient = eval_shape_sdf(8, scale, points[i], sdf_index, table)
    distances[i] = value


def main():
    cfg = app.env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.commands.object_pose.debug_vis = False
    assert not cfg.sim.physics.solver_cfg.use_mujoco_contacts
    env = gym.make(args.task, cfg=cfg)
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        model = NewtonManager.get_model()
        solver = NewtonManager._solver
        # Read-only diagnostics of Newton's experimental packed SDF storage.
        # Deferred cooking intentionally leaves Mesh.sdf empty; Model owns it.
        palm_ids = [i for i, name in enumerate(model.shape_label) if name.endswith("/palm_sdf_collision")]
        assert len(palm_ids) == args.num_envs
        sdf_indices = model._shape_sdf_index.numpy()
        assert np.all(sdf_indices[palm_ids] >= 0), "SDF was not cooked; refusing mesh/hull fallback"
        assert np.all(model.shape_type.numpy()[palm_ids] == int(newton.GeoType.MESH))
        table = model._texture_sdf_data
        assert table is not None and table.size > 0
        first = palm_ids[0]
        src = model.shape_source[first]
        assert len(src.vertices) == 602 and len(src.indices) // 3 == 1200
        candidates = np.stack(
            np.meshgrid(np.linspace(0.008, 0.020, 7), np.linspace(0.015, 0.045, 13), np.linspace(0.045, 0.100, 23)),
            axis=-1,
        ).reshape(-1, 3)
        points = wp.array(candidates.astype(np.float32), dtype=wp.vec3, device=raw.device)
        distances = wp.empty(len(candidates), dtype=float, device=raw.device)
        scale = model.shape_scale.numpy()[first]
        wp.launch(
            query_palm,
            dim=len(candidates),
            inputs=[table, int(sdf_indices[first]), wp.vec3(*scale), points, distances],
            device=raw.device,
        )
        values = distances.numpy()
        assert np.isfinite(values).all()
        source_stage = Usd.Stage.Open(str(REVO3_MULTIPHYSICS_USD))
        old_prim = source_stage.GetPrimAtPath(
            "/revo3_single/Geometry/world/right_plam_backtrim_v2/right_plam_backtrim_v2"
        )
        hull = ConvexHull(np.asarray(UsdGeom.Mesh(old_prim).GetPointsAttr().Get()))
        old_depth = -(candidates @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
        valid = (old_depth > 0.002) & (values > 0.002)
        assert valid.any(), "No clear free-space probe within the old palm hull"
        selected = np.flatnonzero(valid)[np.argmax(values[valid])]
        probe = {
            "local_position_m": candidates[selected].tolist(),
            "old_hull_interior_depth_m": float(old_depth[selected]),
            "actual_sdf_clearance_m": float(values[selected]),
            "sdf_index": int(sdf_indices[first]),
            "cooked_sdf_count": int(table.size),
        }
        print("SDF_RECESS_PROBE", json.dumps(probe), flush=True)
        robot = raw.scene["robot"]
        ids, _ = robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
        limits = robot.data.soft_joint_pos_limits.torch[:, ids]
        q = robot.data.default_joint_pos.torch[:, ids]
        hold = (2 * (q - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1).clamp(-1, 1)
        peak_contacts = 0
        peak_constraints = 0
        overflow = 0
        resets = 0
        started = time.monotonic()
        with torch.inference_mode():
            for step in range(args.steps):
                action = hold if step < args.steps // 3 else torch.zeros_like(hold)
                if step >= args.steps * 2 // 3:
                    action = torch.rand_like(hold) * 2 - 1
                obs, reward, _, _, _ = env.step(action)
                assert obs["policy"].shape == (args.num_envs, 49)
                assert obs["privileged"].shape == (args.num_envs, 38)
                assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
                contacts = NewtonManager.get_contacts()
                count = int(contacts.rigid_contact_count.numpy()[0])
                assert count <= contacts.rigid_contact_max, (count, contacts.rigid_contact_max)
                assert count <= solver.mjw_data.naconmax, (count, solver.mjw_data.naconmax)
                peak_contacts = max(peak_contacts, count)
                peak_constraints = max(peak_constraints, int(solver.mjw_data.nefc.numpy().max()))
                overflow |= int(np.bitwise_or.reduce(solver.mjw_data.overflow.numpy().ravel()))
                assert overflow == 0
                if step and step % max(1, args.steps // 3) == 0:
                    env.reset()
                    resets += 1
                if step == args.steps // 2:
                    raw._reset_idx(torch.tensor([0], device=raw.device))
        result = {
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "resets": resets,
            "seconds": time.monotonic() - started,
            "use_cuda_graph": cfg.sim.physics.use_cuda_graph,
            "use_mujoco_contacts": cfg.sim.physics.solver_cfg.use_mujoco_contacts,
            "num_substeps": cfg.sim.physics.num_substeps,
            "probe": probe,
            "peak_newton_contacts": peak_contacts,
            "newton_contact_capacity": contacts.rigid_contact_max,
            "max_constraints_per_env": peak_constraints,
            "overflow_bitmask": overflow,
            "num_solver_geoms_per_world": solver.mj_model.ngeom,
            "sdf_voxel_size_m": table.numpy()["voxel_size"][int(sdf_indices[first])].tolist(),
            "sdf_subgrid_count": int(table.numpy()["num_subgrids"][int(sdf_indices[first])]),
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print("PALM_SDF_SMOKE_OK", json.dumps(result), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
