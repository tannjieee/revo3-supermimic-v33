"""Untrained-on holdout rollout error, with episode-boundary accounting."""

import torch
from .state_layout import layout
from .core import (
    advance,
    ROT,
    GOAL_ROT,
    action_target,
    physical_delta,
    quat_mul,
    quat_conjugate,
    rotation_log,
    reward_from_orientation_error,
    reward,
)


@torch.no_grad()
def audit_real_reward(state, physical_next, action, world_error, actual_reward, dt, success_bonus=250., action_rate_weight=.01, joint_vel_weight=2.5e-5, action_l2_weight=1e-4, action_second_diff_weight=0.0, *, hold_cfg=None, success_event=None):
    """Check frame conversion separately from the discontinuous success bonus.

    World angles are independently captured from official pre-reset poses. A few
    FP32 ulps across the 0.1-rad threshold must not turn into a false reward alarm.
    Wrong poses/goals and wrong rewards still fail their respective assertions.
    """
    local_error = rotation_log(quat_mul(physical_next[..., ROT], quat_conjugate(state[..., layout(state).goal_rot]))).norm(dim=-1)
    torch.testing.assert_close(local_error, world_error, atol=2e-6, rtol=1e-5)
    expected = reward_from_orientation_error(state, physical_next, action, world_error, dt, success_bonus, action_rate_weight, joint_vel_weight, action_l2_weight, action_second_diff_weight, hold_cfg=hold_cfg, success_event=success_event)
    torch.testing.assert_close(expected, actual_reward, atol=1e-4, rtol=1e-4)
    return {
        "reward": float((expected - actual_reward).abs().max()),
        "orientation_frame": float((local_error - world_error).abs().max()),
        "success_boundary_disagreements": int(((local_error < 0.1) != (world_error < 0.1)).sum()),
    }


@torch.no_grad()
def open_loop_errors(model, records, context, limits, cfg=None):
    """The records must be collected AFTER the last update and never fitted first."""
    from .core import Config
    cfg = cfg or Config()
    state = records[0]["state"].clone()
    initial = state.clone()
    mask = torch.ones(len(state), device=state.device, dtype=torch.bool)
    output = {}
    groups = dict(
        joint_position_rad=slice(0, 21),
        joint_velocity_rad_s=slice(21, 42),
        object_position_m=slice(42, 45),
        object_orientation_rad=slice(45, 48),
        object_linear_velocity_m_s=slice(48, 51),
        object_angular_velocity_rad_s=slice(51, 54),
    )
    for i, row in enumerate(records, 1):
        if cfg.dex_reference:
            state = torch.cat((state[:, :125], row["state"][:, 125:131]), -1)
        target = action_target(state, row["action"], limits, cfg)
        nxt = model(state, target, context)
        if True:
            error = physical_delta(row["next_physical"], nxt)
            baseline = physical_delta(row["next_physical"], initial)
            entry = {"valid_environments": int(mask.sum())}
            for name, sl in groups.items():
                entry[name] = float(error[mask, sl].square().mean().sqrt()) if mask.any() else None
                entry["persistence_" + name] = float(baseline[mask, sl].square().mean().sqrt()) if mask.any() else None
            # Retain historical per-component metrics, add unambiguous vector
            # position RMS and SO(3) geodesic RMS (not rotation-component RMS).
            for name, sl in (("object_position_vector_rmse_m", slice(42,45)),
                             ("object_orientation_geodesic_rmse_rad", slice(45,48))):
                entry[name] = float(error[mask, sl].square().sum(-1).mean().sqrt()) if mask.any() else None
            entry["physical_group_metric_definition"] = "root mean square per scalar component"
            if getattr(model, "geometry", None) is not None:
                predicted = model.geometry.readout(nxt)
                actual = model.geometry.readout(row["next_physical"])
                names = ("hand_point_position_m", "hand_point_velocity_m_s", "relative_position_m", "relative_velocity_m_s", "surface_gap_m")
                for name, a, b in zip(names, predicted[:5], actual[:5]):
                    entry[name] = float((a[mask]-b[mask]).square().mean().sqrt()) if mask.any() else None
            real_r = row.get("reward", reward(row["state"], row["next_physical"], row["action"], cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight, cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight, hold_cfg=cfg))
            predicted_r = reward(torch.cat((state[:, :55], row["state"][:, 55:]), -1), nxt, row["action"], cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight, cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight, hold_cfg=cfg)
            entry["real_reward_mean"] = float(real_r[mask].mean()) if mask.any() else None
            entry["predicted_reward_mean"] = float(predicted_r[mask].mean()) if mask.any() else None
            entry["reward_mae"] = float((real_r[mask]-predicted_r[mask]).abs().mean()) if mask.any() else None
            output[str(i)] = entry
        mask = mask & ~row["done"]
        state = (advance(state, nxt, row["action"], target, state[:, layout(state).goal_rot], hold_cfg=cfg)
                 if cfg.compact_state else torch.cat((nxt, target, state[:, 76:]), -1))
    return output
