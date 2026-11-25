"""Unit and regression tests for V34 Phase 0 + Phase 1 SuperMimic optimizations."""
from dataclasses import replace
import math
from pathlib import Path
import pytest
import torch

from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.distillation.noise import quat_unique, rotation_vector_quat
from viserdex.supermimic.core import (
    hand_center_config, integrate_velocity_physical, physical_delta, Q, QD, POS, ROT, VEL, OMEGA,
)
from viserdex.supermimic.dex_reference import load_profile, profile_options, target, unclipped_target
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.orientation_reward import orientation_reward
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.sensors import initial_sensors, reset_sensors, observe
from viserdex.supermimic.stable_values import value_bounds, cap_values
from viserdex.supermimic.trainer import Trainer, checkpoint_version, imagination_protocol, holdout_env_count

V34_PROFILE = Path(__file__).resolve().parents[1] / "configs/supermimic_v34/control_opt_p1.yaml"


def make_v34_cfg(**overrides):
    profile, digest = load_profile(V34_PROFILE)
    base = hand_center_config(
        wm_task_losses=True, stable_value_learning=True, absolute_policy_optimization=True,
        paper_critic=True, quiet_hold_enabled=True, cost_penalty=0.05, actor_lr_scale=0.25,
        critic_td_steps=1, input_joint_velocity_std_floor_rad_s=math.pi / 2,
        cost_velocity_limit_rad_s=math.pi / 2, cost_target_rate_limit_rad_s=math.radians(100.0),
        stability_reward_scale=4.0, stability_angle_sigma_rad=math.radians(8.0),
        gamma=0.98, lam=0.9025, horizon=8,
        wm_batch=16, actor_batch=16, microbatch=16, wm_microbatch=16,
        actor_microbatch=16, critic_microbatch=16,
        wm_updates=1, wm_only_iters=0, warmup_steps=8, buffer_steps=24,
        critic_epochs=1, critic_minibatches=2,
        wm_hidden=(64, 64), actor_hidden=(64, 64), critic_hidden=(64, 64),
    )
    opts = profile_options(profile, digest)
    opts.update(overrides)
    return replace(base, **opts)


def synthetic_step(cfg, n, device="cpu"):
    limits = torch.tensor(cfg.network_joint_limits, device=device, dtype=torch.float32)[None].expand(n, -1, -1)
    home = torch.tensor(cfg.action_home, device=device, dtype=torch.float32)[None].expand(n, -1)
    q = home + 0.05 * torch.randn(n, 21, device=device)
    qd = 0.2 * torch.randn(n, 21, device=device)
    pos = torch.tensor([0.04, 0.0, 0.10], device=device)[None] + 0.01 * torch.randn(n, 3, device=device)
    rot = quat_unique(rotation_vector_quat(0.3 * torch.randn(n, 3, device=device)))
    vel = 0.05 * torch.randn(n, 3, device=device)
    omega = 0.4 * torch.randn(n, 3, device=device)
    last_action = torch.zeros(n, 21, device=device)
    goal_pos = torch.tensor([0.04, 0.0, 0.10], device=device)[None].expand(n, -1)
    goal_rot = quat_unique(rotation_vector_quat(0.8 * torch.randn(n, 3, device=device)))
    zero = 0.005 * torch.randn(n, 21, device=device)
    ema = home + zero
    wrench = torch.zeros(n, 6, device=device)
    state = torch.cat((q, qd, pos, rot, vel, omega, last_action, goal_pos, goal_rot, ema, zero, wrench), -1)
    return state, limits


def populate_replay(cfg, n_envs=10, steps=14, device="cpu", term_env=0, term_step=10):
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    context = torch.ones(n_envs, 65, device=device)
    _, limits = synthetic_step(cfg, n_envs, device=device)
    base_inv = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)[None].expand(n_envs, -1)
    trainer = Trainer(cfg, noise, context, limits, base_inv, device, episode_steps=250)
    replay = Replay(
        cfg.buffer_steps, n_envs, device, cfg.event_capacity, cfg.event_fraction, cfg.wm_window,
        adaptive=cfg.adaptive_replay, wm_holdout_envs=holdout_env_count(n_envs, cfg.wm_holdout_fraction),
        sampling_mode=cfg.replay_sampling_mode, failure_sample_fraction=cfg.failure_sample_fraction,
    )
    sensors = reset_sensors(initial_sensors(n_envs, device, noise), torch.ones(n_envs, dtype=torch.bool, device=device), noise)
    state, _ = synthetic_step(cfg, n_envs, device=device)
    obs, sensors = observe(state, sensors, limits, noise, cfg)
    history = obs[:, None].expand(-1, cfg.history, -1).clone()
    for step in range(steps):
        action = 0.2 * torch.randn(n_envs, 21, device=device).clamp(-1, 1)
        dv = 0.1 * torch.randn(n_envs, 27, device=device)
        nxt = integrate_velocity_physical(state, dv, cfg.control_dt, cfg.wm_integration_coeff)
        done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros(n_envs, dtype=torch.bool, device=device)
        if step == term_step:
            done[term_env] = True
            terminated[term_env] = True
        contact = (torch.rand(n_envs, len(cfg.contact_body_names), device=device) < 0.2)
        record = dict(
            state=state, next_physical=nxt, action=action, done=done, terminated=terminated,
            reward=torch.full((n_envs,), 0.2, device=device), contact=contact,
            history=history, exploration_noise_rad=torch.zeros(n_envs, 21, device=device),
            **{f"sensor/{k}": v.clone() for k, v in sensors.items()},
        )
        tgt = target(state, action, limits, cfg.ema_alpha, cfg.target_slew_rate_rad_s * cfg.control_dt)
        trainer.update_real_statistics(state, tgt, nxt, obs)
        replay.add(record)
        state = torch.cat((nxt, action, state[:, 76:83], tgt + state[:, 104:125], state[:, 104:125], state[:, 125:131]), -1)
        obs, sensors = observe(state, sensors, limits, noise, cfg)
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
    return trainer, replay


def test_v34_profile_loads_and_bumps_version():
    cfg = make_v34_cfg()
    assert cfg.uses_v34_optimizations
    assert checkpoint_version(cfg) == 34
    assert imagination_protocol(cfg).startswith("supermimic_opt_v34_velocity_integrated_bryson_linear_fine")
    assert cfg.wm_reward_loss_weight == 0.0
    assert cfg.wm_observation_loss_weight == 0.0
    assert cfg.wm_cost_loss_weight == 0.0


def test_velocity_integrated_dynamics_exact_kinematics_and_gradients():
    cfg = make_v34_cfg()
    state, _ = synthetic_step(cfg, 6)
    dt = cfg.control_dt
    c = cfg.wm_integration_coeff
    zero_dv = torch.zeros(6, 27, requires_grad=True)
    nxt = integrate_velocity_physical(state, zero_dv, dt, c)
    delta = physical_delta(state, nxt)
    # Zero velocity increment => exact inertial motion: dq = dt*qd, dp = dt*vel, drot = dt*omega.
    torch.testing.assert_close(delta[:, Q], dt * state[:, QD], atol=1e-6, rtol=0)
    torch.testing.assert_close(delta[:, QD], torch.zeros_like(state[:, QD]), atol=1e-6, rtol=0)
    torch.testing.assert_close(delta[:, POS], dt * state[:, VEL], atol=1e-6, rtol=0)
    torch.testing.assert_close(delta[:, 45:48], dt * state[:, OMEGA], atol=1e-5, rtol=0)
    # Gradients flow cleanly through both velocity and analytically integrated pose/SO(3).
    loss = (nxt[:, POS].square().sum() + nxt[:, ROT].square().sum() + nxt[:, Q].square().sum())
    loss.backward()
    assert torch.isfinite(zero_dv.grad).all() and zero_dv.grad.abs().sum() > 0


def test_pre_slew_target_rate_cost_restores_nonzero_gradient():
    cfg_v33_like = make_v34_cfg(target_rate_pre_slew_cost=False)
    cfg_v34 = make_v34_cfg(target_rate_pre_slew_cost=True)
    state, limits = synthetic_step(cfg_v34, 4)
    # Push raw action far from previous EMA so slew clamp saturates.
    action = torch.full((4, 21), 0.95, requires_grad=True)
    cost_clipped, ratios_clipped = violation_costs(state, state[:, :55], action, limits, cfg_v33_like)
    assert float(cost_clipped[:, 1].detach().max()) == pytest.approx(0.0, abs=1e-10)
    assert float(ratios_clipped[:, 1].detach().max()) <= 1.0 + 1e-5

    action2 = torch.full((4, 21), 0.95, requires_grad=True)
    cost_pre, ratios_pre = violation_costs(state, state[:, :55], action2, limits, cfg_v34)
    assert float(ratios_pre[:, 1].detach().max()) > 1.5
    assert float(cost_pre[:, 1].detach().min()) > 0.0
    cost_pre[:, 1].sum().backward()
    assert action2.grad.abs().sum() > 0


def test_linear_fine_orientation_reward_eliminates_26deg_plateau():
    cfg_g = make_v34_cfg(orientation_reward_mode="gaussian", orientation_tolerance_rad=math.radians(10.0),
                         orientation_coarse_fraction=1.0)
    cfg_lf = make_v34_cfg(orientation_reward_mode="linear_fine", orientation_tolerance_rad=0.0)
    angles = torch.tensor([0.0, math.radians(10.0), math.radians(26.0), math.radians(90.0), math.radians(150.0)],
                          requires_grad=True)
    r_g = orientation_reward(angles, cfg_g)
    r_lf = orientation_reward(angles, cfg_lf)
    # Exact upper bound at 0 deg; strictly monotonic decreasing.
    assert float(r_lf[0].detach()) == pytest.approx(cfg_lf.orientation_reward_scale, abs=1e-6)
    assert (r_lf[:-1] > r_lf[1:]).all()
    # At 26 deg: Gaussian already gave ~96.9% of max; linear_fine gives ~61%, leaving strong pull to <10 deg.
    assert float((r_g[2] / cfg_g.orientation_reward_scale).detach()) > 0.95
    assert float((r_lf[2] / cfg_lf.orientation_reward_scale).detach()) < 0.65
    # Stronger gradient at both 26 deg and 150 deg.
    g_lf = torch.autograd.grad(r_lf.sum(), angles)[0].abs()
    assert float(g_lf[2]) > 3.0 and float(g_lf[4]) > 1.5


def test_wm_holdout_zero_leakage_and_uniform_failure_sampling():
    cfg = make_v34_cfg(wm_holdout_fraction=0.2, replay_sampling_mode="uniform_failure", failure_sample_fraction=0.25)
    trainer, replay = populate_replay(cfg, n_envs=10, steps=14, term_env=2, term_step=10)
    assert replay.wm_holdout_envs == 2
    assert replay.wm_train_envs == 8
    assert trainer.wm_train_envs == 8
    # Normalizer count must only include the 8 training envs across 14 steps.
    assert int(trainer.wm.input_norm.count.item()) == 8 * 14
    assert int(trainer.actor.obs_norm.count.item()) == 10 * 14
    # WM training batches strictly draw from [0, 8), holdout strictly from [8, 10).
    train_batch = replay.sample(64, cfg.wm_window, include_events=False)
    holdout_batch = replay.sample_holdout(32, cfg.wm_window)
    assert int(train_batch["env_id"].max()) < 8
    assert int(holdout_batch["env_id"].min()) >= 8
    # Near-failure archive captured the terminated window from env 2.
    assert replay.archive_size >= 1
    assert (train_batch["terminated"].any(dim=1)).float().mean() > 0.1
    # Leak-free holdout evaluation produces finite loss and 8-step RMSE vs persistence metrics.
    ho = trainer.evaluate_wm_holdout(replay, batch_size=16)
    for k in ("wm_holdout_loss", "wm_holdout_loss_object_rotation", "wm_holdout_rot_rmse_deg",
              "wm_holdout_rot_baseline_deg", "wm_holdout_rot_gain_ratio"):
        assert k in ho and math.isfinite(ho[k])


def test_deterministic_imagination_critic_value_cap_and_full_update():
    cfg = make_v34_cfg()
    trainer, replay = populate_replay(cfg, n_envs=10, steps=14)
    # Check cap_values enforces exact reward upper bound and [0, 1/(1-gamma)] cost bounds.
    bounds = torch.tensor(value_bounds(cfg))
    extreme = torch.tensor([[999.0, -5.0, 500.0, 10.0, 999.0, -1.0]])
    capped = cap_values(extreme, bounds)
    assert float(capped[0, 0]) == pytest.approx(float(bounds[0]))
    assert float(capped[0, 1]) == 0.0
    assert float(capped[0, 2]) == pytest.approx(float(bounds[2]))
    # Negative reward is not clipped from below (signed rewards allowed).
    neg = torch.tensor([[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert float(cap_values(neg, bounds)[0, 0]) == -3.0

    # Critic uses LayerNorm + GELU.
    assert any(isinstance(m, torch.nn.LayerNorm) for m in trainer.critic.net)
    assert any(isinstance(m, torch.nn.GELU) for m in trainer.critic.net)
    assert any(isinstance(m, torch.nn.GELU) for m in trainer.wm.net)

    # Full joint update (WM + failure + Actor + Critic) runs cleanly with finite metrics.
    metrics = trainer.update(replay)
    assert math.isfinite(metrics["wm_loss"])
    assert math.isfinite(metrics["actor_loss"])
    assert math.isfinite(metrics["critic_loss"])
    assert "critic_reward_cap_hit_fraction" in metrics

    # Save & restore roundtrip preserves V34 version and all buffers/weights.
    tmp = Path("/tmp/test_v34_checkpoint_roundtrip.pt")
    meta = dict(
        environment_contract="c", joint_names=[], context_schema={}, observation_protocol=cfg.observation_protocol,
        state_protocol=cfg.state_protocol, context_manifest={}, identified_hand={},
        joint_limits_rad=[], sdk_software_limits={}, training_soft_limits={}, environment_binding=dict(assets={}),
    )
    trainer.save(tmp, meta)
    payload = torch.load(tmp, map_location="cpu", weights_only=False)
    assert payload["version"] == 34
    trainer2 = Trainer(cfg, trainer.noise_cfg, trainer.context, trainer.limits, trainer.base_inverse, "cpu", 250)
    trainer2.restore(payload, meta)
    assert trainer2.iteration == trainer.iteration
    tmp.unlink(missing_ok=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for WM CUDA Graph parity check")
def test_v34_wm_cuda_graph_matches_eager():
    cfg = make_v34_cfg()
    torch.manual_seed(123)
    eager, replay = populate_replay(cfg, n_envs=10, steps=14, device="cuda")
    graph = Trainer(cfg, eager.noise_cfg, eager.context, eager.limits, eager.base_inverse, "cuda", 250, wm_cuda_graph=True)
    graph.wm.load_state_dict(eager.wm.state_dict())
    batch = replay.sample(cfg.wm_batch, cfg.wm_window, include_events=False)

    # Single-batch parity between eager loss/backward and CapturedWMLoss.
    from viserdex.supermimic.wm_cuda_graph import CapturedWMLoss
    captured = CapturedWMLoss(graph, batch)
    eager.opt_wm.zero_grad(set_to_none=True)
    eager.opt_failure.zero_grad(set_to_none=True)
    loss_e, err_e, phys_e, _, _ = eager._wm_loss(batch)
    loss_e.backward()
    eager._independent_failure_loss.backward()
    loss_g, err_g, phys_g, _, _ = captured(batch)
    torch.testing.assert_close(loss_g, loss_e, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(err_g, err_e, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(phys_g, phys_e, atol=1e-5, rtol=1e-5)
    for (n, pe), (_, pg) in zip(eager.wm.named_parameters(), graph.wm.named_parameters()):
        torch.testing.assert_close(pg.grad, pe.grad, atol=1e-4, rtol=1e-4, msg=f"Grad mismatch on {n}")
