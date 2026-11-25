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
V34_P2_PROFILE = Path(__file__).resolve().parents[1] / "configs/supermimic_v34/control_opt_p2.yaml"


def make_v34_cfg(*, profile_path=V34_PROFILE, **overrides):
    profile, digest = load_profile(profile_path)
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


def test_phase2_home_residual_actions_and_centered_actor_init():
    from viserdex.supermimic.core import action_target
    from viserdex.supermimic.networks import Actor
    from viserdex.supermimic.revised_contract import HARDWARE_JOINT_NAMES
    cfg = make_v34_cfg(profile_path=V34_P2_PROFILE, cost_penalty=0.25, horizon=12)
    assert len(cfg.dex_action_residual_rad) == 21
    # Thumb opposition joints have wider residual allowance than default fingers.
    cmp_idx = HARDWARE_JOINT_NAMES.index("right_thumb_CMP_joint")
    idx_idx = HARDWARE_JOINT_NAMES.index("right_index_PIP_joint")
    assert cfg.dex_action_residual_rad[cmp_idx] == pytest.approx(math.radians(55.0))
    assert cfg.dex_action_residual_rad[idx_idx] == pytest.approx(math.radians(42.0))

    # Centered zero-bias Actor outputs action == 0 -> exact HOME geometric target at startup.
    actor = Actor(cfg)
    assert float(actor.net[-1].bias.detach().abs().max()) == 0.0
    state, limits = synthetic_step(cfg, 4)
    home = torch.tensor(cfg.action_home, dtype=torch.float32)[None].expand(4, -1)
    # Zero encoder offset: action == 0 -> exact HOME target.
    state[:, 104:125] = 0.0
    state[:, 83:104] = home
    zero_action = torch.zeros(4, 21)
    tgt = action_target(state, zero_action, limits, cfg)
    torch.testing.assert_close(tgt, home, atol=1e-6, rtol=0)
    # With nonzero zero_bias delta: action == 0 produces home - alpha * delta (exact EMA math).
    delta = 0.01 * torch.ones(4, 21)
    state[:, 104:125] = delta
    state[:, 83:104] = home + delta
    tgt_biased = action_target(state, zero_action, limits, cfg)
    torch.testing.assert_close(tgt_biased, home - cfg.ema_alpha * delta, atol=1e-6, rtol=0)

    # Full extreme actions [-1, +1] stay strictly within soft limits before zero subtraction.
    for sign in (-1.0, 1.0):
        raw_enc = target(state, torch.full((4, 21), sign), limits, 1.0, 0.0,
                         home=cfg.action_home, residual_rad=cfg.dex_action_residual_rad) + state[:, 104:125]
        assert (raw_enc >= limits[..., 0] - 1e-6).all()
        assert (raw_enc <= limits[..., 1] + 1e-6).all()


def test_phase2_regularization_penalties_and_gradients():
    from viserdex.supermimic.quiet_hold import dex_regularization_penalty, imagined_dex_regularization
    cfg = make_v34_cfg(profile_path=V34_P2_PROFILE)
    state, limits = synthetic_step(cfg, 4)
    nxt = state[:, :55].clone().requires_grad_(True)
    action = torch.full((4, 21), 0.5, requires_grad=True)
    # Inside safe cage radius (2 cm < 4.5 cm): zero cage penalty.
    d_in = torch.full((4,), 0.02, requires_grad=True)
    p_in = dex_regularization_penalty(d_in, torch.zeros(4, 21), torch.zeros(4, 21), torch.zeros(4, 21), cfg.control_dt, cfg)
    assert float(p_in.detach().max()) == 0.0
    # Outside safe cage radius (7 cm > 4.5 cm): smooth positive penalty pulling object back inward.
    d_out = torch.full((4,), 0.07, requires_grad=True)
    p_out = dex_regularization_penalty(d_out, torch.zeros(4, 21), torch.zeros(4, 21), torch.zeros(4, 21), cfg.control_dt, cfg)
    assert float(p_out.detach().min()) == pytest.approx(cfg.cage_penalty_weight, abs=1e-5)
    g_d = torch.autograd.grad(p_out.sum(), d_out)[0]
    assert (g_d > 0).all()

    # Full imagined regularization delivers finite nonzero gradients to both action and next_physical.
    dist = (nxt[:, POS] - state[:, 76:79]).norm(dim=-1) + 0.08
    reg = imagined_dex_regularization(state, nxt, action, dist, cfg, limits=limits)
    reg.sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    assert torch.isfinite(nxt.grad).all() and nxt.grad[:, QD].abs().sum() > 0


def test_phase2_compact_two_head_critic_update_export_and_inference():
    from viserdex.supermimic.constraints import cost_names
    from viserdex.supermimic.inference import export_actor, ManualConvexPolicy
    from viserdex.supermimic.revised_contract import HARDWARE_JOINT_NAMES
    cfg = make_v34_cfg(profile_path=V34_P2_PROFILE, cost_penalty=0.25, horizon=8)
    assert cost_names(cfg) == ("drop",)
    trainer, replay = populate_replay(cfg, n_envs=10, steps=14)
    # Critic outputs exactly 2 heads: [reward, drop_risk].
    assert trainer.critic.net[-1].out_features == 2
    assert "_res_" in imagination_protocol(cfg) and "_seq_" in imagination_protocol(cfg) and "_c2_" in imagination_protocol(cfg)

    metrics = trainer.update(replay)
    assert math.isfinite(metrics["wm_loss"])
    assert math.isfinite(metrics["actor_loss"])
    assert math.isfinite(metrics["critic_loss"])
    assert "imagined_cost_drop" in metrics
    assert "imagined_cost_joint_speed" not in metrics

    # Export and run offline ManualConvexPolicy inference end-to-end.
    ckpt = Path("/tmp/test_v34_p2_ckpt.pt")
    exported = Path("/tmp/test_v34_p2_actor.pt")
    meta = dict(
        environment_contract="c", joint_names=list(HARDWARE_JOINT_NAMES), context_schema={},
        observation_protocol=cfg.observation_protocol, state_protocol=cfg.state_protocol,
        observation_dim=cfg.observation_dim, quaternion_order="XYZW",
        joint_limits_rad=[list(lim) for lim in cfg.network_joint_limits],
        context_manifest={}, identified_hand={}, sdk_software_limits={}, training_soft_limits={},
        environment_binding=dict(assets={}),
    )
    trainer.save(ckpt, meta)
    export_actor(ckpt, exported)
    policy = ManualConvexPolicy(exported, device="cpu")
    policy.set_goal([0.0, 0.0, 0.0, 1.0], [0.04, 0.0, 0.10])
    tgt, raw, obs = policy.propose(list(cfg.action_home), [0.0] * 21, [0.04, 0.0, 0.10], [0.0, 0.0, 0.0, 1.0])
    policy.commit()
    assert tgt.shape == (21,) and raw.shape == (21,) and obs.shape == (75,)
    ckpt.unlink(missing_ok=True)
    exported.unlink(missing_ok=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Phase 2 CUDA Graph parity check")
def test_phase2_wm_cuda_graph_matches_eager():
    cfg = make_v34_cfg(profile_path=V34_P2_PROFILE, cost_penalty=0.25, horizon=8)
    torch.manual_seed(456)
    eager, replay = populate_replay(cfg, n_envs=10, steps=14, device="cuda")
    graph = Trainer(cfg, eager.noise_cfg, eager.context, eager.limits, eager.base_inverse, "cuda", 250, wm_cuda_graph=True)
    graph.wm.load_state_dict(eager.wm.state_dict())
    batch = replay.sample(cfg.wm_batch, cfg.wm_window, include_events=False)

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


# ---------------------------------------------------------------------------
# Phase 3: Fast-Convergence Optimizations (V35)
# ---------------------------------------------------------------------------

V35_P3_PROFILE = Path(__file__).resolve().parents[1] / "configs/supermimic_v34/control_opt_p3.yaml"


def test_phase3_angular_progress_reward_bounds_symmetry_and_gradient():
    from viserdex.supermimic.orientation_reward import orientation_progress_reward
    from viserdex.supermimic.stability_reward import task_reward_bound
    cfg = make_v34_cfg(profile_path=V35_P3_PROFILE, cost_penalty=0.25, horizon=8)
    assert cfg.orientation_progress_scale == 4.0
    assert math.isclose(task_reward_bound(cfg), 0.04 * (10.0 + 4.0 + 4.0), rel_tol=1e-6)

    prev = torch.tensor([1.5, 1.5, 1.5])
    nxt = torch.tensor([1.3, 1.5, 1.7], requires_grad=True)
    prog = orientation_progress_reward(prev, nxt, cfg.control_dt, cfg)
    assert prog[0] > 0.0 and abs(float(prog[1].detach())) < 1e-7 and prog[2] < 0.0
    assert torch.allclose(prog[0], -prog[2], atol=1e-6)
    assert prog.abs().max() < cfg.orientation_progress_scale
    prog.sum().backward()
    assert (nxt.grad < 0).all()  # smaller next error increases reward


def test_phase3_ste_slew_limiter_exact_forward_and_nonzero_gradient():
    from viserdex.supermimic.dex_reference import filter_target
    prev = torch.zeros(4, 21)
    raw = torch.full((4, 21), 1.0, requires_grad=True)
    hard = filter_target(prev, raw, 0.51, max_delta=0.1, ste=False)
    hard.sum().backward()
    assert float(raw.grad.abs().max()) == 0.0  # hard clamp kills gradient

    raw2 = torch.full((4, 21), 1.0, requires_grad=True)
    ste = filter_target(prev, raw2, 0.51, max_delta=0.1, ste=True)
    torch.testing.assert_close(ste, hard.detach(), atol=1e-7, rtol=1e-6)
    ste.sum().backward()
    torch.testing.assert_close(raw2.grad, torch.full_like(raw2, 0.51), atol=1e-6, rtol=1e-6)


def test_phase3_recency_replay_and_bounded_so3_curriculum():
    from viserdex.supermimic.core import sample_bounded_so3_goal, quat_mul, quat_conjugate, rotation_log
    cfg = make_v34_cfg(profile_path=V35_P3_PROFILE, cost_penalty=0.25, horizon=8)
    assert cfg.recent_sample_fraction == 0.50
    assert cfg.failure_sample_fraction == 0.08
    replay = Replay(
        capacity=100, num_envs=20, device="cpu", event_window=8,
        wm_holdout_envs=2, sampling_mode="uniform_failure",
        failure_sample_fraction=0.08, recent_sample_fraction=0.50, recent_window_steps=15,
    )
    for step in range(80):
        replay.add({
            "state": torch.full((20, 4), float(step)),
            "next_physical": torch.zeros(20, 4),
            "action": torch.zeros(20, 2),
            "done": torch.zeros(20, dtype=torch.bool),
            "terminated": torch.zeros(20, dtype=torch.bool),
            "reward": torch.zeros(20),
        })
    batch = replay.sample(1000, window=1, include_events=False)
    steps = batch["state"][:, 0]
    recent_frac = float((steps >= 65).float().mean())
    # 50% forced from last 15 steps + ~50% * (15/80) from uniform = ~59%
    assert recent_frac > 0.52

    ref = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 256)
    max_ang = math.radians(60.0)
    goals = sample_bounded_so3_goal(ref, max_ang)
    errs = rotation_log(quat_mul(goals, quat_conjugate(ref))).norm(dim=-1)
    assert float(errs.min()) >= 0.25 * max_ang - 1e-4
    assert float(errs.max()) <= max_ang + 1e-4


def test_phase3_actor_lambda_return_and_multi_actor_updates():
    cfg = make_v34_cfg(profile_path=V35_P3_PROFILE, cost_penalty=0.25, horizon=8)
    assert cfg.actor_updates == 2
    assert cfg.critic_epochs == 2
    assert cfg.actor_lr_scale == 0.75
    assert cfg.actor_ramp_iters == 150
    assert cfg.actor_lambda_return is True
    assert math.isclose(cfg.exploration_position_std_rad, math.radians(2.5), rel_tol=1e-6)
    proto = imagination_protocol(cfg)
    assert all(tag in proto for tag in ("_prog", "_ste", "_alam", "_curr"))

    trainer, replay = populate_replay(cfg, n_envs=10, steps=14)
    metrics = trainer.update(replay)
    assert trainer.optimizer_steps["actor"] == 2
    assert trainer.optimizer_steps["critic"] == cfg.critic_epochs * cfg.critic_minibatches
    assert math.isfinite(metrics["actor_loss"])
    assert math.isfinite(metrics["critic_loss"])
    assert math.isfinite(metrics["wm_loss"])
    assert "actor_reward_rollout" in metrics and "actor_reward_bootstrap" in metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Phase 3 bf16 CUDA Graph check")
def test_phase3_bf16_wm_cuda_graph_matches_eager():
    cfg = make_v34_cfg(profile_path=V35_P3_PROFILE, cost_penalty=0.25, horizon=8, amp="bf16")
    torch.manual_seed(789)
    eager, replay = populate_replay(cfg, n_envs=10, steps=14, device="cuda")
    graph = Trainer(cfg, eager.noise_cfg, eager.context, eager.limits, eager.base_inverse, "cuda", 250, wm_cuda_graph=True)
    graph.wm.load_state_dict(eager.wm.state_dict())
    batch = replay.sample(cfg.wm_batch, cfg.wm_window, include_events=False)

    from viserdex.supermimic.wm_cuda_graph import CapturedWMLoss
    captured = CapturedWMLoss(graph, batch)
    eager.opt_wm.zero_grad(set_to_none=True)
    eager.opt_failure.zero_grad(set_to_none=True)
    loss_e, err_e, phys_e, _, _ = eager._wm_loss(batch)
    loss_e.backward()
    eager._independent_failure_loss.backward()
    loss_g, err_g, phys_g, _, _ = captured(batch)
    torch.testing.assert_close(loss_g, loss_e, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(err_g, err_e, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(phys_g, phys_e, atol=2e-3, rtol=2e-3)
    for (n, pe), (_, pg) in zip(eager.wm.named_parameters(), graph.wm.named_parameters()):
        torch.testing.assert_close(pg.grad, pe.grad, atol=5e-3, rtol=5e-3, msg=f"bf16 grad mismatch on {n}")


V36_P4_PROFILE = Path(__file__).resolve().parents[1] / "configs/supermimic_v34/control_opt_p4.yaml"


def test_phase4_identified_hand_profiles_and_checksums():
    from viserdex.supermimic.identified_hand import SUPPORTED_PROFILES, load_identified_hand
    assert set(SUPPORTED_PROFILES) == {"fit_03", "revo3_sysid_fresh50hz_v1", "revo3_sysid_20261124"}
    p_fit, info_fit = load_identified_hand("fit_03")
    p_v1, info_v1 = load_identified_hand("revo3_sysid_fresh50hz_v1")
    p_24, info_24 = load_identified_hand("revo3_sysid_20261124")
    assert info_fit["model"] == "fit_03"
    assert info_v1["model"] == "revo3_sysid_fresh50hz_v1"
    assert info_24["model"] == "revo3_sysid_20261124"
    for p in (p_v1, p_24):
        assert len(p["effective_kp_Nm_rad"]) == 21
        assert len(p["effective_kd_Nm_s_rad"]) == 21
        assert len(p["armature_kg_m2"]) == 21
        assert len(p["effective_effort_limit_Nm"]) == 21
        assert min(p["effective_kp_Nm_rad"]) > 3.0 and max(p["effective_kp_Nm_rad"]) < 10.0
        assert min(p["effective_effort_limit_Nm"]) >= 0.5 and max(p["effective_effort_limit_Nm"]) <= 1.2
    assert p_v1["effective_kp_Nm_rad"] != p_fit["effective_kp_Nm_rad"]


def test_phase4_twin_conservative_critic_ema_huber_and_residual_wm():
    from viserdex.supermimic.networks import ResBlock
    from viserdex.supermimic.absolute_policy import scheduled_exploration_std_rad
    cfg = make_v34_cfg(profile_path=V36_P4_PROFILE, cost_penalty=0.35, horizon=8)
    assert cfg.identified_hand_profile == "revo3_sysid_fresh50hz_v1"
    assert cfg.wm_huber_delta == 2.0
    assert cfg.wm_residual is True
    assert cfg.weight_decay == 0.0001
    assert cfg.wm_updates == 8
    assert cfg.critic_ensemble_size == 2
    assert cfg.critic_ema_tau == 0.05
    assert cfg.critic_huber_delta == 0.5
    assert cfg.has_target_critic is True and cfg.uses_paper_target_critic is True
    assert cfg.lr_decay_iters == 30000 and cfg.lr_min_ratio == 0.15
    proto = imagination_protocol(cfg)
    for tag in ("_sysid", "_huber", "_wres", "_twin2", "_cema", "_chuber"):
        assert tag in proto, f"Missing {tag} in {proto}"

    trainer, replay = populate_replay(cfg, n_envs=10, steps=14)
    assert any(isinstance(m, ResBlock) for m in trainer.wm.net.modules())
    assert isinstance(trainer.opt_wm, torch.optim.AdamW)
    assert isinstance(trainer.opt_actor, torch.optim.AdamW)
    assert isinstance(trainer.opt_critic, torch.optim.AdamW)
    assert hasattr(trainer, "target_critic") and hasattr(trainer.critic, "nets") and len(trainer.critic.nets) == 2

    # Verify conservative aggregation: reward = min across heads, costs = max across heads.
    batch = replay.sample(8, include_events=False)
    s, ctx = batch["state"], trainer.context[batch["env_id"]]
    with torch.no_grad():
        trainer.critic.nets[0][-1].bias.copy_(torch.tensor([5.0, 1.0]))
        trainer.critic.nets[1][-1].bias.copy_(torch.tensor([3.0, 4.0]))
        all_v = trainer.critic.forward_all(s, ctx)
        agg_v = trainer.critic(s, ctx)
        assert all_v.shape == (2, 8, 2)
        torch.testing.assert_close(agg_v[:, 0], torch.minimum(all_v[0, :, 0], all_v[1, :, 0]))
        torch.testing.assert_close(agg_v[:, 1:], torch.maximum(all_v[0, :, 1:], all_v[1, :, 1:]))
        # Reset bias before running update + Polyak EMA check.
        for net in trainer.critic.nets:
            net[-1].bias.zero_()
        trainer.target_critic.load_state_dict(trainer.critic.state_dict())

    target_before = [p.clone() for p in trainer.target_critic.parameters()]
    metrics = trainer.update(replay)
    assert math.isfinite(metrics["wm_loss"])
    assert math.isfinite(metrics["critic_loss"])
    assert math.isfinite(metrics["actor_loss"])
    # Polyak EMA moved target_critic toward critic without copying 100%.
    diffs_prev = [float((p - b).abs().max()) for p, b in zip(trainer.target_critic.parameters(), target_before)]
    diffs_live = [float((p - c).abs().max()) for p, c in zip(trainer.target_critic.parameters(), trainer.critic.parameters())]
    assert max(diffs_prev) > 0.0
    assert max(diffs_live) > 0.0

    # Verify cosine exploration annealing schedule.
    assert math.isclose(scheduled_exploration_std_rad(cfg, 0), math.radians(2.5), rel_tol=1e-6)
    mid = scheduled_exploration_std_rad(cfg, cfg.wm_only_iters + cfg.exploration_decay_iters // 2)
    assert math.isclose(mid, math.radians(1.5), rel_tol=1e-4)
    end = scheduled_exploration_std_rad(cfg, cfg.wm_only_iters + cfg.exploration_decay_iters + 500)
    assert math.isclose(end, math.radians(0.5), rel_tol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Phase 4 bf16 CUDA Graph check")
def test_phase4_huber_resmlp_bf16_wm_cuda_graph_matches_eager():
    cfg = make_v34_cfg(profile_path=V36_P4_PROFILE, cost_penalty=0.35, horizon=8, amp="bf16")
    torch.manual_seed(901)
    eager, replay = populate_replay(cfg, n_envs=10, steps=14, device="cuda")
    graph = Trainer(cfg, eager.noise_cfg, eager.context, eager.limits, eager.base_inverse, "cuda", 250, wm_cuda_graph=True)
    graph.wm.load_state_dict(eager.wm.state_dict())
    batch = replay.sample(cfg.wm_batch, cfg.wm_window, include_events=False)

    from viserdex.supermimic.wm_cuda_graph import CapturedWMLoss
    captured = CapturedWMLoss(graph, batch)
    eager.opt_wm.zero_grad(set_to_none=True)
    eager.opt_failure.zero_grad(set_to_none=True)
    loss_e, err_e, phys_e, _, _ = eager._wm_loss(batch)
    loss_e.backward()
    eager._independent_failure_loss.backward()
    loss_g, err_g, phys_g, _, _ = captured(batch)
    torch.testing.assert_close(loss_g, loss_e, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(err_g, err_e, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(phys_g, phys_e, atol=2e-3, rtol=2e-3)
    for (n, pe), (_, pg) in zip(eager.wm.named_parameters(), graph.wm.named_parameters()):
        torch.testing.assert_close(pg.grad, pe.grad, atol=5e-3, rtol=5e-3, msg=f"bf16 grad mismatch on {n}")
