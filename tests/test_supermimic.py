"""Pure Torch contract tests; run without launching Isaac Sim."""

from dataclasses import asdict
import copy
import torch
import pytest
from viserdex.supermimic.core import *
from viserdex.supermimic.sensors import initial_sensors, reset_sensors, observe
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.trainer import Trainer, frozen, lambda_returns
from viserdex.supermimic.networks import Dynamics
from viserdex.supermimic.validation import audit_real_reward
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.distillation.noise import JointObservationNoise, PerceptionNoise


def fixture(n=6):
    s = torch.zeros(n, STATE_DIM)
    s[:, 48] = 1
    s[:, 166] = 1
    s[:, TIME] = 1
    s[:, POS] = torch.tensor([0.02, 0.03, 0.04])
    limits = torch.tensor([[-1.0, 1.0]]).expand(n, 21, 2).clone()
    return s, limits


def config(**kwargs):
    return Config(
        wm_hidden=(32, 32),
        actor_hidden=(32, 16),
        critic_hidden=(16,),
        wm_batch=6,
        actor_batch=6,
        wm_updates=1,
        critic_epochs=2,
        critic_minibatches=3,
        wm_only_iters=0,
        microbatch=2,
        buffer_steps=20,
        warmup_steps=8,
        **kwargs,
    )


def test_so3_roundtrip_and_zero_gradient():
    v = torch.randn(20, 3) * 0.3
    torch.testing.assert_close(rotation_log(rotation_vector_quat(v)), v)
    for vector in (torch.zeros(4, 3), torch.tensor([[3.14159, 0.0, 0.0]])):
        vector.requires_grad_()
        rotation_log(rotation_vector_quat(vector)).sum().backward()
        assert torch.isfinite(vector.grad).all()


def test_physical_delta_roundtrip_and_sign_invariance():
    s, _ = fixture()
    delta = torch.randn(6, 54) * 0.02
    nxt = integrate_physical(s, delta)
    torch.testing.assert_close(physical_delta(s, nxt), delta, atol=1e-6, rtol=1e-5)
    flipped = s.clone()
    flipped[:, ROT] *= -1
    torch.testing.assert_close(physical_features(s), physical_features(flipped))
    assert full_features(s).shape == (6, 177)


def test_ema_and_reward_boundaries():
    s, limits = fixture()
    s[:, EMA] = 0.2
    a = torch.full((6, 21), 2.0)
    torch.testing.assert_close(ema_target(s, a, limits), torch.full((6, 21), 0.96))
    physical = s[:, :55].clone()
    s[:, HOLD] = 14  # final qualifying sample at 30 Hz
    expected = (10 + 250 - 0.0001 * 21 * 4 - 0.01 * 21 * 4) / 30
    torch.testing.assert_close(reward(s, physical, a), torch.full((6,), expected))
    physical[:, ROT] = rotation_vector_quat(torch.tensor([[0.1001, 0.0, 0.0]]).expand(6, -1))
    assert (reward(s, physical, a) < 1).all()


def test_history_is_oldest_first_and_reset_local():
    h = torch.arange(10.0).reshape(1, 10, 1).expand(2, -1, 124).clone()
    o = torch.full((2, 124), 10.0)
    new = push_history(h, o, torch.tensor([False, True]))
    torch.testing.assert_close(new[0, :, 0], torch.arange(1.0, 11.0))
    assert (new[1] == 10).all()
    assert (h[:, 0] == 0).all()


def test_reward_audit_accepts_frame_roundoff_but_rejects_wrong_reward_or_goal():
    s, _ = fixture(2)
    s[:, HOLD] = 14
    physical = s[:, :55].clone()
    physical[:, ROT] = rotation_vector_quat(torch.tensor([[0.1, 0.0, 0.0]]).expand(2, -1))
    action = torch.zeros(2, 21)
    local = rotation_log(quat_mul(physical[:, ROT], quat_conjugate(s[:, GOAL_ROT]))).norm(dim=-1)
    # Move to the other side of the discontinuity by only a few FP32 ulps.
    world = torch.where(local < 0.1, torch.full_like(local, 0.1000001), torch.full_like(local, 0.0999999))
    actual = reward_from_orientation_error(s, physical, action, world)
    raw = reward(s, physical, action)
    torch.testing.assert_close((raw - actual).abs(), torch.full_like(raw, 250 / 30), atol=1e-5, rtol=1e-5)
    audit = audit_real_reward(s, physical, action, world, actual, 1 / 30)
    assert audit["success_boundary_disagreements"] == 2
    for wrong in (raw, actual + 0.01):
        with pytest.raises(AssertionError):
            audit_real_reward(s, physical, action, world, wrong, 1 / 30)
    with pytest.raises(AssertionError):
        audit_real_reward(s, physical, action, world + 0.001, actual, 1 / 30)


def test_sensor_process_matches_existing_implementation_over_resets():
    cfg = PerceptionNoiseCfg()
    s, limits = fixture()
    joint = JointObservationNoise(6, 21, "cpu", cfg)
    pose = PerceptionNoise(6, "cpu", cfg)
    d = initial_sensors(6, "cpu", cfg)

    def copy_state():
        for name in ("bias", "history", "delay", "fresh"):
            key = {"bias": "joint_bias", "history": "joint_history", "fresh": "joint_fresh"}.get(name, name)
            d[key] = getattr(joint, name).clone()
        for name in ("age", "period", "jitter", "failure", "bias", "held_pos", "held_quat", "fresh"):
            d[name] = getattr(pose, name).clone()

    copy_state()
    for t in range(14):
        if t == 7:
            mask = torch.tensor([True, False, True, False, False, True])
            joint.reset(mask)
            pose.reset(mask)
            copy_state()
        s[:, Q] = torch.randn(6, 21) * 0.1
        s[:, POS] += 0.001
        rng = torch.get_rng_state()
        joints = joint(s[:, Q])
        p, q = pose(s[:, POS], s[:, ROT])
        expected = torch.cat(
            (joints, s[:, ACTIONS], s[:, GOAL_ROT], s[:, BASE_POS], s[:, TIME], pose_features(p, q, s[:, GOAL_ROT])), -1
        )
        torch.set_rng_state(rng)
        actual, d = observe(s, d, limits, cfg)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-5)
        torch.testing.assert_close(d["age"], pose.age)
        torch.testing.assert_close(d["joint_history"], joint.history)


def test_sensor_bptt_has_no_inplace_version_errors():
    cfg = PerceptionNoiseCfg()
    s, limits = fixture()
    s.requires_grad_()
    d = initial_sensors(6, "cpu", cfg)
    d = reset_sensors(d, torch.ones(6, dtype=torch.bool), cfg)
    losses = []
    for _ in range(12):
        o, d = observe(s, d, limits, cfg)
        losses.append(o.square().mean())
    sum(losses).backward()
    assert torch.isfinite(s.grad).all()


def make_replay(cfg, n=6):
    s, limits = fixture(n)
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    sensors = initial_sensors(n, "cpu", noise)
    o, sensors = observe(s, sensors, limits, noise)
    replay = Replay(cfg.buffer_steps, n, "cpu")
    for t in range(20):
        transition = dict(
            state=s.clone(),
            next_physical=s[:, :55].clone(),
            next_aux=s[:, 55:].clone(),
            action=torch.zeros(n, 21),
            history=o[:, None].expand(-1, 10, -1).clone(),
            done=torch.full((n,), t % 7 == 3),
            reward=torch.zeros(n),
            terminated=torch.zeros(n, dtype=torch.bool),
        )
        transition.update({"sensor/" + k: v for k, v in sensors.items()})
        replay.add(transition)
    return replay, s, limits, noise


def test_ring_wrap_and_terminal_masks():
    r = Replay(5, 1, "cpu")
    for t in range(9):
        r.add(
            dict(
                state=torch.tensor([[float(t)]]),
                next_physical=torch.zeros(1, 55),
                action=torch.zeros(1, 21),
                done=torch.tensor([t == 6]),
            )
        )
    b = r.sample(100, 4)
    assert (b["state"][:, :, 0].diff(dim=1) == 1).all()
    assert (b["state"] >= 4).all()
    for i in range(100):
        terminal = (b["state"][i, :, 0] == 6).nonzero().flatten()
        if len(terminal):
            k = int(terminal[0])
            assert (b["mask"][i, : k + 1] == 1).all()
            assert (b["mask"][i, k + 1 :] == 0).all()


def test_lambda_returns_terminal_and_timeout_bootstrap():
    r = torch.tensor([[1.0], [2.0], [3.0]])
    c = torch.tensor([[1.0], [0.0], [1.0]])
    v = torch.ones(4, 1) * 10
    out = lambda_returns(r, c, v, 0.9, 1.0)
    torch.testing.assert_close(out[:, 0], torch.tensor([2.8, 2.0, 12.0]))


def test_gradient_through_frozen_world_model():
    cfg = config()
    s, limits = fixture()
    model = Dynamics(2, cfg)
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.01)
    action = torch.zeros(6, 21, requires_grad=True)
    with frozen(model):
        nxt = model(s, ema_target(s, action, limits), torch.ones(6, 2))
        nxt[:, POS].sum().backward()
    assert action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())


def test_optimizer_counts_no_leak_and_checkpoint_contract(tmp_path):
    cfg = config()
    r, s, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    trainer.update_real_statistics(s, torch.zeros(6, 21), s[:, :55], r.data["history"][0, :, 0])
    wm_before = copy.deepcopy(trainer.wm.state_dict())
    metrics = trainer.policy_update(r)
    for k, v in trainer.wm.state_dict().items():
        torch.testing.assert_close(v, wm_before[k], rtol=0, atol=0)
    assert trainer.optimizer_steps == dict(wm=0, actor=1, critic=6)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    trainer.wm_update(r)
    assert trainer.optimizer_steps["wm"] == 1
    metadata = dict(environment_contract="a", joint_names=[], context_schema={}, observation_protocol="v1")
    trainer.save(tmp_path / "model.pt", metadata)
    payload = torch.load(tmp_path / "model.pt", weights_only=False)
    torch.rand(13)
    evaluation_rng = torch.get_rng_state().clone()
    trainer.restore(payload, metadata, restore_rng=False)
    torch.testing.assert_close(torch.get_rng_state(), evaluation_rng, atol=0, rtol=0)
    trainer.restore(payload, metadata)
    torch.testing.assert_close(torch.get_rng_state(), payload["rng_cpu"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="contract changed"):
        trainer.restore(payload, {**metadata, "environment_contract": "b"})


def test_wm_microbatch_accumulation_matches_full_batch():
    cfg = config()
    r, s, limits, noise = make_replay(cfg)
    a = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    b = copy.deepcopy(a)
    b.cfg = copy.deepcopy(cfg)
    b.cfg.microbatch = 6
    rng = torch.get_rng_state()
    a.wm_update(r)
    torch.set_rng_state(rng)
    b.wm_update(r)
    for p, q in zip(a.wm.parameters(), b.wm.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph needs GPU")
def test_wm_cuda_graph_matches_eager_with_new_samples_and_normalizers():
    cfg = config(wm_microbatch=6)
    r, _, limits, noise = make_replay(cfg)
    r.device = "cuda:0"
    a = Trainer(cfg, noise, torch.ones(6, 2, device='cuda'), limits.cuda(),
                torch.tensor([[0., 0., 0., 1.]], device='cuda').expand(6, -1), 'cuda:0')
    with torch.no_grad():
        a.wm.net[-1].weight.normal_(0, .001)
    b = copy.deepcopy(a)
    b.wm_cuda_graph = True
    for _ in range(3):
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state()
        eager = a.wm_update(r)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        captured = b.wm_update(r)
        assert captured['wm_loss'] == pytest.approx(eager['wm_loss'], rel=2e-5, abs=1e-6)
        assert captured['wm_grad_norm'] == pytest.approx(eager['wm_grad_norm'], rel=2e-5, abs=1e-6)
        for p, q in zip(a.wm.parameters(), b.wm.parameters()):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
        # Model buffers evolve after real data collection; policy updates clear
        # WM gradients. Capture must use the new values and restore grad storage.
        for trainer in (a, b):
            trainer.opt_wm.zero_grad(set_to_none=True)
            trainer.wm.delta_norm.mean.add_(.0001)
            trainer.wm.delta_norm.variance.mul_(.99)
    assert a.optimizer_steps == b.optimizer_steps == dict(wm=3, actor=0, critic=0)


def test_world_model_action_derivative_matches_finite_difference():
    torch.manual_seed(321)
    cfg = config()
    s, limits = fixture()
    wm = Dynamics(2, cfg)
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, 0.02)
    a = torch.randn(6, 21) * 0.1
    a.requires_grad_()
    context = torch.ones(6, 2)

    def f(action):
        return wm(s, ema_target(s, action, limits), context)[:, POS].sum()

    grad = torch.autograd.grad(f(a), a)[0]
    direction = torch.randn_like(a)
    direction = direction / direction.norm()
    eps = 0.002
    finite = (f(a + eps * direction) - f(a - eps * direction)) / (2 * eps)
    torch.testing.assert_close((grad * direction).sum(), finite, atol=1e-4, rtol=1e-2)


def test_imagination_excludes_post_timeout_critic_samples():
    cfg = config()
    r, s, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    batch = r.sample(6)
    batch["state"][:, TIME] = 1 / 600
    _, _, targets, _, valid = trainer.imagine(batch)
    assert valid[0].all() and not valid[1:].any()
    assert torch.isfinite(targets).all() and not targets.requires_grad


def test_statistics_never_update_in_imagination():
    cfg = config()
    r, s, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    counts = [
        m.count.clone()
        for net in (trainer.wm, trainer.actor, trainer.critic)
        for m in net.modules()
        if hasattr(m, "count")
    ]
    trainer.imagine(r.sample(6))
    now = [m.count for net in (trainer.wm, trainer.actor, trainer.critic) for m in net.modules() if hasattr(m, "count")]
    for a, b in zip(counts, now):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_domain_context_preconditioning_retains_tiny_inertias():
    cfg = config()
    s, limits = fixture()
    context = torch.arange(1.0, 7.0)[:, None] * torch.tensor([[1e-7, 1e-3, 2.0]])
    trainer = Trainer(
        cfg, PerceptionNoiseCfg(), context, limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu"
    )
    features = trainer.wm.inputs(s, torch.zeros(6, 21), context)[:, -3:]
    torch.testing.assert_close(features[:, 0], features[:, 1])
    torch.testing.assert_close(features[:, 0], features[:, 2])


def test_warmup_updates_only_wm_then_joint_optimizer_counts():
    cfg = config()
    cfg.wm_only_iters = 1
    r, s, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    trainer.update(r)
    assert trainer.optimizer_steps == dict(wm=1, actor=0, critic=0)
    trainer.update(r)
    assert trainer.optimizer_steps == dict(wm=2, actor=1, critic=6)


def test_completed_hold_switch_keeps_action_history_and_uses_angle_threshold():
    s, limits = fixture(2)
    s[:, HOLD] = 14
    s[:, 76:160] = torch.arange(84.0)
    physical = s[:, :55].clone()
    physical[1, ROT] = rotation_vector_quat(torch.tensor([[0.11, 0.0, 0.0]]))[0]
    action = torch.ones(2, 21) * 0.25
    goal = rotation_vector_quat(torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    nxt = advance(s, physical, action, ema_target(s, action, limits), goal)
    torch.testing.assert_close(nxt[0, GOAL_ROT], goal[0])
    torch.testing.assert_close(nxt[1, GOAL_ROT], s[1, GOAL_ROT])
    torch.testing.assert_close(nxt[:, 97:160], s[:, 76:139])
    torch.testing.assert_close(nxt[:, 168], torch.tensor([1.0, 0.0]))


def test_bf16_mlp_scope_and_resume_continuation(tmp_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = config(amp="bf16", wm_microbatch=6, actor_microbatch=6, critic_microbatch=64)
    r, s, limits, noise = make_replay(cfg)
    r.device = device
    context = torch.ones(6, 2, device=device)
    limits = limits.to(device)
    inverse = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).expand(6, -1)
    trainer = Trainer(cfg, noise, context, limits, inverse, device)
    dtypes = []
    handle = trainer.actor.net[-1].register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    history = r.data["history"][0].to(device)
    assert trainer.actor(history).dtype == torch.float32
    assert dtypes == [torch.bfloat16]
    handle.remove()
    trainer.update(r)
    assert trainer.optimizer_steps == dict(wm=1, actor=1, critic=6)
    for model in (trainer.wm, trainer.actor, trainer.critic):
        assert model.amp == "bf16"
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(p.grad is None or p.grad.dtype == torch.float32 for p in model.parameters())
    # Frozen WM still provides finite nonzero derivatives to FP32 actions.
    derivative_wm = copy.deepcopy(trainer.wm)
    with torch.no_grad():
        # This fixture has zero physical change, so the zero-initialized output
        # layer need not learn a response; exercise a known nonzero mapping.
        derivative_wm.net[-1].weight.normal_(0, 0.01)
    action = torch.zeros(6, 21, device=device, requires_grad=True)
    with frozen(derivative_wm):
        derivative_wm(s.to(device), ema_target(s.to(device), action, limits), context)[:, POS].sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    metadata = dict(environment_contract="a", joint_names=[], context_schema={}, observation_protocol="v1")
    trainer.save(tmp_path / "bf16.pt", metadata)
    payload = torch.load(tmp_path / "bf16.pt", map_location="cpu", weights_only=False)
    assert payload["version"] == 17 and payload["config"]["amp"] == "bf16"
    restored_cfg = Config(**payload["config"])
    restored = Trainer(restored_cfg, noise, context, limits, inverse, device)
    trainer.restore(copy.deepcopy(payload), metadata)
    trainer.update(r)
    restored.restore(copy.deepcopy(payload), metadata)
    restored.update(r)
    for name in ("wm", "actor", "critic"):
        for original, resumed in zip(getattr(trainer, name).parameters(), getattr(restored, name).parameters()):
            torch.testing.assert_close(original, resumed, atol=1e-7, rtol=1e-6)
    assert trainer.optimizer_steps == restored.optimizer_steps == dict(wm=2, actor=2, critic=12)
    restored.cfg = copy.deepcopy(restored.cfg)
    restored.cfg.amp = "none"
    with pytest.raises(ValueError, match="configuration changed"):
        restored.restore(payload, metadata)


def test_legacy_fp32_checkpoint_execution_defaults(tmp_path):
    cfg = config()
    _, _, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(6, -1), "cpu")
    metadata = dict(environment_contract="a", joint_names=[], context_schema={}, observation_protocol="v1")
    trainer.save(tmp_path / "legacy.pt", metadata)
    payload = torch.load(tmp_path / "legacy.pt", weights_only=False)
    payload["version"] = 2
    for key in ("amp", "wm_microbatch", "actor_microbatch", "critic_microbatch"):
        payload["config"].pop(key)
    with pytest.raises(ValueError, match="Version 17 required"):
        trainer.restore(payload, metadata, restore_rng=False)


def test_resume_microbatch_override_preserves_training_contract(tmp_path):
    cfg = config()
    r, _, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits,
                      torch.tensor([[0., 0., 0., 1.]]).expand(6, -1), "cpu")
    trainer.update(r)
    metadata = dict(environment_contract="a", joint_names=[], context_schema={}, observation_protocol="v1")
    trainer.save(tmp_path / "resume.pt", metadata)
    payload = torch.load(tmp_path / "resume.pt", weights_only=False)
    trainer.cfg = copy.deepcopy(cfg)
    trainer.cfg.wm_microbatch = 6
    with pytest.raises(ValueError, match="configuration changed"):
        trainer.restore(payload, metadata)
    trainer.restore(payload, metadata, allow_microbatch_override=True)
    assert trainer.iteration == 1 and trainer.optimizer_steps == dict(wm=1, actor=1, critic=6)
    for name in ("wm", "actor", "critic"):
        for key, value in getattr(trainer, name).state_dict().items():
            torch.testing.assert_close(value.cpu(), payload[name][key])
    assert all(state['step'] == 1 for state in trainer.opt_wm.state.values())
    trainer.cfg.wm_batch *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        trainer.restore(payload, metadata, allow_microbatch_override=True)


def test_gamma_resume_only_before_policy_training(tmp_path):
    cfg = config()
    cfg.wm_only_iters = 500
    r, _, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6, 2), limits,
                      torch.tensor([[0., 0., 0., 1.]]).expand(6, -1), "cpu")
    trainer.update(r)
    metadata = dict(environment_contract="a", joint_names=[], context_schema={}, observation_protocol="v1")
    trainer.save(tmp_path / "warmup.pt", metadata)
    payload = torch.load(tmp_path / "warmup.pt", weights_only=False)
    trainer.cfg = copy.deepcopy(cfg)
    trainer.cfg.gamma = .98
    with pytest.raises(ValueError, match="configuration changed"):
        trainer.restore(payload, metadata, allow_microbatch_override=True)
    trainer.restore(payload, metadata, allow_gamma_override=True)
    assert trainer.optimizer_steps == dict(wm=1, actor=0, critic=0)
    assert trainer.cfg.gamma == .98 and payload['config']['gamma'] == .99
    payload['optimizer_steps']['critic'] = 1
    with pytest.raises(ValueError, match="untrained actor and critic"):
        trainer.restore(payload, metadata, allow_gamma_override=True)
