"""Readiness and audit regressions without Isaac Sim or a CUDA device."""
import copy
import math

import pytest
import torch

from test_supermimic import config, fixture, make_replay
from viserdex.supermimic.core import POS, ROT, reward, rotation_vector_quat
from viserdex.supermimic.trainer import Trainer, checked_step
from viserdex.supermimic.stream_audit import StreamAudit, audit_rollout


def trainer_and_replay():
    cfg = config(wm_window=2, horizon=2)
    replay, state, limits, noise = make_replay(cfg)
    # The freshly initialized WM predicts persistence; give it a real residual
    # so the freeze test verifies an actual optimizer change rather than zero loss.
    replay.data["next_physical"][:, :, 42].add_(.01)
    inverse = torch.tensor([0., 0., 0., 1.]).expand(6, -1)
    trainer = Trainer(cfg, noise, torch.ones(6, 1), limits, inverse, "cpu")
    return trainer, replay, state


def test_readiness_refits_wm_but_preserves_policy_optimizers_and_training_clock():
    trainer, replay, _ = trainer_and_replay()
    trainer.update(replay)  # Populate all three Adam states before the freeze.
    trainer.iteration, trainer.step_credit, trainer.real_steps = 1510, .25, 123
    before = {name: copy.deepcopy(getattr(trainer, name).state_dict())
              for name in ("wm", "actor", "critic", "opt_actor", "opt_critic")}
    counts = dict(trainer.optimizer_steps)
    result = trainer.update_world_model(replay, updates=3)
    assert (trainer.iteration, trainer.step_credit, trainer.real_steps) == (1510, .25, 123)
    assert trainer.optimizer_steps == {**counts, "wm": counts["wm"] + 3}
    for name in ("actor", "critic", "opt_actor", "opt_critic"):
        torch.testing.assert_close(getattr(trainer, name).state_dict(), before[name], rtol=0, atol=0)
    assert any(not torch.equal(value, before["wm"][name]) for name, value in trainer.wm.state_dict().items())
    assert result["readiness_wm_updates"] == 3
    assert isinstance(result["wm_loss"], float)
    assert not any(key.startswith("actor_") or key.startswith("critic_") for key in result)


def test_readiness_statistics_can_freeze_effective_policy_normalizers():
    trainer, _, state = trainer_and_replay()
    actor = copy.deepcopy(trainer.actor.state_dict())
    critic = copy.deepcopy(trainer.critic.state_dict())
    trainer.update_real_statistics(state, state[:, 55:76], state[:, :55], torch.ones(6, 124),
                                   policy_statistics=False)
    assert trainer.wm.input_norm.count == 6 and trainer.wm.delta_norm.count == 6
    torch.testing.assert_close(trainer.actor.state_dict(), actor, rtol=0, atol=0)
    torch.testing.assert_close(trainer.critic.state_dict(), critic, rtol=0, atol=0)


def test_batched_wm_diagnostics_preserve_updates_and_last_step_metrics():
    trainer, replay, _ = trainer_and_replay()
    reference = copy.deepcopy(trainer)
    rng = torch.get_rng_state()
    actual = trainer.update_world_model(replay, updates=3)
    torch.set_rng_state(rng)
    for _ in range(3):
        expected = reference.wm_update(replay)
    for key, value in expected.items():
        assert actual[key] == value
    torch.testing.assert_close(trainer.wm.state_dict(), reference.wm.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(trainer.opt_wm.state_dict(), reference.opt_wm.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_readiness_rejects_invalid_update_counts(count):
    trainer, replay, _ = trainer_and_replay()
    with pytest.raises(ValueError, match="positive integer"):
        trainer.update_world_model(replay, updates=count)


def test_nonfinite_gradient_guard_remains_per_optimizer_step(monkeypatch):
    trainer, replay, _ = trainer_and_replay()
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def checked(*args, **kwargs):
        calls.append(kwargs.get("error_if_nonfinite"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", checked)
    trainer.update_world_model(replay, updates=3)
    assert calls == [True, True, True]
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    before = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        checked_step(model, optimizer, tensor_norm=True)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert not optimizer.state


def test_critic_diagnostic_batching_preserves_adam_and_fixed_targets():
    trainer, replay, _ = trainer_and_replay()
    reference = copy.deepcopy(trainer)
    batch = replay.sample(11)
    states = batch["state"]
    context = trainer.context[batch["env_id"]]
    targets = torch.linspace(-.5, .8, len(states))
    fixed_targets = targets.clone()
    rng = torch.get_rng_state()
    actual = trainer.fit_critic((states, targets, context))
    torch.set_rng_state(rng)
    losses = []
    for _ in range(reference.cfg.critic_epochs):
        for indices in torch.tensor_split(torch.randperm(len(states)), reference.cfg.critic_minibatches):
            reference.opt_critic.zero_grad(set_to_none=True)
            total = 0.
            for chunk in indices.split(reference.cfg.microbatch):
                loss = (reference.critic(states[chunk], context[chunk]) - targets[chunk]).square().sum() / len(indices)
                loss.backward()
                total += float(loss.detach())
            checked_step(reference.critic, reference.opt_critic)
            losses.append(total)
    torch.testing.assert_close(trainer.critic.state_dict(), reference.critic.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(trainer.opt_critic.state_dict(), reference.opt_critic.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(targets, fixed_targets, rtol=0, atol=0)
    assert actual["critic_loss"] == pytest.approx(sum(losses) / len(losses), rel=1e-12, abs=1e-12)


class Persistence(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("offset", torch.tensor(0.))

    def forward(self, state, target, context):
        result = state[:, :55].clone()
        result[:, 42] += self.offset
        return result


def audit_record(n, cfg):
    state, limits = fixture(n)
    action = torch.zeros(n, 21)
    physical = state[:, :55].clone()
    row = dict(state=state, next_physical=physical, action=action,
               done=torch.zeros(n, dtype=torch.bool), reward=reward(state, physical, action))
    return row, limits


def test_stream_audit_starts_immediately_then_covers_all_worlds_without_rng_change(tmp_path, monkeypatch):
    monkeypatch.setattr("viserdex.supermimic.stream_audit.Dynamics", Persistence)
    cfg = config()
    row, limits = audit_record(1024, cfg)
    model = Persistence()
    rng = torch.get_rng_state()
    audit = StreamAudit(tmp_path / "audit.jsonl", torch.ones(1024, 1), limits, cfg, None)
    seen = set()
    for window in range(8):
        iteration = 1510 + 200 * window
        result = None
        for _ in range(32):
            result = audit.observe(model, row, iteration)
        assert result is not None and audit.latest_result is result
        assert result["iteration"] == iteration
        assert result["metrics"]["8"]["object_position_vector_p99_m"] == 0
        assert len(result["environment_ids"]) == 128
        if window == 1:
            assert 801 in result["environment_ids"]
        seen.update(result["environment_ids"])
        assert audit.observe(model, row, iteration) is None
    assert seen == set(range(1024))
    assert audit.completed == 8
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 8
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_explicit_audit_uses_frozen_model_and_supports_fixed_iteration_gate(tmp_path, monkeypatch):
    monkeypatch.setattr("viserdex.supermimic.stream_audit.Dynamics", Persistence)
    cfg = config()
    row, limits = audit_record(6, cfg)
    model = Persistence()
    audit = StreamAudit(tmp_path / "audit.jsonl", torch.ones(6, 1), limits, cfg, None)
    audit.start(model, 1510, [1, 5])
    with pytest.raises(ValueError, match="Finish"):
        audit.start(model, 1510)
    model.offset.fill_(.1)
    for _ in range(32):
        result = audit.observe(model, row, 1510)
    assert result["environment_ids"] == [1, 5]
    assert result["metrics"]["8"]["object_position_vector_rmse_m"] == 0
    audit.start(model, 1510, [0])
    for _ in range(32):
        result = audit.observe(model, row, 1510)
    assert result["metrics"]["8"]["object_position_vector_rmse_m"] == pytest.approx(.8)


def test_stream_tanh_slope_uses_actor_output_before_real_angle_noise(tmp_path, monkeypatch):
    from viserdex.supermimic.networks import Actor
    from viserdex.supermimic.absolute_policy import explore_absolute
    from viserdex.supermimic.revised_contract import urdf_limits, CENTER_REVISION
    monkeypatch.setattr('viserdex.supermimic.stream_audit.Dynamics', Persistence)
    cfg = config(control_dt=.04, hold_duration=.52)
    row, _ = audit_record(6, cfg)
    real_limits, _ = urdf_limits(CENTER_REVISION)
    limits = torch.tensor(real_limits, dtype=torch.float32)[None].expand(6, -1, -1)
    actor = Actor(cfg)
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.fill_(8.)
        policy_action = actor(torch.zeros(6, cfg.history, cfg.observation_dim))
    rho = math.exp(-cfg.control_dt/cfg.exploration_correlation_s)
    previous_noise = torch.full_like(policy_action, cfg.exploration_position_std_rad/rho)
    row['action'], noise = explore_absolute(policy_action, previous_noise, limits, cfg,
                                            innovation=torch.zeros_like(policy_action))
    torch.testing.assert_close(noise, torch.full_like(noise, math.radians(.5)))
    expected = float((1-policy_action.square()).mean())
    assert float((1-row['action'].square()).mean()) > expected*10000
    audit = StreamAudit(tmp_path/'slope.jsonl', torch.ones(6,1), limits, cfg, None)
    model = Persistence()
    for _ in range(32):
        result = audit.observe(model, row, 0, policy_action=policy_action)
    assert result['action_tanh_slope_mean'] == pytest.approx(expected, abs=1e-12)
    assert result['action_tanh_slope_valid_samples'] == result['action_tanh_slope_total_samples'] == 192
    assert result['action_near_bound_fraction'] == pytest.approx(float((row['action'].abs()>.98).float().mean()))
    assert not audit.policy_tanh_slopes


@pytest.mark.parametrize('available_steps', [0, 8])
def test_stream_tanh_slope_reports_missing_policy_source(tmp_path, monkeypatch, available_steps):
    monkeypatch.setattr('viserdex.supermimic.stream_audit.Dynamics', Persistence)
    cfg = config()
    row, limits = audit_record(6, cfg)
    audit = StreamAudit(tmp_path/'missing_slope.jsonl', torch.ones(6,1), limits, cfg, None)
    policy_action = torch.full_like(row['action'], .5)
    for step in range(32):
        result = audit.observe(Persistence(), row, 0,
                               policy_action=policy_action if step < available_steps else None)
    assert result['action_tanh_slope_mean'] == (.75 if available_steps else None)
    assert result['action_tanh_slope_valid_samples'] == available_steps*6
    assert result['action_tanh_slope_total_samples'] == 192
    assert not audit.policy_tanh_slopes


def test_audit_tail_errors_respect_pre_reset_targets_and_exclude_later_states():
    cfg = config()
    row, limits = audit_record(4, cfg)
    records = [copy.deepcopy(row) for _ in range(8)]
    records[0]["done"][3] = True
    for record in records:
        record["next_physical"][:, POS] += torch.tensor([.01, .02, .03, 99.])[:, None] * torch.tensor([1., 0., 0.])
        record["next_physical"][:, ROT] = rotation_vector_quat(torch.tensor([[.1, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [1., 0., 0.]]))
    result = audit_rollout(Persistence(), records, torch.ones(4, 1), limits, cfg)
    assert result["1"]["object_position_vector_p99_m"] > 90
    step = result["8"]
    assert step["valid_environments"] == 3 and step["valid_fraction"] == .75
    assert step["object_position_vector_p95_m"] == pytest.approx(.029)
    assert step["object_position_vector_p99_m"] == pytest.approx(.0298)
    assert step["object_orientation_geodesic_p95_rad"] == pytest.approx(.29)
    assert step["persistence_object_orientation_geodesic_p99_rad"] == pytest.approx(.298)


def test_real_audit_model_construction_preserves_random_stream(tmp_path):
    cfg = config()
    _, limits = audit_record(6, cfg)
    rng = torch.get_rng_state()
    StreamAudit(tmp_path / "audit.jsonl", torch.ones(6, 1), limits, cfg, None)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("substeps", [False, True])
def test_compact_resume_allows_only_proven_solver_metadata_change(tmp_path, substeps):
    from test_solver_budget import contracts, substep_proof
    from viserdex.distillation.config import PerceptionNoiseCfg
    from viserdex.supermimic.core import manual_convex_config
    from viserdex.supermimic.solver_budget import digest, substep_experiment_record

    source, target = contracts()
    proof = substep_proof() if substeps else dict(source=source, target=target)
    cfg = manual_convex_config(geometry_enabled=False, compile_geometry=False,
                              wm_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,))
    _, limits = fixture(2)
    trainer = Trainer(cfg, PerceptionNoiseCfg(), torch.ones(2, 1), limits,
                      torch.tensor([0., 0., 0., 1.]).expand(2, -1), "cpu")
    identified = dict(physics_substeps=2, pd_period_s=.005, delay_steps=3,
                      effective_delay_s=.015, encoder_bias_rad=[.001])
    metadata = dict(environment_contract=digest(proof["source"]), joint_names=[], context_schema={},
                    observation_protocol=cfg.observation_protocol, identified_hand=identified)
    trainer.save(tmp_path / "model.pt", metadata)
    payload = torch.load(tmp_path / "model.pt", weights_only=False)
    runtime = copy.deepcopy(metadata)
    runtime["environment_contract"] = digest(proof["target"])
    if substeps:
        runtime["identified_hand"]["solver_substep_experiment"] = substep_experiment_record(identified, 4)
    trainer.restore(payload, runtime, restore_rng=False, solver_iteration_override=proof)
    with pytest.raises(ValueError, match="environment_contract"):
        trainer.restore(payload, runtime, restore_rng=False)
    modified = copy.deepcopy(runtime)
    modified["identified_hand"]["encoder_bias_rad"] = [.01]
    with pytest.raises(ValueError):
        trainer.restore(payload, modified, restore_rng=False, solver_iteration_override=proof)
