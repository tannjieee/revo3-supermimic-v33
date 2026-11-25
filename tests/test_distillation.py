"""CPU regression tests for the recurrent student and DAgger data boundaries."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from viserdex.distillation.config import DistillationCfg, PerceptionNoiseCfg, StudentCfg
from viserdex.distillation.frames import pose_frame_metadata
from viserdex.distillation.model import BeliefStudent
from viserdex.distillation.noise import (
    JointObservationNoise,
    PerceptionNoise,
    pose_features,
    quat_apply,
    quat_mul,
    rotation_vector_quat,
)
from viserdex.distillation.observations import StudentObservations
from viserdex.distillation.teacher import (
    audit_clean_state_observations,
    configure_clean_state_observations,
    teacher_metadata,
    audit_teacher_observations,
    teacher_observations,
)
from viserdex.distillation.trainer import DAggerTrainer, load_student_checkpoint, mix_actions


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(19)
    torch.set_num_threads(1)


def small_model():
    cfg = StudentCfg(
        extero_hidden=(12,),
        extero_latent=8,
        belief_dim=12,
        rnn_hidden=16,
        latent_hidden=(12,),
        gate_hidden=(8,),
        decoder_hidden=(12,),
        actor_hidden=(24,),
    )
    return BeliefStudent(cfg)


def identity(n):
    quat = torch.zeros(n, 4)
    quat[:, 3] = 1
    return quat


def make_batch(model):
    t, n = 5, 3
    return {
        "prop": torch.randn(t, n, 113),
        "extero": torch.randn(t, n, 11),
        "target": torch.randn(t, n, 38),
        "teacher": torch.randn(t, n, 21),
        "hidden": model.initial_state(n, "cpu"),
        "reset": torch.zeros(t, n, dtype=torch.bool),
    }


def test_xyzw_composition_and_noisy_goal_difference():
    rot = rotation_vector_quat(torch.tensor([[0.0, 0.0, torch.pi / 2]]))
    torch.testing.assert_close(
        quat_apply(rot, torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=1e-6
    )
    features = pose_features(torch.zeros(1, 3), rot, identity(1))
    torch.testing.assert_close(features[:, 7:], rot)
    torch.testing.assert_close(pose_features(torch.zeros(1, 3), -rot, identity(1)), features)
    torch.testing.assert_close(quat_mul(identity(1), rot), rot)


def test_no_noise_returns_clean_pose():
    noise = PerceptionNoise(3, "cpu", PerceptionNoiseCfg(enabled=False))
    pos = torch.randn(3, 3)
    out, quat = noise(pos, identity(3))
    torch.testing.assert_close(out, pos)
    torch.testing.assert_close(quat, identity(3))


def test_encoder_bias_is_episode_local_and_does_not_modify_teacher():
    cfg = PerceptionNoiseCfg(joint_noise_std=0)
    noise = JointObservationNoise(4, 21, "cpu", cfg)
    clean = torch.zeros(4, 21)
    first = noise(clean)
    torch.testing.assert_close(noise(clean), first)
    assert torch.count_nonzero(clean) == 0
    assert torch.count_nonzero(first) > 0
    noise.reset(torch.tensor([True, False, True, False]))
    second = noise(clean)
    torch.testing.assert_close(second[[1, 3]], first[[1, 3]])
    assert not torch.equal(second[[0, 2]], first[[0, 2]])


def test_encoder_white_noise_scale_and_disabled_path():
    cfg = PerceptionNoiseCfg(joint_bias_std=0)
    noise = JointObservationNoise(10000, 21, "cpu", cfg)
    clean = torch.zeros(10000, 21)
    first, second = noise(clean), noise(clean)
    assert first.std().item() == pytest.approx(0.005, rel=0.01)
    assert not torch.equal(first, second)
    cfg.joint_enabled = False
    torch.testing.assert_close(noise(clean), clean)


def test_encoder_delay_is_bounded_and_reset_cannot_reuse_previous_episode():
    cfg = PerceptionNoiseCfg(joint_noise_std=0, joint_bias_std=0, joint_delay_max_steps=2)
    noise = JointObservationNoise(2, 21, "cpu", cfg)
    noise.delay.fill_(2)
    for value, expected in ((1, 1), (2, 1), (3, 1), (4, 2)):
        torch.testing.assert_close(noise(torch.full((2, 21), float(value))), torch.full((2, 21), float(expected)))
    noise.reset(torch.tensor([True, False]))
    out = noise(torch.full((2, 21), 9.0))
    torch.testing.assert_close(out[0], torch.full((21,), 9.0))
    torch.testing.assert_close(out[1], torch.full((21,), 3.0))


def test_teacher_audit_rejects_corruption_and_mismatched_observations():
    def term(value, scale=None):
        return SimpleNamespace(
            func=lambda _: value.clone(), params={}, scale=scale, clip=None, modifiers=None, history_length=0
        )

    cfg = SimpleNamespace(
        observations=SimpleNamespace(
            policy=SimpleNamespace(enable_corruption=True, joint=term(torch.ones(2, 21))),
            privileged=SimpleNamespace(enable_corruption=True, velocity=term(torch.ones(2, 27), 0.2)),
        )
    )
    env = SimpleNamespace(
        cfg=cfg,
        observation_manager=SimpleNamespace(
            cfg=cfg.observations, active_terms={"policy": ["joint"], "privileged": ["velocity"]}
        ),
    )
    obs = {"policy": torch.ones(2, 21), "privileged": torch.full((2, 27), 0.2)}
    with pytest.raises(AssertionError, match="corruption"):
        audit_clean_state_observations(env, obs)
    configure_clean_state_observations(cfg)
    assert all(value == 0 for value in audit_clean_state_observations(env, obs).values())
    obs["privileged"][0, 0] += 0.001
    with pytest.raises(AssertionError):
        audit_clean_state_observations(env, obs)


def test_downsampling_and_reset_cannot_hold_previous_episode_pose():
    cfg = PerceptionNoiseCfg(
        position_noise=0,
        position_bias=0,
        rotation_noise_deg=0,
        rotation_bias_deg=0,
        failure_probability=0,
        jitter_probability=0,
    )
    noise = PerceptionNoise(2, "cpu", cfg)
    noise.period.fill_(3)
    noise(torch.ones(2, 3), identity(2))
    out, _ = noise(torch.ones(2, 3) * 2, identity(2))
    torch.testing.assert_close(out, torch.ones(2, 3))
    noise.reset(torch.tensor([True, False]))
    noise.jitter.fill_(1)  # Fresh reset still publishes current episode, even if delayed.
    out, _ = noise(torch.ones(2, 3) * 9, identity(2))
    torch.testing.assert_close(out[0], torch.ones(3) * 9)
    torch.testing.assert_close(out[1], torch.ones(3))


def test_tracking_failure_is_independent_of_current_true_pose():
    noise = PerceptionNoise(4, "cpu", PerceptionNoiseCfg(max_period=1, jitter_probability=0))
    noise.failure.fill_(1)
    state = torch.get_rng_state()
    pos1, quat1 = noise(torch.ones(4, 3), identity(4))
    torch.set_rng_state(state)
    pos2, quat2 = noise(torch.ones(4, 3) * 20, -identity(4))
    torch.testing.assert_close(pos1, pos2)
    torch.testing.assert_close(quat1, quat2)
    torch.testing.assert_close(quat1.norm(dim=-1), torch.ones(4))
    assert pos1.abs().max() <= 0.15


def test_dagger_selects_whole_policy_and_probability_boundaries():
    teacher = torch.ones(256, 21)
    student = -teacher
    assert torch.equal(mix_actions(teacher, student, 1)[0], teacher)
    assert torch.equal(mix_actions(teacher, student, 0)[0], student)
    mixed, mask = mix_actions(teacher, student, 0.5)
    assert 0 < mask.sum() < 256
    assert torch.all(mixed == mixed[:, :1])


def test_recurrent_reset_matches_fresh_policy_and_blocks_gradients():
    model = small_model()
    prop = torch.randn(4, 2, 113, requires_grad=True)
    ext = torch.randn(4, 2, 11)
    resets = torch.zeros(4, 2, dtype=torch.bool)
    resets[2, 0] = True
    hidden = torch.randn(2, 2, 16)
    actions, _, _ = model.sequence(prop, ext, hidden, resets)
    fresh, _, _ = model(prop[2, :1], ext[2, :1], model.initial_state(1, "cpu"))
    torch.testing.assert_close(actions[2, :1], fresh)
    actions[3, 0].sum().backward()
    assert torch.count_nonzero(prop.grad[:2, 0]) == 0
    assert torch.count_nonzero(prop.grad[2:, 0]) > 0


def test_auxiliary_targets_do_not_enter_policy():
    model = small_model().eval()
    batch = make_batch(model)
    first = model(batch["prop"][0], batch["extero"][0], batch["hidden"])[0]
    batch["target"].fill_(1000)
    second = model(batch["prop"][0], batch["extero"][0], batch["hidden"])[0]
    torch.testing.assert_close(first, second)


def test_update_checkpoint_resume_preserves_optimizer_beta_and_normalization(tmp_path):
    model = small_model()
    cfg = DistillationCfg(epochs=2, rollout_steps=5)
    trainer = DAggerTrainer(model, cfg)
    batch = make_batch(model)
    before = {k: v.clone() for k, v in model.named_parameters()}
    metrics = trainer.update(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(before[k], v) for k, v in model.named_parameters())
    assert model.prop_norm.count == 15  # not 30: normalization must not count each learning epoch
    path = tmp_path / "student.pt"
    trainer.save(path, {"test": True}, PerceptionNoiseCfg())
    assert not path.with_suffix(".pt.tmp").exists()
    payload = load_student_checkpoint(path)
    restored = DAggerTrainer(small_model(), cfg)
    restored.restore(payload)
    assert restored.iteration == 1 and restored.total_steps == 15
    assert cfg.beta(restored.iteration) == pytest.approx(0.855)
    assert restored.optimizer.state_dict()["state"]
    # The next complete update must agree, including AdamW moments.
    left = trainer.update(batch)
    right = restored.update(batch)
    assert right == pytest.approx(left)
    for key, value in trainer.student.state_dict().items():
        torch.testing.assert_close(value, restored.student.state_dict()[key])


def test_reject_wrong_checkpoint_and_changed_resume_config(tmp_path):
    path = tmp_path / "ppo.pt"
    torch.save({"actor_state_dict": {}}, path)
    with pytest.raises(ValueError, match="belief-student"):
        load_student_checkpoint(path)
    trainer = DAggerTrainer(small_model(), DistillationCfg(epochs=2))
    with pytest.raises(ValueError, match="preserve"):
        trainer.restore({"training_cfg": asdict(DistillationCfg(epochs=3))})


class Scene(dict):
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def fake_env():
    def proxy(x):
        return SimpleNamespace(torch=x)

    scene = Scene()
    for name, height in (("robot", 0.5), ("object", 0.6)):
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=proxy(scene.env_origins + torch.tensor([[0.0, 0.0, height]])), root_quat_w=proxy(identity(2))
            )
        )
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _: torch.cat((torch.zeros(2, 3), identity(2)), -1)),
        episode_length_buf=torch.tensor([0, 10]),
        max_episode_length=20,
    )


def test_observation_adapter_history_frames_and_external_pose_no_leak():
    env = fake_env()
    adapter = StudentObservations(env, StudentCfg(), PerceptionNoiseCfg(enabled=False, joint_enabled=False))
    obs = {"policy": torch.zeros(2, 49), "privileged": torch.randn(2, 38)}
    estimated = (torch.ones(2, 3) * 0.02, identity(2))
    prop, ext, target = adapter(obs, estimated)
    assert prop.shape == (2, 113) and ext.shape == (2, 11) and target.shape == (2, 38)
    torch.testing.assert_close(prop[:, -1], torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(prop[:, -4:-1], torch.tensor([[0.0, 0.0, 0.5]]).expand(2, 3))
    # Replace all privileged values and the simulator object pose; estimated student inputs stay fixed.
    env.scene["object"].data.root_pos_w.torch += 5
    obs["privileged"].fill_(999)
    prop2, ext2, target2 = adapter(obs, estimated)
    torch.testing.assert_close(prop2, prop)
    torch.testing.assert_close(ext2, ext)
    assert not torch.equal(target2, target)
    adapter.record_action(torch.ones(2, 21), torch.tensor([True, False]))
    assert torch.count_nonzero(adapter.history[0]) == 0
    torch.testing.assert_close(adapter.history[1, 0], torch.ones(21))
    adapter.record_action(torch.ones(2, 21) * 2, torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(adapter.history[1, :2, 0], torch.tensor([2.0, 1.0]))


def test_student_noise_leaves_source_state_and_reconstruction_targets_clean():
    env = fake_env()
    obs = {"policy": torch.randn(2, 49), "privileged": torch.randn(2, 38)}
    original = {k: v.clone() for k, v in obs.items()}
    noisy = StudentObservations(env, StudentCfg(), PerceptionNoiseCfg())
    clean = StudentObservations(env, StudentCfg(), PerceptionNoiseCfg(enabled=False, joint_enabled=False))
    prop, ext, target = noisy(obs)
    clean_prop, clean_ext, clean_target = clean(obs)
    assert not torch.equal(prop[:, :21], clean_prop[:, :21])
    assert not torch.equal(ext, clean_ext)
    torch.testing.assert_close(prop[:, 21:], clean_prop[:, 21:])
    torch.testing.assert_close(target, clean_target)
    for name in original:
        torch.testing.assert_close(obs[name], original[name])


def test_teacher_shares_student_encoder_sample_but_keeps_true_pose_and_privileged_state():
    env = fake_env()
    obs = {"policy": torch.randn(2, 49), "privileged": torch.randn(2, 38)}
    original = {name: value.clone() for name, value in obs.items()}
    adapter = StudentObservations(env, StudentCfg(), PerceptionNoiseCfg())
    prop, _, target = adapter(obs)
    noisy_teacher = teacher_observations(obs, prop)
    audit = audit_teacher_observations(noisy_teacher, obs, prop)
    assert audit["teacher_joint_noise_rms"] > 0
    assert audit["teacher_student_joint_max_abs_difference"] == 0
    assert audit["teacher_privileged_max_abs_error"] == 0
    torch.testing.assert_close(target[:, 11:32], obs["privileged"][:, :21])
    for name in original:
        torch.testing.assert_close(obs[name], original[name])
    # Visual corruption must be rejected if it accidentally reaches the teacher.
    noisy_teacher["privileged"] = noisy_teacher["privileged"].clone()
    noisy_teacher["privileged"][:, 21:24] += 0.1
    with pytest.raises(AssertionError):
        audit_teacher_observations(noisy_teacher, obs, prop)


@pytest.mark.parametrize("metadata", [pose_frame_metadata(), {"pose_frame": "palm", "quaternion_order": "xyzw"}])
def test_export_roundtrip_different_batch_and_nonzero_state(tmp_path, metadata):
    module_path = Path(__file__).parents[1] / "scripts/rsl_rl/export_student.py"
    spec = importlib.util.spec_from_file_location("export_student", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "student.pt"
    DAggerTrainer(small_model(), DistillationCfg()).save(checkpoint, metadata, PerceptionNoiseCfg())
    module.export(checkpoint, tmp_path / "export")
    assert (tmp_path / "export/student_policy.pt").exists()
    assert (tmp_path / "export/student_policy.json").exists()
    import json
    info = json.loads((tmp_path / "export/student_policy.json").read_text())
    assert info["metadata"]["pose_frame"] == "right_hand_base_link"
    assert "right_hand_base_link" in info["extero_layout"][0]


def test_teacher_object_and_version_checks(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"metadata-only")
    (tmp_path / "params").mkdir()
    (tmp_path / "params/agent.yaml").write_text(
        "experiment_name: viserdex_revo3_sim6_ppo\nobs_groups:\n  actor: [policy, privileged]\nactor:\n  class_name: MLPModel\n  hidden_dims: !!python/tuple [512, 256, 128]\n"
    )
    (tmp_path / "params/env.yaml").write_text("usd_path: /assets/usd/RSLCube/RSLCube.usd\nrobot: revo3_right.usd\n")
    assert len(teacher_metadata(checkpoint, "RSLCube")["sha256"]) == 64
    with pytest.raises(ValueError, match="does not match"):
        teacher_metadata(checkpoint, "Duck")

    (tmp_path / "params/env.yaml").write_text(
        "usd_path: /assets/generated/usd/RSLCube/RSLCube.usda\nrobot: revo3_right.usda\n"
        "sim:\n  physics:\n    class_type: isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager\n"
    )
    newton = "isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager"
    assert teacher_metadata(checkpoint, "RSLCube", expected_physics=newton)["physics"] == newton
    with pytest.raises(ValueError, match="differs from rollout physics"):
        teacher_metadata(checkpoint, "RSLCube", expected_physics="isaacsim_physx")
