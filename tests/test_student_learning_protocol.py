"""Logical optimizer batches, recurrent refresh and quaternion-label invariance."""

from copy import deepcopy
from dataclasses import asdict

import torch

from viserdex.distillation.config import DistillationCfg
from viserdex.distillation.teacher import audit_teacher_observations, teacher_observations
from viserdex.distillation.trainer import DAggerTrainer
from test_distillation import make_batch, small_model


def test_optimizer_batches_visit_every_environment_and_update_normalization_once():
    torch.manual_seed(12)
    model = small_model()
    batch = make_batch(model)
    batch['prop'][0, :, 0] = torch.arange(3)
    observed = []
    original = model.sequence

    def record(prop, *args):
        observed.extend(prop[0, :, 0].tolist())
        return original(prop, *args)

    model.sequence = record
    trainer = DAggerTrainer(model, DistillationCfg(epochs=2, optimizer_env_batch_size=2, update_env_batch_size=1))
    metrics = trainer.update(batch)
    assert metrics['optimizer_steps'] == 4
    assert sorted(observed) == [0., 0., 1., 1., 2., 2.]
    assert model.prop_norm.count == 15
    assert {int(state['step']) for state in trainer.optimizer.state.values()} == {4}


def test_memory_chunks_preserve_each_logical_optimizer_step():
    torch.manual_seed(13)
    first = small_model()
    second = deepcopy(first)
    batch = make_batch(first)
    a = DAggerTrainer(first, DistillationCfg(epochs=2, optimizer_env_batch_size=2, update_env_batch_size=1))
    b = DAggerTrainer(second, DistillationCfg(epochs=2, optimizer_env_batch_size=2, update_env_batch_size=2))
    torch.manual_seed(22)
    a.update(batch)
    torch.manual_seed(22)
    b.update(batch)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=2e-6, rtol=2e-5)


def test_full_optimizer_batch_retains_legacy_update():
    torch.manual_seed(14)
    first = small_model()
    second = deepcopy(first)
    batch = make_batch(first)
    a = DAggerTrainer(first, DistillationCfg(epochs=2, update_env_batch_size=1))
    b = DAggerTrainer(second, DistillationCfg(epochs=2, update_env_batch_size=1, optimizer_env_batch_size=3))
    a.update(batch)
    b.update(batch)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)


def test_refreshed_hidden_uses_updated_model_and_respects_episode_boundaries():
    torch.manual_seed(15)
    model = small_model()
    batch = make_batch(model)
    batch['reset'][2, 1] = True
    batch['reset'][-1, 2] = True
    trainer = DAggerTrainer(model, DistillationCfg(epochs=2, update_env_batch_size=1))
    with torch.no_grad():
        old = model.sequence(batch['prop'], batch['extero'], batch['hidden'], batch['reset'])[2]
    trainer.update(batch)
    actual = trainer.replay_hidden(batch)
    with torch.no_grad():
        expected = batch['hidden']
        for step in range(batch['prop'].shape[0]):
            expected = expected * (~batch['reset'][step])[None, :, None]
            _, _, expected = model(batch['prop'][step], batch['extero'][step], expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert not torch.equal(actual, old)


def test_teacher_quaternion_labels_are_sign_invariant_without_changing_velocities():
    torch.manual_seed(16)
    clean = {'policy': torch.randn(5, 49), 'privileged': torch.randn(5, 38)}
    original = {key: value.clone() for key, value in clean.items()}
    opposite = {key: value.clone() for key, value in clean.items()}
    opposite['privileged'][:, 24:28] *= -1
    opposite['privileged'][:, 34:38] *= -1
    prop = torch.randn(5, 113)
    first = teacher_observations(clean, prop, 'positive_world_w')
    second = teacher_observations(opposite, prop, 'positive_world_w')
    torch.testing.assert_close(first['privileged'], second['privileged'], atol=0, rtol=0)
    for ids in [slice(0, 24), slice(28, 34)]:
        torch.testing.assert_close(first['privileged'][:, ids], clean['privileged'][:, ids], atol=0, rtol=0)
    for key, value in clean.items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    assert audit_teacher_observations(first, clean, prop, 'positive_world_w')['teacher_privileged_max_abs_error'] == 0


def test_legacy_configuration_does_not_silently_change_training_protocol():
    old = asdict(DistillationCfg())
    for key in ['optimizer_env_batch_size', 'refresh_hidden_after_update', 'teacher_quaternion_sign']:
        old.pop(key)
    cfg = DistillationCfg(**old)
    assert cfg.optimizer_env_batch_size is None
    assert not cfg.refresh_hidden_after_update
    assert cfg.teacher_quaternion_sign == 'raw'
