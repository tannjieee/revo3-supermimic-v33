"""Actor-only runtime LR recovery schedules and checkpoint continuity."""
import copy
from dataclasses import asdict

import pytest
import torch

from test_supermimic_readiness import trainer_and_replay
from viserdex.supermimic.trainer import actor_ramp


METADATA = dict(environment_contract='test', joint_names=[], context_schema={}, observation_protocol='test')


def test_arming_schedule_changes_no_parameters_optimizers_or_frozen_config():
    trainer, replay, _ = trainer_and_replay()
    assert trainer.resume_actor_ramp is None
    trainer.update(replay)  # Populate Adam moments before testing the no-op arm.
    trainer.iteration = 1800
    before = {name: copy.deepcopy(getattr(trainer, name).state_dict())
              for name in ('wm', 'actor', 'critic', 'opt_wm', 'opt_actor', 'opt_critic')}
    config = asdict(trainer.cfg)
    counts = copy.deepcopy(trainer.optimizer_steps)
    trainer.arm_resume_actor_ramp(1000)
    assert trainer.resume_actor_ramp == dict(start_iteration=1800, duration=1000)
    assert trainer.iteration == 1800 and trainer.optimizer_steps == counts
    assert asdict(trainer.cfg) == config
    for name, state in before.items():
        torch.testing.assert_close(getattr(trainer, name).state_dict(), state, rtol=0, atol=0)


def test_first_actor_steps_ramp_gradually_without_changing_wm_or_critic_lr():
    trainer, replay, _ = trainer_and_replay()
    trainer.iteration = 1800
    trainer.arm_resume_actor_ramp(1000)
    for expected in (.001, .002, .003):
        result = trainer.update(replay)
        assert result['resume_actor_ramp'] == expected
        assert result['actor_ramp'] == expected
        assert result['actor_learning_rate'] == result['learning_rate'] * expected * trainer.cfg.actor_lr_scale
        assert trainer.opt_actor.param_groups[0]['lr'] == result['actor_learning_rate']
        assert trainer.opt_wm.param_groups[0]['lr'] == result['learning_rate']
        assert trainer.opt_critic.param_groups[0]['lr'] == result['learning_rate']


def test_resume_schedule_cannot_bypass_global_actor_warmup():
    trainer, replay, _ = trainer_and_replay()
    trainer.cfg.wm_only_iters = 500
    trainer.arm_resume_actor_ramp(10)
    result = trainer.update(replay)
    assert result['resume_actor_ramp'] == .1
    assert result['actor_ramp'] == 0 and result['actor_learning_rate'] == 0
    assert trainer.optimizer_steps['actor'] == 0 and trainer.optimizer_steps['critic'] == 0
    trainer.iteration = 500
    trainer.arm_resume_actor_ramp(10)
    result = trainer.update(replay)
    assert result['resume_actor_ramp'] == .1
    assert result['actor_ramp'] == actor_ramp(500, 500, trainer.cfg.actor_ramp_iters)


def test_save_restore_preserves_ramp_start_and_next_update_exactly(tmp_path):
    trainer, replay, _ = trainer_and_replay()
    trainer.iteration = 1800
    trainer.arm_resume_actor_ramp(1000)
    trainer.update(replay)
    trainer.save(tmp_path / 'model.pt', METADATA)
    payload = torch.load(tmp_path / 'model.pt', weights_only=False)
    assert payload['resume_actor_ramp'] == dict(start_iteration=1800, duration=1000)
    restored, _, _ = trainer_and_replay()
    restored.restore(payload, METADATA, restore_rng=False)
    assert restored.iteration == 1801
    assert restored.resume_actor_ramp == trainer.resume_actor_ramp
    # Restoring must not leave an alias into the caller's mutable payload.
    payload['resume_actor_ramp']['duration'] = 2
    assert restored.resume_actor_ramp['duration'] == 1000
    rng = torch.get_rng_state()
    expected = trainer.update(replay)
    torch.set_rng_state(rng)
    actual = restored.update(replay)
    assert actual == expected
    assert actual['resume_actor_ramp'] == .002
    for name in ('wm', 'actor', 'critic', 'opt_wm', 'opt_actor', 'opt_critic'):
        torch.testing.assert_close(getattr(restored, name).state_dict(), getattr(trainer, name).state_dict(), rtol=0, atol=0)
    restored.arm_resume_actor_ramp(1000)
    assert restored.resume_actor_ramp['start_iteration'] == 1802


def test_legacy_checkpoint_without_schedule_preserves_original_behavior(tmp_path):
    trainer, replay, _ = trainer_and_replay()
    trainer.iteration = 1800
    trainer.save(tmp_path / 'legacy.pt', METADATA)
    payload = torch.load(tmp_path / 'legacy.pt', weights_only=False)
    payload.pop('resume_actor_ramp')
    restored, _, _ = trainer_and_replay()
    restored.arm_resume_actor_ramp(1000)
    restored.restore(payload, METADATA, restore_rng=False)
    assert restored.resume_actor_ramp is None
    result = restored.update(replay)
    assert result['resume_actor_ramp'] == 1 and result['actor_ramp'] == 1


def test_completed_schedule_caps_at_full_actor_learning_rate():
    trainer, replay, _ = trainer_and_replay()
    trainer.iteration = 1800
    trainer.arm_resume_actor_ramp(1)
    for _ in range(2):
        result = trainer.update(replay)
        assert result['resume_actor_ramp'] == 1
        assert result['actor_learning_rate'] == result['learning_rate'] * trainer.cfg.actor_lr_scale


@pytest.mark.parametrize('duration', [0, -1, True, 1.5, float('inf'), '1000', None])
def test_arm_rejects_nonpositive_or_noninteger_duration(duration):
    trainer, _, _ = trainer_and_replay()
    with pytest.raises(ValueError, match='Resume actor ramp'):
        trainer.arm_resume_actor_ramp(duration)
    assert trainer.resume_actor_ramp is None


@pytest.mark.parametrize('schedule', [
    {}, [], True, {'start_iteration': 0}, {'start_iteration': 0, 'duration': 1, 'extra': 2},
    {'start_iteration': -1, 'duration': 1000}, {'start_iteration': 1801, 'duration': 1000},
    {'start_iteration': True, 'duration': 1000}, {'start_iteration': 0., 'duration': 1000},
    {'start_iteration': 1800, 'duration': 0}, {'start_iteration': 1800, 'duration': -1},
    {'start_iteration': 1800, 'duration': True}, {'start_iteration': 1800, 'duration': 1000.},
])
def test_restore_rejects_invalid_runtime_schedule_before_model_mutation(tmp_path, schedule):
    trainer, _, _ = trainer_and_replay()
    trainer.iteration = 1800
    trainer.save(tmp_path / 'invalid.pt', METADATA)
    payload = torch.load(tmp_path / 'invalid.pt', weights_only=False)
    payload['resume_actor_ramp'] = schedule
    restored, _, _ = trainer_and_replay()
    before = copy.deepcopy(restored.actor.state_dict())
    with pytest.raises(ValueError, match='Resume actor ramp'):
        restored.restore(payload, METADATA, restore_rng=False)
    assert restored.iteration == 0 and restored.resume_actor_ramp is None
    torch.testing.assert_close(restored.actor.state_dict(), before, rtol=0, atol=0)
