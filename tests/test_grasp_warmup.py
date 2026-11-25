"""CPU contracts for temporary grasp warmup and prediction-readiness gates."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from viserdex.supermimic import grasp_runtime
from viserdex.supermimic.warmup import WarmupGraspResetManager, readiness_decision


def make_warmup(monkeypatch, *, split='train', seed=12, **kwargs):
    records = [dict(id=f'grasp-{i}', split=split, q_geometry=[i * .1] * 21,
                    hold_target_encoder=[-.4 + .4 * i] * 21,
                    object_pose_xyzw=[.05, 0., .12, 0., 0., 0., 1.]) for i in range(3)]
    bank = SimpleNamespace(records=records, split=split, path='/test/bank.json', sha256='bound-hash')
    env = SimpleNamespace(num_envs=4, device='cpu')
    adapter = SimpleNamespace(env=env, limits=torch.tensor([-1., 1.]).expand(4, 21, 2).clone())
    assignments = []
    monkeypatch.setattr(grasp_runtime, 'write_grasp_states',
                        lambda *args, **kw: assignments.append((args, kw)))
    manager = WarmupGraspResetManager(adapter, bank, seed=seed, **kwargs)
    return manager, assignments


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_warmup_never_accepts_held_out_bank_split(monkeypatch, split):
    with pytest.raises(ValueError, match='training split'):
        make_warmup(monkeypatch, split=split)


def test_reset_preserves_ordinary_goal_and_does_not_preload_actuator_memory(monkeypatch):
    manager, writes = make_warmup(monkeypatch)
    manager.reset(None)
    assert len(writes) == 1
    args, kwargs = writes[0]
    assert args[0] is manager.adapter
    torch.testing.assert_close(args[1], torch.arange(4))
    assert kwargs == {'preserve_goal': True}  # No goals, velocity or delay-memory injection.
    expected = torch.tensor([record['hold_target_encoder'] for record in args[2]])
    torch.testing.assert_close(manager.hold, expected)
    assert manager.transition_ids() == {}  # Replay schema remains valid after detaching.
    metadata = manager.metadata()
    assert metadata['split'] == 'train' and metadata['bank_sha256'] == 'bound-hash'
    assert metadata['actor_updated'] is False
    assert sum(metadata['reset_counts']) == 4


def test_hold_is_exact_for_40_steps_then_offsets_are_bounded_and_piecewise_constant(monkeypatch):
    manager, _ = make_warmup(monkeypatch, exploration=.03)
    manager.reset(None)
    hold = manager.hold.clone()
    for step in range(40):
        assert manager.age.tolist() == [step] * 4
        torch.testing.assert_close(manager.action(), hold, rtol=0, atol=0)
        manager.post_step()
    first = manager.action()
    assert not torch.equal(first, hold)
    assert (first - hold).abs().max() <= .0300001
    for _ in range(3):
        manager.post_step()
        torch.testing.assert_close(manager.action(), first, rtol=0, atol=0)
    manager.post_step()
    second = manager.action()
    assert not torch.equal(first, second)
    assert (second - hold).abs().max() <= .0300001
    manager.hold[:, 0] = 1.
    manager.hold[:, 1] = -1.
    for _ in range(50):
        action = manager.action()
        assert action.abs().max() <= 1
        assert (action - manager.hold).abs().max() <= .0300001
        manager.post_step()


def test_partial_reset_restarts_only_selected_world_settling(monkeypatch):
    manager, writes = make_warmup(monkeypatch)
    manager.reset(None)
    for _ in range(41):
        manager.action()
        manager.post_step()
    old_action, old_counts = manager.action().clone(), manager.counts.copy()
    manager.reset(torch.tensor([1, 3]))
    assert manager.age.tolist() == [41, 0, 41, 0]
    assert manager.counts.sum() == old_counts.sum() + 2
    action = manager.action()
    torch.testing.assert_close(action[[0, 2]], old_action[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(action[[1, 3]], manager.hold[[1, 3]], rtol=0, atol=0)
    assert writes[-1][1] == {'preserve_goal': True}


def test_empty_reset_is_a_noop(monkeypatch):
    manager, writes = make_warmup(monkeypatch)
    before = manager.metadata()
    manager.reset(torch.empty(0, dtype=torch.long))
    assert not writes and manager.metadata() == before


def test_warmup_owns_only_its_temporary_attachment(monkeypatch):
    manager, _ = make_warmup(monkeypatch)
    with pytest.raises(ValueError, match='another grasp reset manager'):
        WarmupGraspResetManager(manager.adapter, manager.bank)
    replacement = object()
    manager.adapter.env.grasp_reset_manager = replacement
    with pytest.raises(ValueError, match='ownership changed'):
        manager.detach()
    assert manager.adapter.env.grasp_reset_manager is replacement
    manager.adapter.env.grasp_reset_manager = manager
    manager.detach()
    assert not hasattr(manager.adapter.env, 'grasp_reset_manager')
    with pytest.raises(ValueError, match='ownership changed'):
        manager.detach()


def test_warmup_sampling_is_reproducible_without_advancing_training_rng(monkeypatch):
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    first, _ = make_warmup(monkeypatch, seed=9, settle_steps=0)
    second, _ = make_warmup(monkeypatch, seed=9, settle_steps=0)
    first.reset(None)
    second.reset(None)
    np.testing.assert_array_equal(first.ids, second.ids)
    for _ in range(8):
        torch.testing.assert_close(first.action(), second.action(), rtol=0, atol=0)
        first.post_step()
        second.post_step()
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    now = np.random.get_state()
    assert now[0] == np_state[0] and now[2:] == np_state[2:]
    np.testing.assert_array_equal(now[1], np_state[1])


@pytest.mark.parametrize('kwargs', [dict(exploration=float('nan')), dict(exploration=.21),
                                    dict(settle_steps=40.5), dict(settle_steps=True),
                                    dict(refresh_steps=0), dict(refresh_steps=float('nan'))])
def test_invalid_behavior_settings_fail_before_attachment(monkeypatch, kwargs):
    with pytest.raises(ValueError, match='behavior settings'):
        make_warmup(monkeypatch, **kwargs)


def readiness_result(position=.02, orientation=.35, valid=.9):
    return {'metrics': {'8': {'object_position_vector_rmse_m': position,
                             'object_orientation_geodesic_rmse_rad': orientation,
                             'valid_fraction': valid}}}


def test_readiness_thresholds_accept_boundary_and_reject_exceedances():
    decision = readiness_decision(readiness_result())
    assert decision['passed'] is True
    assert decision['thresholds'] == dict(position_rmse_m=.02, orientation_rmse_rad=.35,
                                         minimum_valid_fraction=.9)
    assert readiness_decision(readiness_result(position=.020001))['passed'] is False
    assert readiness_decision(readiness_result(orientation=.350001))['passed'] is False
    assert readiness_decision(readiness_result(valid=.899999))['passed'] is False


@pytest.mark.parametrize('field', ['position', 'orientation', 'valid'])
@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf'), -1.])
def test_readiness_rejects_missing_nonfinite_or_negative_metrics(field, value):
    assert readiness_decision(readiness_result(**{field: value}))['passed'] is False


def test_readiness_rejects_impossible_valid_fraction_and_empty_coverage():
    assert readiness_decision(readiness_result(valid=1.01))['passed'] is False
    result = readiness_result()
    del result['metrics']['8']['valid_fraction']
    result['metrics']['8']['valid_environments'] = 0
    result['environments'] = 0
    assert readiness_decision(result)['passed'] is False


def test_readiness_uses_real_sample_count_when_fraction_is_absent():
    result = readiness_result()
    del result['metrics']['8']['valid_fraction']
    result['metrics']['8']['valid_environments'] = 126
    result['environment_ids'] = list(range(128))
    decision = readiness_decision(result)
    assert decision['passed'] is True
    assert decision['observed']['valid_fraction'] == 126 / 128
    result['metrics']['8']['valid_environments'] = 114
    assert readiness_decision(result)['passed'] is False
    # An explicitly invalid reported fraction must not be replaced by a nicer fallback.
    result['metrics']['8']['valid_fraction'] = None
    result['metrics']['8']['valid_environments'] = 128
    assert readiness_decision(result)['passed'] is False


@pytest.mark.parametrize('kwargs', [dict(position_rmse_m=float('nan')), dict(position_rmse_m=0),
                                    dict(orientation_rmse_rad=-1), dict(minimum_valid_fraction=1.1),
                                    dict(minimum_valid_fraction=0)])
def test_invalid_readiness_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError, match='thresholds'):
        readiness_decision(readiness_result(), **kwargs)
