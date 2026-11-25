from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from viserdex.supermimic import grasp_runtime
from viserdex.supermimic.grasp_runtime import GraspResetManager


def make_manager(monkeypatch, evaluate=False, split='train', sha='same'):
    rows = [dict(id=str(i), source_group=str(i), q_geometry=[i*.1]*21,
                 object_pose_xyzw=[.05, 0., .12, 0., 0., np.sin(i*.4), np.cos(i*.4)]) for i in range(3)]
    bank = SimpleNamespace(records=rows, sha256=sha, split=split)
    adapter = SimpleNamespace(env=SimpleNamespace(num_envs=4, device='cpu'))
    assignments = []
    monkeypatch.setattr(grasp_runtime, 'write_grasp_states', lambda *args, **kwargs: assignments.append(args))
    return GraspResetManager(adapter, bank, seed=12, adaptive=True, evaluate=evaluate), assignments


def test_success_updates_statistics_only_at_episode_end(monkeypatch):
    manager, assignments = make_manager(monkeypatch)
    manager.reset(None)
    labels = manager.transition_ids()
    old = copy.deepcopy(manager.state_dict())
    manager.post_step(torch.zeros(4, dtype=torch.bool), torch.tensor([True, False, False, False]))
    assert manager.state_dict() == old
    assert len(assignments) == 1  # no successful-goal replacement
    manager.post_step(torch.tensor([True, False, False, False]), torch.zeros(4, dtype=torch.bool))
    assert manager.sampler.count.sum() == 1
    assert manager.sampler.success[labels['grasp_bucket'][0]] > .5
    manager.reset(torch.tensor([0]))
    assert len(assignments) == 2
    assert labels['grasp_start_id'].data_ptr() != manager.transition_ids()['grasp_start_id'].data_ptr()


def test_bank_hash_and_split_restore_semantics(monkeypatch):
    original, _ = make_manager(monkeypatch)
    original.reset(None)
    original.post_step(torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
    state = original.state_dict()
    restored, _ = make_manager(monkeypatch)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    with pytest.raises(ValueError):
        make_manager(monkeypatch, sha='changed')[0].load_state_dict(state)
    with pytest.raises(ValueError):
        make_manager(monkeypatch, split='test')[0].load_state_dict(state)
    evaluation, _ = make_manager(monkeypatch, evaluate=True, split='validation')
    initial = evaluation.state_dict()
    evaluation.load_state_dict(state)
    assert evaluation.state_dict() == initial


def test_trainer_checkpoint_binds_bank_and_sampler(monkeypatch, tmp_path):
    from test_supermimic import config, make_replay
    from viserdex.supermimic.trainer import Trainer
    cfg = config()
    _, state, limits, noise = make_replay(cfg)
    def trainer():
        return Trainer(cfg, noise, torch.ones(len(state), 2), limits,
                       torch.tensor([0., 0., 0., 1.]).expand(len(state), -1), 'cpu')
    first = trainer()
    first.grasp_manager, _ = make_manager(monkeypatch)
    first.grasp_manager.reset(None)
    first.grasp_manager.post_step(torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool))
    metadata = dict(environment_contract='test', joint_names=[], context_schema={}, observation_protocol='test')
    first.save(tmp_path/'state.pt', metadata)
    saved = torch.load(tmp_path/'state.pt', weights_only=False)
    second = trainer()
    with pytest.raises(ValueError, match='same grasp bank mode'): second.restore(saved, metadata)
    second.grasp_manager, _ = make_manager(monkeypatch)
    second.restore(saved, metadata)
    assert second.grasp_manager.state_dict() == first.grasp_manager.state_dict()
