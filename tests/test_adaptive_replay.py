import torch
import pytest
from viserdex.supermimic.replay import Replay


def row(t, done=False):
    state = torch.full((2, 84), float(t))
    return dict(state=state, next_physical=state[:, :55]+1,
                action=torch.full((2, 21), float(t)), done=torch.full((2,), done))


def test_real_windows_masks_and_stale_tokens():
    replay = Replay(6, 2, 'cpu', event_window=3, adaptive=True)
    for i in range(8): replay.add(row(i, i == 4))
    batch = replay.sample(64, 3)
    assert replay.archive_size == 0
    assert torch.equal(batch['action'][:, 1:, 0]-batch['action'][:, :-1, 0], torch.ones(64, 2))
    assert (batch['state'][:, :, 0] >= 2).all()
    expected = torch.cat((torch.ones(64, 1), (~batch['done'][:, :-1]).float().cumprod(1)), 1)
    torch.testing.assert_close(batch['mask'], expected)
    assert replay.update_priorities(batch, torch.arange(64).float()) > 0
    for i in range(8, 14): replay.add(row(i))
    before = replay.priority.clone()
    assert replay.update_priorities(batch, torch.ones(64)*1000) == 0
    torch.testing.assert_close(replay.priority, before)
    assert 'importance_weight' not in replay.sample(64, 1)
    with pytest.raises(ValueError): replay.sample(1, 2)


def test_importance_corrects_priority_bias_with_uniform_floor():
    torch.manual_seed(20)
    replay = Replay(5, 2, 'cpu', event_window=2, adaptive=True)
    for i in range(5): replay.add(row(i))
    replay.priority[0] = 1000.
    b = replay.sample(60000, 2)
    p = b['sample_probability']
    assert (p >= .2/8 - 1e-8).all()
    torch.testing.assert_close(b['importance_weight'], 1/(8*p))
    assert b['state'][:, 0, 0].mean() < .6  # intentionally biased draws
    corrected = (b['state'][:, 0, 0]*b['importance_weight']).mean()
    assert abs(float(corrected)-1.5) < .05


def test_wm_updates_sequence_priorities():
    from test_supermimic import config, make_replay
    from viserdex.supermimic.trainer import Trainer
    cfg = config()
    replay, state, limits, noise = make_replay(cfg)
    replay.adaptive = True
    replay.event_window = cfg.wm_window
    trainer = Trainer(cfg, noise, torch.ones(len(state), 2), limits,
                      torch.tensor([0., 0., 0., 1.]).expand(len(state), -1), 'cpu')
    before = replay.priority.clone()
    metrics = trainer.wm_update(replay)
    assert torch.isfinite(torch.tensor(metrics['wm_loss']))
    assert not torch.equal(before, replay.priority)


def test_priority_duplicate_max_clamp_and_stale_generation():
    replay = Replay(6, 2, 'cpu', event_window=3, adaptive=True)
    for i in range(6):
        replay.add(row(i))
    batch = dict(priority_slot=torch.tensor([1, 1, 2, 3, 4]),
                 env_id=torch.tensor([0, 0, 1, 0, 1]),
                 priority_generation=replay.generation[torch.tensor([1, 1, 2, 3, 4])].clone())
    batch['priority_generation'][-1] -= 1
    errors = torch.tensor([.25, 2., 2e6, 0., 7e6])
    assert replay.update_priorities(batch, errors) == 3
    assert replay.priority[1, 0] == torch.tensor(2.00001)
    assert replay.priority[2, 1] == 1e6
    assert replay.priority[3, 0] == torch.tensor(1e-5)
    assert replay.priority[4, 1] == 1
    assert replay.max_priority == 1e6
    for value in (float('nan'), float('inf'), -1.):
        with pytest.raises(ValueError, match='finite nonnegative'):
            replay.update_priorities(batch, torch.full((5,), value))
