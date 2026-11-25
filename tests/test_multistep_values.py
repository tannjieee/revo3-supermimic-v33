"""Real TD returns: failure, timeout, ring wrap, bounds and training integration."""
from dataclasses import replace
import math
import pytest
import torch
from torch import nn
from test_absolute_policy import fixture
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.stable_values import real_td_batch, real_prefix, fit_real_critic
from viserdex.supermimic.trainer import imagination_protocol


def sequence(n=4, steps=8):
    tr, b = fixture(n, critic_td_steps=steps)
    keys = ('state', 'action', 'next_physical', 'reward', 'done', 'terminated')
    seq = {k: b[k][:, None].expand(-1, steps, *b[k].shape[1:]).clone() for k in keys}
    seq['env_id'] = b['env_id']
    return tr, b, seq


class FixedValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, state, context):
        self.seen.append(state.clone())
        return state.new_tensor([4., 20., 20., 20., 20., 20.]).expand(len(state), -1)


def test_real_multistep_failure_timeout_and_full_return_do_not_cross_reset():
    tr, b, seq = sequence()
    # failure at t=2; timeout at t=1; full window; immediate failure.
    seq['terminated'][0, 2] = seq['done'][0, 2] = True
    seq['done'][1, 1] = True
    seq['terminated'][3, 0] = seq['done'][3, 0] = True
    seq['next_physical'][1, 1, 42] += .02
    tr.target_critic = FixedValue()
    state, context, labels, _ = real_td_batch(tr, seq)
    gamma = tr.cfg.gamma
    lengths = [3, 2, 8, 1]
    for i, length in enumerate(lengths):
        r = .4 * sum(gamma**t for t in range(length))
        risk = gamma**(length-1)*100 if i in (0, 3) else gamma**length*20
        if i not in (0, 3): r += gamma**length*4
        assert labels[i, 0].item() == pytest.approx(r/tr.critic.value_bounds[0].item(), rel=1e-6)
        torch.testing.assert_close(labels[i, 1:], torch.full((5,), risk/100))
    torch.testing.assert_close(state, b['state'])
    torch.testing.assert_close(context, tr.context)
    torch.testing.assert_close(tr.target_critic.seen[0][1, :55], seq['next_physical'][1, 1])
    torch.testing.assert_close(tr.target_critic.seen[0][[0, 3]], b['state'][[0, 3]])
    valid, lengths = real_prefix(seq)
    assert lengths.tolist() == [3, 2, 8, 1]
    # Unrelated reset data must not affect the labels, including NaNs/Infs.
    for key in ('state', 'next_physical', 'action', 'reward'):
        seq[key][~valid] = float('nan')
    seq['terminated'][~valid] = True
    seq['done'][~valid] = True
    _, _, masked_labels, _ = real_td_batch(tr, seq)
    torch.testing.assert_close(masked_labels, labels, atol=0, rtol=0)


def test_boundary_on_first_step_matches_historical_one_step():
    tr, b, seq = sequence()
    b['terminated'][::2] = True
    b['done'][:] = True
    seq['terminated'][:, 0] = b['terminated']
    seq['done'][:, 0] = True
    tr.target_critic = FixedValue()
    multi = real_td_batch(tr, seq)[2]
    tr.cfg = replace(tr.cfg, critic_td_steps=1)
    one = real_td_batch(tr, b)[2]
    torch.testing.assert_close(multi, one, atol=0, rtol=0)


def test_multistep_targets_keep_bounds_and_fail_on_bad_window():
    tr, _, seq = sequence()
    from viserdex.supermimic.stability_reward import task_reward_bound
    seq['reward'].fill_(task_reward_bound(tr.cfg))
    seq['next_physical'][:, :, 21:42] = 1e6
    class MaxValue(nn.Module):
        def forward(self, state, context): return tr.critic.value_bounds.expand(len(state), -1)
    tr.target_critic = MaxValue()
    labels = real_td_batch(tr, seq)[2]
    assert torch.isfinite(labels).all() and (labels <= 1).all() and (labels >= 0).all()
    assert labels[:, 0].min() > .99999 and labels[:, 1].min() > .99999
    with pytest.raises(ValueError, match='window'): real_td_batch(tr, {**seq, 'state': seq['state'][:, 0]})
    with pytest.raises(ValueError, match='boundaries'): real_td_batch(tr, {k: v for k,v in seq.items() if k != 'done'})


def test_uniform_critic_ring_windows_bypass_wm_priority_and_preserve_boundaries(monkeypatch):
    tr, b, _ = sequence()
    replay = Replay(9, 4, 'cpu', event_window=3, adaptive=True)
    row = {k: b[k] for k in ('state', 'action', 'next_physical', 'reward', 'done', 'terminated')}
    for t in range(15):
        row['reward'] = torch.full((4,), float(t))
        row['done'] = torch.full((4,), t == 9)
        replay.add(row)
    def fail(*args): raise AssertionError('Critic must not use WM error priorities')
    monkeypatch.setattr(replay, '_sample_prioritized', fail)
    sampled = replay.sample(100, window=8, include_events=False, prioritized=False)
    torch.testing.assert_close(sampled['reward'][:, 1:]-sampled['reward'][:, :-1], torch.ones(100, 7))
    assert sampled['reward'].min() >= 6 and sampled['reward'].max() <= 14
    valid, length = real_prefix(sampled)
    assert (length == (10-sampled['reward'][:, 0]).long()).all()
    torch.testing.assert_close(sampled['mask'], valid.float())
    assert 'importance_weight' not in sampled


def test_warmup_and_policy_update_use_uniform_multistep_critic():
    tr, b, seq = sequence()
    from test_wm_task_supervision import window
    wm = window(tr, b['state'])
    calls = []
    class Buffer:
        archive_size = 0
        def sample(self, size, window=1, **kwargs):
            calls.append((window, kwargs))
            return b if window == 1 else (seq if window == 8 else wm)
        def update_priorities(self, *args): return 0
    for iteration in (0, 500):
        tr.iteration = iteration
        metrics = tr.update(Buffer())
        assert math.isfinite(metrics['critic_loss'])
        assert metrics['critic_td_steps'] == metrics['critic_real_mean_td_steps'] == 8
    critic_calls = [opts for k, opts in calls if k == 8]
    assert len(critic_calls) == 2
    assert all(opts == dict(include_events=False, prioritized=False) for opts in critic_calls)
    assert tr.optimizer_steps['actor'] == 1
    assert all(p.grad is None for p in tr.target_critic.parameters())


@pytest.mark.parametrize('steps', [0, -1, True, 1.5, 1001])
def test_invalid_td_steps_and_protocol_separation(steps):
    tr, _, _ = sequence()
    with pytest.raises(ValueError, match='TD steps'): replace(tr.cfg, critic_td_steps=steps)
    assert imagination_protocol(tr.cfg) != imagination_protocol(replace(tr.cfg, critic_td_steps=1))
    with pytest.raises(ValueError, match='v30'): replace(tr.cfg, absolute_policy_optimization=False)


@pytest.mark.parametrize('steps', [65, 128])
def test_td_window_cannot_exceed_initial_replay_collection(steps):
    tr, _, _ = sequence()
    assert tr.cfg.warmup_steps == 64
    with pytest.raises(ValueError, match='warmup.*Critic TD window'):
        replace(tr.cfg, critic_td_steps=steps)


@pytest.mark.parametrize('steps', [8, 64])
def test_default_and_boundary_td_windows_are_available_after_warmup(steps):
    tr, b, _ = sequence()
    cfg = replace(tr.cfg, critic_td_steps=steps)
    replay = Replay(cfg.buffer_steps, len(b['state']), 'cpu', event_capacity=0)
    row = {k: b[k] for k in ('state', 'action', 'next_physical', 'reward', 'done', 'terminated')}
    for _ in range(cfg.warmup_steps):
        replay.add(row)
    sampled = replay.sample(cfg.actor_batch, window=cfg.critic_td_steps,
                            include_events=False, prioritized=False)
    assert sampled['state'].shape == (cfg.actor_batch, steps, cfg.state_dim)
    assert sampled['mask'].all()
