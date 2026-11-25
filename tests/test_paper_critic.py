"""SuperMimic Eq. (3): imagined fixed lambda targets, online V, raw MSE."""
from dataclasses import asdict, replace

import pytest
import torch

from test_absolute_policy import fixture as absolute_fixture
from viserdex.supermimic.core import Config
from viserdex.supermimic.trainer import checkpoint_version, frozen, lambda_returns


def fixture():
    return absolute_fixture(paper_critic=True)


def test_lambda_recursion_matches_hand_computed_returns_and_terminal_boundary():
    rewards = torch.tensor([[1.], [2.], [3.]])
    continuation = torch.tensor([[1.], [.5], [0.]])
    values = torch.tensor([[4.], [5.], [6.], [7.]])
    # R2=3; R1=2+.9*.5*(.3*6+.7*3)=3.755;
    # R0=1+.9*(.3*5+.7*3.755)=4.71565.
    expected = torch.tensor([[4.71565], [3.755], [3.]])
    torch.testing.assert_close(lambda_returns(rewards, continuation, values, .9, .7), expected)
    continuation[-1] = 1.
    out = lambda_returns(rewards, continuation, values, .9, .7)
    torch.testing.assert_close(out[-1], rewards[-1]+.9*values[-1])


def test_online_linear_values_and_lambda_actually_change_imagined_targets():
    tr, batch = fixture()
    assert not hasattr(tr, 'target_critic')
    assert tr.sync_models == (tr.wm, tr.actor, tr.critic)
    with torch.no_grad():
        tr.critic.net[-1].bias.fill_(-2.)
    torch.testing.assert_close(tr.critic(batch['state'], tr.context), torch.full((4, 6), -2.))
    with frozen(tr.wm, tr.critic):
        tr.cfg = replace(tr.cfg, lam=0.)
        torch.manual_seed(31)
        loss, states, td0, _, _ = tr.imagine(batch)
        tr.cfg = replace(tr.cfg, lam=1.)
        torch.manual_seed(31)
        _, _, td1, _, _ = tr.imagine(batch)
        assert not td0.requires_grad and not td1.requires_grad and not states.requires_grad
        assert not torch.allclose(td0, td1)
        loss.backward()
    assert all(p.grad is None for p in tr.wm.parameters())
    assert all(p.grad is None for p in tr.critic.parameters())


def test_mse_uses_return_units_and_time_sum_without_target_rescaling():
    tr, batch = fixture()
    n, h = len(batch['state']), tr.cfg.horizon
    states = batch['state'].repeat(h, 1)
    context = tr.context.repeat(h, 1)
    labels = torch.arange(n*h*6, dtype=torch.float32).reshape(n*h, 6) / 3
    before = labels.clone()
    for group in tr.opt_critic.param_groups:
        group['lr'] = 0.
    result = tr.fit_critic((states, labels, context), rollout_count=n)
    assert result['critic_loss'] == pytest.approx(float(labels.square().sum()/n), rel=1e-6)
    torch.testing.assert_close(labels, before, atol=0, rtol=0)
    # An omitted known-terminal suffix contributes zero, not a renormalized mean.
    tr.opt_critic.zero_grad(set_to_none=True)
    result = tr.fit_critic((states[:n], labels[:n], context[:n]), rollout_count=n)
    assert result['critic_loss'] == pytest.approx(float(labels[:n].square().sum()/n), rel=1e-6)


def test_joint_update_uses_fixed_imagined_targets_and_never_real_td(monkeypatch):
    from viserdex.supermimic import stable_values
    tr, batch = fixture()
    torch.nn.init.normal_(tr.wm.net[-1].weight, std=.01)
    tr.cfg = replace(tr.cfg, critic_epochs=2)
    tr.iteration = tr.cfg.wm_only_iters
    # Real labels deliberately cannot serve as the critic targets.
    batch['reward'].fill_(float('nan'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Real-TD/EMA path was entered')
    monkeypatch.setattr(stable_values, 'fit_real_critic', forbidden)
    monkeypatch.setattr(stable_values, 'update_target', forbidden)
    calls = []
    original = tr.fit_critic
    def fit(imagined, **kwargs):
        states, labels, context = imagined
        assert not labels.requires_grad and torch.isfinite(labels).all()
        saved = labels.clone()
        result = original(imagined, **kwargs)
        torch.testing.assert_close(labels, saved, atol=0, rtol=0)
        calls.append(len(states))
        return result
    monkeypatch.setattr(tr, 'fit_critic', fit)
    class Buffer:
        def sample(self, count, **kwargs):
            assert 'window' not in kwargs
            return batch
    result = tr.policy_update(Buffer())
    assert len(calls) == 1 and calls[0] > len(batch['state'])
    assert tr.optimizer_steps == {'wm': 0, 'actor': 1, 'critic': 2}
    assert 'critic_target_tau' not in result and 'critic_real_normalized_mse' not in result
    assert tr._actor_objective_terms is None and tr._actor_reward_components is None


def test_warmup_does_not_train_critic_on_real_data(monkeypatch):
    tr, _ = fixture()
    monkeypatch.setattr(tr, 'wm_update', lambda *args, **kwargs: {})
    class Buffer:
        archive_size = 0
        def sample(self, *args, **kwargs):
            raise AssertionError('WM-only warmup must not sample real Critic targets')
    tr.update(Buffer())
    assert tr.optimizer_steps['actor'] == tr.optimizer_steps['critic'] == 0


def test_v32_roundtrip_and_old_checkpoint_protocol_remains_separate(tmp_path):
    tr, _ = fixture()
    metadata = dict(environment_contract='test', joint_names=[], context_schema={}, observation_protocol='test')
    tr.save(tmp_path/'paper.pt', metadata)
    payload = torch.load(tmp_path/'paper.pt', weights_only=False)
    assert checkpoint_version(tr.cfg) == payload['version'] == 32
    assert 'target_critic' not in payload
    other, _ = fixture()
    other.restore(payload, metadata, restore_rng=False)
    for name in ('wm', 'actor', 'critic'):
        for key, value in getattr(tr, name).state_dict().items():
            torch.testing.assert_close(value, getattr(other, name).state_dict()[key])
    legacy, _ = absolute_fixture()
    with pytest.raises(ValueError, match='Version'):
        legacy.restore(payload, metadata, restore_rng=False)
    old_config = asdict(legacy.cfg)
    old_config.pop('paper_critic')
    assert Config(**old_config).uses_target_critic
    legacy.save(tmp_path/'legacy.pt', metadata)
    old = torch.load(tmp_path/'legacy.pt', weights_only=False)
    old['config'].pop('paper_critic')
    restored, _ = absolute_fixture()
    restored.restore(old, metadata, restore_rng=False)
    assert hasattr(restored, 'target_critic')
    with pytest.raises(ValueError, match='not a real TD window'):
        replace(tr.cfg, critic_td_steps=8)


def test_v32_actor_export_preserves_actions(tmp_path):
    from viserdex.supermimic.inference import export_actor, ManualConvexPolicy
    from viserdex.supermimic.revised_contract import joint_names
    tr, batch = fixture()
    metadata = dict(state_protocol=tr.cfg.state_protocol,
                    observation_protocol=tr.cfg.observation_protocol,
                    observation_dim=tr.cfg.observation_dim, quaternion_order='XYZW',
                    joint_names=list(joint_names(tr.cfg)),
                    joint_limits_rad=list(tr.cfg.network_joint_limits))
    tr.save(tmp_path/'paper.pt', metadata)
    export_actor(tmp_path/'paper.pt', tmp_path/'actor.pt')
    policy = ManualConvexPolicy(tmp_path/'actor.pt')
    history = torch.randn_like(batch['history'])
    torch.testing.assert_close(policy.actor(history), tr.actor(history), atol=0, rtol=0)
