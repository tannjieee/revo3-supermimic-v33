"""Regression tests for time conversion, true boundaries, and bounded-action imagined learning."""
import copy
import torch
import pytest
from test_supermimic import config, fixture, make_replay
from viserdex.supermimic.core import *
from viserdex.supermimic.trainer import Trainer, imagined_continuation
import viserdex.supermimic.trainer as trainer_module


def trainer_fixture():
    cfg = config(horizon=8)
    cfg.control_dt = .05
    replay, state, limits, noise = make_replay(cfg)
    trainer = Trainer(cfg, noise, torch.ones(6,2), limits,
                      torch.tensor([[0.,0.,0.,1.]]).expand(6,-1), 'cpu',400)
    return trainer, replay, state, limits


def test_smooth_20hz_preserves_event_reward_and_discount_time():
    c=config20()
    assert c.horizon==8 and c.wm_window==8
    assert abs(c.control_dt*c.success_bonus-250/30)<1e-12
    assert c.ema_alpha == .5
    assert abs(c.gamma**20-.99**30)<1e-12
    assert abs(c.lam**20-.95**30)<1e-12
    # Hold-Penalty2x-v1 doubles the Smooth/Hold converted first-difference penalty.
    assert abs(c.action_rate_weight*c.control_dt**2-2.5*.01*(1/30)**2)<1e-12
    s,limits=fixture(1)
    nxt=advance(s,s[:,:55],torch.zeros(1,21),s[:,55:76],s[:,GOAL_ROT],400)
    torch.testing.assert_close(nxt[:,TIME],s[:,TIME]-1/400)


def test_eight_step_closed_loop_retains_terminal_value_and_full_bptt(monkeypatch):
    """A short horizon still propagates the terminal value through WM and history."""
    t, replay, _, _ = trainer_fixture()
    c = config20()
    t.cfg.horizon, t.cfg.gamma, t.cfg.ema_alpha = c.horizon, c.gamma, c.ema_alpha
    predictions = []

    class RecurrentActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(.02))
        def forward(self, history):
            return .1 * history[:, -1, :21] + self.bias

    class Dynamics(torch.nn.Module):
        def forward(self, state, target, context):
            physical = state[:, :55].clone()
            physical[:, Q] = .5 * state[:, Q] + .1 * target
            predictions.append(physical)
            return physical

    class Value(torch.nn.Module):
        def forward(self, state, context):
            return state[:, Q].sum(-1)

    t.actor, t.wm, t.critic = RecurrentActor(), Dynamics(), Value()
    monkeypatch.setattr(trainer_module, 'reward', lambda state, *args, **kwargs: state[:, 0] * 0)
    monkeypatch.setattr(trainer_module, 'observe', lambda state, sensors, *args:
                        (torch.cat((state[:, Q], state.new_zeros(len(state), 103)), -1), sensors))
    batch = replay.sample(6)
    batch['state'][:, Q] = .03
    batch['state'][:, EMA] = .01
    batch['state'][:, TIME] = 1
    batch['state'][:, COUNT] = 0
    batch['history'].zero_()
    batch['history'][:, -1, :21] = .03
    loss, states, targets, _, valid = t.imagine(batch)
    assert len(predictions) == 8 and states.shape[0] == 8 and valid.all()
    # Independent scalar recurrence with all 21 identical joints and no truncation.
    bias = t.actor.bias.detach().clone().requires_grad_()
    q, ema = torch.tensor(.03), torch.tensor(.01)
    for _ in range(8):
        action = .1 * q + bias
        ema = c.ema_alpha * action + (1 - c.ema_alpha) * ema
        q = .5 * q + .1 * ema
    expected = -(c.gamma ** 8) * 21 * q
    torch.testing.assert_close(loss, expected)
    actual_gradient, first_prediction_gradient = torch.autograd.grad(loss, (t.actor.bias, predictions[0]))
    expected_gradient, = torch.autograd.grad(expected, bias)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert first_prediction_gradient[:, Q].abs().sum() > 0
    assert not targets.requires_grad


def test_bounded_action_motor_gradient_and_latent_exploration():
    from viserdex.supermimic.networks import Actor
    actor = Actor(config())
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.fill_(3.)
    s, limits = fixture(6)
    history = torch.zeros(6, 10, 124)
    bounded = actor(history)
    target = ema_target(s, bounded, limits)
    gradient, = torch.autograd.grad(target.sum(), actor.net[-1].bias)
    assert (gradient > .05).all()  # raw=3 -> old clamp had exactly zero motor derivative
    raw = torch.full((6, 21), 3., requires_grad=True)
    old_gradient, = torch.autograd.grad(ema_target(s, raw, limits).sum(), raw)
    assert not old_gradient.any()
    torch.manual_seed(19)
    expected = (actor.logits(history)+.1*torch.randn_like(raw)).tanh()
    torch.manual_seed(19)
    torch.testing.assert_close(actor.sample(history,.1), expected, rtol=0, atol=0)
    assert (actor.sample(history, 100.).abs() <= 1).all()
    # Store the same bounded command in recurrent task state; no hidden raw action.
    nxt = advance(s,s[:,:55],bounded,target,s[:,GOAL_ROT],400)
    torch.testing.assert_close(nxt[:,76:97],bounded)


def test_policy_uses_only_imagined_targets_and_frozen_models(monkeypatch):
    t, r, _, _ = trainer_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Critic must not sample real TD transitions")
    monkeypatch.setattr(r, 'sample_real', forbidden)
    before = copy.deepcopy(t.wm.state_dict())
    captured = []
    original = t.fit_critic
    def checked(data):
        targets = data[1].clone()
        assert not data[1].requires_grad
        result = original(data)
        torch.testing.assert_close(data[1], targets, rtol=0, atol=0)
        captured.append(targets)
        return result
    monkeypatch.setattr(t, 'fit_critic', checked)
    t.policy_update(r)
    assert captured and not hasattr(t, 'target_critic')
    torch.testing.assert_close(t.wm.state_dict(), before, rtol=0, atol=0)
    assert all(p.grad is None for p in t.wm.parameters())


def test_imagined_drop_keeps_sequence_but_timeout_and_count_limit_stop_it():
    t,r,_,_=trainer_fixture()
    class Drop(torch.nn.Module):
        def forward(self,s,target,context):
            p=s[:,:55].clone()
            p[:,42]=.4
            return p
    batch=r.sample(6)
    t.wm=Drop()
    _,_,_,_,valid=t.imagine(batch)
    assert valid.all()
    batch["state"][:, COUNT] = 50
    _,_,_,_,valid=t.imagine(batch)
    assert valid[0].all() and not valid[1:].any()
    batch["state"][:, COUNT] = 0
    class Identity(torch.nn.Module):
        def forward(self,s,target,context): return s[:,:55]
    t.wm=Identity()
    batch['state'][:,TIME]=1/400
    _,_,_,_,valid=t.imagine(batch)
    assert valid[0].all() and not valid[1:].any()


def test_warmup_updates_real_normalizers_but_no_actor_or_critic_weights():
    t,r,s,limits=trainer_fixture()
    t.cfg.wm_only_iters=1
    actor=[p.clone() for p in t.actor.parameters()]
    critic=[p.clone() for p in t.critic.parameters()]
    t.update_real_statistics(s,s[:,55:76],s[:,:55],torch.randn(6,124))
    assert t.actor.obs_norm.count == 6
    metrics = t.update(r)
    for a,b in zip(actor,t.actor.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(critic,t.critic.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert t.optimizer_steps==dict(wm=1,actor=0,critic=0)
    assert metrics['actor_learning_rate'] == 0
    metrics = t.update(r)
    assert t.cfg.actor_ramp_iters == 1000
    assert metrics['actor_ramp'] == .001
    assert metrics['actor_learning_rate'] == metrics['learning_rate'] * .001
    assert t.optimizer_steps==dict(wm=2,actor=1,critic=6)


@pytest.mark.parametrize('iteration', [499, 500, 999, 1499])
def test_production_actor_ramp_survives_checkpoint_resume(tmp_path, iteration):
    t, replay, _, _ = trainer_fixture()
    t.cfg.wm_only_iters = config20().wm_only_iters
    t.iteration = iteration
    metadata = dict(environment_contract='test', joint_names=[], context_schema={}, observation_protocol='test')
    path = tmp_path / 'ramp.pt'
    t.save(path, metadata)
    restored, _, _, _ = trainer_fixture()
    restored.cfg.wm_only_iters = t.cfg.wm_only_iters
    restored.restore(torch.load(path, weights_only=False), metadata, restore_rng=False)
    metrics = restored.update(replay)
    expected = max(0., min((iteration - 499) / 1000, 1.))
    assert metrics['actor_ramp'] == expected
    assert metrics['actor_learning_rate'] == metrics['learning_rate'] * expected
    assert restored.opt_actor.param_groups[0]['lr'] == metrics['actor_learning_rate']
    assert restored.optimizer_steps['actor'] == int(iteration >= 500)


def test_real_replay_keeps_pre_reset_auxiliary_state():
    t,r,_,_=trainer_fixture()
    r.data['next_aux'].fill_(.314)
    batch=r.sample_real(17)
    torch.testing.assert_close(batch['next_state'][:,55:],torch.full((17,118),.314))


def test_soft_drop_continuation_is_smooth_with_gradient_on_both_sides():
    state, _ = fixture(3)
    position = torch.tensor([.2999, .3, .3001], requires_grad=True)
    physical = state[:, :55].clone()
    physical[:, POS] = torch.stack((position, position * 0, position * 0), -1)
    continuation = imagined_continuation(state, physical, .01)
    torch.testing.assert_close(continuation, torch.sigmoid((.3-position)/.01))
    gradient, = torch.autograd.grad(continuation.sum(), position)
    assert (gradient < -24.9).all()
    assert (continuation > .49).all() and (continuation < .51).all()
    # Goal-count termination has priority, including when it coincides with timeout.
    state[:, COUNT] = 50
    assert not imagined_continuation(state, physical, .01).any()


@pytest.mark.parametrize("lam", [0., .7, 1.])
@pytest.mark.parametrize("timeout_step", [None, 1, 2])
def test_actor_and_critic_match_reference_soft_returns(monkeypatch, lam, timeout_step):
    t, replay, _, _ = trainer_fixture()
    t.cfg.horizon, t.cfg.gamma, t.cfg.lam = 3, .9, lam
    class PredictedPositions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.positions = torch.nn.Parameter(torch.tensor([.29, .31, .295]))
            self.step = 0
        def forward(self, state, target, context):
            physical = state[:, :55].clone()
            physical[:, POS] = 0
            physical[:, 42] = self.positions[self.step]
            self.step += 1
            return physical
    class ConstantValue(torch.nn.Module):
        def forward(self, state, context):
            return state[:, 0] * 0 + 3.5
    t.wm = PredictedPositions()
    t.critic = ConstantValue()
    monkeypatch.setattr(trainer_module, "reward", lambda state, *args, **kwargs: state[:, 0]*0+2.)
    batch = replay.sample(6)
    batch['state'][:, COUNT] = 0
    batch['state'][:, TIME] = timeout_step/400 if timeout_step else 1.
    loss, _, targets, _, valid = t.imagine(batch)
    length = timeout_step or 3
    c = torch.sigmoid((.3-t.wm.positions)/t.cfg.survival_width)
    # Independent reference: explicit survival products, plus a final value.
    def n_step_return(start, n):
        result = sum(t.cfg.gamma**j * c[start:start+j].prod() * 2 for j in range(n))
        return result + t.cfg.gamma**n * c[start:start+n].prod() * 3.5
    expected = n_step_return(0, length)
    torch.testing.assert_close(-loss, expected)
    actual_gradient, = torch.autograd.grad(loss, t.wm.positions, retain_graph=True)
    expected_gradient, = torch.autograd.grad(-expected, t.wm.positions)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert (actual_gradient[:length] > 0).all()  # Loss pushes away from falling, also beyond 0.3 m.
    for start in range(length):
        remaining = length-start
        # Lambda return as a weighted mixture of n-step returns, not a reverse recursion.
        mixed = sum((1-lam)*lam**(n-1)*n_step_return(start,n) for n in range(1,remaining))
        mixed = mixed + lam**(remaining-1)*n_step_return(start,remaining)
        torch.testing.assert_close(targets[start], mixed.detach().expand(6))
    assert valid[:length].all() and not valid[length:].any()
    assert not targets[length:].any()
    if lam == 1.:
        torch.testing.assert_close(targets[0], (-loss).detach().expand(6))


def test_checkpoint_contract_rejects_old_action_semantics_and_restores_v15(tmp_path):
    t, replay, _, _ = trainer_fixture()
    t.update(replay)
    metadata = dict(environment_contract="test", joint_names=[], context_schema={}, observation_protocol="test")
    path = tmp_path/'checkpoint.pt'
    t.save(path, metadata)
    payload = torch.load(path, weights_only=False)
    assert payload['version'] == 17 and 'target_critic' not in payload
    restored, _, _, _ = trainer_fixture()
    for legacy in (4, 10, 11, 12, 13, 14):
        old = dict(payload, version=legacy)
        with pytest.raises(ValueError, match="Version 17 required"):
            restored.restore(old, metadata, restore_rng=False)
    restored.restore(payload, metadata, restore_rng=False)
    for name in ('wm','actor','critic','opt_wm','opt_actor','opt_critic'):
        torch.testing.assert_close(getattr(restored,name).state_dict(), getattr(t,name).state_dict(), rtol=0, atol=0)
    assert restored.iteration == t.iteration and restored.optimizer_steps == t.optimizer_steps
    assert restored.real_steps == t.real_steps and restored.step_credit == t.step_credit
    payload['imagination_protocol'] = 'wrong'
    with pytest.raises(ValueError, match="imagination protocol"):
        restored.restore(payload, metadata, restore_rng=False)


def test_count_termination_suppresses_simultaneous_timeout_bootstrap(monkeypatch):
    t, replay, _, _ = trainer_fixture()
    class Identity(torch.nn.Module):
        def forward(self, state, target, context):
            return state[:, :55]
    class Value(torch.nn.Module):
        def forward(self, state, context):
            return state[:, 0]*0+100.
    t.wm, t.critic = Identity(), Value()
    monkeypatch.setattr(trainer_module, 'reward', lambda state, *args, **kwargs: state[:, 0]*0+2.)
    batch = replay.sample(6)
    batch['state'][:, COUNT] = 50
    batch['state'][:, TIME] = 1/400
    loss, _, targets, _, valid = t.imagine(batch)
    torch.testing.assert_close(loss, torch.tensor(-2.))
    torch.testing.assert_close(targets[0], torch.full((6,), 2.))
    assert not targets[1:].any() and not valid[1:].any()


def test_holdout_reports_full_position_and_geodesic_rmse():
    from viserdex.supermimic.validation import open_loop_errors
    s, limits = fixture(2)
    nxt = s[:,:55].clone()
    nxt[:,42:45] += torch.tensor([.003,.004,0.])
    nxt[:,ROT] = rotation_vector_quat(torch.tensor([[.1,0.,0.]]).expand(2,-1))
    class Identity:
        def __call__(self,s,target,context): return s[:,:55]
    row = dict(state=s, next_physical=nxt, action=torch.zeros(2,21), done=torch.zeros(2,dtype=torch.bool))
    result = open_loop_errors(Identity(),[row],torch.ones(2,1),limits,config20())['1']
    assert result['object_position_vector_rmse_m'] == pytest.approx(.005)
    assert result['object_orientation_geodesic_rmse_rad'] == pytest.approx(.1)
    assert result['object_orientation_rad'] == pytest.approx(.1/(3**.5))
