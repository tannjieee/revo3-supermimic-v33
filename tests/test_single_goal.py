"""Stationary single-command rewards and continuing-return behavioral tests."""
from dataclasses import replace
import math
import pytest
import torch
from viserdex.supermimic.core import *
from viserdex.supermimic.trainer import Trainer, imagined_continuation, lambda_returns
from viserdex.supermimic.sensors import initial_sensors
from viserdex.supermimic.inference import ManualConvexPolicy
from viserdex.distillation.config import PerceptionNoiseCfg
from test_manual_convex_policy import make_checkpoint
from viserdex.supermimic.state_layout import COMPACT
GOAL_POS, GOAL_ROT = COMPACT.goal_pos, COMPACT.goal_rot


def fixture(n=1):
    cfg = manual_convex_config()
    s = torch.zeros(n, COMPACT.dim)
    s[:, ROT] = s[:, GOAL_ROT] = torch.tensor([0., 0., 0., 1.])
    s[:, POS] = torch.tensor([.02, .01, .1])
    s[:, GOAL_POS] = s[:, POS]
    return cfg, s, torch.zeros(n, 21)


def evaluate(s, physical, a, cfg):
    return reward(s, physical, a, cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight,
                  cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight, hold_cfg=cfg)


def test_reward_is_active_from_first_step_and_goal_persists_forever():
    cfg, s, a = fixture()
    alternate = rotation_vector_quat(torch.tensor([[1., 0., 0.]]))
    for step in range(1, 1001):
        physical = s[:, :55]
        actual = evaluate(s, physical, a, cfg).item()
        assert actual == pytest.approx(.7)
        s = advance(s, physical, a, a, alternate, 400, hold_cfg=cfg)
        torch.testing.assert_close(s[:, GOAL_ROT], torch.tensor([[0., 0., 0., 1.]]))
    assert COMPACT.time is None
    assert s.shape[-1] == 84 and s[:, COMPACT.delay_ready].item() == 1


def test_removed_penalties_cannot_bias_midpoint_or_home():
    cfg, s, a = fixture(2)
    a[1] = .9
    s[:, COMPACT.last_action] = a  # equally steady commands, different magnitudes
    physical = s[:, :55].clone()
    physical[0, Q] = torch.tensor(cfg.action_home)
    physical[1, Q] = -5
    torch.testing.assert_close(evaluate(s, physical, a, cfg), torch.full((2,), .7))


def test_recovery_after_losing_hold_gets_no_new_bonus_or_goal():
    cfg, s, a = fixture()
    ticks = torch.full((len(s),), 10.)
    physical = s[:, :55].clone()
    physical[:, OMEGA] = torch.tensor([0., 0., 1.])
    ticks, event = persistent_hold_transition(ticks, torch.zeros(len(s)), physical[:, OMEGA].norm(dim=-1), .05,
                                              already_completed=torch.ones(len(s), dtype=torch.bool))
    assert not event.any() and ticks.item() == 0
    assert evaluate(s, physical, a, cfg).item() == pytest.approx(.05*(11+3*math.exp(-4)))
    physical[:, OMEGA] = 0
    ticks[:] = 9
    ticks, event = persistent_hold_transition(ticks, torch.zeros(len(s)), physical[:, OMEGA].norm(dim=-1), .05,
                                              already_completed=torch.ones(len(s), dtype=torch.bool))
    assert ticks.item() == 10 and not event.any()
    assert evaluate(s, physical, a, cfg).item() == pytest.approx(.7)


def test_dense_target_attraction_and_motion_costs_apply_together():
    cfg, s, a = fixture(2)
    physical = s[:, :55].clone()
    physical[:, QD] = 2
    physical[1, ROT] = rotation_vector_quat(torch.tensor([1., 0., 0.]))
    a[:] = .3
    value = evaluate(s, physical, a, cfg)
    no_cost = evaluate(s, physical, a, replace(cfg, joint_vel_weight=0, action_rate_weight=0))
    expected_cost = .05*(cfg.joint_vel_weight*21*4 + cfg.action_rate_weight*21*.3**2)
    torch.testing.assert_close(no_cost - value, torch.full((2,), expected_cost))
    assert value[0] > value[1] > 0


def test_evaluation_time_has_no_reward_or_transition_effect():
    cfg, s, a = fixture()
    diagnostic = replace(cfg, goal_evaluation_s=3.37)
    torch.testing.assert_close(evaluate(s, s[:, :55], a, cfg), evaluate(s, s[:, :55], a, diagnostic))
    torch.testing.assert_close(advance(s, s[:, :55], a, a, s[:, GOAL_ROT], 400, hold_cfg=cfg),
                               advance(s, s[:, :55], a, a, s[:, GOAL_ROT], 400, hold_cfg=diagnostic))
    for kwargs in ({'goal_evaluation_s': 0}, {'goal_evaluation_s': float('nan')},
                   {'gamma': 1}, {'hand_pose_weight': .1}, {'action_l2_weight': .1}, {'success_bonus': 1}):
        with pytest.raises(ValueError):
            manual_convex_config(**kwargs)


def test_gamma_and_lambda_bootstrap_continuing_hold():
    cfg, s, a = fixture()
    assert cfg.gamma == .99
    assert cfg.gamma**200 == pytest.approx(.1339796749)
    assert .99**(1.5*200) == pytest.approx(.0490408941)
    value = .5/(1-cfg.gamma)
    returns = lambda_returns(torch.full((8, 1), .5), torch.ones(8, 1),
                             torch.full((9, 1), value), cfg.gamma, cfg.lam)
    torch.testing.assert_close(returns, torch.full((8, 1), value))
    assert imagined_continuation(s, s[:, :55], cfg.survival_width, single_goal=True).item() > .999
    dropped = s[:, :55].clone()
    dropped[:, POS] = 1
    assert imagined_continuation(s, dropped, cfg.survival_width, single_goal=True).item() < 1e-5


def test_imagination_keeps_goal_without_clock_and_has_valid_gradient(monkeypatch):
    cfg, s, a = fixture(2)
    cfg = replace(cfg, wm_hidden=(32,), actor_hidden=(32,), critic_hidden=(16,),
                  actor_batch=2, critic_minibatches=1, horizon=12)
    noise = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
    limits = torch.tensor([[-1., 1.]]).expand(2, 21, 2)
    trainer = Trainer(cfg, noise, torch.ones(2, 2), limits,
                      torch.tensor([[0., 0., 0., 1.]]).expand(2, -1), 'cpu', 400)
    class StaticPhysical(torch.nn.Module):
        def forward(self, state, target, context):
            return state[:, :55]
    trainer.wm = StaticPhysical()
    batch = dict(state=s, history=torch.zeros(2, 10, 81), env_id=torch.arange(2))
    batch.update({'sensor/'+k: v for k, v in initial_sensors(2, 'cpu', noise).items()})
    import viserdex.supermimic.trainer as module
    def fail_sampling(_):
        raise AssertionError('A single-goal rollout sampled a replacement command')
    monkeypatch.setattr(module, 'sample_xy_goal', fail_sampling)
    loss, states, targets, context, valid = trainer.imagine(batch)
    assert valid.all() and torch.isfinite(targets).all()
    torch.testing.assert_close(states[:, :, GOAL_ROT], s[:, GOAL_ROT].expand(12, -1, -1))
    assert states.shape[-1] == 84 and (states[1:, :, COMPACT.delay_ready] == 1).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in trainer.actor.parameters() if p.grad is not None)


def test_inference_one_command_persists_without_clock_and_reissue_preserves_action(tmp_path):
    path = tmp_path/'actor.pt'
    make_checkpoint(path)
    policy = ManualConvexPolicy(path)
    q = [0., 0., 0., 1.]
    policy.set_goal(q, [.02, 0., .1])
    for i in range(420):
        _, raw, obs = policy.propose([0.]*21, [0.]*21, [.02, 0., .1], q)
        assert obs.shape == (81,)
        torch.testing.assert_close(torch.tensor(obs[66:69]), torch.tensor([.02, 0., .1]))
        policy.commit()
    old_action = policy.state[:, COMPACT.last_action].clone()
    policy.set_goal(q, [.02, 0., .1])
    _, _, obs = policy.propose([0.]*21, [0.]*21, [.02, 0., .1], q)
    torch.testing.assert_close(torch.tensor(obs[42:63]), old_action[0])
