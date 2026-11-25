"""Independent arithmetic, gradient and partial-reset checks for Hold-Penalty2x-v1."""
import torch
from viserdex.supermimic.core import config20, reward_from_orientation_error, ema_target


def test_smooth_weights_ema_and_second_difference_gradient():
    cfg = config20()
    assert cfg.ema_alpha == .5
    assert cfg.joint_vel_weight == 6.25e-5
    assert cfg.action_l2_weight == .00025
    assert abs(cfg.action_rate_weight - 1/90) < 1e-12
    assert cfg.action_second_diff_weight == cfg.action_rate_weight
    state = torch.zeros(2, 173)
    state[:, 76:97] = .2
    state[:, 97:118] = -.1
    state[:, 55:76] = .4
    action = torch.full((2, 21), .3, requires_grad=True)
    physical = state[:, :55].clone()
    physical[:, 21:42] = 2
    error = torch.full((2,), .2)
    result = reward_from_orientation_error(state, physical, action, error, cfg.control_dt, cfg.success_bonus, cfg.action_rate_weight, cfg.joint_vel_weight, cfg.action_l2_weight, cfg.action_second_diff_weight)
    manual = .05 * (1/.3 - 6.25e-5 * 21*4 - .00025*21*.09 - (1/90)*21*.01 - (1/90)*21*.04)
    torch.testing.assert_close(result, torch.full((2,), manual))
    result.sum().backward()
    expected_grad = -.05 * 2 * (.00025*.3 + (1/90)*.1 + (1/90)*(-.2))
    torch.testing.assert_close(action.grad, torch.full_like(action, expected_grad))
    limits = torch.tensor([-1., 1.]).expand(2, 21, 2)
    torch.testing.assert_close(ema_target(state, action, limits, cfg.ema_alpha), torch.full_like(action, .35))


def test_real_reward_history_partial_reset():
    # Load just the reward class with a minimal manager base, no simulator start.
    import importlib.util
    import sys
    from types import SimpleNamespace, ModuleType
    from pathlib import Path
    manager = ModuleType('isaaclab.managers')
    class Base:
        def __init__(self, cfg, env): self._env = env
    manager.ManagerTermBase = Base
    prior = sys.modules.get('isaaclab.managers')
    sys.modules['isaaclab.managers'] = manager
    try:
        path = Path(__file__).parents[1]/'source/viserdex/viserdex/supermimic/action_smoothness.py'
        spec = importlib.util.spec_from_file_location('smooth_reward_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        if prior is None: sys.modules.pop('isaaclab.managers')
        else: sys.modules['isaaclab.managers'] = prior
    actions = SimpleNamespace(action=torch.zeros(2, 21), prev_action=torch.zeros(2, 21))
    env = SimpleNamespace(action_manager=actions)
    term = module.ActionSecondDifference(None, env)
    actions.action.fill_(.2)
    torch.testing.assert_close(term(env), torch.full((2,), 21*.04))
    actions.prev_action.fill_(.2); actions.action.fill_(.4)
    torch.testing.assert_close(term(env), torch.zeros(2))
    term.reset(torch.tensor([0]))
    actions.prev_action[0] = 0; actions.prev_action[1] = .4
    actions.action.fill_(.6)
    expected = torch.tensor([21*.36, 0.])
    torch.testing.assert_close(term(env), expected, atol=1e-6, rtol=1e-5)
    term.reset()
    assert not term.previous_previous.any()
