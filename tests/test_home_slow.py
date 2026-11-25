"""HOME preference, physical controller limits and full-time object costs."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import math

import pytest
import torch

from viserdex.supermimic import dex_reference as dex
from viserdex.supermimic.core import action_target, advance, reward
from viserdex.supermimic.constraints import violation_costs
from viserdex.supermimic.quiet_hold import fresh_options, home_pose_penalty
from test_dex_reference_v33 import config


def restrained_config():
    profile, digest = dex.load_profile(dex.DEFAULT_PROFILE.with_name('control_home_slow.yaml'))
    options = fresh_options()
    options.update(dex.profile_options(profile, digest))
    return replace(config(), **options, stability_reward_scale=10.)


def state_for(cfg, n=2):
    state = torch.zeros(n, cfg.state_dim)
    state[:, :21] = torch.tensor(cfg.action_home)
    state[:, 48] = state[:, 82] = 1.
    state[:, dex.ZERO] = torch.linspace(-math.pi/180, math.pi/180, n)[:, None]
    state[:, dex.EMA] = state[:, :21] + state[:, dex.ZERO]
    return state


def test_home_preference_free_band_and_restoring_gradient():
    cfg = restrained_config()
    home = torch.tensor(cfg.action_home, dtype=torch.float64)
    offsets = torch.tensor([0., 14., 15., 30., -30., 45.], dtype=torch.float64)
    joints = (home + torch.deg2rad(offsets[:, None])).requires_grad_()
    costs = home_pose_penalty(joints, home, cfg)
    # Mean across 21 joints, not a joint-count-dependent sum.
    torch.testing.assert_close(costs, torch.tensor([0., 0., 0., .5, .5, 2.], dtype=torch.float64), atol=1e-12, rtol=0)
    costs.sum().backward()
    assert joints.grad[:3].abs().max() < 1e-12
    assert (joints.grad[3] > 0).all() and (joints.grad[4] < 0).all()


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('ablation', [False, True])
def test_measured_home_reward_matches_imagination_in_joint_order(wrapped, ablation):
    from test_single_goal_command import fixture
    from viserdex.supermimic.single_goal import dense_stability_reward, dense_orientation_reward
    command = fixture()
    cfg = restrained_config()
    if ablation:
        profile, digest = dex.load_profile(dex.DEFAULT_PROFILE.with_name('control_no_target_home.yaml'))
        cfg = replace(cfg, **dex.profile_options(profile, digest))
        assert cfg.home_pose_weight == cfg.hold_target_motion_weight == 0.
        assert cfg.hold_joint_motion_weight == 1.
    for name in fresh_options():
        setattr(command.cfg, name, getattr(cfg, name))
    command.cfg.stability_reward_scale = cfg.stability_reward_scale
    command.cfg.hold_duration = cfg.hold_duration
    command._env.step_dt = cfg.control_dt
    state = state_for(cfg)
    physical = state[:, :55].clone()
    physical[0, 0] += math.radians(45.)
    physical[1, 2] -= math.radians(30.)
    physical[:, 21:42] = math.pi / 9
    # Native articulation order intentionally differs from policy order.
    ids = torch.arange(20, -1, -1)
    def wrap(x):
        return SimpleNamespace(torch=x) if wrapped else x
    data = SimpleNamespace(joint_pos=wrap(physical[:, :21].flip(-1)),
        default_joint_pos=wrap(state[:, :21].flip(-1)), joint_vel=wrap(physical[:, 21:42].flip(-1)))
    limits = torch.tensor(cfg.network_joint_limits)
    raw = torch.ones(2, 21)
    target = action_target(state, raw, limits, cfg)
    action = SimpleNamespace(_asset=SimpleNamespace(data=data), _joint_ids=ids,
        processed_actions=target, previous_processed_actions=state[:, dex.EMA]-state[:, dex.ZERO])
    command._env.action_manager = SimpleNamespace(get_term=lambda _: action)
    actual = cfg.control_dt * (dense_orientation_reward(command._env) + dense_stability_reward(command._env))
    predicted = reward(state, physical, raw, cfg.control_dt, 0., 0., 0., 0., hold_cfg=cfg)
    torch.testing.assert_close(actual, predicted, atol=2e-6, rtol=1e-5)


def test_controller_and_imagination_slew_match_through_partial_resets():
    from viserdex.supermimic.direct_actions import DirectJointPositionAction
    cfg = restrained_config()
    state = state_for(cfg)
    limits = torch.tensor(cfg.network_joint_limits).repeat(2,1,1)
    action = DirectJointPositionAction.__new__(DirectJointPositionAction)
    action.cfg = SimpleNamespace(dex_reference_profile=cfg.control_profile, alpha=cfg.ema_alpha)
    action._raw_actions = torch.zeros(2,21)
    action._processed_actions = state[:, :21].clone()
    action.previous_processed_actions = torch.zeros(2,21)
    action._quiet_target_ready = torch.zeros(2,1,dtype=torch.bool)
    action.encoder_bias = torch.zeros(21)
    action._joint_ids = torch.arange(21)
    home = state[:, :21].clone()
    action._asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=home, joint_pos=home))
    action.position_limits = limits
    action.zero_bias = state[:, dex.ZERO].clone()
    action.ema_encoder = state[:, dex.EMA].clone()
    action.target_max_delta = cfg.target_slew_rate_rad_s * cfg.control_dt
    for step in range(60):
        if step == 20:
            other = action.processed_actions[1].clone()
            action.reset([0])
            torch.testing.assert_close(action.processed_actions[1], other)
        state[:, dex.ZERO] = action.zero_bias
        state[:, dex.EMA] = action.ema_encoder
        previous = state[:, dex.EMA] - state[:, dex.ZERO]
        raw = torch.full((2,21), 1. if step < 30 else -1.)
        expected = action_target(state, raw, limits, cfg)
        action.process_actions(raw)
        torch.testing.assert_close(action.processed_actions, expected, atol=2e-7, rtol=0)
        assert (action.processed_actions - previous).abs().max() <= math.radians(2.4) + 2e-7
        nxt = advance(state, state[:,:55], raw, expected, state[:,79:83], hold_cfg=cfg)
        torch.testing.assert_close(nxt[:,dex.EMA], action.ema_encoder, atol=2e-7, rtol=0)


def test_far_goal_object_cost_has_nonzero_speed_gradient():
    cfg = restrained_config()
    state = state_for(cfg)
    state[:, 79:83] = torch.tensor([1.,0.,0.,0.])  # 180-degree error
    physical = state[:,:55].clone()
    physical[:, 52] = 2.
    physical[:, 49] = .1
    physical.requires_grad_()
    costs,_ = violation_costs(state, physical, torch.zeros(2,21), torch.tensor(cfg.network_joint_limits), cfg)
    torch.testing.assert_close(costs[:,2:4], torch.ones(2,2))
    costs[:,2:4].sum().backward()
    assert (physical.grad[:,52] > 0).all() and (physical.grad[:,49] > 0).all()
    # Historical v33 keeps its old vanishing far-goal speed cost.
    old = config()
    old_cost,_ = violation_costs(state, physical.detach(), torch.zeros(2,21), torch.tensor(old.network_joint_limits), old)
    assert old_cost[:,2:4].max() < 1e-20


def test_hard_limit_gradients_are_finite_and_preserve_unsaturated_direction():
    previous = torch.zeros(2,21)
    raw = torch.stack((torch.full((21,), .01), torch.ones(21))).requires_grad_()
    filtered = dex.filter_target(previous, raw, .51, math.radians(2.4))
    filtered.sum().backward()
    torch.testing.assert_close(raw.grad[0], torch.full((21,), .51))
    torch.testing.assert_close(raw.grad[1], torch.zeros(21))


def test_exported_policy_keeps_slew_limit_and_committed_memory(tmp_path):
    from viserdex.supermimic.inference import export_actor, ManualConvexPolicy
    from viserdex.supermimic.trainer import Trainer
    from viserdex.supermimic.revised_contract import joint_names
    from viserdex.distillation.config import PerceptionNoiseCfg
    cfg = restrained_config()
    state = state_for(cfg, 1)
    limits = torch.tensor(cfg.network_joint_limits)[None]
    trainer = Trainer(cfg, PerceptionNoiseCfg(enabled=False, joint_enabled=False),
        torch.ones(1,65), limits, torch.tensor([[0.,0.,0.,1.]]), 'cpu')
    # Large commands explicitly exercise clipping, including after reset.
    with torch.no_grad():
        trainer.actor.net[-1].weight.zero_()
        trainer.actor.net[-1].bias.fill_(4.)
    metadata = dict(environment_contract='test', joint_names=list(joint_names(cfg)), context_schema={},
        observation_protocol=cfg.observation_protocol, state_protocol=cfg.state_protocol,
        observation_dim=75, quaternion_order='XYZW', joint_limits_rad=cfg.network_joint_limits)
    trainer.save(tmp_path/'model.pt', metadata)
    export_actor(tmp_path/'model.pt', tmp_path/'actor.pt')
    policy = ManualConvexPolicy(tmp_path/'actor.pt')
    for _ in range(2):
        policy.reset(state[0,dex.ZERO])
        policy.set_goal(state[0,79:83], state[0,76:79])
        previous = state[0,:21].clone()
        for _ in range(4):
            target,_,_ = policy.propose(state[0,:21]+state[0,dex.ZERO], state[0,21:42],
                                        state[0,42:45], state[0,45:49])
            target = torch.tensor(target)
            assert (target-previous).abs().max() <= math.radians(2.4)+2e-7
            before_commit = policy.state[:,dex.EMA].clone()
            policy.commit()
            torch.testing.assert_close(policy.state[0,dex.EMA], target+state[0,dex.ZERO])
            assert not torch.equal(before_commit, policy.state[:,dex.EMA])
            previous = target


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_home_and_motion_world_model_cuda_capture_matches_eager():
    from test_dex_reference_v33 import fixture, window
    from viserdex.supermimic.trainer import Trainer
    cpu, state = fixture()
    cpu.cfg = restrained_config()
    eager = Trainer(cpu.cfg, cpu.noise_cfg, cpu.context.cuda(), cpu.limits.cuda(), cpu.base_inverse.cuda(), 'cuda')
    captured = copy.deepcopy(eager)
    captured.wm_cuda_graph = True
    batch = {k:v.cuda() for k,v in window(cpu, state).items()}
    class Buffer:
        def sample(self, *args, **kwargs): return batch
        def update_priorities(self, *args): pass
    for step in range(2):
        # Exercise nonzero HOME and moving-object targets, not just the deadband.
        batch['next_physical'][:,:,0] = state[0,0].item() + .6 + .1*step
        batch['next_physical'][:,:,52] = 2. + step
        batch['action'].fill_(.9)
        a,b = eager.wm_update(Buffer()), captured.wm_update(Buffer())
        for name in a:
            assert b[name] == pytest.approx(a[name], rel=3e-5, abs=3e-6), name
        for name, value in eager.wm.state_dict().items():
            torch.testing.assert_close(value, captured.wm.state_dict()[name], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('name,value', [('target_slew_rate_deg_s', -1.), ('home_pose_weight', float('nan')),
    ('home_pose_scale_deg', 0.), ('object_motion_linear_speed_m_s', 0.),
    ('hold_target_motion_weight', -1.), ('hold_target_motion_weight', float('nan'))])
def test_invalid_profile_fails_before_training(name, value):
    p,_ = dex.load_profile(dex.DEFAULT_PROFILE.with_name('control_home_slow.yaml'))
    p[name] = value
    with pytest.raises(ValueError):
        dex.validate_profile(p)
