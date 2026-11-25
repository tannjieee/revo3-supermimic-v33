"""Compact context/state and the frozen identified actuator response contract."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from viserdex.supermimic.context import pack_context
from viserdex.supermimic.identified_hand import load_identified_hand
from viserdex.supermimic.core import manual_convex_config, advance, full_features
from viserdex.supermimic.networks import Dynamics
from viserdex.supermimic.state_layout import COMPACT, fields


def test_frozen_fit_is_complete_and_uses_identification_pd_clock():
    p, info = load_identified_hand()
    assert info['model'] == 'fit_03'
    assert info['sha256']['best_parameters.json'] == '5c3ae8f00334d20941bf51e58e0cddba7d7b1efb95437e1061aa518302bc7736'
    assert info['delay_steps'] == 3 and info['effective_delay_s'] == .015
    assert p['encoder_bias_rad'][0] == pytest.approx(np.pi / 180)


@pytest.mark.parametrize('n', [1, 4])
def test_context_selection_uses_declared_randomization_even_with_no_sample_variance(n):
    values = {'mu': torch.ones(n, 144), 'fixed': torch.zeros(n, 144), 'kp': torch.ones(n, 21)}
    context, schema, fixed = pack_context(values, {'mu': True, 'fixed': False, 'kp': True})
    assert context.shape == (n, 165)
    assert schema == {'mu': [0, 144], 'kp': [144, 165]}
    assert fixed == {'fixed': [0.] * 144}
    # Independent per-collider changes remain visible, not averaged by body.
    values['mu'][0, 73] = 1.234
    other, _, _ = pack_context(values, {'mu': True, 'fixed': False, 'kp': True})
    assert (other != context).sum() == 1
    assert other[0, 73] == pytest.approx(1.234)


def test_context_refuses_silently_omitting_varying_field():
    with pytest.raises(AssertionError):
        pack_context({'fixed': torch.tensor([[1.], [2.]])}, {'fixed': False})


def test_compact_state_preserves_goal_and_pending_last_command_without_old_memory():
    cfg = manual_convex_config(wm_hidden=(16,))
    s = torch.zeros(2, cfg.state_dim)
    s[:, 48] = s[:, 82] = 1
    s[:, COMPACT.goal_pos] = .05
    action = torch.full((2, 21), .2, requires_grad=True)
    nxt = advance(s, s[:, :55], action, action, torch.zeros(2, 4), hold_cfg=cfg)
    assert nxt.shape[-1] == 84 and full_features(nxt).shape[-1] == 88
    assert list(fields())[-1] == 'actuator_delay_ready'
    torch.testing.assert_close(nxt[:, COMPACT.goal_pos], s[:, COMPACT.goal_pos])
    torch.testing.assert_close(nxt[:, COMPACT.last_action], action)
    assert (nxt[:, COMPACT.delay_ready] == 1).all()
    wm = Dynamics(65, cfg)
    x = wm.inputs(nxt, action, torch.ones(2, 65))
    assert x.shape[-1] == 165  # 465 with 300 geometric features
    torch.testing.assert_close(x[:, 78:99], action)
    assert (x[:, 99:100] == 1).all()
    x[:, 78:99].sum().backward()
    torch.testing.assert_close(action.grad, torch.ones_like(action))
    with pytest.raises(ValueError, match='first-order'):
        replace(cfg, action_second_diff_weight=.01)


def test_native_identified_pd_response_and_delay_match_independent_oracle():
    wp = pytest.importorskip('warp')
    pytest.importorskip('newton')
    from newton.actuators import Actuator, DrivePD, Delay, ClampingMaxEffort
    p, info = load_identified_hand()
    def arr(x, dtype=wp.float32):
        return wp.array(x, dtype=dtype, device='cpu')
    kp, kd, cap, bias = [np.array(p[k], dtype=np.float32) for k in (
        'effective_kp_Nm_rad', 'effective_kd_Nm_s_rad', 'effective_effort_limit_Nm', 'encoder_bias_rad')]
    actuator = Actuator(indices=arr(np.arange(21), wp.uint32),
        drive=DrivePD(kp=arr(kp), kd=arr(kd)),
        delay=Delay(delay_steps=arr([3] * 21, wp.int32), max_delay=3),
        clamping=[ClampingMaxEffort(max_effort=arr(cap))])
    first, second = actuator.state(), actuator.state()
    q = np.linspace(.1, .3, 21, dtype=np.float32)
    qd = np.linspace(-.1, .1, 21, dtype=np.float32)
    state = SimpleNamespace(joint_q=arr(q), joint_qd=arr(qd))
    control = SimpleNamespace(joint_target_q=arr(q), joint_target_qd=arr(np.zeros(21)),
                              joint_f=arr(np.zeros(21)))
    commands, torques = [], []
    for tick in range(30):
        # Three distinct 50 ms commands; actuator updates at 5 ms.
        command = q + (.02 if tick < 10 else -.03 if tick < 20 else 2.)
        commands.append(command.copy())
        control.joint_target_q.assign(command + bias)
        control.joint_f.zero_()
        actuator.step(state, control, first, second, dt=info['pd_period_s'])
        first, second = second, first
        delayed = commands[max(0, tick - 3)]
        expected = np.clip(kp * (delayed - (q-bias)) - kd * qd, -cap, cap)
        np.testing.assert_allclose(control.joint_f.numpy(), expected, atol=4e-7, rtol=3e-6)
        torques.append(control.joint_f.numpy().copy())
    np.testing.assert_array_equal(torques[9], torques[12])
    assert np.max(np.abs(torques[13] - torques[12])) > .1
    np.testing.assert_allclose(torques[-1], cap, atol=1e-7)
    # Reset starts with the new command; it must not reuse pre-reset history.
    first.reset()
    second.reset()
    control.joint_target_q.assign(q + bias)
    control.joint_f.zero_()
    actuator.step(state, control, first, second, dt=info['pd_period_s'])
    np.testing.assert_allclose(control.joint_f.numpy(), np.clip(kp*bias-kd*qd, -cap, cap), atol=4e-7)


def palm_context_fixture(n=2, palm_index=7):
    from viserdex.supermimic.context import compact_context
    names = [f'link_{i}' for i in range(21)]
    names.insert(palm_index, 'world')
    events, values = {}, {}
    for name, count in (('robot', 22), ('object', 1)):
        bodies = names if name == 'robot' else ['die']
        keep = [i for i, body in enumerate(bodies) if body != 'world']
        mass = torch.linspace(.01, .03, count).repeat(n, 1)
        inertia = torch.eye(3).flatten().expand(n, count, 9).clone() * 1e-5
        live = mass.clone()
        live[:, keep] *= torch.linspace(.96, 1.04, n)[:, None]
        live_inertia = inertia * (live / mass)[..., None]
        events[name + '_scale_mass'] = SimpleNamespace(mode='startup', params=dict(
            operation='scale', mass_distribution_params=(.95,1.05),
            asset_cfg=SimpleNamespace(body_ids=keep)), func=SimpleNamespace(
                default_mass=mass, default_inertia=inertia,
                asset=SimpleNamespace(body_names=bodies, is_fixed_base=True)))
        events[name + '_physics_material'] = SimpleNamespace(mode='startup', params=dict(
            static_friction_range=(.7,.7) if name == 'robot' else (.7,1.5), restitution_range=(0.,0.)))
        values[name + '.body_mass'], values[name + '.body_inertia'] = live, live_inertia
        values[name + '.shape_material_mu'] = torch.full((n, 144 if name == 'robot' else 1), .7 if name == 'robot' else 1.)
        values[name + '.shape_material_restitution'] = torch.zeros_like(values[name + '.shape_material_mu'])
    events['robot_joint_stiffness_and_damping'] = SimpleNamespace(mode='startup', params=dict(
        operation='scale', distribution='log_uniform', stiffness_distribution_params=(.6,1.5),
        damping_distribution_params=(.75,1.5)))
    values['actuator.fingers.kp'] = torch.ones(n, 21)
    values['actuator.fingers.kd'] = torch.ones(n, 21)
    values['joint_limits'] = torch.tensor([[-1.,1.]]).expand(n, 21, 2)
    env = SimpleNamespace(device='cpu', event_manager=SimpleNamespace(get_term_cfg=lambda key: events[key]))
    return env, values, events, compact_context


@pytest.mark.parametrize('n,palm_index', [(1, 0), (4, 7), (2, 21)])
def test_fixed_palm_mass_and_inertia_removed_by_name_not_index(n, palm_index):
    env, values, events, compact_context = palm_context_fixture(n, palm_index)
    context, schema, manifest = compact_context(env, values)
    assert context.shape == (n, 65)
    assert schema == {'robot.body_mass': [0,21],
        'object.body_mass': [21,22], 'object.shape_material_mu': [22,23],
        'actuator.fingers.kp': [23,44], 'actuator.fingers.kd': [44,65]}
    np.testing.assert_allclose(manifest['fixed']['robot.shape_material_mu'], [.7] * 144)
    derived = manifest['derived']['robot.body_inertia']
    assert len(derived['default_mass']) == len(derived['default_inertia']) == 21
    assert 'world' not in derived['body_names']
    assert manifest['excluded_bodies']['robot']['mass_and_inertia'] == ['world']
    # Even a different authored palm mass/inertia cannot change xi or reconstruction metadata.
    values['robot.body_mass'][:, palm_index] *= 100
    values['robot.body_inertia'][:, palm_index] *= 50
    events['robot_scale_mass'].func.default_mass[:, palm_index] *= 100
    events['robot_scale_mass'].func.default_inertia[:, palm_index] *= 50
    other, other_schema, other_manifest = compact_context(env, values)
    torch.testing.assert_close(other, context)
    assert other_schema == schema and other_manifest == manifest
    # Object friction remains an independent input; fixed hand friction does not.
    values['object.shape_material_mu'][:, 0] += .1
    changed, _, _ = compact_context(env, values)
    assert (changed != context).sum() == n


@pytest.mark.parametrize('n', [1, 4])
def test_fixed_hand_material_must_match_declared_value_in_every_shape(n):
    env, values, _, compact_context = palm_context_fixture(n)
    # Identical across environments is insufficient: a stray shape must not be
    # silently recorded as fixed metadata when the event declares uniform 0.7.
    values['robot.shape_material_mu'][:, 73] = .9
    with pytest.raises(AssertionError):
        compact_context(env, values)


@pytest.mark.parametrize('failure', ['floating_base', 'palm_randomization'])
def test_palm_omission_refuses_incompatible_physics_or_event_selector(failure):
    env, values, events, compact_context = palm_context_fixture()
    if failure == 'floating_base':
        events['robot_scale_mass'].func.asset.is_fixed_base = False
    else:
        events['robot_scale_mass'].params['asset_cfg'].body_ids = slice(None)
    with pytest.raises(ValueError):
        compact_context(env, values)
