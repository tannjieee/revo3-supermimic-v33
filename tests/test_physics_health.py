import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from viserdex.supermimic.physics_health import PhysicsHealthMonitor, audit_compiled_joint_limits


def monitor(capacity=3):
    return PhysicsHealthMonitor(torch.tensor([[-.1, 1.57], [-.5, .5]]), ['mcp', 'mpr'], .05, capacity)


def test_small_mechanical_overshoot_and_fast_valid_motion_are_diagnostic_only():
    health = monitor()
    state = torch.tensor([[1.57 + math.radians(4.9), 0., 30., 0., 0.]])
    original = state.clone()
    assert health.inspect(state, 'post_step_before_reset') is None
    torch.testing.assert_close(state, original)


def test_gross_mechanical_violation_is_fatal_evidence_not_clipped_state():
    health = monitor()
    state = torch.tensor([[0., 0., 0., 0., 0.], [4.52847, 0., 289.25, 0., 0.]])
    original = state.clone()
    report = health.inspect(state, 'post_step_before_reset')
    assert report['affected_worlds'] == [1]
    assert report['flags'] == ['MECHANICAL_LIMIT']
    assert report['worst_joint'] == 'mcp'
    assert report['mechanical_limit_tolerance_deg'] == 5.
    torch.testing.assert_close(state, original)


def test_object_drop_is_kept_but_nonfinite_object_state_is_rejected():
    health = monitor()
    state = torch.tensor([[0., 0., 0., 0., -10.]])
    assert health.inspect(state, 'post_step_before_reset') is None
    state[0, -1] = float('nan')
    report = health.inspect(state, 'post_step_before_reset')
    assert report['flags'] == ['NONFINITE_PHYSICAL_STATE']
    json.dumps(report, allow_nan=False)


def test_trace_survives_wrap_and_does_not_mislabel_stale_post_step_data():
    health = monitor(3)
    for step in range(5):
        state = torch.full((2, 6), float(step))
        action = torch.full((2, 2), float(step + 10))
        health.begin(state, action)
        if step != 4:
            health.finish(state[:, :5] + 1, action, action + 2, action + 3)
    trace = health.snapshot([1])
    assert trace['control_steps'] == [3, 4, 5]
    assert trace['complete'] == [True, True, False]
    assert trace['world_ids'] == [1]
    assert trace['state'][:, 0, 0].tolist() == [2., 3., 4.]
    assert torch.isnan(trace['next_physical'][-1]).all()
    assert trace['applied_torque'][0, 0, 0] == 15.


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


def compiled_solver():
    return SimpleNamespace(
        mj_model=SimpleNamespace(njnt=2, joint=lambda i: SimpleNamespace(name=['env_mcp', 'env_mpr'][i])),
        mjw_model=SimpleNamespace(jnt_range=Array([[[-.1, 1.57], [-.5, .5]]]),
            jnt_limited=Array([True, True]), jnt_solref=Array([[[.02, 1.], [.02, 1.]]]),
            jnt_solimp=Array([[[.9, .95, .001, .5, 2.], [.9, .95, .001, .5, 2.]]]),
            jnt_margin=Array([[0., 0.]])))


def test_compiled_limit_audit_detects_disabled_or_misimported_limits():
    solver = compiled_solver()
    limits = [[-.1, 1.57], [-.5, .5]]
    report = audit_compiled_joint_limits(solver, ['mcp', 'mpr'], limits)
    assert report['all_compiled_limits_enabled']
    solver.mjw_model.jnt_limited = Array([False, True])
    with pytest.raises(ValueError, match='differ'):
        audit_compiled_joint_limits(solver, ['mcp', 'mpr'], limits)
    solver = compiled_solver()
    solver.mjw_model.jnt_range = Array([[[-.1, 10.], [-.5, .5]]])
    with pytest.raises(ValueError, match='differ'):
        audit_compiled_joint_limits(solver, ['mcp', 'mpr'], limits)
