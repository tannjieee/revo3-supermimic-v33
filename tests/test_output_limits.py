"""Runtime override preserves provenance, delay, joint coverage and torque clipping."""
import math
from types import SimpleNamespace
import numpy as np
import pytest

from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.supermimic.identified_hand import load_identified_hand, actuator_config
from viserdex.supermimic.output_limits import load_output_limits, configured_output_limits


def test_new_limits_use_joint_side_si_units_and_preserve_identified_gains():
    fit, info = load_identified_hand()
    cfg = actuator_config()
    profile = configured_output_limits(cfg)
    assert profile['velocity_limit_rad_s'] == pytest.approx(10*math.pi/3)
    assert profile['velocity_limit_deg_s'] == 600
    assert profile['effort_limit_Nm'] == 1.28
    assert profile['hardware_rating_verified'] is False
    assert cfg.joint_names_expr == list(REVO3_JOINT_NAMES)
    assert list(cfg.stiffness.values()) == fit['effective_kp_Nm_rad']
    assert list(cfg.damping.values()) == fit['effective_kd_Nm_s_rad']
    assert cfg.min_delay == cfg.max_delay == info['delay_steps'] == 3
    assert info['sha256']['best_parameters.json'] == '5c3ae8f00334d20941bf51e58e0cddba7d7b1efb95437e1061aa518302bc7736'


def test_saved_legacy_profile_is_still_recognized_but_partial_overrides_fail():
    fit, _ = load_identified_hand()
    cfg = actuator_config()
    legacy = dict(zip(REVO3_JOINT_NAMES,fit['effective_effort_limit_Nm']))
    cfg.from_dict(dict(actuator_effort_limit=legacy,joint_effort_limit=legacy.copy(),
                      actuator_velocity_limit=None,joint_velocity_limit=None))
    assert configured_output_limits(cfg) is None
    cfg.actuator_effort_limit[REVO3_JOINT_NAMES[0]] = 1.28
    with pytest.raises(ValueError, match='declared user profile'):
        configured_output_limits(cfg)
    cfg.actuator_effort_limit.pop(REVO3_JOINT_NAMES[0])
    with pytest.raises(ValueError, match='21 joints'):
        configured_output_limits(cfg)


def test_native_pd_caps_all_21_joints_both_directions_and_keeps_15ms_delay():
    import warp as wp
    from newton.actuators import Actuator, DrivePD, Delay, ClampingMaxEffort
    fit, info = load_identified_hand()
    profile = load_output_limits()
    def arr(x,dtype=wp.float32):
        return wp.array(x,dtype=dtype,device='cpu')
    actuator = Actuator(indices=arr(np.arange(21),wp.uint32),
        drive=DrivePD(kp=arr(fit['effective_kp_Nm_rad']),kd=arr(fit['effective_kd_Nm_s_rad'])),
        delay=Delay(delay_steps=arr([3]*21,wp.int32),max_delay=3),
        clamping=[ClampingMaxEffort(max_effort=arr([profile['effort_limit_Nm']]*21))])
    a,b=actuator.state(),actuator.state()
    state=SimpleNamespace(joint_q=arr(np.zeros(21)),joint_qd=arr(np.zeros(21)))
    control=SimpleNamespace(joint_target_q=arr(np.zeros(21)),joint_target_qd=arr(np.zeros(21)),joint_f=arr(np.zeros(21)))
    # Large synthetic actuator-only errors force saturation; no physical state
    # or robot is driven by this test.
    for tick in range(12):
        control.joint_target_q.assign(np.full(21,20. if tick<4 else -20.,np.float32))
        control.joint_f.zero_()
        actuator.step(state,control,a,b,dt=info['pd_period_s']);a,b=b,a
        expected=profile['effort_limit_Nm'] if tick<7 else -profile['effort_limit_Nm']
        np.testing.assert_allclose(control.joint_f.numpy(),expected,atol=1e-7,rtol=0)
