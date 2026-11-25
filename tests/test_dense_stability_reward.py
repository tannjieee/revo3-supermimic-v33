"""Independent scalar and gradient checks for continuous positive holding."""
import math
import pytest
import torch
from viserdex.supermimic.core import manual_convex_config
from viserdex.supermimic.stability_reward import stability_reward


@pytest.mark.parametrize('theta,distance,omega,speed', [
    (0., 0., 0., 0.), (0., .005, 0., 0.),
    (math.pi/12, .01, 0., 0.), (0., .03, .5, .05),
    (math.pi/2, .01, 0., 0.), (.1, .02, .2, .02),
])
def test_scalar_value_and_gradient_direction(theta, distance, omega, speed):
    inputs = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for v in (theta, distance, omega, speed)]
    reward = stability_reward(*inputs, manual_convex_config())
    reward.backward()
    expected = 3*math.exp(-(theta/(math.pi/12))**2
                         -(max(distance-.01, 0)/.02)**2 -(omega/.5)**2 -(speed/.05)**2)
    assert reward.item() == pytest.approx(expected, rel=1e-12, abs=1e-30)
    for scalar, tensor in zip((theta, max(distance-.01, 0), omega, speed), inputs):
        assert torch.isfinite(tensor.grad)
        assert tensor.grad <= 0
        assert (tensor.grad == 0) if scalar == 0 else (tensor.grad < 0)


def test_no_far_from_goal_stop_incentive_and_positive_hold():
    cfg = manual_convex_config()
    zero = torch.tensor(0.)
    assert stability_reward(zero, zero, zero, zero, cfg) == 3
    stopped_far = stability_reward(torch.tensor(math.pi/2), zero, zero, zero, cfg)
    rotating_far = stability_reward(torch.tensor(math.pi/2), zero, torch.tensor(2.), zero, cfg)
    assert 0 <= stopped_far-rotating_far < 1e-14
    # Near the target, settling has a material positive benefit.
    assert stability_reward(zero, zero, zero, zero, cfg) - stability_reward(zero, zero, torch.tensor(.5), zero, cfg) > 1.8


@pytest.mark.parametrize('field', [
    'stability_angle_sigma_rad', 'stability_position_sigma_m',
    'stability_angular_speed_scale', 'stability_linear_speed_scale',
])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_width_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        manual_convex_config(**{field: value})
