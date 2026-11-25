"""Independent scalar/gradient oracles and SO(3) reward invariants."""
import math
import pytest
import torch
from viserdex.supermimic.core import manual_convex_config, rotation_log
from viserdex.supermimic.orientation_reward import orientation_reward
from viserdex.distillation.noise import rotation_vector_quat


@pytest.mark.parametrize('degrees', [0., 5., 10., 30., 90., 180.])
def test_gaussian_reward_and_gradient_match_independent_scalar_oracle(degrees):
    cfg = manual_convex_config()
    theta = math.radians(degrees)
    angle = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    value = orientation_reward(angle, cfg)
    grad, = torch.autograd.grad(value, angle)
    expected = 10*(.3*math.exp(-(degrees/90)**2) + .7*math.exp(-(degrees/10)**2))
    slope = sum(-2*theta/(sigma*sigma)*weight*math.exp(-(theta/sigma)**2)
                for sigma, weight in [(math.pi/2, 3.), (math.pi/18, 7.)])
    assert value.item() == pytest.approx(expected, abs=1e-12)
    assert grad.item() == pytest.approx(slope, abs=1e-12)


def test_monotonic_bounded_and_large_angle_has_signal():
    cfg = manual_convex_config()
    angles = torch.linspace(0, math.pi, 10001, dtype=torch.float64, requires_grad=True)
    values = orientation_reward(angles, cfg)
    assert values[0] == 10 and torch.all(values > 0) and torch.all(values <= 10)
    assert torch.all(values[1:] < values[:-1])
    gradients, = torch.autograd.grad(values.sum(), angles)
    assert gradients[0] == 0 and (gradients[1:] < 0).all()
    # Single 10-degree Gaussian loses its useful signal at 90 degrees.
    assert 10*math.exp(-(90/10)**2) < 1e-30
    assert values[5000] > 1 and abs(gradients[-1]) > .05


@pytest.mark.parametrize('vector', [[0., 0., 0.], [1e-6, 0., 0.], [.2, -.4, .7], [math.pi-1e-5, 0., 0.]])
def test_so3_backprop_finite_sign_invariant_and_gradcheck(vector):
    cfg = manual_convex_config()
    v = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
    def function(rotvec):
        return orientation_reward(rotation_log(rotation_vector_quat(rotvec)).norm(dim=-1), cfg)
    assert torch.autograd.gradcheck(function, (v,), eps=1e-7, atol=1e-5, rtol=1e-4)
    q = rotation_vector_quat(v)
    torch.testing.assert_close(orientation_reward(rotation_log(q).norm(dim=-1), cfg),
                               orientation_reward(rotation_log(-q).norm(dim=-1), cfg))


@pytest.mark.parametrize('kwargs', [
    {'orientation_coarse_sigma_rad': 0}, {'orientation_fine_sigma_rad': float('nan')},
    {'orientation_reward_scale': -1}, {'orientation_reward_scale': float('inf')},
    {'orientation_coarse_fraction': 1.1}, {'orientation_coarse_fraction': float('nan')},
    {'orientation_fine_sigma_rad': 2.},
])
def test_invalid_kernel_rejected(kwargs):
    with pytest.raises(ValueError, match='orientation'):
        manual_convex_config(**kwargs)
