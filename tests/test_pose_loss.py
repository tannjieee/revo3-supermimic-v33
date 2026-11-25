"""Every keypoint's pixel coordinates and depth must get the intended weight."""

import pytest
import torch

from viserdex.utils.pose_loss import keypoint_loss


@pytest.mark.parametrize("keypoint", range(9))
@pytest.mark.parametrize("coordinate,expected", [(0, 100 / 18), (1, 100 / 18), (2, 200 / 9)])
def test_weight_is_independent_of_keypoint_order(keypoint, coordinate, expected):
    prediction = torch.zeros(1, 27, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, keypoint * 3 + coordinate] = 1
    loss = keypoint_loss(prediction, target, torch.nn.L1Loss(), depth_loss_coef=2)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert prediction.grad[0, keypoint * 3 + coordinate].abs().item() == pytest.approx(expected)
    assert prediction.grad.count_nonzero() == 1
