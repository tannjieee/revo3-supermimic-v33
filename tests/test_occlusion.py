import torch

from viserdex.utils.occlusion import apply_hand_occlusion


def test_depth_ordering_background_and_missed_rays():
    rgb = torch.full((1, 3, 1, 6), 200, dtype=torch.uint8)
    obj = torch.tensor([[[[0.5, 0.5, 0.5, 0.0, 0.5, 0.5]]]])
    hand = torch.tensor([[[[0.4, 0.6, torch.inf, 0.1, 0.5, torch.nan]]]])
    out, hidden, fraction, visible = apply_hand_occlusion(rgb, obj, hand)
    assert hidden.flatten().tolist() == [True, False, False, False, False, False]
    assert out[0, :, 0, 0].count_nonzero() == 0
    torch.testing.assert_close(out[..., 1:], rgb[..., 1:])
    torch.testing.assert_close(fraction, torch.tensor([0.2]))
    assert visible.item() == 4


def test_empty_frame_is_finite_and_environment_masks_do_not_leak():
    rgb = torch.ones(2, 3, 2, 2, dtype=torch.uint8)
    obj = torch.ones(2, 1, 2, 2) * 0.5
    obj[0] = 0
    hand = torch.ones_like(obj) * 0.4
    out, hidden, fraction, visible = apply_hand_occlusion(rgb, obj, hand)
    assert fraction.tolist() == [0, 1]
    assert visible.tolist() == [0, 0]
    assert hidden[0].count_nonzero() == 0
    assert out[1].count_nonzero() == 0
