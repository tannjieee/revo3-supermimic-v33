"""Depth composition from ViserDex III-C2, independent of the simulator."""

import torch


def apply_hand_occlusion(rgb, object_depth, hand_depth):
    """Mask a CHW batch using optical-axis metric depths from the same camera.

    Missed hand rays are +inf. Zero/invalid object depth is background.
    A finger behind the object must not erase the object.
    """
    if object_depth.shape != hand_depth.shape or object_depth.shape[1] != 1:
        raise ValueError("Aligned object/hand depth must both have shape (N, 1, H, W)")
    foreground = torch.isfinite(object_depth) & (object_depth > 0) & (object_depth < 1.0)
    foreground &= rgb.any(dim=1, keepdim=True).bool()
    hidden = foreground & torch.isfinite(hand_depth) & (hand_depth > 0) & (hand_depth < object_depth)
    output = rgb.masked_fill(hidden.expand_as(rgb), 0)
    foreground_pixels = foreground.flatten(1).sum(1)
    hidden_pixels = hidden.flatten(1).sum(1)
    fraction = hidden_pixels / foreground_pixels.clamp_min(1)
    return output, hidden, fraction, foreground_pixels - hidden_pixels
