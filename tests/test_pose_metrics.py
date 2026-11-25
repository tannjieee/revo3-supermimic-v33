import torch
import pytest
from viserdex.utils.pose_metrics import fit_rigid_pose, pose_metrics


def test_rigid_fit_recovers_rotation_translation():
    reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([0.1, -0.2, 0.5])
    found_r, found_t = fit_rigid_pose(reference, (reference @ rotation.T + translation).unsqueeze(0))
    torch.testing.assert_close(found_r[0], rotation, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(found_t[0], translation, atol=1e-6, rtol=1e-6)


def test_exact_projection_has_zero_metric_error():
    corners = torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)
    reference = torch.cat((torch.zeros(1, 3), corners))
    xyz = reference + torch.tensor([0.01, -0.01, 0.4])
    k = torch.tensor([[250.0, 0.0, 60.0], [0.0, 250.0, 60.0], [0.0, 0.0, 1.0]])
    pixels = xyz @ k.T
    uvd = torch.cat((pixels[:, :2] / pixels[:, 2:] / 120, xyz[:, 2:]), dim=-1).reshape(1, 27)
    metrics = pose_metrics(uvd, uvd, k, reference)
    for name in ("keypoint_mm", "pixel_error", "depth_mm", "translation_mm", "add_mm"):
        torch.testing.assert_close(metrics[name], torch.zeros(1), atol=1e-4, rtol=0)
    assert metrics["rotation_deg"].item() < 0.1
    assert metrics["success_10mm_10deg"].item() == 1


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TF32 regression")
        ),
    ],
)
def test_identical_arbitrary_rotation_has_zero_error_with_tf32_enabled(device):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        reference = torch.cat((torch.zeros(1, 3), torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3)))
        rotation = torch.tensor([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
        xyz = reference @ rotation.T + torch.tensor([0.01, -0.02, 0.4])
        k = torch.tensor([[250.0, 0.0, 60.0], [0.0, 250.0, 60.0], [0.0, 0.0, 1.0]])
        pixels = xyz @ k.T
        uvd = torch.cat((pixels[:, :2] / pixels[:, 2:] / 120, xyz[:, 2:]), -1).reshape(1, 27).to(device)
        metrics = pose_metrics(uvd, uvd, k.to(device), reference.to(device))
        assert metrics["rotation_deg"].item() < 1e-4
        assert metrics["add_mm"].item() == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
