"""Camera-frame keypoint and rigid pose metrics, independent of the simulator."""

import torch


def unproject_uvd(uvd, intrinsics, image_size):
    points = uvd.reshape(-1, 9, 3)
    uv = points[..., :2] * points.new_tensor(image_size)
    rays = torch.cat((uv, torch.ones_like(points[..., :1])), dim=-1)
    return (rays @ torch.linalg.inv(intrinsics).T) * points[..., 2:3]


def fit_rigid_pose(reference, points):
    """Fit points = reference @ rotation.T + translation, enforcing det(R)=1."""
    center = reference.mean(dim=-2)
    target_center = points.mean(dim=-2)
    covariance = (reference - center).T @ (points - target_center.unsqueeze(-2))
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=points.device, dtype=points.dtype).expand(len(points), 3, 3).clone()
    correction[:, 2, 2] = torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    translation = target_center - (rotation @ center.unsqueeze(-1)).squeeze(-1)
    return rotation, translation


def pose_metrics(predictions, targets, intrinsics, reference, image_size=(120, 120)):
    # Low-angle rotation errors are sensitive to reduced-precision matmul:
    # TF32 can make R.T @ R non-orthogonal even for identical predictions.
    # Keep these tiny geometry matrices in float64 without changing CNN precision.
    output_dtype = predictions.dtype
    predictions, targets = predictions.double(), targets.double()
    intrinsics, reference = intrinsics.double(), reference.double()
    predicted = unproject_uvd(predictions, intrinsics, image_size)
    truth = unproject_uvd(targets, intrinsics, image_size)
    pred_r, pred_t = fit_rigid_pose(reference, predicted)
    gt_r, gt_t = fit_rigid_pose(reference, truth)
    cosine = ((pred_r.transpose(-1, -2) @ gt_r).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    translation = (pred_t - gt_t).norm(dim=-1) * 1000
    error = (predictions - targets).reshape(-1, 9, 3)
    pred_fit = reference @ pred_r.transpose(-1, -2) + pred_t.unsqueeze(-2)
    gt_fit = reference @ gt_r.transpose(-1, -2) + gt_t.unsqueeze(-2)
    result = {
        "keypoint_mm": (predicted - truth).norm(dim=-1).mean(-1) * 1000,
        "pixel_error": (error[..., :2] * error.new_tensor(image_size)).norm(dim=-1).mean(-1),
        "depth_mm": error[..., 2].abs().mean(-1) * 1000,
        "translation_mm": translation,
        "rotation_deg": angle,
        "add_mm": (pred_fit - gt_fit).norm(dim=-1).mean(-1) * 1000,
        "success_10mm_10deg": ((translation < 10) & (angle < 10)).float(),
    }
    return {name: value.to(output_dtype) for name, value in result.items()}
