"""Synthetic depth measurement noise; preserves optical rays and global RNG."""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class PointNoiseCfg:
    depth_std_m: float = 0.0005
    depth_clip_m: float = 0.0015

    def __post_init__(self):
        if not (math.isfinite(self.depth_std_m) and math.isfinite(self.depth_clip_m)
                and 0 < self.depth_std_m <= self.depth_clip_m):
            raise ValueError('Require finite 0 < depth_std_m <= depth_clip_m')

    def contract(self):
        return dict(**asdict(self), model='clipped_gaussian_optical_depth',
                    provenance='synthetic augmentation, not calibrated sensor noise or native ViserDex default',
                    projection='preserve X/Z and Y/Z; depth and GT remain clean',
                    input_key='points_camera_m', clean_key='points_camera_clean_m')


@torch.no_grad()
def add_point_noise(points, valid, cfg, seed):
    if points.shape[:-1] != valid.shape or points.shape[-1] != 3 or valid.dtype != torch.bool:
        raise ValueError('Expected [...,3] points and matching boolean validity')
    if not torch.isfinite(points).all() or (points[...,2][valid] <= 0).any():
        raise ValueError('Point measurements must be finite with positive valid depth')
    if (points[~valid] != 0).any():
        raise ValueError('Invalid points must be zero padded')
    generator = torch.Generator(device=points.device).manual_seed(seed)
    delta = (torch.randn(valid.shape, generator=generator, device=points.device, dtype=points.dtype)
             * cfg.depth_std_m).clamp(-cfg.depth_clip_m, cfg.depth_clip_m)
    delta = torch.where(valid, delta, 0.)
    clean_z = points[...,2]
    new_z = clean_z + delta
    if (new_z[valid] <= 0).any():
        raise ValueError('Noise would move a valid point behind the camera')
    scale = torch.where(valid, new_z / torch.where(valid, clean_z, 1.), 1.)
    noisy = points * scale[...,None]
    actual = noisy[...,2] - clean_z
    values = actual[valid]
    metadata = dict(seed=int(seed), valid_points=int(valid.sum()),
                    depth_std_observed_m=float(values.std(unbiased=False)) if len(values) else 0.,
                    depth_abs_max_m=float(values.abs().max()) if len(values) else 0.)
    return noisy, actual, metadata
