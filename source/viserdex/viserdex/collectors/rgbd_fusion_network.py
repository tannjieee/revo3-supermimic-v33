"""Single-frame RGB/visible-point fusion with fixed-identity 3D keypoints.

Points are optical-camera meters. Pixel coordinates refer to the unwarped RGB.
The only centering anchor is the observed point centroid, never a truth pose.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34
from torchvision.models.feature_extraction import create_feature_extractor


class RGBDKeypointNetwork(nn.Module):
    def __init__(self, reference, rgb_checkpoint=None):
        super().__init__()
        # Preserve the exact historical feature tap; no simulator import/download.
        self.rgb_features = create_feature_extractor(
            resnet34(weights=None), return_nodes={"layer4.1.relu": "features"}
        )
        if rgb_checkpoint:
            state = torch.load(rgb_checkpoint, map_location="cpu", weights_only=True)
            prefix = "_feature_extractor."
            backbone = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            self.rgb_features.load_state_dict(backbone, strict=True)
        self.rgb_projection = nn.Conv1d(512, 128, 1)
        self.point_encoder = nn.Sequential(nn.Conv1d(3, 64, 1), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv1d(256, 256, 1), nn.ReLU(), nn.Conv1d(256, 256, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 27))
        reference = torch.as_tensor(reference, dtype=torch.float32)
        self.register_buffer("reference", reference)
        self.register_buffer("edge", (reference.max(0).values - reference.min(0).values).mean())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        nn.init.normal_(self.head[-1].weight, std=0.001)
        with torch.no_grad():
            self.head[-1].bias.copy_((reference / self.edge).flatten())

    def forward(self, rgb, points, uv, valid):
        image = self.rgb_features((rgb - self.mean) / self.std)["features"]
        height, width = rgb.shape[-2:]
        # Capture UV already uses continuous pixel centres (x+0.5,y+0.5).
        grid = uv / uv.new_tensor([width, height]) * 2 - 1
        local_rgb = F.grid_sample(image, grid[:, :, None], align_corners=False).squeeze(-1)
        local_rgb = self.rgb_projection(local_rgb)
        weights = valid[..., None].to(points.dtype)
        center = (points * weights).sum(1) / weights.sum(1).clamp_min(1)
        xyz = ((points - center[:, None]) / self.edge).transpose(1, 2)
        local_points = self.point_encoder(xyz)
        fused = self.fusion(torch.cat((local_rgb, local_points), 1))
        pooled = fused.masked_fill(~valid[:, None], -1e4).amax(-1)
        return center[:, None] + self.head(pooled).reshape(-1, 9, 3) * self.edge
