"""Calibrated full-frame RGB input, independent of object/hand simulator state."""

import torch
from torch.nn import functional as F


class HikrobotImageProcessor:
    """Rectify Brown-Conrady RGB and resample the entire frame in one operation.

    Input is native uint8 NHWC RGB(A). Output is float32 NCHW RGB in [0, 1].
    The 120x120 default preserves the complete calibrated field of view through
    separate x/y scaling, without a center/object crop. Pixel-center offsets
    agree with grid_sample(align_corners=False), including the resized K.
    """

    def __init__(self, intrinsics, device="cpu", image_size=(120, 120)):
        self.native_size = (intrinsics["width"], intrinsics["height"])
        self.image_size = tuple(image_size)
        if min(*self.native_size, *self.image_size) <= 0:
            raise ValueError("Image dimensions must be positive")
        if intrinsics["distortion_order"] != ["k1", "k2", "p1", "p2", "k3"]:
            raise ValueError("Expected OpenCV Brown-Conrady coefficient order")
        width, height = self.native_size
        out_width, out_height = self.image_size
        matrix = torch.tensor(intrinsics["camera_matrix"], dtype=torch.float64, device=device)
        scale_x, scale_y = out_width / width, out_height / height
        resize = matrix.new_tensor([
            [scale_x, 0, (scale_x - 1) / 2],
            [0, scale_y, (scale_y - 1) / 2],
            [0, 0, 1],
        ])
        self.intrinsics = resize @ matrix
        v, u = torch.meshgrid(
            torch.arange(out_height, dtype=matrix.dtype, device=device),
            torch.arange(out_width, dtype=matrix.dtype, device=device), indexing="ij",
        )
        x = (u - self.intrinsics[0, 2]) / self.intrinsics[0, 0]
        y = (v - self.intrinsics[1, 2]) / self.intrinsics[1, 1]
        k1, k2, p1, p2, k3 = intrinsics["dist_coeffs"]
        radius2 = x.square() + y.square()
        radial = 1 + k1 * radius2 + k2 * radius2.square() + k3 * radius2.pow(3)
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x.square())
        distorted_y = y * radial + p1 * (radius2 + 2 * y.square()) + 2 * p2 * x * y
        source_u = matrix[0, 0] * distorted_x + matrix[0, 2]
        source_v = matrix[1, 1] * distorted_y + matrix[1, 2]
        self.grid = torch.stack((2 * (source_u + 0.5) / width - 1, 2 * (source_v + 0.5) / height - 1), -1).float()[None]
        self.metadata = {
            "source": "sensor:hikrobot_camera:rgb",
            "native_image_size": list(self.native_size),
            "native_intrinsics": intrinsics["camera_matrix"],
            "native_distortion_order": intrinsics["distortion_order"],
            "native_distortion": intrinsics["dist_coeffs"],
            "cnn_image_size": list(self.image_size),
            "cnn_intrinsics": self.intrinsics.tolist(),
            "cnn_image_distortion": "rectified pinhole",
            "crop": None,
            "resampling": "full-frame bilinear; pixel centers; align_corners=False",
            "object_state_used": False,
            "hand_state_used": False,
        }

    def __call__(self, native_rgb):
        width, height = self.native_size
        if native_rgb.ndim != 4 or native_rgb.shape[1:3] != (height, width) or native_rgb.shape[-1] not in (3, 4):
            raise ValueError("Expected calibrated native NHWC RGB(A) image dimensions")
        if native_rgb.dtype != torch.uint8:
            raise ValueError("Expected native uint8 RGB; do not rescale twice")
        if native_rgb.device != self.grid.device:
            raise ValueError("Image processor and RGB must be on the same device")
        pixels = native_rgb[..., :3].permute(0, 3, 1, 2).float() / 255
        return F.grid_sample(
            pixels, self.grid.expand(len(pixels), -1, -1, -1),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )


def diagnostic_keypoints_uvd(reference, object_position, object_rotation, camera_position, camera_rotation, intrinsics, image_size):
    """Ground-truth overlay/metrics only; never call to construct a policy input."""
    points_world = reference.double() @ object_rotation.double().transpose(-1, -2) + object_position.double()[:, None]
    points_camera = (points_world - camera_position.double()[:, None]) @ camera_rotation.double()
    projected = points_camera @ intrinsics.double().T
    uv = projected[..., :2] / projected[..., 2:3]
    uv = uv / uv.new_tensor(image_size)
    return torch.cat((uv, points_camera[..., 2:3]), -1).flatten(1).float()
