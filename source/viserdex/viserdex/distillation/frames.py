"""Explicit student pose frames and calibrated camera/URDF-palm conversion.

The existing student uses the articulation root, right_hand_base_link. Keep that
learned convention; convert external measurements before building pose features.
T_A_B maps coordinates in B into A. Positions are meters, poses describe objects
relative to the named source frame, and this module never reads simulator state.
"""

import hashlib
import json
from pathlib import Path

import torch

from .noise import quat_apply, quat_mul, quat_unique

STUDENT_POSE_FRAME = "right_hand_base_link"
PALM_FRAME = "right_palm"
CAMERA_FRAME = "camera_opencv"
POSE_PROTOCOL = "revo3_base_pose_xyzw_m_v1"


def pose_frame_metadata():
    return {
        "pose_frame": STUDENT_POSE_FRAME,
        "pose_protocol": POSE_PROTOCOL,
        "position_unit": "m",
        "quaternion_order": "xyzw",
    }


def validate_pose_metadata(metadata):
    """Resolve this implementation's historical 'palm' label without moving data.

    Its v1 checkpoints used the root all along. Accept that known legacy label
    only when no explicit protocol was saved; reject incompatible new metadata.
    """
    frame = metadata.get("pose_frame")
    protocol = metadata.get("pose_protocol")
    legacy = frame == "palm" and protocol is None
    if not (frame == STUDENT_POSE_FRAME or legacy):
        raise ValueError(f"Student pose frame must be {STUDENT_POSE_FRAME}, got {frame!r}")
    if protocol not in (None, POSE_PROTOCOL):
        raise ValueError(f"Unsupported student pose protocol: {protocol!r}")
    if metadata.get("quaternion_order", "xyzw") != "xyzw" or metadata.get("position_unit", "m") != "m":
        raise ValueError("Student pose protocol requires meters and XYZW quaternions")
    result = pose_frame_metadata()
    if legacy:
        result["legacy_pose_frame_label"] = "palm (historically articulation root; no numeric conversion)"
    return result


def canonical_pose(position, quaternion, quaternion_order="xyzw"):
    if position.shape[-1] != 3 or quaternion.shape != (*position.shape[:-1], 4):
        raise ValueError("Expected matching [..., 3] positions and [..., 4] quaternions")
    if not position.is_floating_point() or not quaternion.is_floating_point():
        raise ValueError("Pose tensors must be floating point")
    if position.device != quaternion.device or position.dtype != quaternion.dtype:
        raise ValueError("Pose tensors must share device and dtype")
    if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all():
        raise ValueError("Pose contains non-finite values")
    if (quaternion.norm(dim=-1) < 1e-8).any():
        raise ValueError("Pose quaternion cannot have zero norm")
    if quaternion_order == "wxyz":
        quaternion = quaternion[..., [1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quaternion_order!r}")
    return position, quat_unique(quaternion)


class CalibratedPoseTransform:
    """Convert measured object poses using the delivered calibration, once only.

    camera_opencv is the raw optical frame: X right, Y down, Z forward. Do not
    supply a graphics-camera pose or apply an additional 180-degree rotation.
    """

    def __init__(self, extrinsics_path):
        path = Path(extrinsics_path)
        content = path.read_bytes()
        data = json.loads(content)
        if data["frames"]["base"] != STUDENT_POSE_FRAME or data["frames"]["palm"] != PALM_FRAME:
            raise ValueError("Calibration base/palm frames do not match the Revo3 student")
        self.transforms = {}
        for frame, name in ((PALM_FRAME, "T_base_palm"), (CAMERA_FRAME, "T_base_camera")):
            record = data["transforms"][name]
            matrix = torch.tensor(record["matrix"], dtype=torch.float64)
            if matrix.shape != (4, 4) or not torch.isfinite(matrix).all():
                raise ValueError(f"Invalid calibration transform: {name}")
            torch.testing.assert_close(matrix[3], matrix.new_tensor([0, 0, 0, 1]), atol=1e-10, rtol=0)
            rotation = matrix[:3, :3]
            torch.testing.assert_close(rotation.T @ rotation, torch.eye(3, dtype=matrix.dtype), atol=1e-10, rtol=0)
            torch.testing.assert_close(torch.linalg.det(rotation), matrix.new_tensor(1), atol=1e-10, rtol=0)
            _, quaternion = canonical_pose(matrix[:3, 3], matrix.new_tensor(record["quaternion_wxyz"]), "wxyz")
            # Compare the redundant quaternion to the matrix, preventing order mistakes.
            axes = torch.eye(3, dtype=matrix.dtype)
            reconstructed = quat_apply(quaternion.expand(3, 4), axes).T
            torch.testing.assert_close(reconstructed, rotation, atol=1e-10, rtol=0)
            self.transforms[frame] = (matrix, quaternion)
        palm_camera = torch.tensor(data["transforms"]["T_palm_camera"]["matrix"], dtype=torch.float64)
        torch.testing.assert_close(
            self.transforms[PALM_FRAME][0] @ palm_camera,
            self.transforms[CAMERA_FRAME][0], atol=1e-10, rtol=0,
        )
        self.metadata = {
            **pose_frame_metadata(),
            "calibration_path": str(path.resolve()),
            "calibration_sha256": hashlib.sha256(content).hexdigest(),
            "camera_frame": CAMERA_FRAME,
            "camera_convention": "OpenCV X right, Y down, Z forward",
        }

    def to_student(self, position, quaternion, *, source_frame, quaternion_order="xyzw"):
        """Return object (position, XYZW quaternion) in right_hand_base_link."""
        position, quaternion = canonical_pose(position, quaternion, quaternion_order)
        if source_frame == STUDENT_POSE_FRAME:
            return position, quaternion
        if source_frame not in self.transforms:
            raise ValueError(f"Unknown pose source frame: {source_frame!r}")
        matrix, rotation = self.transforms[source_frame]
        # Float64 geometry avoids TF32 effects, then returns the estimator dtype.
        rotation = rotation.to(position.device).expand(*position.shape[:-1], 4)
        translation = matrix[:3, 3].to(position.device)
        converted_position = quat_apply(rotation, position.double()) + translation
        converted_quaternion = quat_unique(quat_mul(rotation, quaternion.double()))
        return converted_position.to(position.dtype), converted_quaternion.to(quaternion.dtype)
