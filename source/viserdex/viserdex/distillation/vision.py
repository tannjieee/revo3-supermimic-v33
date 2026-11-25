"""Decode RGB keypoints into an explicitly supplied target frame."""

from viserdex.utils.pose_metrics import fit_rigid_pose, unproject_uvd


def keypoints_to_frame_pose(
    uvd,
    intrinsics,
    reference,
    camera_rotation_w,
    camera_position_w,
    frame_rotation_w,
    frame_position_w,
    image_size=(120, 120),
):
    """Return (rotation, translation) in the supplied frame, without object truth.

    Rotations map local coordinates to world coordinates. Positions must both be
    absolute world positions, including the same environment origin. All small
    geometric operations use float64 to avoid TF32 rotation errors.
    For the current student, supply right_hand_base_link's world pose, not the
    offset URDF right_palm pose. This pinhole UVD decoder expects undistorted UV;
    raw Hikrobot Brown-Conrady pixels must be undistorted by the estimator first.
    """
    points = unproject_uvd(uvd.double(), intrinsics.double(), image_size)
    rotation_cam, position_cam = fit_rigid_pose(reference.double(), points)
    world_to_frame = frame_rotation_w.double().transpose(-1, -2)
    camera_to_frame = world_to_frame @ camera_rotation_w.double()
    camera_position_frame = (
        world_to_frame @ (camera_position_w.double() - frame_position_w.double()).unsqueeze(-1)
    ).squeeze(-1)
    rotation = camera_to_frame @ rotation_cam
    position = (camera_to_frame @ position_cam.unsqueeze(-1)).squeeze(-1) + camera_position_frame
    return rotation.to(uvd.dtype), position.to(uvd.dtype)


def keypoints_to_palm_pose(
    uvd, intrinsics, reference, camera_rotation_w, camera_position_w,
    palm_rotation_w, palm_position_w, image_size=(120, 120),
):
    """Legacy API: its supplied 'palm' pose defines the actual target frame.

    Current callers use keypoints_to_frame_pose and explicitly supply the base.
    Keep the old keyword arguments working for existing standalone consumers.
    """
    return keypoints_to_frame_pose(
        uvd, intrinsics, reference, camera_rotation_w, camera_position_w,
        palm_rotation_w, palm_position_w, image_size,
    )
