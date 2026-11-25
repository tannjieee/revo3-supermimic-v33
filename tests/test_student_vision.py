import torch

from viserdex.distillation.vision import keypoints_to_palm_pose


def test_keypoint_pose_to_palm_with_rotated_camera_palm_and_translated_environments():
    # Non-identity frames and large environment offsets expose inverse/order and
    # local-vs-world mistakes that an origin-aligned camera would conceal.
    dtype = torch.float64
    reference = torch.cat((torch.zeros(1, 3), torch.cartesian_prod(*[torch.tensor([-0.02, 0.02])] * 3))).to(dtype)
    object_camera = torch.tensor([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]], dtype=dtype)
    translation_camera = torch.tensor([0.01, -0.03, 0.4], dtype=dtype)
    points = reference @ object_camera.T + translation_camera
    intrinsics = torch.tensor([[246.0, 0.0, 40.0], [0.0, 246.0, 24.0], [0.0, 0.0, 1.0]], dtype=dtype)
    pixels = points @ intrinsics.T
    uvd = torch.cat((pixels[:, :2] / pixels[:, 2:] / 120, points[:, 2:]), -1).reshape(1, 27).repeat(2, 1)
    camera_rotation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=dtype).repeat(2, 1, 1)
    palm_rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype).repeat(2, 1, 1)
    offsets = torch.tensor([[0.0, 0.0, 0.0], [32.0, -18.0, 4.0]], dtype=dtype)
    camera_position = offsets + torch.tensor([-0.15, 0.03, 0.88], dtype=dtype)
    palm_position = offsets + torch.tensor([0.02, 0.01, 0.5], dtype=dtype)
    rotation, position = keypoints_to_palm_pose(
        uvd, intrinsics, reference, camera_rotation, camera_position, palm_rotation, palm_position
    )
    expected_rotation = palm_rotation.transpose(-1, -2) @ camera_rotation @ object_camera
    object_world = (camera_rotation @ translation_camera[:, None]).squeeze(-1) + camera_position
    expected_position = (palm_rotation.transpose(-1, -2) @ (object_world - palm_position)[..., None]).squeeze(-1)
    torch.testing.assert_close(rotation, expected_rotation)
    torch.testing.assert_close(position, expected_position)
    torch.testing.assert_close(rotation[0], rotation[1])
    torch.testing.assert_close(position[0], position[1])
