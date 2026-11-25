"""Build student observations from clean state; share measured joints with the teacher."""

import torch

from .config import PerceptionNoiseCfg, StudentCfg
from .frames import STUDENT_POSE_FRAME, canonical_pose
from .noise import (
    JointObservationNoise,
    PerceptionNoise,
    pose_features,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_unique,
)


class StudentObservations:
    def __init__(self, env, cfg: StudentCfg, noise_cfg: PerceptionNoiseCfg, pose_transform=None):
        self.env = env
        self.cfg = cfg
        self.pose_transform = pose_transform
        self.history = torch.zeros(env.num_envs, cfg.action_history, cfg.num_actions, device=env.device)
        self.noise = PerceptionNoise(env.num_envs, env.device, noise_cfg)
        self.joint_noise = JointObservationNoise(env.num_envs, cfg.num_actions, env.device, noise_cfg)

    def record_action(self, action, dones):
        # Most recent raw command first, matching the teacher's last_action convention.
        self.history = torch.cat((action[:, None], self.history[:, :-1]), dim=1)
        self.history[dones] = 0
        self.noise.reset(dones)
        self.joint_noise.reset(dones)

    def __call__(
        self, clean_obs, estimated_pose=None, *,
        estimated_pose_frame=STUDENT_POSE_FRAME, estimated_quaternion_order="xyzw",
    ):
        """Accept measured object (position in meters, quaternion) in a named frame.

        The default frame is right_hand_base_link, the existing student protocol.
        right_palm/camera_opencv require a CalibratedPoseTransform at construction.
        This path bypasses simulated pose noise. Ground truth remains a separate
        auxiliary target and is never used to convert an external measurement.
        """
        env = self.env
        if clean_obs["policy"].shape[-1] != 49 or clean_obs["privileged"].shape[-1] != 38:
            raise ValueError("Expected Revo3 Sim 6 observations: policy=49, privileged=38")
        robot, obj = env.scene["robot"], env.scene["object"]
        base_quat_inv = quat_conjugate(robot.data.root_quat_w.torch)
        base_pos = robot.data.root_pos_w.torch
        position = quat_apply(base_quat_inv, obj.data.root_pos_w.torch - base_pos)
        orientation = quat_unique(quat_mul(base_quat_inv, obj.data.root_quat_w.torch))
        goal = env.command_manager.get_command("object_pose")[:, 3:7]
        goal = quat_unique(quat_mul(base_quat_inv, goal))
        remaining = (1 - env.episode_length_buf.float() / env.max_episode_length).clamp(0, 1)[:, None]
        noisy_joints = self.joint_noise(clean_obs["policy"][:, :21])
        prop = torch.cat(
            (noisy_joints, self.history.flatten(1), goal, base_pos - env.scene.env_origins, remaining), dim=-1
        )
        if estimated_pose is None:
            noisy_pos, noisy_quat = self.noise(position, orientation)
        elif estimated_pose_frame == STUDENT_POSE_FRAME:
            noisy_pos, noisy_quat = canonical_pose(*estimated_pose, estimated_quaternion_order)
        elif self.pose_transform is not None:
            noisy_pos, noisy_quat = self.pose_transform.to_student(
                *estimated_pose, source_frame=estimated_pose_frame, quaternion_order=estimated_quaternion_order,
            )
        else:
            raise ValueError("A calibrated pose transform is required for external poses outside right_hand_base_link")
        extero = pose_features(noisy_pos, noisy_quat, goal)
        clean_extero = pose_features(position, orientation, goal)
        # Use the existing teacher's scaled velocity convention for auxiliary targets.
        privileged = torch.cat((clean_obs["privileged"][:, :21], clean_obs["privileged"][:, 28:34]), -1)
        target = torch.cat((clean_extero, privileged), -1)
        return prop, extero, target
