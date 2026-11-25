"""Shared joint sensor errors and student-only right_hand_base_link pose errors."""

import math

import torch

from .config import PerceptionNoiseCfg


def quat_unique(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., 3:] < 0, -q, q)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_mul(a, b):
    v = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((v, w), dim=-1)


def quat_apply(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def rotation_vector_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    xyz = 0.5 * torch.sinc(angle / (2 * math.pi)) * v
    return torch.cat((xyz, torch.cos(angle / 2)), dim=-1)


def pose_features(position, orientation, goal_orientation):
    """Orientation difference is recomputed from the *provided* pose, never ground truth."""
    orientation = quat_unique(orientation)
    diff = quat_unique(quat_mul(orientation, quat_conjugate(goal_orientation)))
    return torch.cat((position, orientation, diff), dim=-1)


class JointObservationNoise:
    """Encoder noise, episode calibration bias and a bounded observation delay.

    Units are normalized joint positions, not radians. Never modify the clean
    tensor supplied by the teacher's observation manager in place.
    """

    def __init__(self, num_envs, num_joints, device, cfg: PerceptionNoiseCfg):
        self.cfg = cfg
        self.bias = torch.zeros(num_envs, num_joints, device=device)
        self.history = torch.zeros(num_envs, cfg.joint_delay_max_steps + 1, num_joints, device=device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.env_ids = torch.arange(num_envs, device=device)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, dones):
        ids = dones.nonzero(as_tuple=False).flatten()
        self.fresh[ids] = True
        if self.cfg.joint_enabled:
            self.bias[ids] = (
                torch.randn(ids.numel(), self.bias.shape[1], device=self.bias.device) * self.cfg.joint_bias_std
            )
            self.delay[ids] = torch.randint(self.cfg.joint_delay_max_steps + 1, (ids.numel(),), device=self.bias.device)

    def __call__(self, clean_joints):
        if not self.cfg.joint_enabled:
            return clean_joints.clone()
        measured = clean_joints + self.bias + torch.randn_like(clean_joints) * self.cfg.joint_noise_std
        self.history = torch.cat((measured[:, None], self.history[:, :-1]), dim=1)
        # First observation after a reset cannot refer to the previous episode.
        self.history[self.fresh] = measured[self.fresh, None]
        self.fresh.zero_()
        return self.history[self.env_ids, self.delay].clone()


class PerceptionNoise:
    def __init__(self, num_envs, device, cfg: PerceptionNoiseCfg):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.period = torch.ones(num_envs, 2, dtype=torch.long, device=device)
        self.jitter = torch.zeros(num_envs, 2, device=device)
        self.failure = torch.zeros(num_envs, 2, device=device)
        self.bias = torch.zeros(num_envs, 6, device=device)
        self.held_pos = torch.zeros(num_envs, 3, device=device)
        self.held_quat = torch.zeros(num_envs, 4, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.reset(self.fresh.clone())

    def reset(self, dones):
        ids = dones.nonzero(as_tuple=False).flatten()
        n = ids.numel()
        self.age[ids] = 0
        self.fresh[ids] = True
        self.period[ids] = torch.randint(1, self.cfg.max_period + 1, (n, 2), device=self.device)
        self.jitter[ids] = torch.rand(n, 2, device=self.device) * self.cfg.jitter_probability
        self.failure[ids] = torch.rand(n, 2, device=self.device) * self.cfg.failure_probability
        self.bias[ids, :3] = (2 * torch.rand(n, 3, device=self.device) - 1) * self.cfg.position_bias
        self.bias[ids, 3:] = (2 * torch.rand(n, 3, device=self.device) - 1) * math.radians(self.cfg.rotation_bias_deg)

    def __call__(self, position, orientation):
        if not self.cfg.enabled:
            return position.clone(), quat_unique(orientation)
        n = self.num_envs
        pos = position + self.bias[:, :3] + (2 * torch.rand_like(position) - 1) * self.cfg.position_noise
        rotvec = self.bias[:, 3:] + (2 * torch.rand_like(position) - 1) * math.radians(self.cfg.rotation_noise_deg)
        quat = quat_unique(quat_mul(rotation_vector_quat(rotvec), orientation))
        failed = torch.rand(n, 2, device=self.device) < self.failure
        # A failure cannot reuse the true pose (including its true orientation).
        replacement_pos = (2 * torch.rand_like(position) - 1) * self.cfg.failure_position_extent
        replacement_quat = quat_unique(torch.randn_like(orientation))
        pos = torch.where(failed[:, :1], replacement_pos, pos)
        quat = torch.where(failed[:, 1:], replacement_quat, quat)
        update = ((self.age[:, None] % self.period) == 0) & (torch.rand(n, 2, device=self.device) >= self.jitter)
        update |= self.fresh[:, None]
        self.held_pos = torch.where(update[:, :1], pos, self.held_pos)
        self.held_quat = torch.where(update[:, 1:], quat, self.held_quat)
        self.age += 1
        self.fresh.zero_()
        return self.held_pos.clone(), self.held_quat.clone()
