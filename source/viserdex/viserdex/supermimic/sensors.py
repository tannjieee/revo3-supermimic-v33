"""Functional version of the existing student sensor process, suitable for BPTT.

Sensor state is persisted in replay, including held poses and encoder delay.
No privileged measurement enters the actor except through this observation path.
"""

import math
import torch
from viserdex.distillation.config import PerceptionNoiseCfg
from .core import Q, QD, POS, ROT, ACTIONS, GOAL_POS, GOAL_ROT, TIME, BASE_POS
from .state_layout import layout
from .policy_contract import DIRECT_OBSERVATION_PROTOCOL, direct_observation
from viserdex.distillation.noise import (
    quat_unique,
    quat_mul,
    rotation_vector_quat,
    pose_features,
)


def initial_sensors(n, device, cfg):
    def zeros(*shape):
        return torch.zeros(n, *shape, device=device)

    return dict(
        joint_bias=zeros(21),
        joint_history=zeros(cfg.joint_delay_max_steps + 1, 21),
        joint_velocity_history=zeros(cfg.joint_delay_max_steps + 1, 21),
        delay=torch.zeros(n, device=device, dtype=torch.long),
        joint_fresh=torch.ones(n, device=device, dtype=torch.bool),
        age=torch.zeros(n, device=device, dtype=torch.long),
        period=torch.ones(n, 2, device=device, dtype=torch.long),
        jitter=zeros(2),
        failure=zeros(2),
        bias=zeros(6),
        held_pos=zeros(3),
        held_quat=zeros(4),
        fresh=torch.ones(n, device=device, dtype=torch.bool),
    )


def reset_sensors(sensor, mask, cfg):
    d = {k: v.clone() for k, v in sensor.items()}
    ids = mask.nonzero().flatten()
    n = ids.numel()
    dev = mask.device
    d["joint_fresh"][ids] = True
    if cfg.joint_enabled:
        d["joint_bias"][ids] = torch.randn(n, 21, device=dev) * cfg.joint_bias_std
        d["delay"][ids] = torch.randint(cfg.joint_delay_max_steps + 1, (n,), device=dev)
    d["age"][ids] = 0
    d["fresh"][ids] = True
    d["period"][ids] = torch.randint(1, cfg.max_period + 1, (n, 2), device=dev)
    d["jitter"][ids] = torch.rand(n, 2, device=dev) * cfg.jitter_probability
    d["failure"][ids] = torch.rand(n, 2, device=dev) * cfg.failure_probability
    d["bias"][ids, :3] = (2 * torch.rand(n, 3, device=dev) - 1) * cfg.position_bias
    d["bias"][ids, 3:] = (2 * torch.rand(n, 3, device=dev) - 1) * math.radians(cfg.rotation_bias_deg)
    return d


def observe(s, sensor, limits, cfg, policy_cfg=None):
    d = dict(sensor)  # All changed tensors are replaced, never mutated.
    lower, upper = limits[..., 0], limits[..., 1]
    bias = s.new_tensor(policy_cfg.encoder_bias_rad) if policy_cfg is not None and policy_cfg.encoder_bias_rad else 0.
    if policy_cfg is not None and policy_cfg.dex_reference:
        from .dex_reference import ZERO
        bias = -s[:, ZERO]
    joints = 2 * (s[:, Q] - bias - lower) / (upper - lower) - 1
    velocity = s[:, QD]
    if cfg.joint_enabled:
        measured = joints + d["joint_bias"] + torch.randn_like(joints) * cfg.joint_noise_std
        if cfg.joint_delay_max_steps == 0:
            # Exact zero-delay fast path: avoids arange allocation, cat, and 2D fancy indexing.
            d["joint_history"] = measured[:, None]
            d["joint_velocity_history"] = velocity[:, None]
            joints = measured
            d["joint_fresh"] = torch.zeros_like(d["joint_fresh"])
        else:
            history = torch.cat((measured[:, None], d["joint_history"][:, :-1]), 1)
            d["joint_history"] = torch.where(d["joint_fresh"][:, None, None], measured[:, None], history)
            velocity_history = torch.cat((velocity[:, None], d["joint_velocity_history"][:, :-1]), 1)
            d["joint_velocity_history"] = torch.where(d["joint_fresh"][:, None, None], velocity[:, None], velocity_history)
            velocity = d["joint_velocity_history"][torch.arange(len(s), device=s.device), d["delay"]]
            joints = d["joint_history"][torch.arange(len(s), device=s.device), d["delay"]]
            d["joint_fresh"] = torch.zeros_like(d["joint_fresh"])
    position, orientation = s[:, POS], s[:, ROT]
    if cfg.enabled:
        pos = position + d["bias"][:, :3] + (2 * torch.rand_like(position) - 1) * cfg.position_noise
        rv = d["bias"][:, 3:] + (2 * torch.rand_like(position) - 1) * math.radians(cfg.rotation_noise_deg)
        quat = quat_unique(quat_mul(rotation_vector_quat(rv), orientation))
        failed = torch.rand(len(s), 2, device=s.device) < d["failure"]
        replacement_pos = (2 * torch.rand_like(position) - 1) * cfg.failure_position_extent
        replacement_quat = quat_unique(torch.randn_like(orientation))
        pos = torch.where(failed[:, :1], replacement_pos, pos)
        quat = torch.where(failed[:, 1:], replacement_quat, quat)
        update = ((d["age"][:, None] % d["period"]) == 0) & (torch.rand(len(s), 2, device=s.device) >= d["jitter"])
        update = update | d["fresh"][:, None]
        d["held_pos"] = torch.where(update[:, :1], pos, d["held_pos"])
        d["held_quat"] = torch.where(update[:, 1:], quat, d["held_quat"])
        d["age"] = d["age"] + 1
        d["fresh"] = torch.zeros_like(d["fresh"])
        position, orientation = d["held_pos"], d["held_quat"]
    if policy_cfg is not None and policy_cfg.dex_reference:
        from .dex_reference import observation
        return observation(joints, s, limits, position, orientation), d
    if policy_cfg is not None and policy_cfg.observation_protocol == DIRECT_OBSERVATION_PROTOCOL:
        return direct_observation(joints, velocity, s[:, layout(s).last_action], s[:, layout(s).goal_rot],
                                  position, orientation, s[:, layout(s).goal_pos],
                                  velocity_scale=policy_cfg.joint_velocity_scale,
                                  include_object_position=policy_cfg.include_object_position), d
    obs = torch.cat(
        (
            joints,
            s[:, ACTIONS],
            quat_unique(s[:, GOAL_ROT]),
            s[:, BASE_POS],
            s[:, TIME],
            pose_features(position, orientation, s[:, GOAL_ROT]),
        ),
        -1,
    )
    return obs, d
