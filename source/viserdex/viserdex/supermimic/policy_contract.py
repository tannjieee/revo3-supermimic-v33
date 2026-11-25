"""Simulator-independent observation and action contract for the manual-convex hand.

Isaac Lab 3.0 / Newton / SciPy quaternions are XYZW. Rotation features store
the first two *columns* of the active rotation matrix, column by column.
"""

import torch
from viserdex.distillation.noise import quat_conjugate, quat_mul

LEGACY_OBSERVATION_PROTOCOL = "revo3_student124_history10_xyzw_base_v1"
DIRECT_OBSERVATION_PROTOCOL = "revo3_manual_convex_single_pose_encoder_rot6d_no_clock_v5"


def rotation_6d_xyzw(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w),
        2 * (x*y - z*w), 1 - 2 * (x*x + z*z), 2 * (y*z + x*w),
    ), -1)


def direct_observation(joints_normalized, joint_velocity, last_action, goal_xyzw,
                       object_position, object_xyzw, goal_position, *,
                       velocity_scale=0.1, include_object_position=True):
    pieces = [joints_normalized, joint_velocity * velocity_scale, last_action]
    if include_object_position:
        pieces.append(object_position)
    pieces.append(goal_position)
    relative = quat_mul(object_xyzw, quat_conjugate(goal_xyzw))
    pieces.extend((rotation_6d_xyzw(object_xyzw), rotation_6d_xyzw(relative)))
    return torch.cat(pieces, -1)


def absolute_joint_target(action, limits):
    """One affine decode AFTER Actor.tanh(); no clamp, EMA, offset or warmup."""
    lower, upper = limits[..., 0], limits[..., 1]
    return lower + (action + 1) * 0.5 * (upper - lower)


def observation_fields(include_object_position=True):
    fields = [('joint_position_normalized', 21), ('joint_velocity_scaled', 21),
              ('last_action_after_tanh', 21)]
    if include_object_position:
        fields.append(('object_position_base_m', 3))
    fields.append(('goal_position_base_m', 3))
    fields.extend([('object_rotation_6d', 6), ('relative_rotation_6d', 6)])
    result, offset = [], 0
    for name, width in fields:
        result.append(dict(name=name, start=offset, stop=offset + width))
        offset += width
    return result
