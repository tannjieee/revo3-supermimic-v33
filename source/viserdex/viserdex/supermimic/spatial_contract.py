"""One translated, axis-aligned hand frame for network positions and drop tests.

Simulator/replay states retain SI base coordinates. All learned inputs subtract
this fixed origin; deltas and velocities are invariant to the translation.
"""
import torch


def centered_position(position_base, origin):
    return position_base - position_base.new_tensor(origin)


def drop_distance(physical, cfg, *, origin=None):
    """Legacy metres, or radius*rho for the COM ellipsoid.

    Retaining this scale makes every existing ratio, hard gate and analytic
    calibration use the same surface. It is NOT Euclidean distance in ellipsoid
    mode. Network input centering remains independent of this boundary.
    """
    if cfg.drop_boundary_mode == 'com_ellipsoid':
        com = object_com_position(physical[...,42:45], physical[...,45:49], cfg.drop_object_com_local_m)
        return ellipsoid_rho(com, cfg.drop_ellipsoid_center_m, cfg.drop_ellipsoid_semi_axes_m) * cfg.drop_radius_m
    return hand_center_distance(physical, cfg, origin=origin)


def hand_center_distance(physical, cfg, *, origin=None):
    """Actual root-to-input-origin distance in metres, for legacy metrics."""
    centered = (centered_position(physical[...,42:45], cfg.hand_center_base_m) if origin is None
                else physical[...,42:45] - origin)
    return centered.norm(dim=-1)


def object_com_position(root_position, xyzw, local_com):
    # Scalar constants avoid CPU-to-device tensor copies inside CUDA graphs.
    q = torch.nn.functional.normalize(xyzw, dim=-1)
    x, y, z, w = q.unbind(-1)
    a, b, c = local_com
    tx, ty, tz = 2*(y*c-z*b), 2*(z*a-x*c), 2*(x*b-y*a)
    offset = torch.stack((a+w*tx+y*tz-z*ty, b+w*ty+z*tx-x*tz, c+w*tz+x*ty-y*tx), -1)
    return root_position + offset


def ellipsoid_rho(position, center, semi_axes):
    return torch.stack(tuple((position[...,j]-center[j])/semi_axes[j] for j in range(3)), -1).norm(dim=-1)


def object_outside_com_ellipsoid(env, center, semi_axes, object_com_local, boundary_sha256=''):
    """Real callback: pinned Isaac Lab v3/Newton and policy both use XYZW."""
    from viserdex.distillation.noise import quat_apply, quat_conjugate, quat_mul
    def tensor(x):
        return x if isinstance(x, torch.Tensor) else x.torch
    robot, obj = env.scene['robot'].data, env.scene['object'].data
    base_q = tensor(robot.root_quat_w)
    object_q = tensor(obj.root_quat_w)
    inverse = quat_conjugate(base_q)
    root_base = quat_apply(inverse, tensor(obj.root_pos_w)-tensor(robot.root_pos_w))
    com = object_com_position(root_base, quat_mul(inverse, object_q), object_com_local)
    return ellipsoid_rho(com, center, semi_axes) > 1.


def object_away_from_hand_center(env, origin, threshold):
    from isaaclab.utils.math import quat_apply_inverse
    robot, obj = env.scene['robot'], env.scene['object']
    p = quat_apply_inverse(robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch - robot.data.root_pos_w.torch)
    return centered_position(p, origin).norm(dim=-1) > threshold
