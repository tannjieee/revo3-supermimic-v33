"""One matched HOME/die pose, written directly before any physics step.

The user explicitly selected fixed initial conditions, with randomized rotation
goals and startup physical domain parameters retained. No online pose search,
settling, grasp library, hidden support force or change of commanded action.
"""
from pathlib import Path
import numpy as np
from .wuji_pose import HOME_POSITION

PROTOCOL = 'fixed_palm_v1'
# Offline-verified basin_2: three palm hulls plus passive ring/little MCP
# support at the unchanged HOME. No PIP/DIP clamping or reset-time search.
POSITION_BASE_M = (0.03425173400653465, -0.008562948749966633, 0.1016880128170303)
QUATERNION_BASE_XYZW = (0.036851804416084884, 0.6479978003253494,
                        0.027523838655177456, 0.7602520855548317)
POSITION_58_BASE_M = (0.0386589006342399, 0.009324460881936406, 0.10230070124364225)
QUATERNION_58_BASE_XYZW = (-0.007058343875964488, 0.6477668926544154,
                         -0.017655594971331106, 0.7616012818586769)


def fixed_pose_for_asset(cfg):
    """Select an offline fixed pose; the 50 mm saved contract is unchanged."""
    from .object_variant import VARIANTS
    path = Path(cfg.scene.object.spawn.usd_path).resolve()
    if path == VARIANTS['die58'].resolve():
        return POSITION_58_BASE_M, QUATERNION_58_BASE_XYZW, 2, 4
    if path in {VARIANTS[name].resolve() for name in ('die50', 'die50_47g')}:
        return POSITION_BASE_M, QUATERNION_BASE_XYZW, 3, 3
    raise ValueError('Fixed palm initialization requires a validated 50 mm or 58 mm die profile')


def validate_fixed_palm_policy(policy_cfg):
    """The fixed profile requires the 25 Hz Actor initialized to absolute HOME."""
    from .revised_contract import CENTER_REVISION
    if (policy_cfg.network_revision != CENTER_REVISION
            or policy_cfg.absolute_policy_optimization is not True):
        raise ValueError('Fixed palm initialization requires hardware_center25_v3 '
                         'with absolute_policy_optimization enabled for HOME initialization')


def validated_up_world(cfg):
    """Reject configurations outside the upright 25 Hz fixed-palm profile."""
    from scipy.spatial.transform import Rotation
    from .revised_contract import CENTER_REVISION
    if cfg.actions.joint_pos.limit_contract != CENTER_REVISION:
        raise ValueError('Fixed palm initialization requires hardware_center25_v3')
    gravity = np.asarray(cfg.sim.gravity, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError('Fixed palm initialization requires finite nonzero gravity')
    magnitude = np.linalg.norm(gravity)
    if not np.isfinite(magnitude) or magnitude <= 1e-8:
        raise ValueError('Fixed palm initialization requires finite nonzero gravity')
    up = -gravity / magnitude
    quaternion = np.asarray(cfg.scene.robot.init_state.rot, dtype=float)
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion)-1.) > 1e-6):
        raise ValueError('Fixed palm initialization requires a unit XYZW hand root quaternion')
    palm_normal = Rotation.from_quat(quaternion).apply([1., 0., 0.])
    if not np.allclose(palm_normal, up, atol=1e-5, rtol=0):
        raise ValueError('Fixed palm initialization requires hand local +X aligned opposite gravity')
    return up


def configured_pose_base(cfg):
    """The existing serialized init_state fields carry the complete fixed pose."""
    from scipy.spatial.transform import Rotation
    hand, obj = cfg.scene.robot.init_state, cfg.scene.object.init_state
    inverse = Rotation.from_quat(hand.rot).inv()
    position = inverse.apply(np.asarray(obj.pos)-np.asarray(hand.pos))
    quaternion = (inverse*Rotation.from_quat(obj.rot)).as_quat()
    return position, quaternion


def configure_fixed_palm_reset(cfg):
    """Fresh supported die runs; saved environments retain their named reset term."""
    from scipy.spatial.transform import Rotation
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    position, quaternion, _, _ = fixed_pose_for_asset(cfg)
    validated_up_world(cfg)
    hand, obj = cfg.scene.robot.init_state, cfg.scene.object.init_state
    hand.joint_pos = dict(zip(REVO3_JOINT_NAMES, HOME_POSITION))
    hand.joint_vel = {'.*': 0.}
    hand.lin_vel = hand.ang_vel = (0., 0., 0.)
    base = Rotation.from_quat(hand.rot)
    obj.pos = tuple(float(v) for v in np.asarray(hand.pos)+base.apply(position))
    obj.rot = tuple(float(v) for v in (base*Rotation.from_quat(quaternion)).as_quat())
    obj.lin_vel = obj.ang_vel = (0., 0., 0.)
    # Keep the existing dictionary schema loadable by Isaac Lab's strict YAML
    # loader. The final event writes exact zero velocities regardless of offset.
    for name in cfg.events.reset_object.params['pose_range']:
        cfg.events.reset_object.params['pose_range'][name] = (0., 0.)
    cfg.events.reset_object.params['velocity_range'] = {}
    event = cfg.events.reset_robot_joints
    event.func = reset_fixed_palm_v1
    event.params['position_range'] = (0., 0.)
    event.params['velocity_range'] = (0., 0.)


def reset_fixed_palm_v1(env, env_ids, position_range, velocity_range, asset_cfg):
    """Reset HOME and the fixed relative pose; audit actual written state."""
    import torch
    from isaaclab.envs.mdp.events import reset_joints_by_offset
    from isaaclab_newton.physics import NewtonManager
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    from .core import quat_apply, quat_conjugate, quat_mul
    from .geometry_asset import build_geometry_spec
    from .support_placement import SupportPlacement

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    if tuple(position_range) != (0., 0.) or tuple(velocity_range) != (0., 0.):
        raise ValueError('Fixed palm reset prohibits joint position/velocity jitter')
    if getattr(env, 'grasp_reset_manager', None) is not None:
        raise ValueError('A grasp bank cannot override the fixed palm initial state')
    up_world = validated_up_world(env.cfg)
    env_ids = slice(None) if env_ids is None else env_ids
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if not len(ids):
        return
    robot, obj = env.scene[asset_cfg.name], env.scene['object']
    base_quaternion = tensor(robot.data.root_quat_w)[ids]
    if (not torch.isfinite(base_quaternion).all()
            or not torch.allclose(base_quaternion.norm(dim=-1),
                                  base_quaternion.new_ones(len(ids)), atol=1e-6, rtol=0)):
        raise ValueError('Fixed palm reset requires unit XYZW live hand root quaternions')
    palm_normal = quat_apply(base_quaternion, base_quaternion.new_tensor([1., 0., 0.]).expand(len(ids), -1))
    if not torch.allclose(palm_normal, base_quaternion.new_tensor(up_world).expand_as(palm_normal),
                          atol=1e-5, rtol=0):
        raise ValueError('Fixed palm reset requires every selected live hand root to face upward')
    # Validate the live frame before mutating any selected physical state.
    reset_joints_by_offset(env, env_ids, position_range, velocity_range, asset_cfg)
    joint_ids, names = robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
    if list(names) != list(REVO3_JOINT_NAMES):
        raise ValueError('Fixed HOME reset joint ordering changed')
    q = tensor(robot.data.joint_pos)[ids][:, joint_ids]
    torch.testing.assert_close(q, q.new_tensor(HOME_POSITION).expand_as(q), atol=1e-7, rtol=0)
    torch.testing.assert_close(tensor(robot.data.joint_vel)[ids][:, joint_ids], torch.zeros_like(q), atol=0, rtol=0)
    position, rotation = configured_pose_base(env.cfg)
    expected_position, expected_rotation, min_palm_contacts, min_contacts = fixed_pose_for_asset(env.cfg)
    np.testing.assert_allclose(position, expected_position, atol=1e-10, rtol=0)
    if abs(float(np.dot(rotation, expected_rotation))) < 1-1e-10:
        raise ValueError('Fixed palm reset pose differs from the validated v1 profile')
    base_position = tensor(robot.data.root_pos_w)[ids]
    local_position = q.new_tensor(position).expand(len(ids), -1)
    local_quaternion = q.new_tensor(rotation).expand(len(ids), -1)
    root_state = torch.cat((base_position+quat_apply(base_quaternion, local_position),
                           quat_mul(base_quaternion, local_quaternion), q.new_zeros(len(ids), 6)), -1)
    obj.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=ids)
    inverse = quat_conjugate(base_quaternion)
    read_position = quat_apply(inverse, tensor(obj.data.root_pos_w)[ids]-base_position)
    read_quaternion = quat_mul(inverse, tensor(obj.data.root_quat_w)[ids])
    torch.testing.assert_close(read_position, local_position, atol=6e-6, rtol=0)
    torch.testing.assert_close(read_quaternion, local_quaternion, atol=3e-7, rtol=0)
    torch.testing.assert_close(tensor(obj.data.root_lin_vel_w)[ids], torch.zeros_like(local_position), atol=0, rtol=0)
    torch.testing.assert_close(tensor(obj.data.root_ang_vel_w)[ids], torch.zeros_like(local_position), atol=0, rtol=0)

    query = getattr(env, '_fixed_palm_query', None)
    if query is None:
        rc, oc = env.cfg.scene.robot, env.cfg.scene.object
        spec = build_geometry_spec(rc.spawn.usd_path, oc.spawn.usd_path, rc.spawn.scale, oc.spawn.scale)
        query = SupportPlacement(rc.spawn.usd_path, spec)
        query._fixed_palm_mask = np.array([query.model.body(query.model.geom_bodyid[g]).name == 'query_world'
                                           for g in query.hand_geoms])
        env._fixed_palm_query = query
        env.object_support_audit = dict(protocol=PROTOCOL,
            position_base_m=position.tolist(), quaternion_base_xyzw=rotation.tolist(),
            joint_names=list(REVO3_JOINT_NAMES), home_joint_position_rad=list(HOME_POSITION),
            zero_joint_and_object_velocity=True, online_contact_search=False, settling_steps=0,
            meshes=query.audit_solver_meshes(NewtonManager._solver.mj_model),
            reset_count=0, min_gap_m=1., max_gap_m=-1., max_local_position_error_m=0., last=[])
    # Query is verification only: it never changes the fixed position/rotation.
    q_np, p_np, r_np = (value.cpu().numpy() for value in (q, read_position, read_quaternion))
    rows = []
    for i, world in enumerate(ids.cpu().tolist()):
        query.set_pose(q_np[i], p_np[i], r_np[i])
        gaps = query.all_gaps()
        gap = float(gaps.min())
        if not -5e-6 <= gap <= 1e-5:
            raise RuntimeError(f'Fixed palm pose failed contact readback: env {world}, gap {gap} m')
        support = query.hand_geoms[int(gaps.argmin())]
        body = query.model.body(query.model.geom_bodyid[support]).name
        palm_contacts = int(np.count_nonzero(gaps[query._fixed_palm_mask] <= 1e-5))
        if palm_contacts < min_palm_contacts or np.count_nonzero(gaps <= 1e-5) < min_contacts:
            raise RuntimeError(f'Fixed pose lost its validated supports: env {world}, palm contacts {palm_contacts}')
        rows.append(dict(env_id=world, gap_m=gap, support_body=body, palm_contact_hulls=palm_contacts))
    audit = env.object_support_audit
    audit['reset_count'] += len(ids)
    audit['min_gap_m'] = min(audit['min_gap_m'], min(row['gap_m'] for row in rows))
    audit['max_gap_m'] = max(audit['max_gap_m'], max(row['gap_m'] for row in rows))
    audit['max_local_position_error_m'] = max(audit['max_local_position_error_m'], float((read_position-local_position).abs().max()))
    audit['last'] = rows
