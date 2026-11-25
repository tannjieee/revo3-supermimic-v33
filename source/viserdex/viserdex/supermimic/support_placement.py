"""Read-only convex shape cast for placing an object directly on the hand.

No dynamics, settling steps, springs, or commanded joint targets are involved.
The query hulls come from the same authored collision geometry as Newton.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_candidates import ConvexQueries


class SupportPlacement(ConvexQueries):
    def __init__(self, robot_path, spec):
        super().__init__(robot_path, spec)
        # Query-model setting only: use the native convex distance algorithm,
        # matching the actual MJWarp shape query, not historical libccd queries.
        self.model.opt.disableflags &= ~int(self.mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self._boxes = []
        for geom in self.hand_geoms:
            mesh = self.model.geom_dataid[geom]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            points = self.model.mesh_vert[start:start+count]
            low, high = points.min(0), points.max(0)
            self._boxes.append(np.array([[x,y,z] for x in (low[0],high[0])
                                        for y in (low[1],high[1]) for z in (low[2],high[2])]))
        self._boxes = np.asarray(self._boxes)

    def place(self, q, position, xyzw, up, *, clearance_m=2e-6, max_iterations=256):
        """Preserve lateral position/orientation; descend to first convex contact.

        Closest-point separating planes bound translation to convex contact,
        so each step cannot pass through a thin hull. Broad phase uses enclosing
        boxes only to reject horizontally disjoint pairs, never as the surface.
        """
        p = np.asarray(position, dtype=float).copy()
        up = np.asarray(up, dtype=float)
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
            raise ValueError('Support placement requires a finite upward direction')
        up = up/np.linalg.norm(up)
        helper = np.eye(3)[np.argmin(abs(up))]
        side = np.cross(up,helper);side /= np.linalg.norm(side)
        basis = np.stack([side,np.cross(up,side),up],axis=1)
        self.set_pose(q,p,xyzw)
        matrices = self.data.geom_xmat[self.hand_geoms].reshape(-1,3,3)
        boxes = np.einsum('gij,gvj->gvi',matrices,self._boxes)+self.data.geom_xpos[self.hand_geoms,None,:]
        boxes = boxes@basis
        low, high = boxes.min(1),boxes.max(1)
        object_points = Rotation.from_quat(xyzw).apply(self.spec['convex_vertices'])@basis
        obj_low,obj_high = object_points.min(0),object_points.max(0)
        lateral = p-up*(p@up)
        lateral_basis = lateral@basis
        overlap = ((high[:,:2]+clearance_m >= lateral_basis[:2]+obj_low[:2])
                   & (low[:,:2]-clearance_m <= lateral_basis[:2]+obj_high[:2])).all(1)
        candidates = np.asarray(self.hand_geoms)[overlap]
        if not len(candidates):
            raise ValueError('No hand support under this randomized lateral position')
        height = float(np.max(high[overlap,2])-obj_low[2]+.001)
        # Use the already sampled free cavity when possible. Starting above
        # every finger would unnecessarily perch a small die on a fingertip.
        initial_gaps = self.all_gaps()
        if initial_gaps.min() >= clearance_m:
            height = min(height, float(p@up))
        lowest = float(np.min(low[overlap,2])-obj_high[2]-.001)
        for iteration in range(max_iterations):
            p = lateral+height*up
            self.data.mocap_pos[self.object_mocap] = p
            self.mujoco.mj_kinematics(self.model,self.data)
            distances, slopes = [], []
            for geom in candidates:
                points = np.zeros(6)
                distance = self.mujoco.mj_geomDistance(self.model,self.data,geom,self.object_id,1.,points)
                normal = points[3:]-points[:3]
                normal /= max(np.linalg.norm(normal),1e-15)
                distances.append(distance)
                slopes.append(float(normal@up))
            gaps = np.asarray(distances)
            if not np.isfinite(gaps).all():
                raise RuntimeError('Nonfinite support distance')
            gap = float(gaps.min())
            if gap < -2e-6:
                raise RuntimeError(f'Support cast penetrated a hull: {gap} m')
            if gap <= clearance_m+2e-7:
                support = int(candidates[gaps.argmin()])
                points = np.zeros(6)
                self.mujoco.mj_geomDistance(self.model,self.data,support,self.object_id,1.,points)
                normal = points[3:]-points[:3]
                normal /= max(np.linalg.norm(normal),1e-15)
                return p,dict(gap_m=gap,height_shift_m=float((p-np.asarray(position))@up),
                    query_iterations=iteration+1,support_geom=support,
                    support_body=self.model.body(self.model.geom_bodyid[support]).name,
                    normal_up_dot=float(normal@up))
            # The object must cross each pair's separating plane before that
            # convex pair can touch. Take the earliest plane crossing among ALL
            # candidates, not just the currently nearest hull. A sideways
            # near-contact no longer forces thousands of micron-sized steps.
            slopes = np.asarray(slopes)
            crossing = np.full_like(gaps, np.inf)
            approaching = slopes > 1e-10
            crossing[approaching] = (gaps[approaching]-clearance_m)/slopes[approaching]
            height -= crossing.min()
            if height < lowest:
                raise ValueError('The vertical sweep missed all hand hulls')
        raise RuntimeError('Support cast did not converge; refusing an airborne reset')

    def place_world(self, q, pose_xyzw, base_position, base_xyzw, up_world):
        """Isaac Lab v3 and SciPy both use XYZW; only MuJoCo uses WXYZ."""
        pose = np.asarray(pose_xyzw, dtype=float).copy()
        base = Rotation.from_quat(base_xyzw)
        inverse = base.inv()
        position = inverse.apply(pose[:3]-base_position)
        rotation = inverse*Rotation.from_quat(pose[3:])
        placed, audit = self.place(q, position, rotation.as_quat(), inverse.apply(up_world))
        pose[:3] = base.apply(placed)+base_position
        return pose, audit


def reset_joints_with_supported_object(env, env_ids, position_range, velocity_range, asset_cfg):
    raise ValueError('Legacy support reset used the wrong quaternion order. '
                     'Start a fresh run with reset_joints_with_supported_object_xyzw_v2; '
                     'do not resume the affected environment contract.')


def reset_joints_with_supported_object_xyzw_v2(env, env_ids, position_range, velocity_range, asset_cfg):
    """Official joint randomization, then zero-step object support placement.

    Reuses the existing final reset event's schema so saved free-drop runs keep
    their original function and new runs serialize this function explicitly.
    """
    import torch
    from isaaclab.envs.mdp.events import reset_joints_by_offset
    from isaaclab_newton.physics import NewtonManager
    from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
    from .geometry_asset import build_geometry_spec
    from .core import quat_conjugate, quat_apply, quat_mul

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    reset_joints_by_offset(env, env_ids, position_range, velocity_range, asset_cfg)
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if not len(ids):
        return
    query = getattr(env, '_object_support_query', None)
    if query is None:
        robot_cfg, object_cfg = env.cfg.scene.robot, env.cfg.scene.object
        spec = build_geometry_spec(robot_cfg.spawn.usd_path, object_cfg.spawn.usd_path,
                                   robot_cfg.spawn.scale, object_cfg.spawn.scale)
        query = SupportPlacement(robot_cfg.spawn.usd_path, spec)
        env.object_support_audit = dict(protocol='world_xyzw_v2',
                                       meshes=query.audit_solver_meshes(NewtonManager._solver.mj_model),
                                       reset_count=0, max_gap_m=0., min_gap_m=1., last=[])
        env._object_support_query = query
    robot, obj = env.scene[asset_cfg.name], env.scene['object']
    joint_ids, _ = robot.find_joints(list(REVO3_JOINT_NAMES), preserve_order=True)
    q = tensor(robot.data.joint_pos)[ids][:, joint_ids].cpu().numpy()
    base_pos = tensor(robot.data.root_pos_w)[ids].cpu().numpy()
    base_quat = tensor(robot.data.root_quat_w)[ids].cpu().numpy()
    pose = torch.cat((tensor(obj.data.root_pos_w)[ids], tensor(obj.data.root_quat_w)[ids]), dim=-1)
    poses = pose.cpu().numpy().copy()
    up_world = -np.asarray(env.cfg.sim.gravity, dtype=float)
    rows = []
    for i, world_id in enumerate(ids.cpu().tolist()):
        placed, row = query.place_world(q[i], poses[i], base_pos[i], base_quat[i], up_world)
        poses[i,:3] = placed[:3]
        row.update(env_id=world_id, position_w=poses[i,:3].tolist())
        rows.append(row)
    # Keep the sampled orientation and velocities exactly; only position moves.
    pose[:,:3] = torch.as_tensor(poses[:,:3], device=pose.device, dtype=pose.dtype)
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    # Read back the actual written buffers and independently transform them
    # with the project's XYZW tensor math (not place_world's SciPy conversion).
    inverse = quat_conjugate(tensor(robot.data.root_quat_w)[ids])
    actual_pos = quat_apply(inverse, tensor(obj.data.root_pos_w)[ids]
                           - tensor(robot.data.root_pos_w)[ids]).cpu().numpy()
    actual_quat = quat_mul(inverse, tensor(obj.data.root_quat_w)[ids]).cpu().numpy()
    for i, row in enumerate(rows):
        query.set_pose(q[i], actual_pos[i], actual_quat[i])
        gap = query.minimum_gap()
        if not -5e-6 <= gap <= 1e-5:
            raise RuntimeError(f'Written reset pose is not supported: env {row["env_id"]}, gap {gap} m')
        row['cast_gap_m'] = row['gap_m']
        row['gap_m'] = gap
    audit = env.object_support_audit
    audit['reset_count'] += len(rows)
    audit['max_gap_m'] = max(audit['max_gap_m'], max(row['gap_m'] for row in rows))
    audit['min_gap_m'] = min(audit['min_gap_m'], min(row['gap_m'] for row in rows))
    audit['last'] = rows  # bounded to the latest reset batch


def configure_supported_reset(cfg):
    """Fresh manual-convex runs start on the hand, including later resets."""
    cfg.events.reset_robot_joints.func = reset_joints_with_supported_object_xyzw_v2
    cfg.events.reset_object.params['pose_range']['z'] = (0., 0.)
