"""Offline grasp proposals migrated from the Revo3 contact-search experiments.

The query model is rebuilt from the *current USD convex pieces*. MuJoCo/libccd
is used only for closest-point queries; every physical trial uses Isaac Lab
Newton. A bounded nonnegative wrench QP scores candidates (not a certificate).
This is contact optimization + QP screening, not official GraspQP/MALA*.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation

FINGERS = ('thumb','index','middle','ring','little')


class ConvexQueries:
    def __init__(self, robot_path, spec):
        import mujoco
        from pxr import Usd, UsdGeom, UsdPhysics
        from .geometry_asset import active_colliders, vertices_in
        self.mujoco, self.spec = mujoco, spec
        stage=Usd.Stage.Open(str(robot_path));cache=UsdGeom.XformCache()
        root=ET.Element('mujoco');asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
        ET.SubElement(root,'option',timestep='.0025')
        bodies={name:ET.SubElement(world,'body',name='query_'+name,mocap='true') for name in spec['body_names']}
        self.groups=[[] for _ in range(5)];self.hand_geoms=[]
        for i,prim in enumerate(active_colliders(stage)):
            body=prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI): body=body.GetParent()
            name=body.GetName();points=vertices_in(prim,body,cache)
            mesh='hand_'+str(i)
            ET.SubElement(asset,'mesh',name=mesh,vertex=' '.join(map(str,points.ravel())),maxhullvert='-1')
            ET.SubElement(bodies[name],'geom',name=mesh,type='mesh',mesh=mesh,contype='0',conaffinity='0')
            self.hand_geoms.append(mesh)
            for g,finger in enumerate(FINGERS):
                if 'right_'+finger+'_' in name and ('_PIP_' in name or '_DIP_' in name): self.groups[g].append(mesh)
        points=np.asarray(spec['convex_vertices'])
        ET.SubElement(asset,'mesh',name='object_mesh',vertex=' '.join(map(str,points.ravel())),maxhullvert='-1')
        ob=ET.SubElement(world,'body',name='query_object',mocap='true')
        ET.SubElement(ob,'geom',name='object',type='mesh',mesh='object_mesh',contype='0',conaffinity='0')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.data=mujoco.MjData(self.model)
        self.groups=[[self.model.geom(n).id for n in names] for names in self.groups]
        self.hand_geoms=[self.model.geom(n).id for n in self.hand_geoms]
        self.object_id=self.model.geom('object').id
        self.mocap=[self.model.body('query_'+n).mocapid[0] for n in spec['body_names']]
        self.object_mocap=self.model.body('query_object').mocapid[0]
        self.levels=[]
        for group in spec['levels']:
            self.levels.append({k:np.asarray(v) for k,v in group.items()})

    def set_pose(self,q,p,xyzw):
        positions=[np.zeros(3)];rotations=[Rotation.identity()]
        for level in self.levels:
            for i,j in enumerate(level['joint']):
                parent=level['parent'][i];rp=rotations[parent]
                joint=rp*Rotation.from_quat(level['r0'][i])
                rotation=joint*Rotation.from_rotvec(level['axis'][i]*q[j])*Rotation.from_quat(level['r1'][i]).inv()
                position=positions[parent]+rp.apply(level['p0'][i])-rotation.apply(level['p1'][i])
                positions.append(position);rotations.append(rotation)
        self.data.mocap_pos[self.mocap]=positions
        self.data.mocap_quat[self.mocap]=np.array([r.as_quat()[[3,0,1,2]] for r in rotations])
        self.data.mocap_pos[self.object_mocap]=p
        self.data.mocap_quat[self.object_mocap]=np.asarray(xyzw)[[3,0,1,2]]
        self.mujoco.mj_kinematics(self.model,self.data)

    def closest(self,group):
        rows=[]
        for geom in self.groups[group]:
            points=np.zeros(6)
            distance=self.mujoco.mj_geomDistance(self.model,self.data,geom,self.object_id,.2,points)
            vector=points[3:]-points[:3];length=np.linalg.norm(vector)
            normal=vector/length*np.sign(distance) if length>1e-8 and abs(distance)>1e-8 else self.data.geom_xpos[self.object_id]-self.data.geom_xpos[geom]
            normal=normal/max(np.linalg.norm(normal),1e-12)
            rows.append((float(distance),normal,points[3:].copy()))
        return min(rows,key=lambda row:row[0])

    def all_gaps(self):
        return np.array([self.mujoco.mj_geomDistance(self.model,self.data,g,self.object_id,.2,None) for g in self.hand_geoms])

    def minimum_gap(self):
        return float(self.all_gaps().min())

    def audit_solver_meshes(self, model):
        """Independently compare the final MJWarp collision hulls to query hulls."""
        from scipy.spatial import ConvexHull, cKDTree
        def cloud(m, g, support=False):
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start:start+count]
            if support and m.mesh_graphadr[mesh] >= 0:
                graph = m.mesh_graph[m.mesh_graphadr[mesh]:]
                n = int(graph[0])
                vertices = vertices[graph[2+n:2+2*n]]
            return Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).apply(vertices) + m.geom_pos[g]
        expected = {}
        for g in self.hand_geoms:
            name = self.model.body(self.model.geom_bodyid[g]).name.removeprefix('query_')
            expected.setdefault(name, []).append(cloud(self.model, g))
        errors, support_errors = [], []
        for g in range(model.ngeom):
            label = model.body(model.geom_bodyid[g]).name
            name = next((n for n in expected if label == n or label.endswith('_'+n) or label.endswith('/'+n)), None)
            if name not in expected or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            points = cloud(model, g)
            candidates = expected[name]
            distances = [max(cKDTree(v).query(points)[0].max(), cKDTree(points).query(v)[0].max()) for v in candidates]
            best = int(np.argmin(distances))
            reference = candidates.pop(best)
            errors.append(float(distances[best]))
            support = cloud(model, g, True)
            planes = ConvexHull(support).equations
            support_errors.append(float(max(0., (reference@planes[:, :3].T+planes[:, 3]).max())))
        if len(errors) != 144 or any(expected.values()) or max(errors) > 2e-6 or max(support_errors) > 2e-6:
            raise ValueError(f'MJWarp/query hull mismatch: {len(errors)} shapes, {max(errors, default=-1)} m vertices, {max(support_errors, default=-1)} m hull')
        objects = [g for g in range(model.ngeom) if 'object' in model.geom(g).name.lower()
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if len(objects) != 1:
            raise ValueError('Expected one final MJWarp object convex hull')
        reference, actual = cloud(self.model, self.object_id), cloud(model, objects[0])
        error = float(max(cKDTree(reference).query(actual)[0].max(), cKDTree(actual).query(reference)[0].max()))
        planes = ConvexHull(cloud(model, objects[0], True)).equations
        support_error = float(max(0., (reference@planes[:, :3].T+planes[:, 3]).max()))
        if max(error, support_error) > 2e-6:
            raise ValueError('Final MJWarp object hull differs from query model')
        return dict(shapes=len(errors), max_vertex_error_m=max(errors), max_support_error_m=max(support_errors),
                    object_vertex_error_m=error, object_support_error_m=support_error)


def wrench_qp(contacts, center, friction=.7, radius=.025):
    """Bounded least-squares QP for all 12 signed unit wrench directions.

    Torques normalized by object radius. Residual is a proposal ranking, not
    an assertion of force closure or stability under identified PD dynamics.
    """
    columns=[]
    for _, inward, point in contacts:
        helper=np.eye(3)[np.argmin(abs(inward))]
        t=np.cross(inward,helper);t/=np.linalg.norm(t);u=np.cross(inward,t)
        for angle in np.arange(4)*np.pi/2:
            force=inward+friction*(np.cos(angle)*t+np.sin(angle)*u)
            columns.append(np.r_[force,np.cross(point-center,force)/radius])
    matrix=np.stack(columns,axis=1)
    residuals=[]
    for direction in np.r_[np.eye(6),-np.eye(6)]:
        result=lsq_linear(matrix,direction,bounds=(0,10),tol=1e-6,max_iter=100)
        residuals.append(float(np.linalg.norm(matrix@result.x-direction)))
    return dict(max_residual=max(residuals),mean_residual=float(np.mean(residuals)),rank=int(np.linalg.matrix_rank(matrix)))


def propose(query, limits, bias, seed_q, seed_pose, index, rng, max_nfev=60, *, geometry_limits):
    q0=np.clip(np.asarray(seed_q)+rng.normal(0,.025,21),geometry_limits[:,0]+1e-5,geometry_limits[:,1]-1e-5)
    p0=np.asarray(seed_pose[:3])+rng.normal(0,.002,3)
    xyzw=(Rotation.from_rotvec(rng.normal(0,.2,3))*Rotation.from_quat(seed_pose[3:])).as_quat()
    selected=((0,1,2),(0,2,3),(0,1,3),(0,2,4))[index%4]
    low=np.r_[geometry_limits[:,0]+1e-5,[.015,-.06,.055]]
    high=np.r_[geometry_limits[:,1]-1e-5,[.115,.065,.19]]
    z0=np.clip(np.r_[q0,p0],low+1e-6,high-1e-6)
    def residual(z):
        query.set_pose(z[:21],z[21:],xyzw)
        contacts=[query.closest(g) for g in selected]
        gaps=np.array([c[0] for c in contacts])
        opposing=np.array([contacts[0][1]@c[1] for c in contacts[1:]])
        return np.r_[(gaps+.00015)/.004,.6*np.maximum(opposing+.3,0),
                     np.maximum(-query.all_gaps()-.0004,0)/.002,.012*(z[:21]-q0),.025*(z[21:]-p0)/.02]
    fit=least_squares(residual,z0,bounds=(low,high),max_nfev=max_nfev,diff_step=2e-4,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    q,p=fit.x[:21],fit.x[21:];query.set_pose(q,p,xyzw)
    contacts=[query.closest(g) for g in selected];minimum=query.minimum_gap()
    gaps=[c[0] for c in contacts]
    score=wrench_qp(contacts,p)
    target=q.copy()
    for group in selected:
        ids=[i for i,n in enumerate(query.spec['joint_names']) if 'right_'+FINGERS[group]+'_' in n]
        grad=[]
        for j in ids:
            q1=q.copy();q1[j]+=.0001;query.set_pose(q1,p,xyzw);above=query.closest(group)[0]
            q1[j]-=.0002;query.set_pose(q1,p,xyzw);below=query.closest(group)[0]
            grad.append((above-below)/.0002)
        grad=np.array(grad)
        target[ids]+=np.clip(-.0008*grad/max(grad@grad,1e-8),-.06,.06)
    target=np.clip(target-bias,limits[:,0],limits[:,1])
    return dict(q_geometry=q.tolist(),hold_target_encoder=target.tolist(),object_pose_xyzw=np.r_[p,xyzw].tolist(),
                candidate_method='usd_contact_optimization_wrench_qp_v1',selected_fingers=list(selected),
                geometry_passed=bool(max(gaps)<.002 and minimum>-.0015),query_min_gap_m=float(minimum),
                contact_gaps_m=gaps,qp=score,nfev=fit.nfev,certified=False)
