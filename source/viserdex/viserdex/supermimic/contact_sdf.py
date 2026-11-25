"""Experimental Torch ContactSDF and a delayed-PD dynamic adaptation.

Derived from ContactSDF equations 12-18 (Yang & Jin, arXiv:2408.09612).
The quasi model uses a spring hand; the dynamic adaptation uses diagonal
articulated inertia and implicit PD, omits Coriolis, and freezes the contact
linearization within 40 ms. Neither is Newton nor an exact contact solver.
Signed geometry margins preserve penetration information (author-code style);
velocity-space distance includes the smooth zero facet from the paper.
"""
import math
import torch
from torch import nn
from .geometry import HandObjectGeometry
from .core import quat_apply, quat_conjugate, quat_mul, rotation_vector_quat


def smooth_polytope(queries, planes, temperature, *, truncate=False):
    values=queries@planes[:,:3].T+planes[:,3]
    logits=values/temperature
    if truncate:logits=torch.cat((torch.zeros_like(logits[...,:1]),logits),-1)
    distance=temperature*torch.logsumexp(logits,-1)
    weights=logits.softmax(-1)
    if truncate:weights=weights[...,1:]
    gradient=weights@planes[:,:3]
    return distance,gradient


def velocity_projection(z, normals, offsets, temperature):
    """One SDF gradient projection in whitened velocity coordinates; approximate."""
    violations=(normals*z[:,None]).sum(-1)+offsets
    logits=torch.cat((torch.zeros_like(violations[:,:1]),violations/temperature),-1)
    distance=temperature*torch.logsumexp(logits,-1)
    gradient=(logits.softmax(-1)[:,1:,None]*normals).sum(1)
    return z-distance[:,None]*gradient


def skew(x):
    a,b,c=x.unbind(-1);o=torch.zeros_like(a)
    return torch.stack((o,-c,b,c,o,-a,-b,a,o),-1).reshape(*x.shape[:-1],3,3)


def rotation_matrix(q):
    eye=torch.eye(3,device=q.device,dtype=q.dtype)
    return quat_apply(q[...,None,:].expand(*q.shape[:-1],3,4),eye.expand(*q.shape[:-1],3,3)).transpose(-1,-2)


class ContactSDF(nn.Module):
    def __init__(self, spec, surface, cfg, metadata, mode='quasi'):
        super().__init__()
        if mode not in ('quasi','dynamic'):raise ValueError('Unknown ContactSDF mode')
        if cfg.control_dt!=.04 or not cfg.include_delay_ready:raise ValueError('This experiment requires the verified 25Hz/ready contract')
        self.mode=mode;self.cfg=cfg
        self.fk=HandObjectGeometry(spec)
        def buffer(name,value,dtype=torch.float32):self.register_buffer(name,torch.as_tensor(value,dtype=dtype).detach().clone())
        buffer('points_local',surface['point_local']);buffer('point_body',surface['point_body'],torch.long)
        buffer('body_com',surface['body_com']);buffer('planes',spec['convex_planes'])
        buffer('bias',cfg.encoder_bias_rad);buffer('limits',cfg.network_joint_limits)
        from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
        identified=metadata['context_manifest']['identified_fixed']
        joint_order=[REVO3_JOINT_NAMES.index(name) for name in metadata['joint_names']]
        for name,key in (('armature','armature_kg_m2'),('viscous','viscous_Nm_s_rad'),('joint_friction','friction_Nm')):
            buffer(name,[identified[key][j] for j in joint_order])
        # Geometry is base-relative; fixed-palm profile requires local +X up.
        buffer('gravity',[-9.81,0.,0.]);buffer('object_inertia',spec['object_inertia'])
        self.object_mass=spec['object_mass']
        derived=metadata['context_manifest']['derived']['robot.body_inertia']
        order=[derived['body_names'].index(n) for n in spec['body_names'][1:]]
        buffer('mass_indices',order,torch.long)
        buffer('body_mass_nominal',[derived['default_mass'][j] for j in order])
        buffer('body_inertia',[derived['default_inertia'][j] for j in order]);self.body_inertia=self.body_inertia.reshape(21,3,3)
        # Kinematic tree, represented in exactly the checkpoint's joint order.
        ancestor=torch.zeros(22,21);child=1
        for level in spec['levels']:
            for parent,joint in zip(level['parent'],level['joint']):
                ancestor[child]=ancestor[parent];ancestor[child,joint]=1;child+=1
        buffer('ancestor',ancestor)
        self.log_scales=nn.Parameter(torch.zeros(5)) # bounded effective mass, inertia, K, D, friction multipliers
        self.raw_velocity_temperature=nn.Parameter(torch.tensor(-5.0))
        self.raw_geometry_temperature=nn.Parameter(torch.tensor(-3.0))

    def kinematics(self,s):
        p,r,_,_=self.fk.body_kinematics(s)
        origins=[];axes=[];indices=[]
        for level in self.fk.groups:
            parent=getattr(self.fk,f'parent_{level}');j=getattr(self.fk,f'joint_{level}')
            origins.append(p[:,parent]+quat_apply(r[:,parent],getattr(self.fk,f'p0_{level}').expand(len(s),-1,-1)))
            rot=quat_mul(r[:,parent],getattr(self.fk,f'r0_{level}').expand(len(s),-1,-1))
            axes.append(quat_apply(rot,getattr(self.fk,f'axis_{level}').expand(len(s),-1,-1)));indices.append(j)
        order=torch.cat(indices).argsort();origin=torch.cat(origins,1)[:,order];axis=torch.cat(axes,1)[:,order]
        points=p[:,self.point_body]+quat_apply(r[:,self.point_body],self.points_local[None].expand(len(s),-1,-1))
        jp=torch.cross(axis[:,None],points[:,:,None]-origin[:,None],dim=-1)*self.ancestor[self.point_body][None,:,:,None]
        return p,r,origin,axis,points,jp.transpose(-1,-2)

    def contact(self,s,kin):
        p,r,origin,axis,points,jp=kin
        objrot=s[:,45:49];local=quat_apply(quat_conjugate(objrot)[:,None].expand(-1,len(self.point_body),-1),points-s[:,None,42:45])
        temperature=1e-5+1.9e-4*torch.sigmoid(self.raw_geometry_temperature)
        phi,gradient=smooth_polytope(local,self.planes,temperature)
        unit=gradient/gradient.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        # Basis is defined in object coordinates; world translation cannot enter a normal.
        helper=torch.nn.functional.one_hot(unit.abs().argmin(-1),3).to(s.dtype)
        tangent=torch.cross(unit,helper,dim=-1);tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        second=torch.cross(unit,tangent,dim=-1)
        directions=torch.stack((unit,tangent,second,-tangent,-second),2)
        directions=quat_apply(objrot[:,None,None].expand(-1,len(self.point_body),5,-1),directions)
        cp=points-quat_apply(objrot[:,None].expand(-1,len(self.point_body),-1),phi[:,:,None]*gradient)
        # gap rate = outward_normal dot (hand_point_velocity-object_point_velocity).
        object_j=torch.cat((-torch.eye(3,device=s.device,dtype=s.dtype).expand(len(s),len(self.point_body),3,3),skew(cp-s[:,None,42:45])),-1)
        relative=torch.cat((object_j,jp),-1)
        normal_j=torch.einsum('bpc,bpcv->bpv',directions[:,:,0],relative)
        tangent_j=torch.einsum('bpdc,bpcv->bpdv',directions[:,:,1:],relative)
        return phi,normal_j,tangent_j

    def inertia(self,s,context,kin):
        p,r,origin,axis,_,_=kin
        com=p[:,1:]+quat_apply(r[:,1:],self.body_com[None,1:].expand(len(s),-1,-1))
        jv=torch.cross(axis[:,None],com[:,:,None]-origin[:,None],dim=-1)*self.ancestor[None,1:,:,None]
        jw=axis[:,None]*self.ancestor[None,1:,:,None]
        mass=context[:,self.mass_indices]
        rot=rotation_matrix(r[:,1:]);ib=rot@self.body_inertia@rot.transpose(-1,-2)*(mass/self.body_mass_nominal)[:,:,None,None]
        diag=(mass[:,:,None]*jv.square().sum(-1)+torch.einsum('bljc,blcd,bljd->blj',jw,ib,jw)).sum(1).clamp_min(1e-7)
        gravity=(mass[:,:,None]*(jv*self.gravity).sum(-1)).sum(1)
        # Body inertia transformed into base coordinates; diagonal approximation is explicit.
        ro=rotation_matrix(s[:,45:49]);io=ro@self.object_inertia@ro.transpose(-1,-2)*(context[:,21]/self.object_mass)[:,None,None]
        object_diag=torch.cat((context[:,21:22].expand(-1,3),io.diagonal(dim1=-2,dim2=-1)),-1)
        return torch.cat((object_diag,diag+self.armature),-1),gravity

    def forward(self,s,target,context):
        kin=self.kinematics(s);phi,jn,jt=self.contact(s,kin);mass,g_hand=self.inertia(s,context,kin)
        scales=torch.exp(math.log(4)*torch.tanh(self.log_scales))
        mass=mass*torch.cat((scales[0].expand(3),scales[1].expand(3),scales[1].expand(21)))
        kp=context[:,23:44]*scales[2];kd=context[:,44:65]*scales[3]
        # Actual solver combines materials; treat object friction as a bounded effective prior.
        mu=context[:,22]*scales[4]
        jac=(jn[:,:,None]-mu[:,None,None,None]*jt).flatten(1,2)
        gaps=phi.repeat_interleave(4,dim=1)
        velocity=torch.cat((s[:,49:55],s[:,21:42]),-1)
        previous=(self.limits[:,0]+self.limits[:,1])/2+s[:,55:76]*(self.limits[:,1]-self.limits[:,0])/2
        q=s[:,:21];position=s[:,42:45];quat=s[:,45:49]
        displacement=torch.zeros_like(velocity)
        h=.04 if self.mode=='quasi' else .005
        for tick in range(1 if self.mode=='quasi' else 8):
            active_target=torch.where((s[:,83:84]>0)&(tick<3),previous,target) if self.mode=='dynamic' else target
            error=active_target+self.bias-q
            if self.mode=='quasi':
                metric=torch.cat((mass[:,:6]/h**2,kp),-1)
                force=torch.cat((context[:,21:22]*self.gravity,torch.zeros_like(position),kp*error+g_hand),-1)
            else:
                # Known passive losses are not part of the clipped actuator effort.
                # Smooth implicit dry-friction approximation; not an exact stiction solve.
                passive=self.viscous+self.joint_friction/torch.sqrt(velocity[:,6:].square()+.01**2)
                metric=mass/h**2+torch.cat((torch.zeros_like(velocity[:,:6]),(kd+passive)/h+kp),-1)
                torque=(kp*error-kd*velocity[:,6:]).clamp(-1.28,1.28)
                force=mass*velocity/h+torch.cat((context[:,21:22]*self.gravity,torch.zeros_like(position),kd*velocity[:,6:]+torque+g_hand),-1)
            inv=metric.rsqrt();query=force*inv
            a=jac*inv[:,None];norm=a.norm(dim=-1).clamp_min(1e-10)
            normals=-a/norm[:,:,None]
            offsets=-(gaps+(jn.repeat_interleave(4,dim=1)*displacement[:,None]).sum(-1))/norm
            temperature=1e-6+.02*torch.sigmoid(self.raw_velocity_temperature)
            projected=velocity_projection(query,normals,offsets,temperature)
            velocity=projected*inv/h
            displacement=displacement+h*velocity
            position=position+h*velocity[:,:3]
            quat=quat_mul(rotation_vector_quat(h*velocity[:,3:6]),quat)
            quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-10)
            q=q+h*velocity[:,6:]
        return torch.cat((q,velocity[:,6:],position,quat,velocity[:,:6]),-1)
