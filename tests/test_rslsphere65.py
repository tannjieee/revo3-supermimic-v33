import math
import torch
import pytest
from test_supermimic_geometry import spec
from test_supermimic import fixture
from viserdex.supermimic.geometry import HandObjectGeometry
from viserdex.supermimic.core import POS, ROT, VEL, OMEGA, Q, QD, quat_apply, quat_conjugate, quat_mul, rotation_vector_quat

def sphere():
 s=spec();s.pop('box_center');s.pop('box_half')
 return dict(s,object_shape='sphere',sphere_center=[0,0,0],sphere_radius=.0325)

def test_sphere_distance_normal_nearest_and_center_gradient():
 g=HandObjectGeometry(sphere()).double()
 x=torch.tensor([[.0425,0,0],[0,.0225,0],[0,0,-.0325],[0,0,0]],dtype=torch.double,requires_grad=True)
 d,n=g.surface(x)
 torch.testing.assert_close(d[:,0],x.new_tensor([.01,-.01,0,-.0325]),atol=2e-9,rtol=0)
 torch.testing.assert_close(n.norm(dim=-1),torch.ones(4,dtype=torch.double))
 torch.testing.assert_close((x-d*n).norm(dim=-1),torch.full((4,),float(g.sphere_radius),dtype=torch.double))
 (d.square().sum()+n.square().sum()).backward();assert torch.isfinite(x.grad).all()

def test_surface_gradcheck_away_from_nondifferentiable_center():
 g=HandObjectGeometry(sphere()).double()
 x=torch.tensor([[.01,.03,.04],[-.01,.001,-.003]],dtype=torch.double,requires_grad=True)
 assert torch.autograd.gradcheck(g.surface,(x,),eps=1e-6,atol=1e-5)

def test_object_rotation_preserves_sphere_gap_but_rotates_relative_normal():
 g=HandObjectGeometry(sphere()).double();s,_=fixture(1);s=s.double();s[:,POS]=0;s[:,ROT]=s.new_tensor([0,0,0,1])
 a=g.readout(s);r=rotation_vector_quat(s.new_tensor([[.2,.3,-.7]]));s[:,ROT]=r;b=g.readout(s)
 torch.testing.assert_close(a[4],b[4])
 torch.testing.assert_close(b[2],quat_apply(quat_conjugate(r)[:,None],a[2]))
 torch.testing.assert_close(b[5],quat_apply(quat_conjugate(r)[:,None],a[5]))

def test_relative_velocity_matches_moving_rotating_frame_finite_difference():
 g=HandObjectGeometry(sphere()).double();s,_=fixture(3);s=s.double()
 torch.manual_seed(7);s[:,Q]=torch.randn(3,21,dtype=torch.double)*.2;s[:,QD]=torch.randn(3,21,dtype=torch.double)
 s[:,POS]=s.new_tensor([.01,.02,.03]);s[:,VEL]=s.new_tensor([.1,.2,.3]);s[:,OMEGA]=s.new_tensor([.3,-.2,.4])
 s[:,ROT]=rotation_vector_quat(s.new_tensor([[.2,.1,.3]]).expand(3,-1))
 eps=1e-6;positions=[]
 for dt in (eps,-eps):
  t=s.clone();t[:,Q]+=dt*s[:,QD];t[:,POS]+=dt*s[:,VEL]
  t[:,ROT]=quat_mul(rotation_vector_quat(s[:,OMEGA]*dt),s[:,ROT]);positions.append(g.readout(t)[2])
 torch.testing.assert_close(g.readout(s)[3],(positions[0]-positions[1])/(2*eps),atol=1e-8,rtol=1e-6)

def test_geometry_supervision_reaches_pose_twist_and_joints():
 g=HandObjectGeometry(sphere());s,_=fixture(2);t=s.clone();t[:,Q]+=.1;t[:,POS]+=.01;t[:,VEL]+=.1;t[:,OMEGA]+=.2;t.requires_grad_()
 g.supervised_errors(t,s).mean().backward()
 for field in (Q,POS,VEL,OMEGA):assert t.grad[:,field].abs().sum()>0
 assert torch.isfinite(t.grad).all()

def test_actual_asset_is_sphere_and_30_points():
 from pathlib import Path
 from viserdex.supermimic.geometry_asset import build_geometry_spec
 from viserdex.assets.revo3_spec import REVO3_COMPACT_USD
 root=Path(__file__).resolve().parents[1]
 s=build_geometry_spec(REVO3_COMPACT_USD,root/'assets/rslsphere65/RSLSphere65/RSLSphere65.usda')
 assert s['object_shape']=='sphere' and s['sphere_radius']==.0325
 assert s['sphere_center']==[0,0,0] and len(s['point_names'])==30
 assert 'box_half' not in s
