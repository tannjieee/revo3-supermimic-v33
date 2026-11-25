"""Collision-derived convex features, gradients, and checkpoint isolation."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from viserdex.supermimic.geometry_asset import build_geometry_spec
from viserdex.supermimic.geometry import HandObjectGeometry
from viserdex.assets.revo3_spec import REVO3_COMPACT_USD
from test_supermimic import fixture
from viserdex.supermimic.core import POS,ROT,Q,VEL,OMEGA,rotation_vector_quat,quat_mul,quat_apply,quat_conjugate
ROOT=Path(__file__).resolve().parents[1]

def actual():return build_geometry_spec(REVO3_COMPACT_USD,ROOT/'assets/rounded_die/object_newton.usda')

def test_actual_convex_geometry_and_mass():
    s=actual();v=np.asarray(s['convex_vertices']);p=np.asarray(s['convex_planes'])
    assert s['object_shape']=='convex_mesh' and len(v)==200 and len(s['point_names'])==30
    assert np.max(v@p[:,:3].T+p[:,3])<1e-7
    assert np.max(np.abs(np.ptp(v,axis=0)-.07))<1e-6
    assert abs(s['object_mass']-.06)<1e-8 and (np.linalg.eigvalsh(s['object_inertia'])>0).all()
    assert 'sphere_radius' not in s and 'box_half' not in s

def test_convex_margin_surface_and_gradient():
    s=actual();g=HandObjectGeometry(s).double()
    v=torch.tensor(s['convex_vertices'],dtype=torch.double)
    gap,_=g.surface(v)
    torch.testing.assert_close(gap,torch.zeros_like(gap),atol=2e-8,rtol=0)
    assert g.surface(torch.zeros(1,3,dtype=torch.double))[0].item()<-.03
    x=torch.tensor([[.06,.011,.004],[-.01,.002,.005]],dtype=torch.double,requires_grad=True)
    assert g.surface(x)[0][0]>0 and g.surface(x)[0][1]<0
    assert torch.autograd.gradcheck(lambda p:g.surface(p)[0],(x,),eps=1e-6,atol=1e-5)
    _,n=g.surface(x);torch.testing.assert_close(n.norm(dim=-1),torch.ones(2,dtype=torch.double),atol=1e-7,rtol=0)

def test_convex_feature_gradient_and_rotation_dependence():
    g=HandObjectGeometry(actual())
    s,_=fixture(2);s[:,POS]=s.new_tensor([.03,.01,.04]);s[:,ROT]=s.new_tensor([0.,0.,0.,1.])
    t=s.clone();t[:,Q]+=.1;t[:,POS]+=.01;t[:,VEL]+=.1;t[:,OMEGA]+=.2;t.requires_grad_()
    g.supervised_errors(t,s).mean().backward()
    assert torch.isfinite(t.grad).all()
    for field in (Q,POS,VEL,OMEGA):assert t.grad[:,field].abs().sum()>0
    assert g(t).shape==(2,300)
    a=g.readout(s)[4];s[:,ROT]=rotation_vector_quat(s.new_tensor([[.2,.4,.1]]).expand(2,-1))
    assert not torch.allclose(a,g.readout(s)[4])

def test_checkpoint_format_is_independent():
    from viserdex.supermimic.trainer import CHECKPOINT_VERSION
    assert CHECKPOINT_VERSION==17
    text=(ROOT/'source/viserdex/viserdex/supermimic/simulation.py').read_text()
    assert 'TASK_ID = "Isaac-ViserDex-Repose-Revo3-SuperMimic-YellowDie-WujiOffset-HandPose-Hold-v1"' in text
