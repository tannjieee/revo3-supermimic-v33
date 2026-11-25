"""Hard COM boundary parity, quaternion/frame conversion and legacy loading."""
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from viserdex.supermimic.core import Config, hand_center_config
from viserdex.supermimic.position_boundary import (FIELDS, load_boundary, configure_fresh_boundary,
    policy_boundary_options, prepare_saved_boundary, verify_policy_boundary, verify_geometry_com)
from viserdex.supermimic.spatial_contract import drop_distance, object_outside_com_ellipsoid, object_com_position
from viserdex.supermimic.trainer import Trainer, imagined_continuation
from viserdex.supermimic.constraints import violation_costs

COM = (-.00014158611884340644, -.000024407045202679, -.00015451501531060785)


def config(**kw):
    p = load_boundary()
    return hand_center_config(wm_task_losses=True, stable_value_learning=True,
        drop_boundary_mode='com_ellipsoid', drop_ellipsoid_center_m=p['center'],
        drop_ellipsoid_semi_axes_m=p['semi_axes'], drop_object_com_local_m=COM,
        drop_boundary_sha256=p['boundary_sha256'], **kw)


def samples(cfg):
    # Every axis, both signs, 0.1% inside/outside, with nontrivial object rotation.
    rotations = Rotation.from_euler('xyz', np.arange(36).reshape(12,3)*.17)
    xyz = []
    for axis in range(3):
        for sign in (-1,1):
            for radius in (.999,1.001):
                p = np.array(cfg.drop_ellipsoid_center_m)
                p[axis] += sign*radius*cfg.drop_ellipsoid_semi_axes_m[axis]
                xyz.append(p)
    root = np.array(xyz)-rotations.apply(np.tile(COM,(12,1)))
    state = torch.zeros(12,55,dtype=torch.float64)
    state[:,42:45] = torch.tensor(root)
    state[:,45:49] = torch.tensor(rotations.as_quat())
    return state, rotations, torch.tensor([False,True]*6)


@pytest.mark.parametrize('wrapped',[False,True])
def test_real_callback_matches_predicted_com_boundary_in_rotated_translated_base(wrapped):
    cfg = config(); physical, rotations, expected = samples(cfg)
    base = Rotation.from_euler('xyz',[.9,-.7,1.2])
    origin = np.array([3.2,-4.1,2.3])
    root_world = base.apply(physical[:,42:45].numpy())+origin
    wrap = (lambda v:NS(torch=v)) if wrapped else (lambda v:v)
    env = NS(scene={'robot':NS(data=NS(root_pos_w=wrap(torch.tensor(np.tile(origin,(12,1)))),
        root_quat_w=wrap(torch.tensor(np.tile(base.as_quat(),(12,1)))))),
        'object':NS(data=NS(root_pos_w=wrap(torch.tensor(root_world)),
        root_quat_w=wrap(torch.tensor((base*rotations).as_quat()))))})
    actual = object_outside_com_ellipsoid(env,cfg.drop_ellipsoid_center_m,cfg.drop_ellipsoid_semi_axes_m,COM)
    assert torch.equal(actual,expected)
    assert torch.equal(drop_distance(physical,cfg)>cfg.drop_radius_m,expected)


def test_nonzero_com_rotates_with_object_and_root_is_not_boundary_reference():
    cfg=config(); angle=math.pi/2
    q=torch.tensor([[0.,0.,math.sin(angle/2),math.cos(angle/2)]],dtype=torch.float64)
    root=torch.tensor([cfg.drop_ellipsoid_center_m],dtype=torch.float64)
    expected=root+torch.tensor([[-COM[1],COM[0],COM[2]]])
    torch.testing.assert_close(object_com_position(root,q,COM),expected,rtol=1e-7,atol=1e-12)
    # Put the root just INSIDE the +X surface, but COM just OUTSIDE it.
    root[:,0]+=cfg.drop_ellipsoid_semi_axes_m[0]-1e-5
    physical=torch.zeros(1,55,dtype=torch.float64);physical[:,42:45]=root;physical[:,45:49]=q
    assert drop_distance(physical,cfg).item()>cfg.drop_radius_m


@pytest.mark.parametrize('stable',[False,True])
def test_learned_and_analytic_continuation_zero_outside(stable):
    cfg=replace(config(),stable_value_learning=stable)
    physical,_,outside=samples(cfg)
    trainer=Trainer.__new__(Trainer);trainer.cfg=cfg
    trainer.wm=NS(termination_probability=lambda x:torch.full(x.shape[:-1],.2,dtype=x.dtype))
    continuation=trainer.continuation(physical,physical)
    torch.testing.assert_close(continuation,torch.where(outside,0.,.8).double(),rtol=1e-6,atol=1e-7)
    analytic=imagined_continuation(physical,physical,.01,True,cfg)
    assert torch.equal(analytic[outside],torch.zeros(6,dtype=analytic.dtype))
    assert (analytic[~outside]>0).all()


def test_drop_cost_uses_same_rho_and_has_finite_gradients():
    cfg=config();physical,_,_=samples(cfg);physical.requires_grad_()
    state=torch.zeros(12,cfg.state_dim,dtype=torch.float64)
    state[:,:21]=torch.tensor(cfg.action_home);state[:,45:49]=state[:,79:83]=torch.tensor([0.,0.,0.,1.])
    limits=torch.tensor(cfg.network_joint_limits,dtype=torch.float64).expand(12,-1,-1)
    costs,ratios=violation_costs(state,physical,torch.zeros(12,21,dtype=torch.float64),limits,cfg,
        hand_center=torch.tensor([100.,200.,300.]))
    expected=torch.tensor([.999,1.001]*6,dtype=torch.float64)
    torch.testing.assert_close(ratios[:,-1],expected,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(costs[:,-1],(expected-1).clamp_min(0).square())
    costs[:,-1].sum().backward()
    assert torch.isfinite(physical.grad).all() and physical.grad[:,42:45].abs().sum()>0


def test_old_serialized_configs_keep_root_sphere():
    original=hand_center_config(); saved=asdict(original)
    for field in FIELDS:saved.pop(field)
    cfg=Config(**saved)
    p=torch.zeros(2,55);p[:,42:45]=torch.tensor(cfg.hand_center_base_m);p[1,42]+=.31
    assert cfg.drop_boundary_mode=='root_sphere'
    torch.testing.assert_close(drop_distance(p,cfg),torch.tensor([0.,.31]))


@pytest.mark.parametrize('bad',[(0.,.1,.2),(-.1,.1,.2),(float('nan'),.1,.2)])
def test_invalid_axes_rejected(bad):
    with pytest.raises(ValueError,match='semi_axes'):
        replace(config(),drop_ellipsoid_semi_axes_m=bad)


def test_fresh_asset_binding_and_saved_schema_roundtrip():
    from viserdex.supermimic.object_variant import VARIANTS
    from isaaclab.managers import TerminationTermCfg
    def env():return NS(scene=NS(object=NS(spawn=NS(usd_path=str(VARIANTS['die58']),scale=None))),
                        terminations=NS(object_out_of_reach=None))
    fresh=env();configure_fresh_boundary(fresh)
    options=policy_boundary_options(fresh)
    assert options['drop_object_com_local_m']==pytest.approx(COM,abs=1e-12)
    cfg=replace(hand_center_config(),**options);verify_policy_boundary(cfg,fresh)
    term=fresh.terminations.object_out_of_reach
    assert term.time_out is False
    saved={'terminations':{'object_out_of_reach':term.to_dict()}}
    restored=env();prepare_saved_boundary(restored,saved)
    restored.terminations.object_out_of_reach.from_dict(saved['terminations']['object_out_of_reach'])
    verify_policy_boundary(cfg,restored)
    assert policy_boundary_options(restored)==options
    with pytest.raises(ValueError,match='differ'):
        verify_policy_boundary(replace(cfg,drop_ellipsoid_center_m=(0.,0.,0.)),restored)
    with pytest.raises(ValueError,match='COM differs'):
        verify_geometry_com(cfg,{'object_com':[0.,0.,0.]})
    verify_geometry_com(cfg,{'object_com':list(COM)})


def test_boundary_parameters_are_the_user_selected_values():
    p=load_boundary()
    assert p['center']==(.0337,.009358916334299222,.10065586490765209)
    assert p['semi_axes']==(.1061,.112,.1521)
