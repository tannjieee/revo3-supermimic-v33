"""Version-26 semantic boundaries: ordering, physical limits, scaling and compatibility."""
from dataclasses import asdict, replace
import torch
from viserdex.supermimic.core import revised_manual_config, manual_convex_config, action_target
from viserdex.supermimic.networks import Actor, Dynamics, Critic
from viserdex.supermimic.revised_contract import JOINT_NAMES, ORDER, urdf_limits
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.trainer import Trainer, checkpoint_version
from viserdex.supermimic.inference import ManualConvexPolicy
from viserdex.distillation.config import PerceptionNoiseCfg


def fixture():
    cfg = revised_manual_config()
    s = torch.zeros(3,84)
    limits = torch.tensor(cfg.network_joint_limits,dtype=torch.float32)[None].expand(3,-1,-1)
    s[:,:21] = limits.mean(-1)+torch.tensor(cfg.encoder_bias_rad)
    s[:,45:49] = s[:,79:83] = torch.tensor([0.,0.,0.,1.])
    s[:,42:45] = s[:,76:79] = .1
    return cfg,s,limits


def test_joint_identity_and_urdf_endpoint_mapping():
    cfg,s,limits=fixture()
    assert JOINT_NAMES[0]=='right_little_MPR_joint' and JOINT_NAMES[16]=='right_thumb_CMP_joint'
    assert sorted(ORDER)==list(range(21))
    _,src=urdf_limits()
    geometric=torch.tensor(src['geometric_limits_rad'],dtype=torch.float32)
    for a,end in ((-1.,0),(1.,1)):
        target=action_target(s,torch.full((3,21),a),limits,cfg)
        torch.testing.assert_close(target+torch.tensor(cfg.encoder_bias_rad),geometric[:,end][None].expand(3,-1))
    assert cfg.encoder_bias_rad[16] > 0 and sum(x!=0 for x in cfg.encoder_bias_rad)==1


def test_network_shapes_and_removal_of_delay_and_geometry():
    cfg,s,limits=fixture();wm=Dynamics(65,cfg);critic=Critic(65,cfg);actor=Actor(cfg)
    assert actor.net[0].out_features==1024 and critic.net[0].out_features==512
    assert wm.net[0].in_features==164 and critic.net[0].in_features==152
    assert wm.geometry is None and critic.geometry is None
    context=torch.ones(3,65);target=limits.mean(-1)
    a,b=wm.inputs(s,target,context),critic.inputs(s,context)
    s[:,83]=1
    torch.testing.assert_close(a,wm.inputs(s,target,context),atol=0,rtol=0)
    torch.testing.assert_close(b,critic.inputs(s,context),atol=0,rtol=0)
    assert checkpoint_version(cfg)==26 and checkpoint_version(manual_convex_config())==25


def test_input_scaling_has_physical_meaning():
    cfg,s,limits=fixture();wm=Dynamics(65,cfg);critic=Critic(65,cfg)
    s[:,21:42]=10.;s[:,49:52]=.5;s[:,52:55]=10.
    w=wm.inputs(s,limits.mean(-1),torch.ones(3,65));v=critic.inputs(s,torch.ones(3,65))
    torch.testing.assert_close(w[:,:21],torch.zeros(3,21),atol=2e-7,rtol=0)
    torch.testing.assert_close(w[:,57:78],torch.zeros(3,21),atol=2e-7,rtol=0)
    for part in (w[:,21:45],w[:,51:57],v[:,78:81]):
        torch.testing.assert_close(part,torch.ones_like(part))
    actor=Actor(cfg);obs=torch.zeros(3,81);obs[:,63:69]=.1
    torch.testing.assert_close(actor.scaled_observation(obs)[:,63:69],torch.ones(3,6))


def test_statistics_use_same_scaled_inputs_as_forward():
    cfg,s,limits=fixture();cfg=replace(cfg,wm_hidden=(16,),actor_hidden=(16,),critic_hidden=(16,))
    trainer=Trainer(cfg,PerceptionNoiseCfg(),torch.ones(3,65),limits,torch.tensor([[0.,0.,0.,1.]]).expand(3,-1),'cpu')
    obs=torch.randn(3,81);trainer.update_real_statistics(s,limits.mean(-1),s[:,:55],obs)
    torch.testing.assert_close(trainer.actor.obs_norm.mean,trainer.actor.scaled_observation(obs).mean(0))


def test_exported_actor_joint_contract_and_inference(tmp_path):
    cfg,s,limits=fixture();cfg=replace(cfg,actor_hidden=(16,))
    payload=dict(version=26,config=asdict(cfg),actor=Actor(cfg).state_dict(),metadata=dict(
        state_protocol=cfg.state_protocol,observation_protocol=cfg.observation_protocol,
        observation_dim=81,quaternion_order='XYZW',joint_names=list(JOINT_NAMES),joint_limits_rad=limits[0].tolist()))
    path=tmp_path/'actor.pt';torch.save(payload,path)
    policy=ManualConvexPolicy(path)
    policy.set_goal([0,0,0,1],[.1,.1,.1])
    result=policy.propose(limits[0].mean(-1),torch.zeros(21),[.1,.1,.1],[0,0,0,1])
    assert result is not None


def test_legacy_environment_hash_ignores_only_equivalent_default():
    from copy import deepcopy
    from viserdex.distillation.environment import contract_sha256
    old = dict(sim={'physics': {}}, scene={'robot': {'spawn': {}}, 'object': {'spawn': {}}},
               observations={'policy': {}, 'privileged': {}}, commands={'object_pose': {}},
               actions={'joint_pos': {}})
    default = deepcopy(old)
    default['actions']['joint_pos']['limit_contract'] = 'legacy'
    assert contract_sha256(old) == contract_sha256(default)
    default['actions']['joint_pos']['limit_contract'] = 'urdf_little_v2'
    assert contract_sha256(old) != contract_sha256(default)
