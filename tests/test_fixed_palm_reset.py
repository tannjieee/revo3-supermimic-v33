"""Fixed-palm reset semantics and the serialized rollout environment contract."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('isaaclab')
from isaaclab.utils.io import dump_yaml
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.distillation.environment import SavedEnvironmentBinding, contract_sha256
from viserdex.supermimic.core import hand_center_config, sample_xy_goal
from viserdex.supermimic.fixed_palm_reset import (
    configure_fixed_palm_reset, reset_fixed_palm_v1, configured_pose_base,
    validate_fixed_palm_policy, POSITION_BASE_M, QUATERNION_BASE_XYZW,
)
from viserdex.supermimic.manual_convex import ManualConvexEnvCfg
from viserdex.supermimic.object_variant import bind_object
from viserdex.supermimic.revised_contract import configure_environment, joint_names
from viserdex.supermimic.support_placement import configure_supported_reset
from viserdex.supermimic.wuji_pose import HOME_POSITION


def environment():
    cfg = ManualConvexEnvCfg()
    # The official launcher resolves this preset before configure callbacks.
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    configure_environment(cfg, 'hardware_center25_v3')
    bind_object(cfg, 'die50')
    return cfg


def test_fixed_state_preserves_goal_and_startup_domain_distributions():
    cfg = environment()
    before = deepcopy(cfg.to_dict())
    configure_fixed_palm_reset(cfg)
    assert cfg.events.reset_robot_joints.func is reset_fixed_palm_v1
    assert tuple(cfg.events.reset_robot_joints.params['position_range']) == (0., 0.)
    assert tuple(cfg.events.reset_robot_joints.params['velocity_range']) == (0., 0.)
    assert set(cfg.events.reset_object.params['pose_range']) == {'x', 'y', 'z', 'roll', 'pitch', 'yaw'}
    assert all(tuple(v) == (0., 0.) for v in cfg.events.reset_object.params['pose_range'].values())
    assert cfg.events.reset_object.params['velocity_range'] == {}
    for name in ('robot', 'object'):
        state = getattr(cfg.scene, name).init_state
        assert tuple(state.lin_vel) == tuple(state.ang_vel) == (0., 0., 0.)
    assert cfg.scene.robot.init_state.joint_vel == {'.*': 0.}
    assert cfg.scene.robot.init_state.joint_pos == dict(zip(REVO3_JOINT_NAMES, HOME_POSITION))
    after = cfg.to_dict()
    for name, event in before['events'].items():
        if event is not None and event.get('mode') == 'startup':
            assert after['events'][name] == event
    assert after['commands'] == before['commands']
    assert after['actions'] == before['actions']
    assert after['scene']['object']['spawn'] == before['scene']['object']['spawn']


def test_fixed_reset_home_matches_actor_initialization_in_policy_joint_order():
    cfg = environment()
    configure_fixed_palm_reset(cfg)
    policy = hand_center_config()
    actual = [cfg.scene.robot.init_state.joint_pos[name] for name in joint_names(policy)]
    np.testing.assert_array_equal(actual, policy.action_home)
    # HOME stores geometric radians; Actor subtracts the identified encoder bias.
    encoder = np.asarray(actual) - np.asarray(policy.encoder_bias_rad)
    bounds = np.asarray(policy.network_joint_limits)
    assert np.all((encoder > bounds[:, 0]) & (encoder < bounds[:, 1]))


@pytest.mark.parametrize('fixed', [False, True])
def test_saved_environment_roundtrip_keeps_the_selected_reset_protocol(tmp_path, fixed):
    cfg = environment()
    (configure_fixed_palm_reset if fixed else configure_supported_reset)(cfg)
    path = tmp_path / 'environment.yaml'
    dump_yaml(str(path), cfg)
    binding = SavedEnvironmentBinding(path)
    # Resume and video instantiate the base task, then restore the saved data.
    # This catches new event-param keys missing from that original typed schema.
    restored = environment()
    args = SimpleNamespace(num_envs=1, device='cpu', preset_overrides=[])
    binding.configure(restored, args)
    assert contract_sha256(restored.to_dict()) == binding.contract_sha
    assert str(restored.events.reset_robot_joints.func) == binding.saved['events']['reset_robot_joints']['func']
    if fixed:
        # train.configure's early grasp-bank guard must recognize saved callables.
        assert restored.events.reset_robot_joints.func in (
            reset_fixed_palm_v1, 'viserdex.supermimic.fixed_palm_reset:reset_fixed_palm_v1')
    assert restored.events.reset_robot_joints.params['position_range'] == cfg.events.reset_robot_joints.params['position_range']
    assert restored.scene.object.init_state.to_dict() == cfg.scene.object.init_state.to_dict()


def test_fixed_and_random_contact_resets_have_distinct_saved_contracts():
    legacy, fixed = environment(), environment()
    configure_supported_reset(legacy)
    original = deepcopy(legacy.to_dict())
    configure_fixed_palm_reset(fixed)
    assert contract_sha256(original) != contract_sha256(fixed.to_dict())
    assert legacy.to_dict() == original


def test_fixed_pose_rejects_unvalidated_70mm_die_before_mutating_reset():
    cfg = environment()
    bind_object(cfg, 'die70')
    before = deepcopy(cfg.to_dict())
    with pytest.raises(ValueError, match='50 mm'):
        configure_fixed_palm_reset(cfg)
    assert cfg.to_dict() == before


@pytest.mark.parametrize('position_range,velocity_range', [((-.03, .03), (0., 0.)), ((0., 0.), (-.1, .1))])
def test_runtime_rejects_jitter_before_touching_the_scene(position_range, velocity_range):
    with pytest.raises(ValueError, match='jitter'):
        reset_fixed_palm_v1(SimpleNamespace(), None, position_range, velocity_range, None)


def test_runtime_rejects_a_grasp_bank_override_before_touching_the_scene():
    with pytest.raises(ValueError, match='grasp bank'):
        reset_fixed_palm_v1(SimpleNamespace(grasp_reset_manager=object()), None, (0., 0.), (0., 0.), None)


def test_goal_sampler_remains_random_for_identical_initial_object_states():
    import torch
    # Reset's fixed object pose is not the command. The existing command samples
    # goals independently from a reference tensor's shape/dtype/device only.
    reference = torch.tensor([0., 0., 0., 1.]).expand(32, -1).clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        first = sample_xy_goal(reference)
        second = sample_xy_goal(reference)
    assert len(torch.unique(first, dim=0)) == 32
    assert not torch.equal(first, second)
    torch.testing.assert_close(first.norm(dim=-1), torch.ones(32), atol=2e-7, rtol=0)


@pytest.mark.parametrize('revision', ['legacy', 'urdf_little_v2'])
def test_fixed_profile_rejects_other_action_contracts_before_mutation(revision):
    cfg = environment()
    cfg.actions.joint_pos.limit_contract = revision
    before = deepcopy(cfg.to_dict())
    with pytest.raises(ValueError, match='hardware_center25_v3'):
        configure_fixed_palm_reset(cfg)
    assert cfg.to_dict() == before


@pytest.mark.parametrize('quaternion', [
    (0., 2**-.5, 0., 2**-.5),  # palm facing down
    (0., 0., 0., 1.),          # palm facing sideways
    (0., 0., 0., 0.),          # invalid quaternion
    (0., -2**.5, 0., 2**.5),   # non-unit quaternion
])
def test_fixed_profile_rejects_non_upright_or_invalid_root_rotation(quaternion):
    cfg = environment()
    cfg.scene.robot.init_state.rot = quaternion
    before = deepcopy(cfg.to_dict())
    with pytest.raises(ValueError, match='gravity|quaternion'):
        configure_fixed_palm_reset(cfg)
    assert cfg.to_dict() == before


@pytest.mark.parametrize('gravity', [
    (0., 0., 0.), (0., 0., float('nan')), (0., 0., float('inf')),
    (0., 0.), (0., 0., 9.81), (0., -9.81, 0.),
])
def test_fixed_profile_rejects_invalid_or_misaligned_gravity(gravity):
    cfg = environment()
    cfg.sim.gravity = gravity
    object_before = cfg.scene.object.init_state.to_dict()
    with pytest.raises(ValueError, match='gravity'):
        configure_fixed_palm_reset(cfg)
    assert cfg.scene.object.init_state.to_dict() == object_before


@pytest.mark.parametrize('yaw', [-2.3, 0.7, 3.14])
def test_fixed_profile_allows_yaw_and_global_translation(yaw):
    from scipy.spatial.transform import Rotation
    cfg = environment()
    cfg.scene.robot.init_state.rot = tuple((Rotation.from_euler('z', yaw)
                                          * Rotation.from_quat(cfg.scene.robot.init_state.rot)).as_quat())
    cfg.scene.robot.init_state.pos = (3.2, -5.4, 0.8)
    configure_fixed_palm_reset(cfg)
    position, quaternion = configured_pose_base(cfg)
    np.testing.assert_allclose(position, POSITION_BASE_M, atol=1e-14, rtol=0)
    assert abs(float(np.dot(quaternion, QUATERNION_BASE_XYZW))) > 1-1e-14


def test_runtime_rejects_selected_live_frame_before_any_state_write(monkeypatch):
    import torch
    import isaaclab.envs.mdp.events as events
    cfg = environment()
    configure_fixed_palm_reset(cfg)
    upright = torch.tensor(cfg.scene.robot.init_state.rot)
    roots = upright.expand(3, -1).clone()
    roots[1] = torch.tensor([0., 2**-.5, 0., 2**-.5])
    env = SimpleNamespace(cfg=cfg, num_envs=3, device='cpu', scene={
        'robot': SimpleNamespace(data=SimpleNamespace(root_quat_w=roots)), 'object': object()})
    writes = []
    monkeypatch.setattr(events, 'reset_joints_by_offset', lambda *args, **kwargs: writes.append(True))
    with pytest.raises(ValueError, match='face upward'):
        reset_fixed_palm_v1(env, torch.tensor([1], dtype=torch.int32), (0., 0.), (0., 0.),
                            SimpleNamespace(name='robot'))
    assert writes == []


@pytest.mark.parametrize('revision,optimized', [('legacy', True), ('urdf_little_v2', True),
                                               ('hardware_center25_v3', False)])
def test_fixed_policy_rejects_missing_25hz_home_initialization(revision, optimized):
    with pytest.raises(ValueError, match='HOME initialization'):
        validate_fixed_palm_policy(SimpleNamespace(network_revision=revision,
                                                   absolute_policy_optimization=optimized))


def test_fixed_policy_accepts_current_home_initialized_profile():
    cfg = hand_center_config(absolute_policy_optimization=True, stable_value_learning=True,
                             wm_task_losses=True, include_delay_ready=True)
    validate_fixed_palm_policy(cfg)


def test_58mm_fixed_pose_uses_live_manual_hulls_and_retains_home():
    from viserdex.supermimic.fixed_palm_reset import fixed_pose_for_asset
    from viserdex.supermimic.geometry_asset import build_geometry_spec
    from viserdex.supermimic.support_placement import SupportPlacement
    cfg=environment();bind_object(cfg,'die58');configure_fixed_palm_reset(cfg)
    position,rotation,palm_min,total_min=fixed_pose_for_asset(cfg)
    actual_p,actual_q=configured_pose_base(cfg)
    np.testing.assert_allclose(actual_p,position,atol=1e-14)
    assert abs(np.dot(actual_q,rotation))>1-1e-14
    assert cfg.scene.robot.init_state.joint_pos==dict(zip(REVO3_JOINT_NAMES,HOME_POSITION))
    rc,oc=cfg.scene.robot,cfg.scene.object
    query=SupportPlacement(rc.spawn.usd_path,build_geometry_spec(rc.spawn.usd_path,oc.spawn.usd_path,rc.spawn.scale,oc.spawn.scale))
    query.set_pose(HOME_POSITION,position,rotation);gaps=query.all_gaps()
    palm=np.array([query.model.body(query.model.geom_bodyid[g]).name=='query_world' for g in query.hand_geoms])
    assert -5e-6<=gaps.min()<=1e-5
    assert sum(gaps[palm]<=1e-5)>=palm_min and sum(gaps<=1e-5)>=total_min
