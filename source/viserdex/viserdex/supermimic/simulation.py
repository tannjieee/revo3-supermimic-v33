"""Official Isaac Lab recorder/manager integration, isolated from original tasks.

Only Newton material/gain context inspection uses backend-specific read access,
following the same bindings as the pinned official randomization event classes.
No simulator step, randomizer, asset, or collision implementation is replaced.
"""

import torch
from copy import deepcopy
from dataclasses import fields
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.managers import RecorderTermCfg, RewardTermCfg
from .action_smoothness import ActionSecondDifference
from .hold_command import HoldReorientCommandCfg, hold_success_bonus
from .wuji_actions import WujiHomeOffsetActionCfg
from .wuji_pose import HOME_POSITION
from .hand_pose import hand_pose_penalty
from .goal_hold import required_steps
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManagerBaseCfg, DatasetExportMode
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES, REVO3_COMPACT_USD
from viserdex.tasks.manipulation.inhand.config.revo3_hand.revo3_env_cfg import ViserDexRevo3EnvCfg
from viserdex.distillation.config import StudentCfg, PerceptionNoiseCfg
from viserdex.distillation.observations import StudentObservations
from .core import *
from .sensors import initial_sensors, reset_sensors, observe
from .validation import audit_real_reward
from .geometry_asset import build_geometry_spec
from .geometry import HandObjectGeometry
from .position_reward import POSITION_CONFIG_FIELDS, position_hold_valid
from .stability_reward import STABILITY_CONFIG_FIELDS
from .quiet_hold import QUIET_HOLD_CONFIG_FIELDS, DEX_REG_CONFIG_FIELDS, validate_episode_timing
from .state_layout import COMPACT, LEGACY, REFERENCE
from .goal_hold import hold_transition, persistent_hold_transition, hold_kwargs

TASK_ID = "Isaac-ViserDex-Repose-Revo3-SuperMimic-YellowDie-WujiOffset-HandPose-Hold-v1"


class TransitionRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        manager = getattr(self._env, "grasp_reset_manager", None)
        if manager is not None:
            manager.reset(env_ids)
        return None, None

    def record_post_step(self):
        adapter = getattr(self._env, "supermimic_adapter", None)
        if adapter is not None:
            adapter.captured_physical = adapter.physical().clone()
            if adapter.hold_cfg.dex_reference:
                adapter.captured_contacts = adapter.contact_labels.read()
            # This recorder runs before automatic resets. Reject grossly invalid
            # mechanical states here, before they can be hidden by a reset or
            # recorded as a training transition.
            if hasattr(adapter, 'physics_health'):
                adapter.capture_physics_health_post_step()
            adapter.captured_hand_pose = hand_pose_penalty(self._env).clone()
            if not adapter.hold_cfg.compact_state:
                adapter.captured_second_diff = self._env.reward_manager.get_term_cfg("action_second_diff_l2").func.last_penalty.clone()
            if adapter.geometry_audit_steps < 8 or adapter.geometry_audit_steps % 500 == 0:
                adapter.audit_geometry(adapter.captured_physical)
            adapter.geometry_audit_steps += 1
            command = self._env.command_manager.get_term('object_pose')
            adapter.captured_completion = command.just_completed.clone()
            adapter.captured_hold = command.hold_ticks.clone()
            adapter.captured_angular_speed = command.last_speed.clone()
            if adapter.hold_cfg.goal_mode == 'single_dense_hold':
                adapter.captured_goal_elapsed = command.elapsed_seconds.clone()
                adapter.captured_position_error = command.last_position_error.clone()
                adapter.captured_linear_speed = command.last_linear_speed.clone()
            adapter.captured_target = self._env.action_manager.get_term("joint_pos").processed_actions.clone()
            # Match the official reward's frame and arithmetic before reset or
            # goal resampling. Do not read a stale command metric here.
            adapter.captured_orientation_error = quat_error_magnitude(
                tensor(adapter.obj.data.root_quat_w),
                self._env.command_manager.get_term("object_pose").quat_command_w,
            ).clone()
            manager = getattr(self._env, "grasp_reset_manager", None)
            if manager is not None:
                manager.post_step(self._env.reset_buf, adapter.captured_completion)
        return None, None


@configclass
class TransitionRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    transition = RecorderTermCfg(class_type=TransitionRecorder)


@configclass
class SuperMimicEnvCfg(ViserDexRevo3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(REVO3_COMPACT_USD)
        self.scene.robot.actuators['fingers'].stiffness = 3.0
        self.scene.robot.init_state.joint_pos = dict(zip(REVO3_JOINT_NAMES, HOME_POSITION))
        self.actions.joint_pos = WujiHomeOffsetActionCfg(
            asset_name="robot", joint_names=list(REVO3_JOINT_NAMES), preserve_order=True,
            scale=1.0, alpha=.5, offset_scale=.5, warmup_time_s=.4, rescale_to_limits=False,
        )
        from viserdex.assets.objects import _OBJECT_CFGS
        self.scene.object.spawn.usd_path = _OBJECT_CFGS["RoundedDie"]["usd_path"]
        self.scene.object.init_state.pos = (0.02, -0.01, 0.565)
        self.scene.num_envs = 4096
        self.sim.dt = 1 / 120
        self.decimation = 6
        self.sim.render_interval = 6
        self.recorders = TransitionRecorderCfg()
        settings = config20()
        old_command = self.commands.object_pose
        # Initialize required fields with their typed values (from_dict cannot
        # replace dataclasses.MISSING, and serialized marker configs lose types).
        self.commands.object_pose = HoldReorientCommandCfg(**{
            field.name: deepcopy(getattr(old_command, field.name))
            for field in fields(old_command) if field.name != 'class_type'
        })
        self.commands.object_pose.orientation_success_threshold = settings.hold_angle_threshold
        self.commands.object_pose.hold_speed_threshold = settings.hold_speed_threshold
        self.commands.object_pose.hold_duration = settings.hold_duration
        self.rewards.success_bonus.func = hold_success_bonus
        self.rewards.success_bonus.params = {'command_name': 'object_pose'}
        self.actions.joint_pos.alpha = settings.ema_alpha
        self.rewards.success_bonus.weight = settings.success_bonus
        self.rewards.action_rate_l2.weight = -settings.action_rate_weight
        self.rewards.joint_vel_l2.weight = -settings.joint_vel_weight
        self.rewards.action_l2.weight = -settings.action_l2_weight
        self.rewards.action_second_diff_l2 = RewardTermCfg(
            func=ActionSecondDifference, weight=-settings.action_second_diff_weight
        )
        self.rewards.hand_pose = RewardTermCfg(func=hand_pose_penalty, weight=-settings.hand_pose_weight)
        # Solver capacities are applied by the official preset / saved binding.


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:SuperMimicEnvCfg"},
    )


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp

    return wp.to_torch(value.contiguous())


def collision_shape_mask(asset, model, num_envs):
    """The official renderer also imports non-colliding visual-only shapes."""
    from newton import ShapeFlags

    flags = tensor(asset._root_view.get_attribute("shape_flags", model)).reshape(num_envs, -1)
    mask = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    if not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError("Collision shape layout differs across replicated environments")
    return mask


class Adapter:
    def __init__(self, env, noise_cfg, history_length=10, policy_cfg=None):
        self.env, self.noise_cfg = env, noise_cfg
        command_cfg = env.command_manager.get_term('object_pose').cfg
        self.hold_cfg = policy_cfg or config20()
        if (command_cfg.orientation_success_threshold != self.hold_cfg.hold_angle_threshold
                or command_cfg.hold_speed_threshold != self.hold_cfg.hold_speed_threshold
                or command_cfg.hold_duration != self.hold_cfg.hold_duration
                or command_cfg.update_goal_on_success != (self.hold_cfg.goal_mode == 'resample_on_success')):
            raise ValueError('Hold success protocol differs from the task contract')
        if self.hold_cfg.goal_mode == 'single_dense_hold':
            for name in ('goal_evaluation_s', *STABILITY_CONFIG_FIELDS, *QUIET_HOLD_CONFIG_FIELDS,
                         'orientation_reward_scale', 'orientation_coarse_sigma_rad',
                         'orientation_fine_sigma_rad', 'orientation_coarse_fraction', *POSITION_CONFIG_FIELDS):
                if getattr(command_cfg, name) != getattr(self.hold_cfg, name):
                    raise ValueError(f'Single-goal reward/command mismatch: {name}')
            if getattr(command_cfg, 'orientation_reward_mode', 'gaussian') != self.hold_cfg.orientation_reward_mode:
                raise ValueError('Single-goal reward/command mismatch: orientation_reward_mode')
            if getattr(command_cfg, 'sequential_goals', False) != self.hold_cfg.sequential_goals:
                raise ValueError('Single-goal reward/command mismatch: sequential_goals')
            for name in DEX_REG_CONFIG_FIELDS:
                if getattr(command_cfg, name, getattr(self.hold_cfg, name)) != getattr(self.hold_cfg, name):
                    raise ValueError(f'Single-goal reward/command mismatch: {name}')
            if env.cfg.rewards.track_orientation_inv_l2 is not None or env.cfg.rewards.track_orientation_exp.weight != 1.:
                raise ValueError('Single-goal task requires the exponential orientation reward contract')
            if any(getattr(env.cfg.rewards, name) is not None for name in ('action_l2', 'hand_pose', 'success_bonus')):
                raise ValueError('Removed single-goal reward terms are still configured')
            if env.cfg.terminations.max_consecutive_success is not None:
                raise ValueError('Single goal must not terminate on success')
            if env.cfg.rewards.object_ang_vel_l2 is not None or env.cfg.rewards.object_lin_vel_l2 is not None:
                raise ValueError('Dense holding uses positive stability reward, not object velocity penalties')
            if env.cfg.rewards.track_position_exp.weight != 1. or env.cfg.rewards.stable_pose_exp.weight != 1.:
                raise ValueError('Dense position/stability reward differs from policy config')
        self.robot, self.obj = env.scene["robot"], env.scene["object"]
        from .revised_contract import joint_names
        self.joint_names = joint_names(self.hold_cfg)
        ids, names = self.robot.find_joints(list(self.joint_names), preserve_order=True)
        if list(names) != list(self.joint_names):
            raise ValueError("Joint order changed")
        self.ids = ids
        self.limits = tensor(self.robot.data.soft_joint_pos_limits)[:, ids].clone()
        if self.hold_cfg.action_mode == 'direct_limits':
            self.limits = env.action_manager.get_term('joint_pos').position_limits.clone()
        self.base_inverse = quat_conjugate(tensor(self.robot.data.root_quat_w)).clone()
        self.sl = REFERENCE if self.hold_cfg.dex_reference else COMPACT if self.hold_cfg.compact_state else LEGACY
        if self.hold_cfg.dex_reference:
            self.wrench = dex.sample_wrench(self.limits[..., 0], self.hold_cfg)
        self.delay_ready = None if self.hold_cfg.dex_reference else torch.zeros(env.num_envs, 1, device=env.device)
        self.raw_history = torch.zeros(env.num_envs, 21 if self.hold_cfg.compact_state else 84, device=env.device)
        self.ema = None if self.hold_cfg.compact_state else tensor(self.robot.data.default_joint_pos)[:, ids].clone()
        self.sensors = initial_sensors(env.num_envs, env.device, noise_cfg)
        self.sensors = reset_sensors(
            self.sensors, torch.ones(env.num_envs, device=env.device, dtype=torch.bool), noise_cfg
        )
        self.state = self.read_state()
        obs, self.sensors = observe(self.state, self.sensors, self.limits, noise_cfg, self.hold_cfg)
        self.history = obs[:, None].expand(-1, history_length, -1).clone()
        self.context, self.context_schema = self.read_context()
        self.captured_physical = None
        self.captured_target = None
        self.captured_second_diff = None
        self.captured_hand_pose = None
        self.captured_orientation_error = None
        self.max_errors = dict(reward=0.0, ema=0.0, clean_student_observation=0.0)
        self.geometry_spec = build_geometry_spec(
            env.cfg.scene.robot.spawn.usd_path, env.cfg.scene.object.spawn.usd_path,
            getattr(env.cfg.scene.robot.spawn, "scale", None), getattr(env.cfg.scene.object.spawn, "scale", None),
        )
        if self.geometry_spec["object_shape"] != "convex_mesh":
            raise ValueError("RoundedDie requires its authored convex mesh; sphere checkpoints are incompatible")
        from .position_boundary import verify_geometry_com
        verify_geometry_com(self.hold_cfg, self.geometry_spec)
        if self.hold_cfg.network_revision != 'legacy':
            # Keep body traversal stable; only remap FK's joint-vector indices.
            remap = [self.joint_names.index(name) for name in REVO3_JOINT_NAMES]
            for level in self.geometry_spec['levels']:
                level['joint'] = [remap[i] for i in level['joint']]
        self.geometry = HandObjectGeometry(self.geometry_spec).to(env.device)
        from .convex_audit import audit_live_convex
        self.convex_audit = audit_live_convex(self.obj, env, self.geometry_spec)
        self.max_errors.update(self.convex_audit["errors"])
        self.geometry_body_ids, found = self.robot.find_bodies(self.geometry.body_names, preserve_order=True)
        if list(found) != self.geometry.body_names:
            raise ValueError("Geometry body ordering differs from simulator")
        self.geometry_audit_steps = 0
        self.audit_geometry(self.state)
        self.reset_count = 0
        if self.hold_cfg.compact_state and not self.hold_cfg.dex_reference:
            from .context import audit_delay_memory
            audit_delay_memory(self.delay_ready)
        env.supermimic_adapter = self
        validate_episode_timing(env.step_dt, env.max_episode_length, compact=self.hold_cfg.compact_state)
        action_cfg = env.action_manager.get_term("joint_pos").cfg
        self.action_alpha = getattr(action_cfg, 'alpha', 1.0)
        if self.hold_cfg.dex_reference:
            max_delta = env.action_manager.get_term('joint_pos').target_max_delta
            if abs(max_delta - self.hold_cfg.target_slew_rate_rad_s * env.step_dt) > 1e-9:
                raise ValueError('Executed target slew limit differs from policy configuration')
        self.success_bonus = env.cfg.rewards.success_bonus.weight if env.cfg.rewards.success_bonus else 0.
        self.action_rate_weight = -env.cfg.rewards.action_rate_l2.weight
        self.joint_vel_weight = -env.cfg.rewards.joint_vel_l2.weight
        self.action_l2_weight = -env.cfg.rewards.action_l2.weight if env.cfg.rewards.action_l2 else 0.
        self.action_second_diff_weight = (-env.cfg.rewards.action_second_diff_l2.weight
                                          if env.cfg.rewards.action_second_diff_l2 else 0.)
        torch.testing.assert_close(tensor(self.robot.data.default_joint_pos)[:, ids],
            self.limits.new_tensor(self.hold_cfg.action_home).expand(env.num_envs, -1), atol=1e-7, rtol=0)
        if self.hold_cfg.action_mode == 'direct_limits':
            from .direct_actions import DirectJointPositionAction
            from viserdex.assets.prepare_manual_convex import inspect_manual_asset
            if not isinstance(env.action_manager.get_term('joint_pos'), DirectJointPositionAction):
                raise ValueError('Direct policy requires the direct absolute-position action term')
            asset = inspect_manual_asset(env.cfg.scene.robot.spawn.usd_path)
            from .training_limits import load_training_limits
            if self.hold_cfg.network_revision != 'legacy':
                from .revised_contract import urdf_limits, ORDER
                software_limits, self.training_limit_source = urdf_limits(self.hold_cfg.network_revision)
                if self.hold_cfg.dex_reference:
                    software_limits, mechanical = dex.validate_profile(self.hold_cfg.control_profile)
                    self.training_limit_source = dict(mechanical, schema='revo3_ema_soft_limits_v33',
                        sha256=self.hold_cfg.control_profile_sha256, profile=self.hold_cfg.control_profile,
                        encoder_bias_rad=[0.]*21, conversion='encoder = geometry + episode zero; target_geometry = EMA_encoder - zero')
                ORDER = [REVO3_JOINT_NAMES.index(n) for n in self.joint_names]
                asset['limits_rad'] = [asset['limits_rad'][i] for i in ORDER]
                asset['joint_names'] = list(self.joint_names)
                torch.testing.assert_close(self.limits.new_tensor(asset['limits_rad']),
                    self.limits.new_tensor(self.training_limit_source['geometric_limits_rad']), atol=2e-7, rtol=0)
                torch.testing.assert_close(self.limits.new_tensor(self.hold_cfg.network_joint_limits),
                    self.limits.new_tensor(software_limits), atol=2e-7, rtol=0)
            else:
                software_limits, self.training_limit_source = load_training_limits()
            self.sdk_limit_source = self.training_limit_source['sdk_software_limits']
            torch.testing.assert_close(self.limits, self.limits.new_tensor(software_limits)[None].expand_as(self.limits), atol=2e-7, rtol=0)
            torch.testing.assert_close(tensor(self.robot.data.soft_joint_pos_limits)[:, ids],
                self.limits.new_tensor(asset['limits_rad'])[None].expand_as(self.limits), atol=2e-7, rtol=0)
            if env.cfg.scene.robot.soft_joint_pos_limit_factor != 1.0:
                raise ValueError('Direct policy requires full authored joint limits')
            torch.testing.assert_close(self.limits.new_tensor(action_cfg.encoder_bias_rad),
                                       self.limits.new_tensor(self.hold_cfg.encoder_bias_rad))
            self.manual_asset_audit = asset
            self.manual_asset_audit['sdk_software_limits'] = self.sdk_limit_source
            self.manual_asset_audit['training_soft_limits'] = self.training_limit_source
            self.manual_asset_audit['policy_encoder_limits_rad'] = software_limits.tolist()
            from .convex_audit import audit_live_hand_convex
            self.manual_asset_audit['live_convex'] = audit_live_hand_convex(self.robot, env, env.cfg.scene.robot.spawn.usd_path)
            self.max_errors['hand_convex_vertices_m'] = self.manual_asset_audit['live_convex']['max_vertex_error_m']
            self.contact_peaks = dict(candidates=0, contacts=0, constraints=0)
            from .physics_health import PhysicsHealthMonitor
            self.physics_health = PhysicsHealthMonitor(
                self.limits.new_tensor(asset['limits_rad']), self.joint_names, env.step_dt)
            self.physics_health_contract = self.physics_health.manifest()
            # Collider topology is independent of whether its material is random
            # (and therefore present in xi) or a fixed manifest value.
            count = int(collision_shape_mask(self.robot, env.sim.physics_manager.get_model(), env.num_envs)[0].sum())
            if count != asset['collision_instances']:
                raise ValueError(f'Live hand has {count} colliders, expected the 144 authored convex pieces')
        elif action_cfg.rescale_to_limits or action_cfg.offset_scale != self.hold_cfg.action_offset_scale or action_cfg.warmup_time_s != self.hold_cfg.action_warmup_s:
            raise ValueError("Wuji home-offset action contract changed")
        hand_pose_weight = env.cfg.rewards.hand_pose.weight if env.cfg.rewards.hand_pose else 0.
        if abs(hand_pose_weight + self.hold_cfg.hand_pose_weight) > 1e-9:
            raise ValueError("Hand-pose reward weight differs between simulator and imagination")
        if self.hold_cfg.dex_reference:
            from .contact_prediction import LiveObjectContacts
            self.contact_labels = LiveObjectContacts(env, self.hold_cfg.contact_body_names)
        expected_alpha = self.hold_cfg.ema_alpha
        if abs(self.action_alpha - expected_alpha) > 1e-8 or action_cfg.scale != 1.0 or action_cfg.clip is not None:
            raise ValueError("Action contract changed")

    def physical(self):
        rb, ob = self.robot.data, self.obj.data
        return torch.cat(
            (
                tensor(rb.joint_pos)[:, self.ids],
                tensor(rb.joint_vel)[:, self.ids],
                quat_apply(self.base_inverse, tensor(ob.root_pos_w) - tensor(rb.root_pos_w)),
                quat_unique(quat_mul(self.base_inverse, tensor(ob.root_quat_w))),
                quat_apply(self.base_inverse, tensor(ob.root_lin_vel_w)),
                quat_apply(self.base_inverse, tensor(ob.root_ang_vel_w)),
            ),
            -1,
        )

    def _solver_diagnostic(self):
        from isaaclab_newton.physics import NewtonManager
        from mujoco_warp._src.types import OverflowType
        import numpy as np
        data = NewtonManager._solver.mjw_data
        for key, field in (('candidates', 'ncollision'), ('contacts', 'nacon'), ('constraints', 'nefc')):
            self.contact_peaks[key] = max(self.contact_peaks[key], int(getattr(data, field).numpy().max()))
        overflow = data.overflow.numpy().ravel()
        mask = int(np.bitwise_or.reduce(overflow))
        return dict(overflow_mask=mask, flags=[flag.name for flag in OverflowType if mask & int(flag)],
                    affected_worlds=np.flatnonzero(overflow).tolist(),
                    world_flags=overflow[overflow != 0].tolist(), contact_peaks=dict(self.contact_peaks),
                    solver_iterations_max=int(data.solver_niter.numpy().max()),
                    solver_config={name: getattr(self.env.cfg.sim.physics.solver_cfg, name)
                                   for name in ('iterations', 'ls_iterations', 'nconmax', 'njmax', 'tolerance')},
                    num_substeps=self.env.cfg.sim.physics.num_substeps)

    def _check_physical_health(self, physical, phase):
        failure = self.physics_health.inspect(physical, phase)
        if failure is not None:
            import json
            failure['solver_diagnostic'] = self._solver_diagnostic()
            self.physics_failure = failure
            raise RuntimeError('Invalid physical state before replay: ' + json.dumps(failure))

    def capture_physics_health_post_step(self):
        """Keep the actual final PD torque and targets before reset changes them."""
        data = self.robot.data
        computed = getattr(data, 'computed_torque', None)
        applied = getattr(data, 'applied_torque', None)
        self.physics_health.finish(
            self.captured_physical,
            self.env.action_manager.get_term('joint_pos').processed_actions,
            None if computed is None else tensor(computed)[:, self.ids],
            None if applied is None else tensor(applied)[:, self.ids])
        self._check_physical_health(self.captured_physical, 'post_step_before_reset')

    def physics_failure_snapshot(self):
        """Read detailed solver/actuator evidence only on failure (no routine CPU copy)."""
        from isaaclab_newton.physics import NewtonManager
        import numpy as np
        worlds = self.physics_failure.get('affected_worlds', [])
        # Current full state/context remains in the entrypoint's failure file.
        # Bound historical per-world evidence when many worlds fail together.
        selected = worlds[:32]
        result = dict(trace=self.physics_health.snapshot(selected),
                      trace_worlds_truncated=len(worlds) > len(selected),
                      raw_state_phase=self.physics_failure.get('phase', 'after_env_step_may_follow_resets'),
                      solver_data={}, solver_model={}, actuator_delay=[], read_errors={})
        solver = NewtonManager._solver
        for group, owner, names in (
            ('solver_data', solver.mjw_data,
             ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'qfrc_applied', 'qfrc_constraint',
              'qfrc_actuator', 'efc_force', 'nefc', 'solver_niter', 'overflow')),
            ('solver_model', solver.mjw_model,
             ('jnt_limited', 'jnt_range', 'jnt_solref', 'jnt_solimp', 'jnt_margin',
              'dof_armature', 'dof_damping', 'dof_frictionloss'))):
            for name in names:
                value = getattr(owner, name, None)
                if value is None:
                    continue
                try:
                    array = value.numpy()
                    if array.ndim and array.shape[0] == self.env.num_envs:
                        array = array[selected]
                    result[group][name] = torch.from_numpy(np.array(array, copy=True))
                except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
                    result['read_errors'][group + '.' + name] = str(exc)
        result['solver_joint_names'] = [solver.mj_model.joint(i).name for i in range(solver.mj_model.njnt)]
        adapter = NewtonManager._adapter
        if adapter is not None:
            for actuator, state in zip(adapter.actuators, adapter._states_a):
                delay = None if state is None else state.delay_state
                if delay is None:
                    continue
                indices = actuator.indices.numpy()
                mask = np.isin(indices // adapter.num_joints, selected)
                record = dict(global_dof_indices=torch.from_numpy(indices[mask].copy()))
                for name in ('buffer_pos', 'buffer_vel', 'buffer_act', 'num_pushes'):
                    values = getattr(delay, name).numpy()
                    record[name] = torch.from_numpy(np.array(values[..., mask], copy=True))
                result['actuator_delay'].append(record)
        return result

    @torch.no_grad()
    def audit_geometry(self, physical):
        """Independent live link-frame check, also executed BEFORE terminal resets."""
        count = min(64, len(physical))
        ids = (torch.arange(count, device=physical.device) + self.geometry_audit_steps) % len(physical)
        inv = self.base_inverse[ids, None].expand(-1, len(self.geometry.body_names), -1)
        data = self.robot.data
        root = tensor(data.root_pos_w)[ids, None]
        real_p = quat_apply(inv, tensor(data.body_link_pos_w)[ids][:, self.geometry_body_ids] - root)
        real_r = quat_mul(inv, tensor(data.body_link_quat_w)[ids][:, self.geometry_body_ids])
        real_v = quat_apply(inv, tensor(data.body_link_lin_vel_w)[ids][:, self.geometry_body_ids])
        real_w = quat_apply(inv, tensor(data.body_link_ang_vel_w)[ids][:, self.geometry_body_ids])
        p, r, v, w = self.geometry.body_kinematics(physical[ids])
        for name, predicted, actual, atol in (
            ("fk_position_m", p, real_p, 3e-5),
            ("fk_linear_velocity_m_s", v, real_v, 3e-4),
            ("fk_angular_velocity_rad_s", w, real_w, 3e-3),
        ):
            self.max_errors[name] = max(self.max_errors.get(name, 0.), float((predicted-actual).abs().max()))
            torch.testing.assert_close(predicted, actual, atol=atol, rtol=2e-4)
        # Sign-invariant orientation comparison avoids q/-q ambiguity.
        sign = torch.where((r*real_r).sum(-1, keepdim=True) >= 0, 1., -1.)
        self.max_errors["fk_quaternion"] = max(self.max_errors.get("fk_quaternion", 0.), float((r-real_r*sign).abs().max()))
        torch.testing.assert_close(r, real_r*sign, atol=3e-5, rtol=2e-4)

    def read_state(self):
        env = self.env
        command = env.command_manager.get_command("object_pose")
        base_local = tensor(self.robot.data.root_pos_w) - env.scene.env_origins
        goal_pos = quat_apply(self.base_inverse, command[:, :3] - base_local)
        goal_quat = quat_unique(quat_mul(self.base_inverse, command[:, 3:7]))
        if self.hold_cfg.dex_reference:
            action = env.action_manager.get_term("joint_pos")
            return torch.cat((self.physical(), self.raw_history, goal_pos, goal_quat,
                              action.ema_encoder.clone(), action.zero_bias.clone(), self.wrench.clone()), -1)
        if self.hold_cfg.compact_state:
            return torch.cat((self.physical(), self.raw_history, goal_pos, goal_quat, self.delay_ready), -1)
        count = env.command_manager.get_term("object_pose").metrics["consecutive_success"][:, None]
        time = (1 - env.episode_length_buf[:, None].float() / env.max_episode_length).clamp(0, 1)
        return torch.cat(
            (
                self.physical(),
                self.ema,
                self.raw_history,
                goal_pos,
                goal_quat,
                time,
                count,
                base_local,
                env.command_manager.get_term("object_pose").hold_ticks[:, None].clone(),
            ),
            -1,
        )

    def read_context(self):
        env = self.env
        fields = {}
        model = env.sim.physics_manager.get_model()
        for name, asset in (("robot", self.robot), ("object", self.obj)):
            fields[name + ".body_mass"] = tensor(asset.data.body_mass)
            fields[name + ".body_inertia"] = tensor(asset.data.body_inertia)
            collision_mask = collision_shape_mask(asset, model, env.num_envs)
            for attr in ("shape_material_mu", "shape_material_restitution"):
                # Same read-only view binding used by official Newton material events.
                values = tensor(asset._root_view.get_attribute(attr, model)).reshape(env.num_envs, -1)
                fields[name + "." + attr] = values[collision_mask].reshape(env.num_envs, -1)
        event = env.event_manager.get_term_cfg("robot_joint_stiffness_and_damping").func
        from isaaclab.actuators.newton import read_group_parameter

        for name, actuator in event._gain_actuators.items():
            if name in event._native_group_names:
                kp = read_group_parameter(self.robot.actuators, name, "controller", "kp")
                kd = read_group_parameter(self.robot.actuators, name, "controller", "kd")
            else:
                kp, kd = actuator.stiffness, actuator.damping
            if self.hold_cfg.compact_state:
                from .context import policy_order_parameter
                kp = policy_order_parameter(self.robot, name, 'controller', 'kp', self.joint_names)
                kd = policy_order_parameter(self.robot, name, 'controller', 'kd', self.joint_names)
            fields[f"actuator.{name}.kp"] = tensor(kp)
            fields[f"actuator.{name}.kd"] = tensor(kd)
        fields["joint_limits"] = self.limits
        if self.hold_cfg.compact_state:
            from .context import compact_context, audit_identified_actuator
            context, schema, self.context_manifest = compact_context(env, fields)
            self.identified_hand_audit = audit_identified_actuator(env, self.robot, self.ids, fields, self.joint_names)
            return context, schema
        flattened = []
        schema = {}
        start = 0
        for name, values in fields.items():
            values = values.reshape(env.num_envs, -1).to(env.device).clone()
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite domain context {name}")
            flattened.append(values)
            schema[name] = [start, start + values.shape[-1]]
            start += values.shape[-1]
        return torch.cat(flattened, -1), schema

    @torch.no_grad()
    def refresh_task_goal(self, env_ids):
        """Publish an external command without advancing sensors, actions or time."""
        if not self.hold_cfg.compact_state:
            raise ValueError('Explicit persistent commands require the compact task')
        current = self.read_state()
        self.state = self.state.clone()
        for field in (self.sl.goal_pos, self.sl.goal_rot):
            self.state[env_ids, field] = current[env_ids, field]
        self.history = self.history.clone()
        last = self.history[env_ids, -1].clone()
        offset = 66 if self.hold_cfg.include_object_position else 63
        last[:, offset:offset+3] = current[env_ids, self.sl.goal_pos]
        # Reuse the already observed object rotation, including its sensor noise.
        # A command change must not generate a second camera/encoder measurement.
        def matrix(six):
            columns = six.reshape(-1, 2, 3).transpose(-1, -2)
            third = torch.linalg.cross(columns[..., 0], columns[..., 1], dim=-1)
            return torch.cat((columns, third[..., None]), -1)
        obj = (matrix(rotation_features(self.sensors['held_quat'][env_ids])) if self.hold_cfg.dex_reference and self.noise_cfg.enabled
               else matrix(rotation_features(current[env_ids, ROT])) if self.hold_cfg.dex_reference
               else matrix(last[:, -12:-6]))
        goal = matrix(rotation_features(current[env_ids, self.sl.goal_rot]))
        last[:, -6:] = (obj @ goal.transpose(-1, -2))[..., :, :2].transpose(-1, -2).flatten(-2)
        self.history[env_ids, -1] = last

    @torch.no_grad()
    def audit_observation(self, clean_obs):
        cfg = PerceptionNoiseCfg(enabled=False, joint_enabled=False)
        if self.hold_cfg.dex_reference:
            from isaaclab.utils.math import matrix_from_quat
            s = self.state
            half = (self.limits[..., 1]-self.limits[..., 0])*.5
            q = s[:, Q]+s[:, dex.ZERO]
            rel = matrix_from_quat(quat_mul(s[:, ROT], quat_conjugate(s[:, self.sl.goal_rot])))
            expected = torch.cat(((q-self.limits.mean(-1))/half, (q-s[:, dex.EMA])/half,
                self.raw_history, s[:, POS], s[:, self.sl.goal_pos], rel[..., :, :2].transpose(-1, -2).flatten(-2)), -1)
            ours, _ = observe(s, self.sensors, self.limits, cfg, self.hold_cfg)
            torch.testing.assert_close(ours, expected, atol=2e-6, rtol=1e-5)
            self.max_errors['reference_observation'] = float((ours-expected).abs().max())
            return
        if self.hold_cfg.observation_protocol == DIRECT_OBSERVATION_PROTOCOL:
            # Independent oracle uses Isaac Lab's XYZW matrix conversion, not the Actor helper.
            from isaaclab.utils.math import matrix_from_quat
            s = self.state
            def six(q):
                return matrix_from_quat(q)[..., :, :2].transpose(-1, -2).flatten(-2)
            bias = s.new_tensor(self.hold_cfg.encoder_bias_rad or [0.] * 21)
            pieces = [2 * (s[:, Q] - bias - self.limits[..., 0]) / (self.limits[..., 1] - self.limits[..., 0]) - 1,
                      s[:, QD] * self.hold_cfg.joint_velocity_scale, self.raw_history[:, :21]]
            if self.hold_cfg.include_object_position:
                pieces.append(s[:, POS])
            pieces.append(s[:, self.sl.goal_pos])
            pieces.extend((six(s[:, ROT]), six(quat_mul(s[:, ROT], quat_conjugate(s[:, self.sl.goal_rot])))))
            expected = torch.cat(pieces, -1)
            ours, _ = observe(s, self.sensors, self.limits, cfg, self.hold_cfg)
            torch.testing.assert_close(ours, expected, atol=2e-6, rtol=1e-5)
            self.max_errors['direct_observation'] = max(self.max_errors.get('direct_observation', 0.), float((ours-expected).abs().max()))
            if ours.shape[-1] != self.hold_cfg.observation_dim:
                raise ValueError('Direct observation dimension mismatch')
            return
        original = StudentObservations(self.env, StudentCfg(), cfg)
        original.history.copy_(self.raw_history.reshape(-1, 4, 21))
        prop, extero, _ = original(clean_obs)
        ours, _ = observe(self.state, self.sensors, self.limits, cfg)
        error = float((ours - torch.cat((prop, extero), -1)).abs().max())
        self.max_errors["clean_student_observation"] = max(error, self.max_errors["clean_student_observation"])
        torch.testing.assert_close(ours, torch.cat((prop, extero), -1), atol=2e-5, rtol=1e-5)
        # Reconstruct every original 49+38 clean teacher feature. Quaternion signs
        # are canonicalized on BOTH sides; no physical orientation is discarded.
        s = self.state
        base = quat_conjugate(self.base_inverse)
        world_goal = quat_unique(quat_mul(base, s[:, self.sl.goal_rot]))
        world_obj = quat_unique(quat_mul(base, s[:, ROT]))
        policy = torch.cat(
            (ours[:, :21], quat_apply(base, s[:, self.sl.goal_pos]) + s[:, BASE_POS], world_goal, s[:, 76:97]), -1
        )
        expected_policy = clean_obs["policy"].clone()
        expected_policy[:, 24:28] = quat_unique(expected_policy[:, 24:28])
        privileged = torch.cat(
            (
                (s[:, QD] - tensor(self.robot.data.default_joint_vel)[:, self.ids]) * 0.2,
                quat_apply(base, s[:, POS]) + s[:, BASE_POS],
                world_obj,
                quat_apply(base, s[:, VEL]),
                quat_apply(base, s[:, OMEGA]) * 0.2,
                quat_unique(quat_mul(world_obj, quat_conjugate(world_goal))),
            ),
            -1,
        )
        expected_privileged = clean_obs["privileged"].clone()
        expected_privileged[:, 24:28] = quat_unique(expected_privileged[:, 24:28])
        expected_privileged[:, 34:38] = quat_unique(expected_privileged[:, 34:38])
        torch.testing.assert_close(policy, expected_policy, atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(privileged, expected_privileged, atol=3e-5, rtol=1e-5)
        self.max_errors["teacher87_reconstruction"] = max(
            self.max_errors.get("teacher87_reconstruction", 0.0),
            float((policy - expected_policy).abs().max()),
            float((privileged - expected_privileged).abs().max()),
        )

    @torch.no_grad()
    def step(self, action):
        state, history, sensors = self.state, self.history, self.sensors
        manager = getattr(self.env, "grasp_reset_manager", None)
        grasp_ids = manager.transition_ids() if manager is not None else {}
        expected_target = action_target(state, action, self.limits, self.hold_cfg)
        if self.hold_cfg.dex_reference:
            base = quat_conjugate(self.base_inverse)
            wrench = state[:, dex.WRENCH]
            self.obj.permanent_wrench_composer.set_forces_and_torques_index(
                forces=quat_apply(base, wrench[:, :3])[:, None].contiguous(),
                torques=quat_apply(base, wrench[:, 3:])[:, None].contiguous(), is_global=True)
        self.captured_physical = None
        self.captured_orientation_error = None
        if hasattr(self, 'physics_health'):
            self.physics_health.begin(state, action)
            self._check_physical_health(state[:, :55], 'before_step')
        command = self.env.command_manager.get_term("object_pose")
        previous_hold = command.hold_ticks.clone()
        previous_completed = command.metrics["consecutive_success"].clone() > 0
        obs, actual_reward, terminated, truncated, info = self.env.step(action)
        if self.hold_cfg.action_mode == 'direct_limits':
            diagnostic = self._solver_diagnostic()
            if diagnostic['overflow_mask']:
                import json
                self.physics_failure = diagnostic
                # ITERATIONS/LS_ITERATIONS share the overflow mask with capacity
                # failures. Report their real type, but reject every nonzero bit.
                raise RuntimeError('Newton/MJWarp physics diagnostic: ' + json.dumps(self.physics_failure))
            if not torch.isfinite(self.captured_physical).all() or not torch.isfinite(actual_reward).all():
                raise RuntimeError('Nonfinite physical state or reward')
        if self.captured_physical is None or self.captured_orientation_error is None:
            raise RuntimeError("Official pre-reset recorder did not execute")
        done = terminated | truncated
        actual_reward = actual_reward.clone()
        done = done.clone()
        expected_pose = (self.captured_physical[:, Q] - state.new_tensor(self.hold_cfg.action_home)).square().sum(-1)
        torch.testing.assert_close(self.captured_hand_pose, expected_pose, atol=2e-6, rtol=1e-5)
        self.max_errors["hand_pose"] = max(self.max_errors.get("hand_pose", 0.), float((self.captured_hand_pose - expected_pose).abs().max()))
        if not self.hold_cfg.compact_state:
            expected_second = (action - 2 * state[:, 76:97] + state[:, 97:118]).square().sum(-1)
            torch.testing.assert_close(self.captured_second_diff, expected_second, atol=2e-6, rtol=1e-5)
            self.max_errors["action_second_diff"] = max(self.max_errors.get("action_second_diff", 0.), float((self.captured_second_diff - expected_second).abs().max()))
        torch.testing.assert_close(self.captured_angular_speed, self.captured_physical[:, OMEGA].norm(dim=-1), atol=2e-6, rtol=1e-5)
        if self.hold_cfg.goal_mode == 'single_dense_hold':
            local_error = (self.captured_physical[:, POS] - state[:, self.sl.goal_pos]).norm(dim=-1)
            torch.testing.assert_close(local_error, self.captured_position_error, atol=2e-6, rtol=1e-5)
            self.max_errors['position_frame'] = max(self.max_errors.get('position_frame', 0.),
                                                    float((local_error-self.captured_position_error).abs().max()))
            torch.testing.assert_close(self.captured_linear_speed, self.captured_physical[:, VEL].norm(dim=-1), atol=2e-6, rtol=1e-5)
        if self.hold_cfg.compact_state:
            blocked = done | ~position_hold_valid(self.captured_position_error, self.captured_linear_speed, self.hold_cfg)
            if self.hold_cfg.sequential_goals:
                expected_hold, expected_event = hold_transition(
                    previous_hold, self.captured_orientation_error, self.captured_angular_speed,
                    self.env.step_dt, blocked=blocked, **hold_kwargs(self.hold_cfg))
            else:
                expected_hold, expected_event = persistent_hold_transition(
                    previous_hold, self.captured_orientation_error, self.captured_angular_speed,
                    self.env.step_dt, already_completed=previous_completed,
                    blocked=blocked, **hold_kwargs(self.hold_cfg))
        else:
            expected_hold, expected_event = goal_event(state, self.captured_physical, self.env.step_dt,
                error=self.captured_orientation_error, speed=self.captured_angular_speed,
                hold_cfg=self.hold_cfg, blocked=done)
        torch.testing.assert_close(expected_event, self.captured_completion)
        torch.testing.assert_close(expected_hold, self.captured_hold, atol=0, rtol=0)
        self.max_errors['hold_counter'] = float((expected_hold-self.captured_hold).abs().max())
        audit = audit_real_reward(
            state, self.captured_physical, action, self.captured_orientation_error, actual_reward, self.env.step_dt, self.success_bonus, self.action_rate_weight, self.joint_vel_weight, self.action_l2_weight, self.action_second_diff_weight, hold_cfg=self.hold_cfg, success_event=expected_event
        )
        for key in ("reward", "orientation_frame"):
            self.max_errors[key] = max(self.max_errors.get(key, 0.0), audit[key])
        self.max_errors["success_boundary_disagreements"] = (
            self.max_errors.get("success_boundary_disagreements", 0) + audit["success_boundary_disagreements"]
        )
        self.max_errors["ema"] = max(
            self.max_errors["ema"], float((expected_target - self.captured_target).abs().max())
        )
        torch.testing.assert_close(expected_target, self.captured_target, atol=2e-6, rtol=1e-5)
        record = dict(
            state=state,
            history=history,
            action=action.clone(),
            next_physical=self.captured_physical,
            done=done,
            terminated=terminated.clone(),
            truncated=truncated.clone(),
            reward=actual_reward,
            success=self.captured_completion.clone(),
        )
        if self.hold_cfg.dex_reference:
            record['contact'] = self.captured_contacts
        if self.hold_cfg.goal_mode == 'single_dense_hold':
            record['goal_elapsed_s'] = self.captured_goal_elapsed
        record.update(grasp_ids)
        record.update({"sensor/" + k: v for k, v in sensors.items()})
        if not self.hold_cfg.compact_state:
            self.ema = torch.where(done[:, None], tensor(self.robot.data.default_joint_pos)[:, self.ids], self.captured_target)
        if not self.hold_cfg.dex_reference:
            self.delay_ready = (~done).float()[:, None]
        self.raw_history = (action.clone() if self.hold_cfg.compact_state else
                            torch.cat((action, self.raw_history[:, :63]), -1))
        self.raw_history = torch.where(done[:, None], 0.0, self.raw_history)
        if not getattr(self, 'capture_terminal_history', False):
            self.sensors = reset_sensors(sensors, done, self.noise_cfg)
        if self.hold_cfg.dex_reference:
            self.wrench = dex.sample_wrench(state, self.hold_cfg)
        self.state = self.read_state()
        if self.hold_cfg.compact_state and not self.hold_cfg.dex_reference and (done.any() or self.geometry_audit_steps < 8):
            from .context import audit_delay_memory
            audit_delay_memory(self.delay_ready)
        terminal_state = advance(state, self.captured_physical, action, self.captured_target,
                                 self.state[:, self.sl.goal_rot], self.env.max_episode_length, hold_cfg=self.hold_cfg,
                                 event_override=(expected_hold, expected_event))
        if not self.hold_cfg.compact_state:
            torch.testing.assert_close(self.state[~done, HOLD].squeeze(-1), expected_hold[~done], atol=0, rtol=0)
            if (~done).any():
                torch.testing.assert_close(self.state[~done, COUNT].squeeze(-1),
                                           state[~done, COUNT].squeeze(-1) + expected_event[~done])
        unchanged = (~done & ~expected_event
                     if (self.hold_cfg.goal_mode != 'single_dense_hold' or self.hold_cfg.sequential_goals)
                     else ~done)
        torch.testing.assert_close(self.state[unchanged, self.sl.goal_rot], state[unchanged, self.sl.goal_rot])
        if self.hold_cfg.goal_mode == 'single_dense_hold':
            torch.testing.assert_close(self.state[~done, self.sl.goal_pos], state[~done, self.sl.goal_pos])
        record["next_aux"] = torch.where(done[:, None], terminal_state, self.state)[:, 55:].clone()
        if manager is not None and hasattr(manager, "audit_transition") and (done.any() or self.geometry_audit_steps < 8):
            manager.audit_transition(record, self.state)
        if getattr(self, 'capture_terminal_history', False):
            from viserdex.flashsac.observations import transition_history
            record['next_history'], self.history, observation, self.sensors = transition_history(
                history, sensors, terminal_state, self.state, done, self.limits, self.noise_cfg, self.hold_cfg)
        else:
            observation, self.sensors = observe(self.state, self.sensors, self.limits, self.noise_cfg, self.hold_cfg)
            self.history = push_history(history, observation, done)
        self.reset_count += int(done.sum())
        if done.any():
            separation = (self.captured_physical[done, :45] - self.state[done, :45]).abs().max()
            self.max_errors["terminal_reset_state_separation"] = max(
                self.max_errors.get("terminal_reset_state_separation", 0.0), float(separation)
            )
        return record, expected_target, observation, obs
