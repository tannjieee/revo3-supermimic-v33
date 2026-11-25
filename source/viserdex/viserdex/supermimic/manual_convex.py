"""Retraining task for the manual convex hand and single-pose 81D Actor."""

import gymnasium as gym
from copy import deepcopy
from dataclasses import fields
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from viserdex.assets.prepare_manual_convex import MANUAL_USD
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from .simulation import SuperMimicEnvCfg
from .direct_actions import DirectJointPositionActionCfg
from .single_goal import (SingleGoalCommandCfg, dense_orientation_reward,
                          dense_position_reward, dense_stability_reward)
from .context import MOVING_HAND_BODIES_PATTERN

TASK_ID = 'Isaac-ViserDex-Repose-Revo3-SuperMimic-ManualConvex-SinglePose-Dense-Identified-v8'


@configclass
class ManualConvexEnvCfg(SuperMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(MANUAL_USD)
        # Mechanical stops remain authored; policy limits use the training JSON.
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        # A world-fixed palm has no inertial DOFs. Retain all palm contact shapes,
        # but randomize mass/inertia of moving links only.
        self.events.robot_scale_mass.params['asset_cfg'] = SceneEntityCfg(
            'robot', body_names=MOVING_HAND_BODIES_PATTERN)
        # MJWarp mixes equal-priority contact friction with max(). Fix the hand
        # at the object's lower bound so the full object range remains effective,
        # including low-friction objects. This is a nominal value, not a fit.
        material = self.events.robot_physics_material.params
        material['static_friction_range'] = (0.7, 0.7)
        material['dynamic_friction_range'] = (0.7, 0.7)
        from .training_limits import load_training_limits
        software_limits, software_source = load_training_limits()
        self.actions.joint_pos = DirectJointPositionActionCfg(
            asset_name='robot', joint_names=list(REVO3_JOINT_NAMES), preserve_order=True,
            position_limits_rad=tuple(map(tuple, software_limits.tolist())),
            software_limits_sha256=software_source['sdk_software_limits']['sha256'],
            training_limits_sha256=software_source['sha256'],
        )
        from .identified_hand import actuator_config, load_identified_hand
        parameters, identified = load_identified_hand()
        self.scene.robot.actuators = {"fingers": actuator_config()}
        self.actions.joint_pos.encoder_bias_rad = tuple(parameters["encoder_bias_rad"])
        self.sim.dt = identified["pd_period_s"]
        self.decimation = 10
        self.sim.render_interval = 10
        old_command = self.commands.object_pose
        self.commands.object_pose = SingleGoalCommandCfg(**{
            field.name: deepcopy(getattr(old_command, field.name))
            for field in fields(old_command) if field.name not in ('class_type', 'update_goal_on_success')
        })
        # The legacy orientation-only task offset its marker by -4 cm. A real
        # position target starts at the nominal object pose instead.
        self.commands.object_pose.init_pos_offset = (0., 0., 0.)
        self.rewards.track_orientation_inv_l2 = None
        self.rewards.track_orientation_exp = RewardTermCfg(func=dense_orientation_reward, weight=1.0)
        self.rewards.track_position_exp = RewardTermCfg(func=dense_position_reward, weight=1.0)
        self.rewards.stable_pose_exp = RewardTermCfg(func=dense_stability_reward, weight=1.0)
        self.rewards.object_ang_vel_l2 = None
        self.rewards.object_lin_vel_l2 = None
        self.rewards.action_second_diff_l2 = None
        self.rewards.action_l2 = None
        self.rewards.hand_pose = None
        self.rewards.success_bonus = None
        self.terminations.max_consecutive_success = None


if TASK_ID not in gym.registry:
    gym.register(id=TASK_ID, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': f'{__name__}:ManualConvexEnvCfg'})
