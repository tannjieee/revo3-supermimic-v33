"""Train/evaluate the independent Revo3 SuperMimic task using official launcher.

No teacher checkpoint is accepted or loaded. --env_config binds the existing
validated asset/dynamics configuration, not a teacher policy.
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "viserdex"))
from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args
from viserdex.distillation.environment import SavedEnvironmentBinding, contract_sha256
from viserdex.distillation.teacher import configure_clean_state_observations
from viserdex.supermimic.simulation import TASK_ID as LEGACY_TASK_ID, TransitionRecorderCfg
from viserdex.supermimic.manual_convex import TASK_ID as DIRECT_TASK_ID
from viserdex.supermimic.distributed import Distributed
from viserdex.supermimic.position_reward import POSITION_CONFIG_FIELDS, POSITION_REWARD_PROTOCOL, position_hold_valid
from viserdex.supermimic.stability_reward import STABILITY_CONFIG_FIELDS, DENSE_REWARD_PROTOCOL, ORIENTATION_HOLD_REWARD_PROTOCOL
from viserdex.supermimic.quiet_hold import QUIET_HOLD_CONFIG_FIELDS, configure_fresh_environment, reward_protocol as quiet_hold_reward_protocol

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--joint_speed_limit_deg_s", type=float, default=90.)
parser.add_argument("--target_rate_limit_deg_s", type=float, default=60.)
parser.add_argument("--resume_speed_cost_override", action="store_true",
                    help="Explicitly override only the two speed-cost thresholds on resume; retain models, optimizers and normalization floors")
parser.add_argument("--object_hold_costs", action=argparse.BooleanOptionalAction, default=None,
                    help="New 25Hz runs use joint plus object holding/drop costs; saved runs keep their cost heads")
parser.add_argument("--network_revision", choices=("legacy", "urdf_little_v2", "hardware_center25_v3"),
                    help="Fresh manual-convex default: hardware_center25_v3; resume keeps its serialized revision")
parser.add_argument("--dex_reference_config", type=Path, help="Fresh v33 Actor/EMA/WM profile YAML")
parser.add_argument('--position_boundary_config', type=Path,
                    help='Fresh die58 hand-center COM ellipsoid; defaults to configs/supermimic_v33/position_boundary.yaml')
parser.add_argument("--env_config", type=Path, help="Saved environment; optional for a new manual_convex run")
parser.add_argument("--policy_variant", choices=('legacy', 'manual_convex'), default='legacy')
parser.add_argument('--object_variant', choices=('die50', 'die50_47g', 'die58', 'die70'),
                    help='Fresh manual-convex runs default to the 58 mm die; saved runs retain their asset')
parser.add_argument("--joint_velocity_scale", type=float, default=0.1)
parser.add_argument("--omit_object_position", action='store_true', help="Explicit 78D ablation; target position remains observed")
parser.add_argument("--track_object_position", action=argparse.BooleanOptionalAction, default=None,
                    help="Fresh manual-convex runs default to orientation-only stable holding; saved runs keep their task")
parser.add_argument("--goal_position_base_m", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help="Fixed target in hand-base meters; default is the nominal object position")
parser.add_argument("--position_reward_scale", type=float, default=1.)
parser.add_argument("--position_tolerance_m", type=float, default=.01)
parser.add_argument("--position_sigma_m", type=float, default=.05)
parser.add_argument("--hold_position_threshold_m", type=float, default=.02)
parser.add_argument("--hold_linear_speed_threshold_m_s", type=float, default=.05)
parser.add_argument("--goal_evaluation_s", type=float, default=10., help="Evaluation marker only; no reward gate or policy clock")
parser.add_argument("--stability_reward_scale", type=float,
                    help="Fresh quiet holding defaults to 10; saved runs retain their reward")
parser.add_argument('--quiet_hold', action=argparse.BooleanOptionalAction, default=None,
                    help='Fresh manual-convex default: tolerant arrival, stable object and quiet joints/targets')
parser.add_argument('--episode_length_s', type=float, choices=(10., 20.),
                    help='Fresh manual-convex default 10 seconds; saved environments retain their duration')
parser.add_argument("--stability_angle_sigma_deg", type=float, default=15.)
parser.add_argument("--stability_position_sigma_m", type=float, default=.02)
parser.add_argument("--stability_angular_speed_scale", type=float, default=.3)
parser.add_argument("--stability_linear_speed_scale", type=float, default=.02)
parser.add_argument("--gamma", type=float, help="Manual-convex default 0.99 per policy control step")
parser.add_argument("--lam", type=float, help="Fresh-run GAE lambda override (e.g. 0.9025 for 25 Hz time alignment)")
parser.add_argument("--horizon", type=int, help="Fresh-run imagination horizon H override")
parser.add_argument("--wm_holdout_fraction", type=float, help="Fresh-run fraction of environments held out strictly for WM validation")
parser.add_argument("--replay_sampling_mode", choices=("legacy", "uniform_failure"), help="Fresh-run WM replay sampling mode")
parser.add_argument("--resume_gamma_override", action="store_true",
                    help="Apply explicit --gamma to a resume only before any actor/critic optimizer update")
parser.add_argument("--orientation_reward_mode", choices=("gaussian", "linear_fine"),
                    help="Fresh-run orientation reward shape (gaussian or linear_fine)")
parser.add_argument("--orientation_tolerance_deg", type=float,
                    help="Fresh-run orientation tolerance plateau in degrees (0 disables plateau)")
parser.add_argument("--orientation_coarse_sigma_deg", type=float, default=90.)
parser.add_argument("--orientation_fine_sigma_deg", type=float, default=10.)
parser.add_argument("--orientation_coarse_fraction", type=float, default=.3)
parser.add_argument("--orientation_reward_scale", type=float, default=10.)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument('--console_log', choices=('pretty', 'json'), default='pretty',
                    help='Terminal progress format; metrics.jsonl always retains every metric')
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=Path)
parser.add_argument("--resume_solver_iterations", type=int,
                    help="Explicitly increase only Newton solver iterations; preserve all other environment contracts")
parser.add_argument("--resume_solver_substeps", type=int,
                    help="Declared numerical experiment; preserve 5ms PD, 15ms delay and the saved policy rate")
parser.add_argument("--replay_storage", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--buffer_steps", type=int,
                    help="Fresh runs only: replay time slots per environment; total capacity is buffer_steps * num_envs")
parser.add_argument("--warmup_grasp_bank", type=Path, help="Training-split real-transition WM warmup only")
parser.add_argument('--warmup_joint_amplitude_deg', type=float, help='Real warmup only: bounded random HOME excursion in degrees')
parser.add_argument('--warmup_joint_segment_steps', type=int, help='Smooth random HOME segment duration in control steps')
parser.add_argument('--wm_only_iterations', type=int, help='Fresh-run Actor freeze duration; WM and real Critic continue learning')
parser.add_argument("--warmup_grasp_steps", type=int, default=256)
parser.add_argument("--warmup_grasp_updates", type=int, default=200)
parser.add_argument("--warmup_grasp_exploration", type=float, default=.03)
parser.add_argument("--wm_readiness_rounds", type=int, default=0,
                    help="Maximum frozen-policy fit/holdout rounds before joint updates; zero disables")
parser.add_argument("--wm_readiness_updates", type=int, default=50)
parser.add_argument("--wm_readiness_passes", type=int, default=2)
parser.add_argument("--resume_actor_ramp_iterations", type=int,
                    help="After WM readiness, progressively restore Actor LR over this many iterations")
parser.add_argument("--grasp_bank", type=Path)
parser.add_argument("--grasp_split", choices=("train", "validation", "test"), default="train")
parser.add_argument("--adaptive_grasps", action="store_true")
parser.add_argument("--adaptive_replay", action="store_true")
parser.add_argument("--audit_interval", type=int, default=200)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval_steps", type=int, default=1200)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--latest_interval", type=int, default=100, help="Atomically replace rolling latest.pt")
parser.add_argument('--video_interval', type=int, default=None,
                    help='Record every N completed iterations; manual_convex default 500; 0 disables')
parser.add_argument('--video_steps', type=int, default=500, help='Frames per periodic video; 500 is 20 seconds at 25 Hz')
parser.add_argument('--video_seed', type=int, default=43, help='Independent fixed evaluation seed for periodic videos')
parser.add_argument('--video_timeout_s', type=float, default=240., help='Fail recording after this wall time, at most 240 seconds')
parser.add_argument("--microbatch", type=int, default=128)
parser.add_argument("--wm_microbatch", type=int)
parser.add_argument("--actor_microbatch", type=int)
parser.add_argument("--critic_microbatch", type=int)
parser.add_argument("--resume_microbatch_override", action="store_true",
                    help="Explicitly change only execution microbatches when resuming; retain logical batches and all training contracts")
parser.add_argument("--benchmark_execution", action="store_true",
                    help="Benchmark full-size updates on real warmup data, then exit without training/checkpointing the diagnostic updates")
parser.add_argument("--wm_cuda_graph", action="store_true",
                    help="Capture full-batch WM loss/backward; optimizer and replay remain unchanged")
parser.add_argument('--wm_task_losses', action=argparse.BooleanOptionalAction, default=None,
                    help='Fresh 25Hz default: reward/clean-observation/cost supervision and learned termination (v28)')
parser.add_argument('--stable_value_learning', action=argparse.BooleanOptionalAction, default=None,
                    help='Enable bounded failure-risk cost semantics; Critic learning is selected separately')
parser.add_argument('--paper_critic', action=argparse.BooleanOptionalAction, default=None,
                    help='Fresh absolute-position default: imagined lambda-return MSE with online linear Critic, no EMA (Eq. 3)')
parser.add_argument('--absolute_policy_optimization', action=argparse.BooleanOptionalAction, default=None,
                    help='Fresh stable 25Hz v30: absolute HOME initialization, correlated angle exploration, logit regularization')
parser.add_argument('--actor_cost_weight', type=float,
                    help='Fresh v30 only: effective penalty on normalized bounded cost returns')
parser.add_argument('--critic_td_steps', type=int,
                    help='Legacy --no-paper_critic only: real TD window (default 8); paper mode requires 1 and does not use it')
parser.add_argument('--joint_velocity_norm_floor_rad_s', type=float,
                    help='Fresh v30 default: joint speed cost threshold; shared Actor/WM/Critic std floor')
parser.add_argument('--calibrate_termination', action='store_true',
                    help='Freeze a resumed v28 model, collect fresh data, fit and validate termination calibration, then exit')
parser.add_argument("--amp", choices=("none", "bf16"), default="none")
parser.add_argument("--eager_geometry", action="store_true", help="Diagnostic comparison: execute the same FP32 geometry without compiler fusion")
parser.add_argument(
    "--smoke", action="store_true", help="Explicit small-batch diagnostic; NOT production hyperparameters"
)
parser.add_argument("--audit", action="store_true", help="Force a timeout during warmup to verify pre-reset capture")
parser.add_argument(
    "--holdout", action="store_true", help="After fitting, collect 32 fresh steps for open-loop error checks"
)
parser.add_argument(
    "--check_joint_update",
    action="store_true",
    help="Diagnostic only: exercise actor immediately; keep full batches/update counts",
)
add_runtime_args(parser, LEGACY_TASK_ID)
if "--distributed" not in parser._option_string_actions:
    parser.add_argument("--distributed", action="store_true")
args = parse_runtime_args(parser)
if args.network_revision is None:
    args.network_revision = ('hardware_center25_v3' if args.policy_variant == 'manual_convex'
                             and args.env_config is None else 'legacy')
TASK_ID = DIRECT_TASK_ID if args.policy_variant == 'manual_convex' else LEGACY_TASK_ID
if args.task not in (LEGACY_TASK_ID, TASK_ID):
    parser.error('Task and policy variant differ')
args.task = TASK_ID
if args.object_variant and args.policy_variant != 'manual_convex':
    parser.error('--object_variant requires manual_convex')
if args.resume and not args.env_config:
    parser.error('--resume requires its saved --env_config so object and dynamics cannot change')
if args.buffer_steps is not None and (args.buffer_steps < 8 or args.resume or args.smoke):
    parser.error('--buffer_steps requires a fresh non-smoke run and at least 8 time slots')
if args.position_boundary_config is not None and (args.resume or args.env_config or
        args.policy_variant != 'manual_convex' or args.network_revision != 'hardware_center25_v3'):
    parser.error('--position_boundary_config requires a fresh manual-convex hand-center run')
from viserdex.supermimic.periodic_video import VideoSettings, record_checkpoint, record_resume_boundary
video_settings = VideoSettings(interval=(500 if args.policy_variant == 'manual_convex' else 0)
                               if args.video_interval is None else args.video_interval,
                               steps=args.video_steps, seed=args.video_seed, timeout_s=args.video_timeout_s)
if video_settings.interval and args.policy_variant != 'manual_convex':
    parser.error('Periodic video currently requires the manual_convex policy variant')
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
if world_size > 1:
    args.distributed = True
    args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
elif args.distributed:
    parser.error("--distributed requires torchrun with at least two processes")
if min(args.num_envs, args.iterations, args.microbatch, args.save_interval, args.latest_interval) < 1:
    parser.error("Counts must be positive")
if args.evaluate and not args.resume:
    parser.error("Evaluation requires --resume")
if args.eval_steps < 1:
    parser.error('--eval_steps must be positive')
if args.calibrate_termination and (not args.resume or not args.env_config or world_size != 1 or args.num_envs < 4
                                  or args.audit or args.evaluate or args.smoke or args.benchmark_execution
                                  or args.warmup_grasp_bank or args.wm_readiness_rounds):
    parser.error('Calibration requires --resume, --env_config, >=4 environments and a single process; no forced audits/training modes')
if args.resume_microbatch_override and not args.resume:
    parser.error("--resume_microbatch_override requires --resume")
if args.resume_speed_cost_override and not args.resume:
    parser.error("--resume_speed_cost_override requires --resume")
if args.resume_gamma_override and (not args.resume or args.gamma is None):
    parser.error("--resume_gamma_override requires --resume and --gamma")
if args.gamma is not None and not 0 < args.gamma < 1:
    parser.error("Gamma must be between zero and one")
if any(value is not None and value < 1 for value in (args.wm_microbatch, args.actor_microbatch, args.critic_microbatch)):
    parser.error("Microbatches must be positive")
if args.benchmark_execution and (world_size != 1 or args.evaluate or args.smoke):
    parser.error("Execution benchmark requires full-size single-process training")
if args.env_config is None and args.policy_variant == 'legacy':
    parser.error('Legacy training requires --env_config')
if args.grasp_bank and (world_size > 1 or args.policy_variant != "manual_convex"):
    parser.error("Grasp sampling currently requires single-process manual_convex training")
if args.adaptive_grasps and not args.grasp_bank:
    parser.error("--adaptive_grasps requires --grasp_bank")
if args.warmup_grasp_bank and (args.grasp_bank or args.evaluate or world_size != 1 or args.policy_variant != 'manual_convex'):
    parser.error('Temporary grasp warmup requires single-process manual-convex training without persistent grasp resets')
if args.warmup_grasp_bank and args.wm_readiness_rounds < 1:
    parser.error('Grasp warmup requires a normal-inhand WM readiness check')
if args.wm_readiness_rounds < 0 or min(args.warmup_grasp_steps, args.warmup_grasp_updates, args.wm_readiness_updates, args.wm_readiness_passes) < 1:
    parser.error('Warmup counts must be positive; readiness rounds may be zero')
if args.wm_readiness_rounds and (args.audit_interval <= 0 or world_size != 1):
    parser.error('Readiness requires enabled audit and a single process')
if args.resume_actor_ramp_iterations is not None and (not args.resume or args.resume_actor_ramp_iterations < 1
                                                      or args.wm_readiness_rounds < 1 or args.evaluate):
    parser.error('Resume Actor ramp requires --resume, a positive duration, and training with WM readiness')
if args.resume_solver_iterations is not None or args.resume_solver_substeps is not None:
    if not args.resume or not args.env_config:
        parser.error('Solver overrides require --resume and --env_config')
    from viserdex.supermimic.solver_budget import SolverBudgetBinding
    binding = SolverBudgetBinding(args.env_config, args.resume_solver_iterations, args.resume_solver_substeps)
else:
    binding = SavedEnvironmentBinding(args.env_config) if args.env_config else None


def configure(cfg):
    if binding is not None:
        if args.policy_variant == 'manual_convex':
            # Instantiate the saved typed task schema before from_dict: the
            # hand-center termination has an origin parameter absent in v25.
            revision = binding.saved['actions']['joint_pos'].get('limit_contract', 'legacy')
            if revision in ('urdf_little_v2', 'hardware_center25_v3'):
                from viserdex.supermimic.revised_contract import configure_environment
                configure_environment(cfg, revision)
        from viserdex.supermimic.position_boundary import prepare_saved_boundary
        prepare_saved_boundary(cfg, binding.saved)
        binding.configure(cfg, args)
    configure_clean_state_observations(cfg)
    cfg.recorders = TransitionRecorderCfg()
    cfg.scene.clone_in_fabric = False
    cfg.scene.robot.spawn.activate_contact_sensors = False
    cfg.scene.object.spawn.activate_contact_sensors = False
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed + rank
    if args.network_revision != 'legacy' and args.policy_variant == 'manual_convex' and not args.resume:
        if binding is not None:
            raise ValueError('Fresh revised training needs its own environment, not a legacy --env_config')
        from viserdex.supermimic.revised_contract import configure_environment
        configure_environment(cfg, args.network_revision)
    if args.policy_variant == 'manual_convex' and binding is None:
        from viserdex.supermimic.core import manual_convex_config
        object_costs = (args.network_revision == 'hardware_center25_v3'
                        and args.object_hold_costs is not False)
        settings = manual_convex_config(goal_evaluation_s=args.goal_evaluation_s,
            stability_angle_sigma_rad=math.radians(args.stability_angle_sigma_deg),
            stability_reward_scale=(args.stability_reward_scale if args.stability_reward_scale is not None
                                    else 0. if object_costs else 3.),
            **{name: getattr(args, name) for name in STABILITY_CONFIG_FIELDS
               if name not in ('stability_angle_sigma_rad', 'stability_reward_scale')},
            orientation_coarse_sigma_rad=math.radians(args.orientation_coarse_sigma_deg),
            orientation_fine_sigma_rad=math.radians(args.orientation_fine_sigma_deg),
            orientation_coarse_fraction=args.orientation_coarse_fraction,
            orientation_reward_mode=(args.orientation_reward_mode or 'gaussian'),
            orientation_reward_scale=args.orientation_reward_scale,
            **{name: getattr(args, name) for name in POSITION_CONFIG_FIELDS if name != 'track_object_position'},
            track_object_position=args.track_object_position if args.track_object_position is not None else False)
        if args.goal_position_base_m is not None:
            if not all(math.isfinite(v) for v in args.goal_position_base_m):
                raise ValueError('goal_position_base_m must be finite')
            cfg.commands.object_pose.goal_position_base_m = tuple(args.goal_position_base_m)
        for name in ('goal_evaluation_s', *STABILITY_CONFIG_FIELDS, 'orientation_coarse_sigma_rad', 'orientation_fine_sigma_rad',
                     'orientation_coarse_fraction', 'orientation_reward_mode', 'orientation_reward_scale', *POSITION_CONFIG_FIELDS):
            setattr(cfg.commands.object_pose, name, getattr(settings, name))
    if binding is not None and args.track_object_position is not None:
        if getattr(cfg.commands.object_pose, 'track_object_position', True) != args.track_object_position:
            raise ValueError('Saved environment cannot change position tracking; start a fresh training run')
    if binding is not None and args.policy_variant == 'manual_convex' and args.goal_position_base_m is not None:
        from viserdex.supermimic.goal_contract import verify_saved_goal_position
        verify_saved_goal_position(cfg, args.goal_position_base_m)
    if args.policy_variant == 'manual_convex':
        from viserdex.supermimic.object_variant import bind_object
        bind_object(cfg, args.object_variant, saved=binding is not None)
        if binding is None:
            from viserdex.supermimic.fixed_palm_reset import configure_fixed_palm_reset
            configure_fixed_palm_reset(cfg)
        from viserdex.supermimic.fixed_palm_reset import reset_fixed_palm_v1
        reset = cfg.events.reset_robot_joints.func
        if (reset in (reset_fixed_palm_v1, 'viserdex.supermimic.fixed_palm_reset:reset_fixed_palm_v1')
                and (args.grasp_bank or args.warmup_grasp_bank)):
            raise ValueError('Fixed palm initialization cannot be overridden by a grasp bank')
    if args.dex_reference_config is not None:
        if binding is not None or args.resume or args.policy_variant != "manual_convex":
            raise ValueError("--dex_reference_config requires a fresh manual-convex run")
        from viserdex.supermimic.dex_reference import load_profile, configure_environment
        profile, digest = load_profile(args.dex_reference_config)
        configure_environment(cfg, profile, digest)
    if args.policy_variant == 'manual_convex' and binding is None:
        # Apply after control-profile configuration, which initializes legacy rewards.
        configure_fresh_environment(cfg, enabled=args.quiet_hold is not False,
            duration=10. if args.episode_length_s is None else args.episode_length_s,
            stability_scale=args.stability_reward_scale)
        if args.dex_reference_config is not None:
            from viserdex.supermimic.dex_reference import reward_options
            from viserdex.supermimic.quiet_hold import validate_quiet_hold
            for name, value in reward_options(profile).items():
                setattr(cfg.commands.object_pose, name, value)
        if args.orientation_reward_mode is not None:
            cfg.commands.object_pose.orientation_reward_mode = args.orientation_reward_mode
        if args.orientation_tolerance_deg is not None:
            cfg.commands.object_pose.orientation_tolerance_rad = math.radians(args.orientation_tolerance_deg)
        if args.dex_reference_config is not None or args.orientation_reward_mode is not None or args.orientation_tolerance_deg is not None:
            from viserdex.supermimic.quiet_hold import validate_quiet_hold
            from viserdex.supermimic.orientation_reward import validate_orientation_reward
            validate_quiet_hold(cfg.commands.object_pose)
            validate_orientation_reward(cfg.commands.object_pose)
    elif binding is not None:
        if args.quiet_hold is not None and args.quiet_hold != getattr(cfg.commands.object_pose, 'quiet_hold_enabled', False):
            raise ValueError('Saved reward cannot change on resume; start a fresh run')
        if args.episode_length_s is not None and args.episode_length_s != cfg.episode_length_s:
            raise ValueError('Saved episode duration cannot change on resume; start a fresh run')
    elif args.quiet_hold or args.episode_length_s is not None:
        raise ValueError('Quiet holding and episode overrides require manual_convex')
    if binding is None and args.policy_variant == 'manual_convex' and args.network_revision == 'hardware_center25_v3':
        from viserdex.supermimic.position_boundary import configure_fresh_boundary, DEFAULT_CONFIG
        from viserdex.supermimic.object_variant import object_metadata
        if args.position_boundary_config is not None or object_metadata(cfg.scene.object.spawn.usd_path)['variant'] == 'die58':
            configure_fresh_boundary(cfg, args.position_boundary_config or DEFAULT_CONFIG)
    if binding is not None:
        binding.verify(cfg)


def environment_metadata(cfg):
    import hashlib
    if binding is not None:
        result = binding.verify(cfg)
        assets = dict(result['assets'])
    else:
        result = dict(contract_sha256=contract_sha256(cfg.to_dict()),
                      physics=str(cfg.sim.physics.class_type), step_dt=cfg.sim.dt * cfg.decimation)
        assets = {}
        for name in ('robot', 'object'):
            path = Path(getattr(cfg.scene, name).spawn.usd_path).resolve(strict=True)
            assets[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if args.policy_variant == 'manual_convex':
        from viserdex.assets.prepare_manual_convex import MANUAL_USD, MANUAL_SOURCE_SHA256
        asset_report = json.loads(MANUAL_USD.with_name('mechanical-audit.json').read_text())
        source = Path(asset_report['source'])
        if not source.is_absolute():
            source = (MANUAL_USD.parent / source).resolve(strict=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != MANUAL_SOURCE_SHA256 or assets['robot']['sha256'] != asset_report['usd_sha256']:
            raise ValueError('Manual convex asset fingerprint changed; regenerate its audited reference')
        assets['manual_source'] = dict(path=str(source), sha256=digest)
    result['assets'] = assets
    return result


app = SimulationRuntime(args, configure=configure)
import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.distillation.config import PerceptionNoiseCfg
from viserdex.supermimic.core import Config, config20, manual_convex_config
from viserdex.supermimic.constraints import cost_names
from viserdex.supermimic.spatial_contract import drop_distance
from viserdex.supermimic.policy_contract import observation_fields
from viserdex.supermimic.state_layout import fields as state_fields
from viserdex.supermimic.orientation_reward import ORIENTATION_REWARD_PROTOCOL
from viserdex.supermimic.object_variant import object_metadata
from viserdex.supermimic.stream_audit import StreamAudit
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.trainer import Trainer, checkpoint_version, imagination_protocol, holdout_env_count
from viserdex.supermimic.validation import open_loop_errors


def main():
    distributed = Distributed.initialize(app.cfg.sim.device)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    rank_dir = args.output_dir if rank == 0 else args.output_dir / f"rank_{rank}"
    if rank != 0:
        rank_dir.mkdir(exist_ok=False)
    factory = manual_convex_config if args.policy_variant == 'manual_convex' else config20
    revised = args.network_revision != 'legacy' and args.policy_variant == 'manual_convex' and not args.resume
    if revised:
        from viserdex.supermimic.core import revised_manual_config
        factory = revised_manual_config
        if args.network_revision == 'hardware_center25_v3':
            from viserdex.supermimic.core import hand_center_config
            factory = hand_center_config
    goal_options = {}
    if args.policy_variant == 'manual_convex':
        for name in ('goal_evaluation_s', *STABILITY_CONFIG_FIELDS, *QUIET_HOLD_CONFIG_FIELDS, 'orientation_coarse_sigma_rad', 'orientation_fine_sigma_rad',
                     'orientation_coarse_fraction', 'orientation_reward_mode', 'orientation_reward_scale', *POSITION_CONFIG_FIELDS):
            goal_options[name] = getattr(app.cfg.commands.object_pose, name)
        if goal_options['quiet_hold_enabled']:
            goal_options['network_joint_limits'] = app.cfg.actions.joint_pos.position_limits_rad
        if args.gamma is not None:
            goal_options['gamma'] = args.gamma
        if args.lam is not None and not args.resume:
            goal_options['lam'] = args.lam
        if args.horizon is not None and not args.resume:
            goal_options['horizon'] = args.horizon
        if args.wm_holdout_fraction is not None and not args.resume:
            goal_options['wm_holdout_fraction'] = args.wm_holdout_fraction
        if args.replay_sampling_mode is not None and not args.resume:
            goal_options['replay_sampling_mode'] = args.replay_sampling_mode
    elif args.gamma is not None:
        raise ValueError('--gamma override is supported by the new manual_convex variant only')
    if revised and args.network_revision == 'hardware_center25_v3':
        goal_options.update(cost_velocity_limit_rad_s=math.radians(args.joint_speed_limit_deg_s),
                            cost_target_rate_limit_rad_s=math.radians(args.target_rate_limit_deg_s),
                            object_hold_costs_enabled=args.object_hold_costs is not False,
                            wm_task_losses=args.wm_task_losses is not False,
                            stable_value_learning=(args.stable_value_learning if args.stable_value_learning is not None
                                else args.wm_task_losses is not False and args.object_hold_costs is not False))
        optimized = (args.absolute_policy_optimization if args.absolute_policy_optimization is not None
                     else goal_options['stable_value_learning'])
        goal_options['absolute_policy_optimization'] = optimized
        if optimized:
            paper_critic = args.paper_critic if args.paper_critic is not None else not bool(args.resume)
            goal_options['paper_critic'] = paper_critic
            goal_options.update(cost_penalty=.05 if args.actor_cost_weight is None else args.actor_cost_weight,
                                include_delay_ready=True,
                                actor_lr_scale=.25,
                                critic_td_steps=(1 if paper_critic else 8) if args.critic_td_steps is None else args.critic_td_steps,
                                input_joint_velocity_std_floor_rad_s=(goal_options['cost_velocity_limit_rad_s']
                                    if args.joint_velocity_norm_floor_rad_s is None else args.joint_velocity_norm_floor_rad_s))
    if not args.resume:
        if args.warmup_joint_amplitude_deg is not None:
            goal_options['warmup_joint_amplitude_rad'] = math.radians(args.warmup_joint_amplitude_deg)
        if args.warmup_joint_segment_steps is not None:
            goal_options['warmup_joint_segment_steps'] = args.warmup_joint_segment_steps
        if args.wm_only_iterations is not None:
            goal_options['wm_only_iters'] = args.wm_only_iterations
    if not args.resume:
        from viserdex.supermimic.position_boundary import policy_boundary_options
        goal_options.update(policy_boundary_options(app.cfg))
    cfg = factory(
        **goal_options,
        adaptive_replay=args.adaptive_replay, adaptive_grasps=args.adaptive_grasps,
        joint_velocity_scale=args.joint_velocity_scale,
        include_object_position=not args.omit_object_position,
        geometry_enabled=not revised,
        compile_geometry=not revised and not args.eager_geometry,
        grouped_physical_loss=True,
        microbatch=args.microbatch,
        wm_microbatch=args.wm_microbatch,
        actor_microbatch=args.actor_microbatch,
        critic_microbatch=args.critic_microbatch,
        amp=args.amp,
    )
    if args.dex_reference_config is not None:
        from dataclasses import replace
        from viserdex.supermimic.dex_reference import load_profile, profile_options
        profile, digest = load_profile(args.dex_reference_config)
        overrides = profile_options(profile, digest)
        if args.orientation_reward_mode is not None:
            overrides['orientation_reward_mode'] = args.orientation_reward_mode
        if args.orientation_tolerance_deg is not None:
            overrides['orientation_tolerance_rad'] = math.radians(args.orientation_tolerance_deg)
        if args.wm_holdout_fraction is not None:
            overrides['wm_holdout_fraction'] = args.wm_holdout_fraction
        if args.replay_sampling_mode is not None:
            overrides['replay_sampling_mode'] = args.replay_sampling_mode
        cfg = replace(cfg, **overrides)
    if args.check_joint_update:
        cfg.wm_only_iters = 0
    if args.smoke:
        # Preserve production network widths, k=8 and H=8. Only counts are reduced.
        cfg.wm_batch = 32
        cfg.actor_batch = 32
        cfg.wm_updates = 1
        cfg.wm_only_iters = 0
        cfg.warmup_steps = 16
        cfg.buffer_steps = 64
        cfg.critic_epochs = 1
        cfg.critic_minibatches = 2
    if args.buffer_steps is not None:
        from dataclasses import replace
        cfg = replace(cfg, buffer_steps=args.buffer_steps)
    payload = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if payload:
        if payload.get("metadata", {}).get("actor_only_preview") or payload.get("metadata", {}).get("actor_only_export"):
            raise ValueError("Actor-only appearance preview cannot resume training; use a full training checkpoint")
        if payload.get("metadata", {}).get("task_id") != TASK_ID:
            raise ValueError(f"Resume requires task {TASK_ID}")
        if payload.get("version") != checkpoint_version(Config(**payload["config"])):
            raise ValueError("Checkpoint version differs from the selected policy variant")
        cfg = Config(**payload["config"])
        if args.resume_speed_cost_override:
            from dataclasses import replace
            cfg = replace(cfg, cost_velocity_limit_rad_s=math.radians(args.joint_speed_limit_deg_s),
                          cost_target_rate_limit_rad_s=math.radians(args.target_rate_limit_deg_s))
        if args.resume_gamma_override:
            cfg.gamma = args.gamma
        if args.resume_microbatch_override:
            from viserdex.supermimic.trainer import MICROBATCH_FIELDS
            for name in MICROBATCH_FIELDS:
                if getattr(args, name) is not None:
                    setattr(cfg, name, getattr(args, name))
        if (args.adaptive_replay and not cfg.adaptive_replay) or (args.adaptive_grasps and not cfg.adaptive_grasps):
            raise ValueError("Resume cannot silently change adaptive sampling configuration")
    from viserdex.supermimic.position_boundary import verify_policy_boundary
    verify_policy_boundary(cfg, app.cfg)
    if args.object_hold_costs is not None and cfg.object_hold_costs_enabled != args.object_hold_costs:
        raise ValueError('Object cost heads cannot be changed on resume or enabled outside the 25Hz constrained profile')
    for requested, saved in ((None if args.warmup_joint_amplitude_deg is None else math.radians(args.warmup_joint_amplitude_deg), cfg.warmup_joint_amplitude_rad),
                             (args.warmup_joint_segment_steps, cfg.warmup_joint_segment_steps),
                             (args.wm_only_iterations, cfg.wm_only_iters)):
        if requested is not None and requested != saved:
            raise ValueError('Warmup excitation/schedule cannot silently change on resume')
    if cfg.warmup_joint_amplitude_rad and args.warmup_grasp_bank:
        raise ValueError('HOME excitation cannot be combined with grasp-bank warmup')
    if args.wm_task_losses is not None and cfg.wm_task_losses != args.wm_task_losses:
        raise ValueError('WM task/termination contract cannot change on resume; start fresh for v28')
    if args.paper_critic is not None and cfg.paper_critic != args.paper_critic:
        raise ValueError('Critic learning protocol cannot change on resume or outside the absolute-position profile')
    if args.stable_value_learning is not None and cfg.stable_value_learning != args.stable_value_learning:
        raise ValueError('Value/cost semantics cannot change on resume; start fresh for v29')
    if args.absolute_policy_optimization is not None and cfg.absolute_policy_optimization != args.absolute_policy_optimization:
        raise ValueError('Absolute exploration/optimization protocol cannot change on resume; start fresh for v30')
    if args.actor_cost_weight is not None and (not cfg.absolute_policy_optimization
                                              or cfg.cost_penalty != args.actor_cost_weight):
        raise ValueError('Actor cost weight is a fresh-v30 setting; cannot silently override a saved run')
    if args.critic_td_steps is not None and cfg.critic_td_steps != args.critic_td_steps:
        raise ValueError('Critic TD window cannot silently change on resume or outside the v30 profile')
    if (args.joint_velocity_norm_floor_rad_s is not None
            and cfg.input_joint_velocity_std_floor_rad_s != args.joint_velocity_norm_floor_rad_s):
        raise ValueError('Velocity normalization floor cannot silently change on resume or outside the v30 profile')
    if args.calibrate_termination and (not payload or not cfg.wm_task_losses or args.evaluate):
        raise ValueError('--calibrate_termination requires a resumed v28 checkpoint, without --evaluate')
    from viserdex.supermimic.fixed_palm_reset import reset_fixed_palm_v1, validate_fixed_palm_policy
    if app.cfg.events.reset_robot_joints.func in (
            reset_fixed_palm_v1, 'viserdex.supermimic.fixed_palm_reset:reset_fixed_palm_v1'):
        validate_fixed_palm_policy(cfg)
    noise = PerceptionNoiseCfg(**payload["noise_config"]) if payload else PerceptionNoiseCfg()
    if payload is None and cfg.network_revision == 'hardware_center25_v3':
        # No measured encoder latency distribution: do not invent a 40 ms lag.
        noise.joint_delay_max_steps = 0
        if cfg.dex_reference:
            # Physical episode zero replaces the older normalized Gaussian bias.
            noise.joint_bias_std = 0.
    if args.wm_cuda_graph and (world_size != 1 or (cfg.wm_microbatch or cfg.microbatch) < cfg.wm_batch):
        raise ValueError("--wm_cuda_graph requires one GPU and wm_microbatch >= wm_batch")
    env = gym.make(TASK_ID, cfg=app.cfg).unwrapped
    env._supermimic_solver_override_proof = getattr(binding, 'override_proof', None)
    if abs(env.step_dt-cfg.control_dt) > 1e-9:
        raise ValueError("Environment control period differs from checkpoint/config")
    stopping = [False]

    def stop(signum, frame):
        stopping[0] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        clean, _ = env.reset()
        adapter = Adapter(env, noise, cfg.history, policy_cfg=cfg)
        adapter.audit_observation(clean)
        trainer = Trainer(
            cfg,
            noise,
            adapter.context,
            adapter.limits,
            adapter.base_inverse,
            env.device,
            env.max_episode_length,
            distributed=distributed,
            geometry_spec=adapter.geometry_spec,
            wm_cuda_graph=args.wm_cuda_graph,
        )
        if args.grasp_bank:
            from viserdex.supermimic.grasp_bank import GraspBank, live_contract
            from viserdex.supermimic.grasp_runtime import GraspResetManager, synchronize_reset
            bank = GraspBank(args.grasp_bank, live_contract(adapter), args.grasp_split)
            trainer.grasp_manager = GraspResetManager(adapter, bank, args.seed, cfg.adaptive_grasps, args.evaluate)
        binding_metadata = environment_metadata(app.cfg)
        metadata = dict(
            task_id=TASK_ID,
            environment_contract=binding_metadata['contract_sha256'],
            environment_binding=binding_metadata,
            object_asset=object_metadata(app.cfg.scene.object.spawn.usd_path),
            joint_names=list(adapter.joint_names),
            context_schema=adapter.context_schema,
            geometry_digest=adapter.geometry.digest,
            geometry_point_count=len(adapter.geometry.point_names),
            geometry_feature_dim=adapter.geometry.feature_dim if cfg.geometry_enabled else 0,
            geometry_audit_feature_dim=adapter.geometry.feature_dim,
            network_revision=cfg.network_revision,
            position_boundary=dict(mode=cfg.drop_boundary_mode,
                position_reference='object_center_of_mass' if cfg.drop_boundary_mode == 'com_ellipsoid' else 'object_root',
                center_m=cfg.drop_ellipsoid_center_m if cfg.drop_boundary_mode == 'com_ellipsoid' else cfg.hand_center_base_m,
                semi_axes_m=cfg.drop_ellipsoid_semi_axes_m if cfg.drop_boundary_mode == 'com_ellipsoid' else (cfg.drop_radius_m,)*3,
                object_com_local_m=cfg.drop_object_com_local_m, config_sha256=cfg.drop_boundary_sha256),
            wm_supervision_protocol=('analytic_task_consistency_learned_termination_v1' if cfg.wm_task_losses
                                     else 'physical_state_loss_analytic_survival_v1'),
            imagined_survival=('learned survival inside configured boundary; zero outside' if cfg.wm_task_losses and (cfg.stable_value_learning or cfg.drop_boundary_mode == 'com_ellipsoid')
                               else '1 - sigmoid(learned_logit / temperature + bias)' if cfg.wm_task_losses
                               else 'analytic survival inside COM ellipsoid; zero outside' if cfg.drop_boundary_mode == 'com_ellipsoid'
                               else 'sigmoid((drop_radius - distance) / survival_width)'),
            critic_training_protocol=('imagined_lambda_linear_online_critic_MSE_no_EMA_v32' if cfg.paper_critic
                                      else 'real_pre_reset_TD_bounded_risk_BCE_EMA_v29' if cfg.stable_value_learning
                                      else 'imagined_lambda_online_critic'),
            training_cost_transform=('raw/(1+raw) for four running costs; each head pays absorbing worst-case 1/(1-gamma) once at failure'
                                     if cfg.stable_value_learning else 'raw squared violations'),
            network_input_scaling=dict(position=cfg.input_position_scale,
                linear_velocity=cfg.input_linear_velocity_scale, angular_velocity=cfg.input_angular_velocity_scale,
                joint_velocity=cfg.joint_velocity_scale, joint_position='encoder bounds affine to [-1,1]',
                context='mean absolute scale then RunningNorm', learned_normalization='real data RunningNorm'),
            observation_protocol=cfg.observation_protocol,
            observation_dim=cfg.observation_dim,
            observation_fields=observation_fields(cfg.include_object_position) if args.policy_variant == 'manual_convex' else None,
            joint_velocity_scale=cfg.joint_velocity_scale,
            quaternion_order='XYZW', rotation_6d_order='first two matrix columns, column-major',
            joint_limits_rad=adapter.limits[0].cpu().tolist(),
            sdk_software_limits=getattr(adapter, 'sdk_limit_source', None),
            training_soft_limits=getattr(adapter, 'training_limit_source', None),
            manual_asset_audit=getattr(adapter, 'manual_asset_audit', None),
            seed=args.seed + rank,
            num_envs=args.num_envs,
            rank=rank,
            world_size=world_size,
            total_num_envs=args.num_envs * world_size,
            batch_scope="per_rank; gradients averaged before clipping and Adam",
            global_wm_batch=cfg.wm_batch * world_size,
            global_actor_batch=cfg.actor_batch * world_size,
            smoke=args.smoke or bool(payload and payload["metadata"].get("smoke")),
            check_joint_update=args.check_joint_update
            or bool(payload and payload["metadata"].get("check_joint_update")),
            forced_reset_audit=args.audit or bool(payload and payload["metadata"].get("forced_reset_audit")),
            grasp_bank=(dict(path=bank.path, sha256=bank.sha256, split=bank.split, adaptive=cfg.adaptive_grasps,
                             statistics_scope="single_process") if args.grasp_bank else None),
            parent_checkpoint=str(args.resume) if args.resume else None,
            actor_transfer=False,
            imagination_protocol=imagination_protocol(cfg),
            action_parameterization=("tanh; affine decode to independent training encoder limits; no clamp/filter/warmup"
                                     if cfg.action_mode == 'direct_limits' else "tanh; home_offset_rad; EMA; 0.4s default-pose warmup"),
            absolute_policy_optimization=(dict(protocol='absolute_position_v30',
                initialization='geometry HOME minus encoder bias; constant absolute target',
                exploration=('AR(1) additive angle offset reflected at encoder limits; executed EMA target has a slew limit'
                             if cfg.target_slew_rate_rad_s else
                             'AR(1) additive angle offset reflected at encoder limits; no target rate limiter'),
                exploration_std_rad=cfg.exploration_position_std_rad,
                exploration_correlation_s=cfg.exploration_correlation_s,
                imagination='same stochastic absolute command transform with replayed noise memory',
                actor_cost_weight=cfg.cost_penalty, actor_logit_penalty=cfg.actor_logit_penalty,
                critic_td_steps=cfg.critic_td_steps,
                input_joint_velocity_std_floor_rad_s=cfg.input_joint_velocity_std_floor_rad_s,
                critic_sampling='uniform real windows; stop at done; bootstrap pre-reset timeout state',
                actor_logit_threshold=cfg.actor_logit_threshold,
                actor_gradient_audit='first actor microbatch, pre-clipping/pre-Adam')
                if cfg.absolute_policy_optimization else None),
            action_home=list(cfg.action_home), action_offset_scale=cfg.action_offset_scale,
            action_warmup_s=cfg.action_warmup_s,
            hand_pose_weight=cfg.hand_pose_weight,
            hand_pose_reference="actual post-physics joint position minus default joint position; squared sum",
            home_pose_preference=dict(weight=cfg.home_pose_weight, deadband_rad=cfg.home_pose_deadband_rad,
                scale_rad=cfg.home_pose_scale_rad, reference='geometric reset HOME in policy joint order; mean square excess'),
            target_slew_rate_rad_s=cfg.target_slew_rate_rad_s,
            control_hz=1/cfg.control_dt,
            goal_success_protocol=(("single_pose_dense_persistent_hold_v4" if cfg.track_object_position
                                    else "single_orientation_dense_persistent_hold_v5") if args.policy_variant == 'manual_convex'
                                   else "angle_speed_consecutive_hold_v1"),
            goal_mode=cfg.goal_mode,
            track_object_position=cfg.track_object_position,
            goal_evaluation_s=cfg.goal_evaluation_s,
            reward_protocol=(quiet_hold_reward_protocol(cfg) if cfg.quiet_hold_enabled else ORIENTATION_REWARD_PROTOCOL + '; '
                             + ('orientation_only_reward; object_stability_in_costs_v1' if cfg.object_hold_costs_enabled
                                else POSITION_REWARD_PROTOCOL + '; ' + DENSE_REWARD_PROTOCOL if cfg.track_object_position
                                else ORIENTATION_HOLD_REWARD_PROTOCOL)
                             + ("; separate_velocity_target_rate_costs" if cfg.constraints_enabled else "; joint_velocity_action_difference")
                             if args.policy_variant == 'manual_convex' else "legacy_dense_orientation_success"),
            cost_protocol=('joint_global_object_motion_and_gated_hold_costs_v2' if cfg.object_motion_angular_speed_rad_s else
                           'joint_and_orientation_gated_object_hold_costs_v1' if cfg.object_hold_costs_enabled
                           else 'joint_motion_costs_v1' if cfg.constraints_enabled else None),
            cost_names=cost_names(cfg) if cfg.constraints_enabled else (),
            critic_outputs=1 + len(cost_names(cfg)) if cfg.constraints_enabled else 1,
            timeout_semantics=(f"{app.cfg.episode_length_s:g}s collection truncation; bootstrap continuing hold; evaluation time has no dynamics/reward effect"
                               if args.policy_variant == 'manual_convex' else "legacy_episode_timeout"),
            state_dim=cfg.state_dim,
            state_protocol=cfg.state_protocol,
            state_fields=state_fields() if cfg.compact_state else None,
            context_manifest=getattr(adapter, "context_manifest", None),
            identified_hand=getattr(adapter, "identified_hand_audit", None),
            hand_center_base_m=list(cfg.hand_center_base_m),
            control_timing=dict(policy_dt=cfg.control_dt, pd_dt=app.cfg.sim.dt,
                decimation=app.cfg.decimation, solver_dt=app.cfg.sim.dt/app.cfg.sim.physics.num_substeps),
            hold_steps=round(cfg.hold_duration/cfg.control_dt),
            source_root=str(ROOT),
            pid=os.getpid(),
            source_revision=os.environ.get("VISERDEX_SOURCE_REVISION"),
            resume_semantics="weights+optimizers; fresh simulator and replay warmup",
            physics_health=getattr(adapter, 'physics_health_contract', None),
            warmup_request=dict(bank=str(args.warmup_grasp_bank) if args.warmup_grasp_bank else None,
                                split='train', grasp_steps=args.warmup_grasp_steps,
                                grasp_wm_updates=args.warmup_grasp_updates,
                                exploration=args.warmup_grasp_exploration,
                                readiness_rounds=args.wm_readiness_rounds,
                                readiness_updates=args.wm_readiness_updates,
                                consecutive_passes=args.wm_readiness_passes,
                                resume_actor_ramp_iterations=args.resume_actor_ramp_iterations,
                                formal_initialization='ordinary in-hand reset; temporary bank replay discarded'),
            inherited_grasp_warmup=((payload['metadata'].get('wm_readiness', {}).get('grasp')
                                    or payload['metadata'].get('inherited_grasp_warmup')) if payload else None),
        )
        if cfg.dex_reference:
            from viserdex.supermimic.dex_reference import state_fields as reference_fields, observation_fields as reference_obs
            phys_out_dim = 27 if cfg.wm_dynamics_mode == 'velocity_integrated' else 54
            wm_proto = (
                f'velocity27_integrated_{cfg.wm_loss_normalization}_contact_{cfg.contact_loss_reduction}_v34'
                if cfg.uses_v34_optimizations
                else 'physical54_dimension_mean_contact_BCE_sum_0.1_independent_failure_v33'
            )
            metadata.update(state_fields=reference_fields(), observation_fields=reference_obs(),
                contact_supervision=adapter.contact_labels.manifest,
                wm_supervision_protocol=wm_proto,
                action_parameterization='tanh -> YAML soft encoder bounds -> EMA -> subtract episode zero -> geometric PD, zero delay',
                wm_input_dim=len(trainer.wm.input_norm.mean), wm_output_dim=phys_out_dim+len(cfg.contact_body_names),
                fixed_encoder_compensation_enabled=False,
                control_profile=cfg.control_profile, control_profile_sha256=cfg.control_profile_sha256)
        if args.policy_variant == 'manual_convex':
            from viserdex.supermimic.goal_contract import goal_position_metadata
            metadata.update(goal_position_metadata(app.cfg))
            torch.testing.assert_close(adapter.state[:, adapter.sl.goal_pos],
                adapter.state.new_tensor(metadata['default_goal_position_base_m']).expand(env.num_envs, -1),
                atol=6e-6, rtol=0)
            from isaaclab_newton.physics import NewtonManager
            from viserdex.supermimic.physics_health import audit_compiled_joint_limits
            metadata['compiled_joint_limits'] = audit_compiled_joint_limits(
                NewtonManager._solver, adapter.joint_names, adapter.manual_asset_audit['limits_rad'])
            metadata['compiled_joint_limits']['sample_phase'] = 'before_first_physics_step'
        if payload:
            trainer.restore(payload, metadata, restore_rng=not args.evaluate and world_size == 1,
                            allow_microbatch_override=args.resume_microbatch_override,
                            allow_gamma_override=args.resume_gamma_override,
                            allow_speed_cost_override=args.resume_speed_cost_override,
                            solver_iteration_override=getattr(binding, 'override_proof', None))
        if trainer.grasp_manager is not None:
            # Restore statistics before drawing the new simulator initial pairs.
            clean, _ = env.reset()
            synchronize_reset(adapter)
            adapter.audit_observation(clean)
        if world_size > 1:
            # Fresh simulators/replay require independent streams, even when
            # both ranks load the same single-GPU checkpoint and Adam state.
            torch.manual_seed(args.seed + rank)
        metadata["initial_model_digest"] = distributed.assert_models_equal(*trainer.sync_models)
        metadata["execution"] = dict(
            amp=cfg.amp,
            wm_cuda_graph=args.wm_cuda_graph,
            solver_iterations=app.cfg.sim.physics.solver_cfg.iterations,
            solver_substeps=app.cfg.sim.physics.num_substeps,
            replay_storage=args.replay_storage,
            periodic_video=asdict(video_settings),
            replay_rng='sampling distribution preserved; cross-device draws are not bitwise identical',
            wm_microbatch=cfg.wm_microbatch or cfg.microbatch,
            actor_microbatch=cfg.actor_microbatch or cfg.microbatch,
            critic_microbatch=cfg.critic_microbatch or cfg.microbatch,
        )
        metadata["resume_microbatch_override"] = args.resume_microbatch_override
        metadata["resume_speed_cost_override"] = (dict(
            previous={k: payload['config'][k] for k in ('cost_velocity_limit_rad_s', 'cost_target_rate_limit_rad_s')},
            current={k: getattr(cfg, k) for k in ('cost_velocity_limit_rad_s', 'cost_target_rate_limit_rad_s')},
            parent_iteration=payload['iteration'], optimizer_state_preserved=True,
            normalization_floor_preserved=True) if args.resume_speed_cost_override else None)
        metadata["resume_gamma_override"] = (dict(previous=payload["config"]["gamma"], current=cfg.gamma,
            policy_optimizer_steps=dict(actor=payload["optimizer_steps"]["actor"], critic=payload["optimizer_steps"]["critic"]))
            if args.resume_gamma_override else None)
        dump_yaml(str(rank_dir / "env.yaml"), app.cfg)
        (rank_dir / "geometry.json").write_text(json.dumps(adapter.geometry_spec, indent=2))
        (rank_dir / "config.json").write_text(
            json.dumps(dict(metadata=metadata, config=asdict(cfg), noise_config=asdict(noise)), indent=2)
        )
        def new_replay():
            return Replay(cfg.buffer_steps, env.num_envs, env.device, cfg.event_capacity, cfg.event_fraction,
                          cfg.wm_window, adaptive=cfg.adaptive_replay,
                          storage_device=env.device if args.replay_storage == 'cuda' else 'cpu',
                          wm_holdout_envs=holdout_env_count(env.num_envs, cfg.wm_holdout_fraction),
                          sampling_mode=cfg.replay_sampling_mode,
                          failure_sample_fraction=cfg.failure_sample_fraction)
        replay = new_replay()
        stream_audit = StreamAudit(rank_dir / "stream_audit.jsonl", adapter.context, adapter.limits,
                                   cfg, adapter.geometry_spec, args.audit_interval)
        start = time.monotonic()
        step = 0
        collection_phase = 'evaluation' if args.evaluate else 'initial_collection'
        episode_goals = torch.zeros(env.num_envs, device=env.device)
        exploration_noise = torch.zeros(env.num_envs, 21, device=env.device)
        from viserdex.supermimic.warmup_excitation import HomeExcitation
        warmup_excitation = HomeExcitation(adapter.limits, cfg) if cfg.warmup_joint_amplitude_rad else None

        def collect(count, evaluate=False, action_provider=None, policy_statistics=True, audit_observe=True,
                    update_statistics=True, record_observer=None):
            nonlocal exploration_noise
            nonlocal step
            position_metric = 'position' if cfg.track_object_position else 'hand_center_distance'
            result = {
                "reward_sum": 0.0,
                "hand_pose_cost_sum": 0.0,
                "hand_pose_reward_sum": 0.0,
                "action_saturation_sum": 0.0,
                "action_near_bound_sum": 0.0,
                "action_tanh_slope_sum": 0.0,
                "action_tanh_slope_steps": 0,
                "terminated": 0,
                "timeouts": 0,
                "successes": 0,
                "drops": 0,
                "completed_episode_goals": 0.0,
                "task_samples": 0,
                "task_angle_sum_rad": 0.,
                f"task_{position_metric}_sum_m": 0.,
                "task_stable_samples": 0,
                "post_evaluation_samples": 0,
                "warmup_excitation_steps": 0,
            }
            with torch.no_grad():
                for _ in range(count):
                    if args.audit and not evaluate and step == 2:
                        env.episode_length_buf[: max(1, env.num_envs // 2)] = env.max_episode_length - 1
                        adapter.state = adapter.read_state()
                    previous_exploration_noise = exploration_noise
                    policy_action = None
                    if action_provider is not None:
                        action = action_provider()
                        exploration_noise = torch.zeros_like(exploration_noise)
                    elif evaluate:
                        action = trainer.actor(adapter.history)
                        policy_action = action
                    elif warmup_excitation is not None and trainer.iteration < cfg.wm_only_iters:
                        action = warmup_excitation.action(trainer.iteration)
                        result['warmup_excitation_steps'] += 1
                        exploration_noise = torch.zeros_like(exploration_noise)
                        previous_exploration_noise = exploration_noise
                    elif cfg.absolute_policy_optimization:
                        from viserdex.supermimic.absolute_policy import explore_absolute
                        policy_action = trainer.actor(adapter.history)
                        action, exploration_noise = explore_absolute(policy_action,
                            exploration_noise, adapter.limits, cfg)
                    else:
                        action = trainer.actor.sample(adapter.history, cfg.exploration)
                        # Historical exploration is before tanh, so the sampled
                        # action is the actual tanh output in this branch.
                        policy_action = action
                    try:
                        record, target, observation, clean = adapter.step(action)
                    except RuntimeError:
                        if hasattr(adapter, 'physics_failure'):
                            failure = dict(adapter.physics_failure, iteration=trainer.iteration,
                                           collection_step=step, parent_checkpoint=str(args.resume),
                                           time_unix=time.time(), collection_phase=collection_phase)
                            (rank_dir / 'physics_failure.json').write_text(json.dumps(failure, indent=2))
                            torch.save(dict(state=adapter.state.detach().cpu(), action=action.detach().cpu(),
                                            next_physical=(adapter.captured_physical.detach().cpu()
                                                           if adapter.captured_physical is not None else None),
                                            context=adapter.context.detach().cpu(),
                                            diagnostics=adapter.physics_failure_snapshot()),
                                       rank_dir / 'physics_failure_state.pt')
                        raise
                    if step < 4:
                        adapter.audit_observation(clean)
                    if cfg.absolute_policy_optimization:
                        record['exploration_noise_rad'] = previous_exploration_noise.clone()
                        exploration_noise = torch.where(record['done'][:, None], 0., exploration_noise)
                    if record_observer is not None:
                        record_observer(record)
                    if warmup_excitation is not None:
                        warmup_excitation.reset(record['done'])
                    if not evaluate:
                        if audit_observe:
                            stream_audit.observe(trainer.wm, record, trainer.iteration,
                                                 policy_action=policy_action)
                        replay.add(record)
                        if update_statistics:
                            trainer.update_real_statistics(record["state"], target, record["next_physical"], observation,
                                                           policy_statistics=policy_statistics)
                        trainer.real_steps += 1
                    result["action_saturation_sum"] += float((action.abs()>1).float().mean())
                    result["action_near_bound_sum"] += float((action.abs()>.98).float().mean())
                    # Post-tanh noise/reflection can move a saturated output
                    # inward without restoring its derivative. External action
                    # providers have no known tanh derivative.
                    if policy_action is not None:
                        result["action_tanh_slope_sum"] += float((1-policy_action.square()).mean())
                        result["action_tanh_slope_steps"] += 1
                    result["reward_sum"] += float(record["reward"].mean())
                    pose_cost = float(adapter.captured_hand_pose.mean())
                    result["hand_pose_cost_sum"] += pose_cost
                    result["hand_pose_reward_sum"] -= cfg.control_dt * cfg.hand_pose_weight * pose_cost
                    if cfg.home_pose_weight:
                        from viserdex.supermimic.quiet_hold import home_pose_penalty
                        preference = home_pose_penalty(record['next_physical'][:, :21], trainer._wm_home, cfg)
                        result['home_pose_penalty_sum'] = result.get('home_pose_penalty_sum', 0.) + float(preference.sum())
                        result['home_pose_reward_sum'] = result.get('home_pose_reward_sum', 0.) - cfg.control_dt * float(preference.mean())
                    result["terminated"] += int(record["terminated"].sum())
                    result["timeouts"] += int((record["done"] & ~record["terminated"]).sum())
                    error = adapter.captured_orientation_error
                    if cfg.goal_mode == 'single_dense_hold':
                        stable = ((error < cfg.hold_angle_threshold)
                                  & (adapter.captured_angular_speed < cfg.hold_speed_threshold)
                                  & position_hold_valid(adapter.captured_position_error, adapter.captured_linear_speed, cfg)
                                  & ~record['done'])
                        result['task_samples'] += error.numel()
                        result['task_angle_sum_rad'] += float(error.sum())
                        from viserdex.supermimic.spatial_contract import hand_center_distance
                        distance = (adapter.captured_position_error if cfg.track_object_position
                                    else hand_center_distance(record['next_physical'], cfg))
                        result[f'task_{position_metric}_sum_m'] += float(distance.sum())
                        result['task_stable_samples'] += int(stable.sum())
                        result['post_evaluation_samples'] += int((record['goal_elapsed_s'] >= cfg.goal_evaluation_s).sum())
                    if cfg.constraints_enabled:
                        from viserdex.supermimic.constraints import violation_costs, object_hold_gate
                        cost, ratios = violation_costs(record['state'], record['next_physical'], record['action'], adapter.limits, cfg)
                        for i, name in enumerate(cost_names(cfg)):
                            result['cost_' + name + '_sum'] = result.get('cost_' + name + '_sum', 0.) + float(cost[:,i].sum())
                            result['cost_' + name + '_violations'] = result.get('cost_' + name + '_violations', 0) + int((ratios[:,i]>1).sum())
                            result['cost_' + name + '_max_ratio'] = max(result.get('cost_' + name + '_max_ratio', 0.), float(ratios[:,i].max()))
                        if cfg.object_hold_costs_enabled:
                            gate = object_hold_gate(record['state'], record['next_physical'], cfg)
                            result['object_hold_cost_gate_sum'] = result.get('object_hold_cost_gate_sum', 0.) + float(gate.sum())
                    reached = record["success"]
                    result["successes"] += int(reached.sum())
                    episode_goals.add_(reached)
                    result["completed_episode_goals"] += float(episode_goals[record["done"]].sum())
                    episode_goals[record["done"]] = 0
                    result["drops"] += int(
                        ((drop_distance(record["next_physical"], cfg) > cfg.drop_radius_m) & record["terminated"]).sum()
                    )
                    step += 1
            completed = result["terminated"] + result["timeouts"]
            result["goals_per_completed_episode"] = result["completed_episode_goals"] / completed if completed else None
            slope_steps = result['action_tanh_slope_steps']
            result['action_tanh_slope_mean'] = result['action_tanh_slope_sum'] / slope_steps if slope_steps else None
            samples = result['task_samples']
            result['task_mean_angle_rad'] = result['task_angle_sum_rad'] / samples if samples else None
            result[f'task_mean_{position_metric}_m'] = result[f'task_{position_metric}_sum_m'] / samples if samples else None
            result['task_stable_fraction'] = result['task_stable_samples'] / samples if samples else None
            if cfg.object_hold_costs_enabled:
                result['object_hold_cost_gate_mean'] = result.get('object_hold_cost_gate_sum', 0.) / samples if samples else None
            return result

        if args.calibrate_termination:
            if distributed.enabled:
                raise ValueError('Frozen termination calibration currently requires single-process execution')
            from viserdex.supermimic.termination_calibration import FrozenTerminationCalibration
            import hashlib
            calibration = FrozenTerminationCalibration(trainer.wm, adapter.context, adapter.limits, cfg)
            collection_phase = 'frozen_termination_calibration'
            real = collect(args.eval_steps, evaluate=True, record_observer=calibration.observe)
            report, samples = calibration.finish()
            report.update(source_checkpoint=str(args.resume.resolve()),
                          source_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                          real_collection=real, optimizer_steps=dict(trainer.optimizer_steps))
            torch.save(samples, rank_dir / 'termination_calibration_samples.pt')
            if report['accepted']:
                trainer.wm.termination_temperature.fill_(report['temperature'])
                trainer.wm.termination_bias.fill_(report['bias'])
                metadata['termination_calibration'] = report
                trainer.save(rank_dir / 'calibrated.pt', metadata)
            (rank_dir / 'termination_calibration.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return
        if args.evaluate:
            metrics = distributed.metrics(collect(args.eval_steps, True))
            (rank_dir / "evaluation.json").write_text(
                json.dumps(
                    dict(metrics=metrics, grasp_sampling=trainer.grasp_manager.state_dict() if trainer.grasp_manager else None, max_errors=adapter.max_errors, contact_peaks=getattr(adapter, 'contact_peaks', None),
                         object_support_reset=getattr(env, 'object_support_audit', None),
                         elapsed=time.monotonic() - start), indent=2
                )
            )
            if rank == 0:
                print(json.dumps(metrics), flush=True)
            return
        if (payload and not args.benchmark_execution and not metadata['smoke']
                and not metadata['check_joint_update'] and video_settings.due(trainer.iteration)):
            video_error = None
            if rank == 0:
                try:
                    # Preserve restored weights/normalizers before warmup, but
                    # record this run's verified environment/allowed overrides.
                    boundary = args.output_dir / f'model_{trainer.iteration}.pt'
                    trainer.save(boundary, metadata)
                    result = record_resume_boundary(boundary, args.output_dir / 'env.yaml',
                        args.output_dir, trainer.iteration, video_settings, device=env.device, repo=ROOT)
                    print(json.dumps(dict(resume_boundary_video=dict(iteration=trainer.iteration,
                        status=result['status'], video=result['video']))), flush=True)
                except Exception as exc:
                    video_error = f'{type(exc).__name__}: {exc}'
            if distributed.any(video_error is not None):
                raise RuntimeError(video_error or 'Resume boundary video failed on rank zero; inspect videos/*.log')
        warmup_summary = dict(parent_iteration=trainer.iteration, actor_optimizer_before=trainer.optimizer_steps['actor'],
                              critic_optimizer_before=trainer.optimizer_steps['critic'])
        if args.warmup_grasp_bank:
            collection_phase = 'grasp_world_model_warmup'
            from viserdex.supermimic.grasp_bank import GraspBank, live_contract
            from viserdex.supermimic.grasp_runtime import synchronize_reset
            from viserdex.supermimic.warmup import WarmupGraspResetManager
            warmup_bank = GraspBank(args.warmup_grasp_bank, live_contract(adapter), 'train')
            manager = WarmupGraspResetManager(adapter, warmup_bank, args.seed, args.warmup_grasp_exploration)
            try:
                env.reset(); synchronize_reset(adapter)
                real = collect(args.warmup_grasp_steps, action_provider=manager.action,
                               policy_statistics=False, audit_observe=False)
                fit = trainer.update_world_model(replay, updates=args.warmup_grasp_updates)
                warmup_summary['grasp'] = dict(manager.metadata(), real=real, fit=fit,
                                               transitions=args.warmup_grasp_steps * env.num_envs)
            finally:
                manager.detach()
            # Bank transitions have trained the WM; formal replay starts from ordinary hand resets.
            del replay
            replay = new_replay()
            env.reset(); synchronize_reset(adapter)
            episode_goals.zero_()
        collection_phase = 'ordinary_inhand_warmup'
        collect(cfg.warmup_steps, policy_statistics=not bool(args.wm_readiness_rounds),
                audit_observe=not bool(args.wm_readiness_rounds))
        if args.wm_readiness_rounds:
            collection_phase = 'ordinary_inhand_readiness'
            from viserdex.supermimic.warmup import readiness_decision
            consecutive = 0
            warmup_summary['readiness'] = []
            with (rank_dir / 'warmup_metrics.jsonl').open('a', buffering=1) as warmup_log:
                for round_index in range(args.wm_readiness_rounds):
                    if stopping[0]:
                        break
                    fit = trainer.update_world_model(replay, updates=args.wm_readiness_updates)
                    stream_audit.start(trainer.wm, trainer.iteration,
                                       environment_ids=torch.arange(env.num_envs, device=env.device))
                    real = collect(32, policy_statistics=False, update_statistics=False)
                    result = stream_audit.latest_result
                    decision = readiness_decision(result)
                    consecutive = consecutive + 1 if decision['passed'] else 0
                    row = dict(round=round_index, iteration=trainer.iteration, phase='wm_readiness',
                               fit=fit, real=real, decision=decision, consecutive_passes=consecutive,
                               elapsed_seconds=time.monotonic()-start, optimizer_steps=dict(trainer.optimizer_steps))
                    warmup_summary['readiness'].append(row)
                    warmup_log.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
                    if consecutive >= args.wm_readiness_passes:
                        break
            warmup_summary['passed'] = consecutive >= args.wm_readiness_passes
            warmup_summary['actor_optimizer_after'] = trainer.optimizer_steps['actor']
            warmup_summary['critic_optimizer_after'] = trainer.optimizer_steps['critic']
            if warmup_summary['passed'] and args.resume_actor_ramp_iterations is not None:
                trainer.arm_resume_actor_ramp(args.resume_actor_ramp_iterations)
                warmup_summary['resume_actor_ramp'] = trainer.resume_actor_ramp
            metadata['wm_readiness'] = warmup_summary
            (rank_dir / 'warmup.json').write_text(json.dumps(warmup_summary, indent=2))
            (rank_dir / 'config.json').write_text(json.dumps(dict(metadata=metadata, config=asdict(cfg), noise_config=asdict(noise)), indent=2))
            trainer.save(rank_dir / 'warmup.pt', metadata)
            if not warmup_summary['passed']:
                raise RuntimeError('WM readiness did not pass; Actor/Critic were kept frozen. See warmup.json')
        collection_phase = 'formal_training'
        if args.benchmark_execution:
            from viserdex.supermimic.execution_benchmark import benchmark_execution
            benchmark_execution(trainer, replay, metadata, rank_dir)
            return
        with (rank_dir / "metrics.jsonl").open("a", buffering=1) as log:
            while trainer.iteration < args.iterations and not distributed.any(stopping[0]):
                trainer.step_credit += 1 / cfg.utd
                count = int(trainer.step_credit)
                trainer.step_credit -= count
                collection_start = time.monotonic()
                real = collect(count)
                collection_seconds = time.monotonic() - collection_start
                before = time.monotonic()
                phase = (("wm_critic_warmup" if cfg.uses_target_critic else "wm_warmup")
                         if trainer.iteration < cfg.wm_only_iters else "joint")
                metrics = trainer.update(replay)
                if replay.wm_holdout_envs > 0:
                    holdout_every = min(args.audit_interval, 50) if args.audit_interval > 0 else 50
                    if trainer.iteration == 1 or trainer.iteration % holdout_every == 0:
                        metrics.update(trainer.evaluate_wm_holdout(replay))
                metrics.update(real)
                metrics.update(
                    iteration=trainer.iteration,
                    phase=phase,
                    real_steps=trainer.real_steps,
                    collection_seconds=collection_seconds,
                    update_seconds=time.monotonic() - before,
                    elapsed_seconds=time.monotonic() - start,
                    gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    replay_allocated_gib=replay.allocated_bytes / 2**30,
                    optimizer_steps=trainer.optimizer_steps,
                    parity_max_errors=adapter.max_errors,
                    resets=adapter.reset_count,
                )
                global_metrics = distributed.metrics(metrics)
                video_due = video_settings.due(trainer.iteration)
                save_model = trainer.iteration % args.save_interval == 0 or video_due
                save_latest = trainer.iteration % args.latest_interval == 0
                if save_model or save_latest:
                    digest = distributed.assert_models_equal(*trainer.sync_models)
                    global_metrics["model_sync_digest"] = digest
                global_metrics["world_size"] = world_size
                global_metrics["total_num_envs"] = env.num_envs * world_size
                global_metrics["real_transitions_since_restart"] = step * env.num_envs * world_size
                if rank == 0:
                    metrics = global_metrics
                line = json.dumps(metrics)
                log.write(line + "\n")
                if rank == 0:
                    if args.console_log == 'pretty':
                        from viserdex.supermimic.console_log import format_training_log
                        print(format_training_log(metrics, args.iterations, count), flush=True)
                    else:
                        print(line, flush=True)
                if rank == 0 and save_model:
                    trainer.save(args.output_dir / f"model_{trainer.iteration}.pt", metadata)
                if rank == 0 and save_latest:
                    trainer.save(args.output_dir / "latest.pt", metadata)
                if video_due:
                    video_error = None
                    if rank == 0:
                        try:
                            video_result = record_checkpoint(
                                args.output_dir / f'model_{trainer.iteration}.pt', args.output_dir / 'env.yaml',
                                args.output_dir, trainer.iteration, video_settings, device=env.device, repo=ROOT)
                            if args.console_log == 'pretty':
                                print(f"[Video] Iteration {trainer.iteration}: {video_result['video']}", flush=True)
                            else:
                                print(json.dumps(dict(periodic_video=video_result)), flush=True)
                        except Exception as exc:
                            video_error = f'{type(exc).__name__}: {exc}'
                    # Every rank waits at a completed-update boundary. Propagate
                    # recording failures together instead of hanging another rank.
                    if distributed.any(video_error is not None):
                        raise RuntimeError(video_error or 'Periodic video failed on rank zero; inspect videos/*.log')
        final_digest = distributed.assert_models_equal(*trainer.sync_models)
        if rank == 0:
            trainer.save(args.output_dir / "latest.pt", metadata)
        if args.holdout:
            rows = []
            with torch.no_grad():
                for _ in range(32):
                    action = trainer.actor(adapter.history)
                    record, _, _, _ = adapter.step(action)
                    keys = ('state', 'action', 'next_physical', 'done', 'reward') + (('terminated',) if cfg.wm_task_losses else ())
                    rows.append({k: record[k] for k in keys})
            if cfg.wm_task_losses:
                from viserdex.supermimic.stream_audit import audit_rollout
                errors = audit_rollout(trainer.wm, rows, adapter.context, adapter.limits, cfg)
            else:
                errors = open_loop_errors(trainer.wm, rows, adapter.context, adapter.limits, cfg)
            (rank_dir / "holdout.json").write_text(json.dumps(errors, indent=2))
        (rank_dir / "validation.json").write_text(
            json.dumps(
                dict(
                    max_errors=adapter.max_errors,
                    contact_peaks=getattr(adapter, 'contact_peaks', None),
                    resets=adapter.reset_count,
                    optimizer_steps=trainer.optimizer_steps,
                    replay_size=replay.size,
                    adaptive_replay=replay.adaptive, event_archive_size=replay.archive_size,
                    grasp_sampling=trainer.grasp_manager.state_dict() if trainer.grasp_manager else None,
                    object_support_reset=getattr(env, 'object_support_audit', None),
                    seconds=time.monotonic() - start,
                    model_sync_digest=final_digest,
                    rank=rank,
                    world_size=world_size,
                ),
                indent=2,
            )
        )
    finally:
        env.close()
        app.close()
        distributed.close()


if __name__ == "__main__":
    main()
