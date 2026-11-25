# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train the pose estimator online using rollouts from a trained manipulation policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

# local imports
from viserdex.utils import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=6200, help="Number of steps to run the agent.")
parser.add_argument("--task", type=str, default="Isaac-ViserDex-Repose-Revo3-Estimator", help="Name of the task.")
parser.add_argument("--policy_run", type=str, default=None, help="Manipulation policy run used for rollouts.")
parser.add_argument("--teacher_checkpoint", type=Path, help="Frozen teacher checkpoint with its saved environment.")
parser.add_argument("--hikrobot_calibration", type=Path, help="Use calibrated object-only GS and live hand-depth images.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_batches", type=int, default=16, help="CNN mini-batches per rollout step.")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--decay_steps", type=int, default=200)
parser.add_argument("--min_lr_factor", type=float, default=0.2)
parser.add_argument("--save_interval", type=int, default=2000, help="Estimator calls between checkpoints.")
parser.add_argument("--log_interval", type=int, default=100)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3")
args_cli = parse_runtime_args(parser)
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be positive")
if (
    min(
        args_cli.num_batches, args_cli.warmup_steps, args_cli.decay_steps, args_cli.save_interval, args_cli.log_interval
    )
    <= 0
):
    parser.error("Batch counts, schedule lengths and logging/save intervals must be positive")
if args_cli.learning_rate <= 0 or not 0 <= args_cli.min_lr_factor <= 1:
    parser.error("Require learning_rate > 0 and min_lr_factor in [0, 1]")
if "-Eval-" in args_cli.task or "-Camera-" in args_cli.task:
    args_cli.enable_cameras = True

prepared_teacher = binding = None
binding_info = {}
if args_cli.hikrobot_calibration:
    from viserdex.assets.objects import TARGET_OBJECT_CFG
    from viserdex.distillation.environment import SavedEnvironmentBinding
    from viserdex.distillation.teacher import teacher_metadata
    from viserdex.renderer.hikrobot_occlusion import configure_original_capture

    if not args_cli.teacher_checkpoint or args_cli.task != "Isaac-ViserDex-Repose-Revo3":
        parser.error("Calibrated online training requires the base Revo3 task and --teacher_checkpoint")
    if args_cli.num_envs is None:
        parser.error("Specify --num_envs for calibrated online training")
    prepared_teacher = teacher_metadata(args_cli.teacher_checkpoint, TARGET_OBJECT_CFG["name"])
    binding = SavedEnvironmentBinding(prepared_teacher["environment_path"], prepared_teacher["environment_sha256"])


def configure(cfg):
    if binding:
        binding.configure(cfg, args_cli)
        binding_info.update(binding.verify(cfg))
        configure_original_capture(cfg, args_cli.hikrobot_calibration)
        pose = cfg.observations.pose.image_features.params["pose_estimator_cfg"]
        pose.collect_only = False
        pose.train = True


# Configure the saved task and calibrated sensors before the official launcher.
simulation_app = SimulationRuntime(args_cli, configure=configure)

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import torch
import json
import signal
import time
from termcolor import colored
import numpy as np
from datetime import datetime
from tqdm import tqdm

from isaaclab_rl.rsl_rl import create_rsl_rl_runner

from viserdex.utils.log_utils import Tee
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from viserdex.utils.log_utils import create_pose_writer
import viserdex.tasks  # noqa: F401
from viserdex.assets.objects import get_rollout_policy_run


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = simulation_app.env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    # load pre-trained policy for rollouts
    if not hasattr(env_cfg.observations, "pose"):
        raise ValueError(f"Task {args_cli.task} has no pose-estimator observation group.")
    if args_cli.num_batches > env_cfg.scene.num_envs:
        raise ValueError("--num_batches must not exceed --num_envs")
    pose_cfg = env_cfg.observations.pose.image_features.params["pose_estimator_cfg"]
    for name in ("num_batches", "learning_rate", "warmup_steps", "decay_steps", "min_lr_factor", "save_interval"):
        setattr(pose_cfg, name, getattr(args_cli, name))
    # Use the same physics settings as the completed teacher run.
    env_cfg.scene.clone_in_fabric = False
    env_cfg.scene.robot.spawn.activate_contact_sensors = False
    env_cfg.scene.object.spawn.activate_contact_sensors = False
    env_cfg.commands.object_pose.debug_vis = False
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if prepared_teacher:
        resume_path = prepared_teacher["path"]
    elif args_cli.teacher_checkpoint:
        resume_path = str(args_cli.teacher_checkpoint.resolve(strict=True))
    else:
        load_run = get_rollout_policy_run(args_cli.task, args_cli.policy_run or args_cli.load_run)
        resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)

    agent_cfg.experiment_name = "pose_estimator"
    run_name = args_cli.run_name if args_cli.run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "rsl_rl", "pose_estimator", run_name)
    env_cfg.log_dir = log_dir

    # Log stdout to file
    if os.path.exists(log_dir):
        raise FileExistsError(f"Use a new --run_name; refusing to overwrite an existing estimator run: {log_dir}")
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, "launch.json"), "w") as stream:
        json.dump({"arguments": vars(args_cli), "teacher_checkpoint": resume_path,
                   "environment_binding": binding_info, "training_mode": "online_teacher_rollout"},
                  stream, indent=2, default=str)
    log_file_path = os.path.join(log_dir, "stdout.log")
    log_file = open(log_file_path, "a")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(colored(f"[INFO]: Loading experiment from directory: {log_root_path}", "green"))
    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))
    # load previously trained model
    if prepared_teacher:
        from viserdex.distillation.teacher import load_teacher
        from viserdex.renderer.hikrobot_occlusion import audit_original_capture
        from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func

        term = _get_observation_term_func(env.unwrapped.observation_manager, "pose", "image_features")
        camera_audit = audit_original_capture(
            term, term._hand_depth_camera, env.unwrapped.scene["robot"], args_cli.hikrobot_calibration
        )
        with open(os.path.join(log_dir, "camera_audit.json"), "w") as stream:
            json.dump(camera_audit, stream, indent=2)
        policy = load_teacher(prepared_teacher, env.get_observations(), env.unwrapped.device)
    else:
        ppo_runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # initialize writer
    writer = create_pose_writer(log_dir, env_cfg, agent_cfg)

    # reset environment
    obs = env.get_observations()

    # Checkpoint only between completed updates, never inside a CUDA operation.
    pending = {"save": False, "stop": None}

    def request_checkpoint(signum, _frame):
        pending["save"] = True
        if signum != signal.SIGUSR1:
            pending["stop"] = signum

    previous_handlers = {
        sig: signal.signal(sig, request_checkpoint) for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM)
    }
    from viserdex.tasks.manipulation.inhand.mdp.observations import _get_observation_term_func

    estimator = _get_observation_term_func(env.unwrapped.observation_manager, "pose", "image_features").pose_estimator
    metrics_file = open(os.path.join(log_dir, "metrics.jsonl"), "a", buffering=1)
    start_time = time.monotonic()
    step_counter = 0
    with tqdm(total=args_cli.num_steps, desc="Pose estimator rollout") as progress_bar:
        while simulation_app.is_running() and step_counter < args_cli.num_steps:
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, extras = env.step(actions)

            string = f"[INFO]: Step: {step_counter}, "
            metrics = {}
            for key, value in extras["log"].items():
                if "pose_estimator" in key:
                    metrics[key] = float(value)
                    string += f"{key}: {value}, "
                    writer.add_scalar(key, value, step_counter)
            string += "\n----------------------------------\n"
            if step_counter % args_cli.log_interval == 0:
                print(colored(string, "green"))
                first_state = next(iter(estimator.optimizer.state.values()), {})
                optimizer_step = first_state.get("step", 0)
                row = {"step": step_counter, "elapsed_s": time.monotonic() - start_time,
                       "estimator_steps": estimator.step_count, "target_rollout_steps": args_cli.num_steps,
                       "optimizer_steps": int(optimizer_step), **metrics}
                metrics_file.write(json.dumps(row) + "\n")
                with open(os.path.join(log_dir, "progress.json"), "w") as stream:
                    json.dump(row, stream, indent=2)

            step_counter += 1
            progress_bar.update(1)

            if pending["save"]:
                checkpoint = os.path.join(estimator.log_dir, f"cnn_{estimator.step_count}_requested.pth")
                torch.save(estimator.pose_estimator.state_dict(), checkpoint)
                print(f"[INFO]: Pose checkpoint saved to {checkpoint}", flush=True)
                pending["save"] = False
            if pending["stop"] is not None:
                print(f"[INFO]: Stop requested by signal {pending['stop']} after step {step_counter}")
                break

            if step_counter >= args_cli.num_steps:
                print(colored(f"[INFO]: Reached maximum number of steps: {args_cli.num_steps}", "green"))
                break
    # Save short runs on normal completion; periodic saves remain available.
    checkpoint = os.path.join(estimator.log_dir, f"cnn_{estimator.step_count}_final.pth")
    torch.save(estimator.pose_estimator.state_dict(), checkpoint)
    print(f"[INFO]: Pose estimator saved to {checkpoint}")
    with open(os.path.join(log_dir, "completion.json"), "w") as stream:
        json.dump(
            {
                "completed": step_counter == args_cli.num_steps,
                "rollout_steps": step_counter,
                "estimator_steps": estimator.step_count,
                "elapsed_s": time.monotonic() - start_time,
                "checkpoint": checkpoint,
                "stop_signal": pending["stop"],
            },
            stream,
            indent=2,
        )
    metrics_file.close()
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    writer.close()

    # close the simulator
    env.close()
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal
    log_file.close()


if __name__ == "__main__":
    import traceback

    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
