# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to roll out a trained policy and log pose-estimator metrics without further training."""

"""Launch Isaac Sim Simulator first."""

import argparse

from viserdex.utils.runtime import SimulationRuntime, add_runtime_args, parse_runtime_args

# local imports
from viserdex.utils import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate the pose estimator in the environment without further training.")
parser.add_argument("--save_images", action="store_true", default=False, help="Save evaluation images.")
parser.add_argument("--play_time", type=int, default=None, help="Time to simulate (in sec).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-ViserDex-Repose-Revo3-Estimator-Eval", help="Name of the task.")
parser.add_argument("--policy_run", type=str, default=None, help="Manipulation policy run used for rollouts.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
add_runtime_args(parser, default_task="Isaac-ViserDex-Repose-Revo3")
args_cli = parse_runtime_args(parser)
if not args_cli.load_run:
    parser.error("--load_run must identify a pose-estimator run")
if "-Eval-" in args_cli.task or "-Camera-" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
simulation_app = SimulationRuntime(args_cli)

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch
from termcolor import colored
import numpy as np
from datetime import datetime
from collections import defaultdict

import viserdex.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import create_rsl_rl_runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.assets import RigidObject
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
import isaaclab.utils.math as math_utils

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from viserdex.utils.log_utils import create_pose_writer
from viserdex.assets.objects import get_rollout_policy_run


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = simulation_app.env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg.terminations.object_out_of_reach.params["threshold"] = 0.05

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    policy_run = get_rollout_policy_run(args_cli.task, args_cli.policy_run)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, policy_run, agent_cfg.load_checkpoint)

    agent_cfg.experiment_name = "pose_estimator"
    agent_cfg.run_name = args_cli.load_run
    log_root_path = os.path.join("logs", "rsl_rl", "pose_estimator", args_cli.load_run)
    env_cfg.log_dir = log_root_path

    # Log stdout to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, "evaluations", f"eval_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(colored(f"[INFO]: Loading experiment from directory: {log_root_path}", "green"))

    # Evaluation must load an existing estimator and never perform training updates.
    from pathlib import Path

    if not list(Path(env_cfg.log_dir, "pose_estimator").glob("cnn_*.pth")):
        raise FileNotFoundError(f"No pose-estimator checkpoint in {env_cfg.log_dir}/pose_estimator")
    for group_name in ("pose", "gaussian_features"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.image_features.params["pose_estimator_cfg"].train = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env_play.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent_play.yaml"), agent_cfg)

    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))
    # load previously trained model
    ppo_runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # initialize writer
    if os.path.exists(os.path.join(log_dir, "wandb_run_id.txt")):
        with open(os.path.join(log_dir, "wandb_run_id.txt"), "r") as f:
            agent_cfg.run_id = f.read().strip()
    writer = create_pose_writer(log_dir, env_cfg, agent_cfg)

    # reset environment
    obs = env.get_observations()
    errors = defaultdict(list)

    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)
            # store logs
            for key in extras["log"].keys():
                if "pose_estimator" in key:
                    errors[key].append(extras["log"][key])
                    writer.add_scalar(f"eval/{key}", extras["log"][key], step_count)

        if args_cli.play_time is not None:
            if env.unwrapped.common_step_counter * env.unwrapped.step_dt >= args_cli.play_time:
                break

        if step_count % 100 == 0:
            print(f"[INFO] Step: {step_count}")

        step_count += 1

    file_name = os.path.join(log_dir, "logs.txt")
    with open(file_name, "w") as f:
        max_key_length = max(len(key.strip("feature_extractor/")) for key in errors.keys())
        for key, value in errors.items():
            errors_array = np.array(value)
            average = np.average(errors_array)
            variance = np.sqrt(np.average((errors_array - average) ** 2))
            f.write(f"{key.strip('feature_extractor/').ljust(max_key_length + 2)}: {average:.5f} +- {variance:.5f}\n")
            writer.add_scalar(f"summary/eval/{key}/average", average, step_count)
            writer.add_scalar(f"summary/eval/{key}/variance", variance, step_count)

    writer.close()

    # close the simulator
    env.close()


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
