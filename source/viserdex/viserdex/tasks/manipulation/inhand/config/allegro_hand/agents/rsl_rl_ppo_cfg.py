# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class SimpleAllegroPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 1000
    experiment_name = "viserdex_allegro_v3_ppo"
    logger = "tensorboard"
    wandb_project = "stable-grasping"
    # RSL-RL 5 names the actor observation set "actor".
    obs_groups = {
        "actor": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
