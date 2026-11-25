# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class Tee:
    """Custom class to duplicate writes to stdout and a file."""

    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def create_pose_writer(log_dir, env_cfg, agent_cfg):
    """Use the requested logger for pose runs without requiring a cloud account."""
    if agent_cfg.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir, flush_secs=10)
    if agent_cfg.logger != "wandb":
        raise ValueError("Pose-estimator scripts support --logger tensorboard or wandb.")
    import os
    from .wandb_utils import WandbSummaryWriter

    writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    writer.log_config(
        env_cfg,
        agent_cfg,
        agent_cfg.algorithm,
        {
            "actor": agent_cfg.actor.to_dict(),
            "critic": agent_cfg.critic.to_dict(),
        },
    )
    with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as stream:
        stream.write(writer.get_run_id())
    return writer
