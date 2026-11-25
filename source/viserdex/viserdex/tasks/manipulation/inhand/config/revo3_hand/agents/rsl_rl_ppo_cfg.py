"""Reuse the task's PPO settings with a separate Revo3 checkpoint namespace."""

from isaaclab.utils import configclass

from ...allegro_hand.agents.rsl_rl_ppo_cfg import SimpleAllegroPPORunnerCfg


@configclass
class Revo3PPORunnerCfg(SimpleAllegroPPORunnerCfg):
    experiment_name = "viserdex_revo3_v3_ppo"
