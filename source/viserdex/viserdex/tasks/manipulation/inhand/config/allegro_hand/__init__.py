"""Register the legacy Allegro task family with the official task naming convention."""

import gymnasium as gym
from . import agents

for suffix, cfg_name in {
    "": "ViserDexAllegroEnvCfg",
    "-Play": "ViserDexAllegroEnvCfg_PLAY",
    "-Estimator": "ViserDexAllegroPoseEstimatorEnvCfg",
    "-Estimator-Eval": "ViserDexAllegroPoseEstimatorEvalEnvCfg",
    "-Estimator-Camera": "ViserDexAllegroPoseEstimatorCameraEnvCfg",
    "-Estimator-NoAug": "ViserDexAllegroPoseEstimatorUnaugmentedEnvCfg",
}.items():
    gym.register(
        id=f"Isaac-ViserDex-Repose-Allegro{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "default_agent": "rsl_rl",
            "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
        },
    )
