"""Register the Revo3 task family."""

import gymnasium as gym

from . import agents

for suffix, cfg_name in {
    "": "ViserDexRevo3EnvCfg",
    "-SDF": "ViserDexRevo3SDFEnvCfg",
    "-V14": "ViserDexRevo3V14EnvCfg",
    "-Play": "ViserDexRevo3EnvCfg_PLAY",
    "-Hikrobot": "ViserDexRevo3HikrobotEnvCfg",
    "-Estimator": "ViserDexRevo3PoseEstimatorEnvCfg",
    "-Estimator-Eval": "ViserDexRevo3PoseEstimatorEvalEnvCfg",
    "-Estimator-Camera": "ViserDexRevo3PoseEstimatorCameraEnvCfg",
    "-Estimator-NoAug": "ViserDexRevo3PoseEstimatorUnaugmentedEnvCfg",
}.items():
    gym.register(
        id=f"Isaac-ViserDex-Repose-Revo3{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "default_agent": "rsl_rl",
            "env_cfg_entry_point": f"{__name__}.revo3_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Revo3PPORunnerCfg",
        },
    )
