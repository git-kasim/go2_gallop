"""Unitree Go2 の平地ギャロップ（前進 0〜5.5 m/s）の学習タスク。

このフォルダを Isaac Lab の isaaclab_tasks/core/velocity/config/ の下に go2_gallop としてコピーすると、
isaaclab_tasks を読み込んだときに自動で登録される。
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Gallop-UnitreeGo2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gallop_env_cfg:UnitreeGo2GallopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2GallopPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
