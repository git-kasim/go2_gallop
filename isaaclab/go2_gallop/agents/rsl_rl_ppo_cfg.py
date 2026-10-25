"""ギャロップ学習の PPO（rsl_rl）設定。"""

from isaaclab.utils import configclass

from ...go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg


@configclass
class UnitreeGo2GallopPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """Isaac Lab 標準の Go2 平地の設定（MLP 128×3、PPO）に対し、回数と行動の上限だけを変える。"""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        # 行動（関節目標のずれ）の上限。ないと一部の台で出力が 1e5 まで膨らみ、2 乗で効く行動変化の罰が
        # -1e9 になって価値関数が壊れ、学習が発散した。普段の行動は ±3 程度
        self.clip_actions = 10.0
        self.experiment_name = "unitree_go2_gallop"
