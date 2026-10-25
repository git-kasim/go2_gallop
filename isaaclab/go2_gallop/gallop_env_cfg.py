"""Unitree Go2 のギャロップ（平地、前進 0〜5.5 m/s）を 1 回の学習で覚えさせる環境設定。

Isaac Lab 標準の Go2 平地設定 :class:`UnitreeGo2FlatEnvCfg` に、次の工夫を足している（説明は ../../README.md）。

1. 這う姿勢の禁止: 胴体が 0.22 m を下回ったときだけ罰（-100）、太ももの接地に罰（-1）
2. 位相時計: 観測に [sin, cos](2π·φ) を足し、時計に合わせた「足ごとの接地の予定表」（横ギャロップ）との一致に報酬
3. 予定表どおりの足上げに報酬（予定で遊脚の足だけ。脚を上げっぱなしにしても得にならない）
4. 足の浮かせっぱなし（0.3 秒超）に罰。3 本脚で走る・立つ抜け道をふさぐ
5. 立ち止まり: 指令 0 の台を 15% に増やし、4 本とも接地で報酬、関節の初期姿勢からのずれに罰
6. 速さで時計を速くする: f = clamp(4 + 0.5·(vx指令 − 3), 4, 5.5) Hz
7. 予定表の報酬に「実速度 ÷ 指令」（0〜1）を掛ける。指令に追いついていないと予定表の報酬が出ないので、
   その場で予定どおりに跳ねて報酬を稼ぐ局所解を防ぐ
8. 発散対策: 物理の破綻（速度の異常）で終了し、その台の報酬を 0 にする。行動の上限は runner 側（±10）

歩容の報酬は、指令 1.5→2.0 m/s で 0→1 になるゲートを掛けて高速だけに効かせ、低速は標準の報酬に任せる。
"""

import functools
import math

import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_dtype, record_shape
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp
from ..go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

##
# 定数
##

MAX_FORWARD_SPEED = 5.5
"""前進速度の指令の上限 [m/s]。"""

STANDING_ENV_RATIO = 0.15
"""指令 0（立ち止まり）の台の割合（標準は 0.02）。"""

MIN_BASE_HEIGHT = 0.22
"""これを下回ると罰を受ける胴体の高さ [m]。"""

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
"""足の順番（左前、右前、左後、右後）。"""

GATE_START = 1.5
GATE_FULL = 2.0
"""歩容の報酬が効き始める / 満額になる前進指令 [m/s]。"""

RELAX_AT_SPEED = 0.75
"""ゲート 1 のとき、胴体の上下・傾きの罰を減らす割合（ギャロップは胴体が上下に揺れるため）。"""

FOOT_CLEARANCE_TARGET = 0.07
"""遊脚の足上げで満点になる高さ [m]。"""

FOOT_STANCE_Z = 0.022
"""平地で接地しているときの足リンク原点の高さ [m]。"""

GAIT_FREQ = 4.0
"""時計の基本周波数 [Hz]。"""

GAIT_FREQ_SLOPE = 0.5
"""GAIT_FREQ_V0 を超えた前進指令 1 m/s あたりの周波数の増加 [Hz/(m/s)]。"""

GAIT_FREQ_V0 = 3.0
"""時計が GAIT_FREQ のままの前進指令の上限 [m/s]。"""

GAIT_FREQ_MAX = 5.5
"""時計の周波数の上限 [Hz]。"""

DUTY = 0.35
"""1 歩の周期のうち各足が接地している割合。"""

FOOT_OFFSETS = (0.55, 0.45, 0.10, 0.0)
"""FL, FR, RL, RR の着地の位相。右後 → 左後 → 右前 → 左前 の横ギャロップ。"""

HELD_UP_THRESHOLD = 0.3
"""この時間 [s] を超えて浮いている足を「上げっぱなし」とみなす（4 Hz の遊脚は約 0.16 秒）。"""

BLOWUP_LIN_VEL = 20.0
BLOWUP_ANG_VEL = 60.0
BLOWUP_JOINT_VEL = 100.0
"""物理が破綻したとみなす胴体の速度 [m/s]・角速度 [rad/s]・関節速度 [rad/s]。"""


##
# 共通の補助関数
##


def _forward_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 0]


def _speed_gate(env) -> torch.Tensor:
    """前進指令で 0→1 になる歩容の報酬のゲート。"""
    return torch.clamp((_forward_command(env) - GATE_START) / (GATE_FULL - GATE_START), 0.0, 1.0)


def _speed_progress(env) -> torch.Tensor:
    """clamp(胴体の前後速度 / 前進指令, 0, 1)。指令にどれだけ追いついているか。"""
    vx_body = torch.nan_to_num(env.scene["robot"].data.root_lin_vel_b.torch[:, 0], nan=0.0)
    return torch.clamp(vx_body / torch.clamp(_forward_command(env), min=0.1), 0.0, 1.0)


def _in_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0


##
# 位相時計
##


def _gait_freq(env) -> torch.Tensor:
    return torch.clamp(GAIT_FREQ + GAIT_FREQ_SLOPE * (_forward_command(env) - GAIT_FREQ_V0), GAIT_FREQ, GAIT_FREQ_MAX)


def _phase(env) -> torch.Tensor:
    """時計の位相 φ ∈ [0, 1)。周波数が指令で変わるので、制御周期ごとに φ += dt·f と積算し、リセットで 0 に戻す。"""
    if getattr(env, "_gait_phase", None) is None:
        env._gait_phase = torch.zeros(env.num_envs, device=env.device)
        env._gait_phase_step = env.common_step_counter
    if env._gait_phase_step != env.common_step_counter:
        env._gait_phase = torch.remainder(env._gait_phase + env.step_dt * _gait_freq(env), 1.0)
        env._gait_phase_step = env.common_step_counter
    env._gait_phase[env.episode_length_buf == 0] = 0.0
    return env._gait_phase


# IO descriptor に周波数の式の係数を書き出し、Isaac Sim 側（../isaacsim/go2_policy_utils.py）で同じ時計を再現する
@generic_io_descriptor(
    observation_type="Clock",
    gait_freq_hz=GAIT_FREQ,
    gait_freq_slope=GAIT_FREQ_SLOPE,
    gait_freq_v0=GAIT_FREQ_V0,
    gait_freq_max=GAIT_FREQ_MAX,
    on_inspect=[record_shape, record_dtype],
)
def gait_clock_variable(env) -> torch.Tensor:
    """観測: [sin, cos](2π·φ)。"""
    angle = 2.0 * math.pi * _phase(env)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def _desired_stance(env) -> torch.Tensor:
    """予定表で各足が接地しているべきか（num_envs × 4）。"""
    offsets = torch.tensor(FOOT_OFFSETS, device=env.device)
    return torch.remainder(_phase(env).unsqueeze(1) - offsets, 1.0) < DUTY


##
# 報酬
##


def base_height_below_l2(env, min_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """胴体が min_height を下回ったときだけの罰（片側 L2）。上は自由に上下してよい。"""
    asset = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(min_height - asset.data.root_pos_w.torch[:, 2], min=0.0))


def gallop_contact_match(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """接地／遊脚が予定表と一致した足の割合 × ゲート × 追いつき度。"""
    match = (_in_contact(env, sensor_cfg) == _desired_stance(env)).float()
    return match.mean(dim=1) * _speed_gate(env) * _speed_progress(env)


def gallop_swing_clearance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """予定表で遊脚の足だけの足上げ（目標の高さで頭打ち）× ゲート × 追いつき度。"""
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - FOOT_STANCE_Z
    lift = torch.clamp(foot_z / FOOT_CLEARANCE_TARGET, 0.0, 1.0) * ~_desired_stance(env)
    return torch.sum(lift, dim=1) / 4.0 * _speed_gate(env) * _speed_progress(env)


def low_speed_feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """標準の滞空時間の報酬。予定表が効く高速では弱める。"""
    return mdp.feet_air_time(env, command_name, sensor_cfg, threshold) * (1.0 - _speed_gate(env))


def feet_held_up(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """threshold を超えて浮いている時間の足ごとの合計（1 本 1 秒で頭打ち）。指令 0 でも効かせる。"""
    air_time = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.clamp(air_time - threshold, 0.0, 1.0).sum(dim=1)


def stand_on_all_feet(env, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.06) -> torch.Tensor:
    """指令がほぼ 0 のとき、4 本とも接地していれば 1。"""
    command = env.command_manager.get_command("base_velocity")[:, :2]
    standing = torch.linalg.norm(command, dim=1) < command_threshold
    return torch.all(_in_contact(env, sensor_cfg), dim=1).float() * standing


def _relax(env) -> torch.Tensor:
    return 1.0 - RELAX_AT_SPEED * _speed_gate(env)


def relaxed_lin_vel_z_l2(env) -> torch.Tensor:
    return base_mdp.lin_vel_z_l2(env) * _relax(env)


def relaxed_ang_vel_xy_l2(env) -> torch.Tensor:
    return base_mdp.ang_vel_xy_l2(env) * _relax(env)


def relaxed_flat_orientation_l2(env) -> torch.Tensor:
    return base_mdp.flat_orientation_l2(env) * _relax(env)


##
# 物理の破綻への対策
##


def _healthy(env) -> torch.Tensor:
    """物理の状態が正常な台は True（速度が有限で、ありえない大きさでない）。"""
    asset = env.scene["robot"]
    lin = asset.data.root_lin_vel_w.torch
    ang = asset.data.root_ang_vel_w.torch
    qd = asset.data.joint_vel.torch
    finite = torch.isfinite(lin).all(dim=1) & torch.isfinite(ang).all(dim=1) & torch.isfinite(qd).all(dim=1)
    sane = (
        (torch.linalg.norm(lin, dim=1) < BLOWUP_LIN_VEL)
        & (torch.linalg.norm(ang, dim=1) < BLOWUP_ANG_VEL)
        & (qd.abs().max(dim=1).values < BLOWUP_JOINT_VEL)
    )
    return finite & sane


def physics_blowup(env) -> torch.Tensor:
    """終了条件: 物理が破綻した台。"""
    return ~_healthy(env)


def _guarded(func):
    """報酬関数を包み、破綻した台の報酬と NaN / inf を 0 にする。"""

    # functools.wraps で元の引数の形を残す（報酬マネージャーが params と照合するため）
    @functools.wraps(func)
    def guarded(env, **kwargs):
        value = torch.nan_to_num(func(env, **kwargs), nan=0.0, posinf=0.0, neginf=0.0)
        return value * _healthy(env)

    return guarded


##
# 環境設定
##


@configclass
class UnitreeGo2GallopEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 の平地ギャロップ（前進 0〜5.5 m/s）。"""

    def __post_init__(self):
        super().__post_init__()

        # 指令: 前進だけ 0〜5.5 m/s、横と旋回は小さく、向きの指令はなし。15% の台は立ち止まり
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = STANDING_ENV_RATIO
        cmd.ranges.lin_vel_x = (0.0, MAX_FORWARD_SPEED)
        cmd.ranges.lin_vel_y = (-0.3, 0.3)
        cmd.ranges.ang_vel_z = (-0.5, 0.5)

        # 観測: 標準の観測＋位相時計
        self.observations.policy.gait_clock = ObsTerm(func=gait_clock_variable)

        feet_sensor = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
        feet_body = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
        r = self.rewards

        # 1. 這う姿勢の禁止
        r.base_height_floor = RewTerm(func=base_height_below_l2, weight=-100.0, params={"min_height": MIN_BASE_HEIGHT})
        r.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
        )
        # 胴体の上下・傾きの罰は標準の重みのまま、高速で弱める
        r.lin_vel_z_l2 = RewTerm(func=relaxed_lin_vel_z_l2, weight=r.lin_vel_z_l2.weight)
        r.ang_vel_xy_l2 = RewTerm(func=relaxed_ang_vel_xy_l2, weight=r.ang_vel_xy_l2.weight)
        r.flat_orientation_l2 = RewTerm(func=relaxed_flat_orientation_l2, weight=r.flat_orientation_l2.weight)

        # 2・3・7. 予定表（横ギャロップ）との一致と足上げ。追いつき度を掛ける
        r.feet_air_time = RewTerm(
            func=low_speed_feet_air_time, weight=r.feet_air_time.weight, params=dict(r.feet_air_time.params)
        )
        r.gallop_contact_match = RewTerm(func=gallop_contact_match, weight=2.0, params={"sensor_cfg": feet_sensor})
        r.gallop_swing_clearance = RewTerm(func=gallop_swing_clearance, weight=0.5, params={"asset_cfg": feet_body})
        r.feet_slide = RewTerm(
            func=mdp.feet_slide, weight=-0.1, params={"sensor_cfg": feet_sensor, "asset_cfg": feet_body}
        )

        # 4. 足の浮かせっぱなし
        r.feet_held_up = RewTerm(
            func=feet_held_up, weight=-1.0, params={"sensor_cfg": feet_sensor, "threshold": HELD_UP_THRESHOLD}
        )

        # 5. 立ち止まり
        r.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1, weight=-1.0, params={"command_name": "base_velocity"}
        )
        r.stand_on_all_feet = RewTerm(func=stand_on_all_feet, weight=1.0, params={"sensor_cfg": feet_sensor})

        # 8. 物理の破綻
        self.terminations.physics_blowup = DoneTerm(func=physics_blowup)
        for term in vars(self.rewards).values():
            if isinstance(term, RewTerm):
                term.func = _guarded(term.func)
