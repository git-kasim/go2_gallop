"""Isaac Sim 6.1: 床のシーンに Go2 を置き、学習済みポリシーで歩かせる準備をする。

- 床（見た目 100m 四方）と物理シーンを作り、摩擦と物理の設定を学習時（Isaac Lab）に合わせる
  - 床: 静止・動摩擦 1.0、反発 0、組み合わせ方 multiply（PhysX の既定は average）
  - ロボット: env.yaml の startup マテリアル（Go2 は静止 0.8・動 0.6）
  - 物理シーン: enableExternalForcesEveryIteration=True、bounceThreshold=0.5
  これを合わせないと、3 m/s 指令で学習時より 0.4〜1.0 m/s 遅く、胴体も低くなった
- 物理を GPU・torch バックエンド・0.005 秒刻み（学習時と同じ 200 Hz、ポリシーは 4 回に 1 回で 50 Hz）にする
- Go2（Isaac Lab 版 go2.usd）を高さ 0.4 m に置き、物理ステップごとにポリシーを呼ぶコールバックを登録する

状態は sys.modules["_go2_walk"] に置く。速度指令は state.command[:] = [vx, vy, 旋回角速度]。
タイムラインを再生すると歩き出す。

注入引数: policy_run（省略可）: 学習結果フォルダ。省略時は Isaac Sim 同梱の Go2 ポリシー
    script_dir（必須）: このフォルダ（isaacsim）の場所。go2_policy_utils.py を読み込むのに使う
実行: isaacsim_send.py --ff -t 0 --file go2_walk_setup.py --arg script_dir=<go2_gallop>/isaacsim [--arg policy_run=<go2_gallop>/models/gallop]
"""
import sys
import types

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade

# Isaac Sim に送ったスクリプトには __file__ が無いため、このフォルダの場所は引数で受け取る
if "script_dir" not in dir():
    raise ValueError("script_dir（このフォルダの場所）を指定してください")
SCRIPT_DIR = str(script_dir).replace("\\", "/")

GROUND_PATH = "/World/GroundPlane"
GROUND_SIZE = 100.0  # 見た目の床の一辺 [m]（衝突判定の平面は無限）
PHYSICS_SCENE_PATH = "/World/PhysicsScene"
GROUND_MATERIAL_PATH = "/World/Looks/GroundPhysicsMaterial"
GO2_PATH = "/World/Go2"
SPAWN_POSITION = (0.0, 0.0, 0.4)  # 学習時の初期位置と同じ高さ

if "policy_run" not in dir():
    policy_run = None

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.policy.examples", True
)
from isaacsim.robot.policy.examples import RobotPolicyRunner  # noqa: E402

# 以前のセットアップが残っていれば片付ける
# stop() は次フレームで反映されるため、停止するまでアプリを進める（再生中は physics dt を変更できない）
timeline = omni.timeline.get_timeline_interface()
timeline.stop()
for _ in range(10):
    if timeline.is_stopped():
        break
    await omni.kit.app.get_app().next_update_async()
old = sys.modules.pop("_go2_walk", None)
if old is not None:
    for cb_id in (old.physics_cb_id, old.stop_cb_id):
        try:
            SimulationManager.deregister_callback(cb_id)
        except Exception:  # noqa: BLE001 - already deregistered
            pass
    old.runner.close()

stage = omni.usd.get_context().get_stage()
for path in (GO2_PATH, GROUND_PATH):
    if stage.GetPrimAtPath(path).IsValid():
        omni.kit.commands.execute("DeletePrimsCommand", paths=[path])

# 物理シーン（Z-up、下向き重力）
if not stage.GetPrimAtPath(PHYSICS_SCENE_PATH).IsValid():
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(PHYSICS_SCENE_PATH))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

# 床と摩擦マテリアル
GroundPlane(GROUND_PATH, sizes=GROUND_SIZE)
material = UsdShade.Material.Define(stage, GROUND_MATERIAL_PATH)
physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physics_material.CreateStaticFrictionAttr().Set(1.0)
physics_material.CreateDynamicFrictionAttr().Set(1.0)
physics_material.CreateRestitutionAttr().Set(0.0)
physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
physx_material.CreateFrictionCombineModeAttr().Set("multiply")
physx_material.CreateRestitutionCombineModeAttr().Set("multiply")
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{GROUND_PATH}/collisionPlane")).Bind(
    material, UsdShade.Tokens.weakerThanDescendants, "physics"
)

# 物理設定を学習時に合わせる（ロボット作成前に行う）
SimulationManager.set_backend("torch")
SimulationManager.set_physics_sim_device("cuda")

# Go2 とポリシー。utils を書き換えたら反映されるよう、毎回読み直す
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import importlib  # noqa: E402

import go2_policy_utils  # noqa: E402

importlib.reload(go2_policy_utils)

spec = go2_policy_utils.make_go2_spec(policy_run)
runner = RobotPolicyRunner(spec, prim_path=GO2_PATH, position=list(SPAWN_POSITION))
runner.spawn()
print("startup materials applied to:", sorted(runner.apply_scene_properties()))
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(PHYSICS_SCENE_PATH))
physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)
physx_scene.CreateBounceThresholdAttr().Set(0.5)
SimulationManager.set_physics_dt(runner.physics_dt)

state = types.SimpleNamespace(
    runner=runner,
    policy_name=spec.name,
    command=np.zeros(3, dtype=np.float32),  # [vx, vy, yaw_rate]（ロボット基準、m/s・rad/s）
    ready=False,
    failed=False,
    physics_cb_id=None,
    stop_cb_id=None,
)
sys.modules["_go2_walk"] = state


def _on_physics_step(dt, context):
    runner = state.runner
    if runner.articulation is None or state.failed:
        return
    if not runner.articulation.is_physics_tensor_entity_valid():
        state.ready = False
    try:
        if state.ready:
            runner.step(dt, state.command.tolist())
        else:
            runner.restart_from_default_state(state.command.tolist())
            state.ready = True
    except Exception as error:  # noqa: BLE001 - physics callback must not raise
        state.failed = True
        carb.log_error(f"go2_walk: policy failed, stopping: {error}")


def _on_timeline_stop(event):
    state.ready = False
    state.failed = False


state.physics_cb_id = SimulationManager.register_callback(_on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
state.stop_cb_id = SimulationManager.register_callback(_on_timeline_stop, IsaacEvents.TIMELINE_STOP)

print("policy:", spec.name)
print("physics_dt:", SimulationManager.get_physics_dt(), "decimation:", runner.decimation)
print("ready. Play the timeline to start walking.")
