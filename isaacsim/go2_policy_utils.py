"""Isaac Sim 6.1: Isaac Lab で学習した Go2 のポリシーを読み込むユーティリティ（go2_walk_setup.py から使う）。

- Isaac Lab 3.x の params/env.yaml を Isaac Sim 6.1 が読める形に変換する
- ギャロップのモデルが観測に使う位相時計（gait_clock_variable）を、Isaac Sim の観測組み立てに追加する
"""
import hashlib

import omni.kit.app

CLOCK_FUNCTION_NAME = "gait_clock_variable"
"""位相時計の観測関数の名前（IO descriptor の full_path の末尾で見分ける）。"""


def _enable_policy_examples():
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.robot.policy.examples", True
    )


def to_isaacsim61_env_yaml(run_dir):
    """Isaac Lab 3.x の params/env.yaml を Isaac Sim 6.1 の読込形式に変換し、params/env_isaacsim61.yaml に保存する。

    - rigid_props / articulation_props: 新形式の func・_usd_* キーを除き、articulation_props のリスト
      （PhysX / Newton 別）から PhysxArticulationAPI の要素だけを取り出す
    - actuators: actuator_effort_limit / actuator_velocity_limit を旧キー effort_limit / velocity_limit に写す
    """
    import yaml

    _enable_policy_examples()
    from isaacsim.robot.policy.examples.env_config import _EnvConfigYamlLoader

    src = f"{run_dir}/params/env.yaml"
    dst = f"{run_dir}/params/env_isaacsim61.yaml"
    with open(src, "rb") as f:
        data = yaml.load(f, Loader=_EnvConfigYamlLoader)

    def strip_new_keys(props):
        return {k: v for k, v in props.items() if k != "func" and not k.startswith("_usd_")}

    spawn = data["scene"]["robot"]["spawn"]
    if isinstance(spawn.get("rigid_props"), dict):
        spawn["rigid_props"] = strip_new_keys(spawn["rigid_props"])
    art_props = spawn.get("articulation_props")
    if isinstance(art_props, list):
        physx = [p for p in art_props if p.get("_usd_applied_schema") == "PhysxArticulationAPI"]
        spawn["articulation_props"] = strip_new_keys(physx[0]) if physx else None
    elif isinstance(art_props, dict):
        spawn["articulation_props"] = strip_new_keys(art_props)
    for actuator in (data["scene"]["robot"].get("actuators") or {}).values():
        for new_key, old_key in (("actuator_effort_limit", "effort_limit"), ("actuator_velocity_limit", "velocity_limit")):
            if actuator.get(old_key) is None and actuator.get(new_key) is not None:
                actuator[old_key] = actuator[new_key]

    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(plain(data), f, sort_keys=False, allow_unicode=True)
    return dst


def _find_key(data, key):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        data = list(data.values())
    if isinstance(data, list):
        for item in data:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def install_gait_clock(io_descriptor_path):
    """IO descriptor に位相時計があれば、Isaac Sim の観測組み立てに時計を追加する。戻り値は時計を使うかどうか。

    同梱の binding.py は Isaac Lab 標準の観測しか組み立てられないため、関数パスの対応表に "gait_clock" を足し、
    _read_observation を包んで [sin, cos](2π·φ) を返す。学習時と同じく、推論のたびに
    φ += 制御周期 × f、f = clamp(f0 + slope·(vx指令 − v0), f0, fmax) と進める（係数は IO descriptor から読む）。
    角度は進める前の φ で出す（学習時も、リセット直後の観測は φ=0 で、次のステップで進む）。
    """
    import math

    import numpy as np
    import yaml
    from isaacsim.robot.policy.examples import binding

    with open(io_descriptor_path, encoding="utf-8") as f:
        descriptor = yaml.safe_load(f)
    records = (descriptor.get("observations") or {}).get("policy") or []
    clock = [r for r in records if str(r.get("full_path", "")).endswith("." + CLOCK_FUNCTION_NAME)]
    if not clock:
        return False

    def value(key):
        return float(_find_key(clock[0], key))

    # 状態は binding モジュール側に持たせる（このファイルを読み直しても、包んだ関数と同じものを使う）
    state = getattr(binding, "_go2_gait_clock_state", None)
    if state is None:
        state = binding._go2_gait_clock_state = {}
    state.update(
        phase=0.0,
        freq=value("gait_freq_hz"),
        slope=value("gait_freq_slope"),
        v0=value("gait_freq_v0"),
        fmax=value("gait_freq_max"),
        dt=float(descriptor["scene"]["dt"]),
    )

    def read_clock(term, *args, **kwargs):
        s = binding._go2_gait_clock_state
        angle = 2.0 * math.pi * s["phase"]
        setpoint_state = kwargs.get("setpoint_state", args[2] if len(args) > 2 else None)
        vx = float(setpoint_state.root.linear_velocity.numpy()[0]) if setpoint_state is not None else 0.0
        freq = min(max(s["freq"] + s["slope"] * (vx - s["v0"]), s["freq"]), s["fmax"])
        s["phase"] = (s["phase"] + s["dt"] * freq) % 1.0
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

    state["read"] = read_clock
    binding._OBSERVATION_PATHS[clock[0]["full_path"]] = "gait_clock"
    if not getattr(binding._read_observation, "_go2_gait_clock", False):
        original = binding._read_observation

        def read_observation(term, *args, **kwargs):
            if term.semantic != "gait_clock":
                return original(term, *args, **kwargs)
            return binding._go2_gait_clock_state["read"](term, *args, **kwargs)

        read_observation._go2_gait_clock = True
        binding._read_observation = read_observation
    return True


def make_go2_spec(policy_run=None):
    """Go2 の PolicySpec を返す。

    policy_run: Isaac Lab の学習結果フォルダ（exported/policy.pt, params/env.yaml,
    io_descriptors/IO_descriptors.yaml を含む）。省略時は Isaac Sim 同梱の Go2 ポリシー（比較の「未学習」）。
    """
    _enable_policy_examples()
    from isaacsim.robot.policy.examples import PolicyArtifact, get_go2_spec

    if not policy_run:
        return get_go2_spec()
    policy_run = str(policy_run).replace("\\", "/").rstrip("/")
    if install_gait_clock(f"{policy_run}/io_descriptors/IO_descriptors.yaml"):
        print("gait clock: on")
    with open(f"{policy_run}/exported/policy.pt", "rb") as f:
        model_sha256 = hashlib.sha256(f.read()).hexdigest()
    artifact = PolicyArtifact.from_files(
        f"{policy_run}/exported/policy.pt",
        to_isaacsim61_env_yaml(policy_run),
        f"{policy_run}/io_descriptors/IO_descriptors.yaml",
        backend="torchscript",
        model_sha256=model_sha256,
    )
    return get_go2_spec(engines={"physx": artifact}, name=f"go2_{policy_run.split('/')[-1]}")
