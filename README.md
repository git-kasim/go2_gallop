# Unitree Go2 をギャロップで走らせる（Isaac Lab / Isaac Sim）

Isaac Lab の強化学習で、四脚ロボット Unitree Go2 にギャロップ（4 本の足が別々のタイミングで着地し、4 本とも浮く瞬間がある走り方）を学習させるタスクと、学習したモデルを Isaac Sim 6.1 で動かすスクリプトです。

- 学習タスク `Isaac-Velocity-Gallop-UnitreeGo2`（Isaac Lab 標準の Go2 平地タスクを拡張）
- 学習済みモデル（ゼロから 6000 回学習、前進 0〜5 m/s）
- 学習したモデルを Isaac Sim 6.1 に読み込んで、速度指令で動かすスクリプト

## 動作確認環境

| ソフトウェア | バージョン |
|---|---|
| NVIDIA Isaac Sim | 6.1.0（pip 版） |
| NVIDIA Isaac Lab | 3.0.0（rsl_rl 5.5.1） |
| PyTorch | 2.12（CUDA 13.0） |
| Python | 3.12 |
| OS | Windows 11 Pro（64 ビット） |

GPU は NVIDIA GeForce RTX 5070 Ti（VRAM 16 GB）で確認しました。4096 体の並列学習が 1 回あたり約 2 秒、6000 回で約 3 時間です。

## 構成

```
isaaclab/go2_gallop/        Isaac Lab の学習タスク
  gallop_env_cfg.py           環境（観測・報酬・指令・終了条件）
  agents/rsl_rl_ppo_cfg.py    PPO の設定
  __init__.py                 タスクの登録
isaacsim/                   Isaac Sim 6.1 で動かすスクリプト
  launch_isaacsim.ps1         Isaac Sim を python_server（127.0.0.1:8226）付きで起動する
  go2_walk_setup.py           床と Go2 を置き、モデルを読み込んで物理ステップに登録する
  go2_policy_utils.py         Isaac Lab 3.x の設定を Isaac Sim 6.1 用に変換し、位相時計の観測を追加する
  quit_isaacsim.py            Isaac Sim を終了する
models/gallop/              学習済みモデル
  exported/policy.pt          Isaac Sim で使うポリシー（TorchScript）
  io_descriptors/             観測・行動の並びと位相時計の係数
  params/                     学習時の環境と PPO の設定
  model_5999.pt               チェックポイント（書き出し直しや続きの学習に使う）
```

## 学習タスクの内容

土台は Isaac Lab 標準の Go2 平地タスク（`UnitreeGo2FlatEnvCfg`）です。ギャロップを覚えさせるために、次の要素を足しています。

| 要素 | 内容 |
|---|---|
| 位相時計 | 観測に (sin 2πφ, cos 2πφ) を追加。周波数は前進指令で変わる：f = clamp(4 + 0.5·(vx指令 − 3), 4, 5.5) Hz |
| 接地の予定表 | 時計に合わせて「どの足がいつ接地しているべきか」を決め、実際の接地との一致に報酬（+2.0）。着地順は 右後 → 左後（+0.10）→ 右前（+0.45）→ 左前（+0.55）、各足の接地は 1 周期の 35% |
| 予定表どおりの足上げ | 予定で遊脚の足だけに足上げの報酬（+0.5） |
| ゲートと追いつき度 | 予定表と足上げの報酬は、前進指令 1.5→2.0 m/s で 0→1 になるゲートと、clamp(実速度 / 指令, 0, 1) を掛ける。低速は標準の報酬に任せ、その場で跳ねるだけでは報酬が出ないようにする |
| 胴体の下限 | 胴体が 0.22 m を下回ったときだけ罰（-100）、太ももの接地に罰（-1）。這う姿勢を防ぎ、上下の揺れは妨げない |
| 足の浮かせっぱなし | 0.3 秒を超えて浮いている足に罰（-1.0） |
| 立ち止まり | 指令 0 の台を 15% にし、4 本とも接地していれば報酬（+1.0）、関節の初期姿勢からのずれに罰（-1.0） |
| 学習の安定化 | 行動に上限 ±10（rsl_rl の `clip_actions`）。物理の破綻（速度の異常）を検出したら終了し、その台の報酬を 0 にする |

| 項目 | 設定 |
|---|---|
| 指令 | 前進 0〜5.5 m/s、横 ±0.3 m/s、旋回 ±0.5 rad/s、立ち止まり 15% |
| 観測 | 標準（胴体の速度・重力の向き・指令・関節の角度と速度・前回の行動）＋位相時計 |
| PPO | Isaac Lab 標準の Go2 平地の設定（MLP 128×3）、6000 回 |

## 使い方

### 1. 学習（Isaac Lab）
1. `isaaclab/go2_gallop` を、Isaac Lab の `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2_gallop` にコピーする
2. Isaac Lab の venv を有効にし、Isaac Lab のフォルダ（`isaaclab.bat` がある所）で学習する
   ```
   .\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Gallop-UnitreeGo2 --num_envs 4096 physics=isaacsim_physx
   ```
   学習の回数は 6000 回。試しに短く動かすときは、末尾に `--max_iterations 50` のように付ける。
3. Isaac Sim 用に書き出す（`exported\policy.pt` と `io_descriptors\` ができたら終了してよい）
   ```
   .\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Gallop-UnitreeGo2 --num_envs 16 --checkpoint logs\rsl_rl\unitree_go2_gallop\<日時>\model_5999.pt env.export_io_descriptors=true physics=isaacsim_physx
   ```
   学習結果のフォルダ（`logs\rsl_rl\unitree_go2_gallop\<日時>\`）が、そのまま Isaac Sim で読み込めるモデルになる。

学習済みのモデル（`models/gallop/`）を使うだけなら、この手順は不要です。

### 2. 再生（Isaac Sim 6.1）
`isaacsim\` のスクリプトは、Isaac Sim の拡張 `isaacsim.code_editor.python_server`（127.0.0.1:8226）に送って実行します。
送るのは Isaac Sim pip 版に同梱の `isaacsim_send.py` で、ほかに必要なものはありません。
Script Editor には貼れません（引数を受け取れず、関数の外の `await` も使えないため）。

以下は、次の場所に置いた例です。自分の環境に合わせて、手順 0 の 2 つの変数だけを書き換えてください。

| もの | 例の場所 |
|---|---|
| Isaac Sim 6.1（pip 版）の venv | `C:\env_isaacsim`（`Scripts\python.exe` と `Scripts\isaacsim.exe` がある所） |
| このリポジトリ | `C:\go2_gallop`（`isaacsim\`、`models\` がある所） |

どちらのパスにも日本語などの英数字以外を含めないでください（Isaac Sim の処理の一部が扱えずに落ちることがあります）。

0. PowerShell を開き、場所を変数に入れてリポジトリのフォルダに移る。以下の手順はすべてこのウィンドウで行う
   ```
   $venv = "C:\env_isaacsim"   # Isaac Sim の venv に書き換える
   $root = "C:\go2_gallop"     # このリポジトリを置いた場所に書き換える
   $env:PYTHONUTF8 = "1"
   $py = "$venv\Scripts\python.exe"
   $send = "$venv\Lib\site-packages\isaacsim\skills\isaac-sim-remote\scripts\isaacsim_send.py"
   cd $root
   ```
1. 送る口を開けて Isaac Sim を起動する（送れるようになるまで 1 分ほどかかる）
   ```
   powershell -ExecutionPolicy Bypass -File isaacsim\launch_isaacsim.ps1 -Venv $venv
   ```
   `-Venv` を省くと環境変数 `ISAACSIM_VENV` を使う。standalone 版などは `-Exe <isaac-sim.bat のパス>` で実行ファイルを直接指定する。
   中でしているのは `<venv>\Scripts\isaacsim.exe --enable isaacsim.code_editor.python_server` だけ。
   GUI から開く場合は、Window → Extensions で `isaacsim.code_editor.python_server` を有効にする。
2. 床と Go2 を置き、モデルを読み込む（`policy_run` を省くと Isaac Sim 同梱のポリシー。`script_dir` は必須）
   ```
   & $py $send --ff -t 0 --file isaacsim\go2_walk_setup.py --arg "script_dir=$root\isaacsim" --arg "policy_run=$root\models\gallop"
   ```
   自分で学習したモデルを使うときは、`policy_run` に学習結果のフォルダ（`exported\` などがある所）を渡す。
3. タイムラインを再生し、速度指令を送る（`[前進 m/s, 横 m/s, 旋回 rad/s]`。下は前進 3 m/s）
   ```
   & $py $send "import sys, omni.timeline; omni.timeline.get_timeline_interface().play(); sys.modules['_go2_walk'].command[:] = [3.0, 0.0, 0.0]"
   ```
   走っている間に速度を変えるときや止めるときは、指令だけ送り直す（下は停止）。
   ```
   & $py $send "import sys; sys.modules['_go2_walk'].command[:] = [0.0, 0.0, 0.0]"
   ```
4. 終了する
   ```
   & $py $send --ff --file isaacsim\quit_isaacsim.py
   ```

- `--ff`（fire-and-forget）はすぐに `task_id` を返し、スクリプトは Isaac Sim の中で動き続ける。
  終わったかは `& $py $send --introspect task <task_id>` で見る（実行中は `null`、終わると `"status": "ok"` と出力）
- `-t 0` はサーバー側の実行時間の上限なし
- python_server は認証なしでどんなコードも実行するため、ループバック（127.0.0.1）でだけ使う

### 学習済みモデルの目安
- 前進 2 m/s 以上でギャロップになる（3 m/s で 4 本とも浮く時間が 1 周期の約 11%）。1 m/s では小さな歩幅の小走り
- 使える速度は前進 0〜5 m/s。5.5 m/s 以上では転ぶ
- 止まった状態から 5 m/s を一気に指令しても走り出せる

## Isaac Sim で学習時と同じ動きにするための設定
`go2_walk_setup.py` は、物理の設定を Isaac Lab の学習時に合わせています。これを合わせないと、同じモデルでも Isaac Sim では遅く、胴体も低くなります（開発中のモデルで確かめたところ、3 m/s 指令で 2.91 m/s 出ていたものが 2.31 m/s に落ちた）。

| 設定 | 値 |
|---|---|
| 物理の刻み | 0.005 秒（200 Hz）。ポリシーは 4 回に 1 回（50 Hz） |
| 床の摩擦 | 静止・動摩擦 1.0、反発 0、組み合わせ方 multiply（PhysX の既定は average） |
| ロボットの足の摩擦 | 学習時の設定（`params/env.yaml` の startup マテリアル。Go2 は静止 0.8・動 0.6） |
| 物理シーン | `enableExternalForcesEveryIteration=True`、`bounceThreshold=0.5` |

## ライセンス

[BSD 3-Clause License](LICENSE)

学習タスクは [Isaac Lab](https://github.com/isaac-sim/IsaacLab)（BSD-3-Clause）の上に作っています。
Unitree Go2 のロボットのモデルは Isaac Lab / Isaac Sim から読み込むため、このリポジトリには含みません。
