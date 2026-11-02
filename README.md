# PAI最終課題：学習型知覚＋模倣学習ポリシーによるガントリーロボットの色分け仕分け＆積み上げ

[English](#english) | 日本語

MuJoCo 上のガントリー（直交）ロボットが、6個のキューブ（3色×2個）を同じ色のゴールパッドへ仕分けて2段に積み上げます。

- **知覚（AI）**：俯瞰カメラ画像から、学習した画素セグメンテーションモデルでキューブを検出
- **動作（AI）**：エキスパートのデモから模倣学習（Behavior Cloning＋Action Chunking）した**ニューラルネットのポリシー**がロボットを制御
- **閉ループ**：毎ピック前に再認識するため、作業中にキューブが動かされても対応

デモ動画（等倍速）: https://youtu.be/-3nVhX7dr_U

第5回講義の「収集 → 学習 → デプロイ → 評価」のパイプラインを、テレオペの代わりにスクリプトのエキスパートを教師として MuJoCo 上で実装しました。

## 実行方法

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python collect.py 200        # デモ収集（200配置 → 約1200エピソード、data/demos.npz）
.venv/bin/python train_bc.py 40        # ポリシー学習（CPUで約20秒、policy.pt と docs/debug_policy.png）
.venv/bin/python run_policy.py eval 50 # 閉ループ評価（未学習の50配置）
.venv/bin/python run_policy.py video   # 学習ポリシーが制御するデモ動画 demo.mp4
.venv/bin/python sort_sim.py 2         # 参考：スクリプト制御版（v2）
```

学習済みの `policy.pt` を同梱しているので、`eval` / `video` はすぐ実行できます。動作確認環境：macOS（Apple M2, CPUのみ）, Python 3.13, MuJoCo 3.x。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `sort_sim.py` | シーン定義、学習型知覚（画素セグメンテーション）、スクリプト制御版デモ |
| `policy_env.py` | 10 Hz の行動インタフェースを持つ環境、エキスパート（教師）、エピソード実行 |
| `collect.py` | DART 方式のノイズ注入付きデモ収集 |
| `train_bc.py` | Action Chunking ポリシーの学習とデバッグ（予測 vs 正解のプロット） |
| `run_policy.py` | 閉ループ評価とデモ動画生成 |

## 構成

| 要素 | 内容 |
|---|---|
| 知覚（AI） | 画素ごとの RGB → {背景, 赤, 緑, 青} を分類する MLP（scikit-learn）。教師ラベルは MuJoCo のセグメンテーション描画から自動生成。連結成分で個体分離 → カメラ幾何で世界座標へ |
| ポリシー（AI） | 入力：目標キューブ・ゴールへの相対位置、高さ、把持状態、サーボ追従誤差（16次元）。出力：次の10ステップ分の行動（dx, dy, dz, 磁石 ON/OFF）。3層MLP（PyTorch）、損失は L1＋BCE |
| デモ収集 | スクリプトのエキスパートが教師。実行時は行動にノイズを加え、ラベルはノイズなしの行動を記録（DART）→ 誤差からの復帰も学習 |
| 実行 | 各ステップでチャンクの先頭行動のみ実行し、毎ステップ再推論（第5回の `n_action_steps: 1` と同じ） |
| ルールベースの部分 | 次に取るキューブの選択（最寄り）と、撮影前の退避動作 |

## 結果（未学習の50配置＝300個、`docs/eval_results.txt`）

| 制御 | 仕分け成功 | 全6個成功した配置 | タイムアウトしたピック |
|---|---|---|---|
| エキスパート（教師） | 300/300 | 50/50 | 0 |
| エキスパート＋外乱 | 300/300 | 50/50 | 0 |
| **学習ポリシー** | **300/300** | **50/50** | 13 |
| **学習ポリシー＋外乱** | **300/300** | **50/50** | 15 |
| ポリシー＋ACT の temporal ensembling（比較） | 183/300（61%） | 18/50 | 218 |

- ピック試行の約4%がタイムアウトしましたが、再認識して再試行することで最終的に全て成功しています（初回成功率は約96%、エキスパートは100%）。
- 知覚：画素分類精度 約99.8%、位置推定誤差 平均 約0.45 cm。

## 工夫した点・開発中に見つけた問題

1. **観測のマルコフ性**：エキスパートは「グリッパが静止するまで待つ」判断をしていたが、ポリシーの入力にはその情報がなく、同じ観測に「待つ」「下降する」の2つのラベルが付いていた → ポリシーが停止したまま動かなくなった。**サーボ追従誤差**を入力に加えて解決（模倣学習では、教師の判断理由が観測から見えることが必要）。
2. **Temporal ensembling の失敗**：古いチャンクと新しいチャンクの平均で行動が打ち消し合い停止 → 先頭行動のみ実行・毎ステップ再推論に変更（61% → 100%）。
3. **入力設計**：絶対座標を除き、相対位置と「1ステップ分にクリップした相対位置」を入力にして、目標付近の小さな誤差を見やすくした。
4. **シミュレーション側の問題**：重力による Z 軸の 6 mm の沈み込み（重力補償で解決）、指令値の先走り（指令位置基準に変更＋サーボ剛性向上）、macOS で2つ目のオフスクリーン描画が壊れる問題（描画器を共有）。

## 制約

- 計算資源：GPU なしの Mac（CPU のみ）で開発・実行したため、軽量な MLP を採用しました。画像を直接入力する end-to-end ポリシーや ACT 本体は使っていません。
- グリッパは「磁石」を簡略化して実装（把持中はキューブを先端に拘束）。摩擦による把持は再現していません。
- 学習・評価とも同一のシミュレーション環境で、実機・実カメラへの汎化は未検証です。
- 次に取るキューブの選択と退避動作はルールベースです。

---

<a id="english"></a>
# PAI Final Assignment: Learned Perception + Imitation-Learned Policy for Gantry Sorting & Stacking

A gantry (Cartesian) robot in MuJoCo sorts six cubes (3 colors x 2) onto goal pads of matching color and stacks them two high.

- **Perception (AI):** a learned pixel-wise segmentation model detects the cubes in a top-camera image
- **Motion (AI):** a **neural-network policy** trained by imitation learning (behavior cloning + action chunking) from expert demos controls the robot
- **Closed loop:** the scene is re-perceived before every pick, so the robot adapts when a cube is moved mid-task

Demo video (1x speed): https://youtu.be/-3nVhX7dr_U

It implements Lecture 5's collect → train → deploy → evaluate pipeline in MuJoCo, with a scripted expert as the teacher instead of teleoperation.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python collect.py 200        # collect demos (200 layouts -> ~1200 episodes, data/demos.npz)
.venv/bin/python train_bc.py 40        # train the policy (~20 s on CPU; policy.pt, docs/debug_policy.png)
.venv/bin/python run_policy.py eval 50 # closed-loop evaluation on 50 unseen layouts
.venv/bin/python run_policy.py video   # demo.mp4 with the learned policy in control
.venv/bin/python sort_sim.py 2         # reference: scripted version (v2)
```

A trained `policy.pt` is included, so `eval` / `video` run immediately. Tested on macOS (Apple M2, CPU only), Python 3.13, MuJoCo 3.x.

## Files

| File | Contents |
|---|---|
| `sort_sim.py` | Scene, learned perception (pixel segmentation), scripted demo |
| `policy_env.py` | Environment with a 10 Hz action interface, the expert (teacher), episode runner |
| `collect.py` | Demo collection with DART-style noise injection |
| `train_bc.py` | Action-chunking policy training and debugging (prediction vs. ground-truth plot) |
| `run_policy.py` | Closed-loop evaluation and demo video |

## Architecture

| Part | Details |
|---|---|
| Perception (AI) | MLP (scikit-learn) classifying each pixel's RGB as background/red/green/blue; labels generated automatically by MuJoCo's segmentation renderer; connected components -> world coordinates via camera geometry |
| Policy (AI) | Input: relative position to target cube and goal, height, holding flag, servo tracking error (16-D). Output: next 10 actions (dx, dy, dz, magnet on/off). 3-layer MLP (PyTorch), L1 + BCE loss |
| Demo collection | Scripted expert as teacher; the robot executes noisy actions while the clean action is recorded (DART), so recovery from errors is learned too |
| Execution | Execute the first action of each chunk and re-infer every step (same as Lecture 5's `n_action_steps: 1`) |
| Rule-based parts | Choosing the next cube (nearest) and parking the gantry before taking an image |

## Results (50 unseen layouts = 300 cubes, `docs/eval_results.txt`)

| Controller | Cubes sorted | Layouts fully solved | Timed-out picks |
|---|---|---|---|
| Expert (teacher) | 300/300 | 50/50 | 0 |
| Expert + disturbance | 300/300 | 50/50 | 0 |
| **Learned policy** | **300/300** | **50/50** | 13 |
| **Learned policy + disturbance** | **300/300** | **50/50** | 15 |
| Policy + ACT temporal ensembling (ablation) | 183/300 (61%) | 18/50 | 218 |

- About 4% of pick attempts timed out, but re-perception and retrying recovered all of them (first-attempt success ~96% vs. the expert's 100%).
- Perception: ~99.8% pixel accuracy, ~0.45 cm mean position error.

## Key ideas and problems found during development

1. **Markov observation:** the expert waited for the gripper to settle, but the policy could not see that, so the same observation had two labels ("wait" and "descend") and the policy froze. Adding the **servo tracking error** to the input fixed it — in imitation learning, the reason for the teacher's decision has to be observable.
2. **Temporal ensembling failure:** averaging old and new chunks cancelled the motion out and the robot stalled -> execute only the first action and re-infer every step (61% -> 100%).
3. **Input design:** no absolute coordinates; relative positions plus versions clipped to one control step, so small errors near a target stand out.
4. **Simulation issues:** 6 mm z-axis sag under gravity (gravity compensation), set-point windup (track the commanded position, stiffer servos), and a broken second offscreen renderer on macOS (one shared renderer).

## Limitations

- Compute: developed and run on a Mac without a GPU (CPU only), hence lightweight MLPs; no end-to-end image-input policy or full ACT.
- The gripper is simplified as a magnet (the held cube is constrained to the tool tip); friction grasping is not simulated.
- Trained and tested in the same simulated scene; transfer to real robots/cameras is untested.
- Choosing the next cube and parking the gantry are rule-based.
