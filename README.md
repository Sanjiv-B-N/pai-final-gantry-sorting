# PAI最終課題：学習型ビジョンによるガントリーロボットの色分け仕分け＆積み上げ

[English](#english) | 日本語

MuJoCo 上のガントリー（直交）ロボットが、俯瞰カメラ画像から**学習した画素セグメンテーションモデル**で6個のキューブ（3色×2個）を検出し、同じ色のゴールパッドへピック＆プレースで**仕分けて2段に積み上げ**ます。毎回のピック前に再認識する**閉ループ**構成で、作業中にキューブが外乱で動かされても対応します。

デモ動画（等倍速）: https://youtu.be/zjxQRYK-bYM

## 実行方法

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python sort_sim.py 2   # 引数 = エピソード数（毎回ランダム配置）
```

`demo.mp4`（1280x720, 30fps, 等倍速）が生成されます。左上に AI の認識結果（セグメンテーション＋検出位置）を表示。動作確認環境：macOS, Python 3.13, MuJoCo 3.x。GPU不要。

## 構成

| 要素 | 内容 |
|---|---|
| ロボット | X/Y/Z 3軸スライド関節のガントリー、位置制御アクチュエータ、磁石式グリッパ |
| 知覚（AI） | 画素ごとの RGB → {背景, 赤, 緑, 青} を分類する MLP（scikit-learn）。教師ラベルは MuJoCo のセグメンテーション描画から自動生成 |
| 個体検出 | 分類マスクの連結成分ラベリング → 各成分の画素中央値 → ピンホールカメラ幾何で世界座標へ |
| 計画 | 未処理キューブのうちエンドエフェクタに最も近いものを選択。パッドごとの段数を数えて積み上げ高さを決定 |
| 閉ループ | 毎ピック前にガントリーを視野外へ退避→再撮影→再推論。2個目の把持後にランダムな外乱でキューブを移動 |

## 設計思想：方策学習を「使わない」という判断

Physical AI では end-to-end の方策学習が注目されますが、すべてを学習で置き換えるのが最適とは限りません。本課題では「不確実性がある所（物体の位置＝知覚）だけ学習し、既知の所（ガントリーの運動学）は古典制御に任せる」ハイブリッド構成を選びました。

- 少量データ（60画像）・CPUのみ数秒で学習できる
- 失敗原因を知覚／制御に切り分けて検証できる
- 動作範囲が予測可能で安全性を説明しやすい（産業用ピッキングでも一般的な構成）
- 限界：接触の多い作業や柔軟物の操作など、モデル化が難しい場面では方策学習が有利

## 結果（seed=1）

- 未学習の評価画像40枚（240個）で：画素分類精度 99.80%、検出 240/240、位置推定誤差 平均 0.44 cm / 最大 1.10 cm
- 2エピソード × 6キューブ = 12/12 個を正しいパッドへ仕分け・積み上げ（外乱あり）

## 制約・正直な注記

- 計算資源の制約：GPUのない Mac（CPUのみ）で開発・実行したため、軽量な MLP と MuJoCo を採用し、強化学習などによる方策（policy）の学習は見送りました（制御はルールベース）。
- グリッパは「磁石」を簡略化して実装（把持中はキューブをエンドエフェクタ位置に拘束）。摩擦による把持の物理シミュレーションではありません。
- 学習・評価とも同一のシミュレーション環境・照明で行っており、実カメラ画像での汎化は未検証です。
- 同色キューブが接触すると連結成分が1つに結合するため、配置時に最小間隔 10 cm を設けています。

---

<a id="english"></a>
# PAI Final Assignment: Vision-Guided Color Sorting & Stacking with a Gantry Robot

A gantry (Cartesian) robot in MuJoCo uses a **learned pixel-wise segmentation model** on a top-camera image to detect six cubes (3 colors x 2), then picks each one and **sorts and stacks it two high** on the goal pad of the same color. It re-perceives the scene before every pick (**closed loop**), so it adapts when a cube is moved mid-task.

Demo video (1x speed): https://youtu.be/zjxQRYK-bYM

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python sort_sim.py 2   # argument = number of episodes (random layout each time)
```

This writes `demo.mp4` (1280x720, 30 fps, 1x speed) with the AI's view (segmentation + detections) shown top-left. Tested on macOS, Python 3.13, MuJoCo 3.x. No GPU needed.

## Architecture

| Part | Details |
|---|---|
| Robot | X/Y/Z prismatic-joint gantry, position actuators, magnetic gripper |
| Perception (AI) | MLP (scikit-learn) classifying each pixel's RGB as background/red/green/blue. Training labels are generated automatically by MuJoCo's segmentation renderer |
| Instance detection | Connected components on the class mask -> per-component pixel median -> world coordinates via pinhole camera geometry |
| Planning | Pick the unsorted cube nearest the end effector; stack height from a per-pad counter |
| Closed loop | Before every pick: park the gantry out of view -> capture -> infer. After the 2nd grasp, a random disturbance moves a cube |

## Design rationale: choosing not to learn a policy

End-to-end policy learning gets much attention in Physical AI, but learning everything is not always best. This project uses a hybrid: learn only where there is uncertainty (where the objects are = perception) and use classical control where things are known (the gantry's kinematics).

- Trains from little data (60 images) in seconds on a CPU
- Failures can be traced to perception or control and verified separately
- Motion is predictable, which makes safety easy to argue (a common architecture in industrial picking)
- Limit: for contact-rich or deformable-object tasks that are hard to model, learned policies have the edge

## Results (seed=1)

- On 40 unseen test images (240 cubes): pixel accuracy 99.80%, detections 240/240, position error mean 0.44 cm / max 1.10 cm
- 2 episodes x 6 cubes = 12/12 correctly sorted and stacked, including a mid-task disturbance

## Limitations

- Compute: developed and run on a Mac without a GPU (CPU only), hence a lightweight MLP + MuJoCo and no policy learning (e.g. RL); control is rule-based.
- The gripper is simplified as a magnet (the held cube is constrained to the tool tip); friction grasping is not simulated.
- Training and testing use the same simulated scene and lighting; generalization to real camera images is untested.
- Touching same-color cubes merge into one connected component, so cubes are spawned at least 10 cm apart.
