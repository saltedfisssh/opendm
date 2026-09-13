# RoboTwin 2.0 表征消融训练（S0–S4）

已实现 S0、S1、S2、S2-pair、S3a、S3、S4 的数据准备、独立注册、归一化和统一训练入口。视频原地引用，不复制。本文与 [Piper 表征实验](piper_representation_ablation.md) 对应，不包含 S5。

## 1. 环境和数据路径

| 用途 | 路径 |
|---|---|
| RoboTwin 仿真环境 | `/kpfs_ssd/data/wzy/dexbotic-benchmark/RoboTwin/.venv` |
| 仿真资产 | `/kpfs_ssd/data/wzy/dexbotic-benchmark/RoboTwin/assets` |
| 源数据 | `/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0` |
| 默认派生数据与统计 | `/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2_ablation` |
| OpenDM 训练环境 | 当前 repo 的 `.venv`，可用 `OPENDM_PYTHON` 覆盖 |

**RoboTwin 的 `.venv` 用于 SAPIEN 运动学校验；训练和转换用 OpenDM 的 Python。** 不需要在仿真环境里重新安装 OpenDM 训练依赖，也不用移动资产。下文命令均在 repo 根目录执行：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENDM_PYTHON="$PWD/.venv/bin/python"
export ROBOTWIN_ROOT=/kpfs_ssd/data/wzy/dexbotic-benchmark/RoboTwin
export SOURCE=/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0
export DATA_ROOT=/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2_ablation
```

已核查的源数据包含 50 个任务、27,500 episodes（clean 2,500、randomized 25,000）。JSONL 在 `jsonl/<task>/<setting>/episode*.jsonl`，视频引用相对 `video/` 解析。此前抽样的 100 episodes / 21,658 帧均为 14 维 state，没有显式 action、EEF 或 base 外参；这不是原始仿真 HDF5。

转换器要求 state 有限且为 14 维，拒绝含显式 action 的输入，避免静默更改监督定义。`BuildActionChunk` 从未来 `state[t+1:t+51]` 构造目标，尾部重复最后一帧；不会写入错误的 `action=state[t]`。这不证明仿真控制命令等于下一帧反馈。源采样率尚未确认，不将 50 步换算成 Piper 的时间长度。

## 2. 运动学与坐标定义

读取 `assets/embodiments/aloha-agilex/config.yml` 及其 `urdf/arx5_description_isaac.urdf`：

- 左右关节分别是 `fl_joint1…6`、`fr_joint1…6`，state 左臂在前，每臂 6 关节 + 夹爪。
- 局部 base 为 `fl_base_link`、`fr_base_link`；共享系 W 为左臂 base。右臂安装变换由 URDF 的固定链计算，包含安装旋转，**不使用 Piper 的 `[0,-0.60,0]` 近似**。
- EEF 采用原始 `move_group` link 坐标：`fl_link6/fr_link6`，不加 TCP offset。这与 RoboTwin `_trans_endpose` 处理后的控制 TCP 不同；未来 rollout adapter 必须显式转换。
- 该 embodiment 是一个固定双臂 articulation。共同 root pose 在左右 base 相对变换中抵消；若换成独立机械臂或可变双臂安装，不能沿用本实现。
- 夹爪保留导出的原始标度，不差分。当前仿真配置有 `gripper_scale=[-0.01,0.045]`，但不把导出值直接称为米制开口宽度。

独立 FK 校验命令（CPU，无渲染）：

```bash
"$ROBOTWIN_ROOT/.venv/bin/python" script/robotwin2_check_fk.py \
  --assets "$ROBOTWIN_ROOT/assets" --samples 32 --out /tmp/robotwin2_fk_check.json
```

脚本在临时目录去掉 URDF visual/collision，只保留运动学和惯性，再通过 SAPIEN 加载；比较零位和关节限位内随机姿态。2026-09-12 实测 32 组 × 双臂，最大矩阵元素误差 `1.35e-6`，低于 `2e-6` 门槛。

**校验确认的是当前资产的 FK 实现，不证明 Dexmal 历史导出与当前资产版本完全一致。** manifest 记录配置/URDF SHA256、关节名、EEF/base link 和 base 变换。正式结果应注明这一版本假设；若取得生成版本或原始 EEF 真值，应进一步比对。

## 3. 七组统一定义

| 组别 | 磁盘表征 | state / action 有效维度 | action 编码 |
|---|---|---|---|
| S0 | `joint` | 14 / 14 | 当前关节锚点的向量差，夹爪未来绝对值 |
| S1 | `eef_local` | 14 / 14 | 各臂 base 下 xyz、轴角向量差 |
| S2 | `eef_local` | 14 / 20 | 当前末端系 `T_t⁻¹T_{t+k}`，xyz + rot6d |
| S2-pair | `eef_local_pair` | 23 / 20 | S2 局部 state + 当前双手相对位姿 AUX；action 同 S2 |
| S3a | `eef_unified` | 14 / 20 | state 统一到 W，action 同 S2 |
| S3 | `eef_unified_pair` | 23 / 20 | S3a state 追加 `T_L⁻¹T_R` 的 xyz + rot6d；9 维 AUX 不进入 action |
| S4 | `eef_gravity_random` | 14 / 20 | 每 episode 共同换系的 state；action 同 S2 |

所有组均 `add_state=True`、chunk=50、相同图像增强和训练参数，rot6d 使用旋转矩阵前两行。S1/S2 共用磁盘数据，但统计文件独立。消融入口默认使用通用 OpenDM 图像增强，区别于原 `dm05_robotwin2.py` 的无增强绝对关节配方。

S4 使用 `T^G=H_e T^W`，`H_e=[Rz(yaw), (a,b,0)]`。默认 `yaw~U[-π,π)`，`a,b~U[-0.5,0.5] m`，SHA256(seed + 完整 episode 相对路径) 决定采样；双臂、整个 episode、所有未来目标共用一个变换。保留 +z 向上、原高度零点，`C=I`，与 S3a 完全一致。这不照搬图示或 HiFi-UMI 的 +z 向下约定。

变换保存于 `episode_frames.json`，解码后的 G 系 EEF 目标先经 `undo_episode_frame(poses,H_e)` 返回 W，再按臂映射回 base。state 会改变，body action 与双臂相对位姿不变；因此 S4 仍需训练，不能用动作不变性测试代替。

### S2-pair：局部 state 加显式双手几何

命令行统一使用 `s2_pair`（下划线），表中写作 S2-pair。磁盘 state 前 14 维与 S2 的 `eef_local` 完全一致；追加的 9 维与 S3 的 `eef_unified_pair` 尾部完全一致，都是当前时刻 `inv(T_L^W) @ T_R^W` 的 xyz + rot6d。它表达右末端在左末端系中的位姿，不是世界系 xyz 的直接相减。

当前实现用左臂 base 作为共同 W，再用 URDF 的真实安装关系把右手变到 W。若以第一帧相机为世界系，同时左乘两手同一个 H，`inv(H T_L) @ (H T_R) = inv(T_L) @ T_R`，因此无需额外相机标定就能得到同一个相对特征。原局部 state 不换系；不能把左右不同 base 系的原始位姿直接相乘。新增观测只使用当前帧，不读取未来信息；AUX 不进入 20 维动作编码。

| state 参考系 | 无相对几何 | 有相对几何 |
|---|---|---|
| 各臂局部 base | S2 | S2-pair |
| 共享 W | S3a | S3 |

S2-pair 对 S2 只增加显式几何；S3 对 S2-pair 只统一原始 state 参考系。S2-pair 使用独立注册与统计缓存，不能复用 S2 的 14 维 state 统计。

**已有数据增量补齐命令**（复用相同 manifest、seed、随机范围与 split，不改原有组）：

```bash
"$OPENDM_PYTHON" script/robotwin2_prepare_ablation.py \
  --source "$SOURCE" --out-root "$DATA_ROOT" --assets "$ROBOTWIN_ROOT/assets" \
  --rung s2_pair --seed 0 --s4-xy-range 0.5 --workers 10
"$OPENDM_PYTHON" script/robotwin2_compute_norm_stats.py \
  --source "$SOURCE" --data-root "$DATA_ROOT" --rung s2_pair --workers 10
# 使用 §5 的统一初始化与 8 卡环境变量
bash script/train_robotwin2_ablation.sh s2_pair
```

输出位于 `eef_local_pair/jsonl/{train,held_out}/`。其余组的数据和统计无需重算。小数据验证可以执行 `bash script/smoke_robotwin2_ablation.sh s2_pair`，并按 §5 指定全新的 smoke 路径。

## 4. 数据转换与断点续跑

先做小规模检查，使用**独立 smoke 目录**：

```bash
"$OPENDM_PYTHON" script/robotwin2_prepare_ablation.py \
  --source "$SOURCE" --assets "$ROBOTWIN_ROOT/assets" \
  --out-root /tmp/robotwin2_ablation_smoke --limit 2 --workers 1

"$OPENDM_PYTHON" script/robotwin2_compute_norm_stats.py \
  --source "$SOURCE" --data-root /tmp/robotwin2_ablation_smoke \
  --workers 0 --batch-size 16 --max-batches 1
```

再生成全量七组（视频保持原地引用）：

```bash
"$OPENDM_PYTHON" script/robotwin2_prepare_ablation.py \
  --source "$SOURCE" --out-root "$DATA_ROOT" \
  --assets "$ROBOTWIN_ROOT/assets" \
  --rung s0 s1 s2 s2_pair s3a s3 s4 --seed 0 --s4-xy-range 0.5 --workers 10

"$OPENDM_PYTHON" script/robotwin2_compute_norm_stats.py \
  --source "$SOURCE" --data-root "$DATA_ROOT" --workers 32
```

**2026-09-13 已修复全量转换停滞。** 原 Python 3.10 `ProcessPoolExecutor.map` 一次提交 27,500 个任务；实测调用栈显示主线程持锁写满 wakeup pipe，管理线程等同一把锁，worker 等待任务。改为 `spawn`、最多 `2×workers` 个在途任务，完成一个补一个，按完成顺序收结果。每 100 个 episode 或约 10 秒打印完成数、跳过数、速率、ETA；无任务完成时每 10 秒打印等待中的最老 episode。`reports/conversion_progress.json` 也保存进度，`limited=true` 表示只处理了 `--limit` 子集。修复保留原有 manifest、划分和数值定义，可使用原命令断点续跑，无需删掉已完成的数据。

可先 `--rung s0`，再用相同参数补其他组。转换按 episode 原子落盘；每个 episode 的完成记录含源文件 SHA256，重跑跳过已完成表征。配置、资产哈希、源 episode 清单或已处理源内容变化会拒绝复用；这时应使用新的输出根目录和统计缓存。不要在训练/统计运行时同时修改派生数据。

划分按 `(task, clean/randomized)` 分层，episode 级固定，共享 `split.json`。每层留出数为 `min(n-1, max(1, floor(n*0.05+0.5)))`：50 clean 留出 3，500 randomized 留出 25，共 **26,100 train / 1,400 held out**。`--limit` 只限制转换量，不改变完整划分。完整 ID 含任务和设置，不会混淆不同任务的 episode0。当前约定一文件一轨迹；若引入同轨迹多文本副本，须先按轨迹归组，不能直接按文件划分。

```text
robotwin2_ablation/
├── manifest.json
├── split.json
├── episode_frames.json
├── joint/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
├── eef_local/jsonl/{train,held_out}/...
├── eef_local_pair/jsonl/{train,held_out}/...
├── eef_unified/jsonl/{train,held_out}/...
├── eef_unified_pair/jsonl/{train,held_out}/...
├── eef_gravity_random/jsonl/{train,held_out}/...
├── norm_stats/
└── reports/episodes/...
```

训练只注册 train；held out 不会被递归扫描进训练集。自然按帧采样，不进行任务或 clean/randomized 重采样。统计只读 train；正式统计不要带 `--max-batches`。smoke 统计不可当作正式统计，且现有文件会被跳过。

## 5. 各实验的 8 卡训练命令

以下命令在 **8 卡服务器的 repo 根目录**执行。先用 §4 的命令完成全量转换和七组完整 train 统计，再启动训练。将路径改为服务器实际挂载；`SOURCE` 指向含 `video/` 的源目录，`DATA_ROOT` 指向派生数据根目录，不能指向 `/tmp` smoke 数据。

```bash
export OPENDM_PYTHON="$PWD/.venv/bin/python"
export SOURCE=/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0
export DATA_ROOT=/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2_ablation
export MODEL_PATH="$PWD/checkpoints/DM05"  # 事先准备；七组共用同一份初始化
export OUTPUT_ROOT="$PWD/user_checkpoints/robotwin2_ablation"
export NPROC_PER_NODE=8
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=8
```

默认训练配方已改为较小的 **每卡 batch=2、梯度累积=8**：8 卡全局 batch 仍是 `8×2×8=128`。统一使用 60k optimizer steps、lr=4e-5、Muon+AdamW、chunk=50、state 输入和同一图像增强；每 10k steps 保存模型及优化器状态。`DM05-MEM` / `DM05-robotwin2-bf16` 是不同初始化，若使用，必须对所有组同时设置同一个 `MODEL_PATH`；不能使用 smoke 产物作为正式初始化。

| 实验 | 数据名 | state/action 有效维度 | 独立训练命令 |
|---|---|---|---|
| S0 | `robotwin2_ablation_s0` | 14 / 14 | `bash script/train_robotwin2_ablation.sh s0` |
| S1 | `robotwin2_ablation_s1` | 14 / 14 | `bash script/train_robotwin2_ablation.sh s1` |
| S2 | `robotwin2_ablation_s2` | 14 / 20 | `bash script/train_robotwin2_ablation.sh s2` |
| S2-pair | `robotwin2_ablation_s2_pair` | 23 / 20 | `bash script/train_robotwin2_ablation.sh s2_pair` |
| S3a | `robotwin2_ablation_s3a` | 14 / 20 | `bash script/train_robotwin2_ablation.sh s3a` |
| S3 | `robotwin2_ablation_s3` | 23 / 20 | `bash script/train_robotwin2_ablation.sh s3` |
| S4 | `robotwin2_ablation_s4` | 14 / 20 | `bash script/train_robotwin2_ablation.sh s4` |

每条命令各占 8 张卡。脚本自动设置对应 JSONL、统计路径和 vector/se3 模式，checkpoint 分别保存在 `$OUTPUT_ROOT/<rung>/checkpoint-<step>`。同一输出目录已有 checkpoint 时入口会自动恢复；更换初始化、数据或配方时请使用新输出目录。

在同一台服务器上顺序跑完七组并保留日志：

```bash
set -euo pipefail
mkdir -p results/robotwin2_ablation
for rung in s0 s1 s2 s2_pair s3a s3 s4; do
  bash script/train_robotwin2_ablation.sh "$rung" \
    2>&1 | tee "results/robotwin2_ablation/train_${rung}.log"
done
```

若显存仍不足，可以对**所有组统一**改成 `BATCH_SIZE=1 GRAD_ACCUM_STEPS=16`，全局 batch 仍为 128。每卡 batch=4 时对应累积=4。不要只修改某一组的有效 batch。也可在命令末尾传入 `--trainer-config.*` 覆盖参数。

跨服务器复制派生数据时，不携带 `index_cache.json`（缓存内有绝对 JSONL 路径），在目的端重新生成；保留 `manifest.json`、`split.json`、S4 的 `episode_frames.json` 和同版本完整 `norm_stats/`。训练使用上面的 SOURCE/DATA_ROOT 覆盖路径，无需重写视频内容。已生成数据可直接训练；若在新路径继续转换，manifest 的源路径检查可能要求使用新的派生根目录。

本地只验证了单卡；**8 卡 FSDP 尚未在本机实测**。原 RoboTwin 文档的 32 卡、global batch=1024、100k steps 是另一套参考配方，本实验不宣称复现它。

### 小数据集端到端验证命令

提供独立脚本，执行 20 episodes 转换、所选组的完整小数据统计，以及每组 2 steps 的单卡训练（batch=1、累积=1）。仅保存模型 checkpoint，节省 smoke 的磁盘占用；正式训练继续保存优化器状态。

```bash
MODEL_PATH="$PWD/checkpoints/DM05-MEM" \
SMOKE_DATA_ROOT=/tmp/robotwin2_ablation_smoke_run \
SMOKE_OUTPUT_ROOT="$PWD/user_checkpoints/robotwin2_smoke_run" \
SMOKE_LOG_ROOT="$PWD/results/robotwin2_smoke_run" \
bash script/smoke_robotwin2_ablation.sh

# 只检查一组时，末尾追加组别，例如 s4
```

此脚本使用 `SMOKE_DATA_ROOT`，不会使用继承的全量 `DATA_ROOT`；同样，smoke 输出由 `SMOKE_OUTPUT_ROOT` 指定。重复验证时选新的 smoke 输出和数据目录：脚本拒绝已有输出或已有统计的 smoke 根目录，避免恢复旧 checkpoint 或复用另一子集的统计。`--limit` 选取排序后的前 N 个 episode，验证的是执行路径，不是覆盖所有任务的质量评测。

## 6. 实现与验证边界

- `opendm/kinematics/robotwin2.py`：从实际 URDF 计算批量 FK 和真实固定 base 关系。
- `opendm/data/episode_frame.py`：S4 确定性采样、双臂变换、逆映射；Piper 共用。
- `script/robotwin2_check_fk.py`：在提供的仿真环境中独立核验。
- `script/robotwin2_prepare_ablation.py`：统一划分、转换、manifest、断点续跑。
- `opendm/dataset/robotwin2_ablation.py`：七组注册。
- `playground/dm05_robotwin2_ablation.py`、`script/train_robotwin2_ablation.sh`：统一训练。
- `script/smoke_robotwin2_ablation.sh`：隔离的小数据转换、统计和逐组训练验证。
- `script/robotwin2_compute_norm_stats.py`：独立 train 统计。

2026-09-13 在 20 个真实 episodes / 2,892 帧上完成六组转换和完整小数据统计；数据只用于执行路径验证，不代表 50 任务分布。转换修复的回归测试还用 27,500 个轻量任务验证了有界提交不会卡住，未借此声称已完成全量真实数据转换。

小数据 GPU 运行记录及 checkpoint 结果见本节末的验证表。此前 S4 在其他进程占用约 27 GiB 显存时于 Muon 更新阶段 OOM；本次使用清理后的单张 A800 80GB、`DM05-MEM` 初始化、batch=1、累积=1、2 个 data worker。`dataloader_num_workers=0` 会触发现有 Trainer 的预取参数校验，示例保持为 2。

尚未实现 RoboTwin checkpoint 的统一离线评测入口和在线 benchmark EEF adapter。不要把 Piper 的 server/rollout 协议直接用于 RoboTwin。EEF rollout 需要从原始 link frame 转换到控制 TCP，或通过真实模型 IK 下发关节；S4 必须应用保存的逆变换。无 S5 也不意味着不需要 IK。

最终应在同一末端定义下比较位置/旋转误差；夹爪映射未确认前报告原始标度误差，不标 mm。在线按任务、clean/randomized 报成功率；把 IK/控制器误差单独记录。原始训练 loss 跨动作表征不可直接比较。

### 2026-09-13 小数据验证结果

六组均在单张 A800 80GB 上完成 2 个 optimizer steps，并成功保存 `checkpoint-2` 和最终模型。逐组读取 `trainer_state.json`、checkpoint 的 `norm_stats.json` 与 safetensors 验证：训练步数=2，loss 有限，统计维度正确且与训练输入统计一致；抽查的 action-expert 参数与初始化相比均发生变化。

| 实验 | state / action 维度 | 完成步数 | checkpoint / 统计 / 参数更新 |
|---|---|---|---|
| S0 | 14 / 14 | 2 | 通过 |
| S1 | 14 / 14 | 2 | 通过 |
| S2 | 14 / 20 | 2 | 通过 |
| S3A | 14 / 20 | 2 | 通过 |
| S3 | 23 / 20 | 2 | 通过 |
| S4 | 14 / 20 | 2 | 通过 |

- [完整验证记录](assets/robotwin2_ablation_smoke_20260913.json)。2 steps 仅证明训练执行路径可用，不评价收敛、跨组优劣或任务成功率。
- 日志与检查脚本：`results/robotwin2_smoke_20260913/`；机器可读汇总：`summary.json`。
- checkpoint：`user_checkpoints/robotwin2_smoke_20260913/<rung>/checkpoint-2/`。
- 本次转换相关回归测试 6 项通过，包括 27,500 任务进程池回归；shell 语法和 `git diff --check` 通过。
- 本次未继续全量转换或全量训练；8 卡执行由目标服务器完成。

### S2-pair 追加验证

新增组单独使用相同的 20 episodes / 2,892 帧，在单张 A800 80GB 上以 batch=1、累积=1 完成 2 steps。`checkpoint-2` 的 state/action 统计为 23/20 维，loss 有限，检查的 action-expert 权重发生更新。详见 [S2-pair 验证记录](assets/robotwin2_s2_pair_smoke_20260913.json)。

新增及相关回归测试共 40 项通过，覆盖：与 S2 的局部 state 和动作完全一致、与 S3 的相对特征完全一致、共同世界系变换不变性、AUX 排除、Piper 离线解码、独立缓存，以及增量补数据不改旧组文件/manifest/split。RoboTwin 实测日志在 `results/robotwin2_s2_pair_smoke/`；Piper 此组已通过数值与解码测试，未开展完整 GPU 训练或真机 rollout。
