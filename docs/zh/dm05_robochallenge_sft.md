# DM05 RoboChallenge 多任务 SFT 配置

本文档给出在 8×B200 上对 Table30 v2 数据进行 SFT 的推荐配置。比赛按
task 等权计分，因此训练默认采用 task-balanced sampling，而不是按各 task
的帧数比例采样。

## 1. 数据集与本体

仓库中有两份 Table30 v2 manifest：

| manifest | 本体 | task 数 | 可训练帧数 | 夹爪 |
|---|---|---:|---:|---|
| `data/table30v2_dexdata_binary/manifest.json` | ARX5 | 7 | 8,864,335 | 已二值化 |
| 同上 | UR5 | 3 | 6,679,218 | 已二值化 |
| `data/table30v2_dexdata_1/manifest.json` | ARX5 | 7 | 8,864,335 | 原始宽度 |
| 同上 | UR5 | 3 | 6,679,218 | 原始宽度 |
| 同上 | ALOHA | 10 | 14,128,477 | 原始宽度 |
| 同上 | DOS-W1 | 10 | 16,122,952 | 原始宽度 |

manifest 中的路径和统计来自当前本地数据。重新转换数据后，应以对应
`index_cache.json` 中的数量为准。

不同本体应分别训练 checkpoint，不建议把它们拼成一个 dataset：

- ARX5/UR5 是 7 维 state/action，ALOHA/DOS-W1 是 14 维；
- 相机数、关节定义和夹爪标定不同；
- RoboChallenge 推理时也按 robot tag 选择不同的 action 接口。

通过环境变量选择 manifest。未设置时默认使用二值化版本：

```bash
# 已二值化的 ARX5/UR5
export OPENDM_TABLE30V2_MANIFEST="$PWD/data/table30v2_dexdata_binary/manifest.json"

# 包含 ALOHA、DOS-W1 的原始宽度版本
export OPENDM_TABLE30V2_MANIFEST="$PWD/data/table30v2_dexdata_1/manifest.json"
```

切换 manifest 后必须启动新的 Python/torchrun 进程，因为数据集注册发生在
模块导入时。两份 manifest 使用相同 dataset name，但数据内容可能不同，所以
必须使用不同的 `norm_stats_root`，避免错误复用归一化统计。

## 2. Task-balanced sampling

`DMTrainer` 对注册信息中包含多个 `tasks` 的 dataset 默认启用均衡采样：

1. 从每个 `<jsonl_dir>/<task>/` 建立独立的 frame 索引；
2. 每个逻辑 epoch 给每个 task 分配相同数量的样本，不能整除时最多相差 1；
3. task 内对 frame 均匀有放回采样；
4. 生成同一个全局采样序列，再由 Accelerate 分给各个 GPU。

因此长轨迹 task 不再自动获得更高权重。例如 ARX5 的
`arrange_flowers` 和 `turn_on_the_light_switch` 在训练中的期望权重均为
`1/7`。首次生成 state/action 归一化统计时也使用相同的 task-balanced
sampling，避免 norm stats 再次偏向长 task。

训练日志应包含：

```text
Using task-balanced sampling for 7 tasks: ...
```

如需复现旧的逐帧均匀采样，可设置：

```bash
--no-trainer-config.balance-tasks
```

单 task 数据集不受该选项影响。

## 3. 8×B200 推荐配置

起始配置：

| 参数 | 推荐值 |
|---|---|
| `chunk_size` | `50` |
| 每卡 batch | `64`；OOM 时降到 `32` |
| gradient accumulation | `1` |
| global batch | `512` |
| optimizer | AdamW |
| learning rate | `2.5e-5` |
| warmup | `1000` steps |
| scheduler | `cosine_with_min_lr`，最低为峰值的 0.1 |
| save interval | `5000` steps |
| dataloader workers | 从 `16` 开始，根据 GPU 等待情况调到 `32/64` |

task-balanced sampling 不改变逻辑 epoch 的长度，仍定义为
`总可训练帧数 / global_batch`。以 global batch 512 计算：

| dataset name | 约 3 epochs | 推荐搜索区间 |
|---|---:|---:|
| `table30v2_ur5` | 39k steps | 30k–45k |
| `table30v2_arx5` | 52k steps | 40k–60k |
| `table30v2_aloha` | 83k steps | 65k–90k |
| `table30v2_dos_w1` | 94k steps | 75k–105k |

这些 epoch 是“均衡后的逻辑 epoch”：小 task 会被重采样，大 task 会被降采样。
比赛 checkpoint 应通过各 task 等权的闭环成功率选择，不能只看训练 loss。

## 4. 训练命令

下面以二值化 ARX5 为例：

```bash
export OPENDM_TABLE30V2_MANIFEST="$PWD/data/table30v2_dexdata_binary/manifest.json"

script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name table30v2_arx5 \
  --data-config.norm-stats-root ./norm_stats/table30v2_binary \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --optimizer-config.base-lr 2.5e-5 \
  --optimizer-config.warmup-steps 1000 \
  --trainer-config.per-device-train-batch-size 64 \
  --trainer-config.gradient-accumulation-steps 1 \
  --trainer-config.dataloader-num-workers 16 \
  --trainer-config.logging-steps 20 \
  --trainer-config.save-steps 5000 \
  --trainer-config.save-total-limit 12 \
  --trainer-config.num-train-steps 55000 \
  --trainer-config.wandb-project opendm_rc \
  --trainer-config.output-dir ./user_checkpoints/DM05_arx5_balanced
```

训练 `data/table30v2_dexdata_1` 中的额外本体时：

```bash
export OPENDM_TABLE30V2_MANIFEST="$PWD/data/table30v2_dexdata_1/manifest.json"
```

ALOHA 使用：

```bash
script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name table30v2_aloha \
  --data-config.norm-stats-root ./norm_stats/table30v2_dexdata_1 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --optimizer-config.base-lr 2.5e-5 \
  --optimizer-config.warmup-steps 1000 \
  --trainer-config.per-device-train-batch-size 64 \
  --trainer-config.dataloader-num-workers 16 \
  --trainer-config.logging-steps 20 \
  --trainer-config.save-steps 5000 \
  --trainer-config.save-total-limit 12 \
  --trainer-config.num-train-steps 85000 \
  --trainer-config.wandb-project opendm_rc \
  --trainer-config.output-dir ./user_checkpoints/DM05_aloha_balanced
```

DOS-W1 使用相同配置，替换以下两项：

```bash
--data-config.dataset-name table30v2_dos_w1
--trainer-config.num-train-steps 95000
```

并使用独立输出目录，例如
`./user_checkpoints/DM05_dos_w1_balanced`。

## 5. Checkpoint 评测

建议分别评测目标步数附近的 3–5 个 checkpoint：

- ARX5：`40k/45k/50k/55k/60k`；
- UR5：`30k/35k/40k/45k`；
- ALOHA：`70k/75k/80k/85k/90k`；
- DOS-W1：`80k/85k/90k/95k/100k`。

使用 `RoboChallengeInference/test.py` 做本地接口检查，并确保：

- `robot-tag`、相机顺序和 checkpoint 本体一致；
- 训练与推理的 `chunk_size` 均为 50；
- checkpoint 中使用训练时复制的 `norm_stats.json`；
- 对每个 task 使用相同的 rollout 次数，最终按 task macro-average 选模型。

对于二值夹爪 checkpoint，部署端继续把模型的 `0/1` 输出映射到各本体真实的
close/open 命令；不要跨本体复用夹爪宽度标定。
