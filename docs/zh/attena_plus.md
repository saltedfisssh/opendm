# DM05 AttenA+ 速度注意力

本文档说明从 `AttenA-Plus` 仓库迁移到 OpenDM 的速度场动作注意力实现、配置方式和验证方法。

## 原理与迁移范围

AttenA+ 根据真实动作的速度大小对每个时间步加权：低速动作通常对应抓取、对齐和精确放置等关键阶段，因此获得更高的训练权重；快速过渡动作获得较低权重。对动作块
`A ∈ R^(B×T×D)`，使用前 `J` 个运动维度计算：

```text
speed = ||A[..., :J]||₂
weight = f(speed)
weight = clip(weight, 1 / clip_max, clip_max)
```

OpenDM 的 DM05 是 flow-matching 模型，实际回归目标为 `u_t = noise - action`。因此权重由原始、已归一化的真实 `action` 计算，再乘到 `v_t` 与 `u_t` 的逐元素 MSE 上。`action_mask` 会同时用于速度计算和 loss reduction，填充的动作维度不参与损失。

本次迁移包括：

- `opendm.losses.VelocityAttention`：可复用的 PyTorch 核心实现；
- DM05 flow-matching loss 集成；
- 可通过命令行控制且会写入 checkpoint 的模型配置；
- 单元测试和本文档。

模型结构、采样过程和推理接口均未改变。`use_velocity_attention` 默认为 `false`，旧 checkpoint 和原训练行为保持兼容。

## 开启训练

在原有 DM05 训练命令后增加：

```bash
script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.use-velocity-attention true \
  --model-config.velocity-weight-strategy inverse_squared \
  --model-config.velocity-clip-max-weight 2.0 \
  --model-config.velocity-joint-dims 6
```

推荐先使用论文默认组合 `inverse_squared + clip_max_weight=2.0`。训练入口会读取数据集的 `state_desc`，自动选择所有标记为 `joint` 的动作维度，并排除夹爪。单臂 6 关节加夹爪会解析为 `[0, 1, 2, 3, 4, 5]`。

Table30 v2 的 Aloha 和 dos_w1 均按下面的 14 维顺序存储：

```text
[左臂关节 0:6, 左夹爪 6, 右臂关节 7:13, 右夹爪 13]
```

开启 AttenA+ 后会自动解析为：

```text
[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
```

因此两个夹爪不会进入速度范数，但加权结果仍会广播到包括夹爪在内的所有有效动作维度。

## 配置项

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `use_velocity_attention` | `false` | 是否启用 AttenA+ |
| `velocity_weight_strategy` | `inverse_squared` | `inverse`、`inverse_squared`、`exp_decay` 或 `log` |
| `velocity_clip_max_weight` | `2.0` | 权重上界，下界为其倒数 |
| `velocity_epsilon` | `1e-3` | 速度下限，避免除零 |
| `velocity_alpha` | `5.0` | `exp_decay` 的衰减系数 |
| `velocity_normalize_weights` | `true` | 按 AttenA+ 公式执行 `weight / clip_max * 2` |
| `velocity_joint_dims` | `6` | 用于计算速度的前置动作维度数 |
| `velocity_joint_indices` | `null` | 显式关节索引；训练时为空则根据数据集 `state_desc` 自动解析 |

注意：权重从归一化后的训练 action 计算，因此不同数据集的归一化统计会影响速度分布。对新机器人建议比较开启前后的 loss 尺度，并在短训练中检查权重分布后再运行完整实验。

如绕过 `DM05Exp` 直接调用模型，或者数据集没有 `state_desc`，则使用 `velocity_joint_dims` 指定的前置维度作为兼容回退。也可以在 Python 配置中显式设置：

```python
model_config.velocity_joint_indices = [
    0, 1, 2, 3, 4, 5,
    7, 8, 9, 10, 11, 12,
]
```

## 独立调用

```python
from opendm.losses import VelocityAttention

attena = VelocityAttention(
    weight_strategy="inverse_squared",
    clip_max_weight=2.0,
    joint_indices=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
)
loss = attena.weighted_loss(
    ground_truth=actions,
    predicted=v_t,
    target=noise - actions,
    loss_type="mse",
    action_mask=action_mask,
)
```

输入形状均为 `(B, T, D)`。`reduction="none"` 可返回 `(B,)` 的逐样本 loss。

## 验证

在仓库同名 conda 环境中运行：

```bash
conda run -n opendm python -m pytest tests/test_velocity_attention.py -q
```

测试覆盖四种权重策略、速度单调性、裁剪、flow target、动作维度 mask、关闭加权时的等价性及非法配置。
