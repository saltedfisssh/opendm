# RoboTwin 2.0 表征消融适配（S0–S4）

本文将 [Piper 表征实验](piper_representation_ablation.md) 迁移到本地 RoboTwin 数据，暂不包含 S5。当前是数据核查与实施规格，尚未生成派生数据或启动训练。

## 1. 已核查的数据

源目录：`/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0`。

2026-09-11 本地检查：50 个任务、27,500 个 JSONL 文件，其中 clean 2,500、randomized 25,000。按每个任务、每种设置抽首个 episode，共检查 100 episodes / 21,658 帧：

- 字段均为 `images_1/2/3, state, prompt, is_robot`，state 均为 14 维；没有显式 action、EEF、base 外参。
- 仓库将其注册为 `ALOHA_ROBOTWIN2`，布局为每臂 6 关节 + 夹爪，左臂在前。
- 抽样夹爪值范围为 `[0,1]`；实际开口映射及关节符号仍需核对数据生成版本，不能照搬 Piper 的米制夹爪定义。
- 视频引用如 `./adjust_bottle/clean/episode0/cam_high.mp4`，应相对源目录的 `video/` 解析。检查的 300 个首帧视频引用均存在；尚未全量解码视频。
- 这是 Dexmal 发布的 OpenDM/Dexdata 格式导出，不是含完整仿真元信息的原始 HDF5。不能假设导出中保留了官方原始数据的所有字段。

`BuildActionChunk` 在缺少 action 时取 `state[t+1:t+51]`，末尾重复最后一帧。因此可以继续用未来状态监督，但这不证明原始仿真控制命令等于下一帧反馈。不要额外写入 `action=state[t]`，否则目标会错一帧。采样频率尚未确认，不能沿用 Piper 的 30 fps 或将 50 步直接换算成相同秒数。

## 2. 各组所需适配

| 组别 | state / action 有效维度 | 适配内容 |
|---|---|---|
| S0 | 14 / 14 | 直接保留关节 state；动作改为关节相对当前 t 的差分，夹爪保持未来绝对值 |
| S1 | 14 / 14 | 用 RoboTwin 对应机械臂 FK 得到各臂 base 系 EEF；动作是位置和轴角向量差 |
| S2 | 14 / 20 | 与 S1 共用磁盘数据；使用 `relative_mode=se3`，整个 chunk 锚定当前末端 |
| S3a | 14 / 20 | 使用仿真真实 base 外参将 state 统一到共享系，动作仍为 body-frame SE(3) |
| S3 | 23 / 20 | 在 S3a state 后附加 `T_L^-1 T_R` 的 xyz + rot6d，9 维标记 AUX，不进入 action |
| S4 | 14 / 20 | 每 episode 固定、双臂共用的水平平移和 yaw；保存变换并实现逆映射；需要新增管线 |

所有组保留 `add_state=True`，旋转 6D 沿用当前实现的前两行约定。夹爪在所有组保持同一种原始标度且不差分；若转换成物理宽度，必须先确认映射，并对所有组一致应用。

**S1–S4 的首要依赖是准确运动学。** 需要获取与这批数据生成版本匹配的 robot URDF/运动学模型、关节名称与顺序、零位与符号、左右 base 位姿、EEF link/TCP 定义。当前项目检索未发现相应 URDF，提供的数据根目录中也没有单独的机器人配置目录。应先在仿真中检查零位和若干随机关节姿态，确认 FK 与仿真末端一致，再全量转换。

不能使用 `opendm/kinematics/piper.py` 或其 `[0,-0.60,0]` 外参。若 base 随 episode 改变，需要逐 episode 的真实配置；缺失时不能以固定外参生成可信的 S3a/S3/S4。这里只读取仿真真值，不做 S5 的虚拟 base 估计。

末端可选仿真控制 TCP 或 flange，但所有组、FK、离线评测、rollout 必须一致。若使用控制 TCP，应明确写入 flange→TCP 变换。相机外参不是当前纯 RGB 训练的必需输入。

## 3. 文件与代码调整

以下文件名是建议新增项，当前尚未实现：

| 模块 | 工作 |
|---|---|
| `opendm/kinematics/robotwin2.py` | 对应仿真 embodiment 的 FK、base/TCP 变换；在线 EEF 控制需要时提供 IK |
| `script/robotwin2_prepare_ablation.py` | 读取已有 JSONL，统一划分，生成各表征，保留 prompt 和原视频引用，支持断点续跑 |
| `opendm/dataset/robotwin2_ablation.py` | 注册独立的 `robotwin2_ablation_s0/s1/s2/s3a/s3/s4`，保持 ALOHA_ROBOTWIN2 robot type，正确声明 EEF/AUX |
| `playground/dm05_robotwin2_ablation.py` | 统一相对动作、state、图像处理和训练参数；检查每组 vector/se3 模式是否正确 |
| 归一化脚本 | 参数化或提取 Piper 脚本通用部分，各组只从 train 统计；S4 独立统计 |
| 离线评测及仿真 adapter | 替换 Piper FK/外参/夹爪单位，支持 S4 逆变换，将预测还原到统一物理空间 |

可以复用通用 JSONL loader、`BuildAction`、SE(3) 数学、Normalize、AUX 排除及 action padding。不要直接把 Piper 专用 rollout/server 协议当作 RoboTwin benchmark 协议。

既有 `playground/dm05_robotwin2.py` 默认为 **absolute** 关节动作，不等于本实验的 S0。它还使用无图像增强的训练 pipeline；消融入口需要让所有组沿用同一图像处理，不能 S0 用 RoboTwin 入口、其余组用 Piper 入口而引入额外变量。

数据路径可通过现有 `--data-config.jsonl-dir`、`--data-config.image-dir` 覆盖，无需复制或移动原数据。归一化缓存 digest 不包含数据路径和完整 state 定义，因此组别、划分或源版本改变时必须使用独立数据名或缓存根目录。

## 4. 派生数据落盘约定

全部处理结果放在用户指定根目录内，建议如下；视频仅引用原文件：

```text
/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/
├── robotwin2.0/                         # 源数据
└── robotwin2_ablation/
    ├── manifest.json                    # 源版本、运动学/TCP/外参、单位、seed、S4分布
    ├── split.json                       # task/setting/episode 联合标识
    ├── episode_frames.json              # S4 episode 变换及必要真实外参
    ├── joint/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
    ├── eef_local/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
    ├── eef_unified/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
    ├── eef_unified_pair/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
    ├── eef_gravity_random/jsonl/{train,held_out}/<task>/<setting>/*.jsonl
    ├── norm_stats/
    └── reports/
```

S1/S2 共用 `eef_local`，但注册名、统计文件不同。各注册项 `image_dir` 统一设为源目录下的 `video`。加载器会往 `jsonl_dir` 写 `index_cache.json`，因此也应在派生 train/held_out 目录构建索引，避免把源目录当处理输出。

按 `(task, clean/randomized)` 分层、episode 级固定划分，所有组共用。若采用 5% 留出，需要明确小样本分层的取整规则；有同轨迹的多文本副本时必须成组划分。不能只以 `episode0` 作为全局 ID，也不能随机拆帧。训练注册只指向 train，因为 loader 递归扫描 JSONL。

clean/randomized 的 episode 数为 1:10，按帧采样的实际比例还取决于轨迹长度。应固定是否按自然帧分布训练或重采样，并分任务、分设置报告结果。原折布单任务与这里 50 任务的差异不能归因给动作表征。

## 5. 验证与实施顺序

1. 先做统一划分和 S0 数据准备，用相对关节动作跑通数据读取、统计与短训练。S0 不依赖 EEF 运动学。
2. 补齐匹配版本的运动学资产，验证关节顺序、FK、base/TCP 和夹爪语义；再生成 S1/S2/S3a/S3。
3. 实现 S4：seed + 完整 episode ID 决定固定变换；双臂及所有时刻共用，保存采样范围。重力符号和固定旋转与 S3a 对照保持一致。
4. 验证 S0 FK 与 EEF 真值一致、SE(3) 编解码往返、AUX 不进入动作、跨组划分相同、S4 state 改变而 body action/双臂相对位姿不变。
5. 各组独立统计并检查有效维度，再按统一初始化、全局 batch、步数、seed、chunk、图像增强训练。原文的 S4 尚未实现，`piper_dual.py` 中“S4 不需训练”的旧注释不适用于此设计。
6. 离线统一比较同一 TCP 的位置/旋转误差，夹爪在映射确认前报告原始标度误差，不能标 mm；在线报告各任务 clean/randomized 成功率。

RoboTwin 参考文档的 32 卡、global batch=1024、100k steps 是其参考训练配方。做新的受控消融可以另定统一资源配置，但不能把新配置结果称为原配方复现，也不能在不同组之间随意更改有效 batch。

**不训练 S5 并不意味着在线评测不需要 IK。** 若仿真 adapter 只接收关节目标，S1–S4 解码出的 EEF 目标仍需用真实机器人模型转换成关节命令；若使用仿真原生 EEF 控制器，也需要核对 TCP、控制语义和失败处理。IK/控制器误差应单独记录，避免归入模型表征误差。
