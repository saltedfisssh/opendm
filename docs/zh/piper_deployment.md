# Piper 云端推理与本地真机部署

配合 [表征阶梯实验](piper_representation_ablation.md) 使用。云端加载 checkpoint，
本地采集 Head / Left wrist / Right wrist 三路相机和双臂反馈，通过
`http://IP:端口/v1/infer` 请求 action chunk，再控制双臂。本地无需 GPU、Torch 或模型权重。

当前可运行的本地客户端是 `script/piper_rollout.py`，直连 SDK/CAN/RealSense。
后文「ROS 部署」保留了复用 `web_console_bundle` 遥操作容器的方案记录，但当前仓库的
`script/piper_ros_rollout.py` 是空文件，暂不能按该章节运行。S4 已接入共享
`Representation` 和 SDK 客户端，以下 S4 命令使用 SDK 入口。

## 云端

使用训练环境以及对应 checkpoint 的 `norm_stats.json`。下面启动 S2：

```bash
script/dm05_launcher.sh --exp playground/dm05_piper.py --task inference \
  --data-config.dataset-name piper_fold_s2 --data-config.relative-mode se3 \
  --model-config.model-name-or-path user_checkpoints/piper_s2/checkpoint-60000 \
  --inference-config.port 7891
```

服务监听 `0.0.0.0`。在云服务器安全组中仅允许本地机器来源 IP 访问此端口，
或通过 VPN/SSH 隧道连接；现有 Flask 服务没有认证和 TLS。

| 级别 | dataset-name | relative-mode | 输入 state | 返回 action |
|---|---|---|---:|---:|
| S0 | piper_fold_s0 | vector | 14 | 14 |
| S1 | piper_fold_s1 | vector | 14 | 14 |
| S2 | piper_fold_s2 | se3 | 14 | 20 |
| S2-pair | piper_fold_s2_pair | se3 | 23 | 20 |
| S3 | piper_fold_s3 | se3 | 23 | 20 |
| S3a | piper_fold_s3a | se3 | 14 | 20 |
| S4 | piper_fold_s4 | se3 | 14 | 20 |
| S5 | piper_fold_s5 | vector | 14 | 14 |

更换实验时同时更换 dataset、relative-mode、checkpoint 和本地 `--rung`。
服务从 dataset 自动设置输出宽度和推理默认配置；错误的 relative-mode 会在加载模型前报错。
不要开启 `--inference-config.compose-eef-rot`：本地已提供轴角 state。
本入口以无 history 的训练配置为基准；当前本地脚本不维护历史图像。

Piper 的 `/v1/infer` 使用专用协议：请求增加 `piper` 对象，服务返回
`metadata.piper`，两端严格核对。返回值是**已反归一化、尚未绝对化**的动作；
关节/EEF 的夹爪维始终是绝对宽度，不再加观测夹爪值。
通用 DM05 客户端不能直接消费该接口的相对动作。请使用下面的客户端。

## 本地安装与连接

将本仓库和 `third_party/pyAgxArm` 目录复制到机械臂机器。已有 uv 环境包含
`pyrealsense2` 时直接复用，无需重新建训练环境；以下命令用 `uv run --no-sync`
运行当前环境，避免部署时自动同步云端训练依赖。

如果是新部署机器，可安装本地最小依赖：

```bash
uv venv .venv
uv pip install numpy requests opencv-python-headless pyrealsense2
uv pip install -e third_party/pyAgxArm
```

OpenCV 仅用于 JPEG 编码，图像采集全部通过 `pyrealsense2`，不使用
`VideoCapture` 或 V4L2 设备编号。若环境已有 OpenCV，不必再安装 headless 版本。
若导入 SDK 报 `libusb-1.0.so.0` 缺失，Ubuntu/Debian 本地机器执行
`sudo apt-get install libusb-1.0-0`；相机访问权限按 RealSense 的 udev 配置处理。
SDK 目录需完整存在；从仓库根目录运行脚本，无需安装云端模型依赖。

在**连接相机的本地机械臂机器**枚举设备：

```bash
uv run --no-sync python script/piper_list_cameras.py
```

脚本输出 JSON，包含 `serial`、设备名、USB 类型及支持的彩色分辨率、帧率和格式，
只枚举设备，不启动图像流或连接机械臂。无设备时输出 `[]` 并以状态码 1 退出。
当前部署默认绑定以下三路相机，不能把枚举顺序当成相机角色：

```bash
HEAD_SERIAL=346522076596
LEFT_SERIAL=346522072780
RIGHT_SERIAL=346522075577
```

按实际设备配置 SocketCAN，例：

```bash
sudo ip link set can_l_slave up type can bitrate 1000000
sudo ip link set can_r_slave up type can bitrate 1000000
```

`can_l_slave` 是左从臂、`can_r_slave` 是右从臂，不能接主臂。
相机通过 `--cameras` 按 **Head / Left wrist / Right wrist** 顺序传入三个不同的
RealSense 序列号，保留前导零。每路仅开启 **640×480、30 FPS、BGR8 彩色流**，
不启用深度或红外；设备不支持该模式时直接报错，不自动替换相机或分辨率。应使用相同机架、相机视角和夹爪；S2-pair/S3/S3a/S4 使用数据约定的
右 base 相对左 base `[0, -0.60, 0]`。不能在换了外参的机架上直接复用。
默认固件为 `v189`，与 `test.py` 的 `PiperFW.V189` 一致；可用
`--firmware default|v183|v188|v189` 覆盖，也接受大写名称。
CAN 后端显式使用 `socketcan`。上述 CAN 名称与三路序列号已是脚本默认值，
因此命令中的 `--left-can`、`--right-can`、`--cameras` 均可省略。

先运行一次只观测与解码（不会使能或下发运动）：

```bash
uv run --no-sync python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s2 \
  --left-can can_l_slave --right-can can_r_slave --cameras "$HEAD_SERIAL" "$LEFT_SERIAL" "$RIGHT_SERIAL" --dry-run
```

输出 JSON 包含解码后 chunk 的长度、首步和末步动作。确认相机顺序、反馈、动作方向后，
在有人看护且可使用实体急停的条件下执行：

```bash
uv run --no-sync python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s2 \
  --left-can can_l_slave --right-can can_r_slave --cameras "$HEAD_SERIAL" "$LEFT_SERIAL" "$RIGHT_SERIAL" \
  --execute --speed 10
```

默认持续运行到 `--max-steps`（默认 100000 步，约 55 分钟），Ctrl-C/SIGTERM 随时停止。
停止只是不再下发新的指令——`move_j`/`move_p`/`move_js` 都是位置/速度或 MIT
跟随控制，本身就会保持在最后下发的目标上，脚本退出时不会调用电子急停或
`disable`。真正需要紧急处置时使用实体急停，而不是依赖驱动层调用；模型推理
误差略微超出目标位置不会使机械臂损坏，因此不再需要软件层面的"停止即急停"。

## 运动模式与 MIT

`--motion-mode {smooth, mit}`（默认 `smooth`）：

- `smooth`：关节档位调用 `move_j`，EEF 档位调用 `move_p`，均为固件的
  位置-速度平滑轨迹，带轨迹规划。
- `mit`：关节档位改用 `move_js`（SDK 文档称为 "MIT pass-through 模式"），
  取消平滑与轨迹规划，直接跟随下发目标，延迟更低但没有缓冲，仅限
  `--rung s0`。EEF 档位（S1/S2/S2-pair/S3/S3a/S4/S5）没有对应的 Cartesian pass-through
  接口，因此其余档位始终使用 `move_p`，`--motion-mode mit` 会在启动时报错拒绝。

## 表征与执行约定

- 所有观测由关节 FK 重算 flange，TCP offset 为 0，与训练转换一致；EEF state
  使用轴角并跨请求解缠，不依赖 SDK 的异步末端反馈。矩阵链避免四元数符号跳变。
- S0：关节相对值加当前关节，按 `--motion-mode` 调用 `move_j` 或 `move_js`。
- S1：位置相加，旋转按 `R_delta @ R_anchor` 合成。
- S2/S3/S3a：按 `T_anchor @ T_delta` 解码 6D body-frame 旋转；S3/S3a
  把右臂目标转回其自身 base。S3 额外构造 9 维双爪相对特征。
- S2-pair：前 14 维观测及动作解码与 S2 一致，仅追加共享系计算出的 9 维双爪
  相对位姿 AUX。右臂动作已在其自身 base 系，不再做共享系到右 base 的逆映射。
- S4：先按同一个 H 把两臂共享系观测映射到 G；body action 解码后经 H⁻¹ 回共享系，
  再映射到各臂自身 base。每个 rollout 内 H 固定，具体参数见下一节。
- EEF 目标调用 `move_p`，使用固件 IK。SDK 高层输入是**米、弧度**，
  `move_gripper_m` 是米和牛顿；不再乘 CAN 单位倍率。`se3.mat_to_rpy`
  每次都对旋转矩阵重新分解，返回的欧拉角本身就是 canonical 范围，
  不存在需要客户端检测或拒绝的跨 ±π 分支问题。
- 夹爪宽度裁剪到 `[0, 0.08]` m 后再下发，而不是拒绝整条指令：推理误差
  略微超出夹爪物理行程不会损坏硬件，裁剪即可。关节限位交给 SDK/固件
  （`set_joint_limits_enabled(True)`），不在客户端重复实现会拒绝整条前缀
  的步长检查。

## S2-pair 部署：保留各臂 base 观测，追加双手几何

云端使用 `piper_fold_s2_pair`，本地使用 `--rung s2_pair`。服务输入 23 维 state、
输出 20 维已反归一化的 body 相对动作。它与 S3 的维度相同，但前 14 维坐标系不同，
协议按 rung 严格区分；S2 的 14 维 state 也不能直接发送给此服务。

```bash
# 云端：使用训练保存的 S2-pair checkpoint 和其中的 norm_stats.json
PATH="$PWD/.venv/bin:$PATH" bash script/dm05_launcher.sh \
  --exp playground/dm05_piper.py --task inference \
  --data-config.dataset-name piper_fold_s2_pair --data-config.relative-mode se3 \
  --model-config.model-name-or-path /kpfs_ssd/data/wzy/data/user_checkpoints/piper_s2_pair/checkpoint-15000 \
  --inference-config.port 7891

# 本地机械臂机器：先只采集、推理和解码
uv run --no-sync python script/piper_rollout.py \
  --server http://CLOUD_IP:7891 --rung s2_pair --dry-run

# 复用现有 SDK/CAN 的 EEF 执行路径
uv run --no-sync python script/piper_rollout.py \
  --server http://CLOUD_IP:7891 --rung s2_pair --execute --speed 10
```

每臂原始观测从真实关节 FK 构造，仍为自身 base 系的 `[xyz, rotvec, gripper]`。
尾部 AUX 为 `T_L⁻¹ T_RIGHT_BASE_TO_LEFT_BASE T_R` 的 xyz 与旋转矩阵前两行（9 维），
与训练 `eef_local_pair` 转换一致。AUX 仅参与 state 条件输入，不加入动作，也不
参与目标解码；两臂目标按各自 `T_anchor @ ΔT` 还原后直接交给固件 IK。
该组无需 S4 的 episode-frame 参数或 S5 的虚拟 base 文件。

服务端按数据集保留 EEF/AUX 描述，并在调用模型前拒绝维度不符或含 NaN/Inf 的
state。checkpoint 中的统计必须对应 23 维 state 和 20 维 action。

2026-09-14 已用上面 `checkpoint-15000` 在单张 A800 80GB 上完成真实模型冒烟测试：
选择留出 episode 36 / 638 / 1240 的第 100 / 300 / 600 帧，从对应真实关节重建
23 维观测并读取三路视频图像，经 Flask `/v1/infer` 处理函数执行完整预处理、模型推理、
反归一化和部署解码。三次均返回 HTTP 200、`(50,20)` 动作及有限的 `(50,14)` 执行目标；
部署与离线解码的位姿矩阵最大差 `<3e-15`。

真实测试还修复了默认后端 suffix CUDA Graph 的时间条件参数不匹配，以及动作投影和
累积状态与 eager 分支精度不一致的问题。修复后捕获 2 个 profile、0 次回退；同一
输入及 seed 的回放与 eager 动作最大绝对差约 `0.00195`（混合单位的 action 分量，
不是米制误差）。该次回放/普通推理耗时约 0.805 / 0.952 秒，仅为本次环境下的观测值。

本地测试产物保存在 `results/piper_s2_pair_smoke/`：`report.json`、逐样本动作/命令、
运行日志和可复现的 `run.py`。在同一工作区可用下列命令重新运行：

```bash
NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python results/piper_s2_pair_smoke/run.py
```

这些是部署链路与数值一致性测试，只有 3 个留出观测，不构成完整任务评测；未下发真机指令。
另有 109 项 Piper 相关回归和 4 项 graph/eager 精度回归通过。

## S4 部署：每个 rollout 固定一个坐标系

云端加载 S4 checkpoint 及其独立归一化统计；服务自动采用 14 维轴角 state 和
20 维 body-frame action，并拒绝其他 rung 的客户端：

```bash
script/dm05_launcher.sh --exp playground/dm05_piper.py --task inference \
  --data-config.dataset-name piper_fold_s4 --data-config.relative-mode se3 \
  --model-config.model-name-or-path user_checkpoints/piper_s4/checkpoint-60000 \
  --inference-config.port 7891

# 在机械臂机器上先验证一次观测、推理和解码
uv run --no-sync python script/piper_rollout.py \
  --server http://CLOUD_IP:7891 --rung s4 \
  --s4-episode-id cloth-001 --s4-seed 0 --s4-xy-range 0.5 --dry-run

# 使用相同参数复现同一个 H，按已有 SDK 路径执行
uv run --no-sync python script/piper_rollout.py \
  --server http://CLOUD_IP:7891 --rung s4 \
  --s4-episode-id cloth-001 --s4-seed 0 --s4-xy-range 0.5 --execute --speed 10
```

`--s4-episode-id` 不传时，每次进程启动生成新的 UUID；它与 `--s4-seed`（默认 0）
共同确定 H。采样复用训练的 `sample_episode_frame`：`C=I`、+z 向上、
`yaw~U[-π,π)`、x/y 各 `U[-range,range]`、z 平移为 0。`--s4-xy-range`
默认 0.5 m，若训练改过该范围，部署也应传入对应值。启动日志的 `s4` JSON
记录 episode ID、seed、范围和完整 W→G 矩阵；保存日志即可复现。

一个进程对应一个 rollout episode，H 在打开硬件前确定，所有观测、HTTP 重试及
chunk 均复用该 H。开始新 episode 时重新启动进程并使用新的 ID；旋转解缠历史
也随新的 `Representation` 重置。需要复现训练某条 episode 的 H 时，ID 使用
`episode_frames.json` 中的完整键（例如 `episode_000123.jsonl`），seed 和范围
使用该数据的 `manifest.json` 配置。

观测换系为 `T_L^G = H FK(q_L)`、`T_R^G = H T_RIGHT_BASE_TO_LEFT_BASE FK(q_R)`。
模型只接收 G 中的 state；每个 chunk 都以发出请求时的原始 state 为锚点，
按 `T_target^G = T_anchor^G ΔT` 解码，再左乘 H⁻¹ 回 W，右臂最后乘自身 base
外参的逆。夹爪始终保留绝对宽度。H 由客户端管理，云端无需加载
`episode_frames.json`，也不需要为不同 H 重启服务。

协议、训练编码到执行位姿的换系、跨请求轴角解缠和模拟 HTTP dry-run 已有测试；
尚未验证 S4 checkpoint 的实际任务成功率或真实 CAN/相机/固件执行。

## RTC（Real-Time Chunking，仅部署侧）

云端 `/v1/infer` 是无状态的一次性 Flask 接口，不支持增量/引导式解码，因此
这里的 RTC 是**仅在客户端实现**的近似方案（参考
[lerobot RTC](https://huggingface.co/docs/lerobot/rtc) 与本仓库
`agilex_ros_infer.py` 的 `StreamActionBuffer`），不修改云端推理服务：

- 后台线程持续请求新的 action chunk；主循环以 30 Hz 从 `RtcActionBuffer`
  中取出下一步指令执行，两者并行，请求延迟不再阻塞控制节奏。
- 新 chunk 到达时，按请求耗时换算成步数，丢弃其中已经被"执行掉"的前缀
  （`--rtc-latency-k` 设置最多丢弃的步数），并将剩余部分与仍在排队的旧
  chunk 做时间维度的交叉淡化（`--rtc-smooth-method temporal`，
  `--rtc-smooth-weight` 控制旧 chunk 权重，`--rtc-min-smooth-steps`
  设置最短淡化窗口），避免旧新 chunk 拼接处的动作跳变。`--rtc-smooth-method
  raw` 关闭淡化，丢弃后直接替换。
- `--rtc-wait-steps` 控制排队步数降到多少时提前发起下一次推理请求，从而
  让新 chunk 尽量在旧 chunk 耗尽前送达。
- 这是近似实现，不是 lerobot/pi0 论文里依赖服务端引导去噪的"真"RTC——服务端
  是不可修改的黑盒，因此客户端只能做丢前缀+淡化，不能做引导式重新采样。

## 已移除的检查

早期版本在客户端维护了一套步长上限检查（关节差/位置差/旋转差/夹爪差超限
即拒绝整条待发前缀）和到点后立即打电子急停的收尾逻辑。这两者都已移除：

- 模型推理误差总会略微超出这些人为设定的步长阈值，但机械臂不会因为这个
  幅度的误差损坏，超限拒绝整条前缀只会让 rollout 无谓中断；真正的关节
  限位改由 SDK/固件在 `set_joint_limits_enabled(True)` 下逐步检查执行。
- 电子急停会让机械臂带阻尼缓慢下垂，且之后需要 `reset()` 才能继续，这是
  不必要的额外操作；`move_j`/`move_p`/`move_js` 停止下发新目标后本身就会
  保持在最后位置。真正的紧急情况应使用实体急停。

S5 必须传训练时的虚拟 base 文件：

```bash
uv run --no-sync python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s5 \
  --cameras "$HEAD_SERIAL" "$LEFT_SERIAL" "$RIGHT_SERIAL" --bases data/piper_fold_cloth/estimated_bases.json
```

S5 在真实 FK → world → 虚拟 base 的位姿上求 IK 构造观测；预测的虚拟关节经
FK → world → 真实 base 后用固件 IK 执行。不能直接下发虚拟关节。
观测 IK 残差超过 3 mm / 0.03 rad 时停止，因此文档中已知的 S5 不可达帧
可能造成真实 rollout 中断，统计实验结果时需记录。

## 时序和停止条件

后台推理线程与 30 Hz 控制主循环并行运行（见上文 RTC 一节）。
每路 RealSense 使用独立 pipeline 和后台线程持续取帧，只缓存最新的彩色图像。
默认丢弃前 15 帧进行曝光预热，`--camera-warmup-frames` 可修改；
`--camera-startup-timeout` 默认 10 秒，限制 pipeline 启动后等待有效预热帧的时间。
三路就绪后才连接机械臂，使能前最多等待 5 秒，确认双臂关节与夹爪反馈时间戳均更新，
再检查一次图像新鲜度。之后每次后台推理请求都会重新检查反馈与图像新鲜度
（复用 `infer_once` 的观测步骤），SDK 断流、图像过期或格式异常会让当次推理
失败并被记录（不会中止整个 rollout，见下）。退出时停止三路 pipeline。
BGR8 直接编码 JPEG，云端正常解码为 RGB，客户端不要额外交换颜色通道。
图像年龄使用本机单调时钟记录的收帧时间，重复帧号不会刷新年龄；三路是独立彩色流，
不声称具备硬件同步，跨相机时钟同步或精确曝光对齐需要额外实现。

`--timeout` 默认 30 秒（可设置 0–120 秒，不含 0），既是 HTTP 超时也是观测新鲜度
上限，直接交给 `requests.Session(timeout=...)` 处理，不再额外维护独立的
watchdog 线程。`--feedback-age` 默认 0.5 秒，检测反馈时间戳停止变化和相机帧过期。
单次推理失败（超时、契约不符、NaN、相机/反馈异常）只会打印一条日志并retry，
不会让整个 rollout 退出——`RtcActionBuffer` 在没有新 chunk 时保持发送最后一个
指令，机械臂原地等待，等下一次推理成功后自然衔接，不需要人为中止。
`--dry-run` 是例外：它只做一次同步推理，任何异常都会直接抛出并让脚本退出，
方便联调时快速定位问题。

这些逻辑不包含双臂碰撞、桌面碰撞或任务空间障碍检测，也不能替代实体急停。
代码测试使用仿真反馈和 HTTP 测试客户端；实际 CAN、相机同步、固件动作和
checkpoint 的成功率需在设备上验证。

## ROS 部署（复用 web_console_bundle 的现有节点，不直连 CAN/RealSense）

`script/piper_ros_rollout.py` 是上面 SDK 直连方案（`piper_rollout.py`）的替代实现，
用于机架上已经跑着 `web_console_bundle` 遥操作容器（roscore + 双臂驱动 + 三路
RealSense 均已启动）的场景：不再单独打开 CAN 口或连接相机，观测和控制全部走
容器已经发布/订阅的 ROS topic，`/v1/infer` 协议、`Representation`、按 `--rung`
解码这些和 SDK 版本完全一致，只是 IO 层换了。

前提：容器已用 `bash run.sh start` 启动（例如 `xiaoyu` 用户下的
`web_console_bundle`），`rostopic list` 能看到：

- 观测：`/camera_f|l|r/color/image_raw`（`sensor_msgs/Image`，`rgb8`）、
  `/puppet/joint_left|right`（`sensor_msgs/JointState`，7 维 = 6 关节 + 夹爪宽度，米）。
- 控制：`/master/joint_left|right`（`JointState`，仅 `--rung s0` 使用）、
  `/puppet/pos_cmd_left|right`（`piper_msgs/PosCmd`，其余 rung 使用）。

### 在 opendm 的 venv 里跑 rospy

系统 ROS 的 Python（3.8）无法 `import opendm`（本仓库的类型标注要求 Python ≥3.9），
但 `rospy`/`sensor_msgs`/`rospkg` 都是纯 Python，没有编译依赖，因此脚本改为在
opendm 自己的 Python 3.10 venv 里把这几个目录**追加**（不是前插）到 `sys.path`：

```python
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")  # rospy, sensor_msgs
sys.path.append("/usr/lib/python3/dist-packages")              # rospkg
sys.path.append("/home/wzy/web_console_bundle-master/dagger_agilex/devel/lib/python3/dist-packages")  # piper_msgs
```

顺序很关键：追加而不是前插，才能保证 venv 自带的 numpy/opendm 优先于系统里更旧的
版本被解析，否则 numpy 的 C 扩展 ABI 会不匹配。`piper_msgs`（`PosCmd` 消息）没有
apt 包，是 catkin 生成的绑定，来自本机已经 `catkin_make` 过的 `web_console_bundle`
checkout；换一台部署机器时如果路径不同，需要改这一行，或者确认该机器上确实存在
对应的 `devel/lib/python3/dist-packages/piper_msgs` 目录（内容和消息定义只要跟容器
里跑的驱动一致即可，不需要用同一份 build 出的文件，ROS 按消息类型名 + md5 校验，
不校验 build 来源）。

### 为什么不直接用 SDK 的 `move_p`

容器里的驱动节点已经通过 SocketCAN 持有 `can_l_slave`/`can_r_slave`；SocketCAN
允许多个进程同时 bind 同一个接口，但 SDK 和 ROS 驱动同时下发指令会互相打架
（两边各自维护 motion mode/使能状态，不是只读监听）。要用 SDK 就必须先停掉驱动的
`piper_slave_left`/`piper_slave_right` 节点，退回 SDK 直连方案；否则应该用驱动已经
订阅好的 ROS topic 代替 SDK 调用。

### 控制路径：`s0` 走关节，其余 rung 走 `PosCmd`

`/master/joint_*` 对应驱动里的 MIT/力控跟随（`move_mit`），只有一个弱低通
（`0.3*旧+0.7*新`）和默认关闭的 jerk clamp，**没有真正的限速**——直接在这条通道上
按 30 Hz 发布原始关节目标会出现动作幅度大、速度快、抓取前就已经移开的问题。这条
通道只用于 `--rung s0`（关节空间，没有 Cartesian 含义，也就没有 `PosCmd` 可用）。

其余 rung（`s1`/`s2`/`s3`/`s3a`/`s5`）解码出来本来就是每臂 `[x, y, z, roll, pitch,
yaw, gripper]`（米、弧度），直接发布为 `piper_msgs/PosCmd` 到 `/puppet/pos_cmd_left|right`
——这条 topic 目前空闲（默认无发布者），驱动侧 `pos_callback` 收到后做：

```python
self.piper.set_motion_mode('p')
self.piper.set_speed_percent(50)       # 固件侧限速，不是客户端限速
self.piper.move_p([x, y, z, roll, pitch, yaw])
self.end_effector.move_gripper_m(value=max(0.0, gripper) - 0.002, force=1.0)
```

也就是驱动自己做 IK + 固件轨迹规划 + 夹爪，一条消息同时带姿态和夹爪；单位与
`move_p` 的 SDK 约定（米、弧度）完全一致，不需要客户端再做单位换算。`mode1`/
`mode2` 只在驱动里被打日志，不影响行为，脚本固定发 `0`/`0`。因此这条路径**不再需要
客户端自己做 IK**（不用 `opendm.kinematics.piper.ik_track`），也就不存在 IK 不收敛/
分支跳变需要拒绝整条前缀的问题——交给固件处理。

### 用法

```bash
export ROS_MASTER_URI=http://机械臂机器IP:11311   # 或用 --ros-master-uri 覆盖

# 一次推理，打印解码结果，不发布
.venv/bin/python3 script/piper_ros_rollout.py --server http://CLOUD_IP:6666 --rung s2 --dry-run

# 整段执行完再等下一次推理结果（lerobot 的 sync 模式，无重叠）
.venv/bin/python3 script/piper_ros_rollout.py --server http://CLOUD_IP:6666 --rung s2 \
  --rtc-mode sync --execute

# 后台推理 + 延迟感知拼接混合（默认模式）
.venv/bin/python3 script/piper_ros_rollout.py --server http://CLOUD_IP:6666 --rung s2 \
  --rtc-mode chunked --execute
```

### RTC：两种模式，对应 lerobot 的 `sync`/`async` 语义

云端 `/v1/infer` 无状态、一次性返回，不支持增量或引导式解码，这里的 RTC 和 SDK
版本一样是纯客户端近似（参考 [lerobot RTC](https://huggingface.co/docs/lerobot/rtc)
与 [Real robot smoothness 博客](https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)
的术语，不是依赖模型去噪采样器逐步引导的"真"RTC——当前 checkpoint 用 FAST 自回归
后端，没有 flow-matching 采样器可以介入，服务端也不可改）：

- `--rtc-mode sync`：执行完整个 chunk 再阻塞等下一次推理，时序上永远连贯，
  代价是 chunk 之间有一段停顿（几百毫秒到几秒，取决于服务端延迟）。
- `--rtc-mode chunked`（默认）：后台线程持续推理，主循环以 `--feedback-age`/
  `rep.spec["fps"]` 对应的节奏从 `RtcActionBuffer` 取下一步执行，新 chunk 到达时
  与仍排队的旧 chunk 做拼接+淡化：
  - 触发时机由**延迟自适应的 margin**决定（`rtc_trigger_margin`）：对最近几次
    推理耗时做 EMA（`LATENCY_EMA_ALPHA=0.3`），乘安全系数 `1.3` 后取整，再夹在
    `--rtc-min-margin`（默认 4）与 `--rtc-max-margin`（默认 45）之间——不用固定
    小常数，是因为固定值一旦小于真实往返延迟（远程 GPU 常见 500ms~1s，相当于
    15~30 步 @30fps）就会导致缓冲区耗尽、机械臂卡顿。
  - 新 chunk 到达后，按实际耗时换算成步数丢弃其中已经"过期"的前缀，最多丢弃
    `--rtc-latency-k`（默认 12，对应 lerobot 文档建议的 8-12 步 execution
    horizon）步——故意保持较小，避免一次网络抖动直接跳过抓取/接近这种精细阶段。
  - 剩余部分与旧 chunk 排队中的尾部做交叉淡化：`--rtc-smooth-curve`（默认
    `exp`，lerobot 推荐的默认衰减形状；`linear`；`raw` 关闭淡化直接替换），
    淡化窗口至少 `--rtc-min-smooth-steps`（默认 8）步，`--rtc-smooth-weight`
    控制旧 chunk 权重（默认 1.0，即完全按曲线走；调小则整体偏向新 chunk）。

调过 SDK 版本参数的话注意：ROS 版本把"何时触发下一次推理"（margin，延迟自适应）
和"新 chunk 里丢多少步当作过期"（`--rtc-latency-k`，独立的小上限）解耦了，不要
用同一个值控制两者——耦合在一起会导致 margin 被强行压到很小，缓冲区反复耗尽。

### 已知限制 / 验证方式

- `--rtc-mode sync` 是本次新增的"不开 RTC"选项：完全对应用户要的
  "执行完所有动作后，等待推理结果再继续执行"，没有拼接/淡化，也没有后台线程。
- 以下为 ROS 客户端恢复后的验证方案；当前空脚本尚不能执行。验证方式是 `--dry-run`（一次同步推理，任何异常
  直接抛出退出）→ `--rtc-mode chunked` 不带 `--execute` 跑一段观察 `remain`/
  `latency_ms` 日志是否健康（`remain` 不应长期贴近 0）→ 有人看护、可用实体急停
  的条件下加 `--max-steps` 限制、带 `--execute` 短测。
- 夹爪、关节限位、双臂/桌面碰撞检测均由驱动/固件负责，脚本本身不做限幅之外的
  安全检查——和 SDK 版本"已移除的检查"一节是同样的设计取舍。

## 与参考脚本的区别及无设备验证

参考 `test.py` 的 SDK 连接方式，扩展为左右两臂；状态与命令始终按
`[左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]` 排列。图像键 `1/2/3`
对应头部／左腕／右腕。当前训练夹爪使用真实宽度（米），不能复制参考脚本的
0/1 夹爪状态或用最后一次指令冒充反馈。启动不会回零或自动张开夹爪。

服务使用本仓库的 `/v1/infer` 相对动作协议，不是参考脚本的
`/process_frame` multipart 协议。`--server` 接受基础 URL 或完整 `/v1/infer` URL。
默认只观测，也可显式传 `--dry-run`；它仍需要相机、CAN 和推理服务，
与 `--execute` 互斥。无设备机器只运行以下帮助和模拟测试，不运行 rollout：

```bash
uv run --no-sync python script/piper_rollout.py --help
uv run --no-sync python -m pytest tests/test_piper_deploy.py tests/test_piper_realsense.py -q
```

测试环境需安装 pytest；其中服务协议测试还需要云端训练依赖。
仅有本地最小依赖时可加 `-k "not server_contract and not matches_training"`，
跳过需要训练环境的服务配置与训练编码器测试。
