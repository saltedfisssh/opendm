# Piper 云端推理与本地真机部署

配合 [表征阶梯实验](piper_representation_ablation.md) 使用。云端加载 checkpoint，
本地采集 Head / Left wrist / Right wrist 三路相机和双臂反馈，通过
`http://IP:端口/v1/infer` 请求 action chunk，再由 `pyAgxArm` 控制双臂。
本地无需 GPU、Torch 或模型权重。

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
| S3 | piper_fold_s3 | se3 | 23 | 20 |
| S3a | piper_fold_s3a | se3 | 14 | 20 |
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
不启用深度或红外；设备不支持该模式时直接报错，不自动替换相机或分辨率。应使用相同机架、相机视角和夹爪；S3/S3a 使用数据约定的
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
  `--rung s0`。EEF 档位（S1/S2/S3/S3a/S5）没有对应的 Cartesian pass-through
  接口，因此其余档位始终使用 `move_p`，`--motion-mode mit` 会在启动时报错拒绝。

## 表征与执行约定

- 所有观测由关节 FK 重算 flange，TCP offset 为 0，与训练转换一致；EEF state
  使用轴角并跨请求解缠，不依赖 SDK 的异步末端反馈。矩阵链避免四元数符号跳变。
- S0：关节相对值加当前关节，按 `--motion-mode` 调用 `move_j` 或 `move_js`。
- S1：位置相加，旋转按 `R_delta @ R_anchor` 合成。
- S2/S3/S3a：按 `T_anchor @ T_delta` 解码 6D body-frame 旋转；S3/S3a
  把右臂目标转回其自身 base。S3 额外构造 9 维双爪相对特征。
- EEF 目标调用 `move_p`，使用固件 IK。SDK 高层输入是**米、弧度**，
  `move_gripper_m` 是米和牛顿；不再乘 CAN 单位倍率。`se3.mat_to_rpy`
  每次都对旋转矩阵重新分解，返回的欧拉角本身就是 canonical 范围，
  不存在需要客户端检测或拒绝的跨 ±π 分支问题。
- 夹爪宽度裁剪到 `[0, 0.08]` m 后再下发，而不是拒绝整条指令：推理误差
  略微超出夹爪物理行程不会损坏硬件，裁剪即可。关节限位交给 SDK/固件
  （`set_joint_limits_enabled(True)`），不在客户端重复实现会拒绝整条前缀
  的步长检查。

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
仅有本地最小依赖时可加 `-k "not server_contract"` 跳过服务端测试。
