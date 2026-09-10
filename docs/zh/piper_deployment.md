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

将本仓库和 `third_party/pyAgxArm` 目录复制到机械臂机器，创建独立环境：

```bash
python3 -m venv .venv-piper
source .venv-piper/bin/activate
pip install numpy requests opencv-python
pip install -e third_party/pyAgxArm
```

SDK 目录需完整存在；当前它是工作区中已有的第三方目录。
从仓库根目录运行脚本，不需要 `pip install -e .` 安装云端训练依赖。

按实际设备配置 SocketCAN，例：

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

`can0` 是左从臂、`can1` 是右从臂，不能接主臂。
相机使用 V4L2 设备编号或 `/dev/v4l/by-id/...` 稳定路径，顺序必须与训练一致，
分辨率要求 640×480。应使用相同机架、相机视角和夹爪；S3/S3a 使用数据约定的
右 base 相对左 base `[0, -0.60, 0]`。不能在换了外参的机架上直接复用。
用 `--firmware default|v183|v188|v189` 选择与机械臂实际固件对应的 SDK profile。

先运行一次只观测与解码（不会使能或下发运动）：

```bash
python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s2 \
  --left-can can0 --right-can can1 --cameras 0 2 4 --cycles 1
```

输出 JSON 包含请求耗时和解码后的执行前缀。确认相机顺序、反馈、动作方向后，
在有人看护且可使用实体急停的条件下执行：

```bash
python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s2 \
  --left-can can0 --right-can can1 --cameras 0 2 4 \
  --execute --speed 10 --steps 5 --cycles 0
```

`--cycles 0` 持续运行，Ctrl-C/SIGTERM 停止；正常完成和异常退出也会发送双臂电子急停。
重新运行前按 SDK/设备操作规范恢复急停；脚本不会自动 reset 或回零。
停止时保留支撑力，不自动 disable。电子急停的实际行为须在所用固件上验证。

## 表征与执行约定

- 所有观测由关节 FK 重算 flange，TCP offset 为 0，与训练转换一致；EEF state
  使用轴角并跨请求解缠，不依赖 SDK 的异步末端反馈。矩阵链避免四元数符号跳变。
- S0：关节相对值加当前关节，调用 `move_j`。
- S1：位置相加，旋转按 `R_delta @ R_anchor` 合成。
- S2/S3/S3a：按 `T_anchor @ T_delta` 解码 6D body-frame 旋转；S3/S3a
  把右臂目标转回其自身 base。S3 额外构造 9 维双爪相对特征。
- EEF 目标调用 `move_p`，使用固件 IK。SDK 高层输入是**米、弧度**，
  `move_gripper_m` 是米和牛顿；不再乘 CAN 单位倍率。
- SDK `move_p` 仅接受 canonical Euler 范围。客户端检测跨 ±π 的欧拉分支，
  直接停止，避免把不满足接口范围的解缠角发给 SDK。这意味着部分跨分支轨迹
  暂不能完整 rollout，应单独记录这类中断。

S5 必须传训练时的虚拟 base 文件：

```bash
python script/piper_rollout.py --server http://CLOUD_IP:7891 --rung s5 \
  --cameras 0 2 4 --bases data/piper_fold_cloth/estimated_bases.json
```

S5 在真实 FK → world → 虚拟 base 的位姿上求 IK 构造观测；预测的虚拟关节经
FK → world → 真实 base 后用固件 IK 执行。不能直接下发虚拟关节。
观测 IK 残差超过 3 mm / 0.03 rad 时停止，因此文档中已知的 S5 不可达帧
可能造成真实 rollout 中断，统计实验结果时需记录。

## 时序和停止条件

采用同步请求、30 Hz 执行前缀、再观测的闭环，默认每次执行 5/50 步。
相机后台持续读取以减少缓存积压。请求期间不继续发送旧 chunk；这会有网络与
推理等待停顿，不能当作 RTC 延迟补偿。对比实验需固定网络、steps 和速度。

`--timeout` 默认 2 秒，同时限制请求与观测到返回结果的年龄；watchdog
独立处理超时和信号。`--feedback-age` 默认 0.5 秒，检测反馈时间戳停止变化和
相机帧过期。完整执行前缀在发送前校验，执行时再对当前反馈校验。
默认单步最大关节差 0.15 rad、位置差 25 mm、旋转差 0.15 rad、夹爪差 20 mm，
夹爪范围 0–80 mm；超限拒绝整个前缀，不静默裁剪。SDK 继续检查关节限位。

这些检查不包含双臂碰撞、桌面碰撞或任务空间障碍检测，也不能替代实体急停。
代码测试使用仿真反馈和 HTTP 测试客户端；实际 CAN、相机同步、固件动作和
checkpoint 的成功率需在设备上验证。
