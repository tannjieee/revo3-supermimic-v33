# Isaac Sim 6 迁移与运行

> 此文保留旧 Sim 6.0 / beta PhysX 基线的安装或测量记录。当前 v3.0.0 Newton 安装、默认值和启动命令以 [Newton 迁移说明](newton_migration.md) 为准。

本项目以 **Isaac Sim 6.0.1（6.0 系列）+ Isaac Lab v3.0.0-beta2.patch1** 为当前运行基线。
默认任务为 Revo3 右手，原 Allegro 任务仍可显式选择。5.1 专用 Storm 显示副本和动作排序补丁已移除。

## 环境基线

先依据 [Isaac Lab 的版本化安装说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html)
准备独立的 Python 3.12 环境、Isaac Sim 和 NVIDIA GPU 驱动，再将 Isaac Lab 固定到下表提交。
本仓库不安装驱动，也不包含完整的 Isaac Sim / Isaac Lab 运行环境。

| 组件 | 已记录的运行版本 |
| --- | --- |
| Python / Isaac Sim | 3.12 / 6.0.1 |
| Isaac Lab | v3.0.0-beta2.patch1，`ffff603eafc6b74264a5261cc0183d6a65390d78` |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| RSL-RL / Gymnasium / NumPy | 5.0.1 / 1.2.1 / 2.3.1 |
| gsplat / Viser | 1.5.3 / 0.2.11 |
| 历史验证硬件 | RTX 5090，驱动 595.99.02 |

版本约束保存在 `requirements/sim6-constraints.txt`。`run_sim6.sh` 支持
`VISERDEX_PYTHON=/path/to/python`，也可查找项目内 `.venv-sim6` 或相邻的 `viserdex_sim6`。
使用其他路径时显式指定解释器和 `CUDA_HOME`，所有命令从仓库根目录运行。

```bash
export OMNI_KIT_ACCEPT_EULA=YES
# 首次训练前，按 README 下载所选物体数据。
bash scripts/run_sim6.sh scripts/rsl_rl/train.py \
  --headless --device cuda:0 --num_envs 64 --max_iterations 5 \
  --run_name my_revo3_run --logger tensorboard
bash scripts/run_sim6.sh scripts/rsl_rl/play.py \
  --headless --device cuda:0 --num_envs 1 --num_steps 120 \
  --load_run my_revo3_run --checkpoint model_4.pt --video --video_length 120
```

策略保存在 `logs/rsl_rl/viserdex_revo3_sim6_ppo/<run>/`。短程验证用的 run 是 `sim6_validation`。
长训练去掉 `--max_iterations 5` 或设置实际需要的轮数。

## 依赖安装与已知元数据冲突

激活已准备好的仿真环境后，安装本项目：

```bash
python -m pip install -c requirements/sim6-constraints.txt -e .
```

Lab 此预发布版本的依赖声明有两处与 Sim 冲突：`isaaclab` 要求 coverage 7.6.1，Sim kernel 要求 7.4.4；
`isaaclab-rl` 要求 packaging <24，Sim core 要求 26.0。历史验证环境选择 Sim 的 7.4.4 和 26.0。
Lab 源码未修改，安装对应可编辑包时使用 `--no-deps`，其余运行依赖单独安装；因此原始 `pip check` 仍报告这两项。
不要对全部依赖执行无约束升级，也不要把 `--no-deps` 当作新环境已装齐依赖的证明。
Viser 固定为 0.2.11，以兼容 Sim 的 websockets 12；不再安装依赖旧 Lightning 的 lightning-bolts。

CUDA 编译器必须匹配 Torch 的 CUDA 13。可建立独立工具链（只安装开发包，不安装驱动）：

```bash
conda create -y -p /path/to/cuda13 --override-channels -c conda-forge \
  cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-libraries-dev=13.0 cuda-version=13.0
export CUDA_HOME=/path/to/cuda13
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include"
export PATH="$CUDA_HOME/bin:$PATH"
```

## Docker

Docker 配方使用官方 `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`，固定 Lab 源码和 Torch/CUDA 版本，
移除了 5.1 的 Python 3.11 / CUDA 12.8 路径。容器构建尚未验证。

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build viserdex
docker compose --env-file .env.base --file docker-compose.yaml up -d
docker exec --interactive --tty -e DISPLAY="${DISPLAY}" viserdex /bin/bash
```

容器内的 `python` 使用 Isaac Sim 的启动包装器。GUI 运行需要主机提供可用的 X11 会话和访问权限。

## 接口变化

- Lab 3 四元数采用 **XYZW**：任务初始姿态、物体、相机外参、位姿差和预处理脚本已统一。
  3DGS PLY 文件和 gsplat API 自身仍采用 WXYZ，这个文件接口保留原格式。
- 资产和相机数据改用 `ProxyArray.torch`，获取对应 Torch 零复制视图。
- PhysX 配置使用 `isaaclab_physx`，仿真配置字段从 `physx` 改为 `physics`。
- Gaussian Splat 相机共享 Lab 的 PhysX Warp simulation view；相机缓冲区使用新的 CameraData 分配接口，
  传感器更新/重置适配 Warp 布尔掩码，保留按环境更新。
- 预处理相机显式设置 XYZW 世界姿态，再将 GS 相机变换到物体坐标系；刚体瞬移后先完成一次物理更新，
  将状态发布到 Fabric，避免 RTX 仍显示旧姿态。深度比较仅用有限前景值确定色阶。
- RSL-RL 5 使用独立 actor/critic 配置、GaussianDistribution 和 `actor` 观测集合；
  通过 Lab 官方转换函数清理弃用字段。策略输入仍为 87，输出为 21。
- 所有姿态估计脚本支持 TensorBoard，默认不需要云账号；训练末尾保存 CNN 检查点。
  评估与真实数据测试统一读取 `logs/rsl_rl/pose_estimator/<run>/pose_estimator/`。
- CNN 学习率调度由 PyTorch 原生实现：500 次实际更新 warmup，200 次 cosine 衰减到 2e-5 后保持，
  避免原配置 warmup 500 / max_epochs 200 的无效时间顺序。初始化的前 6 次观测不推进调度器。

Sim 5.1 的检查点不适用于新的四元数观测和 RSL-RL 格式；必须重新训练，不要直接传入旧检查点。

## 视觉流程

历史验证使用 RSLCube 的 USD、网格、splat、关键点和聚类索引，数据来自
[ViserDexSplats](https://huggingface.co/datasets/leggedrobotics/ViserDexSplats/tree/ffe209a34c7c56516507480e91fb96ced3dbe9f8)。
所有物体数据均需通过 `scripts/download_data.py` 下载。首次运行 CNN 会下载 torchvision ResNet34 权重。

```bash
bash scripts/run_sim6.sh scripts/pose_estimator/train_pose_estimator.py \
  --headless --device cuda:0 --num_envs 4 --num_steps 12 \
  --policy_run sim6_validation --checkpoint model_4.pt \
  --run_name my_pose_run --logger tensorboard
bash scripts/run_sim6.sh scripts/pose_estimator/eval_pose_estimator.py \
  --headless --enable_cameras --device cuda:0 --num_envs 2 --play_time 1 \
  --policy_run sim6_validation --checkpoint model_4.pt --load_run my_pose_run --logger tensorboard
```

Revo3 21 关节顺序和观测布局见 [Revo3 移植说明](revo3_migration.md)。
原项目的 Allegro 遮挡查表不适用 Revo3；现已接入 Revo3 动态网格深度遮挡，见
[位姿训练说明](revo3_pose_training.md)。真机相机标定和真实数据评估仍需单独完成。

## 验证记录（2026-10-13）

日志目录：`logs/revo3_validation/20261013/sim6/`。

| 检查 | 结果 |
| --- | --- |
| 本地资产/关节/限位/校验清单 | 15 项通过 |
| 原生学习率调度边界与停止衰减 | 1 项通过 |
| 真实 RSLCube，4 环境 / 120 步 | `smoke_03.log` 通过；动作 21，policy 49，privileged 38 |
| PPO，64 环境 / 5 轮 | `train_02.log` 通过；7680 步，新格式检查点已保存 |
| RTX 检查点回放，1 环境 / 120 步 | `play_03.log` 通过；1280×720、60 fps 视频，无黑帧，实看 Revo3 和方块正常 |
| RTX 训练录像，8 环境 / 1 轮 | `train_video_01.log` 通过；24 帧完整解码，无黑帧 |
| GS 图像与在线 CNN 训练，4 环境 / 12 步 | `pose_03.log` 通过；最终 `cnn_16_final.pth`，loss 有限且执行了反向更新 |
| GS / NoAug / RTX Camera / Eval 图像测试 | 四种任务均通过；RGB 有内容、有效前景深度、观测有限、投影反投影一致 |
| GS 部分环境重置 | 原生 Warp 掩码重置后，其他环境图像与帧计数保持不变 |
| RTX 与 GS 联合评估入口，2 环境 / 1 秒 | `eval_01.log` 通过；加载 CNN 后只推理，指标正常写入本地日志 |
| 相机预处理对照，2 个视点 | `preprocess_05.log` 通过；检查 RGB/深度图，RTX 与 GS 方块姿态一致 |
| 无效检查点负例 | `bad_checkpoint.log` 明确抛出 FileNotFoundError，退出码 1 |

视觉测试结果和样张：`vision_gs/`、`vision_NoAug/`、`vision_Camera/`、`vision_Eval/` 下的
`result.json` 和相机 PNG；`pose_metrics.json` 保存 loss/LR/关键点指标，`versions.json` 保存实际运行版本。
`revo3_rtx.mp4` 是 RTX 回放视频；`revo3_rtx.json` 记录完整解码 120 帧、无黑帧和帧间运动差异。

复现视觉检查：

```bash
bash scripts/run_sim6.sh scripts/tests/smoke_vision.py --headless --device cuda:0 \
  --num_envs 2 --steps 4 --pose_run logs/rsl_rl/pose_estimator/sim6_pose_validation \
  --output_dir logs/vision_check
# --task 可分别指定 Estimator-NoAug、Estimator-Camera、Estimator-Eval 变体。
```

以上短程检查用于确认迁移后的运行链路，不代表策略或位姿估计模型已经收敛。
真实采集数据集评估和 Docker 构建未执行。后续已用 Revo3 动态网格深度替代手部遮挡查表；
真机相机标定仍待完成。
