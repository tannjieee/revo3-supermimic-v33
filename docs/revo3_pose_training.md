# Revo3 RSLCube 遮挡位姿估计训练

> 当前环境请先按 [Newton 迁移说明](newton_migration.md) 安装与生成资产。本文已有性能或成功率记录来自旧基线，不能视为 Newton 复现结果。

使用同一物体、同一手型且已充分训练的 Revo3 教师（例如 5000 轮的 `model_4999.pt`）生成运动轨迹，采集带物理手部遮挡的
3DGS 图像及真值，再训练图像到关键点的网络。编号从 0 开始，因此最终文件是 4999；
并不存在另一个需要替代它的 `5000.pt`。教师参数、物体 3DGS 均固定。

## 遮挡迁移

依据 [ViserDex III-C2](https://arxiv.org/html/2604.11138v1#S3.SS3.SSS2)，将相机变换到物体
坐标系渲染静态 3DGS，同时从原始相机位姿对运动中的手部网格做深度射线投射。
`D_hand < D_object` 的物体像素置黑，保留正确的物体关键点标签。

- Revo3 使用 21 个关节对应指节及手掌，共 22 个动态视觉网格，逐环境跟踪 PhysX 刚体变换。
- 使用 Isaac Lab 3 / Isaac Sim 6 的 `MultiMeshRayCasterCamera`，不再依赖 Allegro 的离散关节遮挡表。
- 手部相机和 3DGS 相机使用相同的内参、外参、120×120 分辨率；采集启动时断言一致。
- 3DGS 使用 `RGB+ED` 期望米制深度，避免将受不透明度缩放的累积深度误用于物理比较。
- 未命中的射线规范化为正无穷；只有有效、位于物体前面的手部深度产生遮挡。
- Revo3 的 Estimator、Eval、Camera、NoAug 配置均接入动态手部遮挡。

当前保留项目已有相机位姿。数据覆盖的是该仿真视角下实际产生的遮挡；遮挡率分布需要查看
`audit.json`，不能把轻度遮挡训练宣称为重遮挡鲁棒性。真实 Revo3 相机标定和真机验证仍需
单独完成。

## 可复现入口

```bash
bash scripts/train_pose_revo3_occluded.sh \
  logs/rsl_rl/viserdex_revo3_v3_ppo/my_teacher/model_4999.pt my_pose_run
```

该脚本顺序完成采集、数据审计和训练，每一步失败都会中止后续步骤，不接受覆盖已有数据目录。
第一个参数是教师检查点，第二个参数是可选的新 run 名称。脚本不再依赖某次本地实验目录。
默认采集 30 万张训练图像和 3 万张验证图像；运行前请预留存储 RGB、深度和 HDF5 数据所需的磁盘空间。

| 参数 | 配置 |
| --- | --- |
| 训练集 | 300,000 张，256 并行环境，seed 42 |
| 验证集 | 30,000 张，128 并行环境，seed 43 |
| 采样 | 预运行 120 步，每 4 个控制步取一帧；跳过 episode 重置帧 |
| 接受条件 | 遮挡率 5%–95%，至少 32 个可见物体像素 |
| 输入 | 带遮挡 120×120 RGB，网络内 resize 至 224×224 |
| 输出 | 中心和 8 个角点，各为归一化 `(u,v)` 与米制深度，共 27 维 |
| 网络 | 现有 ImageNet ResNet34 特征网络，MLP 512/256，骨干参与训练 |
| 训练 | 10 epochs，batch size 64，约 300 万次图像呈现，46,880 次优化更新 |
| 优化器 | AdamW，峰值学习率 1e-4，默认 weight decay 0.01 |
| 调度 | 1,000 次优化更新 warmup，其后余弦衰减至 2e-5 |
| 损失 | UV 和深度分别计算 Huber(delta=0.05)，权重各 1，整体乘 100 |
| 增强 | 采集时 3DGS 光照/颜色等扰动，训练时 ISO 和 motion blur；binary opening 预处理 |
| 存盘 | 每 500 次优化更新、每 epoch、收到 SIGINT/SIGTERM 时 |
| 评估 | 每 epoch 在独立种子验证集评估关键点、平移、旋转、ADD，并分遮挡率统计 |

30 万/3 万和 10 epochs 是本项目的工程配置，论文未规定这些数量。逐帧同轨迹随机拆分会
导致泄漏，因此这里用独立环境和独立种子生成验证集。

## 数据与检查点

每个数据目录包含：

- `samples.h5`：遮挡前后 RGB、手部/物体深度、27 维标签、物体位姿、21 维关节角、遮挡率、环境/episode/仿真步编号。
- `manifest.json`：教师 SHA256、相机参数、网格和采样配置、完成状态、遮挡直方图。
- `preview.png`：3DGS 原图、Revo3 深度、遮挡像素、实际训练图像。
- `audit.json`：逐样本检查遮挡像素和深度顺序、标签、关节/物体运动变化及遮挡率分布。

训练目录含 `launch.json`、`metrics.jsonl`、`progress.json`、TensorBoard 日志、`completion.json`。
`training_state.pt` 保存模型、优化器、调度器、随机状态和进度；相同参数加 `--resume` 可续训。
`pose_estimator/cnn_<step>_*.pth` 可直接被现有仿真位姿估计器加载；`best.pth` 为验证 ADD 最小的权重。

分别运行或复用已完成数据：

```bash
bash scripts/run_newton.sh scripts/pose_estimator/audit_occlusion_dataset.py <dataset_dir>
bash scripts/run_newton.sh scripts/pose_estimator/train_pose_dataset.py \
  --train_dir <train_dataset_dir> --val_dir <validation_dataset_dir> \
  --output_dir logs/rsl_rl/pose_estimator/<new_unique_run> \
  --epochs 10 --batch_size 64 --learning_rate 1e-4 --warmup_steps 1000 --save_interval 500
```

在线边采集边训练入口仍为 `scripts/train_pose_revo3_5090.sh`，需要显式传入教师 run：

```bash
bash scripts/train_pose_revo3_5090.sh \
  --policy_run my_teacher --checkpoint model_4999.pt --run_name my_online_pose
```

## 历史仿真验证（2026-10-17）

一组已记录的 RSLCube 实验使用 213,280 张训练样本、30,000 张独立种子验证样本。
训练集遮挡率中位数 7.52%、P95 13.84%，验证集没有超过 50% 遮挡的样本。
先训练 10 epochs，再从最佳模型用学习率 1e-5、最低 1e-6 微调 20 epochs。
微调早停条件为连续 5 epochs 未改善超过 1%。

在用于选模的 30,000 帧验证集上重新评估最佳权重，记录结果为：

| 指标 | 结果 |
| --- | ---: |
| 平均 ADD | 0.945831 mm |
| 平均平移误差 | 0.829179 mm |
| 平均旋转误差 | 0.643272° |
| ADD P95 | 2.028000 mm |
| 平移 <10 mm 且旋转 <10° | 99.9467% |

指标用 float64 几何计算，CNN TF32 关闭。该集合用于选模，且缺少重遮挡样本，
结果不能作为独立测试集、真机或重遮挡性能。相关图像、日志和权重不随代码分发。
原相机内外参仍需在真实 Revo3 部署时重新标定。


## 2026-10-23 calibrated Hikrobot pose predictor training

The native calibrated RGB pose task is deployed on server22021 at the same
/home/zhangke/viserDex_revo3 path. The4096-world/30Hz student remains running
on22031 GPU1 with unchanged process2371464 and unchanged student source.
The new collect_hikrobot_pose.py restores frozen teacher4800 environment and
asset hashes before attaching the existing official CameraCfg/NewtonWarpRenderer.
It uses the deliveredDB0447679 intrinsics, distortion and T_base_camera, native
1280x1024 RGB, full-frame rectification and120x120 float32 pixels.
Labels are synchronized center+8corners UVD with optical depth in meters.
The CNN reads RGB only; saved simulator state is exclusively label/audit data.
Native hand occlusion is retained without oracle masking or ground-truth crops.
Unknown occlusion fractions are explicitly-1, not fabricated percentages.

GPU6/7 on22021 collect and GPU3 trains the original ResNet34+512/256head.
Initial8192/2048images use independent seeds42/43,10epochs,AdamW1e-4,batch64.
Concurrently collect131072/16384images with seeds44/45; after completion and
independent audit, initialize a new10-epoch run from initial best.pth with a
fresh optimizer/schedule. Exact plans, memory limits and pipeline failure/exit
records:outputs/hikrobot_pose_training_20261023 on22021.

Validation:57CPU tests pass,1CUDA-only test skipped with CUDA hidden;Ruff passes.
4-world128-image smoke exits0,peak948MiB,all4camera extrinsics within6.62e-8
of calibration. Independent saved-pose reprojection max error8.22e-8.
Native RGB/label previews were visually inspected. Initial datasets completed
and passed audit. CNN optimization and validation run successfully, with finite
model/optimizer checkpoint tensors. This verifies launch, data alignment and
initial learning; it is not real-camera accuracy or closed-loop student proof.
See22021 outputs/revo3_pose_training_active.json for current pipeline phase.
