# Revo3 循环学生蒸馏

> 当前环境请先按 [Newton 迁移说明](newton_migration.md) 安装与生成资产。本文已有性能或成功率记录来自旧基线，不能视为 Newton 复现结果。

对应论文：[ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation](https://arxiv.org/html/2604.11138v1)，
§III-B、附录 B-B、表 XIII–XIV。实现位于 `source/viserdex/viserdex/distillation/`。

本任务在已有 Revo3 教师上训练循环学生，加入感知噪声、在线 DAgger 和状态重建辅助监督。
训练环境沿用 `Isaac-ViserDex-Repose-Revo3`，不需要相机渲染或 CNN 检查点。
视觉位姿估计仍是独立训练阶段；部署时将其输出变换到手部基座 `right_hand_base_link` 坐标系，再送给学生。

## 训练、恢复与评估

按 [环境说明](isaacsim6_migration.md) 配置仿真环境，从仓库根目录运行：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export VISERDEX_OBJECT=RSLCube

bash scripts/distill_revo3_5090.sh \
  --teacher_checkpoint logs/rsl_rl/viserdex_revo3_v3_ppo/my_teacher/model_4999.pt \
  --num_envs 256 --max_iterations 1000 --run_name rslcube_student_01
```

新训练必须显式指定教师文件。加载时校验手型、物体、XYZW 协议和观测布局，
按照教师保存的 `params/agent.yaml` 构建 actor，严格加载 actor 权重及归一化统计。
标签采用冻结教师的确定性动作。训练不会自动切换到后来写出的 PPO 检查点。
现有非旧命名空间教师通过相邻 `params/teacher_protocol.json` 明确声明 XYZW、49/38 维观测、
21 关节顺序及检查点/环境 SHA-256，加载时严格核对，不靠修改实验名绕过协议检查。
启动前使用官方 configclass `from_dict` 恢复教师完整 `params/env.yaml`，再通过官方
`launch_simulation` 启动；只允许环境数、设备、日志、种子和非减小的碰撞容量等操作性差异。
资产、步长、decimation、动作映射、目标、奖励、物理随机化和观测协议按保存的环境校验。
恢复及评估使用学生相邻 `env.yaml` 和 SHA-256，不再取当前基础任务的默认手资产/频率。
蒸馏专用环境先关闭 `policy`、`privileged` 两组自动观测噪声，以保存干净的源状态。
随后只生成一次带噪且带延迟的关节读数，由教师、学生共用；教师的物体位姿、速度等
特权状态保持真值。每次更新分别核验源状态与物理真值一致、教师关节读数与学生一致、
教师特权状态没有受到视觉噪声影响。物理参数随机化保留，辅助重建目标始终干净。

学生日志写入 `logs/distillation/revo3_v3/<run_name>/`：

- `student_<已完成更新次数>.pt`：模型、归一化、AdamW 状态、DAgger 进度、随机数状态、配置与教师 SHA-256。
- `config.json`、`env.yaml`：完整学生参数、观测约定、教师来源及实际环境。
- `metrics.jsonl`、TensorBoard events：BC/重建损失、梯度范数、教师动作占比、奖励与环境指标。
- `progress.json`：最新更新、学生噪声 RMS、干净源状态与仿真真值的最大差值，以及教师/学生关节读数差值。
- `completion.json`：训练是否达到预算、退出原因和最终检查点。第 1 次更新也保存检查点。

目录已存在时拒绝覆盖；恢复时选择新的 run 名称：

```bash
bash scripts/distill_revo3_5090.sh \
  --student_checkpoint logs/distillation/revo3_v3/rslcube_student_01/student_500.pt \
  --max_iterations 1000 --run_name rslcube_student_resume
```

`max_iterations` 表示累计更新上限，上例继续第 501–1000 次更新。
恢复保留学习率、训练参数、优化器、归一化与 beta 衰减进度；环境重新初始化，
循环状态、动作历史及噪声缓存从新回合开始，不恢复模拟器的中途状态。
教师可搬迁，但新路径必须有相邻的 `params/`，且 SHA-256 必须一致。
恢复训练要求 `teacher_observation_protocol=shared_noisy_joints_clean_privileged_v1`。
旧版干净教师检查点不能直接续训为新协议；更换协议时应从头训练。旧版 v2 学生仍可单独评估，
此时保留原本没有关节延迟的设置。10 月 14 日更早的旧噪声格式仍不支持加载。

训练时 `SIGUSR1` 请求在完整更新后保存；`SIGINT` / `SIGTERM` 请求完成当前更新后保存并退出。
不要用 `kill -9` 期待产生最后一次检查点。

纯学生评估不加载教师模型，也不混入教师动作：

```bash
bash scripts/distill_revo3_5090.sh --evaluate \
  --student_checkpoint logs/distillation/revo3_v3/rslcube_student_01/student_1000.pt \
  --num_envs 64 --num_steps 1200 --run_name rslcube_student_eval

# 去掉位姿和关节噪声的消融对照
bash scripts/distill_revo3_5090.sh --evaluate --clean_observations \
  --student_checkpoint logs/distillation/revo3_v3/rslcube_student_01/student_1000.pt \
  --num_envs 64 --num_steps 1200 --run_name rslcube_student_eval_clean

# 冻结教师基线
bash scripts/distill_revo3_5090.sh --evaluate --teacher_only \
  --teacher_checkpoint logs/rsl_rl/viserdex_revo3_v3_ppo/my_teacher/model_4999.pt \
  --num_envs 64 --num_steps 1200 --run_name rslcube_teacher_eval
```

每次评估生成 `evaluation.json`，包含步数、完成回合数、目标成功计数增量、
已结束回合的平均成功次数、平均每步奖励、方向误差与各终止原因。
各终止条件可能同时成立，因此终止计数不是互斥分类。
`--clean_pose` 仅关闭位姿噪声；`--clean_observations` 同时关闭位姿和关节噪声。
评估依赖仿真物体状态生成带噪位姿；这些结果不等同于真实 RGB 闭环或真机评估。

## 当前 4096 环境学生任务（2026-10-22）

按用户要求停止 compact 老师优化，固定其最后已保存并校验有限数值的 `model_4800.pt`。
正式学生任务为 `student_compact4096_30hz_teacher4800_20261022`：4096 环境、30 Hz、
compact116 手、正确的 RSLCube、45 步连续 BPTT、32 epochs、累计 1000 次更新。
第 1 次更新保存检查点，此后每 10 次保存；既有老师不继续优化。
本阶段是带噪位姿 DAgger，不创建相机或 CNN。

4096 环境采用 `--update_env_batch_size 128` 控制反向传播显存。每个环境保持完整的
45 步序列与重置掩码，按环境轴分成若干块；每块损失按环境占比加权并累积梯度，
全部 4096 环境处理后才做一次全局梯度裁剪和 AdamW step。每个 epoch 仍是一批完整数据，
没有环境下采样、缩短序列或把 AdamW step 数放大。归一化统计每次更新仅覆盖一次全体样本。
默认 `update_env_batch_size=None` 保持旧的整批计算行为；恢复保留该配置。

噪声与 BPTT 维持原来的步数约定：30 Hz 下关节延迟上限 1 步约 33.3 ms、视觉更新周期
上限 3 步约 100 ms，BPTT 45 步为 1.5 s。它们是训练扰动，尚非海康跟踪实测误差。
活跃任务指针为服务器 `outputs/revo3_student_training_active.json`，冻结老师及启动证据位于
`outputs/student_dagger_start_20261022/`。读取进度应以正式 run 的 `progress.json`、实际检查点和
监控状态为准，不将短程 smoke 的数字当作正式训练进度。

## 网络与监督

学生默认参数：

| 项目 | 设置 |
| --- | --- |
| 外感知编码 | ELU MLP `[256, 256]` → 64 |
| 循环编码 | 2 层 GRU，隐藏维数 256 |
| 信念状态 | MLP `[256, 256]` → 256 |
| 感知门控 | MLP `[128, 128]` → 64，Sigmoid |
| 控制头 | ELU MLP `[1024, 1024, 512, 512]` → 21 |
| 两个解码头 | 各为 `[256, 256]`，预测干净外感知和特权状态 |
| 优化 | AdamW，学习率 `3e-4`，每批 32 epochs，1 minibatch |
| BPTT | 连续 45 步，回合边界切断隐藏状态与梯度 |
| DAgger | `beta = 0.9 × 0.95^已完成更新次数` |
| 学生采样噪声 | 固定标准差 `0.02`；评估使用确定性输出 |

序列训练将所有时间步的逐帧编码器和输出 MLP 批量计算；GRU 只在存在回合重置的时间点
拆分序列，并仅清除实际重置环境的隐藏状态。无重置区段使用一次完整 GRU 调用，保持 BPTT
梯度和回合边界；在线控制的逐步 `forward` 不变。训练指标先在 GPU 汇总，完成更新后再读回，
避免每个环境分块的 loss/BC/reconstruction 日志触发 CPU–GPU 同步。

每个环境、每一步独立选择整条教师或学生动作向量；无论选择谁，标签都是当前访问状态下的教师动作。
采用在线滚动批次，每个批次保留完整的时间顺序及重置掩码；不维护长期离线重放数据集。

损失为 `MSE(action_student, action_teacher) + 0.2 × L_recon`。
重建目标先做运行均值/方差归一化，`L_recon = 2 × (MSE_extero + 0.2 L1_extero)
+ (MSE_privileged + 0.2 L1_privileged)`。
输入和目标统计每个采样批次更新一次，评估冻结。AdamW weight decay `0.01`、梯度裁剪 `1.0`、
GRU 类型、具体门控连接和归一化方式是本实现补充的工程选择；论文未完整指定这些细节。

## Revo3 观测协议

所有关节与动作采用 `REVO3_JOINT_NAMES` 的显式顺序，四元数采用 **XYZW**。
学生四元数规范为 `w >= 0`，与教师原有观测布局分别处理。

| 输入/监督 | 组成 | 维数 |
| --- | --- | --- |
| 学生本体感知 | 限位归一化关节位置 21、最近 4 条原始动作 84、手部基座坐标目标四元数 4、环境坐标基座位置 3、剩余时间 1 | 113 |
| 学生外感知 | 带噪手部基座坐标物体位置 3、四元数 4、由该带噪四元数计算的目标旋转差 4 | 11 |
| 干净外感知监督 | 手部基座坐标真实位姿与目标差 | 11 |
| 特权监督 | 当前教师的缩放关节速度 21、物体线/角速度 6 | 27 |
| 教师输入/输出 | 原有 policy 49 + privileged 38；原有 21 动作 | 87 / 21 |

动作历史从新到旧排列，保存实际选择并送入环境的原始动作；回合重置清零。
学生动作沿用教师的限位映射和 EMA，不在网络输出端额外加 tanh。
学生推理仅接收本体和外感知两个张量以及显式循环状态；特权监督张量不传入控制头。

关节编码器噪声参考 [NVIDIA DeXtreme ManualDR 官方配置](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/AllegroHandDextremeManualDR.yaml)：

- 对 21 维限位归一化关节位置增加高斯逐帧噪声 `std=0.005`、每回合固定偏差 `std=0.01`；
  教师与学生使用同一个传感器样本，避免重复叠加。
- 每个环境每回合采样固定 0–1 步关节读数延迟（当前 60 Hz 下约 0–16.7 ms）；
  先给测量加噪再通过历史缓存取样，重置时不会读到上一回合的数据。
  此延迟幅值是工程默认值，论文只说明本体感知带有噪声及延迟，没有给出完整范围。
- 单位是归一化关节读数；某关节角度标准差等于上述数值乘以该关节活动范围的一半。
- 借用其噪声幅值；本实现每帧加入白噪声、每回合重采样偏差，与 DeXtreme 的 50% 白噪声触发及定期偏差刷新不同。
- 动作历史、软件目标、时间等不额外注入传感器噪声。回合重置只更新对应环境的偏差。
- 尚无 Revo3 真机误差统计，这些是训练扰动默认值，不是实测传感器精度。

视觉噪声沿用论文表 XIV，按位置、方向分别维护更新周期、延迟概率与跟踪失败概率：

- 每回合采样 1–3 步的更新周期，延迟概率范围 `[0, 0.1]`，失败概率范围 `[0, 0.3]`。
- 平移逐帧噪声和每回合固定偏差均为 `±12 mm`。
- 转动逐帧噪声 `±1°`，每回合固定偏差 `±0.1°`，分别作用于旋转向量三轴并合成单位四元数。
- 失败时位置在手部基座原点周围 `±0.15 m` 立方体内替换，方向使用随机单位四元数；
  替换分布是本实现的明确选择，论文只规定随机替换与概率范围。
- 延迟/降采样保持此前观测；新回合强制发布新观测，不复用上一回合缓存。
- 旋转误差由同一条带噪位姿重新计算，避免通过目标差泄漏真实方向。

## 导出与视觉接入

导出脚本只需要 Torch 和项目 Python 包，不启动 Isaac Sim：

```bash
python scripts/rsl_rl/export_student.py \
  --checkpoint logs/distillation/revo3_v3/rslcube_student_01/student_1000.pt \
  --output_dir logs/distillation/revo3_v3/rslcube_student_01/exported
```

输出 `student_policy.pt`（TorchScript）及 `student_policy.json`（关节顺序、坐标、频率与布局）。
导出后自动重新加载，对非零循环状态和不同 batch size 验证数值一致性。

```python
policy = torch.jit.load("student_policy.pt", map_location=device)
hidden = torch.zeros(2, batch_size, 256, device=device)
# proprio: [batch, 113]，estimated_extero: [batch, 11]
with torch.no_grad():
    action, hidden = policy(proprio, estimated_extero, hidden)
hidden[:, reset_mask] = 0
```

控制周期记录在导出元数据中。蒸馏从老师保存的环境恢复实际周期；本次老师与学生均为 `1/30 s`。海康 RGB 回放任务也是 `1/30 s`，回放拒绝控制周期不匹配的学生权重。基础任务的旧 60 Hz 默认值不会覆盖绑定后的蒸馏设置。
`StudentObservations(...)(teacher_obs, estimated_pose=(position, quaternion), estimated_pose_frame="right_hand_base_link")` 提供仿真内的视觉接入点：
外部位姿必须先变换到手部基座 `right_hand_base_link` 坐标系，输入该参数后绕过模拟位姿噪声。
真实部署需自行构造相同本体观测、动作历史、任务剩余时间与手部基座坐标目标，并遵循导出的关节与动作协议。

## 标定相机与 URDF 手掌的坐标接入（2026-10-22）

学生现有数值协议使用 articulation root **`right_hand_base_link`**，不是 URDF 的
`right_palm`。历史检查点中的 `pose_frame="palm"` 是命名不准确；已知 v1 检查点加载时
按原来的基座协议解释，记录旧标签，不移动其数值。新检查点与导出文件明确记录
`pose_protocol="revo3_base_pose_xyzw_m_v1"`、`pose_frame="right_hand_base_link"`、米和 XYZW。
恢复、导出和 RGB 回放会拒绝不兼容的显式坐标协议。

使用交付的标定文件，`CalibratedPoseTransform` 校验矩阵、WXYZ 四元数及变换链，读取
`T_base_palm`、`T_base_camera`，不硬编码偏移。该包的 `T_base_palm` 平移为
`(0.0155, 0, 0.055) m`、旋转为单位阵，故掌心原点在学生输入中应为这个位置，不能当作零。
`T_base_object = T_base_palm @ T_palm_object`，或者直接
`T_base_object = T_base_camera @ T_camera_object`；已在基座下的位姿不再叠加偏移。

```python
from viserdex.distillation.frames import CalibratedPoseTransform
from viserdex.distillation.observations import StudentObservations

transform = CalibratedPoseTransform(
    "config/cameras/hikrobot_DB0447679_1280x1024_extrinsics_revo3_20261022.json"
)
adapter = StudentObservations(env, student_cfg, noise_cfg, pose_transform=transform)
# 输入必须是物体相对于光学相机的位姿，位置单位为米；不是相机相对于物体。
prop, extero, auxiliary_target = adapter(
    clean_obs, estimated_pose=(position_camera, quaternion_camera),
    estimated_pose_frame="camera_opencv", estimated_quaternion_order="xyzw",
)
# 若估计器已输出 right_palm 下的位姿，改用 estimated_pose_frame="right_palm"。
# 标定/估计器若输出 WXYZ，显式指定 estimated_quaternion_order="wxyz"。
```

纯部署无需模拟器，直接调用
`transform.to_student(position, quaternion, source_frame="camera_opencv", quaternion_order="xyzw")`，
再构造基座坐标下的目标四元数和 11 维特征；导出策略的输入形状与网络权重不变。
`transform.metadata` 提供标定 SHA-256 与坐标协议，供部署日志保存。
转换只读取外部估计与固定标定，不使用物体真值。

`camera_opencv` 表示 X 向右、Y 向下、Z 向前；不要额外叠加用于显示的 180° 翻转。
几何转换接受已估计的三维物体位姿，本身不校正图像畸变。现有 120×120 UVD 解码器
`keypoints_to_frame_pose` 使用针孔模型，接原始 Hikrobot 图像时必须先处理 Brown-Conrady
畸变，并使用与输入缩放/裁剪一致的内参；当前修改不代表真实相机 CNN 已接通或验证。

## 复现范围与验证

现有公开代码的教师是简化 PPO：输入 87、MLP `[512, 256, 128]`，
没有论文完整版教师的接触力、随机动力学属性等全部输入。
学生复用这个教师，因此重建目标为它已有的速度状态；未凭空添加不存在的特权信息。
教师环境的物理参数、奖励、60 Hz 控制和 EMA `alpha=0.95` 保持原实现，
与论文 Allegro 完整训练设置存在差异。当前实现属于论文学生方法的 Revo3 适配。

## RGB 位姿闭环回放

入口 `scripts/rsl_rl/play_student_vision.py` 现在默认使用
`Isaac-ViserDex-Repose-Revo3-Hikrobot`。它复用可视化视频的同一个
`make_calibrated_camera`：海康 DB0447679、1280×1024、完整 K 和 Brown-Conrady 五参数、
`T_world_base @ T_base_palm @ T_palm_camera`，通过官方 `CameraCfg` 和
`NewtonWarpRendererCfg` 读取包含活动手和方块的原生 RGB。旧 3DGS 位姿训练任务保留，
不再作为这个学生回放入口的图像来源。

新任务采用当前 compact116 手资产、正确的 58.5 mm RSLCube、30 Hz、dt=1/120、
decimation=4、两个物理子步。默认单环境渲染原始分辨率；物理容量沿用已经通过的
单环境预览预算 nconmax=2048、njmax=512，CUDA graph 关闭。
基础无相机任务和正在运行的老师训练不受这个新任务影响。

图像路径为：**原生海康 RGB → 校正镜头畸变并全幅缩放到 120×120 → binary opening
→ 冻结 CNN → 归一化 UVD → 校正后内参反投影/SVD 拟合 → 基座位姿 → 导出学生**。
`HikrobotImageProcessor` 不裁剪物体、不使用物体位姿或手部状态，不创建真值遮挡掩码。
横纵方向按完整画幅分别缩放，缩放后的内参包含像素中心偏移：
`cx_new=(cx+0.5)*120/1280-0.5`，`cy_new=(cy+0.5)*120/1024-0.5`；
对应的焦距同步缩放。畸变校正与缩放合并为一次 bilinear grid_sample。
传给 UVD 解码器的是校正后图像对应的 K，不使用原始 1280×1024 K 解码 120×120 预测。

启动时核验实际相机内参、镜头畸变和所有环境的外参，并校验学生的米/XYZW/基座坐标协议
与控制周期。学生权重与 TorchScript 输出仍做数值核对。目标真值只用于辅助监督和诊断，
不输入视觉位姿路径；评估使用公开 command_counter 统计真实成功换目标，排除初始目标与自动重置。
`--no_video` 只关闭视频写盘，相机及 CNN 推理仍运行。

```bash
bash scripts/run_newton.sh scripts/rsl_rl/play_student_vision.py \
  --viz none --device cuda:0 --num_envs 1 --num_steps 300 --video_stride 1 \
  --student_checkpoint logs/distillation/revo3_v3/my_student/student_1000.pt \
  --pose_checkpoint logs/rsl_rl/pose_estimator/my_pose_run/best.pth \
  --mode vision --output_dir logs/deployment/student_hikrobot_example
```

视频输出 `camera_native.mp4`（原生分辨率）和 `student.mp4`（原图、CNN 输入及诊断）。
`manifest.json` 保存相机 K/D/外参、标定 SHA-256、缩放后的 K、学生频率和权重来源。
`observations.npz` 保存实际学生输入、动作和 CNN 预测，图像与这些数据均在同一次 env.step 前采样。
使用 30 Hz、stride=1 时，300 步是 10 秒；stride=2 时视频为 15 fps，仍保持真实仿真时间。
`--mode baseline` 使用原来的模拟位姿噪声控制，CNN 仅用于诊断。

更换视角和完整场景图像后，旧 3DGS CNN 的新视角精度尚未验证，应使用这套相机和同样的
图像处理训练/验证对应的位姿网络。当前相机测试不表示已获得可用的学生或 CNN 权重。
`preview_student_hikrobot.py` 通过官方 play 让现有老师驱动手，并使用完全相同的
生产相机、畸变校正和 binary opening 录制实际 CNN 输入，专门用于核对视角。

## 历史仿真验证（2026-10-17）

一组记录的 RGB 闭环对比使用相同学生、seed 53、64 环境、1,200 步。
基线与 RGB 路径均启用录像相机，整体指标来自全部 64 个环境：

| 指标 | 模拟位姿噪声基线 | 3DGS RGB → CNN 位姿闭环 |
| --- | ---: | ---: |
| 全部环境观察到的重定向成功增量 | 827 | 970 |
| 已结束回合平均成功次数 | 12.21875 | 14.265625 |
| 越界终止 | 10 | 9 |
| 平均每步奖励 | 0.0581835 | 0.0661939 |
| 送入学生的平均位置误差（mm） | 43.2211 | 1.3137 |
| 送入学生的平均旋转误差（°） | 21.2017 | 0.9329 |
| 录像环境 0 的成功次数 | 13 | 11 |

视觉闭环平均位置/角度误差的 P95 为 2.6427 mm / 2.1992°；
平均手部遮挡比例 4.71%，P95 为 12.02%。64 张图像整批 CNN+位姿解码平均
10.07 ms；此数值不是完整仿真或真机控制频率。视觉观测逐步同步刷新，
未模拟真实相机传输延迟，也未额外施加训练中的位姿失败/扰动。
本次成功总数增加 17.29%，仅为一组仿真对比，不能推断真实相机或重遮挡性能。

原始视频、观测日志和训练权重不随代码分发；这些是历史仿真记录，并非本次整理重新运行的结果。

## 2026-10-22 corrected Newton student learning protocol

The active 4096-world job uses the original belief network (113+11 inputs,
21 actions), 45-step sequences, 32 learning epochs, AdamW 3e-4, the original
reconstruction coefficient 0.2, and beta=0.9*0.95^completed_updates. Policy
frequency remains the user-requested 30 Hz. This is the project's custom
ViserDex learner, not the RSL-RL native DistillationRunner.

Three explicit, checkpointed controls were added:

* optimizer_env_batch_size=256: each optimizer step uses complete 45-step
  sequences for at most 256 environments, shuffled by environment each epoch.
  update_env_batch_size=128 only controls gradient-accumulation memory.
  At 4096 environments this gives 16*32=512 optimizer steps per rollout, instead
  of 32 full-4096 steps. This restores the optimizer batch size of the public
  fork's default 256-environment setup; the exact historical successful run's
  training count remains unknown. Statistics count each collected sample once.
* refresh_hidden_after_update=True: replay the just-collected sequence with
  updated weights/normalizers, preserving reset boundaries and clearing
  environments that terminated on the last step. The old loop carried hidden
  states computed before all learning epochs into the updated policy.
* teacher_quaternion_sign=positive_world_w: canonicalize teacher object world
  quaternion and consistently flip its object*conjugate(goal) error. Positions,
  velocity terms and physical state are untouched. Students already canonicalize
  their pose features. In a real sampled state, equivalent q/-q yielded
  identical student features but original teacher labels differed by MSE 4.787;
  canonicalized labels matched exactly. This changes the oracle representation,
  so it is a separately named observation protocol, not a silent resume.

The defaults for all three fields preserve old checkpoint behavior. Resume
rejects command-line changes to learning settings. New checkpoints declare
shared_noisy_joints_canonical_world_quaternions_v2. Old raw-quaternion
checkpoints still load/evaluate with their original behavior.

Example fresh corrected training:

    bash scripts/run_newton.sh scripts/rsl_rl/distill.py \
      --teacher_checkpoint /absolute/path/to/frozen_teacher/model_4800.pt \
      --num_envs 4096 --max_iterations 1000 --save_interval 5 \
      --update_env_batch_size 128 --optimizer_env_batch_size 256 \
      --refresh_hidden_after_update --teacher_quaternion_sign positive_world_w \
      --device cuda:0 --viz none physics=newton_mjwarp

For resume, provide --student_checkpoint and no learning-setting overrides;
use a new output directory. All physical/sensor randomization, saved assets,
reward definitions, joint order, task timing and Newton solver parameters are
preserved. Smaller diagnostic runs only raise nconmax for buffer headroom.

A real-sequence state-refresh test reduced next-action teacher MSE from
1.00564 (stale memory) to 0.66865 (rebuilt memory); hidden RMS change was 0.27653.
This is a single recorded-batch diagnostic, not a closed-loop convergence claim.
The initial 1024-world/10-update pure-student tests are deliberately retained:
legacy 211 earned goals / 271 out-of-reach terminations; minibatch+refresh
331/367; plus canonical oracle 351/338. These early models are still unstable;
higher goal counts alone do not establish success. Follow-up 40-update controls
and independent production-checkpoint evaluations are recorded in
outputs/student_root_cause_20261022/.

Installed CPU regressions: 58 passed; Ruff passed. Candidate packaging initially
omitted the unchanged camera calibration fixtures; the complete fixture rerun
passed and the first failure log is retained. The first standalone state probe
also mistakenly disabled autograd around optimization; its corrected probe is
saved separately. Neither harness error was a production training error.


### 2026-10-22 independent outcome and protocol limits

The restart is running; it is not a convergence certificate. All production
student evaluations below execute student actions exclusively (teacher_used=false),
64 worlds x 600 steps, seed53,30Hz, full training sensor/physical noise.
Earned goal increments / object-out-of-reach terminations:
update1 89/646; update5 389/361; update10 442/251; update20 415/139;
update80 662/111. Initial/reset goals are excluded. These are repeated-goal
counts, not success percentages. Small evaluation capacity is2048/512.

Important counterevidence: matched1024-world controls continued to40 updates
produce raw-teacher minibatch+refresh544/131 versus canonical-teacher472/156.
The separate clean teacher test is1474/6 raw versus1173/5 canonical. Therefore
canonicalization demonstrably removes q/-q label ambiguity, but changes the
oracle and is not established as the better closed-loop protocol. The current
canonical4096 run is retained because its independent progression is improving,
not because canonicalization won the controlled comparison. Original student200
was236/26; current80 has more goals and more drops, so this is not a claim of
uniform improvement. A single-seed short control cannot prove convergence.

The original public fork defaults to256 environments and60Hz; the historical
successful run actual training configuration/checkpoint is unavailable. At4096,
the old full-batch loop still did32AdamW steps per collection round. The new
logical256 batches perform512 steps, while every sample still receives32 epochs
and normalization updates once. Hidden memory reuse existed in both original
and migrated code; refreshing it is a targeted recurrence correction, not proof
that Newton introduced the bug. Replay covers the current truncated45-step
sequence, starting from its supplied initial state, not the entire episode.

The30Hz setting is retained as requested;45steps now span1.5s, and step-defined
sensor delays/refresh periods have doubled physical duration from60Hz. Network,
losses, noise magnitudes, beta formula, task randomization and current assets
remain. Teacher privileged velocities matter, but finite-difference history
ablations show that useful information is available through temporal inputs.
No evidence establishes a unique PhysX/Newton failure cause.

At audit 2026-10-22T23:51:51.654020+08:00, production completed update90;
last10 median iteration 48.22s,
rollout 23.69s, update 24.46s.
Own process GPU peak 26244MiB. Checkpoint
student_85.pt is finite; AdamW step43520,
normalizer count15667200. Installed source
matches the recorded manifest;58CPU tests and Ruff pass. All completed audit
GPU jobs exit0 with no capacity/numeric fatal patterns. Teacher optimization
and SDF training remain stopped; this student does not create a camera/CNN.
Full reproducible plans, actual results, rejected harness logs and source diff:
outputs/student_root_cause_20261022/repair_report.json and final_evidence.zip.
