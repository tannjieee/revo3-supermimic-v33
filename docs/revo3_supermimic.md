# Revo3：直接学习可部署策略的 SuperMimic 实现

本实验位于独立分支 `codex/revo3-supermimic`。原 ViserDex 教师、GRU 学生、蒸馏入口、预测器和权重全部保留。新 Gym 任务为 `Isaac-ViserDex-Repose-Revo3-SuperMimic`；训练入口是 `scripts/supermimic/train.py`。

这里参考的是 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9) 的算法与参数（固定提交 `a1508d42c9d1699a52e8233ab16d08c488441ab9`），**该仓库是论文复现，不是论文作者官方代码**。本项目代码是面向 Revo3 的独立实现，不能把人形机器人运动跟踪的任务函数直接移植为手内重定向。

## 数据和网络

策略仅接收最近 10 帧实际学生观测，按从旧到新的顺序排列。每帧 124 维：归一化关节位置 21、最近四条原始动作 84、目标四元数 4、手基座位置 3、剩余时间 1、带噪物体位置 3、物体四元数 4、物体相对目标四元数差 4。所有四元数使用 XYZW；物体位姿使用 `right_hand_base_link` 坐标系。目标改换不清空历史，整集重置才清空动作历史并重复首帧填充观测历史。

训练侧状态 `s` 为 172 维：物理核心 55、EMA 关节目标 21、四条原始动作 84、目标位姿 7、剩余时间 1、成功计数 1、手基座位置 3。55 维物理核心包含关节位置/速度 42、物体位置 3、四元数 4、线速度/角速度 6。它不是全部特权信息。原老师 49+38 维清洁观测可从此状态还原，代码有真实仿真逐项断言；四元数两侧统一符号后比较。

域参数 `xi` 在本次资产下为 548 维：机器人/物体实际质量、惯量、每个形状的摩擦和恢复系数、实际 Newton 控制器 kp/kd、关节限位。全部从已随机化的当前仿真读出，不能用配置范围或平均数替代。材质使用官方 Newton 事件同一只读 view binding；控制增益使用公开 `read_group_parameter`。域随机化保持开启，当前配置的这些参数仅在 startup 改变。

域参数先除以启动时真实参数的逐维平均绝对量级（最小 1e-12），使质量、惯量等变为无量纲特征，再进行 running normalization。此缩放保存为网络 buffer；恢复时沿用训练缩放，不能在新环境上重新定义输入协议。否则 1e-7 量级的惯量会被通用 eps=0.01 几乎抹掉。新检查点格式为 version 3，保存数值精度和各网络执行分块；仍支持读取 version 2，并按旧版 FP32/统一分块解释缺失的字段。不支持更早的域参数缩放协议。

| 网络 | 输入 | 隐藏层 | 输出 |
|---|---|---|---|
| Dynamics | 物理特征 57 + EMA 目标 21 + xi 548 = 626 | 2048×3，LayerNorm+SiLU | 54 维物理增量 |
| Actor | 10×124 = 1240 | 2048、1024、512、256，ELU | 21 个原始动作 |
| Critic | 完整状态特征 176 + xi 548 = 724 | 256、128，ELU | 标量 V |

旋转输入用 6D 表示，因此物理特征从 55 变为 57；完整状态有两组四元数，从 172 变为 176。WM 输出按六类组织：关节位置 21、关节速度 21、物体位置 3、旋转向量 3、线速度 3、角速度 3。采用一个 MLP 输出，不是六个独立模型。

WM 系统中的观测、奖励、终止三个函数采用解析代码；唯一学习的 WM 网络是动力学。Actor 不接收 `s`、`xi`、Critic 输出或真实物体速度。没有教师动作损失、DAgger、GRU 或额外蒸馏阶段。部署时可把冻结的海康位姿预测器结果按已有坐标协议组装成同样的 124 维观测；本入口尚不运行相机或训练 CNN。

## 一次迭代

1. UTD=0.4，交替采集 2/3 个真实并行控制步。控制频率 30 Hz，物理 120 Hz，保留原求解器配置。
2. WM 做 10 次优化；每次采样 2048 条连续 8 步窗口，以记录的动作自回归推演。中间不回填真实物理状态。终止步自身保留监督，终止后掩掉。
3. Actor 从回放采样 4096 个起点及其实际观测历史、传感器状态，推演 32 步，沿固定 WM 和 Critic 反传一次。冻结参数不等于 `no_grad`。
4. 用更新前的 Critic 一次性计算固定 λ-return 目标；γ=0.99，λ=0.95，5 epochs×4 minibatches，共 20 次 Critic 优化。

初始采集 64 步，前 500 次迭代仅更新 WM；之后才进入联合训练。Adam 基础学习率 2e-4，Actor betas=(0.9,0.95)，250000 次迭代余弦衰减至 3%。默认最大迭代数沿参考代码为 320000，不代表已完成或保证收敛。真实采集探索噪声标准差为 0.1，原动作到关节限位映射和 EMA=0.95 不变。

正式 Trainer 支持分别指定 `wm_microbatch`、`actor_microbatch`、`critic_microbatch`；未指定时才回退到兼容旧版本的 `microbatch=128`。当前 3000 环境方案显式设为 2048/4096/32768，采用完整逻辑批次计算，并设 `amp=bf16`。这些字段全部存入 checkpoint，恢复时沿用并校验。CPU 环形回放仍保存最近 1000 个向量步及真实历史/噪声状态，3000 环境预留约 17.64 GiB 主机内存，按实际采集逐步写入。

[性能排查](revo3_supermimic_performance.md) 记录了从统一分块约 20 秒，到 3000 环境完整批次 BF16 约 0.79 秒网络更新的对照。正式入口已集成该执行路径；短测结果不能代替长期稳定性和收敛验证。

## 重置、噪声与梯度边界

- 使用官方 `RecorderTerm.record_post_step`，在自动重置和 command 更新前抓取下一物理状态。没有覆写官方环境的 step，也没有改 Isaac Lab/Newton 源码。
- 重置后 EMA 从新关节位置开始，四条原始动作历史清零；终止前状态和重置后状态分别保存。
- 原学生噪声的 bias、encoder delay、采样周期、jitter、failure 和 held pose 都跨帧持久化。使用纯函数版本避免 BPTT 中原地修改张量；测试对照原实现逐帧一致。真实采集的传感器状态随起点进入想象，不能重新抽一段毫无关联的噪声历史。
- 成功奖励保留原硬阈值 `angle<=0.1`；换目标保留 `<0.1`，并保持原 `qx* qy` 采样。硬成功阈值和换目标分支对跨边界的导数不存在，当前只对选中分支内的连续路径求导。**没有把它说成精确可微，也没有暗中用平滑奖励替换真实奖励。**
- 想象阶段对掉落距离使用 sigmoid survival，中心保持原 0.3 m，宽度 0.01 m 是本任务明确的工程适配值，不是论文给出的手部参数。真实仿真仍用硬终止。超时按原无限时域任务 bootstrap 一次，此后停止该轨迹的回报并排除后续 Critic 样本。
- WM、Actor、Critic 的归一化统计只从真实采集更新，样本数到 2e8 后冻结；输入 eps=1e-2、增量 eps=1e-4。参考复现代码会用想象状态更新 Critic norm；这里采用此前设计约定的真实数据统计。
- 正式 3000 环境方案与参考采用同样的 MLP BF16 autocast，输出转回 FP32；物理状态积分、旋转、奖励和观测运算保留 FP32，参数与优化器状态也保留 FP32。允许 CUDA matmul TF32。通用入口的 `amp=none` 仍可用于旧检查点和精度对照。
- 人形参考的单独未来运动目标编码器在这里省去：当前 Revo3 目标已经包含在每帧 `o` 中。本实现不凭空添加未来目标序列。
- Critic 是训练侧物理状态价值近似，不接收 Actor 10 帧历史和完整噪声内态；在带持久噪声的部分可观测系统中，它不是严格完整的 Markov value oracle。这与“Actor 不再需要蒸馏”是两回事，需用真实评估验证偏差。

## 启动和部署

在安装过官方锁定运行时的项目根目录，指定已有验证过的环境 YAML。此文件只定义资产/动力学，不会加载任何教师权重。

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU_UUID> bash scripts/run_newton.sh scripts/supermimic/train.py \
  --env_config /absolute/path/to/validated/env.yaml \
  --output_dir outputs/supermimic_run --num_envs 3000 --amp bf16 \
  --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 32768 \
  --iterations 320000 --save_interval 2000 --latest_interval 100 --viz none
```

保留默认 `physics=newton_mjwarp`。独立任务使用项目既有 `SimulationRuntime` 的官方 preset 解析和 `launch_simulation`，自定义 model-based trainer 负责更新三套网络；不是把它伪装成 Isaac Lab 官方现成 PPO 算法。启动会验证环境哈希、资产 SHA、动作、30 Hz、关节顺序和奖励契约，不接受偷改物理的 override。

正式运行从头初始化，不传 `--smoke`、`--audit` 或 `--check_joint_update`，前 500 轮仅更新 WM。每 100 轮原子替换 `latest.pt`，每 2000 轮保留一个独立里程碑 checkpoint；正常停止也会保存。可用 `scripts/supermimic/run_guard.py` 配合 tmux 在后台记录 PID、最新指标和整卡显存状态。

```bash
# 小规模实现检查：保留网络宽度/k8/H32，明确缩小 batch 和更新次数。
bash scripts/run_newton.sh scripts/supermimic/train.py \
  --env_config /absolute/path/to/validated/env.yaml \
  --output_dir outputs/smoke --num_envs 64 --iterations 3 \
  --smoke --audit --holdout --microbatch 16 --viz none

# 恢复；使用检查点保存的模型/超参/优化器/统计量和迭代数。
bash scripts/run_newton.sh scripts/supermimic/train.py \
  --env_config /absolute/path/to/validated/env.yaml --resume /absolute/path/latest.pt \
  --output_dir outputs/resumed --num_envs 4096 --viz none

# 评估，无探索动作噪声，仍保留原学生感知噪声；用多个 seed 分别运行。
bash scripts/run_newton.sh scripts/supermimic/train.py \
  --env_config /absolute/path/to/validated/env.yaml --resume /absolute/path/latest.pt \
  --output_dir outputs/eval_seed43 --num_envs 64 --seed 43 --evaluate --viz none

# 导出只依赖 10×124 测量历史的策略。
.venv-newton/bin/python scripts/supermimic/export_actor.py \
  /absolute/path/latest.pt /absolute/path/actor.jit
```

部署导出继续使用便于 CPU/设备迁移的 FP32 推理，并在 JSON 记录训练精度与导出精度；不声称与 BF16 推理逐 bit 相同。

恢复会重新初始化仿真并采集 warmup 数据，回放和引擎状态没有打进检查点，因此**不是逐 bit 连续恢复**。迭代数是总目标而非额外轮次。导出会用不同 batch 的输入回读验证，输出仍须通过原关节限位映射和 EMA，不能直接当成弧度目标下发。

训练恢复会恢复 checkpoint RNG；独立 `--evaluate --seed` 保留本次评估的随机流，避免被训练 checkpoint 的随机状态覆盖。诊断标记会随恢复链保留，不能通过换输出目录把短测权重变成“正式训练”。

`--check_joint_update` 仅用于完整逻辑 batch 的更新检查：跳过 500 次 WM-only 阶段，保留 k8/H32、batch2048/4096、10/1/20 次优化。它会写进元数据，不可把该诊断权重当作正式训练结果。

## 验证记录与限制

详见 [实测记录](revo3_supermimic_validation.md)。测试区分实现正确性、负载容量和最终任务效果。短程训练 loss、梯度非零、程序退出成功都不能证明 32 步世界模型可信或策略已收敛。新鲜 holdout 会报告 1/8/16/32 步物理误差、有效环境数和“状态保持不变”基线；没有拿训练回放误差冒充泛化。

当前仍需在足够训练后验证：多随机种子的成功次数/掉落率、强接触分布下的长程预测、模型动作扰动方向与真实 Newton 因果响应一致性，以及真实 CNN 输入的闭环表现。自动微分对有限差分的测试只证明模型内部导数一致，不能代替最后这一项真实物理对照。
