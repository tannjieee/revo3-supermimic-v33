# Revo3 SuperMimic 正式训练启动记录

## 第 10292 轮奖励校验中断及恢复（2026-10-24）

2048 环境运行于第 10292 轮后退出，`Adapter.step` 的奖励一致性断言报告 2048 个环境中仅 1 个不匹配，差额 8.333333969116211，恰好是一次成功奖励 250 / 30。GPU 峰值保持 24341 MiB，本次日志没有 OOM。最后可读取的完整 checkpoint 是第 10200 轮，包含 WM102000 / Actor9700 / Critic194000 次优化状态。

中断前最近 500 轮（9793–10292）结束 6929 个 episode，累计完成 36263 个目标，平均 **5.23351 个/episode**；前 500 轮为 4.95398。最近 100 轮为 5.01194，前 100 轮为 5.03589，短窗口大致持平。网络更新/整轮耗时中位数为 0.819/1.710 秒。这是训练采集指标，不能直接与旧学生的独立评测作排名。

官方奖励在世界坐标中调用 `quat_error_magnitude`，旧校验在手掌坐标中重建旋转误差。对 20 万个刻意集中在 0.1±0.000001 弧度的 GPU 合成案例，官方角度与手掌坐标角度最大差 5.2154e-7 弧度；7287 个案例跨过成功阈值。一个案例分别为 0.10000002384 和 0.09999970347 弧度，产生与故障一致的 8.333333969 奖励差。这不是训练中发生概率的估计；原始故障 transition 未保存，不能声称逐位重放了该样本。

修复仅针对真实步骤的校验和统计：官方 recorder 在 reset/换目标前捕获官方世界坐标角度，先独立验证两种坐标下角度一致，再用世界坐标角度检查完整奖励，保留原奖励容差。真实成功计数也采用该角度。官方环境、奖励、可微想象中的奖励公式、模型和超参数未改变。新增回归用例接受边界舍入，但拒绝错误 bonus、错误连续奖励和错误姿态/目标。服务器测试 **20 passed in 5.28s**，Ruff 和 diff 检查通过。恢复运行的结果需另行记录。

恢复于北京时间14:38:13启动，使用服务器22021的 **GPU0**（GPU7已被另一任务占用）。代码提交 `5c71812`，从10200轮恢复至新目录 `outputs/supermimic_revo3_2048_30hz_bf16_resume10200_20261024/`；保留原失败日志及checkpoint。14:42:02复核到 **10315轮、joint**，PID152441仍运行，WM103150 / Actor9815 / Critic196300；**10300轮checkpoint已实际加载**，优化次数为103000 / 9800 / 196000。模型配置、噪声、关节顺序、环境契约、资产及执行精度/批次均与父checkpoint一致。仿真与replay重新预热，非逐位恢复；10201–10292重跑。

恢复后115轮检查：显存及峰值24341 MiB，最近50轮网络/整轮中位数0.883/1.644秒；奖励最大误差2.861e-6，坐标角度最大差1.431e-6弧度。没有warning/error/overflow/OOM；本次短程未再次遇到阈值两侧不一致样本，边界修复由合成案例及回归测试验证。初次恢复的episode尚未覆盖完整600步，不能用这115轮的episode均值替代中断前500轮效果判断。证据：[中断状态](evidence/supermimic_training_20261024/failure10292_status.json)、[恢复及checkpoint检查](evidence/supermimic_training_20261024/resume10200_verification.json)。

## 2048 环境恢复（2026-10-24）

原 3000 环境运行在第 **855 轮**后采集真实状态时失败，退出码 1。异常位于官方 MJWarp `collision_convex.py` 的 EPA 临时缓冲区分配，Warp 无法再分配 331,776,000 字节（约 316.4 MiB）。guard 记录整卡峰值 32065 MiB；故障前最近 100 轮网络更新中位数 0.785 秒，整轮中位数 1.742 秒。累计完成目标 5 次、掉落 3441 次、结束 episode 3469 个；结束 episode 的平均完成目标数约 0.00144，不能视为策略已学成。错误日志没有提供 Torch reserved / allocated 和 Warp pool 的分项数据，因此不能进一步断言显存增长全部来自缓存、碎片或真实接触峰值。

用户随后明确要求显存过大时改为 **2048 环境**。已在同一服务器 22021 / GPU 7 启动恢复，保留同一代码提交、资产、物理参数、30 Hz、BF16、网络结构、WM2048 / Actor4096 / Critic32768 批次及更新次数。仅真实并行环境数从 3000 改为 2048。最后完整 checkpoint 是第 **800 轮**，恢复包含三个网络、归一化统计、Adam 状态及更新计数；仿真场景与 replay 重新预热，因此不是逐位连续恢复，第 801–855 轮需要重跑。

```text
新训练目录：/home/zhangke/viserDex_supermimic_20261024/outputs/supermimic_revo3_2048_30hz_bf16_resume800_20261024/
恢复来源：/home/zhangke/viserDex_supermimic_20261024/outputs/supermimic_revo3_3000_30hz_bf16_20261024/latest.pt
tmux：sm-revo3-2048-resume800-20261024
Python PID（恢复启动时）：43716
原失败状态：outputs/revo3_supermimic_3000_failed855_20261024.json
恢复启动记录：outputs/supermimic_2048_resume_launch_20261024.json
```

北京时间08:57:36恢复启动，09:01:46复核到 **第936轮、joint**，累计 WM9360 / Actor436 / Critic8720 次更新。整卡占用及本次采样峰值均24341 MiB（23.77 GiB），已超过原故障轮数855；第900轮 `latest.pt` 已实际加载验证，包含 WM9000 / Actor400 / Critic8000 次优化计数。与父 checkpoint 的 Config、噪声参数、环境契约、关节顺序、上下文协议逐项一致，真实环境数为2048，step_dt=1/30。

最近100轮（837–936）网络更新中位数0.804秒、整轮中位数1.635秒，累计完成目标8次、掉落908次；结束episode平均完成目标数0.00330。WM loss最近100轮均值0.9758，末轮0.6560。控制台未发现 warning/error/overflow/OOM。策略仍处于早期，短程恢复正常不能证明长期稳定或收敛；不同环境数、重建replay后的这组计数不能直接当作与原运行的公平性能对比。[恢复检查证据](evidence/supermimic_training_20261024/resume2048_verification.json)。

以下保留原 3000 环境启动配置及历史检查记录，不代表当前仍在运行 3000 环境。

北京时间 2026-10-24 02:52:46，在服务器 **22021 / GPU 7** 启动。GPU UUID 为 `GPU-e390d038-f478-84ae-cc23-23f41cac26f5`，型号 RTX 5090。代码提交 `e970bb0d7c387fd0b8bcbed1331044cbb88780fe`，独立分支 `codex/revo3-supermimic`。旧学生蒸馏保持停止，其 `student_755.pt` 保留。

## 正式配置

| 项目 | 配置 |
|---|---|
| 真实环境数 | 3000 |
| 控制 / 物理频率 | 30 / 120 Hz |
| MLP 精度 | BF16 autocast，输出返回 FP32 |
| 参数、梯度、Adam 状态 | FP32 |
| 状态积分、旋转、奖励 | FP32 |
| WM | k=8，batch2048，10 次优化/轮 |
| Actor | 最近 10 帧观测，H=32，batch4096，1 次优化/联合轮 |
| Critic | 5 epochs × 4 minibatches，20 次优化/联合轮 |
| 执行分块 | WM2048 / Actor4096 / Critic32768 |
| 预训练 | 初始采集 64 步；第 1–500 轮只训练 WM，第 501 轮开始联合更新 |
| UTD | 0.4，交替采集 2/3 个真实向量步 |
| 训练目标轮数 | 沿参考入口默认值 320000 |
| 初始化 | seed42，从头开始，无教师/学生/诊断权重恢复 |
| checkpoint | 每 100 轮原子替换 latest.pt，每 2000 轮保留独立 model_N.pt；正常停止也保存 |

未启用 `smoke`、`check_joint_update`、`forced_reset_audit`；未改资产、碰撞体、域随机化、奖励或控制频率。新 checkpoint 为 version3，保存和恢复分块/精度设置，支持旧版 version2 的 FP32 兼容读取。

## 路径与进程

```text
运行根目录：/home/zhangke/viserDex_supermimic_20261024
训练目录：outputs/supermimic_revo3_3000_30hz_bf16_20261024/
控制台日志：outputs/supermimic_revo3_3000_30hz_bf16_20261024.log
实时状态：outputs/revo3_supermimic_training_active.json
tmux session：sm-revo3-3000-bf16-20261024
Python PID（启动时）：3907802
```

guard 每 3 秒写入状态、最近一轮指标、最新 checkpoint 路径和 GPU 占用；启动快照已确认到第 62 轮，处于 `wm_only`，WM 计数递增且 Actor/Critic 未更新。此快照不是持续更新的进度报告，后续状态应读取服务器的实时 JSON 与 metrics.jsonl。

启动后复核已确认到 **第 529 轮、joint 阶段**，同一 Python PID 仍运行。累计 WM5290、Actor29、Critic580 次更新，符合前 500 轮 WM-only 的安排；最近一次网络更新 0.777 秒，WM loss=0.2076，整卡当时占用 30465 MiB。`latest.pt` 已生成（139726141 字节），控制台未发现 warning/error、接触溢出、OOM 或 NaN。该复核是一次时间点状态，不表示最终训练效果。

## 启动前验证

- 19 项测试通过（5.10 秒），包括 BF16 MLP 输出确实为 BF16、网络对外输出 FP32、参数/梯度 FP32、冻结 WM 仍能向动作传梯度、精度/分块 checkpoint 保存恢复、恢复后继续更新一致性，以及 version2 兼容读取。
- 正式 `train.py` 入口用 3000 环境完成两次联合更新；第二轮网络更新 0.796 秒，整卡采样峰值 29951 MiB。累计 WM20、Actor2、Critic40 次优化，无 warning/error、接触容量溢出、OOM 或 NaN。
- version3 checkpoint 成功导出 FP32 actor.jit，并用不同 batch 回读比较通过。导出 JSON 明确标识训练 BF16、部署导出 FP32。
- Ruff 格式/静态检查及 diff 检查通过。

短测验证容量和执行正确性，不能视为最终收敛或长期接触峰值稳定性的证明。正式运行尚处于预训练阶段时，Actor 没有开始学习，不能用这一阶段成功次数判断最终策略效果。

证据：[启动状态快照](evidence/supermimic_training_20261024/startup_state.json)、[正式配置](evidence/supermimic_training_20261024/run_config.json)、[正式入口联合更新](evidence/supermimic_training_20261024/validation_metrics.jsonl)、[接口检查](evidence/supermimic_training_20261024/validation.json)、[显存记录](evidence/supermimic_training_20261024/validation_gpu.json)。
