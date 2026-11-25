# SuperMimic 更新耗时排查（2026-10-24）

更新：本报告记录的执行方案现已接入正式 Trainer，分块/精度写入 version-3 checkpoint，并通过 19 项测试和正式入口的 3000 环境两轮联合更新检查。下文的源码 SHA、临时变体和“未启动”描述保留其性能排查时的语境；当前启动方式以 [训练说明](revo3_supermimic.md) 为准。

结论：此前 20.80 秒主要来自本实现过小的统一 microbatch 和串行 eager 执行，不能解释为 SuperMimic 算法必然需要这个时间。保留全部逻辑训练量，仅改变计算分块，纯网络更新已实测降至 3.67 秒；完整 batch + BF16 为 0.70 秒。进一步按用户允许的 3000 真实环境做同卡短测，完整网络更新约 0.79 秒、整卡显存峰值 29.25 GiB，连续三轮通过。

## 已停止的作业

按用户要求，22021 的原学生蒸馏已通过 SIGTERM 正常结束，退出码 0，完成 755 轮。原 PID 2953386 已不存在；guard 和 active pointer 均已核对。

保留最后权重：

```text
/home/zhangke/viserDex_revo3/logs/distillation/revo3_v3/student_paper4096_30hz_rawteacher_v1_20261023/student_755.pt
```

文件大小 45,507,253 字节。该轮排查没有重启正式训练。性能诊断进程已全部退出，22021 的 GPU 7 最后读数为 5 MiB、0% 利用率。未停止其他作业。

## 参考项目究竟多快

参考的是第三方复现 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)，不是论文作者官方实现。

其 [MILESTONES](https://github.com/zita-ch/supermimic_vibed/blob/main/MILESTONES.md) 自报，在 RTX PRO 6000 Blackwell、48 CPU threads 上，初版完整流水线约 0.66 轮/秒（1.52 秒/轮）；引入编译、批量计算和 BF16 后约 3.1 轮/秒（0.323 秒/轮）。这是作者记录，并非本次在本服务器重新跑出来的参考项目成绩；任务和硬件都与 Revo3/RTX 5090 不同。

固定提交 [a1508d4 的 train.py](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/train.py) 使用 WM k=8、10×2048，Actor H=32、batch4096，Critic 5×4 次更新；编译 WM rollout/error、想象单步、reward/termination 和 Critic 输入；将奖励与 Critic 计算沿 rollout 批处理；MLP 默认 BF16。该来源只支持“参考项目没有必要花 20 秒”，不支持“Revo3 必定达到 0.323 秒”。

## 对照方法

- GPU：22021 的 GPU 7，RTX 5090，UUID `GPU-e390d038-f478-84ae-cc23-23f41cac26f5`，总显存 32607 MiB；Torch 2.11.0+cu130。
- 从既有 validated environment YAML 启动 64 个真实 Newton 环境采集 64 步固定回放，随后退出仿真，再单独测网络。回放不是合成随机状态。
- 网络输入、网络宽度、30 Hz 环境契约、10 帧观测历史、持久传感器状态、奖励/EMA/终止规则保持原样。每次仍是 WM 10×2048×k8、Actor 4096×H32、Critic 5×4。
- 每个方案独立恢复同一诊断 checkpoint，包括深拷贝的 Adam 状态；预热一次、再测三次。阶段边界同步 CUDA。采样时间是 WM/Actor 时间的子集，不能再加到总耗时。
- 各方案都验证了累计 WM=40、Actor=4、Critic=80 次 optimizer.step，含一次预热。microbatch 仅用于梯度累积，不减少逻辑 batch 或更新次数。
- 本测试是网络执行性能对照，不是策略收敛评估。不同分块会改变随机数的消费顺序和浮点累加次序；BF16 还改变 MLP 数值精度。没有宣称最终策略效果等价。
- 原 `trainer.py` 未修改，SHA-256 为 `d362cff0d7c971b15b483e38905d7bdfbaa22d0b1bc62408508eeff4f2fe5272`。分块变体只存在于独立诊断进程。

## 三次重复的中位数

时间均为秒。显存为 Torch allocator 峰值，**不含仿真及全部 CUDA 运行时开销，不能当成整卡占用**。每列独立取中位数，阶段中位数之和不必严格等于总时间中位数。

| 计算方式 | WM | Actor | Critic | 完整更新 | Torch 峰值 |
|---|---:|---:|---:|---:|---:|
| 原版：全部切 128，FP32/TF32 | 4.086 | 9.322 | 6.454 | **19.840** | 0.950 GiB |
| 只取消 Critic 的 128 分块 | 5.040 | 9.347 | 0.045 | **14.433** | 1.299 GiB |
| WM/Actor 切 512，Critic 大批次，FP32/TF32 | 1.299 | 2.323 | 0.046 | **3.671** | 1.932 GiB |
| 上一行加 MLP BF16 | 1.157 | 2.500 | 0.044 | **3.702** | 2.570 GiB |
| WM 2048 / Actor 4096 完整批次，MLP BF16 | 0.344 | 0.315 | 0.046 | **0.698** | 9.832 GiB |

原版三次为 19.840、17.366、21.812 秒；另一次先导测试约 21.30–22.12 秒。应报告这个波动，不能把一次 20.80 秒当成恒定耗时。512 分块 FP32 的三次为 3.645、3.671、3.675 秒，完整批次 BF16 为 0.693、0.707、0.698 秒。

回放采样和 CPU→GPU 传输合计中位数仅约 0.034–0.040 秒；在这个固定小回放试验中不是主因。完整 1000×4096 大回放的缓存/传输表现尚未单独测量，不能由此推导所有规模都相同。

## 为什么原实现慢

1. **Critic 被过度切分。** 最多有 4096×32=131072 个想象样本，4 个逻辑 minibatch 每个 32768 条，再切 128 会成为 256 块。5×4×256=5120 次小块前后向，而优化器仍然只走 20 次。恢复大批次后该阶段降至约 0.045 秒，只有约 0.35 GiB 的峰值增量。这是最直接的实现问题。
2. **Actor 串行展开过多。** 4096/128=32 个分块，每块单独建 32 步计算图并反传；同一轮共执行 1024 次想象单步。增大分块后 GPU 批处理效率提升。保留持久噪声、目标变换、EMA 等规则会增加每步算子，但不需要因此把所有样本切成 128。
3. **WM 同样受影响。** 每个逻辑 batch 2048 被切为 16 块；10 次更新共 160 条独立的 k8 前后向链，完整 batch 则是 10 条。实际数据量相同，Python 调度和小 kernel 启动次数差别很大。
4. **频繁同步和重复特征构造。** WM、Actor、Critic 的每个分块都执行 `float(loss.detach())`；Critic 还每块、每个 epoch 重算输入特征。参考项目把更多张量计算放在批量操作中，并延迟日志标量读取。本轮没有对“同步”单独消融，不能把全部改善都归给同步一项。
5. **缺少参考的编译路径。** 当前生产实现没有 `torch.compile`；本轮所有变体也都未编译，所以上表的改善来自分块/精度变化，尚未测到编译收益。
6. **BF16 本身不是本轮主要收益。** 在 512 分块下，BF16 的 WM 稍快、Actor 稍慢，总时间约 3.70 秒，未优于 FP32 的 3.67 秒。不能把 BF16 说成必然提速或省显存；这个 eager autocast 实现反而保留了更多转换张量。

相关代码：`source/viserdex/viserdex/supermimic/trainer.py` 的 `wm_update`、`imagine`、`policy_update`。原先为先验证显存与梯度正确性，所有网络共用 128 分块，选择过于保守；尤其小 Critic 不应机械沿用 Actor 的显存策略。

## 为什么不能直接用 0.70 秒方案启动 4096 环境

既有 4096 仿真检查整卡峰值约 26.06 GiB，而 Torch 峰值约 0.99 GiB，说明仿真/其他 CUDA 分配本身约占 25 GiB。完整 batch 的网络单独就需要约 9.83 GiB；简单预算已超过这张卡约 31.84 GiB 的容量。这里的近似相加是容量风险估计，不是实测 OOM 结果。

更合适的下一步是分别设置 WM、Actor、Critic 的计算分块，首先采用已测的 512/512/32768 作为候选，再测编译、特征批处理及 4096 仿真共存时的整卡峰值与稳定性。无需为了性能先改 30 Hz、k8/H32、逻辑 batch 或每轮优化次数。长期训练效果仍需实际验证。

本轮未将性能变体写回正式 Trainer，也未重启正式训练。测试结束时显卡已释放。

## 3000 真实环境与完整批次的同卡验证

采用 3000 个真实并行环境，保持 30 Hz 控制、120 Hz 物理频率和原资产/随机化/奖励。WM batch=2048、Actor batch=4096、Critic 每逻辑 minibatch 最多 32768，均使用完整批次；MLP 使用 BF16，物理状态积分、奖励等保留 FP32。环境数独立于回放中采样的网络 batch。

通过 `profile_in_sim.py` 调用原官方 launcher 训练入口，仅在本诊断进程替换执行分块/MLP autocast，并同步分段计时。`--check_joint_update --audit` 明确标记短测，强制测试部分 timeout；从头初始化，在 64 步 warmup 后运行三次联合更新，没有加载教师或学生权重。

| 完整更新 | WM | Actor | Critic | 网络总时间 |
|---|---:|---:|---:|---:|
| 第 1 次（含网络首轮预热） | 0.496 s | 0.474 s | 0.052 s | 1.022 s |
| 第 2 次 | 0.404 s | 0.323 s | 0.054 s | 0.781 s |
| 第 3 次 | 0.398 s | 0.338 s | 0.060 s | 0.796 s |

- 包含真实采集后的第 2/3 轮间隔约 2.01 / 1.62 秒（分别采集 3 / 2 个向量步），不能把 0.79 秒宣传为包含物理采样的完整训练轮耗时。
- 整卡显存 250 ms 轮询峰值 **29951 MiB = 29.25 GiB**，GPU 总容量 32607 MiB，采样峰值剩余约 2.59 GiB。轮询不是引擎逐分配追踪，短于采样间隔的瞬时峰值仍可能遗漏。
- Torch allocator 峰值 9.868 GiB；CPU 回放预留 17.636 GiB，容量仍为 1000 向量步，本次实际写入 71 步。
- 累计 optimizer.step：WM=30、Actor=3、Critic=60。各轮损失与梯度有限。
- 日志无 warning/error、接触容量 overflow、OOM 或 NaN；进程正常退出 0。奖励最大绝对误差 1.04e-7、EMA 2.38e-7，观测接口检查持续通过。
- 只验证了短期容量和执行正确性，尚未证明长期接触峰值下不 OOM、策略收敛或 BF16 的长期数值效果。性能变体尚未接入正式 Trainer 默认配置。

远端目录：`outputs/capacity3000_fullbatch_bf16_20261024/`。保存的 `latest.pt` 带 `check_joint_update`、`forced_reset_audit` 和 `performance_diagnostic` 元数据，属于诊断权重。

## 复现与原始证据

- [逐次计时与中位数 JSON](evidence/supermimic_performance_20261024/network_profile_v2.json)
- [逐行日志](evidence/supermimic_performance_20261024/profile_v2.log)
- [3000 环境逐轮指标](evidence/supermimic_performance_20261024/capacity3000_metrics.jsonl)、[整卡显存采样](evidence/supermimic_performance_20261024/capacity3000_gpu.json)、[接口验证](evidence/supermimic_performance_20261024/capacity3000_validation.json)、[完整日志](evidence/supermimic_performance_20261024/capacity3000.log)。
- `scripts/supermimic/capture_profile_fixture.py`：真实环境回放捕获，无 optimizer 更新。
- `scripts/supermimic/profile_updates.py`：相同 checkpoint 下的短期执行变体、阶段计时、优化次数断言。
- `scripts/supermimic/profile_in_sim.py`：最多三轮、真实环境共存的完整批次性能诊断。
- 远端诊断目录：`/home/zhangke/viserDex_supermimic_20261024/outputs/performance_20261024/`。

正式计时证据为 `network_profile_v2.json`。先导 `network_profile.json` 发现 Adam CPU step tensor 会跨变体共享，已通过深拷贝完整状态修正后全部重测；先导结果不用于上表。
