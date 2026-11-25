# SuperMimic Revo3 实测记录（2026-10-24）

代码分支 `codex/revo3-supermimic`；服务器 **22021** 独立目录 `/home/zhangke/viserDex_supermimic_20261024`。GPU UUID `GPU-c353bd61-5a0a-74b4-de6b-91243ff9931b`（21 号 GPU 2）。原学生在其他 GPU 上继续运行，原项目代码、权重和作业没有被覆盖或停止。

官方锁定环境：Isaac Lab v3.0.0（发行包版本 28.0.0）、Isaac Sim 6.2.0.0、Newton 1.6.1、Warp 1.17.0。使用原 compact 116 碰撞形状的非 SDF Revo3、原 RSLCube、原环境 YAML、30 Hz 控制、nconmax=1152/njmax=512。各启动均经过 `SavedEnvironmentBinding` 校验。

## 自动测试

17 项纯 Torch 测试全部通过，最终运行耗时 3.74 秒。覆盖：

- SO(3) 零角/近 π 梯度、旋转增量往返、q/-q 不变性。
- 原动作 EMA、奖励阈值和目标切换边界。
- 10 帧历史次序和局部 reset；目标切换不清空历史。
- 持久观测噪声与原学生实现连续 14 帧、含局部 reset 的数值对照。
- 跨帧传感器 BPTT 无原地张量版本错误。
- 回放环绕和终止前有效/终止后无效掩码。
- λ-return、超时 bootstrap 与超时后 Critic 样本剔除。
- 梯度可穿过参数冻结的 WM，但不更新 WM；与模型内部有限差分一致。
- microbatch 累积和完整逻辑 batch 的 WM 参数更新一致。
- WM-only 阶段不更新 Actor/Critic；联合阶段 optimizer 次数正确。
- 检查点/环境协议一致性，想象阶段不更新归一化统计；评估保留独立 seed，训练恢复才恢复 checkpoint RNG。
- 极小质量/惯量参数经量级缩放后不会消失。

`ruff check`、`ruff format --check`、Python 编译检查通过。测试文件为 `tests/test_supermimic.py`，不需要启动 Isaac Sim。

## 真实 Newton

先完成 16 环境两次、64 环境三次小批量诊断，再进行两轮独立的 4096 容量检查。第一轮后修正域参数单位缩放；下表采用最终 version-2 实现。

| 检查 | 最终实测 |
|---|---|
| 并行真实环境 | 4096 |
| 网络 | 正式宽度；未缩小 MLP |
| WM | k=8，10×2048 条窗口优化 |
| Actor | H=32，逻辑 batch=4096，一次优化 |
| Critic | 5×4=20 次优化 |
| microbatch | 128 |
| 本次更新部分耗时 | 20.80 秒，包含三网络更新，不含真实采集/初始化 |
| 整卡显存峰值 | 26684 MiB，约 26.06 GiB；nvidia-smi 每秒采样 |
| Torch allocator 峰值 | 约 0.99 GiB；不能用此值代替总显存 |
| CPU 回放预留容量 | 24.08 GiB；1000 向量步，按实际采集逐步写入 |
| 奖励最大绝对误差 | 1.19e-7 |
| EMA 最大绝对误差 | 2.38e-7 |
| 原学生 124 维清洁观测误差 | 1.49e-7 |
| 原老师 87 维状态还原误差 | 1.91e-6（两侧规范化四元数符号） |
| reset | 覆盖强制 timeout 与自然掉落，记录终止前物理状态 |
| 接触容量 | 日志未出现接触容量溢出、OOM 或非有限数错误 |

本次使用 `--check_joint_update --audit`，故跳过正式训练前 500 次 WM-only，并强制部分 timeout 来验证更新/重置链路。诊断标记写入 checkpoint，恢复后仍保留。**不是一次正式收敛训练，也不能把这个诊断权重用于评价算法最终能力。**

恢复测试：从 4096 检查点的第 1 次更新恢复到 64 环境的第 2 次更新，保持原逻辑 batch 和优化次数，累计为 WM=20、Actor=2、Critic=40；保存、恢复和环境协议校验通过。环境数变化不影响网络结构，域参数缩放沿用保存的 buffer。仿真/回放重新 warmup，不宣称逐 bit 续训。

部署导出：`actor.jit` 只接收 `[B,10,124]`，对 B=1、7 的输出与原 Actor 一致，无 `s`、xi、教师、Critic 输入。

最终独立评估（使用一次联合更新的诊断策略，保持感知噪声，无探索动作噪声）：

| seed | 环境×控制步 | 掉落 | 超时 | 成功换目标 |
|---|---:|---:|---:|---:|
| 43 | 64×600 | 7 | 57 | 0 |
| 44 | 64×600 | 2 | 62 | 0 |

两次均退出 0，奖励/EMA/观测断言持续通过，未发现接触溢出或非有限数错误。成功数为零符合尚未正式训练的状态，不能包装为策略有效性结果。早期评估发现 checkpoint RNG 会覆盖指定 seed，已修正并补测试；这里只记录修正后 `eval_final_seed43/44` 的结果。全部本次仿真测试进程已退出。

## 新鲜轨迹上的模型误差

下面是完整更新后另外采集的 32 个真实步；没有先把这些数据训练给 WM。误差为各分量均方根，姿态为三维旋转向量分量 RMS，并非直接的测地角均值。

| 推演长度 | 未跨 reset 的环境数 | 物体位置分量 RMS | 姿态分量 RMS |
|---|---:|---:|---:|
| 1 步 | 4096 | 2.63 mm | 0.0278 rad |
| 8 步 | 4065 | 5.90 mm | 0.0925 rad |
| 16 步 | 4047 | 7.28 mm | 0.0841 rad |
| 32 步 | 4042 | 14.57 mm | 0.1590 rad |

仅十次 WM 优化后的长程误差依然明显；部分物体预测项目差于“保持初始状态”的基线。结果说明应先进行正式 WM 训练，不能据此宣称 32 步预测已经可靠。完整六类误差、基线和有效样本数见 JSON。

## 证据

- [4096 更新指标](evidence/supermimic_20261024/full4096_metrics.jsonl)
- [重置和接口核对](evidence/supermimic_20261024/full4096_validation.json)
- [整卡显存采样](evidence/supermimic_20261024/full4096_gpu.json)
- [新鲜轨迹误差和基线](evidence/supermimic_20261024/full4096_holdout.json)
- [最终单元测试输出](evidence/supermimic_20261024/unit_tests.txt)
- [恢复检查](evidence/supermimic_20261024/resume_validation.json)
- [最终源码 SHA-256](evidence/supermimic_20261024/source_sha256.json)
- [独立 seed 43 评估](evidence/supermimic_20261024/eval_seed43.json) / [seed 44 评估](evidence/supermimic_20261024/eval_seed44.json)

远端完整诊断目录：`outputs/full4096_v2/`；源日志 `outputs/full4096_v2.log`。短测权重、优化器和元数据已保存，但不替换原教师、学生或预测器权重。

尚未完成的效果验证：充分 WM 预训练后的多种子成功率、强接触分布覆盖、模型与真实 Newton 动作扰动方向对照、CNN 相机闭环。模型内部有限差分测试不等同于真实物理导数验证。
