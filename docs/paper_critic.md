# 恢复 SuperMimic 式（3）的 Critic 学习（v32）

更新日期：2026-11-06。完整网络、控制与最新实验结果见 [SuperMimic v32 详细说明](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/docs/revo3_supermimic_v32_zh.md)。

依据用户提供的 `2026_SuperMimic_2610.28480v1.md` 第 3.3 节，恢复：

\[
\hat R_H=V_H,\qquad
\hat R_t=\hat r_{t+1}+\gamma(1-\hat d_{t+1})
[(1-\lambda)V_{t+1}+\lambda\hat R_{t+1}],
\qquad L_V=\mathbb E\sum_t(V_t-\hat R_t)^2.
\]

## 变更

- 新绝对位置训练默认 `paper_critic=True`，也可显式传入 `--paper_critic`。
- Critic 只在 WM 想象状态上拟合 λ-return，真实 replay 只提供想象起点；不读取真实 TD 标签。
- 回报由本次更新前的在线 Critic 计算，`R_H=V_H`；目标 detach 后在所有拟合 epoch 中保持固定。
- Actor 的尾值也取同一个在线 Critic；冻结其参数时仍保留对状态/动作的梯度。
- 不创建、更新或保存 EMA target Critic。
- Critic 输出改为线性值，不使用 sigmoid 或 softplus；MSE 使用原回报单位，不使用 BCE 或每头目标标准差缩放。
- 损失按起点取期望、对时间求和；不是把 H 个时间步再平均一次。已知终止后的无效后缀不计入损失。
- λ 现在真实影响 Critic 目标；预热阶段只训练 WM，不再偷偷进行真实 TD Critic 更新。

## 兼容和启动

新检查点版本为 32，包含 `paper_critic=True` 和独立训练协议标识。保存、恢复、Actor 导出/推理接受新版本。
旧检查点缺失该字段时默认 False，仍走原来的真实 TD/BCE/EMA 分支，不能跨协议直接恢复。
恢复时显式请求不同的 `--paper_critic` 值会报错。

现有启动脚本 `/home/brainco/reorientation/training/start_v31.sh` 已显式使用 `--paper_critic`，
输出目录默认带 `papercritic_v32`。脚本文件名暂保留兼容，但新训练算法已经是 v32。
纸面模式不接受 `--critic_td_steps 8`；配置中该占位字段为 1，实际不参与 λ-return 计算。
要单独复现旧算法，应显式传 `--no-paper_critic --critic_td_steps 8`。

旧 TD/BCE/EMA 长训使用独立源码快照。后续已从头启动 v32 实验
`/home/brainco/reorientation/training/runs/die50_47g_papercritic_v32_20261106_032925`，
50 mm / 47 g、1024 环境、前 1000 轮小幅 HOME 激励只训练 WM，计划 10,000 轮。
该实验最后完整记录为第 2552 轮，随后真实采样时食指 DIP 机械越限，已失败退出；不能再将其标为“未启动”或“仍在运行”。

## 忠实性范围

本次恢复的是式（3）的回报递推、Critic 损失和在线价值网络关系。五类风险信号仍是本任务的扩展，
各头使用同样的 λ-return/MSE；任务奖励、风险定义、Actor 正则化、物理与动作协议保持。
当前任务的 H=8 和重采样后的 λ 保留，没有同时改成论文人形跟踪的 H=32 或更换任务奖励。
因此不能将整个灵巧手任务称为原论文的逐项复现，也不能据此宣称 bootstrap 或 WM 误差已经解决。

回归测试覆盖手算递推、终止、λ 的实际影响、线性值、MSE 单位和时间求和、固定目标、
禁止真实 TD/EMA 路径、预热行为、新旧检查点隔离及 Actor 导出。
启动前 86 项相关测试通过；64 环境真实 Newton 预检完成 60 轮，WM/Actor/Critic 更新数为
600/20/400，预热 Actor/Critic 零更新，零掉落，500 帧录像完成。

长训后又完成 model_1000 / model_2500 在 seed 66、67 上的独立评估，每次 64 环境 ×640 步，
关闭动作探索，保留物理/感知随机化。四组均零成功；model_1000 均零掉落，model_2500 各 19 次掉落。
末 200 轮训练中的 bootstrap/短程奖励参数梯度比中位数为 362.81；恢复此 Critic 形式尚未解决 bootstrap 主导。
末次第 2500 轮训练流审计的 8 步旋转 RMSE 为 14.40°，保持初姿态基线为 3.32°；
该审计不是独立反事实动作方向测试。详细统计与口径见
[analysis_2552/summary.json](/home/brainco/reorientation/training/diagnostics/papercritic_v32_long_20261106_032925/analysis_2552/summary.json)。

还需区分：即时 running risk 有界，但 v32 线性 Critic 输出无界。旧有界 Critic 下的预测回报范围、
Actor loss 范围及任意尾值下的失败代价单调性保证不能直接沿用。Critic 原单位 MSE 增大也可能受目标尺度变化影响，
不能仅凭 loss 曲线断言价值校准恶化。
