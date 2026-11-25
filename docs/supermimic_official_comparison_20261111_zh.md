# SuperMimic 官方开源与既有复刻实现对照

核查日期：2026-11-11。本次为源码与实际配置审计，没有启动训练、修改算法或复跑性能实验。

## 1. 比较对象与结论

本次区分三份实现，避免把第三方复刻、本地任务适配和官方默认配置混为一谈：

1. 官方 `HybridRobotics/supermimic`：提交 [`5efbf114e11f725c31d83e217be987c13c0e4ce2`](https://github.com/HybridRobotics/supermimic/tree/5efbf114e11f725c31d83e217be987c13c0e4ce2)。本次完整拉取并逐项阅读核心训练源码。
2. 之前参考的第三方 `zita-ch/supermimic_vibed`：固定提交 [`a1508d42c9d1699a52e8233ab16d08c488441ab9`](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9)。本地旧版说明明确记录采用该提交；本次通过 GitHub 读取其网络、训练、MDP、回放与 README。
3. 本地 Revo3 SuperMimic V33：以已保存的 [实际运行配置](/home/brainco/reorientation/training/runs/die58_v33_no_target_home_resume500_novideo_20261108_115639/config.json) 和 [冻结源码目录](/home/brainco/reorientation/training/diagnostics/die58_v33_no_target_home_2048x2000_30000_20261108_023654/frozen_source) 为稳定参照，同时检查当前工作树。工作树包含 V35/V36 DMO、V37/V38 PWM 等后续可选分支，不能把它们全部算作 V33。

结论：世界模型监督学习、可微想象更新 Actor、想象 λ-return 拟合 Critic 的主流程相同，但官方与复刻在动力学输出结构、损失尺度、优化器、价值约束和默认观测上存在实质差异。既有实现可以称为基于论文及第三方代码的 Revo3 适配，不能称为官方代码的等价实现。源码差异本身不能证明某个差异就是此前训练失败的原因。

## 2. 最重要的差异：动力学预测结构

| 实现 | 网络预测内容 | 位姿如何更新 |
|---|---|---|
| 官方 GenCoord | 根线速度增量、根角速度增量、关节速度增量；29 关节 G1 合计 35 维 | 位置、四元数、关节角通过速度积分获得，重力方向等由姿态推导 |
| 官方 MaxCoord | 每个跟踪刚体的线速度、角速度增量 | 每个刚体的位置和旋转通过积分更新，随后转换到新的根坐标系 |
| 第三方复刻 | 根位置、旋转、线速度、角速度、关节角、关节速度的独立增量；G1 合计 70 维 | 各增量独立相加／旋转复合 |
| 本地 V33 | 关节角21、关节速度21、物体位置3、旋转向量3、线速度3、角速度3，共54维 | 各增量独立相加／旋转复合 |

官方 GenCoord 的关键关系为：

```text
v_next = v + delta_v
q_next = q + dt * (v + c * delta_v)
c = (decimation + 1) / (2 * decimation)
```

根位姿也采用相应积分，姿态使用四元数积分。这里的网络输出是控制步内速度增量，并非直接输出加速度数值。

这为预测的位置和速度加入了明确的一致性约束。V33 的独立增量允许出现位置几乎不动但预测速度很大、或者预测旋转与角速度不一致的状态。该结构差异值得优先做受控对照；它并不自动保证接触动力学或动作 Jacobian 准确。MaxCoord 也不能因此被描述为自动满足所有关节链约束。

当前工作树的后续 `wm_rotation_coupled` 分支已尝试“角速度梯形积分＋学习残差”，但 V33 冻结源码没有此结构，且该残差方案也不等价于官方全部位置／速度的积分约束。

证据：[官方表示与积分](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/representations.py#L469)、[第三方网络](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/networks.py)、[V33 动力学预测](/home/brainco/reorientation/training/diagnostics/die58_v33_no_target_home_2048x2000_30000_20261108_023654/frozen_source/source/viserdex/viserdex/supermimic/networks.py:168)。

## 3. 世界模型监督与归一化

官方对输入状态与处理后的动作做经验归一化；域参数按随机化区间中心和半宽缩放。动力学网络直接输出物理速度增量，没有第三方／V33 那种输出增量 running mean/std 反归一化。

官方 WM 损失采用固定 nominal error 的 Bryson 权重，即平方误差除以固定容许误差的平方，再按通道数聚合。默认 GenCoord 尺度为：根位置 0.01 m、根旋转 0.02 rad、关节角 0.1 rad、关节速度 3.14 rad/s。根旋转使用旋转矩阵差，乘以 sqrt(1/2) 后计入误差；根速度没有额外直接监督项，而是通过多步积分位姿约束。MaxCoord 另监督非根刚体的位置、旋转6D、线速度和角速度。

第三方及 V33 则使用 `max(真实增量标准差, 物理 floor)` 作为误差分母，并用真实增量统计反归一化网络输出。V33 的 floor 依次为 0.003 rad、0.05 rad/s、0.003 m、0.005 rad、0.02 m/s、0.05 rad/s。这些 floor 不是实际始终采用的分母，不能直接据此声称损失权重相差某个固定倍数。

V33 还加入：解析 reward／干净 observation／cost 的一致性损失、22通道手物接触 BCE，以及独立优化器训练的失败分类器。官方当前 WM learner 只拟合物理状态误差，没有这些辅助损失。官方奖励和终止由状态解析计算，终止概率是高度／倾斜阈值的 sigmoid 平滑。

证据：[官方 WM learner](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/learners/wm_learner.py)、[官方终止公式](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/envs/world_model_env.py)、[V33 误差分母](/home/brainco/reorientation/training/diagnostics/die58_v33_no_target_home_2048x2000_30000_20261108_023654/frozen_source/source/viserdex/viserdex/supermimic/trainer.py:225)。

## 4. 网络与优化器

| 项目 | 官方当前实现 | 第三方复刻 | 本地 V33 |
|---|---|---|---|
| WM 隐藏层 | 2048×3，LayerNorm＋GELU | 2048×3，LayerNorm＋SiLU | 2048×3，LayerNorm＋SiLU |
| Actor | 每任务目标编码器＋共享解码器；各为2048→1024→512→256；latent32，tanh | 单运动编码器＋解码器，相同层宽；ELU | 225→1024→512→256→21；ELU，最终动作 tanh |
| Critic | 256→128→1；LayerNorm＋GELU | 256→128→1；ELU，独立 running norm | 512→256→128→6；ELU，独立 running norm |
| 优化器 | 各 Sequential 内部隐藏 Linear 权重用 Muon，其余参数用 fused/capturable Adam | 全部 Adam | 全部 Adam；失败头独立 Adam |
| 学习率 | 默认2e-4；当前训练循环未设置余弦退火或 Actor ramp | 2e-4，250k轮余弦至3% | 同类余弦退火；Actor另乘0.25并有ramp |

官方的 tanh 位于 latent，而解码器输出本身没有 V33 的最终动作 tanh。不能把两边的原始 action 直接当作同一控制量比较。官方 Dynamics 内部执行 `action * scale + offset`；V33 将 tanh 动作映射到 Revo3 软限位，并经过编码器偏置转换、EMA 和所选配置的目标速率限制。

官方已支持 motion／velocity 多目标编码器以及向 motion latent 对齐的辅助 BC；默认 slot mix 为 motion=1、velocity=0，因此默认训练并没有实际启用 velocity 分支。V33 的目标姿态直接放入观测，没有独立目标编码器。这部分属于任务与策略接口差异，不宜简单判断为本地遗漏。

证据：[官方模型](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/models.py)、[官方优化器](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/learners/learner_utils.py#L17)、[V33网络说明](revo3_supermimic_v33_zh.md)。

## 5. Critic：保留相同主流程，但官方增加价值上界

三者都使用可微想象回报和 Critic 尾值更新 Actor，并用固定的想象 λ-return／GAE return 做 Critic MSE 拟合；默认每轮5 epochs×4 minibatches。官方当前主路径没有 target Critic／EMA，也不是 SAC。

官方新增的关键约束是：

```text
V_max = max_step_reward / (1 - gamma)
Actor尾值 = min(V(s_H), V_max)
Critic训练目标 = min(GAE_return, V_max)
```

其中 `max_step_reward` 从环境正奖励项权重之和乘以控制 dt 得到。只限制上界，不是将网络整体改为 sigmoid，也不限制下界。

第三方复刻没有上述上界；V33 的 `paper_critic=True` 明确返回无界线性六头值，想象尾值和 λ-return 也没有对应上限。V29/V30 的 sigmoid／target Critic 是另一路历史实现，不应误算到 V33。

这一差异与此前观测到的 bootstrap 主导值得关联检查，但只能提出假设：cap 可以抑制超过理论上界的乐观尾值；如果当前值仍低于 cap，局部状态梯度仍可能很大。不能声称加 cap 就能解决 V33 的梯度问题。迁移时还必须分别推导奖励头与各代价头的合法界，不能把人形任务的数值复制到手部。

证据：[官方尾值及目标裁剪](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/learners/policy_learner.py#L161)、[V33线性Critic](/home/brainco/reorientation/training/diagnostics/die58_v33_no_target_home_2048x2000_30000_20261108_023654/frozen_source/source/viserdex/viserdex/supermimic/networks.py:247)、[V33尾值与目标](/home/brainco/reorientation/training/diagnostics/die58_v33_no_target_home_2048x2000_30000_20261108_023654/frozen_source/source/viserdex/viserdex/supermimic/stable_values.py:247)。

## 6. 时序、采样与默认配置

| 项目 | 官方当前默认 | 第三方固定提交 | 本地选定V33运行 |
|---|---|---|---|
| Actor观测历史 | 1帧，`actor_obs=rep_state`；可切换proprio及多帧 | 10帧带噪proprio | 3×75=225维，包含视觉物体位姿信息 |
| 目标 | 10个参考帧，stride5，约0.9秒前视 | 10个未来参考帧 | 当前物体目标姿态／位置信息 |
| WM训练长度 | `horizon=8`表示8个状态，实际7次预测 | 8条transition，8次预测 | 8条transition，8次预测 |
| Actor想象长度 | 32步 | 32步 | 8步 |
| WM更新 | 10×batch2048 | 相同 | 相同 |
| Actor起点batch | 4096 | 相同 | 相同 |
| γ／λ | 0.99／0.95 | 0.99／0.95 | 0.99／0.9259454628 |
| 回放容量 | 1000向量步 | 1000向量步 | 2000向量步 |
| 更新／采集比例 | 从env数、WM batch/window和replay_ratio16计算 | 固定UTD0.4 | 固定UTD0.4 |
| 预热 | buffer满足窗口后开始联合更新；无500轮WM-only阶段 | 先64步采样，再500轮WM-only | 此配置为恢复训练，WM-only=0；不能据此推断所有V33从头训练无预热 |

官方在4096环境、10×2048×8、replay_ratio16下计算出的更新／向量采样步比例也是0.4；改为2048环境而保留其余默认值时为0.2。本地0.4是显式固定值。官方外层 `max_iterations` 计数仿真向量步，本地外层计数网络训练迭代，因此不能直接比较相同 iteration 的预算。

官方 Replay 按环境分层均匀采有效窗口，不跨终止／截断边界；不保存显式 terminal next state 用于跨向终止状态的动力学监督。第三方和V33保存真实 next state，窗口中终止 transition 本身可监督，终止后的部分mask。V33另启用按预测误差的 prioritized replay；这和官方按低回报调整真实运动片段 reset 分布是两个不同机制。第三方人形复刻已经实现了类似 return-based reset sampling，并非官方才有。

证据：[官方Runner默认值及循环](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/runners/supermimic_runner.py)、[官方Replay](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/replay_buffer.py)、[第三方训练](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/train.py)。

## 7. 任务差异与运行条件

官方目前注册 G1 GenCoord／MaxCoord 运动跟踪任务，默认数据入口是 W&B 的 `lafan_run1_subject5`，也允许本地 motion 文件。没有现成的 Revo3 手内旋转任务、Revo3资产或本地视觉／硬件控制协议。

官方与第三方在人形跟踪奖励的大结构上很接近：六类指数跟踪项、action-rate惩罚和joint-limit惩罚。V33改成手内重定向与稳定保持奖励，并加入多项约束代价、接触监督、外扰、编码器偏置和EMA控制。它们应视为任务适配和本地算法扩展，不能机械地全部删除以“对齐官方”。

官方工程使用 Python3.13 和 mjlab；`pyproject.toml` 将 mjlab 指向兄弟目录 `../mjlab_research`，README要求克隆 `kevinzakka/mjlab_research`。本次通过 GitHub 连接访问该依赖返回404；这表示当前访问路径不可用，不足以判断仓库究竟私有、移动还是不存在。未验证W&B数据权限。不能把公开主仓库等同于完整可运行环境。

官方将 WM、Actor／Critic 更新纳入 `torch.compile` 和 CUDA graph；本地V33具有可选 WM CUDA graph，但不等同于官方完整训练更新捕获。第三方默认 MLP BF16，所选V33配置为FP32；官方当前源码没有第三方式BF16 autocast路径。

官方保存 WM、Actor和Critic参数bundle，但当前Runner保存流程没有保存Adam/Muon优化器、Replay和RNG的完整续训状态；本地checkpoint保存优化器与RNG等更多训练状态，虽然也不保存仿真／Replay以实现逐位连续恢复。不能笼统认为官方所有工程能力都更完整。

证据：[官方README](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/README.md)、[依赖配置](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/pyproject.toml)、[任务配置](https://github.com/HybridRobotics/supermimic/blob/5efbf114e11f725c31d83e217be987c13c0e4ce2/src/supermimic/tasks/env_cfgs.py)。

## 8. 建议的后续对照顺序

1. 固定现有Revo3资产、奖励、控制合同和真实数据，比较独立54维增量与速度增量＋解析积分；同时检查开环多步误差、位姿／速度一致性、正负微扰动作的真实响应方向。
2. 单独比较现有动态std损失尺度与固定nominal error尺度；采用适合手部的物理量尺度，不照搬G1值。
3. 单独加入有理论依据的价值上界，记录cap命中率、真实回报校准和bootstrap／短程奖励梯度比。若不命中cap，不能期待它必然改善梯度。
4. 在前述对照稳定后，再评估GELU／LayerNorm、Muon＋Adam及完整CUDA graph的学习和吞吐影响。

不要在同一轮同时改模型输出、损失、优化器、奖励与仿真引擎，否则无法解释收益或回退原因。官方G1默认H32也不构成手部接触模型应立即恢复H32的证据。

本次没有运行两边训练或声称哪一边速度／成功率更高；所有收益判断均需同任务、同数据预算的实验支持。
