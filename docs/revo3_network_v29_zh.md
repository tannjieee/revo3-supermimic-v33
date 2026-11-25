# Revo3 网络与控制协议 v29：有界代价、真实 TD 与目标 Critic

日期：2026-11-05。v28 在第 2114 轮后退出；诊断确认代价 Critic 的估值超过理论上界，
Actor loss 被错误的长期代价值主导。v29 修复训练语义，不继承失真的 v28 Actor/Critic。
已通过 399 项回归测试，以及 64 环境、80 轮完整联合训练和两轮检查点恢复验证。
这验证了实现与数值边界，尚不能证明任务收敛；验证记录见文末。

Actor、WM、关节顺序、物理限位、100 rpm / 1.28 N·m、25 Hz 控制、200 Hz PD、
400 Hz 求解子步、手中坐标系、物体姿态奖励及所有随机化均沿用既有合同。
`stable_value_learning=True` 的检查点版本是 29；旧配置缺少该字段时为 False，仍按
v25–v28 原语义加载。不能用 `--resume` 把 v28 静默改成 v29。

## 1. 真机下发顺序与编码器限位

手指顺序：**小指、无名指、中指、食指、拇指**。四个非拇指是 MPR/MCP/PIP/DIP；拇指明确为 **MCP/PIP/DIP/CMP/CMR**，不是按运动学树近端到远端排列。

所有关节相关张量采用该顺序：动作、q/qd、编码器偏置、限位、HOME、Kp/Kd、物理监督输出。资产与辨识源文件保留原始顺序，通过名称映射；FK 审计也映射，不改 USD/URDF 本体。

| 索引 | 关节 | URDF 几何范围（°） | 编码器范围（°） |
|---:|---|---:|---:|
| 0 | `right_little_MPR_joint` | -30 ～ 10 | -30 ～ 10 |
| 1 | `right_little_MCP_joint` | -5 ～ 90 | -5 ～ 90 |
| 2 | `right_little_PIP_joint` | -10 ～ 90 | -10 ～ 90 |
| 3 | `right_little_DIP_joint` | -20 ～ 90 | -20 ～ 90 |
| 4 | `right_ring_MPR_joint` | -25 ～ 20 | -25 ～ 20 |
| 5 | `right_ring_MCP_joint` | -5 ～ 90 | -5 ～ 90 |
| 6 | `right_ring_PIP_joint` | -10 ～ 90 | -10 ～ 90 |
| 7 | `right_ring_DIP_joint` | -20 ～ 90 | -20 ～ 90 |
| 8 | `right_middle_MPR_joint` | -20 ～ 25 | -20 ～ 25 |
| 9 | `right_middle_MCP_joint` | -5 ～ 90 | -5 ～ 90 |
| 10 | `right_middle_PIP_joint` | -10 ～ 90 | -10 ～ 90 |
| 11 | `right_middle_DIP_joint` | -20 ～ 90 | -20 ～ 90 |
| 12 | `right_index_MPR_joint` | -10 ～ 30 | -10 ～ 30 |
| 13 | `right_index_MCP_joint` | -5 ～ 90 | -5 ～ 90 |
| 14 | `right_index_PIP_joint` | -10 ～ 90 | -10 ～ 90 |
| 15 | `right_index_DIP_joint` | -20 ～ 90 | -20 ～ 90 |
| 16 | `right_thumb_MCP_joint` | 0 ～ 75 | 0 ～ 75 |
| 17 | `right_thumb_PIP_joint` | -10 ～ 90 | -10 ～ 90 |
| 18 | `right_thumb_DIP_joint` | -20 ～ 90 | -20 ～ 90 |
| 19 | `right_thumb_CMP_joint` | 0 ～ 110 | -1 ～ 109 |
| 20 | `right_thumb_CMR_joint` | 0 ～ 106 | 0 ～ 106 |

限位来源为 BrainCoTech/brainco-description dev `2a98621` 的 URDF 快照。物理目标满足 `q_geometry=q_encoder+bias`，因此编码器限位按 `URDF_limit−bias` 计算。CMP 位于索引 19；它采用辨识模型里的 **固定 1° 偏置假设**，不是单独标定得到的零点。该假设在来源 provenance 中有明确说明。由此 CMP 几何范围 0°～110°对应编码器 −1°～109°。

## 2. 25 Hz、PD 与仿真频率

| 层级 | 周期 | 频率 | 对齐关系 |
|---|---:|---:|---|
| Actor 控制 | 40 ms | 25 Hz | 每次命令持续 8 个 PD tick |
| 执行器 PD | 5 ms | 200 Hz | 保留辨识模型采样周期 |
| 当前默认仿真积分 | 2.5 ms | 400 Hz | 每个 PD tick 有 2 个 solver 子步 |
| 等效执行器迟延 | 15 ms | — | 3 个 PD tick，短于 40 ms |

官方 NewtonManager 的执行顺序是每个 decimation 循环先更新 actuator，再执行 solver 子步。把 decimation 从 10 改为 8 不会把 PD 频率改成 25 Hz，也不会在每个 2.5 ms 子步重新运行 PD。

本次实时验证读取到 `policy_dt=.04, pd_dt=.005, decimation=8, solver_dt=.0025`。若使用显式绑定的 4 子步数值实验，则积分为 1.25 ms / 800 Hz，PD 仍为 200 Hz。这里没有修改 solver 或辨识参数去迁就控制频率。

本轮运行使用 `newton_mjwarp`，solver iterations=100、子步数=2、`nconmax=640`、`njmax=1024`。重力保持开启，世界重力为 `(0,0,-9.81) m/s²`；手和物体的 `disable_gravity` 均为 false。solver iterations 是每次积分内部的求解迭代预算，不是新的控制频率，也不能把 `400×100` 当成物理采样频率。

时域变化明确如下：

- 20 s 回合对应 500 个策略步；是采样截断，不是学习状态中的倒计时。
- 10 帧历史首末相隔 0.36 s；8 步 WM 监督与 8 步想象各覆盖 0.32 s。
- 至少保持 0.5 s 的成功条件取 13 步，实际判定持续时间为 0.52 s；不能四舍五入成 12 步的 0.48 s。
- γ 仍为 0.99/策略步，未暗中重新调参。因此其近似折扣尺度从旧 20 Hz 下的 5 s 变为 25 Hz 下的 4 s。这与模型显式预测长度是不同概念。

## 3. 手中坐标系与掉落

取手基座坐标系下 `origin=(0.055,-0.010,0.100) m`，对应当前名义初始方块中心。根据环境初始机器人根位置/旋转与物体中心变换得到；它固定在手上，不随物体移动或每回合重采样。

```text
p_hand_center = p_hand_base − origin
R_hand_center = R_hand_base
v_hand_center = v_hand_base
omega_hand_center = omega_hand_base
```

只平移原点，坐标轴与基座相同；姿态、线速度、角速度和位移增量不需额外旋转。Actor、WM、Critic 的物体位置和目标位置都使用相同原点，原点处网络位置特征为 0。

网络前的实际变换是 `(p_base−origin)×10`，即 10 cm 对应 1。居中消除固定偏移，×10 处理米制数值大小，两者并不互相替代。

仿真、Replay 和外部推理 API 仍保留 SI 基座坐标，在进入网络时统一转换；这避免把引擎状态、传感器噪声、目标命令或旧模型的数据误读成另一坐标系。导出的 Actor 保存相同 offset/scale，部署不能再自行多减一次原点。

真实掉落标签基于手中距离。v29 在工作空间内使用学习概率；预测位置已经越过真实掉落边界时，继续概率直接为零：

```text
d = ||p_base−origin||
real_drop = d > 0.3 m
v29_imagined_continuation = where(d > 0.3, 0, 1 − sigmoid(learned_logit / T + b))
v27_legacy_continuation = sigmoid((0.3−d)/0.01)
```

掉落统计和播放统计同步改用此定义。0.3 m 保持原来的距离阈值数量级，现在以手中原点为球心；它是脱离工作空间判定，不是触觉意义上的失去接触检测。

## 4. 网络结构与缩放一致性

| 模块 | 输入 | 隐藏层 | 输出 | 参数数 |
|---|---|---|---|---:|
| Actor | 810 | 1024→512→256 | 21 + tanh | 1,491,989 |
| WM（含终止头） | 动力学 164；终止头 57 | 动力学 2048→2048→2048；终止 64→64 | 54 个物理增量 + 终止 logit | 8,861,495 |
| 在线 Critic | 152 | 512→256→128 | 6 个有界价值输出 | 243,334 |
| 目标 Critic（冻结） | 152 | 512→256→128 | 同在线 Critic；EMA 参数 | 243,334 |

Actor/Critic 使用 ELU；WM 隐藏层使用 LayerNorm+SiLU。没有 CNN/RNN。Critic 共享 `[512,256,128]` 主干，末层依次输出奖励值、关节速度代价值、目标变化速度代价值、物体角速度代价值、物体线速度代价值、掉落代价值；六个输出先经 sigmoid，再分别乘固定回报上界 `[40,100,100,100,100,100]`（当前 γ=0.99、每步奖励上界 0.4）。训练使用 logits 的软标签 BCE，避免输出饱和时监督梯度消失。旧检查点缺少 `object_hold_costs_enabled` 时仍采用原三头，不把旧末层权重当作新六头权重加载。

### 4.1 同一物理量采用同一尺度

| 变量 | 所有相关网络采用的变换 |
|---|---|
| 编码器位置 | `2(q_encoder−lo)/(hi−lo)−1` |
| 真实几何 q | 先减 encoder bias，再用同一上下限变换 |
| 绝对编码器目标 | 同一限位归一化 |
| 关节速度 | ×0.1 |
| 物体位置、目标位置 | `(p_base−origin)×10` |
| 旋转 6D、上次归一化动作 | 保持自然尺度 |
| 物体线速度 | ×2 |
| 物体角速度 | ×0.1 |
| 域上下文 | 逐维除以真实环境平均绝对值，下限 1e-12 |

固定物理缩放后再进入 RunningNorm。Actor 看含噪/可能采样保持的观测，WM/Critic 看真实状态，因此三类统计量独立估计；目标 Critic 与在线 Critic 同步真实数据统计，不强制共用同一均值方差；**独立统计不意味着物理尺度不同**。

位置 affine 的定义来自同一个 `input_affine`；Actor 前向、真实统计更新、导出和推理复用它。新增测试用同一干净物理状态比对 Actor/WM/Critic 的共同输入，验证 q、qd、位置等缩放一致。

### 4.2 Actor 观测布局与输出

| 单帧切片 | 内容 |
|---|---|
| 0:21 | 归一化编码器位置，小指优先/真机拇指顺序 |
| 21:42 | 关节速度 ×0.1 |
| 42:63 | 上次实际施加的归一化动作 |
| 63:66 | 物体位置，进入网络时居中缩放 |
| 66:69 | 目标位置，进入网络时居中缩放 |
| 69:75 | 物体旋转 6D |
| 75:81 | 相对旋转 `R_object R_goalᵀ` 的 6D |

四元数 XYZW；rotation 6D 是旋转矩阵前两列按列展平。10 帧从旧到新，reset 重复首帧。Actor 的输入 `[B,10,81]` 展平为 `[B,810]`，输出 21 个 logits，经一次 tanh 得动作。

动作映射仍是绝对编码器目标 `lo+(a+1)/2×(hi−lo)`，不是速度动作。训练探索噪声加在 tanh 前。tanh 与映射限制目标位置范围；下发端没有增加目标速率投影或动作滤波。

### 4.3 WM 与 Critic 输入

WM 输入 164 维：`physical_features57 + 当前目标21 + 上次动作21 + context65`。57 维包括 q21、qd21、p3、物体旋转6D、v3、ω3。目标位姿不直接进入动力学模型；目标通过策略动作影响演化。

Critic 输入 152 维：`physical_features57 + 上次动作21 + 目标位置3 + 相对旋转6D + context65`。

两个网络均无几何特征和 delay_ready 输入。域上下文仍是 21 个运动连杆质量、物体质量、物体摩擦、21 个 Kp、21 个 Kd。质量按 body_names，增益按当前关节顺序。

以下是拼接后的准确布局；WM/Critic 共享前 57 维物理特征，但后续字段位置不同。

| 两网络共用的切片 | 内容 |
|---|---|
| 0:21 | 几何 q 转编码器归一化 |
| 21:42 | qd ×0.1 |
| 42:45 | 物体位置居中后 ×10 |
| 45:51 | 物体旋转 6D |
| 51:54 | 物体线速度 ×2 |
| 54:57 | 物体角速度 ×0.1 |

| 网络 | 后续切片 | 内容 |
|---|---|---|
| WM | 57:78 | 当前绝对编码器目标，按限位归一化 |
| WM | 78:99 | 上次归一化动作 |
| WM | 99:164 | 65 维域上下文 |
| Critic | 57:78 | 上次归一化动作 |
| Critic | 78:81 | 目标位置居中后 ×10 |
| Critic | 81:87 | 相对旋转 6D |
| Critic | 87:152 | 65 维域上下文 |

域上下文内部的 65 维布局如下；这些索引相对于 context 起点，而非整张网络输入。

| context 切片 | 内容 | 单位 |
|---|---|---|
| 0:21 | 21 个运动连杆质量 | kg |
| 21:22 | 物体质量 | kg |
| 22:23 | 物体摩擦系数 | 无量纲 |
| 23:44 | 21 个关节 Kp | 仿真执行器原始参数单位 |
| 44:65 | 21 个关节 Kd | 仿真执行器原始参数单位 |

Actor 不接收 context、物体线/角速度、接触力或几何点云；物体动态信息需要从含噪的历史位姿中推断。Critic 使用仿真真实状态，因此属于训练时的特权信息。

### 4.4 固定缩放、RunningNorm 与 WM 输出归一化

网络输入的完整处理为 `z=(x×scale+offset−running_mean)/sqrt(running_variance+eps²)`；context 先除以启动时计算的逐维平均绝对值。输入 RunningNorm 的 `eps=0.01`，统计从真实仿真采样累计，在计数达到 2e8 后停止更新。想象状态不更新这些统计。

Actor 对每一帧使用同一组 81 维统计，归一化后才展平成 810 维；没有为十个历史槽分别维护十组统计。编码器归一化和 qd 缩放发生在观测构造阶段，Actor 的 affine 只补充位置处理，不能在部署端把 q、qd 再缩放一次。

WM 的输出 `delta_norm` 是另一组 54 维真实增量统计，`eps=0.0001`；反变换为 `delta=network_output×delta_std+delta_mean`。输出是状态变化量，输入是状态/目标/context，两者不能共用统计。Actor 输出和 Critic 值输出不经过此反归一化。

## 5. 再次核对：WM 究竟需要预测什么？

它预测 **当前命令经过 40 ms 后、reset 之前的完整物理状态**。当前采用 54 维增量参数化：

| 输出切片 | 内容 | 重建后的意义 |
|---|---|---|
| 0:21 | Δq，rad | 下一时刻 21 个几何关节角 |
| 21:42 | Δqd，rad/s | 下一时刻实际关节速度 |
| 42:45 | Δp，m | 下一时刻物体中心位置 |
| 45:48 | 旋转向量增量，rad | 下一时刻物体四元数 |
| 48:51 | Δv，m/s | 下一时刻物体线速度 |
| 51:54 | Δω，rad/s | 下一时刻物体角速度 |

输出由 delta_norm 反归一化；平移/关节量相加，姿态用 `Exp(Δr)⊗q`。重建出的物理核心为 55 维。Δ 已是 40 ms 的变化量，不再乘 dt。原点平移不改变 Δp，所以无需新增平移输出维度。

q 和 qd 都需要：下一步 PD 响应依赖关节位置/速度，关节观测和速度代价也需要它们。物体 p/姿态/v/ω 都需要：任务奖励、保持、掉落及下一步动力学依赖这些状态。不能只预测物体姿态而丢掉运动速度。

不需要学习预测的量包括：目标位置/姿态（指令保持）、上次动作（已知赋值）、传感器噪声/延迟状态（按观测过程更新）、域参数（启动后固定）、奖励和五项代价（解析计算）。掉落概率改由学习头预测。本版本不预测触觉、接触力、扭矩或加速度，也不能据此宣称已具备准确的力矩约束。

物理状态/Replay 内部仍为 84 维：55 维物理核心、21 维上次动作、3 维目标位置、4 维目标姿态、1 维 reset 审计标志。最后一维不进入 WM/Critic，但成本在 reset 首步需要区分“没有上一策略目标”，以当前测量编码器位置作为第一步目标变化的参考。

15 ms 等效迟延短于 40 ms，所以一个上次命令覆盖当前步初始的迟延区段；steady-state 不需要多周期命令队列输入。reset 首步没有显式 flag 输入是有损简化，仍应单独评估首步预测误差。

SuperMimic 使用可微世界模型与短想象轨迹上的 BPTT 联合训练策略。本工程选择 54 维增量和上述残差重建；这一具体张量协议应以本地实现为准，不能称为论文要求。[原文](https://arxiv.org/abs/2610.28480)

### 5.1 “积分”在代码中的准确含义

这里的 `integrate_physical` 是一步残差重建，不是网络先预测加速度再用欧拉积分更新速度/位置。网络分别预测 Δq、Δqd、Δp、Δv；没有强制 `Δq=qd×dt` 或 `Δp=v×dt`。这种表示把接触、PD、阻尼和迟延在整个 40 ms 内的效果一起拟合，是否准确由真实轨迹监督和开环审计判断。

54 维内部输出与 55 维物理核心相差一维，是因为姿态增量使用三维旋转向量，而重建姿态使用四元数；不是漏掉了一个物理变量。

| 84 维 Replay 状态切片 | 内容 |
|---|---|
| 0:21 / 21:42 | 几何关节角 / 实际关节速度 |
| 42:45 / 45:49 | 基座系物体位置 / XYZW 四元数 |
| 49:52 / 52:55 | 物体线速度 / 角速度 |
| 55:76 | 上次施加的归一化动作 |
| 76:79 / 79:83 | 基座系目标位置 / 目标 XYZW 四元数 |
| 83:84 | 内部执行器队列是否已有上一命令的标志 |

状态、观测历史和传感器缓存是不同对象：84 维状态没有装入全部 10 帧历史，也没有装入位姿保持、噪声偏置等传感器缓存；Replay 另外保存它们，供想象路径复现同一观测过程。

### 5.2 物理状态监督损失

每次 WM 更新采样 2048 条、每条 8 步的真实序列。第一步使用真实起点，之后把预测状态继续送入 WM；使用记录的真实动作，不在中间用真实物理状态替换预测状态。遇到 episode 边界后屏蔽后续项，终止那一步的 reset 前状态仍可监督，避免把 reset 跳变拟合成动力学。

预测误差按 q、qd、p、旋转向量、v、ω 分六组。每组先按真实增量标准差和下表下限中的较大值缩放，再对组内平方误差取平均；最后对六组等权平均，避免 21 维关节量仅凭维数压过三维物体量。

| 误差组 | 缩放分母下限 |
|---|---:|
| 关节位置 | 0.003 rad |
| 关节速度 | 0.05 rad/s |
| 物体位置 | 0.003 m |
| 物体旋转残差 | 0.005 rad |
| 物体线速度 | 0.02 m/s |
| 物体角速度 | 0.05 rad/s |

当前自适应 Replay 对 WM 序列做优先采样，并使用重要性权重修正训练损失。反向传播的损失累加 8 步；日志 `wm_loss` 额外除以 8。六个 `wm_loss_*` 组指标不乘采样重要性权重，因此不能直接把它们平均后当成同一个 `wm_loss`。每轮执行 10 次 WM 更新，当前日志记录的是第 10 次更新的损失，不是十次均值。

### 5.3 新增任务损失与终止头

当前总损失为 `L_s + L_r + L_o + L_d + L_c`，逐步乘有效 mask 和采样重要性权重，
在 8 步上累加。`L_s` 保持上一节；新增各权重为 1，配置可追溯。

| 损失 | 定义 |
|---|---|
| L_r | 预测状态的解析奖励 vs 实际仿真 reward，除以每步上界 0.4 后平方 |
| L_o | 预测/真实下一物理状态构建干净观测，采用 Actor 固定物理 affine 后的均方差 |
| L_d | 57→64→64→1 终止头输出 logit，与真实 terminated 做 BCE |
| L_c | 五项解析预测/真实代价经 log1p 后的均方差；不修改 Actor 代价值 |

无噪声观测和真实代价均来自 reset 前物理真值，未对 noisy observation 学习。
20 秒 timeout 不是掉落正标签；掉落这一 transition 被保留，之后的序列被 mask。
新增目标将任务误差的梯度传给物理预测；奖励/观测/代价没有额外学习型网络。

想象存活采用 `1-sigmoid(logit/T+b)`；T=1、b=0 为未后处理的学习概率。
终止头与 WM 一起更新，Actor 更新时冻结其参数但保留输入梯度。
原人工距离 sigmoid 只留作 v27 路径及审计基线；五项代价的继续系数仍为 1。

新增 TensorBoard 项为 `world_model/wm_loss_reward`、`wm_loss_observation`、
`wm_loss_termination`、`wm_loss_cost` 和五个 `wm_loss_cost_*`。
未来数据审计报告原始/校准/解析对照概率的 NLL、Brier、ECE、类别计数和召回率。
冻结模型后可独立拟合 T/b，用不同环境分组选择并测试；数据不足时拒绝校准。
继续拟合 WM 会清除旧校准参数。细节与用法见 [WM v28 说明](revo3_wm_v28.md)。

## 6. 代价、终止语义与 Actor 目标

### 6.1 保留原始物理违反

实际关节速度阈值为 90°/s，目标变化速度阈值为 60°/s；这些是训练阈值，
不是 100 rpm 的硬件声明上限。辨识文件的 2 rad/s 是激励轨迹的预设约束，
不能作为测得的硬件极限。

原始代价仍由 `violation_costs` 计算并写日志：

```text
c_speed = ReLU(max_j |qd_j| / (pi/2) - 1)^2
c_rate = ReLU(max_j |target_j - previous_j| / (dt*pi/3) - 1)^2
g = exp(-(orientation_error/(pi/12))^2)
c_omega = g * ReLU(norm(object_omega)/0.5 - 1)^2
c_linear = g * ReLU(norm(object_velocity)/0.05 - 1)^2
c_drop_distance = ReLU(distance_from_hand_center/0.3 - 1)^2
```

首次指令没有前一控制目标时，以当前编码器角作为目标变化速度的参照。
WM 的 L_c 继续使用这些原始代价的 log1p 一致性误差，帮助拟合物理预测。

### 6.2 有界风险与一次终止代价

为防止无上界的平方代价与自举共同放大，Actor/Critic 的训练代价改为：

```text
b_i = c_i/(1+c_i), i = speed, rate, omega, linear
b_drop = 0
p = real terminated label                  # 真实 Critic TD
p = 1 - imagined_continuation              # 可微 Actor rollout
C = 1/(1-gamma)                            # 当前为 100
risk_i = (1-p)*b_i + p*C                   # 所有五个代价头
continuation_i = 1-p                       # 奖励/代价一致
```

`c/(1+c)` 保留零违反区域和单调梯度，但压缩大违反的相对权重；这是明确的目标变化，
不是只调整日志缩放。阈值和原始超限统计不变。

**每个代价头都包含共同的掉落失败风险**，不再是纯粹的某一种速度代价。
掉落时一次支付 C，表示后续处在最坏约束状态的折扣成本；随后停止自举和预测状态推进。
这样没有把掉落后的未知物理轨迹当作训练依据，也不会因终止删除未来代价而奖励主动掉落。
单独的 drop 头学习折扣失败风险；距离越界原始代价仍保留为 WM 监督和诊断。

对任意 `b in [0,1]`、`V_next in [0,C]`：

```text
Q_i = (1-p)*b_i + p*C + gamma*(1-p)*V_next_i
0 <= Q_i <= C
∂Q_i/∂p = C - b_i - gamma*V_next_i >= 0
```

因此仅提高失败概率不能降低该头的代价。这里的共同失败处罚比 v28 更强，必须通过
真实姿态成功率、保持率和速度超限率评估是否过于保守，不能只看 loss 已经有界。

### 6.3 可微 Actor 的 8 步回报

`alive_0=1`，`alive_(t+1)=alive_t*(1-p_t)`；奖励、运行代价、终止代价、末端值
都使用同一存活权重。已知硬掉落样本冻结在最后一个有效状态，此后贡献为零。

```text
J_reward = sum_t gamma^t * alive_t * reward_t
           + gamma^H * alive_H * target_V_reward(s_H)
J_cost_i = sum_t gamma^t * alive_t * risk_(t,i)
           + gamma^H * alive_H * target_V_cost_i(s_H)
L_actor = -(1-gamma)*mean(J_reward)/reward_step_max
          + cost_penalty * sum_i (1-gamma)*mean(J_cost_i)
```

当前 H=8、gamma=0.99、cost_penalty=1，每步奖励上界 0.4。
每项归一化代价回报位于 `[0,1]`，总 Actor loss 位于 `[-1,5]`（浮点误差除外）。
这个范围来自目标和价值参数化，而非对最终 loss 做 clamp。
Actor 更新冻结 WM、在线/目标 Critic 的参数，但保留经过 WM 与目标 Critic 输入的梯度。
原始 Actor 梯度范数仍在裁剪前记录；实际全局范数上限为 1。

## 7. 真实 TD、目标 Critic、预热与检查点

在线 Critic 只使用均匀抽取的真实单步转移训练，全部六个头都改成真实 TD。
没有将想象状态作为 Critic 的监督输入。目标网络更新前先固定本轮 TD 标签：

```text
s_next = advance(s, recorded_pre_reset_physical, recorded_action, same_goal)
y_reward = recorded_reward + gamma*(1-terminated)*target_V_reward(s_next)
y_cost_i = risk_i + gamma*(1-terminated)*target_V_cost_i(s_next)
unit_target = y / [reward_step_max/(1-gamma), C, C, C, C, C]
L_critic = sum_heads BCEWithLogits(logits, unit_target)
```

真实 `terminated` 决定失败；`done=True, terminated=False` 的 time limit 继续在
reset 前下一状态上 bootstrap。终止样本不向 Critic 输入失败后的下一状态；标签不使用
WM 预测、学习终止概率或 reset 后观测。超出数学边界的 TD 标签报错，仅容忍浮点尾差。
Critic loss 使用固定回报尺度，不再按每批目标标准差缩放，另记录逐头归一化 TD MAE、
均值和最大估值。BCE 的软标签只是有界均值回归，不表示速度代价变成独立 Bernoulli 事件。

每轮 Critic 完成 5 epochs × 4 minibatches 后，目标参数执行一次
`target = 0.98*target + 0.02*online`；归一化 buffers 从同一真实统计直接同步。
目标网络不进入优化器，随检查点严格保存/恢复，也参与多进程一致性检查。

前 500 轮为 `wm_critic_warmup`：WM 和真实 TD Critic 更新，Actor 冻结。
之后每轮先拟合真实 Critic，再更新一次 Actor。Actor 学习率仍在 1000 轮内渐增；
每轮 WM 更新 10 次。`lam` 为兼容旧配置保留，在 v29 只参与返回的想象诊断目标，
不参与实际 Critic 拟合。旧 v28 的 λ-return 拟合路径完整保留给历史检查点。

新训练默认启用 `--stable_value_learning`。历史 v28 可以用原语义加载；
更换版本、gamma 或代价语义必须新开训练，不能静默继承 v28 的失真 Critic 和 Adam 状态。

## 8. 观测关节延迟是否必须？

**不必须。** 新训练将 `joint_delay_max_steps=0`，保留关节测量噪声与偏置。旧 v25/v26 检查点继续使用保存的原噪声配置。

应区分：

1. 辨识的总等效响应迟延约 12.984 ms，模型按 5 ms tick 量化为 15 ms；不能仅凭它推断单独的传感器读回延迟。
2. 额外观测迟延是训练人为加入的鲁棒性扰动。没有独立测量分布时，再叠加 0/40 ms 可能重复计入迟延，并且离散粒度过粗。
3. 若以后获得实测通信/编码器时间戳，应建立带时间戳的高频缓存，按毫秒分布读取历史状态。两帧线性插值可以作为近似分数延迟，但也引入平滑，不等同于真实异步采样；需要标明近似范围。

位姿传感器的更新周期、保持、抖动和噪声仍保留；本次仅取消额外的关节观测延迟，不是关闭所有传感器随机化。

当前 `noise_config` 的具体含义：

| 配置 | 当前值与作用 |
|---|---|
| `joint_noise_std` / `joint_bias_std` | 0.005 / 0.01；作用于归一化后的关节位置，不是 rad。前者是逐次高斯噪声，后者是 reset 时采样的高斯偏置 |
| `joint_delay_max_steps` | 0；关节位置/速度不附加整控制步延迟。当前关节速度没有单独的加性噪声项 |
| `max_period` | 位姿的位置/旋转各自采样 1–3 个控制步的更新周期，即 40/80/120 ms；中间保持旧观测 |
| `jitter_probability` | reset 时每个通道采样 0–0.1 的跳过更新概率 |
| `failure_probability` | reset 时每个通道采样 0–0.3 的异常测量概率，并非每个环境恒定 30% |
| `position_noise` / `position_bias` | 每轴均匀 ±0.012 m / reset 时每轴均匀 ±0.012 m |
| `rotation_noise_deg` / `rotation_bias_deg` | 旋转向量各分量均匀 ±1° / reset 时均匀 ±0.1° |
| `failure_position_extent` | 异常位置替换为基座系每轴 ±0.15 m 内的随机值；异常旋转用随机单位四元数 |

位姿测量周期可大于 40 ms，与 15 ms 执行器迟延是两件事；因此“取消关节观测延迟”不等于 Actor 总能看到最新物体位姿。这些数值是当前训练随机化配置，不能全部当成实测传感器标定结果。

## 9. 日志与验证

真实 `cost_*_sum / task_samples` 仍表示原始物理违反；它与新的有界风险回报是不同统计量。

| 指标 | 含义 |
|---|---|
| `imagined_cost_<name>` | 存活加权后的归一化风险回报，含共同失败风险和 bootstrap；范围 0–1 |
| `actor_cost_running_<name>` | 8 步运行风险部分 |
| `actor_cost_terminal_<name>` | 8 步内一次失败代价的期望部分 |
| `actor_cost_bootstrap_<name>` | 末端目标 Critic 部分 |
| `actor_reward_rollout` / `actor_reward_bootstrap` | 对 Actor loss 的两项正奖励贡献 |
| `actor_cost_bootstrap_fraction` | 总代价中来自末端估值的比例 |
| `critic_value_mean/max_<name>` | 固定上界归一化的估值，范围 0–1 |
| `critic_target_mean_<name>` / `critic_td_mae_<name>` | 真实 TD 标签及归一化绝对误差 |
| `critic_real_timeout_fraction` / `critic_real_terminated_fraction` | 抽到的真实超时/终止样本比例 |

数学有界不能证明策略已收敛，也不能替代真实速度限制或机械越限保护。
用户此前要求的 100 rpm 和 1.28 N·m 声明上限不代表任意碰撞冲量下都能硬限速。
v28 的食指机械越界记录保留；本次没有修改碰撞体、求解器、力矩或健康阈值。

验证证据已固化于 `docs/versions/v29_20261105/validation.json`。

64 环境验证使用完整网络、正式 batch 和 CUDA Graph，80 轮完成 WM/Actor/Critic
800/80/1600 次更新；Actor loss 0.1570–0.5154，末轮梯度范数 0.3107，
末端代价占比 34.24%。正式更新期间包含 6 个真实掉落；另在初始采集中强制超时，
预复位物理状态分离审计通过。接触容量/机械健康均未报错，奖励核对最大差 1.49e-8。
恢复到第 82 轮后更新计数为 820/82/1640；四个网络的初始 digest 与原检查点一致，
确认目标网络也被恢复。真实 v28 Critic strict load 和 v29 推理加载通过。
399 项回归测试包含 CPU/GPU Replay、CUDA Graph/eager 和旧版本合同。
历史完整说明保留于 [v28](revo3_network_v28_zh.md)；先前物理故障不应被解释为已修复。
