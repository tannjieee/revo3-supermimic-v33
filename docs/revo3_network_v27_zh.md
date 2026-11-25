# Revo3 网络与控制协议 v27：25 Hz、真机顺序、手中坐标和代价约束

核对日期：2026-11-05。新训练默认使用 `hardware_center25_v3` / checkpoint version 27。旧 v25 第 9300 轮权重不改变；v26/v25 仍按原协议加载。已完成单元/回归测试与 4 环境短验证，并于 2026-11-05 14:39（北京时间）启动 1024 环境、10000 轮的新训练。验证通过和训练启动不代表策略已学会任务或满足速度约束。

**当前运行：2026-11-05 16:21（北京时间）已从头启动 1024 环境、10000 轮训练，运行名 `orientation_costs_epa64_1024_10000_20261105_162115`。** 完成 500 轮 WM 预热，已超过第 760 轮并进入联合训练；日志、检查点和 TensorBoard 正常更新。使用下文的姿态奖励、五项代价、六头 Critic；物体从手中初始化，不使用抓握库。每 100 轮保存检查点。实时状态以 `training/current_job/status.json` 为准，本文数字是启动核对快照，不表示训练已收敛。

此次第一次启动在第 107 轮后触发 MJWarp `EPA_HORIZON` 缓存溢出并退出。已将本地依赖的 EPA 边界边临时容量从 24 扩到 64，记录补丁及哈希到运行合同，再以相同种子重启；未改变碰撞几何、随机化或健康检查。528 组已保存接触的距离对照一致，159 项测试通过，新运行已越过原失败点。详见 [诊断记录](/home/brainco/reorientation/experiments/epa_failure_20261105/README.md) 和 [启动核对](/home/brainco/reorientation/training/jobs/orientation_costs_epa64_1024_10000_20261105_162115/startup_validation.json)。这不构成未来所有接触状态或机械限位稳定性的保证。

**历史运行：14:39 启动的旧任务在完成第 1801 轮后的真实采集中触发机械限位异常，已退出；其 10000 轮尚未完成，最近完整检查点为第 1800 轮。** 第 10 节保留该运行退出前快照和失败现场，不能把其中 `training` 当作当前进程状态。

**后续配置更新（2026-11-05）：新建环境的 21 个关节统一采用 100 rpm（600°/s、10.4719755 rad/s）声明速度上限和 1.28 N·m 电机力矩上限。** 力矩限幅已验证生效；当前 Newton/MJWarp 不执行该速度字段的硬限速。原失败运行和第 1800 轮权重仍对应此前的辨识力矩限幅，不能当作已在新参数下训练过。具体文件、验证和边界见本文末节。

**任务更新（2026-11-05）：新训练取消物体位置跟踪，只要求目标姿态到达、稳定保持且不掉落。** `track_object_position=False` 同时关闭独立位置奖励、稳定奖励中的位置误差项和成功判定的 2 cm 位置门槛。当前物体位置仍供抓握状态观测、WM 动力学和手中坐标系掉落检测使用；目标位置槽位为接口兼容而保留。第 6 节已更新为新目标；第 10 节历史训练数据仍对应旧任务。本次只做短验证，没有启动正式训练。138 项测试通过，4 环境、2 轮短验证完成 WM/Actor/Critic 的 2/2/4 次更新，真实与解析奖励最大差 0，保持计数和成功边界核对一致；见 [验证结果](/home/brainco/reorientation/training/runs/orientation_only_validation_20261105/validation.json)。这不表示策略已经学会新目标。

**最新代价更新：经用户确认，物体角速度、线速度和掉落加入独立代价，原正向稳定奖励关闭。** 新配置 `object_hold_costs_enabled=True`，保留姿态到达奖励与 0.52 秒成功统计；速度代价按姿态接近程度连续加权，掉落代价不使用该权重。Critic 由旧三头扩展为六头（1 奖励 + 5 代价），Actor/WM 不变。上段 138 项测试与短验证是此次代价更新之前的结果；当前结构和公式以第 4、6 节为准。

当前代价方案通过 159 项测试，以及 4 环境、2 轮短验证（WM/Actor/Critic 更新 2/2/4 次）。真实与解析奖励最大差 3.73e-9、保持计数差 0、成功边界不一致 0；五项真实/想象代价和连续权重均有日志。证据见 [短验证配置](/home/brainco/reorientation/training/runs/object_hold_costs_validation_20261105/config.json)、[短验证结果](/home/brainco/reorientation/training/runs/object_hold_costs_validation_20261105/validation.json)。完成上述检查后，已按顶部记录启动正式长训练。

历史文档：[v26](/home/brainco/reorientation/training/网络结构与输入输出详细说明_v26.md)、[v25 / 9300](/home/brainco/reorientation/training/网络结构与输入输出详细说明_v25_9300.md)。本文件取代它们作为新训练配置说明。

本文当前配置依据 [config.json](/home/brainco/reorientation/training/runs/orientation_costs_epa64_1024_10000_20261105_162115/config.json)、[env.yaml](/home/brainco/reorientation/training/runs/orientation_costs_epa64_1024_10000_20261105_162115/env.yaml) 和训练源码。第 10 节对应历史 `manual_inhand_v27_1024_10000_20261105_143940` 运行，数字不会随当前训练自动刷新。

阅读顺序：第 1–3 节定义关节、时间和坐标；第 4–5 节解释网络张量与 WM 监督；第 6–7 节解释奖励、代价和传感器；第 9–11 节记录训练循环、当前快照及日志使用方式。所有切片均为 Python 左闭右开区间，`B` 表示批量大小。

```mermaid
flowchart LR
    S[仿真真实状态与域参数] --> O[传感器模型与 10 帧历史]
    O --> A[Actor: 810 → 21]
    A --> D[绝对编码器目标]
    D --> P[200 Hz PD / 400 Hz 积分]
    P --> R[reset 前状态 → GPU Replay]
    R --> W[8 步 WM 监督学习]
    R --> I[4096 个起点 / 8 步可微想象]
    A --> I
    W --> I
    I --> C[姿态奖励 / 五项代价 / 六值 Critic]
    C --> G[路径梯度更新 Actor]
```

WM 与 Critic 用于训练；实际推理只需要 Actor、保存的归一化统计和观测历史。图中的 WM 箭头表示模型参与想象，Actor 更新时冻结 WM/Critic 参数，但保留对输入的梯度。

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

真实终止与想象继续概率共用手中距离：

```text
d = ||p_base−origin||
real_drop = d > 0.3 m
imagined_continuation = sigmoid((0.3−d)/0.01)
```

掉落统计和播放统计同步改用此定义。0.3 m 保持原来的距离阈值数量级，现在以手中原点为球心；它是脱离工作空间判定，不是触觉意义上的失去接触检测。

## 4. 网络结构与缩放一致性

| 模块 | 输入 | 隐藏层 | 输出 | 参数数 |
|---|---|---|---|---:|
| Actor | 810 | 1024→512→256 | 21 + tanh | 1,491,989 |
| WM | 164 | 2048→2048→2048 | 54 个物理增量 | 8,853,558 |
| Critic | 152 | 512→256→128 | 6 个价值输出 | 243,334 |

Actor/Critic 使用 ELU；WM 隐藏层使用 LayerNorm+SiLU。没有 CNN/RNN。Critic 共享 `[512,256,128]` 主干，末层依次输出奖励值、关节速度代价值、目标变化速度代价值、物体角速度代价值、物体线速度代价值、掉落代价值；五个代价值经 Softplus 保持非负。旧检查点缺少 `object_hold_costs_enabled` 时仍采用原三头，不把旧末层权重当作新六头权重加载。

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

固定物理缩放后再进入 RunningNorm。Actor 看含噪/可能采样保持的观测，WM/Critic 看真实状态，因此统计量独立估计，不强制共用同一均值方差；**独立统计不意味着物理尺度不同**。

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

不需要学习预测的量包括：目标位置/姿态（指令保持）、上次动作（已知赋值）、传感器噪声/延迟状态（按观测过程更新）、域参数（启动后固定）、奖励和五项代价（解析计算）、掉落概率（解析计算）。本版本不预测触觉、接触力、扭矩或加速度，也不能据此宣称已具备准确的力矩约束。

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

### 5.2 WM 监督损失

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

## 6. N-P3O 思路已如何接入

实现名称是 **归一化、零违反预算的 constrained BPTT**：借鉴奖励/代价分离和 hinge 约束思想，保留可微世界模型的路径梯度，不复制 PPO 概率比率/clipping。[N-P3O 来源](https://arxiv.org/html/2309.15430)

### 6.1 速度阈值的辨识证据与当前选择

辨识 `hand.json` 中的 `command_speed_limit_rad_s=2.0`（114.6°/s）是轨迹生成/导入校验的预设值。源码 `excitation.py`/`io.py` 用它检查相邻指令差分；它不在 best_parameters 的拟合变量中。`simulation_abort_speed_rad_s=20.0` 是仿真异常阈值。模型本身标注 `EFFECTIVE_RESPONSE_MODEL_NOT_HARDWARE_CALIBRATION`。

因此没有找到可直接采用的“逐关节硬件最高速度辨识结果”。按用户希望更小的要求，新训练先选：

- 实际关节速度训练阈值：90°/s，即 π/2 rad/s。
- 目标变化速度训练阈值：60°/s，即 π/3 rad/s，40 ms 对应 2.4°。
- 可通过 `--joint_speed_limit_deg_s`、`--target_rate_limit_deg_s` 修改；这些是策略训练阈值，不是硬件额定参数。

### 6.2 解析代价与价值输出

```text
c_speed = ReLU(max_j |qd_next,j| / (π/2) − 1)^2
c_rate  = ReLU(max_j |target_j−previous_target_j| / (dt×π/3) − 1)^2

g(theta) = exp(−(theta / 15°)^2)
c_object_omega = g(theta) × ReLU(||omega_next|| / 0.5 rad/s − 1)^2
c_object_v     = g(theta) × ReLU(||v_next|| / 0.05 m/s − 1)^2
c_drop         = ReLU(||p_next−hand_center|| / 0.3 m − 1)^2
```

超限程度用物理阈值归一化，取最严重关节，避免平均值掩盖单关节问题。平方 ReLU 为超限提供可微梯度；0/1 超限用于真实统计，不用于 BPTT 梯度。

物体速度采用向量范数；新代价在阈值以内为 0，超限后平滑增长。`g` 在姿态误差 0°/15°/30° 时分别为 1/0.3679/0.0183，没有时钟、成功锁存或离散切换；姿态尚未到达时减弱速度约束，以允许转动。它的角度梯度完整参与 BPTT。掉落代价不乘 `g`，在手中半径 0.3 m 以内为 0，不把物体拉回某个固定位置。该半径仍是工作空间离开判据，不等价于接触丢失或几何意义上的实际掉落。

任务 reward 只保留姿态到达奖励，`stability_reward_scale=0`；位置跟踪继续关闭，旧 joint_vel/action_rate 负奖励权重仍为 0。Critic 输出 `[V_reward,V_speed_cost,V_rate_cost,V_object_omega_cost,V_object_v_cost,V_drop_cost]`；五个成本头非负。成本和奖励分别构造 λ-return，目标在任何 Critic 更新前固定。姿态误差仍由到达奖励优化，不在随机初始姿态上施加必须已到达的零预算约束；0.52 秒保持要求继续用于成功统计。

```text
Jr = discounted reward rollout + terminal reward value
Jci = discounted cost rollout + terminal cost value
normalized_cost_i = (1−gamma) × mean(Jci)
budget_i = 0

L_actor = −(1−gamma)×mean(Jr)/(dt×10)
          + cost_penalty × Σ ReLU(normalized_cost_i−budget_i)
```

10 是当前姿态奖励密度上界；仅取消位置跟踪的上一版为 13，旧位置跟踪模式为 14。归一化上界随任务开关一致变化；cost_penalty 初始为 1，五项归一化代价直接求和。成本不做减均值处理，零违反预算针对实际使用的非负代价定义，物体速度项含 `g` 权重。预算目前明确固定为 0，没有实现非零预算或自适应乘子，不应把它描述为完整 N-P3O/原始 primal-dual。

奖励使用掉落后的 survival 折扣；五项成本继续计算，不让预计掉落自动抹去剩余代价。真实掉落仍终止 episode，想象中使用解析掉落代价和原有软生存概率；边界以内的提前趋避梯度来自软生存概率，`c_drop` 本身只对越界状态计费。Actor 更新冻结 WM/Critic 参数，但保留经过它们的输入梯度。Critic 回归按本轮固定目标的逐头标准差（下限 1）缩放误差，减少初期成本值过大压制奖励头的情况；不改变值的单位或预算。

真实日志包含五项 cost sum、超限环境步数、最大超限比；想象日志单列折扣成本。物体速度 cost sum 含 `g` 权重，但 violations/max_ratio 使用未加权的实际速度比；`object_hold_cost_gate_mean` 单独记录权重，避免把远离目标时的低代价误读成低速度。所有状态使用 reset 前的真实采样。当前实际速度成本在控制边界取样，未宣称覆盖每个 2.5 ms 子步的瞬时峰值。

### 6.3 当前没有做什么

这是一种可微软约束学习机制，不保证任意时刻都不超速。短验证中的随机初始化策略仍明显超限，说明日志确实在计费，也说明不能把短验证当成速度约束已收敛。严格目标速率投影若需要，必须同时加入真实执行与想象路径；实际关节峰值仍需单独监控。

暂不加入目标加速度代价，因为需要再多一帧历史/目标状态及合理阈值，不能从当前一帧上一动作准确构造。新增物体保持代价使用现有物理状态与成功阈值，不引入未经辨识的硬件加速度上限。连续权重与速度代价之间的权衡仍需实际学习验证，代码接通不代表策略不会停在错误朝向或逃避稳定要求。

### 6.4 全程奖励、成功统计与 Critic 目标

令 θ 为目标姿态的最短 SO(3) 角误差，v/ω 为物体线/角速度。当前每步奖励不再依赖物体与目标位置的距离：

```text
r_orientation = 10 × [0.3 exp(−(θ/90°)²) + 0.7 exp(−(θ/10°)²)]
r_position    = 0
r_stability   = 0
r_step        = 0.04 × r_orientation
```

角度计算使用 rad，上式角度常量需先转换；高斯指数没有额外的 1/2。每步奖励上界为 `0.04×10=0.40`。奖励从每个 episode 的第一步起全程计算，不等待第 10 秒，也不在成功后关闭。10 s 仅为诊断标记，既不是网络输入也不是奖励门控。

成功统计要求连续 13 个控制步（0.52 s）同时满足：姿态误差 <0.1 rad（约 5.73°）、角速度 <0.5 rad/s、线速度 <0.05 m/s，且没有掉落或其他终止；超时步也不能新增成功。物体可在手内选择合适位置，不再要求位置误差 <0.02 m。一条目标指令最多计一次成功，成功后继续保持同一目标，奖励持续计算；只有 reset 或显式新指令才换目标。`task_stable_fraction` 统计未结束的单步是否满足上述三项阈值，不等于完成 13 步保持的成功率。掉落仍使用第 3 节的手中半径 0.3 m 判定，并未增加新的接触传感器条件。

真实仿真回调与可微想象共用奖励和保持判据，代价也由同一函数解析计算。位置误差项关闭后没有到某个固定位置的直接奖励梯度；位置依然通过动力学、接触和掉落生存概率/代价影响长期回报。WM 继续预测物体位置、线速度等完整物理状态。目标位置参数、观测槽和历史接口暂时保留，Actor 81 维单帧输入、WM 结构不变，Critic 改为六头。新训练默认关闭位置跟踪、启用物体保持代价；旧 checkpoint 缺少开关时分别按 `track_object_position=True`、`object_hold_costs_enabled=False` 解释，保留原任务和三头结构，不能静默改成新目标后续训。

六头 Critic 的 λ-return 逐头计算：`G_t=r_t+γ c_t[(1−λ)V_{t+1}+λG_{t+1}]`，最后从 `V_H` bootstrap；奖励头的 c 为软生存概率，五项代价头的 c 恒为 1。当前 γ=0.99、λ=0.9259454627568515。Critic 使用同一在线网络生成更新前的固定目标，没有额外 target network，也没有混入真实 TD 损失。Actor 的目标是第 6.2 节的显式想象回报加终值，不应把 Critic 的 λ-return 公式直接当作 Actor 损失。

## 7. 观测关节延迟是否必须？

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

## 8. 验证、使用和剩余限制

- 143 项相关测试通过，含真机拇指顺序、限位/偏置映射、三网络公共量一致性、原点变换、WM 增量平移不变性、成本单位及梯度、冻结参数的想象优化与价值拟合。
- 4 环境、2 轮完整短验证通过；最终一次启用 WM CUDA Graph，完成 WM/Actor/三值 Critic 更新及真实 FK/观测/奖励审计，无接触容量溢出或非有限异常。
- 实际 v25/9300、v26、v27 检查点均完成 strict state_dict 加载及推理对象构造；v27 完成一次离线 propose。
- 短验证不是新策略的效果评估。长期学习收敛、删除额外观测迟延后的实机鲁棒性、速度成本达标、reset 首步预测质量仍待新训练检验。

新训练入口仍为 `/home/brainco/reorientation/training/run.sh train`，默认 1024 环境、v27、25 Hz、普通手内初始化。2026-11-05 14:39 已启动新训练，`training/current` 和 `training/current_job` 指向第 9 节记录的新运行；旧训练文件保留。

日志：[回归测试](/home/brainco/reorientation/training/validation_v27_tests.log)、[最终 CUDA Graph 短验证](/home/brainco/reorientation/training/validation_hardware_center25_v27_final.log)。

关键源码：

- [revised_contract.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/revised_contract.py)
- [spatial_contract.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/spatial_contract.py)
- [core.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/core.py)
- [networks.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/networks.py)
- [constraints.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/constraints.py)
- [trainer.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/trainer.py)
- [context.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/context.py)
- [simulation.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/simulation.py)
- [inference.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/inference.py)
- [test_hand_center_contract.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/tests/test_hand_center_contract.py)
- [train.py](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/scripts/supermimic/train.py)
- [hand.json](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/data/fit_03/hand.json)
- [excitation.py](/home/brainco/PACE/REVO3_PACE/src/revo3_sysid/excitation.py)

## 9. 本次重新训练

2026-11-05 14:39（北京时间）从随机初始化启动 v27，目标 10000 轮，1024 环境；普通手中初始位姿及原有复位随机化，不加载旧权重或抓握库。前 500 轮只训练 WM，之后联合训练，并在 1000 轮内逐步提高 Actor 学习率。

采用 GPU Replay、WM CUDA Graph，WM/Actor/Critic 执行批量分别为 2048/4096/8192。控制 25 Hz、PD 200 Hz、积分 400 Hz；实际关节速度与目标变化速度的代价阈值分别为 90°/s、60°/s。每 100 轮保存编号权重并更新 `latest.pt`。

### 9.1 一轮训练具体做什么

1. 启动时先采集 64 个控制步，即 65536 条真实环境转换，建立 Replay 和归一化统计。这不是 64 次梯度更新。
2. 每轮按 `utd=0.4` 累积采样额度，交替采集 2、3 个控制步，平均每轮增加 2560 条真实转换。`utd` 在这里表示外层训练轮数/控制步数；每轮又包含 10 次 WM 更新，不能把 0.4 当成 WM 梯度更新与样本之比。
3. 每轮 WM 更新 10 次，每次逻辑 batch=2048、序列长度=8。
4. 第 1–500 轮只更新 WM。第 501 轮起，从 Replay 采 4096 个真实起点，各自想象 8 步，对 Actor 更新一次。
5. 同一轮产生 4096×8=32768 个想象状态与固定六头 λ-return 目标（历史运行使用三头）。Critic 每轮训练 5 个 epoch，每个 epoch 分 4 个 minibatch，每个 8192 样本，合计 20 次 Critic 更新。
6. 每 50 轮触发 WM 流式开环审计，每 100 轮保存编号权重并原子更新 `latest.pt`；指标逐轮追加到 `metrics.jsonl`。

因此正常联合训练阶段的优化器更新次数比例为 **WM : Actor : Critic = 10 : 1 : 20**。终止/超时环境自动复位到配置的手内初始状态；初始位姿、随机化和 Replay 都来自这次新运行。

### 9.2 当前超参数、采样与执行设置

| 项目 | 本轮实际值 |
|---|---|
| 随机种子 / 训练目标 | 42 / 10000 外层迭代 |
| 并行环境 / 训练设备 | 1024 / 单张 CUDA GPU（RTX 5090） |
| Replay 时间容量 | 每个环境最近 1000 个控制步，共 1024000 个转换槽位 |
| WM 窗口 / Actor 想象长度 | 8 / 8，即各 0.32 s |
| Actor 历史 | 10×81，首末跨度 0.36 s |
| WM / Actor / Critic 执行 batch | 2048 / 4096 / 8192；三个专用值覆盖通用 `microbatch=128` |
| 数值精度 | FP32，`amp=none` |
| 加速选项 | GPU Replay、WM loss/backward CUDA Graph；物理后端自身 CUDA Graph 开启 |
| WM 优先采样 | `adaptive_replay=true`，20% 均匀混合 + 80% 优先分布，priority 指数 0.6 |
| 抓握库 / 抓握课程 | 均未启用；`warmup_grasp_bank` 也未启用 |
| WM readiness 自适应闸门 | `wm_readiness_rounds=0`，本轮依靠固定 500 轮 WM 预热，没有启用按误差阈值决定放行的闸门 |
| 探索 | tanh 前 logits 加标准差 0.1 的高斯噪声；Actor 想象动作不添加此探索噪声 |
| 优化器 | WM/Critic：Adam 默认 β=(0.9,0.999)；Actor：Adam β=(0.9,0.95) |
| 基础学习率 | 2e-4，余弦衰减跨度 250000 轮，最低比例 0.03 |
| Actor 学习率渐增 | 第 501 轮为基础学习率的 0.001，第 1500 轮达到 1.0 |
| 梯度处理 | 三网络均做范数上限 1.0；Actor 额外做每元素 ±0.1 裁剪；非有限梯度直接报错 |

学习率公式为 `lr(i)=2e-4×[0.03+0.97×0.5×(1+cos(π×min(i/250000,1)))]`，i 是更新前的零基迭代计数。10000 轮任务并不意味着学习率会在第 10000 轮降至最低；该轮基础学习率仍约为初始值的 99.6%。

WM 的优先采样在本实现中只作用于多步序列。Actor 起点使用单步均匀采样；配置里仍保存 `event_fraction=0.25`，但开启自适应 WM 分支后不会再走另一个事件库替换分支，不能把本轮写成“25% 抓握样本/事件样本”。

### 9.3 启动命令与检查点语义

后台任务实际由本节下方的 `launch.py` 管理，记录子进程身份、心跳和退出状态。以下是等价的新训练命令，供复现参数；输出目录必须另取一个尚不存在的路径，不应对正在运行的目录重复启动：

```bash
bash /home/brainco/reorientation/training/run.sh train \
  --network_revision hardware_center25_v3 \
  --num_envs 1024 --iterations 10000 --seed 42 \
  --joint_speed_limit_deg_s 90 --target_rate_limit_deg_s 60 \
  --adaptive_replay --microbatch 128 \
  --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 8192 \
  --wm_cuda_graph --replay_storage cuda \
  --audit_interval 50 --save_interval 100 --latest_interval 100 \
  --output_dir /home/brainco/reorientation/training/runs/v27_reproduction_new
```

检查点保存三网络、三优化器、归一化 buffers、配置、噪声配置、迭代计数及 RNG 等状态。**不保存完整 Replay 和仿真现场**；恢复是加载学习状态后重新初始化仿真并采集 Replay 预热数据，不是逐 bit 接续同一条物理轨迹。恢复时应使用同一运行的检查点与 `env.yaml`，新输出目录另建；v25/v26 按旧协议加载，不会因启动脚本默认 v27 就被自动升级。Actor-only 导出不含继续训练所需的完整状态。

- [本次训练目录](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940)
- [完整日志](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/train.log)
- [指标 JSONL](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/metrics.jsonl)
- [实际启动参数](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/launch.py)
- [后台任务状态](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/status.json)
- [TensorBoard](http://127.0.0.1:6009/#timeseries)

## 10. 本轮训练快照与失败记录

### 10.1 退出前快照（2026-11-05 14:49:12 北京时间）

本节是采样时点的记录，后续进度以实时日志为准。[原始统计快照](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/documentation_snapshot_20261105_144912.json) 保存了对应指标、GPU 样本和 WM 审计，避免把不同时间的数字混在一起。

| 项目 | 采样结果 |
|---|---:|
| 后台状态 / 阶段 | `training` / `joint` |
| 已完成迭代 / 计划 | 1572 / 10000 |
| WM / Actor / Critic 累计优化次数 | 15720 / 1072 / 21440 |
| 已保存的最近编号权重 | `model_1500.pt` |
| 最近 100 轮区间 | 1473–1572 |
| 最近区间平均每轮耗时 | 0.406 s |
| 平均采集 / 更新耗时 | 0.120 / 0.280 s/轮 |
| 真实转换吞吐量 | 约 6,319 条/s，合计 1024 个环境 |
| GPU 利用率（10 次约 1 s 间隔采样） | 平均 78.4%，范围 48%–97% |
| 整卡显存使用（nvidia-smi） | 17.01 GiB |
| 训练记录的 PyTorch 峰值已分配显存 | 6.63 GiB |

整卡显存包含物理引擎、CUDA 上下文、缓存和其他进程，不能与 PyTorch 的已分配峰值直接视作同一指标。GPU 利用率是这一小段时间的采样，不是全程平均值，也不表示已经持续满载。

最近 100 轮共统计 256,000 个环境控制步，包含训练探索噪声，**不是无探索的独立策略评测**：

| 任务/约束指标 | 样本结果 |
|---|---:|
| 平均姿态误差 | 127.48° |
| 平均位置误差 | 3.61 cm |
| 单步四条件稳定占比 | 0.000% |
| 成功事件 / 掉落 / 超时 | 0 / 1038 / 371 |
| 实际关节速度超限环境步占比 | 97.88% |
| 目标变化速度超限环境步占比 | 99.97% |
| 每环境步实际速度 / 目标速率平均代价 | 13.2360 / 34.9226 |

每项“超限”表示该环境步至少一个关节超阈值，不是超限关节数量比例。随机探索本身也会增大目标变化速度；目前这些结果尚不能支持速度约束已达标或策略已收敛的结论。当前只记录并说明现状，本次文档更新未调整训练超参数。

最新完成的流式 WM 审计锚定第 1550 轮。各预测跨度上的有效样本不同，reset 后不继续累计同一序列：

| 预测步数 / 时长 | 有效环境数 | 物体位置向量 RMSE | 姿态测地角 RMSE |
|---|---:|---:|---:|
| 1 / 0.04 s | 128 | 1.51 mm | 1.92° |
| 8 / 0.32 s | 124 | 10.38 mm | 15.06° |
| 16 / 0.64 s | 118 | 28.71 mm | 24.70° |
| 32 / 1.28 s | 110 | 117.56 mm | 22.19° |

这些误差是当前数据分布上的一次流式检查；16/32 步超过训练的 8 步窗口，作为外推诊断，不表示 Actor 当前使用了更长想象跨度。启动验证和这一时点已记录的损失均为有限值，模型仍需持续观察任务成功、超限率和开环误差。

### 10.2 14:50 的机械限位异常

后台状态最终为 `failed`、退出码 1。已完成并记录第 1801 轮，随后在下一轮采集阶段、reset 前检查到环境 605 的食指 MCP 越过机械下限；异常状态在写入 Replay 前被拒绝。证据来自 [physics_failure.json](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/physics_failure.json)、[现场张量与轨迹](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/physics_failure_state.pt) 和训练日志。

| 失败现场 | 记录值 |
|---|---:|
| 关节 | `right_index_MCP_joint` |
| 实际几何角度 | 约 −15.12° |
| 机械下限 | −5° |
| 超出下限 | 约 10.12°，大于 5° 健康检查容差 |
| 现场最大关节绝对速度诊断 | 83.63 rad/s；后续按 solver 关节名称映射核对，最大值确实来自该食指 MCP，带符号速度为 −83.63 rad/s |
| solver 溢出标志 | `overflow_mask=0`，无已报告容量溢出 |
| 当时 solver 设置 | iterations=100、子步=2 |
| 最后完整权重 | [model_1800.pt](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/model_1800.pt)；`latest.pt` 也保存到第 1800 轮 |

已核对 `latest.pt` 可读取、version=27、iteration=1800，并包含三网络及三优化器状态。第 1801 轮日志存在不意味着它有对应检查点；恢复应以权重实际保存的迭代为准。

目前能确定的是机械限位健康检查触发退出，而非记录到的接触容量溢出。异常根因仍需结合现场动作、速度与子步响应进一步定位，不能仅凭本条记录归因于 GPU、网络损失或某一个 solver 参数。本次文档更新保留了失败现场，没有修改容差或自动重启训练。

### 10.3 后续诊断：真实仿真越界与 WM 预测误差

[机械限位诊断数据](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/mechanical_limit_analysis.json) 保留了失败前 40 个控制步、编译后限位参数，以及最近检查点的离线 WM 对比。

- 出错目标为 −4.818°，仍在 −5°～90° 内；失败步全部环境的解码目标及已保存历史中的目标均未越出各自限位。编译后的食指 MCP 限位启用且范围正确，排除了“目标下发越界”和“限位没有导入”这两个直接解释。
- 最后 6 个控制步的 MCP 实际速度约为 −1473、+1848、−467、−1432、+2995、−4792°/s，已经出现明显往复振荡；末步 PD 请求力矩约 +6.53 Nm，实际被执行器上限截为 +0.364 Nm。
- 本地 MJWarp 对 hinge 的位置更新为 `q_next=q+dt×qd_next`。按末步状态和 2.5 ms 子步反推，最后一次积分前约为 −3.144°，这一子步前进约 −11.979°，到达 −15.122°。这是根据已保存状态和已安装积分公式反推的值，不是单独记录的子步起点。
- 当前 `jnt_margin=0`。限位约束按积分前位置判断是否激活，且属于有限刚度/阻尼的约束，而非对最终 q 的位置裁剪；对应末步该自由度约束力矩为 0，与从界内一次跨出边界的情形一致。关节限位 `solref=[0.0047021,0.326091]` 的阻尼比小于 1，具有欠阻尼回弹特征；这属于需要验证的数值配置，尚不能单凭它认定引擎代码 bug。[MuJoCo 限位机制](https://mujoco.readthedocs.io/en/stable/computation/index.html#limit)、[solver 参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
- 用最近保存的第 1800 轮 WM 对失败步的真实起点/目标/context 做一次离线前向，MCP 预测约 +32.87°、+273.33°/s，而真实结果为 −15.12°、−4791.50°/s，说明该检查点不能准确预测这段异常响应。它比失败时少一轮更新，不能冒充失败瞬间的精确 WM 输出，也不能用一次误差倒推出异常的唯一根因。

实际采样路径是 Actor → 延迟 PD → Newton/MJWarp → 真实状态；WM 不把自己的预测写回真实仿真。因而直接故障是实际动力学越界，WM 误差最多通过学习到的动作间接参与。应优先用记录的动作验证仿真子步、限位响应和带迟延饱和 PD 的稳定性，再评价 WM；单改 WM 损失或放宽健康检查不能保证消除越界。本次仅诊断，未修改物理参数或重启训练。

## 11. 日志、TensorBoard 与核对方法

本次运行 ID 为 `manual_inhand_v27_1024_10000_20261105_143940`。`training/current`、`training/current_job` 是便于查看的软链接；下表使用固定路径，后续启动新实验也不会改变这些链接的含义。

| 文件/界面 | 用途 |
|---|---|
| [config.json](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/config.json) | 实际网络配置、输入/关节协议、噪声、物理时序及执行选项 |
| [env.yaml](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/env.yaml) | 本轮完整环境、重力、PD/solver、复位及域随机化设置 |
| [train.log](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/train.log) | 启动输出、逐轮指标、警告和异常栈 |
| [metrics.jsonl](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/metrics.jsonl) | 每行一轮的结构化训练指标 |
| [stream_audit.jsonl](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/stream_audit.jsonl) | 多步开环 WM 误差与恒定状态基线 |
| [status.json](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/status.json) | 后台任务状态、心跳、命令、进程身份和退出结果 |
| [latest.pt](/home/brainco/reorientation/training/runs/manual_inhand_v27_1024_10000_20261105_143940/latest.pt) | 最近一次周期保存或正常退出保存的完整检查点 |
| [startup_validation.json](/home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/startup_validation.json) | 第 603 轮时对联合训练、GPU、日志和 TensorBoard 的启动验证 |
| [TensorBoard](http://127.0.0.1:6009/#timeseries) | 本轮曲线；由独立 bridge 从 JSONL 生成事件文件 |

TensorBoard 的主要标签及统计口径：

| 标签/字段 | 正确读法 |
|---|---|
| `progress/iteration` / `progress/joint_training_active` | 外层迭代数 / 是否已进入联合训练 |
| `world_model/wm_loss*` | WM 归一化训练误差；不是直接以 m 或 rad 表示的物理误差 |
| `policy/actor_loss` | 当前为归一化姿态奖励目标与五项代价的组合，历史运行仅两项代价；不能等同真实环境 reward |
| `policy/critic_loss` | 当前六头、历史三头价值回归损失；使用目标逐头尺度，不能当成任务成功率 |
| `training/imagined_cost_joint_speed` / `training/imagined_cost_target_rate` | 想象轨迹上的归一化折扣代价，含 Critic 终值；不是实际超限率 |
| `task/task_mean_angle_rad` / `task/task_mean_hand_center_distance_m` | 新任务的平均姿态误差 / 距手中原点距离；分别是 rad / m。后者用于监控掉落风险，不是位置跟踪误差。旧任务保留 `task_mean_position_m` |
| `task/task_stable_fraction` | 新任务中未结束的环境步同时满足姿态、角速度、线速度三项阈值的占比；历史日志仍按原四条件解释 |
| `task/successes` / `task/drops` | 本轮计数，不是累计数，也不是已除以 episode 数的比例 |
| `training/cost_*_violations` | 本轮至少一个关节超阈值的环境步数；除以 `task_samples` 才得到超限比例 |
| `training/cost_*_sum` | 本轮所有环境控制步的代价之和；除以 `task_samples` 得每环境步平均代价 |
| `training/cost_*_max_ratio` | 最严重关节/环境/步的 `测量值÷阈值`；1 为边界，不能再除以样本数 |
| `performance/collection_seconds` / `performance/update_seconds` | 本轮采集/梯度更新时间，不包含所有日志和权重保存开销 |
| `health/training_running` / `health/training_failed` | bridge 读取后台状态生成的运行/失败标志 |

`reward_sum` 的定义与成本求和不同：它是每个采集控制步先对环境求平均，再对采集步数求和。单环境单控制步平均奖励应使用 `reward_sum/(task_samples/num_envs)`，TensorBoard 已提供 `task/reward_per_env_control_step`。跨多轮汇总角度、位置、稳定率和超限率时，应先分别累加分子/分母，不能把采集 2 步与 3 步的轮次简单等权平均。

`stream_audit_by_index/…` 的横轴是 audit_index，关联的训练轮次另有 iteration 字段。bridge 约 2 s 刷新、TensorBoard 约 5 s 重载，因此曲线可能比 JSONL 滞后几秒。查看原始实时日志可运行：

```bash
tail -f /home/brainco/reorientation/training/jobs/manual_inhand_v27_1024_10000_20261105_143940/train.log
```

当前机械限位健康检查允许 5° 数值超调容差，超过后在写入 Replay 前报错；该检查用于发现明显仿真异常，独立于 90°/s 的速度学习代价。若任务状态转为 failed，应结合日志和存在时的 `physics_failure.json` 检查原因；TensorBoard 页面能打开本身不能证明训练仍在运行。

## 12. 推理接口的输入输出与训练协议兼容

[ManualConvexPolicy](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/source/viserdex/viserdex/supermimic/inference.py) 的 `propose` 使用以下外部输入，内部负责 bias、缩放、历史和 Actor 推理：

| 外部量 | 形状 | 单位/坐标 |
|---|---|---|
| `joint_position_rad` | 21 | 编码器 rad，按第 1 节真机顺序；调用者不额外加 CMP bias |
| `joint_velocity_rad_s` | 21 | 实测 rad/s，同一关节顺序；不先乘 0.1 |
| `object_position_base_m` | 3 | 手基座系 m；不先减手中原点，不先乘 10 |
| `object_quaternion_xyzw` | 4 | 基座系单位四元数 XYZW |
| 目标位置/姿态 | 3 / 4 | 基座系 m / 单位 XYZW；先用 `set_goal` 设置，或在 propose 时成对提供 |

`propose` 返回 `(target, raw_action, observation)`：分别为 21 维绝对编码器目标 rad、21 维 tanh 后的动作、81 维当前观测。返回的 observation 仍采用观测接口布局，其中位置保留基座系 m，Actor 内部再做居中缩放；它不等于完成全部 RunningNorm 后的网络张量。

调用者实际发送返回的目标后再调用 `commit()`，把上一动作更新到下一帧历史；该类本身不连接或控制硬件。首次观测会重复构成 10 帧历史，后续按 25 Hz 更新。推理类关闭人工噪声注入，输入应来自实际测量；训练学习到的归一化参数保持冻结。

部署不需要 WM/Critic，但必须保留 Actor 的归一化 buffers、v27 限位、关节顺序和坐标约定。已有实现验证的是加载和推理接口贯通，并非已完成真机性能验证。Actor-only 导出可用于推理，不能替代完整训练检查点。

## 2026-11-05 补充：实际限速能力和碰撞故障诊断

本节为失败后的隔离诊断，不代表训练配置已经修复。完整脚本、数据和图片见
[机械限位与碰撞诊断报告](/home/brainco/reorientation/experiments/limit_failure_20261105/README.md)。

本次故障诊断时，参考 URDF 的 21 个活动关节均声明 `effort="0" velocity="0"`，没有提供有效额定上限。
失败运行另行加载 fit_03 电机力矩限幅：食指 MCP 的 ±0.3642501 N·m 已经现场验证生效。
该运行的 DelayedPD 仅包含指令延迟、PD 和最大力矩限幅；其速度字段没有显式配置，
Lab 实际速度上限缓冲区约为 `5.939e36 rad/s`。更关键的是当前 Newton 1.6.1 MuJoCo/MJWarp
后端不支持 `joint_velocity_limit`，所以不能靠填入 URDF 或该配置字段实现限速。
90°/s、60°/s 仍只是学习代价阈值。电机力矩限幅不限制碰撞和关节限位产生的约束反力。
[Newton 1.6.1 官方支持说明](https://newton-physics.github.io/newton/1.6.1/api/_generated/newton.solvers.SolverMuJoCo.html)

故障前最后一次 reset（步 4556）的 70 mm 物体与固定手掌存在明显穿透：
native CCD 约 9.895 mm / 10 个凸包，libccd 约 11.488 mm / 13 个凸包。
两种距离算法都确认穿透；同算法下，独立查询和导出求解器的最深距离相差约 `4.3e-15 m`。
144 个手部凸包在最终求解器中与资产匹配，最大顶点误差约 `2.6e-9 m`，未发现导入变形。
当前 reset 直接随机位置和欧拉角后写入，没有几何拒绝采样；名义手中位置不保证随机状态无穿透。

使用相同的现场质量、摩擦、PD context（65 维最大误差 0），固定回放 13 步保存动作，
不调用 Actor/WM。保留全部接触、关闭手物接触、关闭自碰撞、关闭全部接触时，MCP 子步峰值
分别约 1559、2588、1577、1779°/s。关闭全部接触后仍出现约 22.66 N·m 的限位反力矩；
因此碰撞穿透不是高速振荡的唯一原因。原目标变化率可达 MCP 1054°/s、全手 1459°/s。
保留接触但将下发目标变化限制到 60°/s 后，该段 MCP 子步峰值约 133°/s，未产生限位反力。
这也说明目标限速本身不等于实际速度硬上限。

这些是 0.52 秒的因果隔离实验，不是长程稳定性认证；接触路径敏感，未精确重复原来
−15.122° 的最后端点，不应由短段对照断言引擎 bug 已证实或所有故障已经修复。
应先修正 reset 几何可行性，并在真实执行和 WM 想象中同步加入目标变化限制，
再验证 PD、软限位参数与积分步长。所有隔离实验已退出，正式训练未重启。

## 2026-11-05 更新：统一 100 rpm / 1.28 N·m 输出上限

按用户指令，新的 `actuator_config()` 对全部 21 个关节按名称设置：

| 项目 | 全关节统一值 | 当前执行效果 |
|---|---:|---|
| `actuator_effort_limit` | 1.28 N·m | DelayedPD 对电机输出做 ±1.28 N·m 限幅 |
| `joint_effort_limit` | 1.28 N·m | 同步声明关节输出上限；不限制接触/限位约束反力 |
| `actuator_velocity_limit` | 10.471975511965976 rad/s | 写入模型侧声明参数；当前 PD 不包含速度依赖的电机包络 |
| `joint_velocity_limit` | 10.471975511965976 rad/s | 写入关节数据，但当前 SolverMuJoCo 不强制执行 |

100 rpm 为关节输出侧 100 转/分钟，即 600°/s，不是 100°/s，也不是传动前电机轴速度。
它与训练代价阈值是不同层次：实际速度代价 90°/s、目标变化代价 60°/s 继续保留。
本次没有引入转矩—转速曲线或目标滤波；PD Kp/Kd、armature、摩擦、15 ms 指令延迟、
25/200/400 Hz 时间安排、位置范围与关节顺序均沿用当前协议。

配置来源是 [revo3_joint_output_limits.json](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/config/revo3_joint_output_limits.json)。
同步的 [revo3_right_output_limits.urdf](/home/brainco/reorientation/source_home/brainco1/workspace/xinyu/viserDex_revo3_20261102/config/revo3_right_output_limits.urdf)
将 21 个关节的 `effort` / `velocity` 写为 1.28 / 10.471975511965976，角度限位不变。
这是本地派生文件，原 vendor、校验固定的参考 URDF 和 PACE fit_03 辨识原件不修改；
运行时仍使用原 144 凸包 USD，并通过执行器配置读取 JSON 上限。

新环境的 context manifest 和执行器审计记录 `actuator_output_limit_override` 及参数文件哈希，
区分原辨识值与用户覆盖值；旧保存环境按原值加载，不静默换成新参数。
现有 resume 的环境/context 合同检查会拒绝把旧完整训练状态直接视为新动力学的续训。

验证：106 项测试通过，包含 21 关节正/负双向 native PD 饱和、15 ms 延迟保持和历史配置合并。
官方 Newton/MJWarp 1024 环境读回全部 21,504 个关节：力矩上限约 1.27999997 N·m，
声明速度上限约 10.47197533 rad/s（float32）。4 步保持诊断通过，碰撞容量溢出为 0。
这不构成实际速度硬限幅或长期稳定性证明，先前发现的 reset 穿透仍需另外处理。
结果见 [验证 JSON](/home/brainco/reorientation/training/validation_output_limits_100rpm.json)、
[测试日志](/home/brainco/reorientation/training/validation_output_limits_100rpm_tests.log)。本次未启动正式训练。
