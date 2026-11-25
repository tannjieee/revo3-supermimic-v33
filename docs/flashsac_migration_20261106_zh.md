# Revo3 从可微世界模型训练迁移到 FlashSAC：代码审计与实施方案

日期：2026-11-06（Asia/Shanghai）。状态：**分析交付，尚未实现新迁移、未启动训练**。

建议新建独立 FlashSAC 训练路径，复用当前真实 Newton 环境合同、观测和评估链路，以真实 replay 的分类分布双 Q 训练取代 WM 想象和 λ-return。首先完成 1-step、单 GPU、带噪历史 Actor 的最小闭环，再增加 3-step 与吞吐优化。动作跳变必须作为独立问题处理：换算法本身没有限速保证；若坚持当前全范围绝对动作映射，只能通过初始化、探索尺度、代价和发送前拒绝机制降低风险，不能声称硬限速。需要硬保证时，应另立显式动作合同，而不是暗加滤波。

## 1. 证据、版本与分析范围

### 1.1 本地证据优先级

当前仓库 HEAD 为 `0ec92382fe14f229eeaa2d809c770b86c3f5be40`，有用户未提交的 v32 paper_critic、console 和相关文档修改。本文不覆盖这些文件，不提交、不切换分支、不启动旧训练。复现基线应取冻结源码及运行配置，不能只取 Git HEAD。

| 证据 | 位置与含义 |
|---|---|
| 当前失败交接 | [analysis_2552/README.md](/home/brainco/reorientation/training/diagnostics/papercritic_v32_long_20261106_032925/analysis_2552/README.md)、同目录 `summary.json`、`training_diagnosis.png`；本轮已读并查看图 |
| 真实训练数据 | `/home/brainco/reorientation/training/runs/die50_47g_papercritic_v32_20261106_032925/` 的 `config.json`、`env.yaml`、`metrics.jsonl`、`stream_audit.jsonl`、`physics_failure.json`、`physics_failure_state.pt`、`model_2500.pt` |
| 冻结代码 | `/home/brainco/reorientation/training/diagnostics/papercritic_v32_long_20261106_032925/frozen_source/` 与 `source_manifest.json`、`working_tree.patch` |
| 代码合同 | `AGENTS.md`、`docs/paper_critic.md`、`docs/revo3_network_v30_zh.md` 第24–25节及实际配置；早期100 g、20 Hz、旧Critic段落是历史，不是当前参数 |
| 已有 FlashSAC 实验 | `/home/brainco/reorientation/experiments/flashsac_hand_20261105/README.txt`、`algorithm.py`、`reports/network_frequency_audit.txt`、`configs/` 和只读 `vendor/FlashSAC/` |

已知 v32 在2552轮日志后机械越限退出，Actor更新1552次。食指DIP实际103.9°，上限90°、容差5°；失败40步轨迹中动作仍在[-1,1]、施加扭矩≤1.28 Nm，而40 ms内目标最大跳变102.9°。末200轮bootstrap/短程奖励参数梯度比中位数362.81，关节速度/目标速率违规36.82%/56.25%。这支持优先移除对不可靠多步模型梯度的依赖，但不是可微WM路线整体无效的因果证明；接触和求解器贡献也未被排除。

seed66/67各64环境×640步、关闭探索的独立结果：HOME model1000均0成功0掉落0违规；model2500均0成功19掉落，目标速率违规20.37%/19.94%。因此仅缩小随机探索不能修复已学习均值策略的跳变。TensorBoard 6010在线仅是面板状态，不表示训练运行。

### 1.2 FlashSAC 确切来源

已联网核实，目标是 Donghu Kim、Youngdo Lee 等人的 **FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control**，不是 FastSAC、普通SAC或其他同名缩写。arXiv元数据标注v1为2026-04-06、v2为2026-05-15；RSS XXII也有正式条目。以下以固定实现为可执行规范，论文用于解释设计；不把不同出版页面的任务数量或时间摘要混成一个实验版本。[arXiv版本页](https://arxiv.org/abs/2604.04539)、[v2正文](https://arxiv.org/html/2604.04539v2)、[RSS正式条目](https://roboticsproceedings.org/rss22/p099.html)。

官方项目页指向 [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC)，另有作者组织的 [RSL-RL实现](https://github.com/Holiday-Robot/rsl_rl_flashsac)。当前本地vendor固定到 `87edc9061150ae9e962dd84e6544e27a1554b3ab`，提交标题 `Create LICENSE`；本轮只读 `git ls-remote origin HEAD` 返回同一SHA，vendor工作树干净。[固定提交](https://github.com/Holiday-Robot/FlashSAC/commit/87edc9061150ae9e962dd84e6544e27a1554b3ab)。论文v2与该4月代码提交分别记录，不声称代码逐项覆盖后来论文版本。

首选复用这个已接入本地PyTorch环境的官方核心，不同时换成RSL-RL分支。后者是后续工程选项，尚未逐文件对齐，不能凭名称宣称二者数值等价。保留MIT许可证；固定依赖，不升级共享IsaacLab环境。

## 2. 实际算法合同：哪些不能被普通 SAC 替代

下面的细节来自本地固定源码逐文件检查，可通过相同SHA的官方链接复核。

| 项目 | 固定实现及GPU基准 | 本任务迁移建议 |
|---|---|---|
| 网络 | Actor宽128、Q宽256，各2个残差块，内部扩展4倍；UnitBatchNorm、UnitRMSNorm、更新后参数归一化 | 保留这些模块，不换成当前ELU MLP后仍称忠实FlashSAC |
| Critic | 两条独立分类分布Q，各101 bins、support [-5,5]；交叉熵拟合投影后的TD分布 | 替换v32六头线性V/MSE，不是仅增加第二个V头 |
| 双Q选择 | 按期望Q较小者选择**整个目标分布**；Actor使用两Q期望最小值 | 不能逐bin取最小概率，不能把Q值当logit |
| 目标网络 | 初始化复制双Q，每次Q更新后参数EMA，τ=.01 | 与v32无EMA在线V根本不同；无目标Actor |
| BN生命周期 | Q更新把当前/下一观测与动作拼批；目标Q也以training=True更新自己的BN统计；EMA函数只遍历parameters | 不额外复制/EMA running buffers；目标网络不是“所有buffer始终冻结” |
| Actor概率 | 高斯重参数采样后一次tanh；log_std∈[-10,2]；稳定Jacobian修正后对21维求和 | 保留log_prob与实际动作坐标一致，不叠加旧反射AR探索 |
| 熵温度 | α初始.01，学习log α；目标熵由σ=.15和动作维数计算 | 21维约−10.04，不直接改为常见−21；构造器重算target_entropy，单改配置字段无效 |
| 探索 | 复用高斯噪声ε，重复长度服从截断ζ分布（指数2、1..16） | 重复的是噪声，不是绝对命令；均值/标准差随观测变化仍会跳变 |
| 更新频率 | 官方IsaacLab脚本1024环境，每向量步2次Q更新，Actor/α每2次Q更新一次 | 用显式计数，不复用v32 `utd=.4` 的含义 |
| replay | 均匀采样，GPU基准容量10,000,000、预热100,000、batch2048 | 最小闭环容量1,000,000，独立声明偏离；不沿用WM误差优先采样/事件混采 |
| n-step | 基础配置n=1；IsaacLab脚本n=3，γ=.99 | 先n=1验终态，再修正有效步长后n=3；不能将基础yaml误称GPU基准 |
| 奖励归一化 | 原始回报尺度估计，除以 `max(sqrt(Var(G)+eps), max_abs_G/5)`，不减奖励均值 | 存原始标量奖励，采样时缩放；评估用原始任务单位 |
| 数值与吞吐 | 官方AMP/compile开启；参数范数控制和分类损失是稳定化核心 | 首轮FP32 eager，然后独立验证AMP/compile；不是加一个通用grad_clip便等价 |

代码来源：[网络/层](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC)、[更新](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/flashSAC/update.py)、[GPU启动参数](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/scripts/run_isaaclab.sh)、[奖励缩放](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/agents/utils/reward_normalization.py)。

行为噪声的当前实现有全局标量重复计数/长度，噪声张量按环境独立；不能误称每个环境独立抽取持续长度。reset时是否清理该环境噪声须由新适配显式定义并测试，不能把原AR状态迁过去。学习时Actor/TD重新采样高斯，不使用行为时缓存ε计算重要性比；这是官方离策略实现，不应再叠PPO式概率比。`training=False`仅控制BN，不保证网络forward确定性；评估必须显式取 `tanh(mean)`。

官方奖励累计统计的边界顺序也是具体实现的一部分：`G_new=gamma*(1-done)*G_old+r`，done包含termination和timeout。不要悄悄换成另一种return normalizer。support投影会裁到[-5,5]，奖励缩放并不保证含熵bootstrap后的所有目标在support内部；必须记录裁剪质量和边缘概率。

### 2.1 新的梯度路径

旧路径：Actor → WM多步状态/奖励/风险 → 在线V尾值 → Actor梯度。

新路径：真实Newton转移 → replay → Q的停止梯度TD标签；Actor → 采样动作 → Q(s,a) → Actor梯度。

对1-step，目标分布是对 `r_scaled + gamma*(1-terminated)*(Z_target - alpha*log_pi(a'|h'))` 的分类投影；Actor最小化 `E[alpha*log_pi(a|h)-min(Q1,Q2)]`。目标在no_grad中计算，Actor更新冻结Q参数但保留Q对动作的导数。不创建WM、不做想象、不调用WM训练/归一化/CUDA graph、不把WM预测塞进replay。由此消除的是模型反传依赖，不是所有bootstrap误差或Q外推误差。

## 3. 必须保持的真实环境合同

基准取47 g失败运行的config/env和冻结源码，不取更早FlashSAC实验的env.yaml。保持官方 `launch_simulation`、统一 `isaaclab train/play` 注册路径及默认 `physics=newton_mjwarp`。IsaacLab v3.0.0（`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`）、Isaac Sim6.2.0.0、Newton1.6.1、Warp1.17.0，依赖覆盖与importer extra沿用仓库流程。资产仍经 `scripts/preprocessing/prepare_revo3_newton.py` 生成；不改vendor BrainCo输入。固定指南见 [官方迁移](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)、[资产准备](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/prepare_asset_for_newton.rst)、[MJWarp调优](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/solver_tuning_mjwarp.rst)。

| 合同 | 保持及验收内容 |
|---|---|
| 物体/碰撞 | 50 mm、名义47 g原骰子造型；惯量沿已有缩放假设，非实测；真实手部凸包、物体凸包及所有引用资产hash；现场形状审计，容量警告即失败 |
| reset | fixed_palm_v1匹配HOME与掌内位姿；关节、物体速度归零；清理延迟、历史、传感器状态；随机旋转目标保持，禁止换随机出生/抓取bank |
| 手接口 | 21关节：小指/无名指/中指/食指各MPR/MCP/PIP/DIP；拇指MCP/PIP/DIP/CMP/CMR；CMP索引19固定bias=1° |
| 坐标 | 外部SI基座坐标、XYZW；6D旋转取前两列按列展平；网络物体位置减(.055,-.010,.100)m后×10；出生中心不等于网络原点 |
| 执行器 | 25 Hz绝对编码器位置目标，200 Hz PD，15 ms=3 PD tick延迟，扭矩≤1.28 Nm；solver默认2子步、400 Hz、iterations100、nconmax640/njmax1024按保存绑定核对 |
| 随机化 | 保留质量/摩擦/PD等物理随机化和噪声/偏置/采样保持/失败感知；不能通过关闭随机化让迁移通过 |
| Actor可见信息 | 81D单帧×10历史=810D，reset重复首帧；不泄露真实物体速度、domain context、未来状态或时间倒计时 |
| 终止 | 真掉落terminated；20 s/500步timeout；success按已有保持逻辑统计而非自动terminate；pre-reset终态与新回合初态严格分开 |
| 保护与录像 | 原机械/非有限/接触容量fail-fast保留，故障保存轨迹与状态；每500个新训练轮保存模型并独立录500帧，记录样本数和更新数以避免“轮”歧义 |

reset首帧last_action=0是旧合同占位，不是上一条真实中点命令。目标速率首步参照实测编码器角，后续参照上一条真实下发目标；新动作审计不能把占位0当成已执行命令。

## 4. 文件级模块映射

下列路径相对仓库根，**是拟议改动，不是已实现**。优先新增 `source/viserdex/viserdex/flashsac/`，保留supermimic行为。

| 当前模块 | 处置 | 拟议文件/具体工作 |
|---|---|---|
| `supermimic/simulation.py` 的Adapter、TransitionRecorder | 复用物理/保护；扩展纯数据接口 | 新 `flashsac/env_adapter.py`，输出行为历史、pre-reset历史、privileged状态、terminated/truncated；若必须动原文件，仅新增可选钩子并回归旧接口 |
| `fixed_palm_reset.py`、`direct_actions.py`、`identified_hand.py`、`context.py`、`policy_contract.py`、`spatial_contract.py` | 复用 | 新 `flashsac/contracts.py` 固定合同指纹；引用同一限位/顺序/bias实现 |
| `sensors.py`、`core.py` 观测与history工具 | 复用 | 在适配层只做一次真实噪声状态推进；独立保存终态分支；共享物理缩放 |
| `networks.py` Actor/WM/V | 替换训练网络 | 新 `flashsac/agent.py` 调用只读官方网络与更新；新 `flashsac/observations.py` 管理布局；WM不实例化 |
| `trainer.py` 想象、paper Critic、WM optimizer | 新建训练器 | 新 `flashsac/runner.py`，采集→存储→Q/Actor/α更新，精确步数、终止、checkpoint；旧trainer不切算法分支 |
| `replay.py` WM序列/PER/事件档案 | 替换 | 新 `flashsac/replay.py`，均匀真实转移，episode_id/env_id，有效n、终态历史；初版可用官方1-step结构适配 |
| `constraints.py`、`stable_values.py` 风险定义 | 复用纯函数，替换学习用途 | 新 `flashsac/rewards.py` 输出原奖励、五风险、标量训练奖励；不导入想象训练入口 |
| `absolute_policy.py` 旧AR反射探索 | 替换 | 新 `flashsac/exploration.py` 管HOME初始化/预热/重复噪声/发送前审计；不能把旧AR直接加在tanh后 |
| `physics_health.py`、`runtime_capacity.py`、`solver_budget.py`、`manual_convex.py` | 复用 | 同样故障类型、阈值和快照，不捕获后继续训练 |
| `periodic_video.py`、`inference.py` | 复用协议、另建模型loader | 新 `flashsac/evaluate.py`、`flashsac/checkpoint.py`，独立确定性进程，模型及BN/normalizer不变，支持新版本导出 |
| 训练入口与配置 | 新增 | `scripts/flashsac/train.py`、`scripts/flashsac/play.py`、`configs/flashsac_die47/{env.yaml,agent.yaml}`；按现有官方入口注册，任务ID和算法ID显式区分 |
| 日志与测试 | 新增 | `flashsac/metrics.py`、`tests/test_flashsac_contract.py`、`test_flashsac_targets.py`、`test_flashsac_actions.py`、`test_flashsac_checkpoint.py`；复用原console风格而非覆盖未提交console文件 |

完整依赖清单、vendor SHA、patch、资产hash和有效配置写入新的run目录。旧Actor/V/WM检查点及旧FlashSAC的149D输入检查点一律拒绝作为resume；可以只读作历史评估，不能隐式warm start。HOME由配置重新构造，不加载旧Actor权重。

### 4.1 观测、归一化与critic信息

推荐首版采用官方已有asymmetric接口：存储 `o=[h810, x153]` 共963维，Actor只取前810维；x为当前物理缩放后的152维critic特征加delay_ready一维（未来实现必须用schema断言维数，不能凭位置切片猜测）。Q实际输入是整个o再拼21维动作，即984维；不能误称“Q只看153维”——固定Agent会将整个observation传给Q。x包含真值物理状态、上一动作、目标与65维域上下文，history同时让Q看见Actor的感知状态。训练用特权信息、部署Actor仅历史；这是本任务适配，GPU官方脚本默认对称观测。

保留SI/编码器/旋转语义和固定缩放；不复用旧RunningNorm统计，也不在官方BN前默认叠一层运行归一化。官方输入BN与v30的关节速度std_floor并不等价，先记录静止预热→运动的速度通道标准化P99、running variance、均值动作变动。若低方差放大复现，再独立比较输入BN的物理std下限补丁；该补丁必须标为算法适配，并同步训练/推理。输出日志不能把官方RMSNorm说成输入RunningNorm。

### 4.2 pre-reset历史是首要接口缺口

现有 `Adapter.step()` 的record已有 `state/history/action/next_physical/next_aux/reward/terminated/done`，compact `advance()`保留旧目标，并将当前动作和ready=1写入终态；随后Adapter内部history是reset后的新历史。旧FlashSAC A版用真值149D，可以只拼终态物理和aux；新810D Actor不能直接使用reset后的 `adapter.history` 作为TD下一历史。

新增终态传感器分支：从采样前传感器状态推进到pre-reset物理终态及旧目标，产生一次带噪 `o_next_terminal`，`h_next_terminal=push(history,o_next_terminal)`。非done时这一结果应与下次行为history逐元素一致；done时另行reset传感器并构造重复新初帧的行为history。不得对同一步重复抽两份不一致噪声、不得用干净终态替代含噪历史、不得让新目标穿透旧回合。恢复时缓存、RNG和统计生命周期均需明示。

## 5. replay、n-step与更新预算的精确定义

每条真实转移至少保留：env_id、episode_id、h_t、x_t、实际动作、原奖励、五风险、训练标量奖励、pre-reset h_next/x_next、terminated、truncated、effective_n/discount；审计旁路记录角度目标、噪声重复状态、配置指纹。GPU容量紧张时不要把所有旁路字段永久复制进训练tensor。

1-step标签使用terminated屏蔽bootstrap；timeout仍从旧目标/旧回合的pre-reset下一状态bootstrap。机械异常、非有限、容量溢出属于故障退出，不伪造成可学习的正常timeout，未完成env.step不能凭空补transition。

固定官方 `torch_buffer.py::_get_n_step_prev_transition` 在terminated或truncated处截断奖励序列、改写next observation；`update_critic()`却固定用γⁿ。对提前timeout（k<n）而terminated=False，正确折扣应是γᵏ。这是静态审计识别出的具体边界风险，尚未在新环境运行复现。首版n=1避开；n=3实现逐样本k与discount，手算验证后再用。终止时bootstrap为0仍需确保不穿越reset；收集结束的n−1条在途样本应统计/flush，不能凭总采样数推断可训练样本数。

学习计数定义：N=并行环境数，C=每轮向量采样步，U=该轮Q更新次数，B=2048。记录真实转移 `N*C`、更新/转移 `U/(N*C)`、回放抽样/新数据 `U*B/(N*C)`，以及Actor/α实际更新数。官方GPU C=1、U=2、N=1024时分别为1/512和4；不是“每条转移更新2次”。v32每轮平均2.5向量步，不能直接比较轮数。

小规模64环境smoke可以每步更新用于接口测试，但不可拿它做公平性能结论。正式实验固定1024环境；若内存迫使降N，用累计update credit维持比例并记录改变，不能沿用U=2而把重放强度翻倍。

## 6. 奖励与五风险如何迁移

真实env reward依旧是当前任务函数：实际config的 `track_object_position=False`、`stability_reward_scale=0`、orientation_scale=10、dt=.04，因此单步奖励上界Rmax=.4；旧FlashSAC实验的环境reward包含不同项，不能复用其数值。当前动作/关节速度env penalty权重均0，风险是在原Actor目标中单独加入；仅把env reward接到SAC会**丢失五风险优化目标**。

| 风险 | 原物理尺度/定义 | 首版迁移 |
|---|---|---|
| joint_speed | max关节速度/90°/s，超额平方 | 保留原ratio、raw、bounded及违规率 |
| target_rate | max目标变化速度/60°/s，首步参考实测位置 | 同上；60°/s在25Hz对应2.4°/步 |
| object_angular_speed | ||ω||/.5 rad/s，超额平方乘目标接近gate | gate保持；原始违规率不被gate隐藏 |
| object_linear_speed | ||v||/.05 m/s，同样gate | 保持单位与门控 |
| drop | 离网络原点距离>.3m为真实终止 | 正常运行drop风险头为0；失败五头各付一次1/(1−γ)=100 |

运行前四项 `b_i=c_i/(1+c_i)`；失败每项 `C_i=100`，否则 `C=[b1,b2,b3,b4,0]`，与当前 `failure_risk_cost` 一致。不能把raw超额直接当bounded值，也不能将掉落处罚误缩成1，或只罚drop头而忽略原先其他四头的吸收态成本。

最小迁移采用一个双Q标量目标：`r_train = r_env/Rmax − 0.05*sum(C_i)`。它保留原Actor目标中任务与风险的相对权重（共同的1−γ因子不影响无熵最优解），再让官方reward normalizer整体缩放。正常步任务项≤1、风险项<.2；真实失败惩罚部分为−25。**这是任务奖励适配，非官方新增安全算法，也不与v32完整目标严格等价**：熵、模型/真实数据、bootstrap和旧logit正则都已改变。旧logit惩罚不默认叠加到新Actor，避免混淆概率分布的优化。

每项独立保存原始单位与贡献，报告奖励缩放分母、α、α·logπ/任务Q量级、目标投影越界质量和support边缘概率。稀少失败会使max_abs_G增大并改变整体奖励尺度，可能压缩常态姿态差异；1M阶段若连续多个评估窗口投影越界质量>1%或边缘概率异常集中，应先检查该问题，不直接扩大训练。0.05并非已验证最优乘子；初版固定它，之后才做0.025/0.1单因素消融，不同时改物理阈值。

将五风险独立做分布Q/拉格朗日约束会增加网络、温度与单位选择，不在最小闭环内。辅助风险头可用于诊断，但不应未经对照就叠加回Actor。严禁以更低奖励或风险饱和值替代机械健康判断。

## 7. 绝对位置动作与探索：两个明确版本

### 7.1 S0：保持现有全范围绝对映射（兼容诊断版）

`u=mu(h)+sigma(h)*epsilon; a=tanh(u); q_enc=lo+(a+1)*(hi-lo)/2; q_geom=q_enc+bias`。

学习log概率为 `sum(log Normal(u;mu,sigma) − log(1−tanh(u)^2))`，使用官方稳定softplus公式。Q/replay都用同一个归一化a。若改成角度密度，还需减 `sum(log((hi−lo)/2))` 且同步目标熵；初版在归一化动作坐标定义熵，避免单位混用。不能tanh两次、不能反射/clamp后仍声称原tanh密度是最终动作密度。

采用现有±2°、30步五次平滑HOME预热，不使用全范围均匀随机预热；这偏离官方预热，但保留当前控制需求。首批正式学习前应验证未训练均值在**全部验证历史**接近HOME。官方UnitLinear更新会归一化权重，不能照搬旧Actor“末层权重全零”当作可靠初始化：需要独立的非归一化输出gain=0和HOME bias方案，或明确的小范围HOME初始化补丁，并验证第一次更新之后的行为。将该改变写入模型版本，不能称原网络完全未改。

初始探索按物理角度0.5°标定各关节std（局部近似 `sigma_u≈sigma_q/[half_range*(1−a_home²)]`），再用实际采样验证；不能只设同一个log_std就声称每关节噪声相同。官方目标σ=.15不会自动维持0.5°：温度适应会推动熵，若继续使用较小物理熵目标，应新建配置并明确偏离；禁止只缩行为噪声而不审计Actor学习/TD采样尺度差异。

重复噪声1..16步在25Hz为40..640ms，重抽时仍可能跳变，网络均值变化也无界于目标速率。因此S0加**发送前审计并拒绝超2.4°/步的命令、保存状态后停止该次实验**，不裁动作、不重试、不伪造转移。可用于确认初始化、采集与首批更新合同，不能作为能长期探索的硬安全保证。即便目标速率通过，也不保证接触动力学不会造成实际超速/越限，原物理保护继续生效。

### 7.2 S1：显式受速率约束的绝对目标分布（推荐后续适配）

若用户愿意修改当前“全范围直接映射、无硬速率约束”的动作合同，可在每步先计算 `L=max(lo,q_prev−vmax*dt)`、`U=min(hi,q_prev+vmax*dt)`，然后直接生成绝对目标 `q_enc=(L+U)/2+(U−L)/2*tanh(u)`。vmax初值60°/s；首步q_prev取reset实测编码器角。这不是把输出当速度下发，PD收到的仍是绝对目标；但它确实改变了策略动作参数化与可达集合，必须版本化并重新训练，不能默默加到旧模型推理。

选择局部归一化z=tanh(u)的熵作为本适配优化定义；Q输入/replay保存经L/U映射后、再按全机械范围归一化的实际a。Actor/TD更新必须用各样本对应的q_prev/ready和同一可微映射后再查Q；行为和目标策略不能一个限速、一个全范围。这样使用的是logπ(z|h)，不得把它标成物理q的密度；如选择物理密度，则加状态相关的 `−sum(log((U−L)/2))`，并重新定义熵目标，否则边界支持宽度会引入额外偏好。

L/U退化或上一目标超界时直接报合同错误；不以epsilon掩盖非法初态。硬目标速率限制不等于实际关节速度/力矩/接触安全保证。S1与S0、官方全范围探索是不同实验条件，不能用S1成功单独归因FlashSAC。可在同一S1动作合同下比较原WM路线和FlashSAC，才能隔离算法影响。

**实施选择**：本轮只记录方案。先做S0接口smoke并允许其发送前停止；准备长训时必须在“继续接受S0软约束且可能中止”与“采用新S1合同”之间明确选择。已有用户坚持绝对位置且无隐藏裁剪的记录不构成自动批准S1。无需为了本文分析向用户重复索要已存在的物理参数。

## 8. 最小可运行范围与阶段门槛

| 阶段 | 交付 | 必须通过的门槛 |
|---|---|---|
| P0 合同冻结 | 新配置、manifest、官方代码pin、旧改动hash清单；纯数据适配 | 47g/25Hz/随机化/坐标/bias/reset检查；不同合同checkpoint拒绝；不依赖WM加载 |
| P1 合成语义测试 | 1-step目标、概率、BN、双Q/EMA、奖励风险单元检查 | 手算terminal/timeout；tanh梯度有限；分布归一化；选最小Q整分布；α更新方向；Actor梯度不进入WM或Q参数；目标Q BN/EMA符合固定实现 |
| P2 Newton小闭环 | 64环境×1024步，预热及少量真实更新、保存/重载、视频 | 正常/强制timeout/受控掉落终态均无reset泄漏；history非done完全一致；reset关节物体速度为零；目标/bias误差≤2e−6 rad；无容量/数值/机械保护触发，或明确以动作发送前拒绝结束而非记为通过 |
| P3 1M诊断 | 1024环境，1-step、FP32、S0或明确选定S1 | 真实奖励/风险一致；Q/α有限、投影边界可控；独立确定性评估未靠冻结HOME伪进展；统计吞吐/峰值内存 |
| P4 5M×3种子 | 同合同，统一评估和日志 | 成功/角误差进展超过HOME，同时无硬保护失败；按下述独立评估门槛筛选；失败run保留，不自动重启补“成功种子” |
| P5 3-step及规模对照 | 逐样本discount修复；单因素n=1/3 | 手算短timeout通过，n-step不跨env/episode；同预算对照后再AMP/compile/10M replay，不一次更换全部开关 |
| P6 50M验证 | 3种子阶段性，必要时5种子对齐论文级预算 | 基于P4/P5决定；仍不等于真机部署验收 |

最小闭环不包含手模型B、ContactSDF、模型辅助Q、Actor模型反传、视觉CNN、分布式多GPU、自动约束乘子、优先replay或新reset课程。不是删除这些研究，而是先获得可解释的model-free基线。

## 9. 预算、内存与评估设计

### 9.1 训练预算（新计划，非本轮已执行）

正式1024环境，预热98向量步=100,352转移；为可复核计数，规定这98步全部不更新，第99步起每向量步2次Q更新、每2次Q更新1次Actor/α。1-step计数如下，n=3另计在途尾项。

| 预算/种子 | 向量步 | 真实转移 | Q更新 | Actor/α更新 | 用途 |
|---|---:|---:|---:|---:|---|
| 约1M | 1,024 | 1,048,576 | 1,852 | 926 | 一个种子诊断，不下效果结论 |
| 约5M | 4,883 | 5,000,192 | 9,570 | 4,785 | 3种子筛选 |
| 约10M | 9,766 | 10,000,384 | 19,336 | 9,668 | n-step或尺度对照 |
| 约50M | 48,829 | 50,000,896 | 97,462 | 48,731 | 官方GPU脚本量级；官方5种子总约250M |

起始训练seed建议42/43/44；新开发评估seed166/167，封存最终seed266/267/268，避免继续用已经看过的66/67调参。不同算法共享初始化域/目标seed，但策略导致轨迹不同，不要求逐步状态相同。

分配建议：P2约0.066M；P3一个1.049M；P4三个独立5M共15.001M；P5两个配置×3种子×10M上限60.002M；只有门槛通过才进入P6三个50M=150.003M（扩五种子250.004M）。这是顺序解锁上限，不是自动排队执行。若P3续到5M可节省重复预算，但其学习率终点必须一开始按5M固定；不能将短程已衰减到终值的策略当公平长程续训。

本机只读查询为RTX5090、32607 MiB显存。旧149D实验约8.5k转移/秒，仅供粗估，不外推为新吞吐。以新有效吞吐2k–8k/s暂估，1M约2.2–8.7分钟，5M约10.4–41.7分钟，50M约1.7–6.9小时/种子；均不含初始化、编译、评估和视频。P3测得端到端速率后替换估计。视频周期若定义新一轮=一向量步，则50M约97次视频，另计成本；同时记录与旧500轮≈1.28M转移的节奏差别。

初步执行时间帽建议P2 20分钟、P3 45分钟、5M每种子2小时、50M每种子12小时（含评估/录像；测量后可重新预注册）。达帽保存并标记预算未完成，不自动扩时或重启。完整阶段开发工时粗估3–5人日：合同/适配1日、算法/动作/终态测试1–2日、评估/恢复/吞吐1–2日；这是工程估算，不是已有实现的完成度。

### 9.2 replay内存不能照搬10M

963D当前/下一观测float32、21D动作、reward/两flag等，基础约 `4*(2*963+21+3)=7800 bytes/transition`：1M约7.8 GB（7.26 GiB），10M约78 GB（72.6 GiB），尚未加审计字段、采样batch、优化器、仿真、CUDA graph。当前32GB卡不适合直接10M全GPU历史replay。

因此首版1M GPU并记录峰值；要求峰值≤可用显存80%，不足时用CPU pinned replay并评估拷贝开销，不能静默丢历史或特权字段。后续可按env/episode保存81D帧并重建10帧历史，终态单独存，带generation防环形覆盖；该优化必须验证重建与显式历史逐元素等价。不能用float16压缩未验证地改变低幅噪声/速度信息，也不能用少容量结果声称完成10M replay官方配置。

### 9.3 独立评估与接受标准

每500轮录像单环境500帧用于可视检查；每约1M转移用独立进程64环境×640步、两个开发seed做快速评估。固定checkpoint，确定性tanh(mean)，保留物理/感知随机化，BN/reward normalizer冻结，检查文件hash及模型buffer不变。行为探索噪声置零不等于关闭感知噪声。

最终评估每个封存seed1024个完整首回合；正常timeout500步，成功不提前停，drop则记该回合终止，后续autoreset回合不混入首回合成功率。三seed最多1,536,000评估转移/检查点。报告回合成功比例（连续保持13步/0.52s、阈值按当前配置）、掉落比例、timeout、有效步平均及回合末SO(3)测地角误差、五项违规率、目标变化P95/P99/max、真实关节速度/扭矩/机械越限、存活时间；不做骰子对称折叠。统计按回合/随机域聚类区间，不能把相邻控制步视独立样本。

预注册“继续投入”门槛（工程选择，非论文承诺）：至少2/3训练种子在独立评估表现出可重复姿态进展，且成功率相对HOME的95%区间支持改善；建议实用筛选线成功≥5%、掉落≤5%、实际关节速度违规≤1%、零机械/数值/容量失败。S1目标速率超限应为0（仅容许明确浮点容差）；S0若触发发送前拒绝则不算完成安全评估。仅零掉落但一直HOME不能通过。5M不通过时先分类失败原因，不声称FlashSAC整体无效，也不无条件投50M。

对照分层：①固定HOME；②已冻结v32 model1000/2500历史结果（不同样本/动作合同只做参考）；③同一新合同下FlashSAC；④后续需要因果比较时，才从头跑同预算WM与FlashSAC。每次只改变n-step、风险权重、探索目标或动作合同其中一个因素。原20Hz149D FlashSAC A/B不能当当前环境基线：其2.097M、1-step、1次更新/向量步、单种子、无部署观测约束的结果仅说明短程未观察到手模型收益。

## 10. 保存、回退与未解决问题

新checkpoint独立标识如 `flashsac_revo3_die47_hist25_v1`，保存Actor、双Q、target Q、α、优化器/调度器、BN、reward normalizer、步数、RNG、动作合同、资产/config/vendor hash。checkpoint不含WM。完整训练续接还需replay、n-step队列、探索噪声状态；若物理与传感器状态无法完整恢复，则明示“重建环境的统计续训”，清空跨回合在途项，不能承诺逐位续接。保存Actor导出需保留输入布局、缩放、BN和HOME/bias及S0/S1映射。

回退通过选择旧入口/旧冻结源码及原检查点进行，不覆盖旧run，不把新replay灌回v32。新实验任一失败保留failure snapshot、最后完整checkpoint、实际计数、退出码和原因；不得自动恢复旧训练。新的部署/真实硬件验证不在本轮范围。

尚待实现验证的风险：带噪终态历史生命周期、输入BN低方差、HOME初始化与UnitLinear相容性、S1动作合同是否被采纳、n-step有效折扣补丁、reward normalizer被终止重罚主导、32GB显存下的replay与仿真共存。当前有充分证据支持独立建设model-free基线，尚无证据证明它会学会该47g固定掌内任务。

本轮完成的是本地源码/配置/历史证据审阅、诊断图查看、线上一手来源及固定SHA核对和本文方案。未运行新训练、未做新策略效果测试；文中所有阶段门槛、预算和新增路径均为拟议方案。
