# Isaac Lab v3.0.0 / Newton 迁移

## 2026-11-05 新版50 mm / 100 g骰子与周期录制

用户指定改用原70 mm造型等比缩小至50 mm的已接受版本，并给出名义质量100 g。
生成独立SI USD，视觉与200顶点外凸包同时乘5/7，惯量按 `(0.1/m70)*(5/7)^2` 缩放，
保留主轴/相对质量分布假设。不是对打印填充惯量的实测。原70 mm与vendor资产保留。
新旧环境完整比较只改变物体和目标标记USD路径，算法Config无变化；原质量随机化0.8–1.2、
摩擦、手部、25/200/400 Hz、健康阈值与容量保持，未改Newton/Isaac Lab引擎。

按本地固定版本官方资产准备/启动流程，在Newton+MJWarp完成64环境2轮预检及500帧视频，
并独立seed44运行HOME Actor 64环境×500步。运行时凸包、质量/惯量、奖励/观测核对通过；
2次掉落、62次超时、0次姿态成功，未出现容量/机械健康失败。新Actor尚未训练，
此结果只验证资产/链路与初始化基线，不证明任务收敛；未做PhysX对照或新接触参数标定。

manual-convex训练默认每500轮先保存完整检查点，再独立进程录制20秒策略视频。
修复旧录制脚本的SolverBudgetBinding误用与typed schema加载缺失，恢复保存的环境合同。
录制前后全部checkpoint内容一致，包含优化器与CPU/CUDA RNG；视频逐帧解码通过。
104项相关检查通过。正式长训练及真机未启动，多GPU录制分支未实机验证。
详见 [本轮验证](versions/v30_video_die50_20261105/README.md) 与 [网络说明第15、16节](revo3_network_v30_zh.md)。

## 2026-11-05 v29：修复代价 Critic 自举失稳

v28 正式训练在第 2114 轮后由机械越限保护停止。独立诊断确认 Actor loss 1025
主要来自失真的代价 Critic；原始快照与诊断保存在 `training/diagnostics/actor_loss_v28_20261105/`。
新版本保留所有原始物理阈值与统计，以有界单调运行风险、每头一次最坏失败处罚和
一致的终止权重替代无终止的成本 rollout。在线/目标 Critic 保持 `[512,256,128]`，
目标参数 EMA=0.02，全部六个头用真实 pre-reset TD 和固定上界归一化 BCE 训练。
不再用想象数据训练 v29 Critic。前 500 轮预热 WM 与 Critic；旧版本路径保留。

验证：399 项回归通过；64 环境、80 轮完整网络/生产 batch/CUDA Graph 联合更新通过，
Actor loss 0.1570–0.5154，末轮梯度范数 0.3107，无容量或机械健康失败；随后实际恢复
两轮，四网络 digest 完整匹配，WM/Actor/Critic 更新计数 820/82/1640。
证据见 `docs/versions/v29_20261105/validation.json`，完整合同见
[v29 网络说明](revo3_network_v29_zh.md)。新 loss 的数值不可直接与 v28 比较，
实际姿态成功率与速度达标仍须长期验证。没有改引擎、碰撞、随机化或放宽健康检查，
也未声称解决先前的物理机械越界。

## 2026-11-05 v28：WM 任务一致性监督与真实终止学习

在 v27 Git 基线上加入奖励、干净观测、终止 BCE 和五项代价误差。
奖励/观测/代价保留解析读出，以其误差辅助训练物理状态预测；新增 57→64→64→1
终止头，想象中的继续概率改为 1−学习终止概率。旧 Config 缺少开关时保持 v27
行为，新启动默认 checkpoint 28。详情见 [WM v28](revo3_wm_v28.md) 和
[完整网络说明](revo3_network_v28_zh.md)。

监督使用 reset 前物理真值与真实 `terminated`；timeout 不视为掉落正样本，
reset 后样本不进入同一 WM 序列。优先采样的重要性权重应用到全部损失；
未校正的事件档案不混入 v28 WM。原速度/力矩参数、物理引擎、随机化、
碰撞和 fail-fast 健康检查不变。冻结模型的新真实数据审计加入 NLL/Brier/ECE
和正负类指标，并提供独立环境分组的后处理概率校准。

恢复校准验证发现，旧初始化 schema 的 object_out_of_reach 没有手中 origin 参数，
因此官方 from_dict 拒绝新 YAML。训练入口现根据保存的 limit_contract 先实例化
对应的既有 typed schema，再应用 SavedEnvironmentBinding；仍严格核对完整环境
合同，没有删除保存字段或忽略不一致。真实 v28 resume 校准路径已通过。

验证：196 项单元/回归测试通过，包含新增损失与 CUDA Graph/eager 等价检查；
另外执行全部 10 项 GPU/CPU Replay 测试，涵盖前一组跳过的 4 项 GPU 检查。
官方 SimulationRuntime/newton_mjwarp 完成 4 环境、2 轮完整网络更新，WM/Actor/Critic
计数 2/2/4，真实奖励核对差 0、成功边界差 0；144 个手部凸包审计通过，
接触峰值 candidates/contacts/constraints=338/41/51，无容量/机械健康异常。
冻结模型后另外收集 64 步验证校准流程，只有独立测试组出现 1 个掉落，拟合/选择组
均无正样本，因此明确返回样本不足且不写 calibrated.pt。不能声称概率已经校准。
仅存在已知 OmniPBR MDL 搜索路径提示。证据见
`docs/versions/v28_20261105/validation.json` 及其中指向的真实运行目录。
未重启正式长训练；旧 v27 第 1656 轮机械越界仍是未解决的独立问题。

## 2026-11-05 正式训练启动与 EPA horizon 容量补丁

用户要求开始训练后，以最新姿态奖励/五项代价配置从头启动 1024 环境、10000 轮。
第一次运行 `orientation_costs_v27_1024_10000_20261105_161219` 完成第 107 轮后，
world 34 真实采集触发 `EPA_HORIZON`（bit 256），健康检查保存现场并停止；
Actor/Critic 尚未更新。碰撞临时数组的边容量固定为 24，不由 nconmax/njmax
或约束迭代数控制，不能通过增加这些数值解决该容量问题。

已对本地安装 `mujoco_warp/_src/types.py` 应用单行补丁，将
`MJ_MAX_EPAHORIZON` 从 24 扩到 64；这是对固定版本依赖的明确本地修改。
版本化补丁 `patches/mujoco_warp_epa_horizon_64.patch` 与配置
`config/mjwarp_capacity_patch.json` 保存原/新 SHA256。`supermimic/runtime_capacity.py`
启动时核对常量、导入别名和源文件哈希；补丁记录进入 context manifest，
检查点严格核对运行合同。官方 SimulationRuntime / newton_mjwarp 路径保留，
几何、域随机化、PD、求解参数和健康检查不变，未通过忽略溢出继续训练。

验证：159 项测试通过；保存的 40 个控制边界上 528 组 GPU 凸包距离查询，
24/64 两种容量的结果完全相同且无溢出。未保存失败中间积分子步，
故静态对照不是精确重放该失败的证明。新运行
`orientation_costs_epa64_1024_10000_20261105_162115` 使用同一 seed 42，
已完成 500 轮 WM 预热并超过 760 轮，Actor/WM/Critic 均在更新。
该运行内尚未出现容量或机械异常，不能据此保证之后 10000 轮全部稳定。

诊断、备份与复现脚本：项目 `experiments/epa_failure_20261105/`。
启动证据：`training/jobs/orientation_costs_epa64_1024_10000_20261105_162115/startup_validation.json`。
当前运行 `training/current`、作业 `training/current_job`，TensorBoard 端口 6009。
保留 25 Hz 控制 / 200 Hz PD / 400 Hz 积分、100 rpm 声明速度和 1.28 N·m 力矩限幅；
速度字段仍不构成真实关节硬限速，代价也不提供硬约束保证。每 100 轮保存检查点。

## 2026-11-05 稳定性和掉落写入独立代价（用户已确认方案）

当前 25 Hz 新训练默认 `object_hold_costs_enabled=True`，姿态到达仍使用原双尺度 SO(3)
奖励；关闭正向稳定奖励，避免重复计费。成功仍需要 13 步（0.52 s）满足姿态、
物体角/线速度阈值且无终止，真实掉落终止仍保留。位置跟踪继续关闭。

除原关节速度/目标变化速度两项代价，增加：

```
g = exp(-(theta / (pi/12))**2)
c_object_angular_speed = g * ReLU(norm(omega)/0.5 - 1)**2
c_object_linear_speed  = g * ReLU(norm(v)/0.05 - 1)**2
c_drop = ReLU(norm(p-hand_center)/0.3 - 1)**2
```

使用 reset 前物理状态、SI 单位和现有手中原点。g 随姿态接近而平滑增强，
保留完整角度梯度，不使用时间/成功锁存；c_drop 不乘 g。手中范围内的不同位置
仍不受位置跟踪约束。c_drop 只在原空间终止边界外计费；提前趋避仍由原软生存
概率提供梯度。代价学习不替代物理硬限幅或保证成功，权重间的行为权衡需训练验证。

Critic 保留 [512,256,128] 主干，末层为奖励加五代价，共六个值；代价头使用 Softplus。
Actor、WM、观察布局和物理频率不变；奖励归一化上界同步为每步 0.4。
五项代价分别构造 λ-return，想象成本不因预测掉落而消失。真实日志中的速度 cost sum
包含 g 权重，violations/max_ratio 则保留未加权物理超限，单列 object_hold_cost_gate_mean。
开关和代价名称、协议、头数保存在 config/metadata；旧配置缺少开关时仍构造三头 Critic，
不能把旧末层加载成六头。`--no-object_hold_costs` 可显式保留旧两代价方案；resume 不允许
静默更换代价头。原 100 rpm 声明速度上限、1.28 N·m 驱动力矩限幅及 vendor/引擎未改动。

实际验证：159 项测试通过，包含独立标量预期、零代价区、阈值外梯度、连续姿态权重、
掉落恢复梯度、位置自由度、六头可微 Actor/Critic 更新及旧三头权重严格加载。
官方 SimulationRuntime + newton_mjwarp 完成 4 环境、2 轮短验证，启用 WM CUDA Graph；
WM/Actor/Critic 更新 2/2/4 次，真实与解析奖励最大差 3.73e-9，保持计数差 0，
成功边界不一致 0。144 手部凸包及物体几何审计通过；接触峰值 candidates/contacts/
constraints 为 322/36/51，日志无容量溢出或非有限异常。
证据：`training/validation_object_hold_costs_tests.log` 与
`training/runs/object_hold_costs_validation_20261105/{config.json,validation.json,metrics.jsonl}`。
本节记录实现与短验证；后续正式启动及遇到的容量问题见顶部新增记录。

## 2026-11-05 取消位置跟踪，保留姿态稳定保持与掉落终止

按用户要求，新建手中 25 Hz 训练配置使用 `track_object_position=False`。
该开关统一关闭独立位置奖励、稳定奖励中的位置误差项和成功判定的位置门槛，
真实回调与 WM 想象路径共享实现。保留姿态奖励 10、稳定奖励 3，以及实际关节速度/
目标变化速度两项代价；constrained BPTT 的每步奖励归一化上界同步从 0.56 改为 0.52。
成功条件为姿态误差 <0.1 rad、物体角速度 <0.5 rad/s、线速度 <0.05 m/s，
连续 13 个控制样本（0.52 s），终止和超时样本不能完成成功事件。
成功后不换目标、不停止奖励，继续保持。物体位置观测与 WM 位置预测保留；
手中原点半径 0.3 m 掉落检测、软生存概率、随机化、关节顺序及碰撞模型均保留。
位置目标槽位为接口兼容保留，新任务不对该位置施加奖励或成功门槛。

训练日志在新模式记录 `task_mean_hand_center_distance_m`，不再把它称为目标位置误差；
单步稳定率同样采用三项阈值且排除结束样本。开关序列化到 checkpoint、env 和任务元数据。
历史 Config/YAML 缺少该字段时解释为 True，哈希只省略此等价旧默认值；
False 是不同的环境合同，旧权重不能被静默作为新任务续训。
未更改引擎、vendor 资产、25/200/400 Hz 时序或上一步的 100 rpm/1.28 N·m 参数。

实际验证：`training/validation_orientation_only_tests.log` 记录 138 项通过，
覆盖位置变化的奖励不变性/零位置梯度、13 步保持、物体速度梯度、终止阻断、
掉落生存概率及旧配置哈希兼容。官方 SimulationRuntime + `physics=newton_mjwarp`
完成 4 环境、2 轮短验证，启用 WM CUDA Graph，WM/Actor/Critic 更新次数为 2/2/4，
5 次 reset；真实奖励与解析奖励最大差 0，保持计数差 0，成功边界不一致 0。
144 手部凸包和物体几何审计通过，日志无接触容量溢出；仅有已知 OmniPBR 材质解析提示。
证据在 `training/runs/orientation_only_validation_20261105/{config.json,env.yaml,validation.json}`。
短验证不表示策略已学会保持，也不是之前机械越界问题的稳定性修复证明；未启动正式长训练。

## 2026-11-04 用户定义的独立训练软限位

当前 `config/revo3_training_limits.json` 按用户指定值定义 21 关节的训练编码器范围；
`config/revo3_sdk_limits.json` 仍是 SDK 2.3.2 / UT2 固件 0.1.5 的只读回读快照。
候选生成（含 Dexonomy）、动作解码、编码器归一化、WM 动作解码和抓取合同共用训练配置。
训练 JSON 独立绑定 SDK 快照及 PACE fit_03 参数哈希，checkpoint/缓存记录两层来源。
辨识参数不变；其 hand.json 旧限位仅为辨识来源记录，不能覆盖当前训练范围。
USD/URDF 机械范围、碰撞几何及 Newton 引擎配置保持不变。

新的 21 个训练区间全部包含于 SDK 区间；加已有 encoder_bias 后也全部处于当前 URDF 范围内。
原始 SDK 与 URDF 并非全体严格嵌套，10 个关节有超过 0.02 度的数值越界；
三层包含图表达设计目标，并标注当前例外，不伪造机械或设备限位。
训练限位与辨识参数一起交付于项目 training/hand_limits/ 下。

本轮 58 项离线单元/回归测试通过，覆盖精确用户值、关节顺序、单位、SDK 包含性、
辨识偏置后的机械包含性、错误来源拒绝与归一化/解码端点。按用户要求未启动训练、
物理认证或硬件运动；新范围下的物理稳定性、旧抓取重新认证和学习性能尚未验证。


## 2026-11-04 Dexonomy 的 Revo3 小规模适配实验

隔离实验位于 `/home/brainco/reorientation/experiments/dexonomy/`，完整方法边界、
命令与结果见该目录 `README.md` / `summary.json`。上游固定
`JYChen18/Dexonomy@d561ff9df8545deef7a3f9e7cfb4045ef66f6913`；实际调用上游
HandObjMatcher、Warp 查询核和 ContactQP。仅在本地子类构造器适配 Warp 1.17
公开 API；上游源码、引擎、vendor 均未修改，附加包隔离到实验目录。

本次是现有 4 个 Revo3 模板 → 上游物体拟合 → 当前固定手掌局部优化 → 原 Newton
认证，不是上游浮动手掌优化器、完整抓取分类及 contact-aware controller 的复现。
32 个候选中 29 个通过几何/QP，4 个通过 14 项 × 4 context 的原门限试验，
去重后新增 3 个。相同初值、对齐环境槽位的无拟合对照也新增 3 个，尚无通过率优势。
新增 Dexonomy 状态离旧缓存最近朝向约 35.7°/64.3°/114.4°（不消除物体对称性），
对照约 2.9°/33.4°/5.9°。主要失败为沉降后穿透超限，未放宽阈值。

`experiments/dexonomy/bank_merged.json` 含原 4 + 新 3 条，train 3 / validation 4，
仍为 2 个来源组；SHA256 `bd1ff26414882246c604f57e1a44738cea7b9a595e723922b2ebf372c17e2356`。
该文件可用于新的训练运行；原缓存和旧 checkpoint 保留。已完成 3 项适配测试及
1 轮真实 smoke 训练/holdout（WM/Actor/Critic 1/1/2 次更新、18 replay 时间槽、
2 次 reset）。没有策略性能提升结论；这次不需要更改策略、奖励或现有训练实现。
当前 144 手部凸包和物体的最终 solver 几何审计通过；认证与训练日志无容量溢出。

## 2026-11-04 当前 RoundedDie 稳态抓取缓存与自适应采样

已在当前手动凸包任务完成候选生成 → Newton 物理认证 → 缓存 reset/单目标 →
真实轨迹 replay → 训练与 checkpoint 的小规模迁移。保留 v25、s84/xi65、
81×10 Actor、465 WM 输入、453 Critic、单 tanh 绝对编码器目标和原密集保持奖励。
没有修改 Isaac Lab/Newton/MJWarp/vendor，没有控制真机或启动正式长训练。
启动器仍是官方 `launch_simulation`，`physics=newton_mjwarp`；固定
IsaacSim 6.2.0.0 / isaaclab 28.0.0 / Newton 1.6.1 / Warp 1.17.0。

最终可用文件（以下 ROOT 为 `/home/brainco/reorientation`）：

- `ROOT/training/grasps/bank_final_v2.json`：4 条，train 2 / validation 2 / test 0，
  两个独立来源组不跨 split；SHA256
  `481418cc56fb8eac6239f07a2b88d1783b0b2976eae573c175fc21a89b83ea78`。
- 同名 `.report.json` / `.env.yaml` / `.solver.mjb`：实际 context、每项试验、
  几何/容量审计、官方环境配置与只读导出的最终求解器碰撞模型。
- `ROOT/training/grasps/query_fk_audit.json`：5 个姿态的独立查询 FK 核对，
  最大位置差 1.67e-8 m，四元数分量差 2.15e-7。

旧项目 `q_actual` 经核实是直接写入 MuJoCo qpos 的几何角，旧 XML 无额外
reference-angle shift；只迁移几何种子，WXYZ 转 XYZW，不导入旧 hold target。
候选重新使用当前 USD 的 144 个凸包与 RoundedDie 凸包优化，逐凸包穿透残差
避免仅优化最深一个接触时的切换。SciPy bounded nonnegative wrench least-squares
只给候选排序：不是上游完整 GraspQP/MALA，也不是稳定性证书。

诊断过程完整保留：v1 的 8 个候选有 3 个通过初始几何、1 个通过姿态保持，
但沉降后查询穿透约 5.3 mm，已排除。v2 的 24 个和 v3 的 18 个候选各产生
2 个满足当时筛选的状态。扩大 context 并再次沉降会改变 canonical q/pose，
部分状态因此超过穿透门限；最终直接复测这 4 个**已有真实沉降记录的原始状态**，
使用 `--preserve_reset_state`，不改写它们的 q、姿态或 hold target。此参数只省略
预备沉降，每项认证仍重新冷 reset 并清空执行器缓存。早期 bank_v1/v2/v3 和
bank_final 是诊断中间文件，不能替代最终缓存；严格的新校验会拒绝其缺失证据。

最终每条在 4 个不同的实际随机化 xi65 context 下完成 14 项试验：冷 reset
2 s、保持 5 s、线速度 3 轴 ±0.05 m/s 和角速度 3 轴 ±0.5 rad/s 各 5 s。
所有 trial 都从同一个保存状态、零关节速度、空动作/延迟历史重新开始。
20 Hz 采样记录每条试验的全程最大漂移/速度及真实 solver 接触，不用末帧均值
掩盖中途失败，不允许任何 done。门限保持为 3 mm / 5° / 0.05 m/s / 0.5 rad/s；
另要求 reset 查询穿透 ≤1.5 mm、solver 物体接触穿透 ≤1.5 mm、接触采样占比 ≥90%。
这些是有限样本的经验性物理验证，不是全随机化域的数学证书。

最终 4 条、全部 context/trial 的最大值：位置漂移 1.402 mm、朝向漂移 4.397°、
线速度 0.005645 m/s、角速度 0.20443 rad/s、solver 物体穿透 0.662 mm；
物体接触采样占比均为 100%。真实手部 144 实例/90 唯一凸包/9678 顶点的审计保留；
额外对比最终 MJWarp 编译后的顶点和 support hull，手部最大差 2.63e-9 m，
物体最大差 2.31e-14 m，没有用简化球体替代。16 环境认证峰值：全局候选 1228、
全局接触 177、单环境约束 45，overflow 标志为 0，运行日志没有容量溢出。
依旧存在原有 OmniPBR 可视材质解析提示，它不改变碰撞网格。

缓存 schema 绑定资产字节哈希、几何摘要、关节顺序/完整限位、XYZW/base frame、
冻结辨识参数、软件版本、完整环境合同、物理配置（含容量）和实际试验状态摘要。
无证书、篡改状态、不同物理合同、重复 ID、跨 split 来源组会被拒绝。
姿态/位置/关节去重作用于缓存全体，保留来源组；静态缓存不写成 WM transition。

`GraspResetManager` 仅在官方 Recorder `record_post_reset` 中写入抓取起点和
持久目标；首次装配后重建 Adapter 的观测历史和传感器状态。reset 的 last_action=0、
ready=0；验证时显式发 hold action，训练时只执行网络动作，不强塞 hold target。
终止前复制起点/目标 ID；next_physical 仍在自动 reset 前捕获。成功只更新统计，
同回合始终保持同一个目标，中间抓取可以作为下一段训练的起点或终点。

PairSampler 先均匀抽来源组，再抽起点；相对朝向按 15/45/90/135° 分 5 桶，
EMA 成功率与访问次数调整桶权重，同时保留 20% 均匀目标底座。
checkpoint 保存 bank 内容哈希、split、统计与 NumPy RNG。恢复训练先恢复统计再
抽新回合；validation/test 评估核对同一个 bank 哈希，但不继承训练难度/RNG。
当前抓取采样明确仅支持单进程，尚未实现跨 GPU 统计同步。
最终训练两态差约 13.4°、验证两态差约 25.7°，大角度覆盖仍不足。

`--adaptive_replay` 对真实连续 WM 窗口按误差 priority^0.6 采样，混入 20% 均匀
概率，并用精确 1/(N*p) 权重校正 WM loss；每条窗口误差更新 priority。
终止之后 mask=0，slot generation token 拒绝环形覆盖后的过期更新。
Actor 起点继续均匀采样。启用时关闭旧固定 25% event archive，避免错用 IS 权重；
未启用时旧行为保持。恢复 checkpoint 后仿真/replay 重新预热，不声称 bitwise 续跑。

复现入口：

```bash
ROOT=/home/brainco/reorientation
# 生成候选，seed_bank 可选旧种子或当前缓存；只读输入
$ROOT/training/run.sh generate_grasps --candidates 24 --max_nfev 80 --seed 43 \
  --seed_bank <只读种子.json> --output <候选.json>
# 普通候选先真实沉降，再认证保存的状态
$ROOT/training/run.sh certify_grasps --candidates <候选.json> --output <缓存.json> \
  --num_envs 16 --contexts 4 --seed 47
# 本次最终 4 个已经沉降的固定状态之原样复测
$ROOT/training/run.sh certify_grasps --candidates $ROOT/training/grasps/candidates_final.json \
  --output <新复测缓存.json> --num_envs 16 --contexts 4 --seed 47 --preserve_reset_state
$ROOT/training/run.sh train --num_envs 4 --iterations 2 --smoke --audit --holdout --eager_geometry \
  --grasp_bank $ROOT/training/grasps/bank_final_v2.json --adaptive_grasps --adaptive_replay \
  --output_dir <新训练目录>
```

214 项相关 CPU 回归通过（`ROOT/training/grasp_regression_final.log`），涵盖老任务、
网络/奖励/观测契约、证书拒绝、来源组、固定目标、IS 分布、过期 priority、统计保存恢复。
最终缓存的真实训练/恢复证据：

- `ROOT/training/runs/grasp_bank_smoke_final`：4 环境、2 轮，WM/Actor/Critic
  更新 2/2/4 次，真实 replay 21 个时间槽，强制 reset 2 次，event archive=0。
- `ROOT/training/runs/grasp_bank_restore_final`：恢复后第 3 轮，更新累计 3/3/6 次；
  抓取已完成回合统计从 2 续至 4，bank SHA 保持一致。
- `ROOT/training/runs/grasp_bank_validation_final/evaluation.json`：validation split
  420 控制步 × 4 环境 =1680 个真实 transition，3 次掉落、2 次正常 20 s 截断，
  10 s 之后的诊断采样 511 次；采样难度 count 仍为全 0，未继承训练统计。
  3 次更新的未训练充分策略成功数为 0，不能据此声称训练性能已达标。
- 三个运行与认证的日志均无 overflow、非有限状态或断言失败。
  认证容量峰值见上文；训练/恢复/验证峰值分别为 374/103/123、399/105/102、
  296/63/66（全局候选/全局接触/单环境约束）。
- 汇总：`ROOT/training/validation/grasp_migration_summary.json`。

恢复命令（输出目录必须是新目录）：

```bash
$ROOT/training/run.sh train --num_envs 4 --iterations 3 --audit \
  --grasp_bank $ROOT/training/grasps/bank_final_v2.json \
  --resume $ROOT/training/runs/grasp_bank_smoke_final/latest.pt --output_dir <新恢复目录>
$ROOT/training/run.sh train --num_envs 4 --evaluate --eval_steps 420 --grasp_split validation \
  --grasp_bank $ROOT/training/grasps/bank_final_v2.json \
  --resume $ROOT/training/runs/grasp_bank_restore_final/latest.pt --output_dir <新验证目录>
```


## 2026-11-02 共同 AprilTag 的真仿相机外参复测

重新采集 D435i 261722074750 的 100 帧 1280×720 RGB，使用双方相同的
tag36h11 ID0、48 mm 黑边及既有固定 T_base_tag，70 帧拟合、30 帧留出。
真机 PnP 独立求得相机到手基座位姿；仿真位姿从保存样本的世界/基座/相机
矩阵还原。相机位置差 0.150 mm、旋转差 0.0369°，低于本次逐帧重复偏差
P95 0.542 mm / 0.0760°，未测出显著外参差异。不能将此解释为绝对精度。
因此此前仅凭整幅画面认为真仿外参不同的判断应撤回。

仿真已知位姿直接投影到真机原生图，与平均检测角点 RMS 差 0.305 px。
单独用低分辨率仿真标签 PnP 相对其已知位姿却有 9.70 mm / 1.44° 偏差，
对应角点栅格化/检测差约 0.556 px，故未将该图像 PnP 偏差当作实际外参差。
证据保存于 `outputs/live_yellow_20261102/tag_extrinsics_comparison/`。
假定标签尺寸及安装未变；没有改写外参、仿真引擎、资产或训练，没有机器人动作。

## 2026-11-02 真机 RGB-D 部署坐标审计

对 D435i 261722074750 的 1024 个实际输入点分别调用 RealSense SDK 反投影、
depth-to-color 变换和 RGB 投影。部署点与 SDK 的最大计算差异 0.0000351 mm，
RGB 投影最大差异 0.0000298 px；未发现三维外参方向、矩阵转置或单位错误。
这是与工厂标定计算的一致性，不是相机物理标定精度验证。

查出部署 RGB 图像 warp 漏掉 SDK/OpenCV 整数像素中心到训练连续像素中心的
半像素转换。实时预览已将每轴图像平移 -0.25 px 修正为 -0.75 px；模型 UV、
真实 XYZ 和权重不变。坐标标尺最大误差由约 0.500 px 降至 0.000061 px。
同一组 60 帧仅改变 RGB warp，网络朝向平均变化 9.585°（不是准确率提升），
关键点刚性拟合残差 25.378→24.037 mm，尚未解决实机识别误差。

另外确认 `examples/yellow_rgbd/pair.npz` 是整数像素反投影的旧示例，与正式
采集修正后的 x+0.5/y+0.5 约定不同；其变换闭环与关键点次序正确。原文件
未覆盖，修正射线的副本仅作消融，单帧真值旋转误差仍约 6.38°。
验证脚本和报告保存在 `outputs/live_yellow_20261102/coordinate_audit/`。
未修改 Isaac Lab/Newton 引擎、训练、随机化或资产，未运行机器人控制；
相机到机器人基座的物理标定及真实骰子面号语义未在本次独立验证。
实时重启时另发现旧提示点偏离移动后的骰子，SAM 分割混入夹爪；已重新框选、
移除旧点提示，并在预览中直接显示实际 mask，便于识别对象选择失败。

## 2026-11-02 黄色骰子可移植推理发布

发布分支保留第 32100 轮 Actor 原始字节，附带第 1943 轮最新快照与第 1887 轮最佳预测器推理权重。仅剔除优化器状态，模型张量逐项相等。SavedEnvironmentBinding 在本发布副本中解析当前仓库内的相同资产，同时保留原环境合同身份；动力学和几何检查不放宽。CPU 上验证三份配置的路径迁移、Actor strict load、几何摘要、错误路径/动力学拒绝和一个真实采集样本的融合推理。未启动新的 GPU play，没有改训练环境。详细记录见 `release/validation.json`，使用说明见根 README。


## 2026-11-02 当前预测器连续轨迹视频

`record_yellow_fusion_preview.py` 在 CPU、2 线程、nice=15 下运行独立推理，未占用
训练 GPU。从原子更新的 best.pt 复制固定快照：epoch 1808 / step 72320，
SHA256 `affafee8d11159ace0ca7c6a8ccbfb61e4be20ceee049d2714b9c672477d8383`。
回放已有 `yellow_capture_dashboard200_v1` 的 seed=42 连续 Actor 32100 轨迹，
200 帧、10 fps、20 秒；没有重新启动仿真，没有将预测反馈给 Actor。
与完整训练/验证索引逐项对照，seed 不重叠、pair 哈希零重叠；相机和 Actor
身份一致，全部源 pair 哈希符合源 manifest。使用干净 RGB 和旧预览中尚未
加噪的可见点云，输入重投影误差 <1e-4 px。显示采用固定同视角裁剪，
模型仍读取完整 640×360 RGB；无输出平滑和对称姿态取最小误差。

四窗口为原始 play、RGB 输入、同相机点云、绿色 GT / 橙色预测叠加。
20 秒全帧平均位置误差 0.9191 mm、旋转误差 2.8744°，P95 分别为
1.6690 mm / 8.3182°；包含初始预热帧。该连续轨迹预览不是最终独立实机测试。
MP4 已逐帧解码确认 200 帧，回传本地后 SHA256 校验一致。
结果、逐帧预测和固定快照保存在独立 outputs/yellow_predictor_preview_20261102_v1；
本地视频和报告位于实验 samples/ 同名目录。录制完成时预测器训练仍在
第 1813 轮推进，原训练 PID 1426919/1426920 仍存活。

## 2026-11-02 黄色骰子离线 RGB-D 预测器训练适配

新增独立入口 `scripts/pose_estimator/train_yellow_rgbd_fusion.py`，沿用归档的
RGB-D 2,000 epoch 训练器与 ResNet34 / PointNet 融合结构。直接构造旧版
`layer4.1.relu` 特征节点并严格加载 RGB 骨干，避免导入仿真注册代码或下载权重。
RGB 初始化文件 SHA256 为 `6a940233e65146bf34f1051a593727f79fa0821d0d8bcc7feec18b07a67794ed`。
只复制该权重至独立项目；融合头、优化器与 2,000 epoch 余弦调度从头初始化。

训练读取 v2 根索引：20,000 train + 1,000 validation，保留所引用 v1 的
3,000 对样本。启动检查根完成状态、索引哈希/计数、10 个分片 manifest
哈希和校验报告、校准/策略身份、样本路径唯一性、轨迹 seed 与 split 归属。
数据只读，不改动原数据。旧分片 training_ready 字段由根审计结果替代。

使用数据内 70 mm 骰子的 9 点模板。输入 UV 已是连续坐标，因此
`grid_sample(align_corners=False)` 使用 `uv/[width,height]*2-1`，不再加半像素。
训练使用 `rgb` 的采集时 SH 随机化，再加亮度 0.75–1.25 / 对比度 0.85–1.15。
读取 `points_camera_clean_m`，每次 batch 新增标准差 0.5 mm、截断 ±1.5 mm
射线深度噪声和 15% 随机丢点；不对已加噪点云再次叠加。验证使用 clean RGB
和 clean points。audit 深度及真值位姿不进入模型输入，真值仅作为标签。

启动器 `launch_yellow_rgbd_training.py` 锁定物理 GPU 2、4 的 UUID，拒绝已有
计算进程或占用显存的 GPU；只启动离线双卡 DDP，同步训练一个模型。
配置：每卡 batch 256、全局 512、BF16、SyncBN、每卡 8 workers、seed 42、
freeze=0、AdamW(骨干 1e-5/新增层 1e-4, decay 1e-4)、cosine eta_min=1e-6、
SmoothL1 beta=0.1、梯度裁剪 5。进程 nice=10。使用现有 Python 只读运行，
无安装/升级依赖、无仿真启动、无原策略修改。启动/结束记录原受保护文件哈希。

适配预检通过：训练和验证各抽查首/中/末样本，重投影最大误差
3.052e-5 px、刚体标签拟合最大误差 1.564e-8 m；骨干权重严格加载通过。
预检记录在独立实验 `evidence/yellow_fusion_preflight_20261102.json`。
短测试和正式运行各有独立输出目录、guard 状态及日志；短测试结果不能计入正式训练。
双卡短测试已完成 2 次同步更新和全量验证，checkpoint 回读与所有模型张量有限性
检查通过；峰值 allocated 21,968 MiB / reserved 26,646 MiB，guard exit=0，
原受保护文件哈希不变。正式运行 `yellow_fusion_20k_1k_ddp256_2000ep_20261102`
已启动，guard PID 1544851、torchrun PID 1544858；已完成前两轮/80 次同步更新，
每轮均遍历 20,000 对训练和 1,000 对验证，latest.pt / best.pt 已保存。
前两轮训练 loss 0.48636 → 0.35092；仅为启动验证，不能视为最终精度。
启动后再次确认原训练 PID 1426919/1426920 存活，7 个受保护文件哈希不变。
正式运行从 epoch 1 开始，没有载入短测试模型或旧魔方融合 checkpoint。
验证指标为仿真内准确率，尚无真实相机测试结果。

## 2026-11-02 按用户要求切换为双卡各 1,000 环境

已停止上一轮单环境采集调度器及其两个采集 guard，仅按已核验的 PID、
命令和独立输出目录停止自有任务；原训练进程未停止。v1 已完成并校验的
6 个分片、共 3,000 对训练数据保留；未完成分片不计入正式数据。

新运行目录 `outputs/yellow_rgbd_20k_train_1k_val_2x1000_20261102_v2`，
调度器 PID 1539616。物理 GPU 2、4 各一个采集进程，各运行 1,000 个
环境，共 2,000 个策略环境；原 Actor 的 action/history/context 按环境批量
运行，原 20 Hz 控制、stride=2、每次重置后 40 步 warmup 保持不变。
两卡各补采 8,500 对训练数据（seed 8200/8201），然后各采 500 对
干净验证数据（seed 9200/9201）；连同保留的 3,000 对，总目标仍为
20,000 train + 1,000 validation。

相机原生缓冲区对应全部 1,000 worlds，GS/点云/独立网格审计按每批 8 个
视角处理；每进程 4 个写盘线程，最多 16 个待写样本，避免展开 1,000 份
高斯中间张量和无限积压 CPU 图像。每个环境独立计数 episode、warmup、
接受样本配额；保存 env_id、episode_id、sequence_id 和 policy_step。
SH 与点云增强分别使用逐样本种子，姿态、相机、点云、RGB 同一时刻。
每次 capture round 中物理状态固定，全部图像批次处理完再运行下一控制步。
policy_rollout.npz 记录所有环境的动作、结束、成功、reward 和误差数组。

正式启动前实测：8 环境 × 8 对样本全量回读通过，包含 1 次自动重置，
每环境独立配额正确；随后 1,000 环境、跨整个网格选取的 64 个环境样本
全量通过，Actor 运行 60 个控制步，即 60,000 个环境控制步，无容量溢出。
显存峰值 18,976 MiB。所有 64 对含 1,024 个有效点，点云原始深度误差 0，
重投影最大误差 3.06e-5 px，独立可视网格深度最大差 0.470 mm；严格的
轮廓/深度审计累计拒绝并补采 61 帧，未为提高通过率放宽该审计阈值。
这是千环境容量和跨网格样本验证，不表示 1,000 个环境都已逐条存盘验证。

大网格 world 坐标为 float32；相机 base-relative 位移检查仅增加按
4*eps*|base_world| 推导的坐标舍入容差，逐批记录实际最大误差。
相机本身的 K、外参和所有物理参数未改。参考了固定 v3.0.0 的迁移、
资产准备、MJWarp 调优指南；未更改引擎、依赖或原策略实现。

原训练 PID 1426919、1426920 仍运行，受保护文件哈希不变。
新调度器最终索引会引用 v1 的已校验分片，旧数据目录须与 v2 一起保留；
训练应按根 train_index.jsonl / validation_index.jsonl 读取完整集合。
实时完成状态仍以服务器根 manifest.json 为准，启动不等于采集完成。

## 2026-11-02 黄色骰子正式 RGB-D 采集（独立副本）

22021 的物理 GPU 2、4 已启动 `collect_yellow_dataset.py`，输出目录为
`outputs/yellow_rgbd_20k_train_1k_val_20261102_v1`。启动 PID 1522537。
目标训练集 20,000 对、验证集 1,000 对；两卡各一个环境，40 个训练分片
各 500 对，种子 6100–6139；验证两个分片各 500 对，种子 7100–7101。
先对前两个训练分片共 1,000 对逐条回读校验，全部通过才继续剩余分片。
实时状态以输出根目录 manifest.json 为准，不能把启动视为完成。

沿用已验证的黄色骰子 32100 Actor、原始策略契约、新相机内外参、
原架子纯视觉资产。策略 20 Hz，采样 stride=2，即每环境 10 Hz。
每次 reset 后先运行 40 个控制步（2 秒），再接受采样；保留自动重置、
原域随机化、动作滤波、观测噪声、Hold 判定，不加载优化器。
训练 RGB 使用原生 ViserDex SH 增强链；点云使用光轴深度噪声
std=0.5 mm、clip=±1.5 mm；同时保存干净 RGB 和干净可见点云。
验证集关闭上述两种输入增强。验证种子与训练种子完全分离。

新增独立 Warp 可视网格射线审计：读取同一 USD 的可见三角网格，排除
碰撞/guide 网格，用实时骰子位姿生成 object-only depth；不向物理仿真
添加任何 body，也不把该完整深度用于输入。对每帧 first-hit target mask
检查轮廓外目标像素、深度差 >0.5 mm、可见深度 <200 或 RGB <32 像素；
异常整帧丢弃、记录原因并继续补采。超过 10 倍尝试上限则停止，不标记可训练。
保存 audit_isolated_depth_m 仅供审计，不能作为预测器输入；遮挡比例
按孤立骰子轮廓中未被场景 target mask 观察到的部分计算。

检查当前固定版本 Newton 的 OpenCV camera kernel 后，点云反投影已
对齐其 (x+0.5,y+0.5) 像素中心；K 保持实际相机输出，point_uv_rgb 为
连续图像坐标，回到像素索引用 floor。只修正采集模块，未改引擎。
同时修正 keypoints 描述中的面号为资产实际 +X=6、+Z=5；角点坐标未变。

启动前实测：带噪 32 对和干净 32 对均逐条校验通过，全部 1024 有效点，
点云观测深度误差为 0、RGB 重投影最大误差 3.06e-5 px，位姿组合误差
2.23e-16。带噪实测 std=0.50024 mm、最大绝对值约 1.50004 mm（float32）；
32 对 SH 均改变，干净验证集 0 对改变。带噪试验拒绝并补采 1 个轮廓边缘
不一致帧，干净试验无拒绝；遮挡比例最高分别为 10.94% 和 12.35%。
无容量溢出，原训练 PID 1426919、1426920 仍存活，7 项原项目文件哈希不变。
证据位于独立项目 evidence/yellow_occlusion_{audit,clean}32_validation.json。

每个正式分片在统计入已校验数量前都进行全量 NPZ 回读、哈希、遮挡、
反投影、姿态标签、fixture、噪声和原文件保护检查。所有分片通过后生成
train_index.jsonl、validation_index.jsonl，才将根 manifest 的 training_ready
置为 true。采集仍是合成数据、共光心虚拟深度，不等同于真实深度相机测试。
本轮仅采集数据，不启动预测器训练，不修改共享依赖或原训练项目。

首批正式试采已完成：train_000、train_001 各 500 对全量回读通过，
1,000 对全部 1024 有效点、全部 SH 输入有变化；共拒绝并补采 36 帧
（23 帧 first-hit 深度不一致、13 帧孤立轮廓边缘不一致）。接受样本的
独立网格深度最大差分别为 0.2304 / 0.3080 mm，最大遮挡比例分别为
13.68% / 16.96%。两分片共覆盖 7 个回合，全部检查通过后已自动进入
train_002、train_003，根状态为 collecting、pilot_passed=true。
此时已校验计数 train=1000、validation=0，不能称完整 21,000 对已完成。
原训练 PID 1426919、1426920 仍在运行。试采校验报告和状态快照已同步到
本地 evidence/yellow_rgbd_20k_train_1k_val_20261102_v1/；服务器根 manifest
持续更新，本地文件只代表这次核验时刻。

## 2026-10-26 Horizon20 第 15500 轮连续视频

冻结当前训练的第 15500 轮 checkpoint，使用 `scripts/supermimic/record.py`
通过官方 SimulationRuntime / launch_simulation 和 NewtonGLVisualizer 的
`render_rgb_array` 录制单环境、seed42、1200 步，共 60 秒，20 fps 原速。
运行必须显式传入 `--viz newton_gl`；第一次误传 none 导致无渲染器，
第二次省略该选项只有空背景，均未作为最终视频。最终录制完整保留成功、
掉落、重置和失败过程，没有按表现挑选片段。

仅加载 Actor，保留 checkpoint 的 BF16 推理、10×124 带噪观测历史、
域随机化、原手和 RSLCube 资产、EMA、20 Hz 控制和奖励/结束规则。
评估不额外叠加训练探索动作噪声。单环境播放沿用已验证的 nconmax2048，
未改动训练进程的 nconmax1536。资产和环境语义校验、几何摘要均匹配。

本段累计完成 52 个目标、2 次掉落、2 次超时重置；已结束回合目标数为
5、14、22、8，最后未结束回合另有 3 个。它是一次单环境展示，
不替代 4096 环境的训练统计。奖励一致性误差 9.54e-7，FK 位置误差
1.68e-7 m，物体半尺寸误差 0，成功阈值分歧 0；没有容量或非有限数值警告。

原视频和固定空间放大版均完整解码为 1200 帧、1280×960、20 fps、
60 秒。放大版保留每帧、时序和原始计数栏。抽查五个时点的模型画面，
并用逐通道空间方差排除只有背景的输出。放大版 SHA-256：
`dc89ea6f5d8c404a2dd22ca3f6cbaacf9f285c393b71556cc824679c8dc09613`。

22031 证据目录：
`outputs/horizon20_video_20261026_1133/rollout_v3/`，
包含视频、manifest、summary、events、trajectory、env 和解码校验。
录制进程已结束；核验时主训练仍运行至 15735 轮，未停止训练。


## 2026-10-25 Horizon20 策略展开调整

按真实留出轨迹误差将策略想象长度从 16 缩至 8 步，WM 监督仍为 8 步。
未改变官方 Newton 入口、引擎、资产、随机化、奖励或控制周期。
67 项 CPU 回归测试通过；两卡各 2048 环境的三轮完整网络更新已完成，
两卡 WM/Actor/Critic 校验值一致。保留 500 轮 WM 预热、1000 轮 Actor
学习率渐增、`nconmax=1536`、`njmax=512`。旧运行正常停止并保存第 1536 轮。
诊断数据、对照限制和启动记录见 [Horizon20 记录](supermimic_horizon20.md)。


## 2026-10-25 Ramp20 碰撞容量修复验证

按已安装 v3.0.0 的 `solver_tuning_mjwarp.rst`，通过官方公开覆盖项
`env.sim.physics.solver_cfg.nconmax=1536` 修复旧容量 1152 的 broadphase
overflow，保留 `njmax=512`，未修改 Newton/MJWarp 源码或接触算法。
两卡各 2048 环境、各 768 控制步的密集接触测试通过；候选总量折算每环境
峰值 1425.7085，约束行峰值 345，无容量溢出、非有限观测。完整双卡三轮
网络更新和模型同步通过，机器人、物体指纹及环境语义合同均与旧运行一致。
有限压力测试不覆盖未来所有姿态。全部证据及等待拟合完成后再启动的用户
决定见 [Ramp20 记录](supermimic_ramp20.md)。

## 2026-10-24 Geometry-v1 独立实验

新增可微手部运动学和手—物体空间特征/监督，复用官方默认 Newton MJWarp
和同一冻结环境。读取 USD Physics 关节帧与实际碰撞体，不修改引擎、资产、
奖励、随机化、30Hz控制或学生观测。运动学独立对照官方 articulation 的
link pose/twist；物体几何保留 USD 根节点的0.9缩放，并对照 Newton 已加载
shape_scale。实现及实际验证范围见 [Geometry-v1 记录](geometry_world_model.md)。

## 2026-10-24 独立 SuperMimic 任务

两卡SuperMimic已按用户许可改在22021/GPU1、GPU5启动，每卡2048环境，总计4096，30Hz，官方torchrun/launcher设备模型。正式任务按用户最新要求从零初始化，不加载旧checkpoint；前500轮保持原WM预训练。21项CPU测试、22020及22021两卡NCCL检查通过，22021真实2048×2两轮联合更新、完整模型同步及checkpoint回读通过，第二轮网络更新0.943秒，整卡峰值24538/24537 MiB。容量测试使用的旧权重未用于正式任务；原GPU0单卡运行保持独立。部署复用现有项目的官方锁定运行时，未修改依赖、资产或物理配置。实际启动证据与长期限制见[两卡记录](revo3_supermimic_multigpu_20261024.md)。

2048 环境后续在第10292轮因奖励校验断言退出，完整 checkpoint 保留到10200。差额恰为一次成功 bonus；官方世界坐标角度与手掌坐标重建角度在0.1弧度边界的舍入差已用GPU合成案例复现。校验现通过官方 pre-reset recorder 捕获世界坐标角度，单独检查坐标变换误差及完整奖励，保持奖励断言容差和原学习设置。20项测试通过；真实运行的恢复结果见[训练记录](revo3_supermimic_training_20261024.md)。未修改官方模拟器、资产或物理配置。

奖励校验修复后，已在22021/GPU0从10200轮恢复并核查到10315轮，10300轮新checkpoint成功加载，配置/环境契约与父checkpoint一致。显存24341 MiB，网络/整轮中位数0.883/1.644秒，无容量或数值错误。原GPU7已由另一任务使用，未干预该任务。此次恢复重建仿真和replay；短测不代表最终收敛。

正式 3000 环境运行随后在第 855 轮后的真实仿真采集遇到 Warp EPA 缓冲区 OOM（请求 331,776,000 字节，整卡峰值32065 MiB）。按用户要求改为 2048 环境，从第800轮checkpoint恢复；保持30 Hz、BF16、网络批次、碰撞容量、资产和物理配置。初始恢复已进入第801轮联合更新，整卡24341 MiB，无容量警告。实际后续检查及失败证据见[训练记录](revo3_supermimic_training_20261024.md)。下面的3000环境短测记录仅是历史容量检查，不能证明长期稳定。

快速执行路径已正式接入：3000 环境、30 Hz，WM/Actor/Critic 完整批次为 2048/4096/32768，MLP BF16、其余任务运算 FP32。19 项测试通过，覆盖 BF16 梯度及 checkpoint 连续恢复；正式入口两轮联合更新的第二轮约 0.796 秒，整卡峰值 29951 MiB，无接触容量或数值错误。version-3 保存精度与分块配置，保留 version-2 FP32 兼容；FP32 策略导出也已回读验证。见 [性能记录](revo3_supermimic_performance.md) 与 [训练说明](revo3_supermimic.md)。长期收敛与接触峰值稳定性仍待正式运行验证。

新增 `Isaac-ViserDex-Repose-Revo3-SuperMimic`，复用官方 preset/`launch_simulation`、ManagerBasedRLEnv 和 RecorderTerm 生命周期。Recorder 在自动 reset 前采集下一物理状态；仿真器与原 ViserDex 训练代码未修改。模型训练使用独立入口 `scripts/supermimic/train.py`，不加载教师动作。

最终 version-2 实现通过 4096 环境、30 Hz、k8/H32、WM/Actor/Critic=10/1/20 次完整更新检查。每次更新部分约 20.80 秒，整卡峰值 26684 MiB；接触容量保持原 1152/512，未发现溢出。包含完整特权状态还原、持久传感器噪声、终止前状态、EMA、奖励、检查点恢复与策略导出检查。短程模型的长程预测仍有明显误差，不能当作已收敛策略。详细证据、Revo3 任务适配和剩余限制见 [SuperMimic 实测记录](revo3_supermimic_validation.md)。

> **2026-10-23 权重发布分支：** 当前安装和运行步骤以[主 README](../README.md)为准。
> 此分支打包了最终 compact 非 SDF 手、RSLCube 和三组权重。
> 下文保留不同日期的迁移/容量/SDF/v14 实验；早期“生成资产不提交”与默认参数描述不代表当前发布包。

最新源码来源、运行时后端和长程复测记录见 [官方流程复核](newton_official_recheck.md)。

本分支以官方 Isaac Lab `v3.0.0`（`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`）为基线：Isaac Sim **6.2.0.0**、Newton **1.6.1**、Warp **1.17.0**、PyTorch **2.11.0+cu130**、RSL-RL **5.5.1**。默认物理为 Newton + MJWarp，Revo3 默认环境数为已验证的 2048。

遵循的官方资料固定到发布标签：

- [2.x → 3.x 迁移](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)
- [Newton 资产准备](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/prepare_asset_for_newton.rst)
- [MJWarp 求解器配置](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/solver_tuning_mjwarp.rst)
- [后端与预设](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/concepts/backends_and_presets.rst)
- [官方依赖覆盖](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/pyproject.toml)

## 环境

在 Linux / Python 3.12 中，先准备官方发布的工作区，再建立本项目独立环境。以下从项目根目录执行；设置实际的 `ISAACLAB_PATH` 和 `UV_BIN`。

```bash
export ISAACLAB_PATH="$(realpath ../IsaacLab-v3.0.0)"
export UV_BIN="$(command -v uv)"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-newton"
export UV_PYTHON_PREFERENCE=only-system
export UV_PYTHON_DOWNLOADS=never
"$UV_BIN" sync --project "$ISAACLAB_PATH" --locked --python /usr/bin/python3 \
  --extra isaacsim --extra importers --extra rsl-rl --extra video --extra test
"$UV_BIN" pip install --python .venv-newton/bin/python \
  -o requirements/isaaclab-v3-overrides.txt -c requirements/newton-constraints.txt \
  -e . 'cuda-toolkit[nvcc,cudart,cccl]==13.0.2'
```

保留官方对 USD、Newton、Torch、fsspec 和 urllib3 的覆盖，不按 Isaac Sim 子包的旧元数据降级。项目额外固定 NumPy 2.3.1（满足官方 `numpy>=2`，同时兼容 Sim / Numba）和 W&B / protobuf。`uv pip check` 可能报告官方覆盖明确允许的元数据差异，不应为消除这些报告而降级 USD / Torch。

CUDA wheel 仅提供带版本的运行库；`setup_cuda_build.py` 在独立环境中建立供编译器使用的 `libcudart.so` 链接，不改包内二进制。

3DGS 首次运行会编译 gsplat CUDA 扩展，需要 C++ 编译器、Python 3.12 开发头文件和 CUDA 13.0 编译器。启动脚本可以发现环境内 NVIDIA CUDA wheel 的编译器。服务器上的头文件安装在独立环境的 `include/`，未改系统 Python。运行用 `--no-sync`，避免 uv 清除额外项目依赖。

## 资产准备

先下载物体数据，再在独立的新 Python 进程中转换资产：

```bash
export VISERDEX_OBJECT=RSLCube
bash scripts/run_newton.sh scripts/download_data.py
bash scripts/run_newton.sh scripts/preprocessing/prepare_revo3_newton.py
bash scripts/run_newton.sh scripts/preprocessing/setup_cuda_build.py
```

转换使用官方 `UrdfConverter`，明确启用 `run_asset_transformer=True` 与 `run_multi_physics_conversion=True`。输出包含中立物理、PhysX、MuJoCo 层。保留源 URDF 的质量、惯量与关节边界，合并固定装饰关节，生成固定基座，不再额外叠加第二个固定关节。机械审计检查 21 个活动关节、22 个刚体、正质量、正主惯量及三角不等式、碰撞和资产依赖。BrainCo 原始文件及校验清单保持不变。

旧 Revo3 / RSLCube USD 将 CollisionAPI 放在 Xform 层，Newton 不能将这些节点作为实际碰撞几何。手部通过官方导入重建；物体在 `assets/generated/usd/<object>/` 中生成副本，将标准 USD 碰撞 API 正确写到 Mesh，保留原始尺度、质量、变换和材质。正的显式质量优先；仅在没有正质量时使用 500 kg/m³ 密度作为后备，避免同时写入互相矛盾的质量与密度。RSLCube 的 `OmniPBR.mdl` 是 Isaac Sim 内建模块；只有这个内建引用允许不作为本地文件解析，其他缺失依赖均报错。默认 RSLCube 数据来源及 SHA-256 另存于服务器 `assets/data/assets-source.json`。

`assets/generated/` 不提交。换物体后重新执行准备脚本；新增扫描物体也走此步骤。不要直接启动旧的平铺 PhysX USD。

## 运行

使用不带 `-v0` 的官方任务命名。v3 默认无可视化窗口，可显式使用 `--viz none`；原 `--headless` 命令需要更新。Gym 不允许同时注册同名的带版本和不带版本任务，所以本分支只注册新名称。

```bash
# 保持、零动作和随机动作，检查多次重置及观测/动作契约。
bash scripts/run_newton.sh scripts/tests/smoke_revo3.py \
  --viz none --device cuda:0 --num_envs 64 --steps 360 --dataset_object

# 官方统一训练入口。
bash scripts/run_newton.sh isaaclab train \
  --task Isaac-ViserDex-Repose-Revo3 --viz none --device cuda:0 \
  --num_envs 64 --max_iterations 3 --logger tensorboard \
  physics=newton_mjwarp env.commands.object_pose.debug_vis=false

# 官方 PhysX 预设用于独立的配对验证。
bash scripts/run_newton.sh scripts/tests/smoke_revo3.py \
  --viz none --device cuda:0 --num_envs 4 --steps 120 --dataset_object physics=physx

# 无已训练 CNN 时，验证 RGB、深度、活动手遮挡、投影和选择性重置。
bash scripts/run_newton.sh scripts/tests/smoke_vision.py \
  --viz none --device cuda:0 --num_envs 2 --steps 4 --output_dir logs/newton_vision_smoke
```

`train.py` / `play.py` 作为兼容文件，直接调用官方统一入口，不再维护一份旧训练循环。官方入口使用 `--checkpoint` 恢复；旧脚本的自定义 `--profile`、信号保存及 `--resume/--load_run` 参数不再由训练兼容文件实现。自定义蒸馏循环仍保留自己的信号保存和恢复机制。`run_sim6.sh` 文件名暂留为 `run_newton.sh` 的转发入口。

教师日志采用新的 `viserdex_revo3_v3_ppo` 命名空间，避免与 beta / PhysX 实验混淆。教师用于蒸馏时会校验保存的实际物理后端；学生恢复也要求同一后端。请重新训练 Newton 教师，保留检查点相邻的 `params/`。

## 迁移边界和差异

- 21 个动作的显式关节顺序、EMA、49 / 38 维观测、奖励、控制周期保持原设定。四元数继续采用项目已迁移的 XYZW；PLY / gsplat 自身的 WXYZ 输入格式不重复转换。
- 目标姿态通过项目 `ViserDexReorientCommand` 对齐 ViserDex 开源版依赖的 Isaac Lab 2.3.2：独立采样两个 `[-pi, pi]` 角度，组合 `q_x(alpha) * q_y(beta)`。不再使用 v3 的均匀 SO(3) 目标分布；继承 v3 的成功计数、重置保护和成功后换目标流程，四元数仍为 XYZW。该设置由共享任务配置绑定，适用于 Revo3、Allegro 及其继承的环境；官方 Isaac Lab / Newton 源码没有修改。
- 资产公共属性使用 USD Physics schema fragments，PhysX / Newton 属性使用各自片段；SimulationCfg 的全局材质槽使用官方公共 RigidBodyMaterialBaseCfg。MJWarp 用 `implicitfast`、`elliptic`、2 个内部子步、`update_data_interval=1`，验证阶段开启 `debug_mode`。保留 `iterations=100` 和 `tolerance=1e-6`；1200 步复核发现默认 50 次线搜索触顶，通过公开 `ls_iterations` 调到 100 后同负载通过，详见官方流程复核记录。
- 接触容量来自实测，不把 PhysX 全局 GPU 缓冲区数字搬到 Newton。初始 128 容量出现约 400 对宽相碰撞；迁移时曾保守设置为 `1024 / 4096`。后续分项审计后，当前默认收紧为 `nconmax=640`、`njmax=1024`，2048 环境 / 600 步检查通过，显存峰值降至约 13.9 GiB；详见 [显存审计](newton_memory_audit.md)。扩大状态分布时仍需检查溢出。
- MJWarp **不强制执行** `joint_velocity_limit=10`。它不是已验证的硬件限速器，实测随机动作速度会超过此值。保留原控制参数，并记录实际速度；需要硬速度界限时应另做控制器设计与验证。
- 域随机化继续通过官方后端分派执行。Newton 的材质随机化使用静摩擦区间，不能把 PhysX 独立静 / 动摩擦分桶当作完全等价。
- 3DGS 相机采用官方 `FrameView` 获取非物理锚点位姿；手深度相机调用官方 `MultiMeshRayCasterCamera` 工厂，在生成资产中发现并跟踪真实视觉 Mesh，不依赖旧的 `link/visuals` 路径或 PhysX 私有视图。
- 当前官方 MJWarp CPU 路径在接触初始化调用 `get_max_contact_count()` 时会抛出 `NotImplementedError`；多世界还受 CPU `separate_worlds` 限制。物理验证使用 GPU，CPU 单元测试单独执行。

## 本次验证记录

代码主目录：服务器 22031 的 `/home/zhangke/viserDex_revo3`，分支 `codex/newton-migration`。31 的 GPU 均在运行其他任务，GPU 验证使用 22020 的独立源码快照及独立环境，没有修改其他任务的环境。

已确认：64 环境、360 步混合动作、2 次全量和一次选择性重置，通过关节动作与观测断言，退出 0，日志无容量溢出。进程显存峰值 **1328 MiB**（nvidia-smi 采样，包含 Warp / MuJoCo；不能只看 Torch 分配器）。随机动作最大关节速度约 20.79 rad/s。CPU 测试初次为 85 通过、2 跳过。

已确认：初始容量配置下，2048 环境、120 步混合动作与多次重置，退出 0，未发现容量溢出，显存峰值 **22954 MiB（约 22.4 GiB）**。后续审计收紧容量，2048 环境 / 600 步检查降至 **14218 MiB（约 13.9 GiB）**。官方统一 PPO 完成 3 次更新，保存 model_0 / model_1 / model_2；学生 DAgger 完成 2 次更新并保存检查点。

已确认：PhysX 配对的 4 环境 / 120 步检查通过；Newton 3DGS RGB、手深度、投影与选择性传感器重置通过；学生可从第 2 次更新恢复到第 3 次更新，TorchScript 导出与回读校验通过。

视觉检查未使用已训练 CNN 权重，验证的是渲染、遮挡与标注链路。其他物体、Allegro 及完整 RGB 策略效果需各自验证。最终回归结果另存于服务器验证报告；短程验证只说明执行链路可运行，不代表策略收敛或真机性能。

Kit 视图默认采用官方 `KitVisualizerCfg(origin_type="env", origin_env_index=0)`。
官方启动扫描会跳过未启用的 `default_visualizer_cfg`，因此这不会让默认无窗口 Newton 启动 Kit。
批量环境的 env_0 原点可能不是世界原点；自定义取帧相机使用 `scene.env_origins[env_id]` 计算世界坐标。
取帧通过可视化器的公开 `render_rgb_array()`，先创建并预热渲染产品，再采集画面。

最终回归：CPU 为 85 项通过、2 项跳过；CUDA 的 pose metrics 测试 4 项全部通过。
无窗口 Newton 和 Kit 截图检查都在最终配置上退出 0，wheel 构建通过，官方 Isaac Lab 工作区没有源码修改。
本次使用的 22020 GPU 2 / 4 已释放。验证摘要和逐项日志保存在
`/home/zhangke/viserdex-newton-migration-20261020/validation/`（22031）；
原始 GPU 检查记录及短程检查点位于 22020 的
`/home/zhangke/viserdex-newton-validation-20261020/`。

## ViserDex 开源目标采样对齐（2026-10-20）

共享任务现使用项目内的 `mdp/commands.py:ViserDexReorientCommandCfg`。
两个角度独立均匀采样于 `[-pi, pi]`，按原版顺序组合 `q_x(alpha) * q_y(beta)`；
对应 [Isaac Lab 2.3.2 原实现](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/mdp/commands/orientation_command.py#L96)。
仅替换目标分布，保留 v3 命令基类的成功记账、选择性重置和成功后换目标机制。
默认 Newton + MJWarp、Revo3 关节与重置配置、观测和 PPO 参数保持现状；此项对齐不代表整套训练条件与 Allegro 原版相同。

实际验证：

- 22031：`tests/test_goal_sampling.py` 为 10 项通过。用独立 SciPy 旋转验证固定随机种子的乘法顺序和 XYZW 输出，检查随机数消耗、空索引、切片、非连续索引、未选中环境及成功记账；核对 Allegro / Revo3 默认配置绑定。Ruff 与 `git diff --check` 通过。
- 22020 GPU 4：64 环境 / 180 步混合动作，含全量和选择性重置；运行时确认命令类型、单位四元数和旧分布约束 `R[0,1]=0`，检查通过。
- 同一空闲卡：官方 `isaaclab train` 入口完成 64 环境 / 3 次 PPO 更新，保存 `model_0.pt`、`model_1.pt`、`model_2.pt`；保存的配置使用 `ViserDexReorientCommand`，未发现接触 / 约束容量溢出。运行名为 `2026-10-20_15-05-04_goal_xy_alignment_smoke`。

31 上变更前备份和清单位于 `/home/zhangke/viserdex-newton-migration-20261020/goal-sampling-alignment/`。
20 上 GPU 原始日志、检查摘要及验证脚本位于 `/home/zhangke/viserdex-newton-validation-20261020/goal-sampling-alignment/`。
这些是采样兼容性与短程执行验证，没有训练收敛的教师。新启动的训练使用新分布；已有进程不会自动热更新。

## 8192 环境容量验证（2026-10-20）

当前默认仍为 2048 环境并启用 CUDA Graph。8192 环境在单张 RTX 5090 32 GB 上需要显式关闭 Graph。

| 8192 环境配置 | 进程显存峰值 | 实测结果 |
|---|---:|---|
| 默认 CUDA Graph | 31946 MiB（31.2 GiB） | Graph 创建时报 CUDA out of memory，未通过 |
| 仅关闭 Graph | 29384 MiB（28.7 GiB） | 120 步混合动作、全量及选择性重置通过 |
| 关闭 Graph，实际 PPO | 29932 MiB（29.2 GiB） | 官方训练入口完成 3 次更新，保存 model_0 / model_1 / model_2 |

关闭方式为公开配置覆盖 `env.sim.physics.use_cuda_graph=false`。
没有降低接触容量、约束容量、碰撞几何或两个物理子步；采样仍为已对齐的 X/Y 组合。
关闭 Graph 会失去对应的加速，不能将其速度当作默认 Graph 模式的速度。

测试在 22021 的空闲 GPU 3 执行。源码和生成物理资产从 22020 验证副本复制；
使用相同官方 v3.0.0、Sim 6.2、Newton 1.6.1、Warp 1.17、MuJoCo Warp 3.12。
测试所需的 NumPy / Kornia 放在项目独立 `.test-deps` 中，没有更改 21 的共享环境或官方源码。
状态 PPO 未启用相机和 3DGS 渲染。

120 步检查读取到的求解器标记始终为 0；接触总数峰值 3013257、宽相候选峰值 3565426，均小于共享容量 5242880；
单世界约束行峰值 486，小于 1024。PPO 日志没有容量或收敛警告。
原始日志、命令和检查摘要位于 22021 的
`/home/zhangke/viserdex-newton-validation-20261020/env8192-audit/`。
运行名为 `2026-10-20_15-33-02_env8192_no_graph_smoke`，测试进程已退出。

在空闲 GPU 上启动同配置训练：

```bash
bash scripts/run_newton.sh isaaclab train \
  --task Isaac-ViserDex-Repose-Revo3 --viz none --device cuda:0 \
  --num_envs 8192 --logger tensorboard \
  env.sim.physics.use_cuda_graph=false env.commands.object_pose.debug_vis=false
```

以上是容量和短程执行验证，不能代表长程训练收敛，也不代表 8192 个视觉环境的显存需求。

## 4096 环境容量验证（2026-10-20）

在 22021 的空闲 RTX 5090 32 GB（GPU 1）实测，4096 环境可以保留默认 CUDA Graph。
使用与上面 8192 检查相同的官方版本、源码和生成物理资产；没有降低碰撞几何、接触容量、约束容量或物理子步。
目标姿态仍为已对齐的 X/Y 两角组合。

| 4096 环境配置 | 进程显存峰值 | 实测结果 |
|---|---:|---|
| CUDA Graph 开启，物理检查 | 27634 MiB（27.0 GiB） | 120 步混合动作、全量及选择性重置通过 |
| CUDA Graph 开启，实际 PPO | 27914 MiB（27.3 GiB） | 官方入口完成 3 次更新，保存 model_0 / model_1 / model_2 |

显存为 nvidia-smi 对测试进程及其子进程每 0.5 秒采样的峰值，包含 Warp / MuJoCo 分配。
120 步物理循环耗时 22.87 秒；3 次 PPO 更新的训练计时为 16.88 秒，启动和初始化不包含在这两个计时中。
求解器溢出标记始终为 0；接触总数峰值 1506601、宽相候选峰值 1782035，
均小于共享容量 2621440；单世界约束行峰值 513，小于 1024。
两项检查的容量及收敛警告均为 0。保存的训练配置确认 `num_envs=4096`、`use_cuda_graph=true`、
`num_substeps=2`、`nconmax=640`、`njmax=1024` 和 `ViserDexReorientCommand`。

启动时使用 `--num_envs 4096` 即可；生产默认仍为 2048，未启动长程训练。
运行名为 `2026-10-20_15-41-42_env4096_graph_smoke`。
原始命令、日志和检查摘要在 22021 的
`/home/zhangke/viserdex-newton-validation-20261020/env8192-audit/graph4096*` 与 `ppo4096_graph*`。
测试进程已全部退出。以上是无相机教师的容量与短程执行验证，不代表长程训练收敛或视觉环境需求。

## 掌部凸包几何检查发现的问题（2026-10-20）

本次碰撞可视化发现：当前掌部的凹口被凸包跨接，外观中可见的空隙并没有完整保留在碰撞几何中。
掌部的 `right_hand_base_link`、`right_plam_touch_link`、`right_plam_rubber_link`、
`right_plam_backtrim_v2`、`right_plam_led_link_1` 在生成 USD 中分别标为 `convexHull`。
22021 单环境的实时检查确认这五个零件在 Newton 中均为 `GeoType.CONVEX_MESH`，
每个为一个 64 顶点凸包；MJWarp / MuJoCo 中对应五个 mesh geom，凸包图均已生成。
单独隐藏手指和物体后，同一姿态的外观 / 实际碰撞对照确认掌部凹口被填平。

因此，之前的 82 个手部碰撞体数量、物理运行、显存和 PPO 短程通过记录并不代表碰撞几何保真。
预处理脚本虽然请求 `Convex Decomposition`，但不能据此假定每个掌部凹形零件已正确分解。
这种近似可能造成凹口附近提前接触，并影响掌部和拇指附近的物体运动。
需要为相关凹形零件生成保留凹口的多凸体分解，并重新检查运行时形状及训练容量；本次仅诊断，没有修改资产或训练默认值。

原始截图及运行时形状诊断位于 22021 的
`/home/zhangke/viserdex-newton-validation-20261020/palm-audit/`，包括 `palm_live_solver.json`
和 `palm_recess_comparison.png`。显示检查仅修改查看器的 `shape_opacity`；物理几何保持原样，测试进程已退出。

## 固定掌部 SDF 碰撞资产（2026-10-21）

依据用户提供的《Revo3手掌碰撞体优化报告》重建固定掌部。报告未附原始网格文件；
本次从现有官方导入资产提取同样的六个 visual 网格（32790 面），并非报告网格的逐字节副本。
在 0.3 mm 体素网格上进行两次闭运算修复拼缝、填充封闭内腔、提取外表面，
Taubin 平滑 20 次后简化为 **602 顶点 / 1200 三角面**。结果为单个封闭、定向一致、Euler 数 2 的外表面。

使用 Isaac Lab 官方 `UsdPhysicsMeshCollisionCfg` 和 `NewtonSDFCollisionCfg` 写入 SDF 意图：
最大分辨率 **512**、内外窄带 -5 / +5 mm、uint16、非 hydroelastic。
Newton 在模型 finalize 时生成并打包实际 SDF；诊断必须读模型的 packed SDF 表，
不能用源 `Mesh.sdf is None` 推断未烘焙，因为官方实现将延迟生成的 SDF 放在临时克隆中再打包。
4096 个环境共享一个已烘焙 SDF，实测体素尺寸约 0.260 mm、50959 个子网格。

生成文件在主项目 `source/viserdex/viserdex/assets/generated/revo3_palm_sdf/`：

- `revo3_right_palm_sdf.usda`：引用原有多物理手模型的完整手部覆盖层。
- `palm_collision_sdf.usda`：无外部依赖的独立碰撞 Mesh，供附加到现有刚体使用。
- `palm_surface_1200.ply`：SDF 输入三角网格。
- `palm_sdf_manifest.json`：源网格、机械属性、输出网格及哈希记录。

只将固定掌部的五个凸包替换为一个 SDF，77 个手指碰撞体逐项保持原网格与近似设置。
质量、惯量和关节属性签名一致。新手共 78 个碰撞体，加任务物体后每世界 79 个 geom。
网格在 22021 从源 visual 重建并验证后复制到 22031；31 用同一源资产重新装配 USD 并复核签名。
31 的重复完整重建因耗时停止，未将该重复尝试计为成功验证。

### 官方运行路径

新增任务 **`Isaac-ViserDex-Repose-Revo3-SDF`**。原任务保留用于几何 / 性能对照。
SDF 任务保留 MJWarp 求解器、implicitfast、elliptic、两个子步、CUDA Graph、21 动作、49/38 观测和 X/Y 目标采样。
根据官方 v3.0.0 求解器指南，非凸 SDF 使用 `use_mujoco_contacts=False`
和 `NewtonCollisionPipelineCfg`；接触生成路径的改变是有意的，不与 PhysX 报告的计时直接等同。

旧 `nconmax=640` 在 64 环境出现 Newton 接触截断，SDF 任务增加到 **1024**；`njmax=1024` 保留。
4096 环境产生约 1670 万候选三角形对，默认 100 万缓冲区会溢出；
据此设置全局 **`max_triangle_pairs=33554432`**。没有降低碰撞几何、域随机化、子步或求解器精度。
含溢出的早期运行均不计为通过；验证脚本同时检查 Newton 接触容量和 MJWarp 容量，并扫描警告。

```bash
# 原始多物理资产已存在时，只生成掌部 SDF。
bash scripts/run_newton.sh scripts/preprocessing/prepare_revo3_palm_sdf.py
# 从官方 URDF 导入流程一起生成时，加 --palm_sdf。
# 几何构建依赖固定在 requirements/palm-sdf-build.txt。
bash scripts/run_newton.sh scripts/preprocessing/prepare_revo3_newton.py --palm_sdf

bash scripts/run_newton.sh isaaclab train \
  --task Isaac-ViserDex-Repose-Revo3-SDF --viz none --device cuda:0 \
  --num_envs 4096 --logger tensorboard env.commands.object_pose.debug_vis=false
```

### 实测验证

- 22031：`tests/test_palm_sdf.py` 与 `tests/test_goal_sampling.py` 共 **14 项通过**；Ruff 和 `git diff --check` 通过。
- 实际 GPU SDF 查询：同一个掌部局部点 `(0.008, 0.0425, 0.0525)` 米，在原运行时 64 顶点凸包内约 **2.88 mm**，
  新 SDF 则有 **14.60 mm** 正距离，确认凹口不再被该凸包封住。高分辨率原网格数学凸包对应深度 3.16 mm，二者记录分开保存。
- 64 环境 / 120 步混合动作，含全量和选择性重置，通过扩容后的容量与有限值检查。
- 4096 环境 / 120 步，CUDA Graph 开启，完整三角形缓冲区下通过。进程显存峰值 **7280 MiB（7.1 GiB）**。
  Newton 接触峰值 2960547；MJWarp 共享接触容量 4194304，单世界约束行峰值 339；溢出标记及容量 / 收敛警告均为 0。
- 官方 `isaaclab train` 完成 4096 环境 / 3 次 PPO 更新，保存 model_0 / model_1 / model_2。
  进程显存峰值 **7562 MiB（7.4 GiB）**；每轮 10.39 / 8.26 / 8.27 秒。
  运行名 `2026-10-20_16-24-36_palm_sdf4096_smoke`。
- 已保存原始外观、旧凸包和新 SDF 输入表面的同视角对照；原始 SDF 模式和精简显示均通过官方 Newton GL 可视化器取帧。

主要 GPU 原始记录位于 22021 `/home/zhangke/viserdex-newton-validation-20261020/palm-sdf-audit/`。
以上为状态教师的几何、容量和短程训练验证；未启动长程训练，未验证视觉学生或训练收敛。

### SDF 采样耗时调优（2026-10-21）

在同一台 22021 的空闲 RTX 5090 GPU 1 上，以 4096 环境、seed=42、
相同 24 步 rollout 和网络分别运行 8 次官方 PPO 更新。以下均剔除第一轮预热，取后 7 轮平均：

| 配置 | 环境采样 / 轮 | PPO 更新 / 轮 | 总耗时 / 轮 | 进程显存峰值 |
| --- | ---: | ---: | ---: | ---: |
| 原 SDF：每形状 gap=10 mm | 7.838 s | 0.393 s | 8.231 s | 7562 MiB |
| 仅关闭 debug_mode | 7.796 s | 0.393 s | 8.191 s | 7562 MiB |
| 仅 njmax=512 | 7.851 s | 0.375 s | 8.229 s | 7148 MiB |
| 仅每形状 gap=5 mm | **4.088 s** | 0.367 s | **4.456 s** | **7562 MiB** |

采用最后一项：在 SDF 专用 `NewtonCfg` 上设置官方公开配置
`default_shape_cfg=NewtonShapeCfg(gap=0.005)`。
环境采样耗时减少 **47.84%**，总迭代耗时减少 **45.87%**。
关闭诊断和缩小约束容量均无明显提速，因此保留 debug_mode=True、njmax=1024、nconmax=1024。
仍保留两个求解子步、每物理 tick 一次碰撞刷新（collision_decimation=0）、
原求解精度、接触材质、域随机化和 512 分辨率 / 1200 面 SDF 资产。

`gap` 是形状表面以外的额外接触检测距离，成对相加；这里从总计 20 mm 调整为总计 10 mm。
`margin` 控制接触表面位置，保持 0，未通过膨胀或简化几何提速。
原配置在混合动作诊断中每步有约 280 万个非负距离的候选接触，缩小搜索范围减少了这部分工作。
CUDA 内核采样显示主要可见开销在 mesh_triangle_contacts_to_reducer、
网格三角形重叠查询和稠密约束矩阵计算，而预计算 shape pair 的 broad phase 较小。
该 kernel 样本不用于声称完整耗时百分比，最终提速以上述未启用 profiler 的 PPO 实测为准。

官方依据：
[Isaac Lab v3.0.0 MJWarp 调优指南](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/solver_tuning_mjwarp.rst)、
[Newton margin / gap 语义](https://newton-physics.github.io/newton/1.6.0/concepts/collisions.html#shape-configuration)。
同时核对了安装的 Newton 1.6.1 和 Isaac Lab 配置源码，未修改引擎文件。

8 轮 PPO 和 4096 环境 / 120 步混合动作、全量及选择性重置检查均通过容量、收敛和有限值检查；
实际 packed SDF 凹口探针仍为 14.60 mm 正距离。另有固定环境 seed=42、动作 seed=2244 的对照记录。
较小 gap 会缩短提前检测范围，不能将短测通过解释为所有高速碰撞或长期训练效果完全等价；
尚未验证最终收敛率和任务成功率。

本轮短测一度将 SDF 任务设为 5 mm；后续精度审查已恢复默认 10 mm（见下节）。复现 10 mm 基线可追加：

```bash
env.sim.physics.default_shape_cfg.gap=0.01
```

原始 benchmark、CUDA 内核摘要和固定动作记录位于
22021 `/home/zhangke/viserdex-newton-validation-20261020/palm-sdf-audit/`，
文件名前缀为 `sdf_tune_` 和 `sdf_gap_fixed`。

### gap 精度与漏接触专项审查（2026-10-21）

结论：5 mm 不是无损的性能开关。保留该值作为显式 CLI 加速选项，
**主项目默认已恢复每形状 gap=0.01 m**。SDF 网格、材质、两个子步、CUDA Graph、
njmax/nconmax、求解精度及原任务随机化不变。10 mm 也不构成连续碰撞检测保证。

完整任务分别用 seed=42、2026，在 256 环境执行 1600 个控制步，
覆盖保持、2 Hz 正弦动作、极限开合、随机跳变、全量和选择性重置。
两种 gap 加上 seed=42 的 5 mm/每子步刷新对照，共 **2048000 个环境步**。
五组均无容量溢出或求解触顶、无非有限状态；物体速度峰值约 5.9 m/s。
平均奖励接近不等于接触时机、轨迹或真实物理精度相同，且这些并非训练收敛试验。

受控参考采用官方 Newton ModelBuilder、CollisionPipeline 和 SolverMuJoCo 接口：
固定的同一份 1200 面掌部网格与 512/uint16 SDF，58.5 mm / 70 g 立方体。
质量与尺寸来自任务资产；参考隔离掌部，不含手部关节、手指自碰撞或随机化材质。
在正面薄壁、凹口壁、侧缘和上缘四处，分别设置 6 个起始间隙、8 档速度，另补 6 m/s。
每个起点按完整方块 y/z 占据范围构造保守分离平面，确认未与掌部初始相交。
引擎 time 数组逐步校验实际模拟时长为 0.15 s，原子求解步均有位置及接触记录。

原设置为碰撞刷新 120 Hz、求解 240 Hz；对照组仅把碰撞刷新改为 240 Hz。
另有 960 / 1920 Hz 的较细步长参考通过求解检查。参考迭代上限为 400，实际最多 17 次，
上限未触发；不能将这些参考称为已达到时间步收敛的真实物理真值。

| 设置 | 0.1～8 m/s，192 场景穿壁 | 16 m/s，24 场景穿壁 | 16 m/s 中穿壁且全程未出现有效掌部接触 |
| --- | ---: | ---: | ---: |
| 10 mm，碰撞 120 Hz | 0 | 17 | 4 |
| 5 mm，碰撞 120 Hz | 0 | 20 | 12 |
| 5 mm，碰撞 240 Hz | 0 | 4 | 0 |
| 1920 Hz 较细参考 | 0 | 0 | 0 |

这里“穿壁”按整个方块超过首层壁面内侧 0.5 mm、中心仍在原射线 y/z 的 3 mm 邻域定义。
它和“完全没有生成有效接触”分开统计：有接触也可能因为时机晚或离散响应不足而穿壁。
5 mm 比 10 mm 额外出现的三个穿壁场景都在 **16 m/s**，位置为凹口壁（11/15 mm 起始间隙）
与侧缘（11 mm 间隙）。这些场景中 5 mm 未出现有效掌部接触，而 10 mm 与较细参考有接触。
16 m/s 明显超出当前完整任务压力测试的观察峰值，不能把此发生率直接用于正常训练。

6 m/s 的 24 场景均未穿壁，但 5/10 mm 的 0.15 s 位置轨迹 RMS 差异：
中位 **8.30 mm**、95 分位 **26.71 mm**、最大 **28.19 mm**。
同一个 10 mm 配置重复运行的对应差异为中位 **0.59 mm**、95 分位 **2.98 mm**、最大 **11.20 mm**；
因此存在数值重复性误差，但不能据此把两种 gap 的差异全部忽略。
上述数值是配置之间的运动差异，不是真实世界的绝对位置误差。

剔除的诊断尝试也保留日志：未修正 Newton policy decimation 的早期直接 sim.step 试验、
仅按中心射线放置而可能初始相交的方块、所有迭代触顶/溢出的更细步长试验，均不计入表格。
4 mm / 70 g 的额外极端小物体试验中两种 gap 均大量穿越薄壁，不能外推为当前训练物体的发生率。
没有将全局 contact.dist 最小值或单纯无 NaN 当作几何精度证明。

当前默认回到 10 mm，故先前 **4.46 s/轮仅代表可选 5 mm 配置的速度**，
不能继续当作当前默认速度。10 mm 已测基线约 8.23 s/轮，显存仍约 7.38 GiB。
5 mm 加每子步碰撞刷新改善了撞击覆盖，但未将其自动升级为训练默认，也未验证长程策略效果。
主项目恢复后，原有 14 项测试、Ruff、git diff --check 通过。

原始记录在 22021 `palm-sdf-audit/accuracy_clean_*`、`accuracy_speed6_*`、
`accuracy_iso_clean_cube_fine*`、`accuracy_stress_*`；
本地导出的 `palm_gap_accuracy/summary.json` 与脚本、NPZ 轨迹记录一起保存。


## 2026-10-21 恢复原 SDF 训练

按用户最新指令停止 V14 凸包任务，保留代码与资产；恢复原 `Isaac-ViserDex-Repose-Revo3-SDF`，8192 环境、10 mm gap、CUDA 图开启。使用原 SDF 的 `model_100.pt` 恢复策略及优化器，剩余 4900 轮，未加载 V14 checkpoint。实际运行在 22021 物理 GPU 3，记录位于 `training-runs/revo3_sdf_8192_gap10_resume_20261021/`。原环境完整配置对照一致，差异仅是日志目录、恢复 checkpoint 路径和剩余训练轮数。主项目镜像记录见 `outputs/sdf_resume_20261021/`。


## 原版 URDF 拇指侧掌部凸碰撞体裁剪（2026-10-21）

以原项目 revo3_right.urdf（SHA-256 c4eae42fddedfd02b6a2d15699fc21a3173f5ae3d5a164b585178a307f6173ca）为起点，在独立资产中裁去拇指根部相邻的掌壳、橡胶层、触觉层凸包。基于原实际 Newton 烹制凸包，每层分为三个封闭凸块，形成 +Y 侧的局部凹口，原 82 块变为 88 块。非 SDF；原输入及当前 SDF 训练没有覆盖或切换。

生成仍通过 scripts/preprocessing/prepare_revo3_newton.py 的官方 UrdfConverter / 多物理转换。新增可选 --exact_hull_prefix 让裁剪凸块通过 NewtonMeshCollisionCfg 保留完整顶点，其余保持 64 顶点，V14 原有默认行为保留。关节驱动可显式传入原版 7 / 0.22，未修改引擎。

运行时逐点比较证实其余 79 个碰撞体完全一致；22 个刚体的质量、质心、惯量以及 21 个关节的轴、边界、锚点和导入驱动属性完全一致。裁剪凸块没有超出原凸包或切面，凹口没有被导入器填回。Newton MJWarp 单环境 240 步混合动作、两次重置通过，观测、奖励和关节状态有限，无接触容量溢出。NewtonGL 同相机截图已检查；GL 互操作不可用时用官方复制回退。该测试不代表大规模新资产训练性能。

资产与证据见 outputs/palm_thumb_crop_20261021/README.md。Newton USD 位于 source/viserdex/viserdex/assets/generated/revo3_palm_thumb_crop_v1/revo3_right_thumbtrim_v1/revo3_right_thumbtrim_v1.usda；URDF 和完整依赖网格在 outputs/palm_thumb_crop_20261021/input/。


## 裁剪掌部 v1 的拇指自碰撞复核（2026-10-21）

25 秒 / 3000 步 Newton PD 驱动检查确认：CMP 内收到约 44° 时，cmr_v4_22 与掌部中段凸块接触；目标 CMP 110°，实际最大约 44.8°。单独 CMR 扫描路径未触掌，完全屈指时末节会触掌，实际最大接触穿透约 0.784 mm。本版尚未通过全内收行程检查。详细报告、视频和接触记录见 outputs/palm_thumb_crop_20261021/thumb_self_collision/README.md。原先 240 步短程 smoke 通过仅代表执行与状态检查，没有覆盖这些特定运动路径。当前 SDF 训练未修改。


## 根部全行程避让资产 v2（2026-10-21）

原 v1 在 CMP 约44°被掌部提前挡住。v2 按 CMR/MCP 根部完整角度运动包络重新裁剪三个掌部碰撞层，保留 1 mm 设计间隙并计入采样位移界；27个封闭凸块、全手106个碰撞体，保持 Newton 凸碰撞，未使用SDF。通过官方预处理入口生成，其他79个运行时碰撞体逐点一致，质量/惯量/关节及视觉内容一致，过滤器未改。

实际25秒3000步测试中，根部往返前15秒零自接触，全动作CMR根部零接触；CMP实际到约109.9°。编译模型36,863个根部姿态细扫零拇指—手掌接触，距离抽查最小约1.707mm。仍保留屈指指尖触掌和对掌时与其他手指的接触，未宣称全五关节空间零接触。详情、完整限制和证据见 outputs/palm_thumb_crop_v2_20261021/README.md。新资产已存到 assets/generated/revo3_palm_thumb_crop_v2/；没有切换当前8192环境SDF训练。


## 2026-10-21 visual-sized palm and thumb-root convex trim

The user clarified that palm collision size must follow the visual mesh and remaining root interference should be removed from the thumb root. This supersedes the earlier palm swept-cavity candidate. The running SDF training asset remains unchanged.

Candidate: `source/viserdex/viserdex/assets/generated/revo3_visual_palm_root_trim/revo3_right_visual_root_v3/revo3_right_visual_root_v3.usda`.

Palm hard/rubber/touch layers were decomposed from the actual visual meshes, 64 convex parts each, with no subsequent swept palm cuts. Seventeen of the original 32 CMR convex parts were clipped using measured root-palm contact halfspaces; all 32 remain. Maximum vertex shave is 1.932 mm and total CMR convex volume retention is 87.00%. The other 62 runtime collision meshes are vertex-identical. Full hand count is 271 convex shapes; no SDF. Actual cooked palm bounding-size discrepancy is at most 0.01459 mm; extra YZ footprint is at most 0.515%. CoACD reached its part-count cap, so the requested nominal concavity tolerance is not claimed; internal cavities have conservative fill.

The public project preprocessing entrypoint and official Isaac Lab v3.0.0, Isaac Sim 6.2.0.0, Newton 1.6.1, Warp 1.17.0 launch/config pipeline were used, with `physics=newton_mjwarp`. No engine, joint, mass/inertia, visual or self-collision filter changes were made. Original and candidate non-collision URDF trees are identical after filename-content normalization. USD audit confirms all 22 mass APIs and 21 revolute joints unchanged.

Validation: 191x193 = 36,863 static CMP/CMR poses over complete limits, MCP/PIP/DIP=0, produced zero negative-distance active CMR/MCP-root versus palm contact events. The clipping target margin was 0.5 mm; this is not a continuous-space clearance proof. A 25 s / 3000-step PD motor run at 120 Hz with two substeps gave zero CMR root contacts. Its initial 15 s CMP/CMR sweep had zero active thumb self-contacts, reached CMP 109.870 degrees and CMR -24.193 to 81.782 degrees. Subsequent full curl/opposition retained distal DIP-palm and distal-thumb/ring-finger contacts (deepest 0.941 / 0.580 mm). Arbitrary full-hand poses are not claimed collision-free.

The first motion run with `nconmax=2048` overflowed its collision candidate capacity and is rejected, archived in `round1/motion_capacity2048_rejected`. The complete accepted rerun used `nconmax=3072`, `njmax=4096`; full logs contain no overflow, NaN or traceback, and recorded solver contact count peaked at 1599. These are single-environment validation capacities, not a batch-training guarantee. No 4096/8192 performance or convergence claim is made for this 271-shape candidate.

Evidence and self-contained URDF/USD bundle: `outputs/revo3_visual_root_20261021/`. See its `README.md`, `accepted.json`, `runtime_geometry_audit.json`, `mechanics_comparison.json`, `installed_dependency_audit.json`, `round1/fine/scan.json`, `round1/motion/{phase_analysis,summary}.json`, and Newton GL images/video in `round1/{capture,motion}`. The independently installed USD and packaged USD were checked for unresolved dependencies; none remain. No default task config or running training was switched.


## 2026-10-21 compact palm: 271 to 116 convex shapes

At the user's request, palm hulls were merged to reduce collision shape count while retaining visual-sized palm and thumb-root clearance. Final full hand count is 116 (57.2% fewer than 271): palm hard/rubber/touch 10/13/14, base+LED 2, original 77 finger/thumb shapes unchanged. Runtime vertex comparison confirms all 79 non-merged shapes are identical to the previous accepted asset, including every thumb-root trim. No further thumb cuts or deletions are accepted. The 22 mass APIs and 21 revolute joint attributes also remain identical to the original USD. Original URDF/USD and SDF training configuration were not overwritten.

Only protecting the YZ outline was insufficient: a naive merge filled internal cavities and produced root contacts. That candidate, including its attempted excessive root deletion, was rejected and archived at `outputs/revo3_compact_20261021/rejected_naive_merge`. The accepted merge additionally excludes sampled CMR/MCP swept volume (43x45 poses, 0.5 mm voxel sampling, about 1.47M points), then validates independently with compiled collision geometry. Maximum actual cooked palm bounding-size discrepancy is 0.01459 mm, YZ extra footprint <=0.663%, missing footprint <=0.0067%. Some internal surfaces are conservative convex approximations; pointwise geometry equivalence is not claimed.

Official public project preprocessing, Isaac Lab v3.0.0 / Isaac Sim 6.2.0.0 / Newton 1.6.1 / Warp 1.17.0, `launch_simulation` and `physics=newton_mjwarp` were used. No engine or self-collision filter edits were made. The 191x193 static root grid (36,863 poses, full CMP/CMR limits, MCP/PIP/DIP=0) has zero negative-distance active root-palm contact events. The 25 s / 3000-step PD motor test (120 Hz, two substeps) has zero CMR/MCP root contacts; its first 15 s root sweep has zero active thumb self-contacts, reaching CMP 109.870 degrees and CMR -24.193 to 81.782 degrees. Distal tip-palm and tip-ring contacts remain in full curl/opposition; deepest dynamic penetrations are 0.903 and 0.580 mm. Arbitrary full-hand poses are not claimed collision-free.

The motor test's peak solver contact count falls from 1599 to 753 on the same trajectory. This is not a speed or memory benchmark. Test capacities `nconmax=3072`, `njmax=4096` had no overflow/NaN/traceback in complete motion/capture logs. New 4096/8192 environment performance and convergence have not been tested. Training/default task configs were not switched.

Installed independent asset: `source/viserdex/viserdex/assets/generated/revo3_compact_visual_palm/revo3_right_compact_v4/revo3_right_compact_v4.usda`. Evidence and self-contained URDF/USD bundle: `outputs/revo3_compact_20261021/`; see README, runtime_geometry_audit.json, mechanics_comparison.json, completion_audit.json, round0/fine/scan.json, and round0/motion/{phase_analysis,summary}.json. Both installed and packaged USD dependencies resolve with no missing assets. Non-collision URDF equality and package file hashes are recorded.


## 2026-10-21 interactive joint/contact inspector

The user requested a browser interface to inspect the 116-shape compact Revo3 asset manually. `outputs/revo3_joint_viewer_20261021/inspect_revo3_contacts.py` uses the public launcher, official ViserVisualizerCfg, and `physics=newton_mjwarp`; it exposes all 21 joint sliders, model/collision views, contact point and wireframe overlays, contact pair/depth details, root-palm counts, camera presets, and pose export. No training/default asset configuration was switched.

Default pose inspection directly writes requested joint states and queries the actual compiled Newton MuJoCo geometry using CPU `mj_forward`; it does not advance dynamics and can show impossible penetrations. Motor mode starts from the open pose, uses public articulation PD targets and Newton MJWarp stepping at 120 Hz/two substeps, reports actual angles, and reads GPU solver contacts. Only active hand-hand constraint contacts deeper than 0.01 mm are counted, deduplicated by geometry pair. Original collision filtering remains in place. The task object is moved away for hand-only inspection.

Interactive integration validation passed neutral pose, static CMP limit, static full curl with tip/palm contact, motor mode actual CMP limit, motor full curl with tip/palm contact, and neutral restoration. Root-palm count was zero in each recorded case. Browser button/highlight behavior and an actual CMP slider drag to 106.5 degrees were verified. Reconnect camera initialization was added so new clients see the hand rather than an all-scene fit. Results: interactive_validation.json, delivery_validation.json and README in the output directory. This is not a guarantee that arbitrary 21-dimensional poses are collision-free.

The legacy project Viser 0.2.11 is incompatible with the official visualizer's newer API. The inspector uses isolated viewer-deps from Isaac Lab's uv.lock: Viser 1.0.30 and zstandard 0.25.0. Existing project dependencies, engine versions and training environment were not changed. The old incompatible startup is retained separately; accepted logs contain no capacity overflow/NaN/traceback. Both browser UI (8816) and diagnostic API (8817) bind only 127.0.0.1 and are reached through a private local SSH forward, with public sharing disabled.


### Interactive viewer latency correction

The official manager live-plot stream was enabled by default, including collapsed plots. A WebSocket measurement with an idle hand recorded 9,082,592 bytes in 6.10 s, 31 frames and 651 GuiUpdateMessage records; decoded payloads show repeated uPlot history arrays. The inspector now sets the public ViserVisualizerCfg enable_live_plots=False. Static pose contact publication and scene rendering occur only when controls or client connection change, duplicate text updates are skipped, and the input event wakes the loop immediately. Newton physics mode retains its original step/render pacing. No engine code, collision geometry/filter or training config was changed.

After correction the same idle measurement recorded zero bytes/frames for 6.01 s after the initial scene transfer (436,190 bytes during warm-up). Six small angle changes measured 110.36-142.22 ms of server loop work, including 101.79-122.02 ms in official scene rendering and 5.00-7.06 ms in static contact query. The local SSH API RTT was approximately 60-114 ms; these are separate component measurements, not a browser input-to-photon benchmark. Browser angle input and contact text were verified, and the user's original pose was restored exactly. That pose has one root-palm pair at 0.11414 mm; it is not reported as collision-free. Evidence: traffic_before.json, traffic_after.json, processing_after.json, latency_validation.json and viewer_before_latency_fix.log.


## 2026-10-21 compact 116-shape teacher training

User requested training with the current interactive compact asset. A separate from-scratch run uses 4096 environments, seed 42 and 5000 PPO iterations on 22031 physical GPU 1. It uses the official unified isaaclab train entrypoint, physics=newton_mjwarp and the already prepared compact USD (SHA256 a13e3f0ac0482d88c57a9050b4095fde62cc01e9db5dd6053d6dd19ce8f0c650), selected by a per-run env.scene.robot.spawn.usd_path override. No default task or SDF checkpoint was switched. The original observation noise, randomization, 21-action order, 49/38 observations, reward, X/Y goal sampler, 120 Hz physics / 60 Hz policy / two internal substeps and actuator parameters remain.

The first 4096 attempt at nconmax/njmax=1024/1024 failed at CUDA Graph instantiation due to OOM. A 64-world/600-step large-buffer audit observed shared contacts 40088 and at most 195 constraint rows per world, with no overflow/warnings. The 896/512 graph attempt also OOMed (31404 MiB process peak). No failed graph run is counted as passed.

Accepted configuration: nconmax=896, njmax=512, use_cuda_graph=False; other integration/convergence/contact settings unchanged. The 4096-world/120-step mixed-action and full/selective-reset audit passed with observation corruption and task randomization retained. Actual hand shapes=475136 (116/world), solver geoms=117/world; shared contact peak=2554915 / 3670016 capacity, maximum per-world constraint rows=231 / 512, overflow=0. Loop time=30.203 s, process GPU memory peak=20014 MiB sampled approximately every second. Complete log contains no capacity or convergence warning. This is short workload validation, not a guarantee for every later policy state or convergence.

Production saves checkpoints every 100 iterations, records process memory and exits on reported capacity overflow or solver non-convergence rather than silently continuing. Saved params were checked for the exact asset/counts/settings. Initial PPO updates and finite checkpoint tensors were verified. Run directory: /home/zhangke/viserDex_revo3/logs/rsl_rl/viserdex_revo3_compact116/2026-10-21_06-55-39_compact116_env4096_seed42. Evidence and launch commands: outputs/revo3_compact_training_20261021/{training_launch,training_verification,teacher4096-plan,teacher4096-state}.json, teacher4096.log, capacity64.json and smoke4096_ncon896_nj512_no_graph.json. Existing SDF training was not stopped or modified.


## 2026-10-21 RSLCube geometry and material audit / correction

The earlier generated RSLCube_0p9.usda is not geometrically equivalent to the source collider in Newton. In a live one-environment SDF task, its boundingCube approximation was refitted to a rotated 82.415 x 58.500 x 82.415 mm box, while the source visual cube measures 58.5 mm on each axis. Its inertia also differed from a uniform cube. The approximated viewport mesh imported as one visual mesh and used only the red T material. These findings supersede any earlier assumption of source/runtime collision equivalence.

The original data/usd/RSLCube/RSLCube_0p9.usd is preserved (SHA-256 b9d9f04a0765c442578a5b3a08fdefe532bd8315cbd0e4ae8052a330e3ce2a4d). Object preparation now expresses boundingCube as a separate standard USD Physics Cube with purpose=guide, at the exact mesh-local bounds and transform. The original mesh remains visual-only, so the official importer reads its six material subsets. The source H face (face 2) has inward winding; only its winding and corresponding face-varying UV order are corrected on the generated copy, preserving every vertex-to-UV association, vertex position, material binding and transform. No Newton/Isaac Lab engine source was edited.

The default RSLCube output is versioned as assets/generated/usd/RSLCube/RSLCube_0p9_exact_box_v1.usda. The earlier unversioned generated asset is retained unchanged so existing checkpoints/params retain their historical inputs. Future task creation uses the new version; currently running SDF-8192 and compact-4096 jobs were not restarted and retain their already-loaded, incorrect collider. Their training success rates must not be represented as performance on the repaired cube.

Live validation on 22021: the normal SDF task, seed 42, standard domain randomization, physics=newton_mjwarp, 10 mm gap; one actual BOX collider measured 58.499999 mm on each axis, centered in the original frame, plus six visual submeshes with six distinct source texture files. Randomized mass was 0.06603409 kg and diagonal inertia approximately 3.766419e-5 kg*m^2, matching m*a^2/6. Multi-view screenshots showed T/H/L and E/R/S. A 300-step zero/small-random-action smoke run with two explicit resets completed with finite state and no contact-capacity or numerical warnings. The display-only close-up moves the cube above the hand without stepping; those images are geometry inspections, not policy performance. The audit process exited 0 and released its allocation (peak 5680 MiB including GL). CUDA/GL registration remains unavailable under the masked device, and the renderer uses its supported copy fallback.

Production asset bytes on 22021 match the fully tested candidate. Both projects also verify the default version through the official Newton ModelBuilder USD importer. Source backups, generation/deployment/import reports and runtime evidence are in outputs/cube_asset_audit_20261021/. Source corrections are in prepare_object.py and objects.py. A new PhysX paired rollout, repaired-asset policy evaluation and repaired-asset teacher training have not been performed; this is asset/import/smoke validation, not a convergence result.


## 2026-10-21 repaired-cube teacher restart at 30 Hz

User requested training restart and 30 Hz policy control. The previous teacher process was interrupted; its checkpoints remain unchanged. This job starts fresh, seed 42, 5000 PPO iterations, using the repaired RSLCube_0p9_exact_box_v1.usda. Official unified training is retained; env.decimation=4 and env.sim.render_interval=4 give a 0.0333333333 s policy step, with sim.dt=1/120 s and two MJWarp substeps (120 Hz collision updates / 240 Hz internal solves). Episode duration remains 20 s (600 policy steps), rollout length remains 24 policy steps (0.8 s per rollout), and task rewards, observation noise, reset/randomization, goal sampler and hand collision asset remain. Contact capacities and CUDA graph settings are the same as this hand variant's prior validated run.

Launch, retained-checkpoint manifest, preflight and startup validation are saved under outputs/revo3_cube30hz_restart_20261021/. This job: Isaac-ViserDex-Repose-Revo3, 4096 environments. Checkpoints: /home/zhangke/viserDex_revo3/logs/rsl_rl/viserdex_revo3_compact116_cube30hz/2026-10-21_10-45-13_teacher_compact4096_cube30hz_20261021. At initial validation, 6 PPO updates completed with no importer, numerical or capacity warnings. Measured recent iteration time: 13.37 s; process GPU peak: 20296 MiB. This confirms startup only, not convergence. The detached monitor retains GPU samples and stops the owned job on capacity/convergence/non-finite/OOM errors.


## 2026-10-22 compact-4096 30 Hz capacity recovery

The 30 Hz repaired-cube teacher stopped at 2026-10-21 20:13 China time after
iteration 383. Its monitor detected genuine `broadphase overflow` messages:
nconmax=896 gives 3,670,016 shared slots at 4096 worlds, while the largest
reported requirement was 3,670,467 (the upstream diagnostic requested at least
897 per world). Model 300 is the last pre-overflow checkpoint; its actor/critic
weights are finite and its SHA-256 is
d70b706c7eec8a778ef17fbe296fcd9f7283cd0a7fc627bac8362b03068dc49b.
Updates 301..383 were not saved and cannot be recovered.

Following the public MJWarp capacity controls, this recovery increases only
nconmax to 1152, giving 4,718,592 shared slots and about 28.6 percent headroom
over the reported requirement. It retains njmax=512, 4096 worlds, 30 Hz policy,
120 Hz physics/collision calls, two solver substeps, CUDA Graph disabled,
the same compact 116-collider hand and corrected 58.5 mm cube, goal sampling,
noise, rewards and randomization.

The 4096-world, 96-step hold/zero/random-action test with full and selective
resets passed with overflow_bitmask=0 and finite state. The final clean repeat
observed 2,590,438 reported MJWarp contacts, 234 maximum constraint rows, and
25294 MiB process GPU peak. The first test completed its physics assertions,
but our log monitor incorrectly matched the JSON field `overflow_bitmask: 0`
as a warning and terminated cleanup. That false-positive record is retained.
The matcher now anchors capacity diagnostics to plain text lines, distinguishes
the word `overflow` from metadata field names, and is checked against all four
actual original overflow messages and normal reports. The clean repeated test
exited 0. This is bounded capacity validation, not proof over all future states.

Official unified training resumed from the copied model_300.pt with 4700
additional iterations, giving the original total target 5000. Its live log
confirms checkpoint loading and iterations starting at 300/5000. Initial
validation reached iteration 305 with no capacity,
convergence, non-finite or OOM warning. Checkpoints now save every 25 iterations
to reduce unsaved work. Recovery records: outputs/revo3_compact30hz_recover300_20261022/.
The current-run pointer is outputs/revo3_compact_training_active.json.
SDF-8192 remains stopped at the user's request.


## 2026-10-22 compact teacher model 3150 video

A continuous, unselected 30-second seed-42 single-environment rollout of model_3150.pt is recorded through the official run_play_cli and launch_simulation, using a display-only task variant and --train_env_cfg plus the active training run's explicit physics/asset/timing overrides. The final resolved config and actual model were audited: policy 30 Hz, physics 120 Hz with two solver substeps, nconmax=1152, njmax=512, CUDA Graph disabled, native MuJoCo contacts, 116 actual hand colliders, one 58.5 mm cube BOX and six distinct visual materials. Normal observation noise, startup/reset randomization and the 20-second episode timeout remain. Only environment count, GL visualizer and video recorder differ from training.

Artifact: outputs/compact_teacher_video_20261022/video/compact_teacher_model_3150_0000.mp4. H.264, 1280x960, 900 decoded frames at 30 fps, exactly 30 seconds at 1x simulation speed. Fixed frame samples at 2/8/14/15/20/26/29.97 seconds are nonblank and show motion. The clip is a qualitative preview, not an independent success-rate evaluation. Source checkpoint SHA-256: da2bd29d72cef8f0147a37ac51297219d4da42ab1cb80ca1ce2247bef29544c6. Video SHA-256: c5059d4515366965abf1c580e11652f25f8a9524366809039f4831c7b66357f1.

The recording process exited 0 and released its allocation (peak 690 MiB); the main 4096-environment training processes remained alive without fatal warnings. On the masked physical GPU 1, CUDA/GL registration reports the known invalid-ordinal error and the official renderer uses its supported copy fallback. No other CUDA error or physics overflow was found. Selection metadata, resolved-model audit, exact command, logs, fixed frame samples and full-frame decoding validation are retained beside the video.

## Hikrobot DB0447679 标定相机仿真预览（2026-10-22）

已将用户交付的 `hikrobot_DB0447679_revo3_calibration_20261022.zip` 导入
22031 的独立单环境预览。相机正式 JSON 保存到 `config/cameras/`；
原始包、校验结果、显示资产和录像位于
`outputs/hikrobot_calibrated_preview_20261022/`。

- `renderer/hikrobot_calibration.py` 使用官方 `CameraCfg`、
  `PinholeCameraCfg.from_intrinsic_matrix`、`OpenCvPinholeDistortionCfg`
  和 `NewtonWarpRendererCfg`；保持原始 1280×1024、非相等 fx/fy、
  偏移主点及五参数 Brown-Conrady 畸变，生成原生畸变相机射线。
- 世界位姿为 `T_world_base @ T_base_palm @ T_palm_camera`。
  标定的 `right_palm` 相对手基座有 (0.0155, 0, 0.055) m 偏移，
  不能直接把 `T_palm_camera` 当作基座相机位姿。矩阵转换为 ROS 光学方向和
  XYZW 四元数，不重复应用观察器 vision 的 Rx(180°) 修正。
- `scripts/visualization/preview_hikrobot_calibration.py` 注册独立显示任务，
  调用官方 `run_play_cli` 和两条 `VideoRecorderCfg`：原生相机 RGB 与
  Newton GL 外部视角。相机视野四角射线和轴只通过 GL 公共 `log_mesh` 绘制，
  不进入原生 RGB 传感器。
- 使用当前 compact116 非 SDF 手和修正的 RSLCube，运行时核对 116 个手碰撞体、
  1 个 58.5 mm 原生 BOX 来源资产、21 个动作；手/物体 USD SHA 保持不变。
  包内支架及 AprilTag 仅添加两个显示形状，不添加物理碰撞、刚体和惯量。
- 支架显示网格沿 raw tag +Z 后移 0.3 mm，消除与标签平面共面造成的闪烁；
  标签平面和标定变换保持精确。负 Z 可见面的纹理 U 朝向与留出实拍图核对。
- 静态对照将关节置零并按关节边界裁切，方块临时移出视野，截图后恢复完整状态；
  老师回放保留正常重置、域随机化、观测噪声和 20 s 回合。该静态姿态不是从照片估计。

实际验证：`model_4225.pt`（SHA
`a4c68bbb483fc7e24d3a92f79339bd751541832fe10522e639e113479c727de6`）；
Newton MJWarp，1 环境，30 Hz，dt=1/120，2 子步，nconmax=1152、
njmax=512、物理 CUDA graph 关闭。最终录像退出 0，独立进程显存峰值 764 MiB，
300 帧/30 fps/10 s，两个源流完整解码并核对同步。
运行时相机矩阵与标定变换链最大绝对差 6.62e-8；
原生相机矩阵 K 与正式标定一致。
OpenCV 在仿真原生 RGB 中识别出 tag36h11 ID 0，
未细化的像素角点与标定投影 RMS 差 0.934 px。
这项栅格化/检测检查不同于原标定静态帧 0.0333 px 的重投影指标。

产物：`hikrobot_calibrated_sim_preview.mp4`、
`calibration_real_vs_sim.png`、`camera_runtime_audit.json`、
`preview_validation.json`；完整执行计划为
`hikrobot_calibrated_verified-plan.json`。准备、验证脚本保存在同一输出目录。
先前的显示问题版本保留在 `before_fixture_depth_fix/`、
`before_tag_uv_fix/`，无引擎或 vendor 文件修改。

边界：当前是标定相机预览和原状态老师策略回放，不是 RGB 闭环控制。
真实桌面/世界坐标未标定，材质/照明不代表实拍，照片关节角未恢复。
学生观测适配器目前使用 robot root/base 作为其名义“palm”原点；
后续视觉闭环需显式统一到该输入坐标系，不能把标定 right_palm 位姿直接混入。
Newton GL 在 CUDA/GL 互操作失败时使用官方复制回退；Newton Warp 原生相机独立于此。
当前 4096 环境老师训练使用独立进程和 GPU，未改动其训练配置或停止其运行。


### 标定相机老师回放：逐目标成功计数（2026-10-22）

原 10 秒预览未记录成功事件。补测增加可选 Gym 只读包装器 `utils/goal_counting.py`，读取公开 `CommandTerm.command_counter`，排除初始/重置目标，同时核对目标四元数确实变化及切换前姿态误差 `<0.1 rad`。没有修改引擎、动作或随机化。逐步记录写入 `success_steps.jsonl`，统计写入 `success_summary.json`。

单环境首次补测遇到 broadphase overflow，监控器终止该进程，结果作废并保存在 `outputs/hikrobot_teacher_success_count_20261022/invalid_due_to_contact_capacity.json`。仅单环境回放将 nconmax 从 1152 增至 2048 后复测；4096 环境老师训练保持 1152 及原配置。

有效补测：`model_4225.pt`，seed 42，1 环境，30 Hz，300 步/10 秒，完成 12 次目标重定向，0 次终止/回合重置。退出 0，完整日志无接触/约束溢出、NaN 或 solver 告警，进程显存峰值 764 MiB。

产物位于 `outputs/hikrobot_teacher_success_count_safe_20261022/`，包含原生双视角视频、带累计计数的 `teacher_4225_success_count.mp4`、逐步事件及执行计划。此结果是同权重同任务的另一次短回放，不能当作原视频未记录的精确计数，也不是独立总体成功率评估。原相机预览完整日志未出现容量溢出。


### 学生实现审计（2026-10-22）

完整报告与源码快照、CPU/GPU 探针保存于 outputs/student_implementation_audit_20261022/。生产学生实现未改。发现当前 compact116_cube30hz 老师被 teacher.py 的实验名白名单拒绝；学生默认 rollout 不继承保存的老师环境，会选择原始手资产、60 Hz、640/1024 容量，而当前老师为 compact_v4、30 Hz、1152/512。物体已为同一修正 RSLCube。

另外复现了学生评估在自动重置保护步把未获得的新目标记作成功的反例（官方实际目标数 0，旧 consecutive_success 差分计 1）；输入元数据 palm 实际是手基座，与海康标定 right_palm 相差 (15.5,0,55) mm，需要在视觉接入处显式换坐标。RGB 学生回放也未检查 checkpoint step_dt。按控制步定义的延迟、更新周期、45 步 BPTT 在 30 Hz 下物理时长翻倍，需明确协议。

19 项现有 CPU 学生/视觉测试通过。隔离 GPU 探针显式对齐 compact116、30 Hz 和正确方块，只在探针内绕开实验名称筛选，严格加载并冻结 model_4400 老师；4 环境、45 步采样、每次 32 epochs、2 次更新，共 360 环境步，退出 0。干净状态、共享编码器读数和特权隔离检查通过，保存恢复逐位一致，无容量溢出、solver 不收敛、NaN 或 CUDA 错误，峰值 842 MiB。nconmax=2048、njmax=512 仅用于这个小环境数诊断。该检查不代表生产入口已修好或学生收敛。

### 学生相机坐标协议修复（2026-10-22）

已保留学生现有 articulation-root 数值观测，明确命名为 `right_hand_base_link`，协议为 `revo3_base_pose_xyzw_m_v1`（米、XYZW）。新增 `distillation/frames.py`，从交付的 Hikrobot 外参读取并校验 `T_base_palm`、`T_base_camera` 及其矩阵/四元数一致性和变换链。`StudentObservations` 支持显式的 `camera_opencv`、`right_palm` 输入；缺少标定时拒绝跨坐标系输入。已经在基座下的输入不再次叠加偏移。外部位姿转换只读取估计值和标定，不读取物体真值。

蒸馏恢复、导出、RGB 学生回放均校验坐标元数据；历史 v1 中含糊的 `palm` 标签按其原本的根坐标数值协议解析并记录，新导出布局使用准确的基座名称。RGB UVD 解码器改用 `keypoints_to_frame_pose` 名称；兼容保留旧函数及关键字签名。现有网络输入形状、教师及学生权重不变。

验证：35 项 CPU 测试通过（含新旧坐标元数据的 TorchScript 导出/重载）；Ruff 通过。启用原感知噪声的修复前后 8 步数值对照包含重置，学生全部观测与监督逐位一致。官方 SimulationRuntime + Newton MJWarp 在 compact116/正确方块资产、30 Hz、4 环境执行 120 步，分别由相机和 URDF 掌部位姿构造学生输入，共 960 次环境位姿对照，最大特征绝对差 1.7881393432617188e-7；无容量/数值警告，退出码 0，独立进程峰值 726 MiB。资产中合并静态根 `world` 到 `right_hand_base_link` 为单位变换，名称合并不改变基座协议。证据位于 `outputs/student_camera_frame_fix_20261022/`。

本次只修复坐标协议及转换，先前审计的老师命名校验、学生默认环境/30 Hz 绑定、成功计数问题仍未修复。没有启动正式学生训练；既有老师训练保持运行。几何测试未使用相机图像，不代表真实 Hikrobot/CNN 闭环已接通；当前针孔 UVD 路径接原始图像还需匹配 Brown-Conrady 畸变与裁剪后的内参。

### 学生视觉入口切换为海康标定视角（2026-10-22）

`play_student_vision.py` 默认任务改为 `Isaac-ViserDex-Repose-Revo3-Hikrobot`，实际图像源为官方 CameraCfg + NewtonWarpRenderer 的 `sensor:hikrobot_camera:rgb`，复用此前可视化视频的同一个标定相机工厂。内外参、1280×1024 和 Brown-Conrady 五参数来自交付文件；全场景 RGB 自然包含活动手部遮挡。新任务使用 compact116、正确的 RSLCube、30 Hz、两个物理子步，单环境容量 2048/512、CUDA graph 关闭。旧基础任务和既有老师训练配置没有修改。

新增 `HikrobotImageProcessor`：输入只接受原生 uint8 RGB，按完整画幅同时去畸变并缩放到 120×120，不使用物体/手部真值、不按物体裁剪。缩放后的 K 含半像素中心修正，供 UVD 解码器使用；默认保持原有 binary opening。学生回放核验坐标元数据和控制周期，视频关掉时仍启用相机，并使用公开 command_counter 排除重置目标计数。输出保存原生相机视频与 CNN 输入对照、K/D/外参和标定哈希。

验证：40 项 CPU 测试通过，含 OpenCV 独立畸变网格对照、全幅像素中心缩放、RGB 通道、校正后 UVD 位姿重建；Ruff 通过。无额外覆盖时的新任务默认配置解析验证通过：30 Hz、compact116、2048/512、无旧 GS 相机或位姿观测项。官方 unified play 使用真实 model_4225 老师运行单环境 300 步，生产相机外参相对于此前视频完全一致（运行时容差 1e-6，标定变换绝对差约 6.62e-8），K/D 相同。10 秒原生 1280×1024 视频与 1280×704 对照视频均完整解码为 300 帧、30 fps；无碰撞容量/数值警告，退出码 0，独立进程峰值 758 MiB。证据位于 `outputs/student_hikrobot_view_20261022/`。

31 上未找到正式学生或 CNN 权重，因此本次 GPU 回放由现有老师驱动手，验证了学生实际使用的相机和相同图像预处理，未声称学生 RGB 控制效果或新视角 CNN 精度已验证。旧 GS 位姿训练任务仍保留；对应 CNN 需按新视角与完整场景输入训练/验证。CPU 隐藏 CUDA 的配置解析出现预期 Warp 无设备提示；Pinhole 工厂的 Omniverse 相机限制提示不代表此处 NewtonWarp 的实际 K/D，后者已通过运行时标定核验。

### 4096 环境学生蒸馏正式启动（2026-10-22）

按用户要求以 SIGINT 停止原 compact4096/30 Hz 老师优化，原监控退出 0、进程释放。固定最后已保存且全张量有限的 `model_4800.pt`（SHA-256 `fc3a2e23b13eef53d8c192672436bc949dece8c5afa423c3bf7138e85e9e52c0`）及相邻参数到 `outputs/student_dagger_start_20261022/frozen_teacher/`。教师用于 DAgger 时只作冻结的确定性标签网络，不再优化。

正式入口问题已修复：通过有检查点/环境哈希、明确 XYZW/49+38 观测/21 关节顺序的 `teacher_protocol.json` 验证当前 compact 命名空间，保留旧协议拒绝；在 SimulationRuntime 调用官方 launch_simulation 前，通过官方 configclass.from_dict 恢复老师完整保存环境。只容许环境数、设备、种子、日志和不减小的接触容量等操作性差异，核验环境契约与手/物体 USD 哈希；源记录中的 spawner.spawn_path 是运行时输出，重新生成而非恢复。恢复/纯学生评估使用相邻 env.yaml 及其 SHA-256，避免默认资产和 60 Hz 漂移。成功次数改用公开 command_counter，排除初始与自动重置目标。

4096 环境的 BPTT 采用每块 128 环境的加权梯度累积：每个 epoch 处理完整 4096×45 序列，全部块累积后才做一次全局裁剪/AdamW step，仍为 32 epochs，归一化统计每批只更新一次。CPU 数值对照验证与整批计算一致，覆盖不整除的块、重置和优化步数；没有环境下采样或缩短序列。噪声维持步数默认值，30 Hz 下记录关节延迟上限约 33.3 ms、视觉更新周期上限约 100 ms、BPTT 1.5 s；不冒充硬件实测值。

验证：47 项 CPU 测试与 Ruff 通过。未旁路正式入口，4 环境完成 2 次完整 45 步/32 epochs 更新，峰值 830 MiB；恢复继续第 3 次更新，冻结老师与 beta/优化器/归一化保留；TorchScript 导出/回读数值校验通过；纯学生 4 环境/120 步评估退出 0，含 6 次重置、teacher_used=false，未使用教师动作。上述 smoke 只验证执行链路，不表示收敛。

正式任务 `logs/distillation/revo3_v3/student_compact4096_30hz_teacher4800_20261022` 已启动，GPU UUID 为原老师使用的 `GPU-ee91e573-32c8-e473-64a6-15dd5bc05e5d`。4096 环境、30 Hz、compact116 和正确 RSLCube、原 nconmax=1152/njmax=512、累计预算 1000 次更新，首次及每 10 次保存。首个正式更新覆盖 184320 环境步，rollout 25.02 s、学习 119.66 s、合计 144.67 s，损失/梯度有限、干净源观测与共享关节/老师特权状态审计误差均为 0，`student_1.pt` 已保存；初期进程显存峰值 26124 MiB（约 25.5 GiB），未见容量/数值警告，持续下一轮。该阶段不创建相机/CNN。活跃指针为 `outputs/revo3_student_training_active.json`；完整启动证据在 `outputs/student_dagger_start_20261022/`。这是启动验证，后续训练表现与完成时间须查实时日志。

### 学生序列训练性能修复（2026-10-22）

已确认原蒸馏慢的主要原因是训练侧序列实现效率：每个 45 步序列逐时间步调用整套网络，4096 环境按 128 分块、32 epochs，每轮约 46080 次完整单步 forward；每个分块又将三个损失转为 Python float，引入重复 CUDA 同步。原训练末 10 轮中位采样 24.24 s、学习 116.93 s、总 141.12 s。没有物理容量或数值异常。

`BeliefStudent.sequence` 已将逐帧归一化、外感知编码及全部 MLP 输出头按时间/环境维批量计算，GRU 仅在存在回合重置的时间点拆段，对实际重置环境清除隐藏状态并阻断梯度；其余连续区段一次处理。在线控制 `forward`、网络参数名称/形状和输入输出布局不变。指标在 GPU 上累积，每个 epoch 的 AdamW 更新前核验有限性，完成更新后一次读回；保留 NaN 失败保护。4096 环境、45 步、32 epochs、128 环境梯度累积块、目标分布、噪声和物理设置均未缩减或更换。

验证：52 项 CPU 测试与 Ruff 通过，新增无重置/稀疏重置/全部重置/首末重置的输出、隐藏状态及全部参数梯度对照，CPU 双精度绝对容差 1e-12；并验证含非有限标签的分块不能推进优化器。独立 GPU 比对中，禁用 TF32 时最大输出绝对差 1.19e-6、梯度相对 L2 差 1.18e-7；启用原有 TF32 时最大差 9.62e-4、梯度相对 L2 差 5.87e-5，来自矩阵批形状改变的舍入，在线单步 forward 的同权重输出逐位一致。最初对 TF32 套用 2e-4 的绝对阈值触发比对失败，已保留初始日志；严格 FP32/CPU 等价对照与 TF32 实测误差分别记录，没有将其当作物理错误或隐去。独立性能测试使用加载真实 student_10 权重、形状与重置概率匹配的合成批次，不是生产采样数据：4096×45、2 epochs、块大小 128，从 9.54 s 降至 1.07 s；一个分块 forward/backward 中 Linear 调用从 1035 次降至 23 次，GRU 从 45 次降至实测 7 段。该 8.9 倍仅描述独立网络更新测试。

正式任务在完成第 21 次更新后通过已有 SIGTERM 处理器原子保存 `student_21.pt`，校验所有张量有限、归一化计数 3870720、每参数 AdamW step=672，再完整恢复继续第 22 次更新。优化器、归一化、beta、冻结 teacher4800 和累计 1000 次预算保留；模拟器按既有恢复约定开启新回合。新目录为 `logs/distillation/revo3_v3/student_compact4096_30hz_teacher4800_batched_20261022`，活跃指针已更新，旧 run 完整保留。

真实 4096 环境第 22–25 轮的整轮耗时分别 43.24、41.49、41.61、40.13 s，中位约 41.55 s；采样约 24–25 s，更新约 16–18 s，整体比原末 10 轮中位快约 3.4 倍。进程显存峰值约 26132 MiB（25.5 GiB），全部源观测/共享关节/老师特权状态审计误差为 0，未出现容量或数值警告。优化后通过 SIGUSR1 额外保存并校验 student_24 检查点，训练继续运行，老师优化仍停止。本次验证的是数值与执行性能，不表示长期收敛或学生成功率已验证。证据位于 `outputs/student_performance_audit_20261022/`。

## 2026-10-22 student learning diagnosis

The earlier status report compared training BC/reward across changing DAgger
mixtures; this is insufficient to establish student-only convergence or failure.
Frozen checkpoint student_200.pt was tested independently, with 64 environments,
600 steps (20 simulated seconds per continuously running environment), seed 53,
30 Hz and the saved teacher/student environment contract. Small-batch diagnostics
only raise nconmax to 2048; task physics, randomization, actions and assets remain
the saved configuration. There are no cameras/CNNs in these tests.

Observed earned goal changes / object-out-of-reach terminations:
teacher clean 1466 / 2; teacher with current shared joint noise/delay 1333 / 10;
student_200 with full training noise 236 / 26; student_200 with both pose and
joint noise disabled 253 / 4; student_40 with full noise 506 / 143. All five
600-step runs exited zero with no capacity/numerical fatal patterns. Student
evaluation executed student actions exclusively. An extra frozen teacher in the
instrumented evaluations computed diagnostic labels only, never control actions.
Initial/reset goals are excluded by the public command_counter accounting.
Counts are repeated goal achievements, not Bernoulli success percentages.
Completed episodes and termination counts differ between policies.

The student is partially learned, but below this teacher. From update 40 to 200,
holding becomes more stable while goal throughput declines; neither BC alone
nor lower average orientation error (which is reset by frequent new goals) can
replace the independent goal/termination measurements. Removing observation
noise does not close the gap. On student_200 states, teacher sensor perturbation
changes its action by MSE 0.1170, versus student/teacher MSE 2.1926.

A real 45x64 rollout batch was replayed and optimized separately. Online
stepwise inference with matching TF32 settings exactly reproduced saved online
actions. Batched/stepwise full FP32 forward RMS difference is 9.11e-7, gradient
relative L2 difference 8.27e-6; the matching-TF32 batched/online action RMS
difference is 4.93e-4. These are precision/batching comparisons, not bitwise
training guarantees. Repeated fitting of this one diagnostic batch reduced BC
from 2.0527 to 0.01227 over 10x32 optimization epochs. This establishes working
optimization on those samples, not held-out or closed-loop success.

The public fork and current code both use beta=0.9*0.95^completed_updates and
latest-rollout-only training. Beta is approximately 0.00533 after 100 updates
and 0.00003155 after 200. The formula was not accidentally changed by the
Newton migration. Its adequacy for the changed assets/30 Hz/4096 setup is not
established. A paired short 256-environment diagnostic derives two clearly
labelled checkpoints from the same student_200 model/optimizer/normalizers,
changing only fixed beta (0 versus 0.3), runs 20 complete updates with otherwise
identical settings, then evaluates both without teacher actions. These are
isolated diagnostic runs; the production 4096-environment run continues.
Final paired results follow below.

Full plans, source snapshots, model provenance, measured losses, capacity logs,
and evaluation JSON are under outputs/student_learning_audit_20261022/.

Paired diagnostic finished: both 256-environment continuations completed updates
201-220 (32 epochs/update) from model/optimizer/RNG-identical derived checkpoints.
All tensors are finite; optimizer step=7040 and normalizer count=37094400 in
both. Subsequent student-only 64-environment/600-step tests observed beta0:
211 earned goals, 36 object-out-of-reach terminations, mean reward 0.0736023;
beta0.3: 622 goals, 127 object-out-of-reach terminations, reward 0.171802.
Both use the same evaluation seed and training-noise distribution. All audit
jobs exit zero with no capacity/numerical fatal patterns. This supports strong
curriculum sensitivity, while the larger drop count prevents calling fixed
beta0.3 a validated repair. It is a short 256-world/single-seed experiment,
not a 4096-world convergence demonstration. No production dynamics, noise,
weights, or curriculum were changed; the 4096 job continues. Complete measured
results and limits: outputs/student_learning_audit_20261022/learning_diagnosis.json.

## 2026-10-22 recurrent student correction and 4096 restart

The previous student finished complete update 298 after SIGTERM and exited 0.
student_298.pt was retained with finite model/optimizer tensors, AdamW step 9536,
normalizer count 54927360 and SHA-256
5c1cf2a3d599b9af088d70e12b2ad6db0b58c2c3e84197b9fbb3383e88bd6807.

The correction preserves the original network, losses, all task randomization,
assets and saved physics. It decouples logical optimizer batch (256 sequences)
from 4096-world rollout parallelism and 128-world memory chunks, rebuilds GRU
memory after parameter/normalizer updates, and gives the frozen teacher a
sign-consistent quaternion representation. Controls and backward compatibility
are documented in revo3_distillation.md and checkpoint training_cfg/metadata.

The frozen teacher is capable under Newton: independent 64-world/600-step clean
input tests observed full teacher 1474 goals / 6 out-of-reach terminations;
zeroing velocity input 307/368; replacing joint velocities by position finite
differences 1364/9; replacing all velocity inputs by finite differences 1280/15;
canonicalizing quaternion representation 1173/5. Physical randomization stays
enabled. These information ablations show important information dependencies,
not a proof of the historical PhysX/Newton causal difference.

Real-data q/-q test: student pose features change 0, raw teacher action-label
MSE 4.78717, canonical teacher label maximum change 0. Saved original labels
recompute exactly. Real GRU update: hidden RMS drift 0.27653; next-action teacher
MSE 1.00564 with stale memory versus 0.66865 after replay. These are reproducible
specific tests and do not imply that every earlier performance difference has
a single proven cause.

The corrected job was freshly initialized with the frozen teacher4800:
logs/distillation/revo3_v3/student_compact4096_30hz_minibatch256_canonical_v2_20261022.
Official v3.0.0 launcher/runtime, physics=newton_mjwarp, policy 30 Hz,
dt1/120 with two solver substeps, compact116, corrected RSLCube,
nconmax1152/njmax512 and CUDA Graph disabled remain in use.
Update1 covered 184320 samples and 512 optimizer steps; all checkpoint tensors
are finite, normalization count184320, teacher/asset/environment hashes verified.
Process GPU peak at initial verification26224MiB; first iteration35.98s
(24.98s collection,10.99s update including memory replay).
No capacity/numeric fatal patterns were observed. Independent student-only
evaluations of checkpoints1,5,10 and two longer 1024-world controls are tracked
separately. Mixed-control training goal counts are not student success results.
Teacher PPO optimization and the old SDF training were not restarted.

Evidence, source backup/manifests, all accepted/rejected harness logs and exact
plans: outputs/student_root_cause_20261022/.


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


## 2026-10-23 calibrated Hikrobot pose predictor training

The native calibrated RGB pose task is deployed on server22021 at the same
/home/zhangke/viserDex_revo3 path. The4096-world/30Hz student remains running
on22031 GPU1 with unchanged process2371464 and unchanged student source.
The new collect_hikrobot_pose.py restores frozen teacher4800 environment and
asset hashes before attaching the existing official CameraCfg/NewtonWarpRenderer.
It uses the deliveredDB0447679 intrinsics, distortion and T_base_camera, native
1280x1024 RGB, full-frame rectification and120x120 float32 pixels.
Labels are synchronized center+8corners UVD with optical depth in meters.
The CNN reads RGB only; saved simulator state is exclusively label/audit data.
Native hand occlusion is retained without oracle masking or ground-truth crops.
Unknown occlusion fractions are explicitly-1, not fabricated percentages.

GPU6/7 on22021 collect and GPU3 trains the original ResNet34+512/256head.
Initial8192/2048images use independent seeds42/43,10epochs,AdamW1e-4,batch64.
Concurrently collect131072/16384images with seeds44/45; after completion and
independent audit, initialize a new10-epoch run from initial best.pth with a
fresh optimizer/schedule. Exact plans, memory limits and pipeline failure/exit
records:outputs/hikrobot_pose_training_20261023 on22021.

Validation:57CPU tests pass,1CUDA-only test skipped with CUDA hidden;Ruff passes.
4-world128-image smoke exits0,peak948MiB,all4camera extrinsics within6.62e-8
of calibration. Independent saved-pose reprojection max error8.22e-8.
Native RGB/label previews were visually inspected. Initial datasets completed
and passed audit. CNN optimization and validation run successfully, with finite
model/optimizer checkpoint tensors. This verifies launch, data alignment and
initial learning; it is not real-camera accuracy or closed-loop student proof.
See22021 outputs/revo3_pose_training_active.json for current pipeline phase.


## 2026-10-23 original object-only pose protocol restored


The prior full-scene RGB experiment in outputs/hikrobot_pose_training_20261023
was rejected by the user because it changed the original image protocol.
Its controller and collection jobs were stopped, its source archived, and its
weights are excluded from this restart. No accuracy from that experiment
establishes correctness of the original protocol.

The restarted path uses the existing GaussianSplatCamera and Revo3 live hand
MultiMeshRayCasterCamera. Only object RGB is rasterized. A foreground object
pixel is set to black when the hand's optical-axis depth is nearer than the
object depth. Hand RGB is never composited into the network input.
The existing renderer perturbations, binary opening, ISO/motion blur,
ResNet34 plus 512/256 head, 27 UVD outputs, Huber loss and offline trainer are
preserved. Nine core source files match the pre-experiment snapshot; see
original_source_contract.json. This restores the project's existing Newton
Revo3 workflow; sample budgets are project settings, not paper requirements.

The delivered DB0447679 calibration and viewpoint are retained. Synthetic
rendering directly uses the rectified 120x120 K produced by the existing
HikrobotImageProcessor. The real-image input path must use the corresponding
rectification. The base-to-camera and base-to-palm transforms are composed
once, using the frozen teacher's live hand-base pose. The 4-world smoke has
camera extrinsic max absolute error 9.8211e-8 and matching GS/ray intrinsics:
fx=122.86667633, fy=153.50335693, cx=60.31119919, cy=58.38762665.

The frozen teacher model_4800.pt and saved task/assets are restored before
attaching the existing pose sensors using the official SimulationRuntime
configure hook. Newton 1.6.1, Warp 1.17.0, Isaac Sim 6.2 and Isaac Lab v3.0.0
remain in use. No shared runtime or vendored asset was modified.

Production on server 22021:
- GPU6: 300,000 train samples, 256 worlds, seed42.
- GPU7: 30,000 validation samples, 128 worlds, seed43.
- Original burn-in120, stride4, occlusion5%-95%, minimum32 visible pixels.
- After both complete, the unchanged audit must pass for both datasets.
- GPU3: fresh ImageNet initialization, 10 epochs, batch64, AdamW1e-4,
  1000 warmup steps, save every500 optimizer steps. No rejected weights.
- run_original_pipeline.py and the three *-plan.json files record exact commands.
- Training only starts after complete datasets pass audit. No pose model is
  automatically installed into the running student's control loop.

Verified before production:
- CPU tests:40 passed,1 CUDA-only test skipped with CUDA hidden; Ruff passed.
- 4-world,128-image smoke exit0, own peak774MiB; dataset audit passed.
- Independent audit verifies saved pixel removals, depth order, every UVD label,
  finite values, articulated hand motion and object rotation.
- Smoke occlusion median26.48%, maximum40.45%; this is not a heavy-occlusion
  or real-camera accuracy claim.
- Actual stored RGB was inspected and exported as actual_object_occlusion_samples.png.

The server22031 student process2371464 remains running with4096worlds,30Hz.
It was not stopped, restarted or changed by this correction.
Live pose status: outputs/revo3_pose_training_active.json.
Evidence: outputs/hikrobot_gsplat_occlusion_20261023/.


## 2026-10-23 author-default6200 online pose training


The user explicitly selected the author's 6200-step online training workflow.
This supersedes the offline ten-epoch plan. The offline controller2191696 was
retired before any offline CNN launch. Both collected datasets completed and
are retained, but the online optimizer does not read them.

Production runs on server22021 GPU3
(GPU-fd7fa4b1-12b3-051a-67c9-6a8be0f5a2f9):
logs/rsl_rl/pose_estimator/revo3_hikrobot_online6200_env1024_v2_20261023.
It uses the existing train_pose_estimator.py loop, 6200 teacher rollout steps,
1024parallel worlds and16minibatches per observation batch (64images each).
The upstream default is6200rollout steps, not6200offline epochs; the paper
does not publish a total pose-estimator epoch count. The world count is an
operational choice for this Revo3/Newton32GiB GPU, not a paper-matched claim.

Only the existing online entrypoint gains optional frozen-teacher and camera
arguments. It restores the saved teacher task/assets before the official
SimulationRuntime/launch_simulation hook, attaches the already validated
calibrated GS/hand-depth sensors, and enables the existing online PoseEstimator.
The frozen teacher4800 generates the trajectory and is not optimized.
Fresh ImageNet initialization is used; smoke/rejected/previous pose weights are
not loaded. Existing ResNet34/MLP, keypoint loss, renderer augmentations, ISO,
motion blur, binary opening, optimizer and schedule implementations are unchanged.
Nine original core files still match the pre-experiment source hashes.

Input remains object-only3DGS RGB with live hand-depth ordered black occlusion.
The delivered Hikrobot DB0447679 calibration and viewpoint are retained;
1024camera transforms/intrinsics pass audit. A fixed1micrometre world-position
assertion initially rejected1.233595micrometres of float32 rounding on the large
world grid. The audit now allows1micrometre plus2float32ULPs per position axis,
while retaining a fixed1e-6 rotation-matrix tolerance. It changes no camera or
rendering values. A regression test accepts the recorded rounding and rejects
1mm translation errors and0.01rad rotation errors; all3geometry tests pass.

Startup validation:
-64-world,30-rollout-step smoke exited0; last recorded optimizer step272.
 Saved weights are finite and backbone weights differ from ImageNet.
-Production reached80/6200rollout steps and1232actual optimizer updates.
 First checkpoint cnn_59_requested.pth is finite; backbone conv1 maximum
 absolute change from ImageNet is0.00242720824.
-Own process GPU peak18274MiB; no capacity, numerical or OOM fatal events.
-Entry source compiles. Ruff excludes pre-existing launcher import-order and
 unused-import findings; calibration/audit tests pass normal Ruff.
-The first smoke needed termcolor3.3.0, installed only in the private runtime,
 matching server31. No engine/shared runtime changes.
-Validation establishes launch and weight updates, not convergence or real-camera accuracy.

Server22031 student process2371464 remains running with4096worlds,30Hz.
Its process, training configuration and policy source are unchanged.
The online pose CNN is not installed automatically into that student.

Exact plan/state/log, camera and saved-task audits, source fingerprints,
smoke/production checkpoint validations and rejected launch evidence are in
outputs/hikrobot_online_pose_20261023/.
Live pointer: outputs/revo3_pose_training_active.json.


## 2026-10-23 completed online predictor and independent-seed evaluation

The online predictor completed6200rollout steps in5635.40s; guard exit0,
no fatal event, own GPU peak18274MiB. Final checkpoint:
logs/rsl_rl/pose_estimator/revo3_hikrobot_online6200_env1024_v2_20261023/pose_estimator/cnn_6202_final.pth
SHA256:6a940233e65146bf34f1051a593727f79fa0821d0d8bcc7feec18b07a67794ed.
All saved tensors are finite. The estimator-call count6202 includes startup
observations; the requested online rollout limit was exactly6200.

The final checkpoint was evaluated once on the retained30000-image seed43
Hikrobot-calibrated GS/live-hand-occlusion dataset, after the unchanged dataset
audit passed. Training used seed42; this evaluation did not select or update
weights. Existing evaluate_pose_dataset.py uses deterministic binary opening,
no stochastic augmentation, float64 pose geometry, matmul TF32 disabled.
Mean translation2.816757mm, rotation1.871527deg, ADD3.141668mm,
pixel error0.393483px; fraction with translation<10mm and rotation<10deg
is0.992566645. ADDp95=6.643908mm. Only7images have occlusion>=50%; these
simulation results establish neither heavy-occlusion robustness nor real-camera
accuracy. Evaluation guard exit0 with no fatal event.

Server22031 student also completed1000updates and184320000environment steps,
guard exit0, no fatal event. student_1000.pt and optimizer state contain227
finite tensors. The latest independent policy evaluation remains older; final
student1000closed-loop task success has not been re-evaluated here.
Evidence:outputs/hikrobot_online_pose_20261023/evaluation_final6200_seed43.json
and training_completed_evaluated_report.json. Active pointers now mark completion.


## 2026-10-23 final student1000 matched independent evaluation

student_1000.pt completed1000updates /184320000environment steps with guard exit0.
A fresh pure-student evaluation used the existing distill.py --evaluate path,
64worlds,600steps,seed53,30Hz,original pose and joint noise, no teacher actions.
It completed767target increments and58object_out_of_reach terminations;
84episodes ended, mean completed-episode successes6.642857. Matched checkpoint225
had704target increments,70out-of-reach terminations,94completed episodes and
mean5.372340successes. This one-seed test gives8.95%more goals and17.14%fewer
out-of-reach events; it does not establish multi-seed statistical significance.
The final student can control independently but still loses objects.
Inputs are the checkpoint's simulated noisy pose and proprioception; the newly
trained image pose CNN is not part of this evaluation. Evaluation guard exit0,
no fatal event, own GPU peak1360MiB. No weights/configuration were changed.
Report:outputs/student_root_cause_20261022/status_20261023_checkpoint1000.json.


## 2026-10-23 teacher4800 paired with final student1000

A fresh teacher-only baseline used distill.py --evaluate --teacher_only with
student1000checkpoint metadata,64worlds,600steps,seed53,30Hz,canonical-world-W
teacher protocol and the same joint-noise settings as final student evaluation.
Teacher4800 retained clean privileged object/velocity state by design; the
student used simulated noisy pose. Teacher joint-input equality and privileged
state audits passed. No weights were updated. Evaluation guard exited0.
Teacher:1123goal increments,7object_out_of_reach events,64completed episodes,
mean17.28125successes per completed episode. Student1000:767goals,58out-of-reach,
84completed episodes,mean6.642857. Student's goal count is68.30%of the teacher's
in this one-seed20s-per-world test; this is not a statistical multi-seed result.
Old teacher_info_full's1474goals used clean observations and a raw-quaternion
protocol, so it is not this matched baseline.
Evidence:outputs/student_root_cause_20261022/teacher_vs_student1000_seed53.json.


## 2026-10-23 final student gap diagnostic ablations


Read-only tests on the existing student1000checkpoint:64worlds,600steps,seed53,30Hz.
No student/teacher weights or training implementation were changed.

| Policy/input | Goals | Object out of reach |
|---|---:|---:|
| Student, full pose/joint noise |767|58|
| Student, clean pose/original joint noise |922|48|
| Student, clean pose and joints |1012|28|
| Teacher, original joint noise/privileged truth |1123|7|
| Teacher, clean joints/privileged truth |1217|7|

This supports two contributors: noise robustness and a residual control imitation/state
inference gap. It does not establish one specific implementation fault. The CNN pose
estimator is not part of these tests. Training completed its budget; it did not reach
teacher performance.

An additional shadow rollout measured38400student states without changing returned
actions. Input position vector RMSE79.16mm was reduced by the auxiliary decoder to
11.37mm; mean orientation error24.60deg became11.68deg. Normalized velocity-readout
R2:joint=.921,object linear=.596,object angular=.546. Action RMSE relative to a frozen
teacher in raw policy-output units=.849; large-observation-error states had1.044 vs
.740elsewhere. These are diagnostic associations, not isolated causal interventions.
The actor uses fused latent features, not decoder outputs directly.

Limitations:single seed, closed-loop trajectories differ across noise conditions.
The instrumented shadow rollout got786goals/46out-of-reach vs uninstrumented767/58;
its readout metrics should not be presented as an identical-trajectory replay.
Teacher forward defaults to deterministic output; source files match pre-test hashes.
All four diagnostic GPU jobs exit0, no capacity/numerical fatal event.

## 2026-10-25 Geometry SuperMimic checkpoint20600 video

`scripts/supermimic/record.py` records the saved actor through the official
SimulationRuntime/launch_simulation and NewtonGLVisualizer.render_rgb_array APIs.
The latest checkpoint was frozen at iteration20600 before recording. Seed42,
one environment,1800 consecutive steps,30Hz,60seconds at1x speed; no trajectory
selection or temporal cuts. Saved BF16 actor inference,10x124 observation history,
observation noise,domain randomization,assets,EMA and episode rules are preserved.
Evaluation omits the training-only additional exploration action noise. Only the
actor is loaded; the world model and optimizer are not needed for deployment.
Single-world nconmax2048 uses the previously verified playback capacity; physics
settings otherwise retain the saved contract. No running training job is stopped.

Enabling the official visualizer imports visual-only shapes alongside colliders.
The original all-shapes assertion therefore rejected the object at initialization.
Adapter now filters shape scales/material context using official
`ShapeFlags.COLLIDE_SHAPES`, without changing any shape or collision flag. Both
rendered and non-rendered synthetic masks pass; the rendered rollout preserves
the checkpoint's entire context schema and geometry digest. Live box-half-extent
error is0, FK position error1.68e-7m, reward error9.54e-7, and success-boundary
disagreements0. No contact-capacity/NaN/fatal error occurs. The known masked-GPU
CUDA/OpenGL interop warning uses the official copy fallback.

The uninterrupted60second recording completes34goals,4drops and2timeouts.
Completed-episode goals are15,7,9,1,0,2. This single-environment demonstration is
not a population success-rate evaluation. Both the native video and its fixed
spatial close-up decode to1800frames,1280x960,30fps. Close-up retains every frame,
timing and HUD; only a fixed crop/resize is applied. Close-up SHA256:
`152f33ce16dc4596384e08286a1bf0160750047ec190b90f1ab3ff005369b3e5`.
Evidence on22021:
`outputs/geometry_video_20261025_1245/rollout_v2/` includes video,manifest,
resolved environment,summary,per-step events,trajectory and validation JSON.
Recording process has exited and released its GPU allocation; main training was
still active at iteration20817 on GPU0/4 after recording.


## 2026-10-27 paired policy video: SuperMimic 51000

Recorded one continuous 60-second seed-42 actor-only rollout from a frozen iteration-51000 checkpoint on physical GPU4, keeping the saved 20 Hz dynamics, assets, domain randomization and sensor-noise protocol. Official SimulationRuntime/NewtonGL recording uses a single-world operational nconmax=2048; the environment contract and geometry checks pass. Training on GPU1/GPU3 continued. The unselected clip contains 61 achieved goals, zero drops and three 20-second timeout episodes (25, 20, 16 goals). These are one video trajectory, not population training or evaluation statistics. The complete recording log has no collision-capacity, solver-convergence, nonfinite or traceback diagnostics; the renderer used its supported CUDA/GL copy fallback. Both source and fixed-close-up output decode as 1200 frames, 60 seconds, 20 fps, 1280x960, with no temporal cuts or speed change. Evidence: outputs/paired_policy_video_20261027/supermimic/{manifest,summary,video_validation}.json; output revo3_supermimic_51000_60s.mp4 SHA256 195215e3c7862e97d0c3de6bbe7d9979e696b80f1dee5081c842e0762d9288bc.


## 2026-10-28 SuperMimic checkpoint 71600 video

Recorded one continuous 60-second seed-42 actor-only rollout on idle physical GPU4 from frozen iteration 71600 (checkpoint SHA256 bfde681c59139462897604c044dbb0a36373f17857d6b0dcfb916665fba0ac61). The official SimulationRuntime/NewtonGL recorder retained the saved 20 Hz environment contract, collision geometry, domain randomization and observation noise; the single-world operational contact capacity remains nconmax=2048. The strict checkpoint, environment, context schema and geometry checks passed. The clip contains 67 achieved goals, one drop and two timeouts; completed episodes have 15, 9 and 29 goals, with 14 goals in the unfinished final episode. These counts describe this one continuous video, not population training statistics. No collision-capacity, nonfinite, fatal or solver-failure diagnostic occurred; the renderer used its supported CUDA/GL copy fallback. Both native and fixed-close-up video decode to 1200 frames, 60 seconds, 20 fps and 1280x960, without temporal cuts or speed changes. Final video SHA256: 35c4cf961664c7ce659d348d576e417534192a381ed9fbc04a9304ec14b81dec. Evidence: outputs/supermimic_video_20261028/rollout/{manifest,summary,video_validation}.json and events.jsonl. Recording and finishing exited successfully and released GPU4; both training ranks continued on GPU1/GPU3, verified at iteration 71733.


## 2026-10-28 14:42 SuperMimic checkpoint 88800 video

Recorded a continuous 60-second seed-42 actor-only rollout from frozen iteration 88800, checkpoint SHA256 356ff3de5d77102f879a12922b502116926889d170a7888b1a705cc8faffcb2f. Official SimulationRuntime/NewtonGL retained the saved 20 Hz environment, assets, geometry, domain randomization and noisy-observation protocol; strict environment/context/geometry checks passed. The one-world operational nconmax remains 2048. The clip contains 58 achieved goals, 2 drops and 2 timeouts. Completed-episode goals are [3, 21, 11, 18], with 5 goals in the unfinished episode. These are trajectory counts, not population training metrics. The complete log contains no collision-capacity, nonfinite, fatal or solver-failure diagnostic; the supported CUDA/GL copy fallback was used. Recording and finishing exited 0, released physical GPU4, and both training ranks remained alive at iteration 89005. Both source and fixed close-up decode to 1200 frames, 60 seconds, 20 fps, 1280x960, retaining every frame and its timing. Final video SHA256 48010d7825cf6392cd1390cbf5fe9e09e1e749c605c8c4b4a883ac0bed499778. Evidence: outputs/paired_video_20261028_1442/rollout/.

## 2026-10-30 RoundedDie production preflight

The accepted lateral reset center is (0.02, -0.01, 0.565) m. Six targeted CPU
regressions and a three-update 64-environment BF16 diagnostic passed, including
forced timeout reset, reward/EMA/second-action-difference parity and actual
convex import checks. The original 200-vertex hull and inertia were preserved.

Two GPUs each passed 768 control steps across eight contact-stress phases with
2048 environments per GPU. Maximum collision candidates were 2,767,412 versus
capacity 3,145,728; maximum contacts 1,904,463; maximum constraint rows 271
versus njmax 512. No overflow or nonfinite observation/reward occurred.

Three full-batch BF16 joint updates passed on the same 4096 environments:
30 WM, 3 actor and 60 critic updates, equal final model hashes across ranks,
and version-14 checkpoint load verified. Last complete iteration was 2.59 s.
Diagnostics used immediate joint updates only to exercise all optimizers.
Production keeps 500 WM warmup iterations and loads no diagnostic weights.
These bounded checks do not establish convergence or all future contact peaks.

Sphere Smooth-v1 stopped gracefully at iteration 6706 (exit 0). Its final
checkpoint, configuration and Git bundle were preserved separately under
`/home/zhangke/rslsphere65_preserved/smooth_stopped_20261030_iter6706`.
Actual launch state and preflight summaries are in the sibling evidence folder.
No engine, installed dependency, sphere source or original asset was modified.

## 2026-10-30 RoundedDie checkpoint 3300 video

`record_die_target.py` recorded an uninterrupted 60 s / 1200-frame, 20 Hz
actor-only rollout from frozen production checkpoint 3300 (version 14).
The right-hand target uses the original textured die mesh at the live world
goal orientation; its recording-only USD has collisions and rigid-body APIs
removed. The existing camera and 1:1 object scale are retained.

Seed 42, original sensor noise/domain randomization, no exploration action
noise. All three 20 s episodes were retained: goals 3, 2, 3; zero drops and
three timeouts. This single rollout is not a population success-rate estimate.
Live collider vertices matched exactly; inertia error 1.09e-11 kg m²; reward
parity error 9.54e-7. Candidate/contact/constraint peaks were 891/592/99,
with no capacity overflow or nonfinite state/reward. NewtonGL used its
supported CUDA/GL copy fallback. Five sampled frames were visually inspected.
Full decode confirmed 1280x960, 1200 frames, 20 fps, 60 s; local/remote SHA-256:
`07015df52dc0f41970e577dda36a4c0cafe7e5443e74ad4142fd2a77bb001536`.
Remote evidence: `outputs/die_video_20261030_141304/rollout/`.
Production training continued; verified alive at iteration 3511.

## 2026-10-31 RoundedDie checkpoint 7500 video

Using the existing recorder, froze production checkpoint 7500 and recorded a
continuous seed-42 rollout with live target orientation, original noise and
randomization, and no exploration action noise. All 1200 frames were retained:
60 s at 20 fps, 1280x960. Goals 47, drops 2, timeout resets 1; completed episode
goals 12/15/14, unfinished episode goals 6. This is one demonstration, not a
population success rate. Full decode passed and five frames were visually checked.
Live convex vertices matched exactly; inertia error 1.09e-11 kg m²; reward
parity error 9.54e-7. Candidate/contact/constraint peaks 878/596/105, with
no capacity overflow or nonfinite state/reward. Supported CUDA/GL copy fallback.
SHA-256: `6a16acf4f12b6686958c0a3d9b43ec93ef812de68bef5e5c536dc48c5ae07412`.
Remote evidence: `outputs/die_video_20261031_010818/rollout/`.
Main training stayed running and was verified at iteration 7721.


## 2026-10-31 RoundedDie Hold-v1 independent implementation

New task: `Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-Hold-v1`.
New isolated repository `/home/zhangke/viserDex_supermimic_rounded_die_hold_20261031/repository`,
branch `codex/rounded-die-hold`, checkpoint v15. The parent Smooth-v1 remains
in its own directory and continues on its original processes. File protection
checks: 675 original local files and 698 original remote files unchanged.

The command now earns a goal only after 10 consecutive 20 Hz samples with
SO(3) error <0.1 rad and object angular speed <0.5 rad/s. Terminating samples
cannot earn a goal. The success pulse drives reward, count, resampling,
training statistics and the target video recorder. The official manager
lifecycle and pre-reset recorder are preserved; no engine/vendor patches.
Hold state is propagated analytically in real and imagined transitions.
Privileged state is 173 dimensions; Actor stays 10x124; physical prediction
stays 55 dimensions. Version 14 is explicitly rejected by new entrypoints.
EMA, reward coefficients, domain randomization and approved object placement
are unchanged. See `rounded_die_hold_v1.md` for the exact semantics.

Actual validation with the pinned official Newton runtime:

- 72 core/geometry/replay/reward/checkpoint tests passed; two additional tests
  of the real command callback passed (74 total). They cover callback
  idempotence, strict thresholds, partial reset, terminal suppression,
  counter propagation through imagination and isolation of Actor/WM inputs.
- Real Newton 64-environment diagnostic, seed 43: 150 steps plus a forced
  all-environment timeout. After settling, goals were aligned to current
  object poses once to exercise actual hold completions. 32 successful
  holds occurred; partial-hold and all-environment resets passed. This is
  a lifecycle diagnostic without a policy, not a training success rate.
- Real/analytic hold counter error 0; reward error 1.12e-8; EMA error
  1.19e-7; clean observation error 1.49e-7. Live convex geometry: one
  collider per environment, 200 vertices, vertex error 0, inertia error
  1.82e-11 kg m^2. MJWarp candidate/contact/constraint peaks
  53798/35958/66; runtime overflow bits stayed zero.
- Two-card BF16 smoke training: 64 environments/rank, seed 44/45, three
  optimizer iterations, production network widths and 8-step horizons.
  Each rank performed WM=3, Actor=3, Critic=6 optimizer updates.
  Initial and final model/normalizer digests matched across ranks.
  Final digest: `e9774df897372923f579a0b3c6ac08981bf01210444bd5eb72727d5eaefd2c42`.
  Per-step hold counter error 0, reward error <=2.24e-8 during fitting.
  Post-fit 32-step holdout completed on both ranks with finite metrics.
  Both jobs exited normally. Logs contained no contact-capacity overflow
  warnings or CUDA memory errors.
- Saved actual v15 checkpoint loaded with correct task/state/config metadata
  and finite model weights. FP32 Actor TorchScript export and reload passed
  numerical comparison for batch sizes 1 and 7.

Evidence is stored in the experiment's sibling `evidence/` directory;
server run outputs are `outputs/hold_environment_check/` and
`outputs/hold_smoke_2x64/`. Smoke weights are diagnostics, not production
weights. Full 2x2048 Hold-v1 training has NOT been launched or capacity-tested
in this change. The original two-card training is still running; the new
production launcher refuses occupied GPUs. GPU validation temporarily shared
only spare memory on those same user-owned GPUs, then exited.
No trained Hold-v1 performance or smoothness is claimed. The updated video
HUD is implemented but no trained Hold-v1 video has been recorded yet.
The unchanged 8-step imagination horizon is shorter than the 10-step dwell;
partial progress is retained in Replay/Critic and beyond-horizon return uses
Critic bootstrap. Actual learning quality requires a production run.


## 2026-10-31 Hold-v1 production-scale approval and preflight

The user explicitly requested stopping the previous RoundedDie Smooth-v1 and
starting Hold-v1 after detailed validation. Smooth-v1 stopped gracefully at
iteration 8579, exit code 0, both ranks exited. Final v14 checkpoint was loaded
and verified finite, then independently copied with config and source bundle to
`/home/zhangke/rounded_die_preserved/smooth_stopped_20261031_iter8579`.
Checkpoint SHA-256: `2fbb5c12ae1ab0c7b1b18b1893a5ad960d5300a67c1cef9b7ea2040ec44b8584`.
All 698 protected parent source files remained unchanged.

The new config was compared against the normalized parent environment contract:
only the command class/hold settings and success reward function/parameters
changed. Asset paths resolve within the new repository. The prior 74 tests,
actual hold-completion diagnostic, and small dual-GPU test remain applicable;
no training algebra or simulator code was changed for this launch.

Added `scripts/supermimic/check_hold_production.py`, a validation-only wrapper
around the normal train entrypoint. It reads MJWarp counters after each control
step and asserts finite states/rewards without changing physics. Full 2x2048
validation (seeds 46/47) passed: 71 control steps / 290816 transitions, three
optimizer iterations, each rank WM=30, Actor=3, Critic=60 updates. This diagnostic
uses production batch sizes, replay capacity, network widths and microbatches;
only the WM warmup is bypassed to exercise all optimizers immediately.

Both ranks had overflow bits 0. Candidate/contact/constraint peaks were
1802780/1241163/120 (rank 0) and 1802873/1241838/117 (rank 1).
Final model/normalizer digests match:
`dc114d88a5cd2355a31f30a775a9645cb1abcb01187bd392699a14b6e451688d`.
Hold counter error 0 and reward parity error <=9.54e-7. All loss/gradient checks
were finite; both jobs exited normally and released GPU memory. No contact
capacity or CUDA memory error was logged. The first optimizer step included
compilation; subsequent optimizer-only timings were 1.423 and 1.434 s. These
exclude environment collection and do not predict long-run iteration timing.

Production uses `start_hold.py` on physical GPUs 5 and 7, 2048 environments each,
seed 42/43, BF16, fresh initialization, WM warmup 500 and original policy LR ramp.
The diagnostic checkpoint is never loaded into production. Actual launch
status and checkpoint verification are recorded in the experiment's sibling
`evidence/task_status.json`; preflight results in `production_preflight_summary.json`.
This supersedes the earlier full-capacity-testing limitation above. No claim
about trained Hold-v1 performance is made by this short validation.

## 2026-11-01 黄色黑点骰子独立资产替换

独立任务 `Isaac-ViserDex-Repose-Revo3-SuperMimic-YellowDie-WujiOffset-HandPose-Hold-v1`。
使用用户提供的 die_camera_assets_v2_with_gaussians.zip，仅提取 rounded_die 的
USD、mesh、材质、配套 Gaussian 与归属信息。原木纹版本目录和训练未改动。

静态检查通过：collision.obj 字节一致；USD 物理参数和变换一致；策略几何摘要一致；
环境合同除新目录路径外一致。保持约 70 mm、60 g、单个 200 顶点凸包。
原版 32100 轮 Actor 经显式来源记录导出为仅用于预览的 actor_preview.pt；
原始权重单独冻结，预览文件不允许恢复训练。

实际通过官方 SimulationRuntime / Newton MJWarp + NewtonGL 运行 1 环境、seed42、
400 步、20 Hz，得到 20 秒连续回放。完成 11 次 Hold、0 掉落、1 次到时重置。
无非有限状态或接触容量溢出；候选接触/实际接触/约束峰值分别 877/605/105；
实时凸包顶点误差 0，惯量误差 1.09e-11，奖励一致性误差 9.54e-7，
HandPose 误差 4.77e-7，Hold 计数和成功边界分歧均为 0。
这是有限的单环境验证，未宣称完整训练或大规模稳定性测试通过。

目标标记通过两个同姿态的纯视觉子网格显示黄色主体与黑色点数，解决官方 marker
加载器只读取单一 mesh 材质而忽略材质分区的问题。仅在录制脚本中处理，未改引擎。
最终视频来自 validation/yellow_die_preview_v3，前两次目标材质显示不完整的预览未交付。
验证结束时原训练仍在第 32241 轮，原两个 PID 保持；父目录 git diff HEAD 为空。

来源、20 个部署文件 SHA-256 与独立配置见 configs/yellow_die_handpose/asset_version.json；
审计和运行结果见 docs/evidence/yellow_die_20261101/。


## 2026-11-01 黄色骰子 32100 Actor 独立 RGB-D 采集

本节属于 `viserDex_yellow_die_rgbd_capture_20261101/repository` 独立副本。
来源为用户指定的黄色骰子任务与 `record_die_target.py` 的原生 Actor/Adapter。
Actor SHA-256 为 `b9b2f53510c8e9d59ca35bed1b87b7f81a19de11309cc9c6808c29fef38a7108`，
版本 17、第 32100 轮，仅推理，不加载优化器。原始环境配置 SHA-256 为
`5811393478264c40260661421c221e2a9340175230f0ac0d42d4445ad3561d93`。
原始环境合同、策略几何、上下文 schema、黄色 PLY 均强校验；采集配置只迁移资产路径。
保留 home_offset、0.5 动作尺度、0.5 EMA、0.4 秒预热和原感知噪声；
Hold 判定保留角度 0.1 rad、速度 0.5 rad/s、持续 0.5 秒。
录制只添加两台虚拟相机并隐藏目标可视标记；未修改原训练代码或依赖。

通过固定的官方 Isaac Lab / Newton 运行入口在空闲物理 GPU4 单环境采集。
`outputs/yellow_policy32100_rgbd200_v1` 实测 200 帧、10 Hz，时间戳 0–19.9 秒，
398 个 20 Hz 策略步骤，398 步非零 Actor 输出，12 次 Hold 成功，无重置、无接触溢出。
实测关节最大变化 0.9544 rad；奖励一致性误差 <=9.54e-7，EMA 误差 0，
Hold 计数误差 0、成功边界分歧 0；实时凸包顶点误差 0、惯量误差 1.10e-11。
启动至退出 GPU 峰值 872 MiB，退出码 0。首次 gsplat 编译缓存保存在独立目录。

RGB 使用同帧黄色高斯模型及原生 ViserDex DefaultAugmentations 的完整 13 步 SH 链，
只扰动 sh0/shN，位置、尺度、旋转、不透明度不变。聚类索引匹配本黄色 PLY。
每帧同时保存增强 RGB、未增强 RGB、完整手部场景、深度、掩码、可见点与 GT 位姿。
渲染后应用场景 first-hit 遮挡，手指与背景在预测器 RGB 内为黑色；
点云来自同帧可见目标深度，无补全；点采样和 SH 使用独立随机流，不改变策略 RNG。
200 帧各含 1024 个有效点。全部帧通过投影、深度、位姿组合及遮挡检查：
最大 RGB 投影误差 3.06e-5 像素、深度误差 0、位姿组合误差 3.34e-16，
掩码外非黑像素 0，全部帧均产生 SH 颜色变化。

验证脚本 `scripts/pose_estimator/validate_yellow_capture.py` 的实际结果在
`../evidence/yellow_capture_validation.json`。原黄色工程配置、播放、训练、
核心模拟文件、PLY 和 Actor 的哈希均未变化；原训练 PID
1426919、1426920、1451418、1451419 保持运行。没有进行预测器训练。

交付视频通过 `record_rgbd_sample_video.py --saved_rgb` 从已采集帧生成，
不重新模拟、不重新随机化，按 10 Hz 正常速度播放。RGB 固定裁剪放大；
点云轴为光学相机坐标毫米，颜色表示深度；另附完整手部场景视频。
限制：仅为短轨迹采集预览，不是完整训练数据集或大规模稳定性验证；
深度采用与 RGB 共位的虚拟相机，不代表真实深度设备已标定。


### 同 RGB 相机视角的视频补充

`record_rgbd_camera_view.py` 使用已采集的 `points_camera_m` 经 `K_rgb` 透视投影，
左右使用同一固定裁剪框 [109,17,309,217] 和 540x540 显示尺寸，未添加观察旋转。
输出 `outputs/yellow_policy32100_same_camera_v1/rgb_pointcloud_camera_view.mp4`，
200 帧、10 Hz、20 秒。逐帧解码通过；投影与保存 UV 最大误差 3.06e-5 像素。
点以半径 1.5 显示像素绘制，按深度着色，不补全表面。
仅 CPU 读取已有数据生成视频，没有启动仿真、策略或训练，也没有改变原数据。


### 2026-11-01 用户提供的新相机标定

已原样归档用户提供的 extrinsics.json 到
`config/cameras/realsense_261722074750_1280x720_extrinsics_revo3_20261101.json`，
SHA-256 `ca7b8ac9cb4dac0733feec8bed8580d387a0b629b2783b77db7bed944c037e2a`。
独立采集脚本默认使用此文件，显式 --calibration 仍可指定其他版本。
新旧 1280x720 RGB 内参和零畸变参数相同；base/palm 定义相同。
T_base_camera 相机位置差 3.3813 mm、相对旋转 5.9252 度。
全部旋转正交性、行列式、四元数、逆变换及 base/tag/palm 坐标链检查通过，
误差 <=5.56e-16。静态检查报告保存在 ../evidence/new_calibration_20261101_validation.json。
文件自身报告同会话 holdout RMS 为 0.10552 px；此项没有重新实测。
本次只导入配置并检查语法，没有启动仿真或修改训练；此前视频/数据仍使用旧标定。
新文件包含 RGB 相机标定，没有真实深度内参或 depth-to-color 外参；
采集器现有虚拟共位深度假设保持，需要与真实深度设备标定区分。


### 新标定的实际仿真录制

新标定首次同步落在额外子目录，未被实际采集入口使用；本次已修正到正式的
独立采集 config/cameras 目录，并在启动前核对 SHA-256 为 ca7b8ac9…c037e2a。
通过显式 --calibration 使用新文件，生成 `outputs/yellow_new_camera_rgbd200_v1`。
实时相机检查通过：两套分辨率的内参及 base-to-optical 外参与新文件一致。
200 帧、398 策略步、12 次 Hold 成功，无重置、无接触容量溢出。
每帧 1024 有效点，投影误差 <=3.06e-5 px，深度误差 0，位姿组合误差 <=2.23e-16。
原工程受保护文件哈希未变化。本次启动前仍在运行的训练 PID 1426919/1426920
在录制结束后保持；以前记录的 1451418/1451419 在本次开始前已不在进程列表中。

`outputs/yellow_new_camera_video_v1` 包含完整相机场景和同相机视角的 RGB/点云对照，
各 200 帧、20 秒，两段视频均全帧解码通过。两侧裁剪和缩放一致，
完整场景保持原始 640x360 视野（另加标题栏）；不通过旧视频变换冒充新机位。
验证证据在 `../evidence/yellow_new_camera_validation.json`。


### 2026-11-02 恢复原采集夹具

用户指出完整画面缺少支架。定位发现黄色采集移植遗漏原交接文档的 --fixture。
从 22031 原路径读取 calibration_fixture_visual.usda 及其 AprilTag PNG，
独立保存至 assets/calibration_fixture；只把绝对贴图引用改为同目录相对引用。
原文件 SHA-256 为 720c24ecf3203dec182cc906487c5afa2859ccdd113769bd355853622260008e；
部署文件为 150c4279c4c94323e14722c537672a1f60674b7a3394fb6e09f3357d1ebec553。
资产没有物理/碰撞 schema，保持纯视觉。按照原 configure_scene，将夹具以
AssetBaseCfg 放置在机器人初始基座变换，添加独立 semantic class，参与
RGB/深度 first-hit 渲染和遮挡，排除在骰子 RGB/点云之外。
新 --fixture 默认指向独立归档资产；源训练工程和依赖未修改。

实测 outputs/yellow_fixture_rgbd200_v1 使用新标定 ca7b8ac9…c037e2a、32100 Actor，
200 帧、398 步、10 次 Hold 成功，无重置、接触容量溢出或非有限状态。
200 帧均可见支架：RGB 至少 12289 像素、深度至少 21580 像素。
每帧 1024 骰子有效点；RGB 投影误差 <=3.06e-5 px，深度误差 0，
位姿组合误差 <=2.23e-16。原工程受保护哈希未变，1426919/1426920 保持运行。
证据 ../evidence/yellow_fixture_validation.json。完整场景和同视角 RGB/点云视频
在 outputs/yellow_fixture_video_v1，各 200 帧、20 秒，全帧解码通过。
此验证只覆盖当前相机机位与短轨迹，不代表夹具在所有目标姿态下的遮挡覆盖已验证。


### 2026-11-02 八窗口采集演示与原始 Play

新增保存 play_raw_rgb (uint8,360x640x3) 到 pair.npz 和 play_raw.png。
它直接来自相机网格渲染输出，不做 SH 扰动或目标抠图；已有 scene.png 是
增强高斯目标与场景合成图，两者明确区分。采集器读取原始画面不改变策略或物理。

最新实际采集 outputs/yellow_capture_dashboard200_v1：新相机标定、原始视觉支架、
32100 Actor、20 Hz 控制、stride=2 得到 10 Hz 的 200 帧，时间戳 0–19.9 秒。
398 个控制步骤、11 次 Hold 成功、无重置/接触容量溢出。
每帧 1024 可见点，投影误差 <=3.06e-5 px，深度误差 0；GT 变换组合误差
<=2.23e-16；全部 200 帧有 SH 颜色变化，原始训练 PID 1426919/1426920 保持。

record_capture_dashboard.py 仅 CPU 读取这批保存数据，生成1920x1120、20秒、
10fps的八窗口视频和640x416的单独原始play视频，两段均全帧解码通过。
八窗口：1 原始网格play全视野；2 干净可见高斯RGB；3 增强RGB；4 可见掩码；
5 原始场景848x480光轴深度；6 骰子可见深度；7 同RGB视角的采样点云；
8 仿真GT位姿轴与中心/8个语义框角点。GT包含不可见角点，仅作监督标签。
窗口2/3/4/6/7/8使用同一固定ROI缩放；窗口6将虚拟共位深度按内参映射到RGB
网格仅用于显示，不改变存储深度；点云颜色只表达深度。窗口1/5保留全视野。
底栏显示帧号、仿真时间、有效点数、最近控制步误差/速度、累计Hold成功及GT中心。

实际采集流程：原Actor根据原带噪状态观测驱动环境；每两个控制步在同一状态
捕获RGB/深度/语义可见性；渲染原高斯及原生13步SH扰动版本；以场景first-hit
掩码去除手、支架、背景及被遮挡目标区域；由可见目标深度反投影、无放回采样
至最多1024点，不补全隐藏表面；用仿真物体真值变换生成相机坐标位姿及9点标签。
同步保存K、世界/相机/基座变换、关节、动作、时间戳、seed、资产和权重摘要。
增强只改SH，点采样和SH随机流不干扰policy RNG；训练没有启动，也没有将预测器
输出接入策略。这里的mask来自仿真语义真值，深度为虚拟共位相机。

这是演示和数据管线检查，不是完整训练集。正式批量采集应覆盖多seed/完整轨迹、
姿态/遮挡/重置状态，并按轨迹划分训练验证，避免相邻帧泄漏；需另行统计覆盖。
当前尚未执行批量扩展或训练。证据 ../evidence/yellow_capture_dashboard_validation.json，
视频 outputs/yellow_capture_dashboard_video_v1；源训练工程受保护哈希未变化。


### 2026-11-02 点云深度测量噪声

用户要求给点云加噪声。新增 collectors/point_noise.py 与
configs/rgbd_capture/point_noise_v1.json；独立采集默认启用，可 --no_point_noise 关闭。
这是新设计的合成噪声，不是 ViserDex 原生配置，也不是实测 RealSense 误差模型。
对每个有效采样点的光轴深度加独立零均值高斯噪声，sigma=0.0005 m，
截断到 ±0.0015 m。XYZ 同比缩放以保持光线方向和 RGB 投影对应；
不旋转整个点云、不抖动 GT、不新增隐藏面，不添加丢点或离群点。

points_camera_m 为加噪输入；points_camera_clean_m 为干净观测点；
point_depth_noise_m 保存实际深度差。depth_m、mask、GT、RGB不变。
每帧噪声 seed=augmentation_seed+2000000+frame_index；局部 torch.Generator
与 SH、采样、物理和策略随机流隔离，重复seed在同设备可复现。
无效点保持零填充和valid=false。manifest记模型和参数，每帧记录seed及实际统计。

CPU检查覆盖旧样本的200帧/204800点，原文件只读。标准差0.500945 mm，
最大深度差1.500041 mm（float32舍入），最大光线比误差2.99e-8；
输入未变、全局RNG未变、可复现性、零填充和全无效输入检查通过。
物理GPU4有其他任务，guard拒绝启动；随后只使用确认空闲的物理GPU2。
实际新采集 outputs/yellow_point_noise32_v1：32帧、62控制步、1次Hold、
无重置/溢出，1024有效点/帧。噪声实测sigma=0.499980 mm，最大深度差
1.500041 mm，均值-0.004113 mm。RGB投影误差<=3.06e-5 px；
干净点与原始深度误差0，GT变换组合误差<=2.23e-16。
原黄色源文件受保护哈希未变化，原训练PID1426919/1426920保持运行。

八窗口视频已支持噪声标注（窗口7），本次生成3.2秒/32帧预览，完整解码通过。
实际验证见 ../evidence/point_noise_cpu_validation.json 和
../evidence/yellow_point_noise_validation.json；视频 outputs/yellow_point_noise_video_v1。
旧数据和视频没有覆盖，后续采集使用新默认值；未开始训练，未修改训练环境。


## 2026-11-03 真机部署 Actor 32100 的同权重仿真录像

在目标机器独立目录 /home/brainco1/yellow_die_policy_from_capture_20261103/simulation_video_20261103 录制。
直接加载 capture_source/checkpoints/policy/yellow_actor_32100.pt，SHA256 b9b2f53510c8e9d59ca35bed1b87b7f81a19de11309cc9c6808c29fef38a7108。
采集源与当前运行时的 29 个 Actor/仿真源文件、15 个资产文件哈希一致；环境仅迁移资产路径。
录制副本开启官方 /isaaclab/render/offscreen 设置，解决离屏捕获时未导入可视网格的问题；
保留全部原始动力学、噪声、动作、随机目标与 Hold 规则，没有修改引擎源码。
seed 42 连续 60 秒、1200 帧、20 fps，34 次目标、0 掉落、3 次 20 秒正常超时重置。
动态几何审计通过、逐步无接触溢出，完整视频成功解码。视频使用仿真状态观测，不代表真机效果；录制未发送硬件指令。

### 2026-11-03 deployed Actor offline replay audit

Official Newton/MJWarp environment, exact deployed Actor b9b2f53510c8, seed 113, 1600 steps, 4 episodes, 44 goals, no terminated episodes. Original noise/randomization retained. Contact peaks: 918 candidates, 613 contacts, 138 constraints; no overflow. Simulator expected/actual action targets match exactly. GPU identical-history and canonical-sensor continuous deployment replay match exactly. SDK-format BF16 continuous replay has max 13.01 deg target drift; firmware clipping independently reaches 9.17 deg. No hardware actions. Full evidence: /home/brainco1/yellow_die_policy_from_capture_20261103/offline_parity_20261103/REPORT.md. No engine or physics behavior changed.

### 2026-11-03 GPU FP32 deployment repair and firmware-command evaluation

Exact Actor b9b2f53510c8, no weight update. Strict CUDA FP32 evaluation with autocast/TF32 off; original environment contracts and sensor noise preserved. Seeds 113/42/7: 3200 baseline steps, 89 goals, 0 non-timeout terminations. Separate official action plugin clamps final actuator commands to saved SDK ranges while preserving original policy EMA and observation normalization: 3200 steps, 88 goals, 1 object-out-of-reach termination. All six deployment replays pass target error <=1e-4 rad; maxima baseline 4.202127e-6 rad and bounded 1.001358e-5 rad. All contact-overflow checks passed. No engine/asset/randomization changes. Physical joint calibration remains pending. Fresh read-only device range snapshot matches the cached ranges; no hardware target or configuration writes. Evidence: /home/brainco1/yellow_die_policy_from_capture_20261103/numeric_repair_20261103/REPORT.md.

### 2026-11-03 real-policy deployment diagnosis (read-only hardware)

Replayed all 239 recorded steps from four real-policy runs on CUDA FP32: observation, raw Actor action and SDK target conversion matched the logs exactly. This checks the computation path, not the physical observation/actuator contract. A separate official Newton diagnostic retained dynamics randomization and saved firmware command bounds while explicitly disabling injected perception/encoder noise to match deployment inputs. Seed 42, 800 steps/40 s: 26 goals, zero non-timeout terminations, no contact overflow. This ablation is not a change to the saved training/evaluation contract and is not evidence of real-hand task success.

Read-only hardware refresh reports controller 0.0.7 and motor firmware 1.5; mode API returned Position=0 three times. Support/semantics of register 900 for this firmware are not independently established, so no mode switch or firmware update was performed. The deployed policy previously assumed MIT command parameters without validating the actual mode or equivalent physical PD response. Full-policy execution is now interlocked after the failed bounded startup-hold diagnostics; loading/preview and separate controlled diagnostics remain available. The first-Send CUDA acceptance test confirms zero hardware sessions, zero motor writes and zero policy steps when blocked. Physical direction/zero, gain/latency identification, dynamic visual accuracy and real task performance remain unresolved. Evidence: /home/brainco1/yellow_die_policy_from_capture_20261103/root_cause_20261103/diagnosis.json.


## 2026-11-04 手动凸包资产与 Direct6D 策略重训接口

按用户明确要求新增独立 ManualConvex-Direct6D-v1 任务和 v18 checkpoint。
使用 right_20261027/revo3_right.usd（SHA256 f64d49af517dded512593bfc7e132484ae1f4f409f4f9e67a02e4bdf4880e249），
由 prepare_revo3_newton.py --manual_convex_usd 创建引用层；不改 vendor，不重新凸分解。
新资产范围用于位置观测归一化及 tanh 输出的唯一仿射角度解码，取消新任务的
EMA、额外 clamp、home offset 和动作 warmup。旧 v17 定义保留用于历史权重。
Actor 每帧85维：q21、缩放qd21、last-action21、goal6D6、time1、position3、
object6D6、relative6D6；移除 Actor 的基座位置与四次动作历史。速度默认0.1×rad/s。
物体位置参与掉落终止/想象存活回报，默认保留；提供82维无位置消融。
四元数按本机及官方v3.0.0迁移文档的XYZW读取，USD Gf显式重排，6D为前两列逐列展开。
保留原有随机化、奖励和终止逻辑，未更改引擎或求解器参数。

验证：14项新协议单测、72项原训练回归测试通过。Newton/MJWarp 4环境完成
2次WM/Actor和4次Critic优化更新、3次重置；完整保存/恢复后运行420控制步×4环境。
全部144个实际手部凸块、9678个顶点实例逐块对照源资产，最大顶点偏差2.09e-10米。
FK位置最大误差2.46e-7米，6D观测最大误差2.39e-7，实际/想象动作映射误差0。
恢复回放接触峰值328候选/46接触/75约束，无容量溢出或收敛上限警告。
恢复回放0成功、2掉落、3超时；这是未收敛的接口测试，不能当作学习性能结论。
完整权重导出、离线推理输入[1,10,85]与绝对角度解码通过。

证据：/home/brainco/reorientation/training/validation/summary.json；
运行目录 training/runs/manual_convex_smoke_v1、manual_convex_restore_v2。
中间 restore_v1 因新审计对uint64指针使用CUDA索引失败，已改为CPU索引；失败结果不计通过。
USD独立依赖解析器报告OmniPBR.mdl是因为它使用Isaac Sim的MDL搜索路径；
已在本机kit/mdl/core/Base/OmniPBR.mdl确认该内置材质存在，未忽略资产/网格依赖。
未启动正式长训练、未验证1024环境容量、未切换原GUI或向机械手发送指令。
新推理入口ManualConvexPolicy独立于旧GUI命令整形，正式新权重接入时必须沿用直接解码。

## 2026-11-04 单目标到达与持续保持（替代上面的 v18 新训练合同）

按用户明确要求删除新任务 action_l2、hand_pose 和成功奖金，成功不再切换目标或终止。
任务升级 ManualConvex-SingleGoal-v2、checkpoint v19。Actor 删除重复的目标6D，
每帧79维（无位置消融76维）；TIME改为目标剩余秒数/固定10秒，归零后保持0。
默认10秒到达时限、最后1秒开始朝向奖励及物体角速度代价，之后永久继续保持。
关节速度、一阶/二阶动作差分代价全程保留；动作输出仍唯一tanh和仿射解码。
仅仿真采样保留20秒截断，推理没有自动停止；目标只有重置或显式set_goal才改变。
真实回调和想象状态机共用一次成功事件合同，持续保持计数饱和，不循环发奖金。

参考 Rudin 等的 https://arxiv.org/pdf/2209.12827 末端窗口奖励结构。
gamma改为0.999，10秒后折扣0.81865；该值是本任务选择，非论文报告值。
继续保持任务保留Critic bootstrap，不复制论文有限任务的超时价值清零。
模型推演仍8步、lambda保持原设置，长程延迟奖励的学习效果待正式训练验证。
完整公式及对照在 /home/brainco/reorientation/training/奖励函数说明.txt。

验证：22项新合同/单目标测试、5项真实命令回调测试、72项旧逻辑回归通过。
4环境2次WM/Actor与4次Critic优化通过，完整checkpoint恢复后420步×4环境回放通过；
其中878个样本进入任务奖励窗口、798个样本在到达时限之后。奖励对照最大误差
1.17e-10，观测2.39e-7，绝对目标角误差0，无接触容量溢出或收敛上限警告。
保持原手动凸包资产及全部随机化，继续核对144个凸块与9678个顶点实例。
恢复回放0成功、2掉落、3采样截断；未收敛烟测不代表任务性能。
实际v19 Actor导出及CPU推理[1,10,79]通过。未启动正式训练或向机械手发送命令。
证据：training/validation/single_goal_summary.json，运行single_goal_smoke_v2、single_goal_restore_v1。
smoke_v1暴露删除hand_pose后旧校验未判空，修复后v2通过；失败运行不计为通过。

## 2026-11-04 指数朝向奖励（v20）

用户要求将朝向奖励改为指数型。新任务为 ManualConvex-SingleGoal-Exp-v3，
checkpoint v20，Actor观测/输出保持79×10及一次tanh直接解码。
朝向奖励改为10*[0.3*exp(-(theta/(pi/2))^2)+0.7*exp(-(theta/(pi/18))^2)]，
theta为最短SO(3)角（弧度），90度宽核和10度窄核提供不同误差范围的信号。
峰值保持10；目标处梯度为0；窄核单独使用时90度误差的奖励仅6.64e-35，
双尺度在同一误差处为1.104。宽度、混合权重和峰值可通过CLI配置并保存。
默认到达时限、持续保持、gamma=0.999及其余惩罚不变；不更改随机化或求解器。
真实回调和想象推演共用核，标量数值及梯度另用独立math/SO(3)测试校验。

42项新合同/指数梯度测试、5项命令回调测试、72项原训练回归通过。
4环境短训练2次WM/Actor和4次Critic更新、完整恢复后420步×4环境通过；
其中878个任务奖励窗口样本、798个时限后样本，奖励对照误差7.46e-9以内，
目标角映射误差0，实际凸块/资产对照通过，无接触容量溢出或收敛上限警告。
v20实际Actor导出及CPU推理通过。0成功/2掉落/3采样截断是未训练模型的接口检查，
不代表指数奖励已改善任务性能。未开展正式训练或硬件动作。
证据：training/validation/exponential_reward_summary.json；
曲线：training/validation/orientation_reward_curve.png；
复现脚本：training/plot_orientation_reward.py。


## 2026-11-04: single pose command, target position and WM field audit (v21)

User selected current object position plus target position, rather than an error observation.
The manual-convex Actor now receives 82 features per frame (10-frame history), protocol
`revo3_manual_convex_single_pose_rot6d_v3`, checkpoint 21, task
`Isaac-ViserDex-Repose-Revo3-SuperMimic-ManualConvex-SinglePose-Exp-v4`.
Both positions are in hand-base meters. Inference and explicit simulator commands require
a base-frame XYZW orientation and XYZ position; q/-q does not restart a repeated command.
Position-only changes do restart the command. Explicit reissue restarts the arrival clock.
The old orientation marker's -4 cm offset is removed from the new task. Default position
is the nominal object position, not a validated optimal grasp. A CLI target override is
available; goal-bank generation, position-goal generalization and adaptive sampling are not implemented.

A shared tolerant position Gaussian, 2*exp(-(max(d-.01,0)/.05)^2), is added to the
existing two-scale orientation reward within the deadline/hold window. Linear velocity
squared has rate weight -1.0 in that same window. Success additionally requires
position error < .02 m and linear speed < .05 m/s for the existing 10 consecutive steps.
These weights/widths are initial tunable choices, not optimized or hardware-calibrated results.
Removed action_l2/home penalties remain absent; direct tanh plus affine targets are preserved.
Distributed metrics now pool task-window sums/counts before deriving ratios, including
ranks with no eligible samples. All physics assets/randomizers/solver presets remain as configured.

Validation: 58 contract/reward/inference tests and 7 official callback tests passed. Expanded
regressions passed 90 tests initially; one fixture still allocated 172 instead of 173
state fields. Updating that fixture to STATE_DIM resolved all 4 affected event/replay tests,
for 91 unique passing regression tests including the two-process CPU optimizer/metric check.
An initial regression invocation lacked IsaacLab module paths and was rerun in the pinned runtime.
Official Newton smoke: 4 environments, 2 WM/Actor updates, 4 Critic updates, 3 resets.
Full checkpoint restore: 420 steps x 4 environments, 878 task-window and 798 post-deadline
samples; max reward difference 4.47035e-8, position-frame difference 5.96046e-8 m,
observation difference 2.38419e-7, direct-target decode difference 0.
Contact peaks 328 candidates / 46 contacts / 75 constraints; no capacity/nonfinite/solver
convergence diagnostics. Existing standalone OmniPBR.mdl resolver warning remains.
Actor export validates [1,10,82] -> 21. Untrained smoke actor achieved no successful holds.
No formal training, hardware motion, or deployed checkpoint replacement was performed.
Evidence: /home/brainco/reorientation/training/validation/pose_goal_summary.json and
training/runs/pose_goal_{smoke,restore}_v1.

WM field audit (no dynamics-network contract change): s is the 173D container, with a
55D physical core; Dynamics converts that core to 57D rotation features and combines
it with 21 targets, 300 deterministic geometry features and 604 context parameters
(xi), for 982 inputs and 54 physical-delta outputs. Goals/time/action history do not
enter the Dynamics MLP. Full field documentation is training/世界模型输入说明.txt.


## 2026-11-04 — identified hand / compact state (checkpoint v22)

The manual-convex task is now `Isaac-ViserDex-Repose-Revo3-SuperMimic-ManualConvex-SinglePose-Identified-v5`. Frozen PACE `fit_03` was copied into `supermimic/data/fit_03` with source hashes and provenance (parameter SHA256 `5c3ae8f00334d20941bf51e58e0cddba7d7b1efb95437e1061aa518302bc7736`). Its identified effective kp/kd, armature, passive viscous damping, dry joint friction, effort limits, encoder bias and command delay are applied through official Isaac Lab/Newton APIs. This does not replace the manual-convex geometry or vendor assets.

The official native DelayedPDActuator runs at 5 ms with the existing two-substep Newton preset; ten PD steps preserve 20 Hz policy commands. Saved delay 12.983980784281389 ms rounds to three PD steps (15 ms), matching the identification backend. Encoder = geometric angle minus bias; only the simulated coordinate boundary applies that offset. Actor remains one tanh followed by affine target decoding. Effort saturation is part of the identified plant, not an additional Actor output clamp. Thumb CMP +1 degree was a fixed fitting assumption, not independent zero calibration. Free-space effective parameters are not a certificate of mounted/new-asset accuracy.

State is 85D: physical 55, last action 21, persistent goal position/quaternion 7, deadline clock 1, actuator-delay-ready flag 1. Old target/EMA memory, three surplus actions, base position, success count and hold counter are absent from the learned state. The last two remain command diagnostics only. The second action difference cost is removed because it would require another previous action. External set_goal now refreshes cached task state/latest instruction observation without new sensor sampling. Runtime checks verify native empty/full delay buffers and that every pending command equals the previous control command at normal policy boundaries.

Context is 210D: hand body masses 22, independently randomized collision friction coefficients 144, object mass/friction 2, identified-randomized kp/kd 42. Fixed restitution (145) and limits (42) are excluded. Inertia (207 elements) is reconstructed from mass scaling and verified per environment; default tensors and all fixed identified values are retained in checkpoint context_manifest. Selection follows declared randomization, not finite-sample variance. Native group parameters are explicitly reordered by joint names. Both 1- and 4-environment runs produced identical context schemas and fixed/derived manifests. WM input is 610D and Critic input 599D; Actor remains 10x82. Resume rejects old versions and mismatched fixed/context/state contracts.

Validation in the pinned Isaac Lab 3 / Newton 1.6.1 runtime:
- 161 regression tests passed; 38 affected tests passed after the last code edits, including one additional command-cache test (162 unique tests). Native CPU Newton PD/delay/saturation was checked against an independent NumPy oracle using all 21 fitted joints.
- Four-environment smoke: 2 WM/Actor updates, 4 Critic updates, 3 resets including a forced cutoff; fresh 32-step open-loop holdout executed.
- Restored checkpoint: 420 steps x 4 environments, 855 task-window samples and 775 post-deadline samples. Maximum reward difference 5.96e-8; direct target decode error zero; object-position frame error 2.98e-8 m.
- Final one-environment smoke: 1 WM/Actor update, 2 Critic updates, forced reset; compact dimensions and delay-memory audit passed.
- Original 144 collision pieces and 9678 vertex instances remain audited; live hand vertex error about 2.09e-10 m. Recorded smoke contact peaks: candidates 358, contacts 42, constraints 66; restore: 309/40/54. No capacity overflow, nonfinite state or convergence failure. Existing standalone OmniPBR.mdl lookup warning remains.
- v22 Actor export/offline inference passed 210 steps across the deadline; no hardware commands were sent.

Artifacts under `/home/brainco/reorientation/training`: `validation/identified_summary.json`, `runs/identified_smoke_v2`, `runs/identified_restore_v1`, `runs/identified_single_env_v1`, `validation/smoke_actor_v22.pt`. These are interface smoke weights, not a learned manipulation policy: restore had zero successes, two drops and three collection cutoffs. Formal learning and production-scale capacity remain unvalidated.

## 2026-11-04 — fixed palm inertial context exclusion (checkpoint v23)

The user requested removal of fixed-palm mass/inertia. The manual-convex task now uses
`Isaac-ViserDex-Repose-Revo3-SuperMimic-ManualConvex-SinglePose-Identified-v6` and checkpoint 23.
Official `SceneEntityCfg` selects the 21 moving bodies for the existing startup mass
randomizer, excluding the fixed articulation root named `world`. Context extraction checks
the fixed-base property, body-name inventory, resolved randomizer body IDs and unchanged
root mass/inertia. It removes root mass and its inertia reconstruction rows. It preserves
all 144 contact pieces and their independent material randomization, all moving-body mass
scales, object randomization and identified actuator parameters. Vendor USD and engine code
are unchanged. The saved manifest names the excluded body without storing its inertial values.

Context is 209D (21 moving-hand masses + 144 shape friction values + object mass/friction 2
+ kp/kd 42). Only 198 moving-hand/object inertia elements are reconstructed; the root's nine
elements are excluded. WM input is 609D and Critic input 598D. Actor remains 10x82 and state
85D. The delay-ready flag at [84:85] is an observed binary condition, not a prediction target:
reset writes zero, the deterministic transition writes one, and Dynamics still predicts only
54 physical increments. It distinguishes empty from populated native command-delay buffers.

Validation with the pinned runtime:
- 44 affected tests passed, including shuffled root-body ordering, 1/2/4-environment context
  fixtures, context invariance to authored fixed-palm mass/inertia, preserved friction,
  rejection of floating bases or root mass randomization, and policy/command contracts.
- Four-environment smoke: 2 WM/Actor and 4 Critic updates, 3 resets, fresh 32-step holdout.
  All native delay, mass/inertia, live geometry and reward/observation consistency checks passed.
- Full v23 checkpoint restored for 420 steps x 4 environments: 854 task-window samples and
  774 post-deadline samples. Reward parity error at most 4.47e-8; target decode error zero;
  position-frame error at most 5.96e-8 m. The untrained actor recorded 0 successes, 2 drops
  and 3 sampling cutoffs; this is interface validation, not a successful learned policy.
- Live collision peaks: smoke 358 candidates / 41 contacts / 66 constraints; restore
  321 / 40 / 54. No capacity/nonfinite/convergence failures. Existing standalone OmniPBR.mdl
  resolver warning remains. Original geometry audit still covers 144 pieces and 9678 vertices.
- Actual v23 Actor export and CPU inference [1,10,82] -> 21 passed, with exact affine target
  decoding. No formal training or hardware commands. Syntax parsing and git diff whitespace
  checks passed.

Evidence: `/home/brainco/reorientation/training/validation/fixed_palm_summary.json`,
`training/runs/fixed_palm_smoke_v1`, `training/runs/fixed_palm_restore_v1`,
`training/validation/fixed_palm_tests.log`, `training/validation/smoke_actor_v23.pt`.

## 2026-11-04 — fixed hand / randomized object contact friction (checkpoint v24)

At the user's request, the manual-convex task fixes contact friction on all 144 hand
collision shapes and randomizes only object contact friction. Both static/dynamic hand
event ranges are `(0.7, 0.7)`; the existing object range remains `(0.7, 1.5)`. The official
startup material event remains in use. Context verifies each fixed material value against
its declared constant, including single-environment runs, and stores it only as metadata.
This is contact friction, not the identified joint dry-friction torque. Mass, actuator,
sensor and object randomization otherwise remain unchanged.

The nominal hand value 0.7 is not a measured material coefficient. It is chosen at the
object range's lower bound because the pinned Newton/MJWarp contact implementation uses
the elementwise maximum of geom friction at equal priority (`newton/_src/solvers/mujoco/
kernels.py:contact_params`, also documented in MuJoCo's contact-parameter mixing rules).
Thus the hand does not mask the low end of object-friction randomization. If the object
range is lowered or geom priorities change, this choice must be revisited. No solver,
vendor asset, contact geometry or contact priority was modified.

Task ID ends in `ManualConvex-SinglePose-Identified-v7`; checkpoint is 24. Context is 65D:
21 moving-hand masses, object mass 1, object friction 1, and kp/kd 42. WM input is 465D and
Critic input 454D. State 85D and Actor 10x82 are unchanged. The saved policy/reward/horizon
configuration and sensor configuration were compared with v23 and are identical. Reward
window curriculum, changes to artificial sensor delay and long-training gradient assessment
are recommendations only, documented in the training text files, not implemented here.

Validation in the pinned runtime:
- 47 affected tests passed, including fixed hand material verification across all shapes,
  1/4-environment tests, retained object-friction sensitivity and rejection of old checkpoints.
- Four-environment smoke completed 2 WM/Actor and 4 Critic updates, 4 resets and a fresh
  32-step open-loop holdout. Live fixed material assertions and all 144 collision pieces passed.
- Full checkpoint restore completed 420 steps x 4 environments, with 884 task-window and
  804 post-deadline samples. Maximum reward parity error 3.73e-8, target decode error zero,
  position-frame error 8.94e-8 m. Untrained smoke policy: 0 successes, 1 drop, 4 truncations.
- Contact peaks: smoke 388 candidates / 43 contacts / 66 constraints; restore 331 / 41 / 54.
  No capacity, nonfinite or convergence failures; the existing OmniPBR.mdl lookup warning remains.
- Actual v24 Actor export and CPU inference [1,10,82] -> 21 passed with exact affine decoding.
  Syntax and diff whitespace checks passed. No formal long training or hardware commands.

Evidence: `/home/brainco/reorientation/training/validation/fixed_hand_friction_summary.json`,
`training/runs/fixed_hand_friction_smoke_v1`, `training/runs/fixed_hand_friction_restore_v1`,
`training/validation/fixed_hand_friction_tests.log`, `training/validation/smoke_actor_v24.pt`.

## 2026-11-04 — stationary dense pose holding without countdown (checkpoint v25)

The user approved full-time dense rewards and removal of the countdown from policy
observations and privileged state. The manual-convex task now uses orientation and
position kernels from the first control step, plus a positive stability kernel:
`3 * exp(-(theta/(pi/12))^2 - (max(d-.01,0)/.02)^2 - (|omega|/.5)^2 - (|v|/.05)^2)`.
The two-scale orientation kernel is unchanged; position peak is 1 instead of 2.
Global object angular/linear speed penalties and the 9-second reward gate are removed.
Joint-speed and first-action-difference costs remain. The complete reward is multiplied
by control dt. Removed action-magnitude, home-pose, second-difference and success-bonus
terms remain absent. Positive stability is continuous, not dependent on success counters.

Actor is now 10x81, state 84, Critic 453; context remains 65 and WM input remains 465.
There is no target/episode countdown in learned state, Actor or Critic. Dynamics already
excluded the target clock. State's final field `[83:84]` is the deterministic actuator
cache-ready flag. Single goals persist until explicitly replaced. The 10-second marker
and elapsed time remain diagnostic metadata only. The 20-second collection cutoff retains
continuing-value bootstrap; drops terminate. Sensor history remains because it represents
measurement delays, not target time. Checkpoint version is 25 and task ID ends in
`ManualConvex-SinglePose-Dense-Identified-v8`; old v17-v24 direct checkpoints are rejected.

The official pinned launch/preset, manually authored 144 collision pieces, identified
fit_03 parameters and native 15ms command delay, sensor delays, fixed palm treatment,
fixed hand friction 0.7 and randomized object friction 0.7-1.5 were not changed.
Gamma 0.999 and H=8 remain. Legacy 173D tasks retain their separate historical clock.
No vendor or Newton/MJWarp code was modified. Full-time reward favors earlier approach;
it no longer implements a specified arrival-time instruction.

Validation with the pinned runtime:
- 127 contract/core tests plus 56 legacy regression tests passed (183 total), including
  independent reward values/gradient directions, no clock in observation/state, persistent
  goals, partial resets, native identified PD, inference and distributed metric aggregation.
- Four-environment smoke: 2 WM/Actor and 4 Critic updates, 4 resets and a fresh 32-step
  holdout. Live asset geometry, actuator memory, material and reward/observation checks passed.
- Full v25 restore completed 420 steps x 4 environments: all 1680 samples contribute task
  metrics, 804 occur after 10 seconds. Reward parity max 2.98e-8, target decode error zero,
  position-frame error 5.96e-8 m; no goal resampling at the evaluation marker.
  The untrained smoke Actor recorded 0 successes, 1 drop and 4 collection truncations.
- Contact peaks: smoke 388 candidates / 42 contacts / 66 constraints; restore 331 / 42 / 54.
  No capacity, nonfinite or convergence failures. Existing OmniPBR.mdl resolver warning remains.
  Live collision comparison still covers all 144 pieces / 9678 vertices, max error 2.09e-10 m.
- Actual v25 Actor export and 420 CPU inference steps passed: [1,10,81] -> 21, exact affine
  decode, repeated identical goals preserve state. Syntax and diff whitespace checks passed.

These checks validate execution and contracts, not task mastery or trustworthy learned
long-horizon action gradients. No formal training or hardware commands were issued.
Evidence: `/home/brainco/reorientation/training/validation/dense_no_clock_summary.json`,
`training/runs/dense_no_clock_smoke_v1`, `training/runs/dense_no_clock_restore_v1`,
`training/validation/dense_no_clock_tests.log`, `training/validation/dense_no_clock_regression.log`,
`training/validation/dense_no_clock_inference.json`, `training/validation/smoke_actor_v25.pt`.

### v25 independent recheck requested by the user

Added eight intervention/gradient regression cases; the combined affected and legacy
suite passed 191 tests in the pinned runtime. The compact Adapter constructs state with
no episode clock or success-tracker attributes available. Reward callbacks return exactly
the same values for fixed measurements/goals across different elapsed times and success
histories. Changing the diagnostic evaluation marker to 0.05 / 3.37 / 10000 seconds,
collection length to 1 / 20 / 400000 steps, and replay elapsed metadata to NaN leaves rollout
losses, states, lambda-return targets, valid masks and Actor gradients exactly unchanged.
The rollout test uses a nonzero action-to-state WM path and identical sensor random draws;
it tests time independence, not the physical accuracy of a learned model's gradients.
Full-reward finite-difference gradcheck passes at rest, generic pose/speed errors and near pi.

Direct inspection of the actual v25 checkpoint confirms Actor normalization 81 and first
layer [2048,810], state 84, context 65, WM input 465 and Critic input 453. Actual exported
CPU Actor outputs exactly match the full checkpoint's training Actor over 12 successive
inputs. Production code and reward parameters were unchanged. Prior simulator validation
was reviewed, not rerun because no runtime implementation changed. Remaining sensor age
and actuator-cache-ready memory serve delayed measurements/actuation; they are not target
countdowns. Legacy-task clocks remain isolated from the v25 path.

Evidence: `/home/brainco/reorientation/training/validation/dense_no_clock_recheck.json`,
`training/validation/dense_no_clock_recheck_tests.log`,
`tests/test_stationary_task_contract.py`.


## 2026-11-04 — upstream right-hand mechanical limit synchronization

Per the user's new PACE delivery, fast-forwarded the clean BrainCo vendor from
`af707e30e5ac24241d12b16e413b174e320131ba` to upstream dev
`2a98621de70d59d182b8075214cebb96fd5375d3`; the GitHub dev HEAD and local delivery
agree. Regenerated the manual-convex wrapper through the official preparation
entrypoint and rebuilt the D405 assembly. No vendor file or physics engine patch.

Independent old/new comparison covers 752 prims: exactly 40 joint-limit attributes
changed, with all 21 joints matching the delivery table. Every other attribute,
relationship, schema and prim type is identical. The 144 collision instances,
22 rigid bodies, zero frames, inertia, identified parameter bytes and D405 mounting
are preserved. Mechanical thumb CMR now ends at 106 degrees; frozen SDK 105 and
independent training 102 degree bounds remain separate. Prior certificates require
recertification because the asset contract changed.

Validation: 61 affected SDK/training-limit, grasp, policy and identified-actuator
tests passed. Official Newton smoke with 4 environments and a fresh holdout completed
WM/Actor/Critic 1/1/2 updates. Live collision vertex error was 2.09e-10 m; contact peaks
365 candidates / 40 contacts / 60 constraints, without capacity overflow. This is an
execution/contract check, not a policy-performance result. Existing OmniPBR.mdl
material lookup warning remains. Evidence and before snapshots:
`/home/brainco/reorientation/hardware_migration/right_asset_sync_20261104/`.


### New-asset Dexonomy expansion and gravity evidence

Re-refined all 640 saved upstream matcher poses against the updated USD and the
independent training limits. Of 570 geometry/QP passes, 37 passed all physical
trials; 35 remain after candidate-only deduplication. Eight historical states were
freshly recertified, and global deduplication yields 42 states: train 15, validation
27, with two ancestry groups. All 42 passed cold reset, hold and 12 signed velocity
perturbations in four randomized contexts; old certificates were discarded.

Live MJWarp gravity is [0,0,-9.81] m/s^2 with gravity enabled and object compensation
zero. After the first physics steps, new-candidate trials verify actual object
masses against recorded randomized contexts exactly (48.006-71.983 g). Randomization
and thresholds were preserved. All collision audits and contact-capacity checks
passed. Most rejected geometry passes had excessive post-settle penetration (480).
These are finite-context grasp checks, not evidence of trained policy performance.

Three additional upstream matcher frame/Warp/QP tests passed (64 total affected
and adapter checks). A fresh 4-environment training smoke loaded the exact final
bank SHA256 2dab9cee1834569c7edeea61f57abdb1c46a444d504454acfced417c97221613,
completed WM/Actor/Critic 1/1/2 updates and a 32-step holdout with adaptive grasp and
replay sampling enabled. Real replay has 18 time slots / 72 environment transitions.
No engine or vendor patches were made.

Published bank: `/home/brainco/reorientation/training/grasps/bank_dexonomy_20261104_2a98621.json`.
Full candidate, trial, source-snapshot, failure and preview artifacts:
`/home/brainco/reorientation/experiments/dexonomy/expansion_20261104/current_asset/`.


## 2026-11-04 — parallel grasp generation and certification scheduling

User requested speed optimization. Added an ordered spawn-based CPU pool (default
8 workers) to saved-pose refinement and the upstream Dexonomy generator. Workers
own independent query models; optimizer math/tolerances are unchanged. Candidate
snapshots are atomic every 16 rows. A persistent official launcher consumes batch
jobs, closing/recreating each scene with its original seed so startup randomization
is independent and matches the reference. Active object-contact statistics now use
an exact per-world vector reduction. No engine, vendor, solver, controller, limits,
randomization or certificate threshold changes.

Same 640 proposals: local optimization 588.90 -> 84.40 s (6.98x). End-to-end refinement
and certification approximately 638.58 -> measured 349.48 s (1.83x, 45.3% less wall time).
The old end-to-end estimate uses file completion times; the new uses a monotonic clock.
Upstream matcher generation and training smoke are outside this timing. All candidate
records and all 570 candidates' randomized contexts match exactly. All 37 previously
passing candidates still pass. One additional borderline state passes at 2.936 mm
instead of failing at 3.021 mm (3 mm threshold); unmodified baseline code repeated on
the same batch also passes at 2.939 mm. Thus physics is not bitwise reproducible and
this benchmark does not establish identical decisions for every near-boundary state.
The benchmark bank is separate; the published 42-state bank is unchanged.

Twelve contact-reduction/grasp/adapter tests passed, plus a fresh 20-candidate matcher
run, the complete 640-candidate physics benchmark, and a four-candidate single-command
pipeline smoke. Invalid geometry contract handling failed promptly and cleaned its
child processes. All live geometry, gravity, mass-context and capacity audits passed.
Evidence: `/home/brainco/reorientation/experiments/dexonomy/speed_optimization_20261104/`.
Single-command entry: `/home/brainco/reorientation/experiments/dexonomy/pipeline.py`.


## 2026-11-04 — unattended pose collection and gallery

Added a detached local `experiments/dexonomy/collect.py` worker with start/status/
resume/stop commands and a finite 640-proposal default configuration. The workflow
keeps the official Newton launcher and unchanged 14-trial randomized certification.
Candidate and matcher checkpoints persist; certified chunks require a completion
receipt binding candidate/bank/report hashes, runner, seed and context/environment
counts. Interrupted batches are archived and rerun; verified completed batches are
reused. A run lock, PID/start-time/boot-ID ownership checks, bounded retries, stall
and render timeouts, atomic status files, and input/source/asset/runtime fingerprints
prevent duplicate runs and mixed evidence. Closing the chat/terminal does not stop
the detached worker. Reboot recovery uses an explicit resume command, not a service.

Validation: five recovery/ownership/locking tests and nine grasp/contact/runtime
regressions passed. A real four-candidate official Newton run survived intentional
SIGKILL of its certification worker after chunk 1: automatic retry reused all chunk
1 files byte-for-byte, completed chunk 2 and produced four certified states plus
its gallery. A completed resume launched no certification; a two-record partial
optimization checkpoint resumed to four records identical to the uninterrupted
result. Missing receipts and altered bank bytes were rejected. A real fresh
640-proposal run is recorded separately below; injected generation-parent SIGKILL
after 48 durable records recovered automatically and preserved that exact prefix.

Rendered all 42 published certified poses as three-view static collision-mesh cards
and six eight-pose overview pages with a searchable offline HTML gallery. Reviewed
an overview and a multi-view card. The existing published bank is unchanged. Static
rendering is not a dynamics rollout or evidence of trained policy performance.
Evidence/usage: `/home/brainco/reorientation/experiments/dexonomy/unattended_20261104/`.
Gallery: `/home/brainco/reorientation/experiments/dexonomy/renders/certified_42_20261104/`.
Fresh run: `/home/brainco/reorientation/experiments/dexonomy/runs/autonomous_20261104_640/`.


## 2026-11-04 — user-started 1024-environment in-hand training

The user changed the newly launched 128-environment grasp-bank run to 1024
environments with normal in-hand initialization and no pose bank. The initial
128-environment process was stopped and retained as superseded evidence. Started
`training/runs/manual_inhand_1024_20261104` from scratch through the pinned official
Newton launcher, with `grasp_bank=null`, adaptive grasp sampling off, adaptive replay
on, FP32 and microbatch 128. No smoke/check-joint-update flags. The first stage is
1000 iterations with the existing 500-iteration world-model warmup. Each iteration
atomically saves latest.pt; every 100 also saves a numbered checkpoint.

The existing HOME_POSITION/reset configuration is used: nominal object world
position (0.02,-0.01,0.565) m, or (0.055,-0.01,0.10) m in hand-base coordinates.
Existing object/joint reset perturbations, mass/friction/actuator randomization,
gravity [0,0,-9.81], limits and control/reward logic are preserved. Rendered and
inspected the nominal initial collision geometry; this is not a grasp certificate.

Startup verification reached iteration 5, with 50 world-model optimizer steps and
77,824 real transitions. Checkpoint loaded successfully on CPU; model tensors and
reported losses/gradients were finite. Live audit covers 144 collision instances /
9678 vertex instances, max vertex error 2.0863e-10 m. No capacity exception occurred
during the observed steps. Actor/Critic updates begin after the normal WM warmup;
training is ongoing and there is no convergence/performance claim yet. Detached
supervisor status, log, initial-pose audit and startup evidence are under
`/home/brainco/reorientation/training/jobs/manual_inhand_1024_20261104/`.

The independent fresh 640-proposal collection also finished: 570 geometry passes,
46 unique certified states and a complete offline gallery. Its separate bank is
not loaded into this user's no-bank training run.


## 2026-11-04 GPU execution tuning and dense-reward discount update

The user retained 1024 environments, no grasp bank and randomized in-hand resets,
and requested higher useful GPU throughput. The original run was cooperatively
saved at iteration 143. Microbatch benchmarking used real replay, full production
networks/batches, FP32/TF32, and the live Newton environment allocated on the GPU.
WM microbatch 128/512/1024/2048 measured 1.016/0.266/0.146/0.077 seconds per update.
Policy+Critic at 128/128 measured 5.537 seconds; 2048/8192 measured 0.255 seconds.
Only execution partitioning changed; logical batches and optimizer counts did not.

An explicit --resume_microbatch_override permits only the four microbatch fields;
all other configuration, asset, environment and observation contracts remain strict.
The subsequent CUDA Graph benchmark on the same snapshot and real fixed batch
measured WM eager 0.06894 vs captured 0.03254 seconds. Its first loss, gradient norm
and every component metric matched exactly. Capture includes only device loss and
backward, never Adam/clipping/priority sampling. Fixed input storage is refreshed;
normalization buffers remain live; gradient storage is restored after policy zeroing.
A dedicated test covers new samples, changing normalizers, cleared gradients and
multiple optimizer steps. Captures are invalidated on checkpoint restore.
Actor 4096 / Critic 8192 measured 0.1503 seconds, with 13.53 GiB total CUDA residency
in the isolated policy benchmark. Joint training also retains the WM capture pool.
Priority updates now use unique/scatter-reduce instead of 2048 Python assignments;
duplicate-max, generation filtering, float64 epsilon/clamp and max-priority semantics
are preserved. CPU replay distribution and importance correction are unchanged.

After consulting SuperMimic Table A.5 (gamma=0.99, 50 Hz) and official ViserDex
released PPO code (gamma=0.998, dt=1/120, decimation=2), the user explicitly agreed
to gamma=0.99 at this task's 20 Hz. Physical-time-equivalent 20 Hz discounts are
0.975187 and 0.994012 respectively; the selected 0.99 is an experimental task choice,
not a published optimum. It replaces the previous deadline-motivated 0.999 default.
--resume_gamma_override is deliberately restricted to checkpoints whose Actor and
Critic optimizer counts are both zero. The change was applied to iteration 373
(WM 3730, Actor/Critic 0), so no previously trained value function was reused under
a different discount. The fresh manual_convex default and bootstrap test are updated.
Sources: https://arxiv.org/pdf/2610.28480 ;
https://github.com/leggedrobotics/ViserDex/blob/main/source/viserdex/viserdex/tasks/manipulation/inhand/config/allegro_hand/agents/rsl_rl_ppo_cfg.py ;
https://github.com/leggedrobotics/ViserDex/blob/main/source/viserdex/viserdex/tasks/manipulation/inhand/inhand_env_cfg.py .

Final detached run: training/runs/manual_inhand_1024_optimized_20261104, resumed
from iteration 581 with WM/Actor/Critic steps 5810/81/1620. The initial model digest
matches the parent's final digest. Execution uses WM/Actor/Critic microbatches
2048/4096/8192, WM CUDA Graph, and gamma=0.99. Latest checkpoint every 10 iterations,
numbered checkpoint every 100, save on cooperative stop. Replay and simulator are
reinitialized at each resume as before; this is not bitwise continuation. The
TensorBoard endpoint remains http://127.0.0.1:6009 with the earlier history copied
and provenance recorded; gamma and execution settings are now scalar tags.

Validation: 108 SuperMimic/task tests plus 8 adaptive replay/event tests pass (116).
The 640–690 joint-training interval averaged 0.63805 s/iteration, update median
0.48243 s. The original 113–143 WM-only interval averaged 12.48475 s/iteration;
these phases differ, so the comparison is not a matched joint-stage speedup claim.
149 GPU samples over 30 seconds averaged 77.85%, ranged 10–91%, max memory 16253 MiB,
mean power 299.62 W. Collection/checkpoint/host work still prevents continuous 100%
GPU occupancy. Current model/optimizer finite checks and parent digest validation
are under training/current_job/checkpoint_validation.json; timing and raw metrics
are in performance_validation.json. Full benchmark outputs remain in the two
execution_*_benchmark_20261104 run directories. No engine, vendor asset, reward
formula, gravity, randomization, UTD, logical batch, rollout horizon or sensor
contract was changed. Throughput checks do not establish policy convergence.


## 2026-11-04 Continue the optimized run to 10000 total iterations

At the user's request, resumed the completed iteration-1000 checkpoint into
`training/runs/manual_inhand_1024_10000_20261104` with --iterations 10000
(9000 additional iterations). The source remains untouched. Kept 1024 environments,
no grasp bank, normal randomized in-hand resets, gamma=0.99, full FP32/TF32 logical
batches, execution microbatches 2048/4096/8192 and WM CUDA capture. All physics,
asset, reward, observation and randomization contracts are unchanged. The runtime
rebuilds the simulator/replay and collects the normal 64-step replay warmup; the
500-iteration WM-only schedule is not repeated. The detached supervisor records
the 10000 target; current/current_job symlinks point to this continuation.

Verified initial model digest equality with the source's final digest. The new
iteration-1080 checkpoint has finite model/optimizer tensors and optimizer counts
10800/580/11600 (WM/Actor/Critic). Live updates reached 1116 with 11160/616/12320
steps; the recent 20-iteration interval averaged 0.6487 seconds per iteration.
TensorBoard http://127.0.0.1:6009 serves copied 1–1000 history plus the growing
continuation, with scalar queries verified above step 1100. latest.pt is written
every 10 iterations, numbered checkpoints every 100, and on cooperative stop.
Validation is in `training/jobs/manual_inhand_1024_10000_20261104/startup_validation.json`
and `continuation_status.json`. Training is ongoing; 10000 is the requested run
limit, not a convergence assertion.


## 2026-11-04 GPU investigation: repeated solver failure and invalid joint states

The 10000-iteration continuation failed after iteration 1513 (22:07:01 local
supervisor status); latest checkpoint was 1510. No trainer remained on the GPU.
The previous adapter mislabeled every nonzero MJWarp overflow bit as contact
capacity. The pinned OverflowType includes ITERATIONS and LS_ITERATIONS; the
warning printer displays the current iteration cap, not a computed required cap.
All bits remain fatal. Diagnostics now retain decoded flags, world IDs, contact
peaks and configured budgets, plus physical state/action/context on failure.

Read the pinned official migration, asset and solver-tuning guides. Added explicit
--resume_solver_iterations with a narrowly verified increasing-only environment
contract proof. All other physics/reward/asset/observation contracts remain strict;
engine and vendor inputs are untouched. Seven override rejection/acceptance tests
and 23 existing SuperMimic tests pass (30 total). Trial 100→200 changed only the
Newton iteration cap, retaining 100 line-search iterations, tolerance 1e-6,
nconmax640/njmax1024, 2 substeps, the identified 5ms PD and 20Hz policy periods.

Frozen iteration-1510 actor, seed42, 1024 environments and 2000 control steps:
both cap100 and cap200 completed without physics warnings in 124.005/125.044 s.
The original failure was NOT reproduced by this frozen-actor test; native
nondeterminism remained and resulting trajectory metrics were not bit-identical.
Both had zero successful holds; these are short rollout validations only.

Resuming exploratory joint training from1510 with cap200 failed after1581.
The new failure report confirms mask512, ITERATIONS only, world801; no capacity
bits. Peak candidates84809, contacts11780, constraints345. Final substep niter13
does not describe the earlier failed substeps: overflow bits accumulate across
the control interval. Failure-state inspection shows right_index_MCP_joint
already at4.52847rad and289.258rad/s before the failed control step, versus
authored mechanical upper limit1.57080rad. Afterward peak absolute joint angle
is134.164rad and joint speed49250.18rad/s. These finite values are physically
invalid; a finiteness-only training audit cannot establish valid dynamics.
Increasing the iteration cap did NOT fix the underlying instability. No further
production restart has been made; contact/limit/control stability requires
further reproduction and validation before resuming. Checkpoint1580 is preserved
for diagnosis, not certified as a policy trained exclusively on valid physics.

The 60-sample/30-second measurement while the resumed run was alive averaged
81.87% GPU utilization (11–93%), max16055MiB and mean282.91W. Iterations1532–1581
averaged0.70487s, with update0.52333s and collection0.16373s. These are active-run
measurements, not current utilization after failure. Added collection time and
solver budget to TensorBoard; the bridge now exposes job running/failed/complete
health. History through1510 is retained in the new branch, without importing
the original failed branch's1511–1513 tail. Evidence is under
training/jobs/manual_inhand_1024_solver200_10000_20261104/ (failure_analysis.json,
solver_budget_validation.json, gpu_30s.json, startup_validation.json) and
training/analysis_20261104_2216.json. The 10000 target remains unfinished.


## 2026-11-04 Certified grasp WM warmup and numerical subdivision experiment

The approved branch retains 1024 worlds, ordinary in-hand formal resets, gravity,
randomization, gamma .99, H8, observations, assets and rewards. The explicit
`--resume_solver_substeps 4` experiment changes solver subdivisions 2→4 only:
5ms explicit PD, 15ms delay and 20Hz policy remain fixed. Solver iterations remain
200. The runtime annotation distinguishes this numerical experiment from the
identified plant's original 2-substep fit; source/target contracts are checked.
No engine or vendor asset is edited.

Actual compiled MJWarp limits are enabled and match the authored mechanical
limits. New pre-step and pre-reset guards reject nonfinite states and gross
mechanical excursions >5 degrees before replay insertion; velocity alone does
not reject samples. They never clip or silently reset. Failure artifacts retain
full state/action/next_physical/context plus a 40-control-step trace (2 seconds),
solver state and actuator-delay diagnostics. This is evidence capture, not an
exact full-engine replay claim.

A fixed checkpoint1580 policy completed1024×2000 real transitions under4
substeps: no guard/solver failure,180.21s collection, contact peak139constraints.
The policy is still weak:2success events,2065drops,4191timeouts over2,048,000
transitions. This is not a matched learning comparison against checkpoint1510.
Artifacts: training/runs/diagnostic_substeps4_guard_20261104/.

The newest46-state bank was re-certified without changing its reset states or
loosening thresholds:184worlds,4randomized contexts per state, all14 cold-reset,
hold and directional-kick trials.35passed(13train/22validation);11were excluded.
First-physics-step audit verifies randomized mass and actual1.25ms timestep.
Bank/receipt/report: experiments/dexonomy/warmup_substeps4_20261104/.
The bank contract matches the exact intended training environment and physics;
old2-substep certificates are never treated as4-substep evidence.

Temporary `--warmup_grasp_bank` uses only13train grasps, ordinary random goals,
40controlsteps of certified hold followed by piecewise-constant bounded ±.03
normalized action offsets.256real controlsteps and200WM updates were collected;
static poses are not invented transitions. The temporary manager is detached,
replay discarded, and ordinary HOME/in-hand initialization restored before
policy training. Fresh ordinary data and frozen1024-world audits gate Actor
updates:8-step position RMSE≤2cm, orientation geodesic RMSE≤.35rad, valid fraction
≥.9, two consecutive passes. Holdout samples do not update any normalization
statistics before being evaluated. The WM fit uses previous data only.

Five readiness rounds(50WM updates each) passed on the final two windows:
10.12mm/.318rad/99.41% and10.48mm/.253rad/99.71%. CPU tensor comparisons verified
Actor/Critic parameters, normalization and Adam state bitwise unchanged from
parent1580; their update counts stayed1080/21600. WM steps15800→16250. Validation
IDs were unused and formal checkpoint grasp_sampling=None. Mean readiness is
not success or uniform accuracy: individual tail errors still require monitoring.

GPU replay retains sampling/importance/generation/masking semantics;10CUDA
regressions passed. Synthetic production-shape1024×1000 replay benchmark measured
60.89→7.13ms per iteration's replay operations,4.331GiB device storage. It excludes
network and physics work. Metrics aggregation reduces host synchronizations but
preserves every optimizer nonfinite-gradient check.89related CPU tests passed,
4GPU tests skipped in that CPU invocation and executed separately onCUDA.

The first production branch ran1581–1870(290iterations) without physics failure,
then stopped cooperatively to add the approved resume Actor LR ramp. During its
30second trace GPU averaged91.3%(73–100%),peak20384MiB. Last100iterations averaged
.819s/iteration(update.462s,collection.335s), versus the old failed2-substep run's
.705s/iteration. GPU feeding/model updates improved; finer physics and health
checks increase collection cost. Do not claim overall throughput improved.
World-model audit errors rose after full-rate Actor updates, motivating the
separate gradual-recovery schedule rather than relaxing readiness thresholds.
Artifacts: training/jobs/manual_inhand_1024_gpu_warmup4_10000_20261104/.


## 2026-11-05 Current-policy playback entrypoint

Added `training/run.sh play` / `scripts/supermimic/record_manual.py` for current
manual-convex checkpoints, adapting the existing NewtonGL target recorder.
It uses the official launcher, saved environment contract, current task/version,
identified-hand/context checks and explicit unchanged runtime-substep proof.
The Actor is frozen, ordinary in-hand resets and observation noise remain, and
no exploration action noise or grasp reset manager is used. Target dice are
visual-only. Videos are continuous at20fps/real simulation speed; resets/drops
are retained, not selected away. The existing legacy recorder is unchanged.
First actual recording: training/play/checkpoint8400_seed42,1200steps/60seconds,
while the independent1024-world trainer continues. Check summary.json for the
completed results; a single seed is a demonstration, not a success-rate estimate.


## 2026-11-05 Command-motion limitation experiments (training remains stopped)

User requested less exaggerated/jittery motion. Added opt-in playback
`--target_velocity_deg_s` and `--target_filter_tau_s` and isolated TargetShaper.
This bounds encoder target slew, not measured joint velocity. It preserves user
position limits, initializes from measured encoder at reset, and passes shaped
actions through Adapter so noisy observation history and reward reflect applied
commands. No PD/asset/randomization changes, no hardware commands, no training
restart. Command limits, alternating targets, reset and NaN rejection passed CPU
assertions. Three continuous60s Newton trials completed and videos decoded.

Checkpoint8400/seed42 baseline target jump max101.83degrees and actual joint-speed
p99498.61deg/s. Excluding each episode's first2s, baseline speedp99 remains117.79
and maximum649.89deg/s: transients are not the only motion issue.
90deg/s+tau.1 yields max target step4.5deg and measured-speedp9976.62deg/s, but
success events fall3→0 and drops0→1.180deg/s+tau.05:0success/0drops, speedp99190.08.
180deg/s without smoothing:1success/1drop, speedp99195.70. These are single-seed
controller ablations, not generalization estimates. None is promoted to a new
default. Constraint-aware training needs consistent real/imagination/inference
action transforms and actual-motion penalties; old-checkpoint playback smoothing
alone demonstrably loses task performance. Evidence/videos in training/play/,
combined report motion_constraint_analysis_20261105.json.

### 2026-11-05: explicit v26 policy revision (not an engine migration)

The user's requested policy changes are implemented by `revised_manual_config`
(`network_revision=urdf_little_v2`, checkpoint version 26). Fresh training via
`/home/brainco/reorientation/training/run.sh train` selects this revision. Actor
hidden widths are 1024/512/256; Critic widths are 512/256/128. WM/Critic omit the
300 geometry features and delay-ready input (164/152 input dimensions). Geometry
is retained only for independent asset/live-FK auditing and old checkpoints.

Policy order is little/ring/middle/index/thumb, proximal to distal. Actions,
state q/qd, bias, home, bounds, gain context and FK indices follow that order.
`config/revo3_right_limits_reference.urdf` is an immutable limits-only snapshot
of upstream dev 2a98621; the imported collision USD and vendor files are unchanged.
Encoder bounds are URDF geometric bounds minus identified encoder offsets.
Fixed physical input scaling precedes real-data RunningNorm; scaler buffers are
serialized. Physical actuator delay remains 15 ms. No solver change is made.

Validation: a 4-world, 2-iteration smoke run completed WM/Actor/Critic updates
and live geometry/reward/observation audits without contact-capacity overflow.
The original v25 iteration-9300 models strictly load; a separate 4-world,
2-step evaluation also completed with the original explicit 4-substep binding.
An initial legacy-evaluation invocation without that numerical-experiment binding
was correctly rejected; the successful invocation uses `--resume_solver_substeps 4`.
Relevant regression and new contract tests are recorded in
`/home/brainco/reorientation/training/validation_network_revision_v26_tests.log`.
These checks validate implementation, not new-policy task performance. Long
training remains stopped. NP3O-inspired cost constraints remain a documented
proposal pending physical thresholds/budgets; the existing dense reward and
regularizers have not yet been replaced with cost critics.

### 2026-11-05: v27 hardware-order / 25 Hz / hand-center contract

The follow-up explicitly corrects thumb order to MCP/PIP/DIP/CMP/CMR after
little/ring/middle/index. New profile `hardware_center25_v3` (checkpoint 27)
remaps all joint-dependent arrays by name; existing v25/v26 remain loadable.
Policy decimation is 8 at unchanged 5 ms PD ticks (25 Hz). Current two solver
substeps yield 2.5 ms integration; no engine or vendor source is edited.
The 15 ms effective delay remains three PD ticks. Live metadata confirms all
four quantities. A 0.5 s minimum hold is conservatively represented by 13 steps
(0.52 s); episode truncation is 500 control steps.

All learned position inputs use a common hand-center origin (0.055,-0.010,0.100)
in base-frame meters, aligned with the same axes and the nominal initial object
center. Replay/engine/API retain explicit base-frame SI values. Real termination,
imagined survival and play/train drop statistics use the same translated radius.
Joint noise/bias remain enabled, but new runs have no invented 0/40 ms observation
lag. The fitted response delay is not an independently measured encoder latency.

Constrained BPTT separates two nonnegative violation costs from task rewards:
actual sampled joint speed (90 deg/s) and target slew (60 deg/s). These lower
training thresholds follow the user's steering; fit_03 has a preconfigured
2 rad/s command-excitation limit, not an identified hardware maximum speed.
The shared Critic trunk has one reward value and two nonnegative cost outputs.
Cost return scales/budgets are explicit; zero budgets and hinge penalties are
used, not PPO likelihood ratios or a claimed hard safety projection. Actual
control-boundary violation counts, maxima and costs are logged. Random smoke
policies still violate the thresholds; no convergence claim is made.

Validation: 143 tests pass; a 4-world two-update run with WM CUDA Graph completes
WM/Actor/three-value Critic optimization and real physics/observation/FK/reward
parity audits. Actual v25/9300, v26 and v27 state dictionaries strictly load;
v27 offline inference also executes. Logs are under
`/home/brainco/reorientation/training/validation_hardware_center25_v27_final.log`
and `validation_v27_tests.log`. Long training remains stopped. Full current
Chinese contract and source links are in
`/home/brainco/reorientation/training/网络结构与输入输出详细说明.md`.

## 2026-11-05 v27 mechanical-stop and collider forensic replay

Isolated diagnostics and reproducible data are in
`/home/brainco/reorientation/experiments/limit_failure_20261105/README.md`.
Five bounded 1024-world official-launcher GPU runs replayed the saved 13-step
world-605 command sequence from its cold reset, without Actor/WM inference.
All 65 domain-context values matched the failure snapshot exactly. Geometry
audits passed for 144 hand hulls and the object; every final run had zero solver
overflow and exited successfully. No engine, asset, or training configuration
was modified. Diagnostic contact ablations and disabled drop termination are
explicitly recorded and are not learning/production settings.

The reset has substantial palm-object penetration: native CCD 9.895 mm / 10
hulls and libccd 11.488 mm / 13 hulls. Matching-algorithm independent queries
agree to 4.3e-15 m; the distance-algorithm difference is not a transform error.
Final solver hulls match the authored vertices within 2.7e-9 m. Thus the reset
sampler lacks geometry feasibility, while this audit finds no import distortion.

High joint speed persists with all contact forces disabled: MCP 1779 deg/s and
22.66 Nm joint-limit reaction. Force decomposition distinguishes motor torque,
joint-limit reaction and contact reaction. The fitted MCP motor clip of
0.36425 Nm works and does not cap constraint forces. Newton 1.6.1 SolverMuJoCo
does not support joint_velocity_limit; the present DelayedPD pipeline has no
speed-dependent motor clamp, and the cost thresholds are not execution limits.
A separate 60 deg/s command-rate projection reduced this short replay's MCP
substep peak to 133 deg/s without hitting its limit; it is not a measured
hardware rating or proof of actual-speed enforcement.

Contact-sensitive replay did not exactly repeat the original terminal overshoot.
These results identify reset penetration and unbounded command changes as
actionable issues, not a proven engine-code bug or long-run stability fix.

## 2026-11-05 user output limits: 100 rpm and 1.28 Nm

New manual-convex environments now read `config/revo3_joint_output_limits.json`
and set all 21 named joints' actuator/joint effort limits to 1.28 Nm and declared
actuator/joint velocity limits to 10.471975511965976 rad/s (100 rpm, joint side).
The official delayed-PD/native max-effort pipeline remains unchanged. The velocity
fields are declared values: SolverMuJoCo does not enforce joint_velocity_limit,
and this change does not add a torque-speed envelope or modify q/qd directly.

The immutable PACE fit and pinned reference/vendor URDF remain untouched.
`scripts/preprocessing/export_revo3_output_limits_urdf.py` generates
`config/revo3_right_output_limits.urdf` with only the 21 effort/velocity entries
changed. Position limits, bias, joint order, gains, inertia, friction and timing
are preserved. Context/audit metadata records the explicit user override with
its source hash; historical saved configs still retain their original limits.

Validation: 106 tests pass, including native bidirectional effort saturation on
all 21 joints, preserved three-tick delay, and historical config dictionary merge.
`scripts/supermimic/check_output_limits.py` runs through the official launcher and
default newton_mjwarp preset. In 1024 environments it reads back all 21,504 joint
limits and passes four holding steps, live collider audits and zero overflow.
Reports: `/home/brainco/reorientation/training/validation_output_limits_100rpm.json`
and `validation_output_limits_100rpm_tests.log`. No training was restarted;
the known reset penetration and lack of hard actual-speed enforcement remain.


## 2026-11-05 v30 absolute-position policy optimization

The policy still emits 21 absolute encoder position targets at 25 Hz, with the
same full-range affine decoder, 200 Hz identified PD and 400 Hz integration.
No engine, collider, asset, randomization, actuator limit or physics-health
threshold changed. Actor initialization, correlated physical-angle exploration,
logit regularization, uniform real eight-step Critic returns, and a common
physical joint-velocity normalization floor address optimization behavior.
Historical checkpoints preserve their original one-step and normalization
semantics; new settings cannot silently override a resumed checkpoint.

The v29 production run independently failed its existing mechanical-health
check at iteration 7578 (index DIP, world 324); its latest valid saved checkpoint
is 7500. These policy changes do not establish an engine fix or enforce the
100 rpm declaration as a hard speed limit. The fail-fast check remains active.

Actual finite validation and independent evaluation results are recorded in
`docs/versions/v30_20261105/validation.json`. The 428-test regression suite covers
historical contracts, CPU/GPU replay, graph/eager WM updates, TD reset boundaries,
normalization floors and strict checkpoint restoration. The source snapshot and
run evidence are under
`/home/brainco/reorientation/training/diagnostics/absolute_v30_validation_20261105`.
No formal 1024-environment / 10000-iteration training is started by this update,
and the existing training/current and TensorBoard pointers remain on v29.


## 2026-11-05 用户要求直接接触手部出生

新建manual-convex训练在官方关节reset后，以已随机化的真实手形进行只读凸包距离查询，
沿重力方向设置对象出生位置。按用户要求取消独立竖直±1 cm悬空偏移；横向/朝向、关节、
质量和摩擦随机化保持。查询几何与实际MJWarp的144手部凸包和物体凸包逐一核对。
未改引擎、碰撞形状/参数、动力学、限位、健康阈值；没有隐藏settling步或额外支撑力。
保存的旧环境保留原reset函数；新环境序列化支持接触的reset函数。
56项检查通过；64环境2轮预检及500帧视频完成；独立seed44的32000转移中，128次reset查询
间隙均约2 μm，1掉落、63超时、0姿态成功，无容量/机械健康失败。接触初始化不保证静力稳定。
证据见 [支撑重置验证](versions/v30_supported_reset_20261105/README.md)。未启动正式长训练或真机。


## 2026-11-05 v31 对齐修复与撤回旧接触结论

旧支撑出生适配器误把Isaac Lab XYZW重排为WXYZ，现已纠正；先前查询内部的2 μm间隙结论撤回。
新reset在真实写回后使用独立Tensor坐标变换和全凸包距离核验；近切向查询改用候选分离平面保守步进。
未改Newton引擎、真实碰撞、摩擦、限位、求解器容量、物理阈值或控制周期。
WM/Critic恢复已存在但遗漏的delay_ready输入，版本31；Actor输入输出合同保持。
237回归检查通过，4096离线姿态通过，1024环境完整网络联合更新及恢复通过，
seed44的32000步真实评估128次reset通过，3掉落/61超时/0成功。新正式长训练未启动。
旧错误reset配置拒绝恢复，证据保留。详见[对齐审计](versions/v31_alignment_20261105/README.md)。

## 2026-11-05 v31 独立严格复核

修复tanh后探索下的饱和日志来源、多卡成本统计聚合、TD窗口/预采集长度校验，
并为视频轨迹声明reset前后时点和坐标，保存对应状态。没有修改Newton、资产、
控制周期、探索采样、任务目标、随机化或任何物理健康阈值。
174项合并回归通过；64环境2轮完整网络预检正常退出，WM/Actor/Critic=20/0/40，
自动500帧视频正常结束且逐帧解码，跨timeout的可见姿态独立核对误差小于0.00001°。
4096例实际reset等价几何采样无失败；32例独立凸半空间线性规划确认首次接触层。
同时量化了未完成的要求：仅61.8%的最近支撑凸包为掌体，存在指尖/近竖直侧接触，
单接触摩擦锥筛查不能证明静力稳定。不得把几何接触继续解释为稳定躺在掌心。
无新长训练或真机。详见[严格复核](versions/v31_strict_audit_20261105/README.md)。

## 2026-11-06 用户明确选择固定托举初态

按用户澄清，新建50 mm／100 g训练采用固定匹配的HOME、掌内局部位置及XYZW朝向，
关节和对象速度均归零；目标旋转、质量、摩擦和PD参数随机化保持。
这是用户授权的初态任务定义变更，不是取消随机化来掩盖旧随机接触方案的缺陷。
固定方块由3个掌体凸包和2个MCP根部被动承托，无PIP/DIP夹紧；reset不搜索、不settle、
不施加支撑力。实际写回后继续审计144手部凸包和对象凸包、零速度及局部pose。
没有修改Newton、碰撞资产、摩擦、求解器、控制频率、限位或物理健康阈值。

先前固定切片候选因22秒漂移6.4–11.8 mm失败，保留证据；最终basin_2通过1024环境seed42
及64环境seed43/44/45的550步检查，全部零掉落，最大漂移0.327 mm、转角0.832°以内。
1024环境2560次出生/reset实际间隙1.636–1.961 μm；seed45逐位验证局部reset不影响其他环境，
并重新读取确认物理参数不变。全部无容量或机械健康失败。另有独立2048次世界坐标float32
几何回读检查及410次单关节远端动作空间检查。

67回归检查通过；完整网络64环境2轮预检WM/Actor/Critic=20/0/40正常退出，自动500帧视频通过。
新旧保存合同分开，版本仍31，旧随机接触配置不被静默重写；固定初态拒绝grasp-bank覆盖。
有限承托测试不证明已学会旋转，正式长训练及真机未启动。
详见[固定掌内初始化](versions/v31_fixed_palm_20261106/README.md)。

## 2026-11-06 固定初态边界继续复核

固定pose与官方物理参数保持。配置及runtime写入前现在验证：25 Hz固定托举profile、
有限单位XYZW、掌面+X与反重力对齐；允许yaw和平移，拒绝未验证的侧掌/翻掌和旧初始化profile。
直接manual-convex新入口默认当前25 Hz HOME初始化。修复恢复时目标位置参数静默忽略，
补充实际默认目标的导出metadata，并在恢复到录像边界时先补录再预热/更新。

109项相关回归通过，另77项训练回归通过（有重叠）。16环境动作后局部reset探针中，
选中55D物理状态精确还原、两缓冲延迟计数清零、历史刷新；1120真实转移/1008 TD窗口独立核验通过。
fresh64环境2轮及恢复到3轮完整运行通过，实际触发恢复边界的500帧视频；无机械/容量异常。
无正式长训练或真机。见[细节复核](versions/v31_fixed_detail_audit_20261106/README.md)。

## 2026-11-06 当前50 mm固定托举配置：1500轮重跑与学习质量审计

按用户要求，以当前工作源码从头完成1024环境、1500轮诊断训练：50 mm/100 g、fixed_palm_v1，
官方Newton MJWarp启动链、物理随机化、完整网络与batch、CUDA Graph/Replay保持。
WM/Actor/Critic更新15000/1000/30000，正常退出；奖励、观测、reset、几何/FK在线校验通过，
无接触容量溢出或机械健康失败文件。已有OmniPBR材质引用告警保留，不属于容量告警。
第500/1000/1500轮录像各500帧，独立完整解码通过。此为有限诊断训练，未进行10000轮正式长训或真机。

独立64环境640步最终评估22次掉落、0成功，关节速度/目标速率违规约99.49%；初始HOME模型同规格
评估0掉落且两项0违规。训练流本身偶有4次成功事件，不能代替独立最终评估。
两组同物理起点±1°真实分支确认多步关节及物体响应不可靠：8步物体仅13/400合法存活起点可辨，
余弦−.0375、区间跨零；90个真实近静止方向中81个出现明显模型响应。
最终奖励bootstrap/短程奖励参数梯度72–308倍，关闭TF32结论保持。
新增分析指出TD8×target EMA的慢冷启动、有界运行风险在大超速区梯度衰减；均不等于已证明全部长期梯度错误。
10.24秒闭环分支的长期差分未通过重复噪声筛选，不能报告长期准确率。

没有以改变求解器、碰撞体、随机化、阈值或动作协议来隐藏策略问题；本轮只新增诊断与文档，未修改生产算法。
详细网络说明第22节及docs/versions/v31_wm_bootstrap_rootcause_20261106记录新结论、复现方法和后续对照顺序。
源码hash、原始命令、检查点、视频及分支证据在training/diagnostics/wm_bootstrap_rootcause_20261106。

## 2026-11-06 冻结WM特征对照

按官方启动链完成seed50–53、每次128环境的真实分支采集；保持50 mm/100 g、fixed_palm_v1、
随机目标、物理随机化和健康检查。四次运行正常退出，模型hash与优化器计数逐位/逐项不变。
离线baseline/geometry/actuator/both各两种采样顺序、200次WM更新全部完成，Actor/Critic冻结。
代理特征未显示稳定旋转方向收益；测试仅11个可辨旋转起点，不能外推完整接触模型效果。
4项诊断合同测试通过；修复空批次统计后重算实际训练统计逐位不变。
未改变引擎、碰撞资产、生产学习算法；ContactSDF只完成接入设计，尚无其物理验证或策略结果。
详见[特征实验与ContactSDF接入建议](versions/v31_contact_features_20261106/README.md)。


### 2026-11-06：47 g 固定托举与预热长训

新增独立50 mm/47 g资产，原100 g保存协议保留；无Newton/vendor修改。
官方Newton MJWarp链64环境550步固定托举检查通过，零掉落，最大位移.308 mm；
完整64环境60轮训练/录像预检完成，47 g随机质量和真实碰撞体读回通过。
启动1024环境10000轮实验，前1000轮±2°平滑HOME激励；Actor学习后已有掉落和违规，不能称策略验证通过。
ContactSDF为独立WM实验，不替换Newton真值；完整范围与运行状态见
`versions/v31_die47_contactsdf_long_20261106/README.md`及`validation.json`。


### 2026-11-06：用户v32 Critic版本长训

按当前用户修改后的在线线性/想象λ-return Critic启动，未修改引擎、资产、随机化或健康阈值。
官方Newton MJWarp链64环境60轮完整预检通过，v32检查点独立500帧录像通过，随后启动1024环境10000轮。
旧TD/BCE/EMA基线在3089轮日志后机械越限退出，原证据保留，不放宽保护。
运行和预检证据：`/home/brainco/reorientation/training/diagnostics/papercritic_v32_long_20261106_032925`。


## 2026-11-06 FlashSAC 1-step 独立迁移小闭环

保持当前47g/50mm固定掌内任务与Newton官方版本链，新增独立FlashSAC backend，未实例化WM。
64环境实际完成100向量步/6400转移、Q4次/Actor2次更新；第101步2.4689°目标变化在发送前被2.4°阈值拒绝，
未完成1024步预检，未启动长训。两个新seed各64个完整回合零掉落/违规、零成功；500帧独立录像完整解码。
79项相关测试通过。原未提交修改与旧实验保留；未停止现有FoundationPose服务。
实现范围、启动桥接、故障修复、剩余限制和证据见 [FlashSAC首轮实施](versions/flashsac_hist25_20261106/README.md)。

### 2026-11-06 FlashSAC S1显式速率约束

用户明确选择S1后，新增局部tanh到每40ms±2.4°可达区间的可微绝对目标映射，采集/Actor/TD一致，局部坐标熵单独版本化；保留S0评估兼容及独立发送前保护。80项测试通过。64环境×1024步小闭环完成，强制timeout/掉落、终态历史、reset速度审计通过。1024环境×1024步共1,048,576条转移完成并保存final，1,852次Q/926次Actor更新，目标速率违规0，无记录到机械/数值/接触容量保护中断；训练进程保存结束后退出码143，原因未确认，不能标为正常进程退出。final完整性及两个独立评估均通过。

seed166/167最终策略各64首完整回合，合计0/128成功、0掉落，关节速度及目标速率违规0；角误差未一致优于同合同HOME，因此不继续扩大训练。训练中自然掉落和超速仍存在，目标限速不代表物理运动安全。详见 [S1合同与实测记录](versions/flashsac_hist25_s1_20261106/README.md)。本次仅仿真，未修改官方FlashSAC、IsaacLab/Newton/Warp或旧WM训练文件。

### 2026-11-06 S1噪声重复单因素诊断

冻结现有S1策略，seed166/167各64首完整回合，重复1..16步探索掉落57/128，每步重采样22/128，两者均0成功。新增显式探索评估模式与关闭阶段记录，冻结网络/BN/奖励统计及源检查点hash均核对不变。

随后从完全相同初始Actor/Q/target/温度/编码器/RNG出发，仅将actor_noise_zeta_max由16改1，完整训练1,048,576转移：训练terminated事件784→571、实际关节速度违规1.5316%→1.0271%，目标速率违规仍0；独立确定性128回合零成功/掉落/五项违规，角误差未优于HOME。末次TD支持外质量12.95%全部在上边界，未解决学习问题，未扩大预算。新增监控不改变官方更新；8项语义测试（含官方更新等价）通过。训练及全部6次评估退出0、关闭记录完整，历史143未复现且原因仍未确认。详见 [后续诊断与证据](versions/flashsac_s1_diagnostics_20261106/README.md)。

### 2026-11-06 用户授权扩大S1训练至5M

用户明确要求扩大训练后，用repeat1同配置从头运行1024环境×4883步，共5,000,192转移；学习率按完整预算衰减，未同时改奖励、温度、随机化或物理合同。完整完成9,570次Q/4,785次Actor更新，训练成功9次、terminated事件623，目标速率违规0，无硬保护中断，退出0且关闭记录完整。第3000步和final各两个seed独立评估均0/128成功、0掉落，角误差未优于HOME。最终Q上边缘概率33.93%，TD超支持质量13.43%，不宣称任务收敛。9段录像共4500帧解码验证，final检查点hash及冻结评估均通过。详见 [5M扩量实测](versions/flashsac_s1_repeat1_5m_20261106/README.md)。

### 2026-11-06 用户授权S1目标速率120°/s对照

仅将cost_target_rate_limit_rad_s由60°/s提高至120°/s，每40ms目标步幅±4.8°，采集/Actor/TD/成本/发送前保护一致，新检查点版本隔离；实际关节90°/s与物体速度风险阈值保持不变。81项回归通过，64环境小闭环终态/reset审计通过。新seed42从头完成5,000,192转移，退出0，无硬保护中断；训练5次成功、3628次terminated事件、实际关节超速79.7441%、目标速率违规0。相对60°/s训练掉落623次，风险明显增加。

最终seed166/167各64首完整回合均零成功/掉落，平均角误差2.268875/2.110308rad，略优于HOME的2.300059/2.131523，但实际速度违规0.265625%/0.284375%；不宣称显著性或任务收敛。11段录像5500帧及final完整性通过。详见 [120°/s对照](versions/flashsac_s1_rate120_20261106/README.md)。

## 2026-11-06 Revo3 v33 Actor / EMA / contact-WM integration

User-authorized protocol change: named YAML soft limits, alpha=0.51 encoder EMA,
episode uniform +/-1 degree encoder-zero offsets, disabled fixed CMP correction,
and explicitly zero DelayedPD delay. Engine, asset geometry, authored mechanical
limits, randomized material/mass/gains, and physics-health checks are unchanged.
Live actuator inspection confirms the zero-delay Newton configuration omits both
the delay component and its queue; all 21 actuators per world are covered.

The official permanent wrench composer applies bounded, step-held force/torque at
object CoM. Read-only MJWarp active constraints provide pre-reset object/hand
contact labels, OR-reduced across 144 hand convex pieces to 22 fused body channels.
WM outputs 54 physical increments plus 22 contact logits. Failure classification
has a disjoint Adam optimizer and detached predicted-state supervision.

Validation: 88 targeted/legacy regression tests passed; 4 opt-in GPU replay tests
skipped. Real 8-world Newton smoke completed three joint-training iterations and
forced-reset capture: exact real/imagined target parity, observation discrepancy
3.58e-7, WM/failure 3 updates each, Actor 3, Critic 6, no physics-health or capacity
failure. Saved environment/checkpoint restored to iteration4 successfully.
A fresh 1024-world, 47g-die, 1500-iteration run was launched at
`training/runs/die50_47g_dex_reference_v33_20261106`; initial production updates
are running on CUDA replay. No convergence or long-run safety claim is made.
See `docs/revo3_supermimic_v33_zh.md` for interface and experiment details.

## 2026-11-07 v33 neural execution optimization

Enabled WM CUDA Graph execution with the independent failure optimizer: capture
both backward paths, retain separate optimizer updates/clipping and detached
predicted-state failure supervision. Wrench sampling avoids a host-to-device
constant copy during capture while preserving the draw order/distribution.
No physics engine, asset, control contract, reward or randomization change.

Validation: 49 targeted regression tests passed, including consecutive eager/
graph parameter and Adam-state comparisons with changing inputs. Production
checkpoint/fixed-real-batch benchmark: WM median 70.78 ms eager versus 30.23 ms
graph (2.34x), with identical first/final update metrics. Profiling is excluded
from timings. Resume from saved iteration 1377 toward the original 8000-iteration
budget, with no excitation warmup. Simulator/replay reinitialize under the
existing resume contract, so the trajectory is not bitwise continuous.
Observed collection+update median 0.930 s before versus 0.512 s after; this is
not a controlled end-to-end comparison because replay and states differ.
Training through iteration 1813 does not establish task convergence.
See `docs/v33_training_gpu_20261107_zh.md` for curves, limitations and evidence.

## 2026-11-07 user-authorized v33 speed-cost continuation

Continue the completed iteration-8000 model for 4000 additional iterations
(target 12000). Change only target-command speed cost threshold 60 -> 180 deg/s
and actual joint speed cost threshold 90 -> 120 deg/s. Cost weight remains 0.05;
holding conditions, normalization floors, model/optimizer state, randomization,
physics, EMA, and no-excitation schedule remain unchanged. These are soft cost
thresholds, not hardware speed certification or a new target limiter.

Added explicit `--resume_speed_cost_override`, with strict rejection of unrelated
configuration changes and recorded old/new thresholds in run metadata. Regression
validation: 35 tests passed, including exact model/Adam restoration and rejection
without explicit override. Live run config differs from its parent in exactly
the two threshold fields. Startup directory/import errors were corrected before
any training update. Simulator/replay initialization follows the existing resume
contract; old Critic cost estimates will adapt to the changed costs.

Run: `training/runs/die50_47g_v33_speed180_120_resume8000_20261107_023635`.
TensorBoard port 6011 includes `speed180_120` alongside the original runs.
Live continuation verified: first update 8001, reached 8030 with WM/failure
80300, Actor 8030, Critic 160600 updates; network update approximately 0.406 s.

## 2026-11-07 58 mm default for future training

User requested the finished 12000-iteration video and replacement of the die for
future training. Verified the existing final recording against model_12000.pt
SHA256 and decoded all 500 frames (25 Hz, 20 s). It uses the original 50 mm die.

Added `die58` from the existing `assets/dice_58mm_isaaclab` package, with its
estimated 73.362112 g mass and authored inertia. Visual/collision points and
topology are unchanged; preserve authored schemas and add the same Newton
200-vertex convex budget as existing assets. Fresh manual-convex defaults now
select die58; saved environments and checkpoints retain strict asset identity.
No engine, solver, health threshold, domain randomization or HOME changes.

The larger object required an independent fixed reset pose. The first offline
candidate used compact-hand geometry and failed live gap checking (-0.117 mm).
It was rejected. Recomputed against the actual manual-convex training hand:
two palm hulls/four total support contacts, 2 micrometer clearance. Position
and orientation are serialized in the existing init_state fields; no online
pose search or settling. Original 50 mm reset checks remain unchanged.

Validation: 56 tests passed. 16 worlds × 550 control steps: no drops, maximum
HOME drift 0.265 mm/0.344 degrees, partial and natural timeout resets verified.
The final asset also passed an 8-world v33 two-update preflight with forced
reset audit, WM/failure/Actor/Critic updates 2/2/2/4, CUDA Graph enabled.
Evidence: `training/diagnostics/die58_fixed_reset_manual_20261107.json` and
`training/diagnostics/die58_v33_preflight_20261107/validation.json`.
No production 58 mm training was launched.

### 2026-11-07: user-requested 10-degree, single-exponential arrival reward

Fresh quiet-hold runs now use `10 * exp(-(max(theta-pi/18, 0)/(pi/2))**2)`.
The environment remains 10 seconds; saved reward configurations retain their
serialized tolerance and mixture. Validation: 112 targeted tests passed;
8-world, 3-update v33/Newton CUDA Graph smoke completed, with real/imagined
reward maximum discrepancy 2.98e-8 and WM/Actor/Critic/failure updates 3/3/6/3.
Evidence: `training/diagnostics/quiet_hold_reward_20261107/smoke_single_exp_10deg/validation.json`.
This verifies implementation consistency, not converged policy performance.

### 2026-11-07: selected COM ellipsoid replaces the 30 cm termination sphere

User explicitly selected hard reset outside the interactive viewer's final
58 mm COM ellipsoid. Fresh manual-convex hand-center die58 runs now load
`configs/supermimic_v33/position_boundary.yaml`. Real termination, imagined
continuation and drop costs share the same asset-COM transform and surface.
Saved runs retain their serialized boundary; no solver or engine change.

Validation: 72 targeted tests passed. Official Newton MJWarp, 4 worlds × 1
control step (8 physics steps): only the two outside worlds reset, no timeout,
zero reset-position error. Live convex audit, CUDA Graph boundary backward and
saved-environment reconstruction passed. No contact-capacity warnings; zero
optimizer updates and no production training launched. Evidence:
`training/diagnostics/position_boundary_activation_20261107/validation.json`.
See `docs/com_ellipsoid_boundary_20261107_zh.md` for coordinates and limitations;
this is a termination envelope, not a proof of recoverability.

### 2026-11-07: fresh die58 v33 30,000-iteration training launch

User requested a new 30,000-iteration run. Started
`die58_v33_quiet_hold_30000_20261107_043752` with 1,024 environments, seed 42,
10-second episodes, the selected COM ellipsoid, single-exponential quiet-hold
reward, and base-frame COM velocity observations. Wrench randomization remains
enabled (probability 0.02 per control step, component limits 0.1 N / 0.001 Nm).
No checkpoint resume or WM-only warmup; CUDA replay and WM CUDA Graph remain
enabled. Saves every 100 iterations; 500-frame videos every 500 iterations.
No engine, solver, health threshold, HOME or domain-randomization changes.

The preflight exposed an old recorder assertion accepting only black pip
subsets. It now also accepts die58's authored `midnight_green_11501` subset,
retaining complete mesh-face partition checking and original material bindings.
Validation: 51 targeted tests passed; 8-environment, three-iteration preflight
using production optimizer batch sizes completed 30/3/60/30 WM/Actor/Critic/
failure updates, forced reset audit, checkpoint load and a 500-frame video.
Maximum real/imagined reward discrepancy was 2.98e-8. This checks execution,
not policy convergence or completion of the requested 30,000 iterations.

Evidence and frozen sources are in
`training/diagnostics/die58_v33_quiet_hold_30000_20261107_043752/` relative to
`/home/brainco/reorientation`. The relocated generated hand USD references the
same original vendor asset by absolute path; composed-stage equality was
verified before refreshing its snapshot audit fingerprint. Original vendor
assets are unchanged. `status.json` records the supervised process outcome;
the separate production run directory contains metrics and checkpoints.

### 2026-11-07: user-selected HOME preference and slower motion profile

The user selected the existing geometric reset HOME as the preferred grasp.
The opt-in `configs/supermimic_v33/control_home_slow.yaml` adds the per-second
penalty `2 * mean((relu(abs(q-HOME)-15deg)/30deg)^2)` on actual post-physics
joint positions. The same expression is used in real rewards, imagined rewards
and WM task supervision. It is active throughout the episode; the 15-degree
deadband permits manipulation without fixing joints at HOME. Constants for WM
CUDA Graph are resident device tensors. HOME, mechanical limits, assets,
observations, network sizes and reset geometry remain unchanged.

The executed EMA command is limited to 60 degrees/second (2.4 degrees per
40 ms step), including the first post-reset action. Limited commands, rather
than unconstrained EMA values, are retained as controller memory. Simulation,
imagination and offline/deployment inference share this transform. This bounds
command slew only, not physical joint speed. Ordinary clamp derivatives are
used; saturated target coordinates have zero local action derivative, with
no straight-through estimator. The fresh launcher sets the joint speed soft
cost to 90 degrees/second and target-rate soft cost to 60 degrees/second.

Object speed costs now also operate far from the goal: 1 rad/s angular and
0.05 m/s linear limits. Each existing object cost head receives the maximum
of its global excess cost and its original orientation-gated holding cost.
Near-goal stability rewards and hold criteria retain their previous values.
Wrench randomization (0.02, 0.1 N, 0.001 Nm), the COM ellipsoid and 10-second
episodes remain enabled. Reward and cost metadata distinguish this profile;
missing fields preserve historical behavior. No solver or engine changes.

Launch a fresh run through `/home/brainco/reorientation/training/run_home_slow.sh`,
supplying explicit `--output_dir` and `--iterations`. The original control
profile and the 30,000-iteration run's frozen sources/checkpoints are untouched.
This change does not authorize or start another production training run.

Validation evidence is in `training/diagnostics/home_slow_20261107/` under
`/home/brainco/reorientation`: targeted tests cover the posture deadband and
gradient, reordered joint callbacks, first/partial-reset slew, exported policy
memory, global object speed gradients, eager/CUDA Graph equivalence and legacy
contracts. An 8-environment, three-iteration production-batch preflight completed
30/3/60/30 WM/Actor/Critic/failure updates, forced resets, checkpoint loading
and a 500-frame video. Maximum reward parity error was 2.98e-8, with no detected
contact-capacity warnings. This validates implementation, not improved policy
appearance, speed distributions or success rate after convergence.

Forty old-checkpoint inference samples (targets, observations, actions and
analytic rewards) were exactly equal between frozen and updated source. Old
frozen-run playback succeeded through its own launcher. Using the updated
workspace launcher directly on the frozen run is rejected by the existing
path-specific fixed-reset asset identity check; that check was not weakened.

### 2026-11-07: authorized HOME/slow production run

The user subsequently requested 30,000 iterations. Launched fresh run
`die58_v33_home_slow_30000_20261107_173657` with 1,024 environments and seed 42,
using the validated HOME/slow profile above. No checkpoint resume. The prior
120-test validation and three-update/video preflight remain applicable: all
validated implementation hashes matched immediately before freezing the source.
Snapshot relocation again passed composed-USD equality verification; vendor
assets, physics preset, health checks and perturbations were not changed.

Run metrics/checkpoints are under `training/runs/` and the supervised launch,
source manifest and outcome status under `training/diagnostics/`, both followed
by the run name, relative to `/home/brainco/reorientation`. Save interval 100,
latest interval 50, audit interval 200, video interval 500 (500 frames, seed 43).
TensorBoard for this run uses local port 6013. Launch verification is separate
from training completion; the supervisor records the actual final exit state.

### 2026-11-07: authorized fresh 2048-environment / 2000-step replay run

At the user's request, gracefully stopped the preceding HOME/slow run at
iteration 9047, saving its final latest checkpoint and preserving all artifacts.
Started `die58_v33_home_slow_2048x2000_30000_20261107_191239` from scratch for
30000 iterations with seed 42, 2048 environments and 2000 replay slots per
environment (4096000 transitions). The new optional `--buffer_steps` argument
applies only to fresh, non-smoke runs; existing defaults and resume contracts
remain unchanged. Compared with the prior serialized training configuration,
only `buffer_steps` changes (1000 to 2000); environment count changes separately.
Logical training batches, update counts, HOME preference, speed constraints,
10-second episodes, COM ellipsoid, wrench randomization and physics are retained.

The frozen source is copied from the preceding validated snapshot, with only
the CLI change and the mechanical-audit file's relocated USD path changed.
Generated USD bytes and their audited hash remain identical; no vendor or
solver edits. Official pinned runtime and `physics=newton_mjwarp` are retained.

Verified through iteration 818: 8180 WM, 818 Actor, 16360 Critic and 8180 failure
updates, finite logged metrics, and 2109 collected slots per environment,
exercising the 2000-slot replay wrap. Replay allocation is 9.598 GiB. Normal
whole-GPU memory is about 29.7 GiB; sampled peak during the successful 500-frame
iteration-500 video is 30.791 GiB out of 31.843 GiB. No detected OOM or contact
capacity warnings; the existing unresolved vendor OmniPBR material warning
remains. Maximum reward parity error is 1.79e-7. This verifies startup and the
first replay/video cycle, not convergence or completion of all 30000 iterations.

Evidence: `/home/brainco/reorientation/training/diagnostics/` plus the run name,
including `plan.json`, `source_manifest.json`, `configuration_comparison.json`,
`launch_verification.json`, resource samples and supervised `status.json`.
Checkpoints and video are in the corresponding `training/runs/` directory.
TensorBoard uses local port 6014. The training remains running after validation.

### 2026-11-07: authorized target slew 60 to 100 deg/s continuation

The user requested 100 deg/s target slew and checkpoint continuation. Gracefully
stopped the 2048x2000 run at iteration 3560 and retained its final checkpoint.
Run `die58_v33_target100_2048x2000_resume_20261107_195342` resumes those learned
states toward the original total of 30000 iterations. Hard target slew and its
matching soft target-rate cost threshold are both 100 deg/s (4 degrees per
40 ms policy step). Actual joint speed cost stays at 90 deg/s; object costs,
HOME preference, perturbations, episode duration, solver and assets are unchanged.
Environment count remains 2048 and buffer_steps remains 2000; simulator and
replay are reinitialized by the existing continuation workflow.

The runtime uses the unchanged preceding frozen source and its official pinned
launcher. No restore checks were relaxed. An explicitly audited checkpoint
migration creates separate `resume_target100.pt`, `env_target100.yaml` and
`control_target100.yaml` artifacts under this run's diagnostics directory.
Only target slew, target-rate cost and the corresponding profile/hash/contract
metadata change. The original checkpoint is untouched. Every other payload
entry, including all network/normalizer tensors, all four optimizers, RNG state,
iteration, step credit and schedules, was compared exactly before serialization.
The serialized result passed CPU strict restore and exact optimizer/model checks;
the original incompatible configuration is still rejected by that restore path.
An action-target test reached 4.00000125 degrees (float32 rounding).

The first simulation startup correctly rejected an old profile copy in the
context audit manifest, before any training update. All mirrored control profile
and profile-hash records were then synchronized; the failed startup is retained
under `attempt1`. Successful runtime startup passed the unchanged metadata and
asset checks. Its restored model digest exactly matches the parent, and the
first resumed row is iteration 3561 with WM/Actor/Critic/failure counts
35610/3561/71220/35610. Finite updates were verified past iteration 3800.
`migration_validation.json`, `migration_diff.json`, `migrate_checkpoint.py`,
`plan.json` and supervised status record the transformation and its provenance.
TensorBoard remains on port 6014 with `target60` and `target100` runs.

Subsequent launch verification passed iteration 4000, including the automatic
500-frame / 20-second checkpoint video and continued updates afterward. Logged
metrics stayed finite with no detected OOM/contact-capacity errors. Detailed
checks are in this continuation's `launch_verification.json`; the complete
30000-iteration production outcome remains pending.

### 2026-11-07: opt-in scalar reward / Critic conversion, no production switch

Implemented checkpoint v34 behind `reward_costs_enabled`, moving all five former
risk channels into the environment/imagined scalar reward. The four bounded
motion penalties each have weight 1/second; genuine failure deducts 20 once,
replacing that step's running penalties. Timeouts are not failures. These values
convert the current reward bound 0.8, cost coefficient 0.05 and gamma 0.99;
Actor objective/gradient and lambda-target projection equivalence are tested.
Critic has one signed linear output. Independent Actor cost optimization and
WM cost consistency loss are absent in v34; raw physical diagnostics remain.
Legacy mode retains six outputs and its old environment hash/restore contract.

The user explicitly requested code and validation only. The running v33
`die58_v33_target100_2048x2000_resume_20261107_195342` remains on its unchanged
frozen source, supervisor and checkpoint path. The fresh opt-in launcher
`training/run_reward_only.sh` defaults to 2048 environments / 2000 replay slots,
100 deg/s target slew and the existing HOME/quiet holding profile. No production
v34 training or checkpoint conversion has been started.

Validation used pinned Isaac Sim 6.2.0.0 / Isaac Lab 28.0.0 from the official
v3.0.0 workspace, Newton 1.6.1 and Warp 1.17.0, default `physics=newton_mjwarp`.
No physics, vendor inputs, randomization, noise, wrench, collision capacities,
COM boundary, velocity frames or 10-second episode limit were relaxed.
Final focused tests: 82 passed, including CUDA Graph/eager equivalence, reward
units, failure/timeouts, real callback parity, strict hash behavior and export.
An 8-environment smoke with production network widths completed four updates:
WM/Actor/Critic/failure counts 4/4/8/4, four audited timeout resets and maximum
reward error 2.98e-8. Strict official-entrypoint continuation reached iteration 6,
counts 6/6/12/6, maximum reward error 3.36e-8. Both passed live geometry and
contact-capacity checks. Actual v33 iteration-16000 and v34 checkpoints passed
CPU model/optimizer exact restore and environment hash checks; cross-version
restore remained rejected.

Limits: the broader official-entrypoint suite exposed three pre-existing
`test_identified_hand` dummy-environment failures (missing `env.cfg`), reproduced
on unchanged production frozen code (3 failed / 10 passed). A separate full
logical-batch 8-environment diagnostic OOMed while competing with production's
periodic recorder; the production job continued normally. Thus full-batch and
2048x2000 v34 execution/convergence are not validated. The recorder correctly
refused the smoke checkpoint; no guard or metadata was bypassed, and no v34
policy video is claimed. Existing vendor OmniPBR material warnings remain.

Evidence: `/home/brainco/reorientation/training/diagnostics/reward_costs_v34_20261107/`
(`tests_final.log`, `preflight/validation.json`, `preflight_resume/validation.json`,
`restore_validation.json`, `weight_analysis.json`, baseline and resource-failure
logs). See `docs/reward_costs_v34_20261107_zh.md` for formulas and implementation.

### 2026-11-07: requested latest-policy 20-second recording

Snapshotted the current production v33 latest checkpoint at iteration 20050 and
recorded it through the unchanged production frozen official launcher with its
saved environment, seed 43 and no playback target override. The completed
1280x960 H.264 video decoded to exactly 500 frames at 25 FPS / 20 seconds.
Live geometry/contact-capacity checks passed; maximum reward parity error was
2.98e-7. The rollout had zero successes, zero drops and two 10-second timeouts;
this single-seed recording is not an aggregate policy evaluation. Production
continued through iteration 20171 without a training switch. Evidence and video:
`/home/brainco/reorientation/training/recordings/latest_policy_20261107_225912/`
(`recording_plan.json`, `recording_verification.json`, `video/rollout.mp4`).

### 2026-11-08: hardware MPR audit and 20-second simulation trajectory export

Audited all 405 logged transitions from candidate v33 iteration 24257's hardware
run without issuing new motor targets. Original USD FK and 144 convex colliders
were checked against the exact model compiled by the official saved-environment
Newton/MJWarp launcher. All 324 index-middle convex pairs are enabled; no
cross-finger exclusion was found. Native MuJoCo signed distance and independent
Coal queries agree on the severe index-middle target conflict (minimum -19.42
mm in model geometry). Measured-joint reconstruction, stalled joints and opposing
currents support physical obstruction, but do not measure physical penetration
or eliminate zero calibration and object-contact differences. Firmware config
was only read; both MPR protection-current settings were 600 mA.

Recorded 500 normal inference transitions (20 simulated seconds, seed 42) with
unchanged saved randomization, observations, gains, rewards and automatic resets.
Two 10-second timeouts occurred; no other termination or contact-capacity overflow
occurred. Exported final post-EMA/rate-limit targets and actual post-physics,
pre-reset angles in explicit SDK logical order. User-requested hardware playback
metadata is 25 Hz, kp=3.0, kd=0.1; simulation gains were not changed. CSV/NPZ/JSONL
round trips and both 250-frame segments passed. Video fully decoded to 500 frames,
25 FPS, 20 seconds. The preview renders saved collision geometry, not a new
physical rollout. Target reset jump is 52.71 degrees and is explicitly separated
from ordinary motion. Actual-angle and target-angle playback are separate exports.

Full-hand checks of 8206 enabled convex pairs at the sampled poses found target
overlap up to 8.89 mm (middle-ring DIP links); actual-state overlap reached 1.10
mm. Both worst pairs were independently verified with Coal. This is a faithful
diagnostic recording, not a collision-free hardware trajectory or a swept-motion
certificate. No training source, vendored geometry, solver behavior, policy weight
or saved operator control parameter was changed. Hardware playback remains for
the user. Evidence is under candidate
`v33_24257_20261107_234931/diagnostic_sessions/20261108_000419/collision_audit/`
and `trajectory_exports/20261108_20s_25hz_kp3_kd0p1/`.

### 2026-11-07: user-requested stop of the v33 target-100 production run

Sent SIGINT to the verified training process at the user's request. It finished
the current update, saved iteration 24257 to `latest.pt`, wrote `validation.json`,
and exited with code zero. Training, launcher and supervisor processes exited.
Preserved the final checkpoint as `model_24257.pt` in
`training/runs/die58_v33_target100_2048x2000_resume_20261107_195342/` under the
workspace root. CPU load, iteration/version and optimizer-state presence passed;
checkpoint optimizer counts match final validation. Free GPU memory increased
to 30469 MiB (1610 MiB used). The supervisor labels any early exit as failed;
its original record is preserved and status was annotated `stopped_by_user`.
Evidence: the run's diagnostic directory contains `stop_request.json`,
`supervisor_status_at_stop.json` and `stop_verification.json`. No automatic
restart, deployment-policy replacement or hardware action was performed.

### 2026-11-07: 58 mm v33 hardware deployment candidate, no motor execution

The user confirmed a 58 mm physical die. Snapshotted production's latest v33
checkpoint at iteration 21100 (23:11:07 Asia/Shanghai), exported its Actor, and
copied the production frozen source into
`/home/brainco/reorientation/deployment/candidates/latest_v33_20261107_231107/`.
The independent entrypoint is
`/home/brainco/reorientation/launch_latest_policy_gui.sh`; the old deployment
configuration and 70 mm perception asset remain byte-identical. Production
continued through observed iteration 21989. No reward-only v34 switch occurred.

The adapter implements 75 x 3 observations at 25 Hz, explicit internal/transport
joint permutations, the existing HOME, EMA alpha 0.51, 100 deg/s command slew,
and applied-command feedback into EMA memory. Raw prior Actor actions stay raw.
The trained fixed goal position, XYZW base-frame poses, and COM ellipsoid are
preserved. Real measurements receive neither randomized encoder bias nor
simulated noise. Runtime checks verify firmware range containment and reject
out-of-contract targets before writes. A read-only diagnostic may approach the
training range from an initially unprepared hand without sending commands.

Exported the exact training USD's authored-frame visual mesh in meters, without
additional recentering or scaling. PLY bounds match 58 mm within float32 export
precision. Live pose ingestion requires the matching mesh SHA-256, preventing
accidental use of the old 70 mm stream. Existing D405 extrinsics are retained.

Validation: 46 focused adapter/control tests and 29 subtests passed; 7 real-Tk
GUI tests passed with device/process calls mocked. CPU and CUDA each passed
500-step full-checkpoint/independent-formula parity, including resets and
external command shaping. Maximum target error was 5.36e-7 rad (0.000031 deg).
All 163 copied source Python files matched the production frozen source hashes.
Read-only UT2 identification, 21-joint firmware range mapping, and the existing
software-stop state were verified. SDK command and servo-command counters
remained zero. D405 imports, pinned SAM2 source/weight checks, and 15 actual
RGB-D frames passed; maximum RGB-depth timestamp skew was 0.305 ms. Camera
stream closed after the check. Shared device/GUI locks preserve mutual exclusion
with the old console.

Limits: full FoundationPose inference and the live vision-policy shadow loop
were not run while production occupied about 30/32 GB VRAM. No active motor
trial, physical success-rate evaluation, or fresh camera/joint calibration was
performed. The package retains existing operator interlocks and non-rate stop
thresholds; passing offline tests does not establish physical readiness. It
uses shared local runtimes, SDK, FoundationPose/SAM2 dependencies and calibration
assets, and is not a standalone installer. Physics, solver, collision capacities,
training observations, rewards and randomization were unchanged. Evidence is
in the candidate's `README.md`, `verification.json`, `package_manifest.json`,
`parity_*.json`, `hand_readback.json`, `perception_preflight.json` and test logs.

Follow-up at 23:28 Asia/Shanghai: the user reported no camera image. The GUI
previously required a separate vision-start click. Starting it also exposed a
real CUDA OOM in FoundationPose initialization before camera acquisition while
production training and its recorder occupied the GPU. The candidate GUI now
automatically opens RGB-D preview, with GPU tracking loaded only after object
selection. A 4096 MiB free-memory startup margin prevents immediate low-memory
retries; it is conservative and is not a validated peak-memory requirement.
Model-load errors keep the preview alive and appear in the GUI. Camera-live
and pose-ready states are separate, so raw preview cannot enable policy motion.
Stale/zombie vision PID records no longer prevent restart. GUI fixtures now
isolate the live perception directory instead of accessing production state.
Nine GUI tests and two startup checks passed. Actual GUI frame counters advanced
at about 30 FPS and RGB-D images were visually inspected. No robot policy worker
or motion command was started; training continued. Successful FoundationPose
tracking and hardware-policy trials remain unverified. Evidence is the candidate
`camera_fix_verification.json`, `tests_camera_fix.log`, and
`perception/live_20261107_232807/`.

### 2026-11-07 23:35: requested latest-policy 20-second recording

Snapshotted production v33 iteration 23250 and recorded 500 steps through its
unchanged frozen official launcher, saved environment and seed 43. No playback
target override, physics change, training switch or hardware action was made.
Full decode verified 500 frames, 25 FPS, 20 seconds, 1280x960; first and last
frames were visually checked. Live geometry and per-step contact-overflow
checks passed; maximum reward parity error was 1.79e-7. This rollout had zero
successes, zero drops and two 10-second timeouts; it is not an aggregate policy
evaluation. Production continued through observed iteration 23385. Evidence:
`/home/brainco/reorientation/training/recordings/latest_policy_20261107_233507/`
(`recording_plan.json`, `recording_verification.json`, `video/rollout.mp4`).


### 2026-11-08: requested rollback from V34 to V33

Restored the 12 V34-modified runtime files byte-for-byte from the verified
production V33 frozen snapshot (1008 source manifest entries checked). Archived
the V34-only reward modules, config and test outside the active source tree;
retired `training/run_reward_only.sh` with a non-launching error. All pre-rollback
files, unrelated edits and historical V34 validation records are preserved.

The selected fresh-run entrypoint is `/home/brainco/reorientation/training/run_v33.sh`.
It uses the final trained V33 target100 profile, 2048 environments and 2000 replay
time slots. The 58 mm object, HOME reward, 10 s timeout, 25 Hz control, EMA 0.51,
100 deg/s target limit, 90 deg/s joint-speed cost threshold, object-motion costs,
external-wrench randomization and physics presets remain as in V33. Actor cost
weight is 0.05, WM cost consistency weight is 1.0, and Critic has reward plus five
separate cost outputs. The historical HOME/60 profile remains available by name.

Validation: 134 relevant existing tests passed; five CUDA tests were explicitly
skipped because validation ran on CPU while the independent hardware deployment
was active. Final production checkpoint `model_24257.pt` strictly restored all
models and four optimizers with six Critic outputs; its saved environment contract
and latest control-profile hash match. A V34 diagnostic checkpoint is rejected.
Wrapper routing was checked with an inert argument-capture leaf for defaults,
smoke mode and explicit overrides; all launcher syntax checks passed. Verified
441 deployment source/config files and the production checkpoint are unchanged.
No new training, simulation or hardware commands were started by this rollback;
there is no new contact-capacity or convergence claim.

Rollback, backup and validation evidence:
`/home/brainco/reorientation/training/diagnostics/revert_v33_20261108_014505/`.
