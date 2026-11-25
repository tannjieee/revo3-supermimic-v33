# FlashSAC 1-step 迁移首轮实施与真实验证

日期：2026-11-06，Asia/Shanghai。用户在迁移分析后要求“开始”。本轮实现独立model-free路径并实际运行；**当前没有FlashSAC训练在后台运行，P2的1024步预算未完成，P3长训未启动**。

## 已实现

- `source/viserdex/viserdex/flashsac/`：固定官方FlashSAC的分类分布双Q、目标参数EMA、温度和均匀1-step replay；不构造WM，不使用想象transition或模型反传。
- Actor仅看带噪81D×10历史，Q看963D历史＋特权观测及21D实际动作；输入采用当前物理缩放、独立官方BN，不加载旧RunningNorm。
- 精确保留pre-reset带噪历史；非终止时replay下一历史与下一次行为历史逐元素相同，reset另生成重复初帧。原 `supermimic/simulation.py` 仅增加 `truncated` 字段及默认关闭的可选终态历史钩子，旧分支保留。
- 五项风险从真实状态计算；训练标量为 `r_env/.4 − .05*sum(C_i)`，失败五头各100；原始env reward和违规指标分别保留。
- S0绝对全范围映射、一次tanh及正确概率密度；非归一化可学习输出gain使初始均值精确为HOME，初始噪声按0.5°标定；其余官方层和更新保留。该HOME输出参数化是明确适配，不宣称原论文逐项复现。
- ±2°平滑HOME预热98向量步；其后每步2次Q更新、每2次Q更新1次Actor/温度。warmup使用的是现有真实采集控制器，不训练WM。
- 发送前拒绝超过60°/s即2.4°/40ms的目标变化，保存现场和完整网络/优化器检查点，不裁动作、不缩机械范围、不放宽保护。
- 每500轮保存检查点后独立进程录500帧；独立评估明确冻结网络/BN/温度/奖励统计并核对源checkpoint hash。
- 新检查点版本 `flashsac_revo3_die47_hist25_s0_v1`；保存网络、优化器/调度器、BN、奖励统计、输入缩放、探索和RNG状态；可选replay。当前只提供训练新启动和检查点评估，**未实现物理状态精确resume或训练resume命令**。
- 实际TD投影越界概率质量、边缘概率、双Q差、α·logπ、奖励缩放分母、关节速度BN P99诊断不额外采样/前向，不改变更新。首轮在两次Actor更新后即停止，未积累长期诊断曲线。

## 启动与环境保持

使用 `scripts/flashsac/run.sh`，调用项目桥接入口 `scripts/flashsac/cli.py train|play --rl_library flashsac`，经固定版官方统一RL dispatcher、task registry和 `launch_simulation` 启动。固定版统一入口没有FlashSAC插件注册接口，因此项目桥接仅在当前进程扩展backend表；没有修改IsaacLab文件或共享安装。不是声称上游原生附带FlashSAC backend。

环境完整复制当前47g运行的 `env.yaml` 和task/noise配置到 `configs/flashsac_die47/`。沿用Newton MJWarp、25Hz策略/200Hz PD/15ms延迟、1.28Nm、固定HOME与掌内reset、速度归零、真实凸包、随机目标及物理/感知随机化。Config中的历史WM字段仅供复用环境/观测合同解析，不会实例化WM或旧Critic。

官方源码仍只读引用 `/home/brainco/reorientation/experiments/flashsac_hand_20261105/vendor/FlashSAC`，SHA `87edc9061150ae9e962dd84e6544e27a1554b3ab`，启动检查工作树干净。只读使用该实验的隔离JAX依赖，未升级共享环境。当前GPU的FoundationPose服务保持原样，没有停止其他进程。

最终版本在每次启动保存source_manifest和源码归档；早期smoke发生在该归档钩子加入前，其原始配置、日志、检查点仍保留，不声称它已有启动时完整源码快照。最终实现快照/清单另见本目录 `implementation_manifest.json`。

## 实测结果

| 项目 | 结果 |
|---|---|
| CPU语义测试 | 7项通过；含HOME/概率、终态历史、发送前拒绝、失败风险、手算分布TD、保存加载和更新等价性 |
| 含旧paper_critic/稳定价值/绝对策略/固定reset回归 | 合计79项通过，包含上述7项，不与其相加；36条旧TorchScript弃用警告 |
| 官方更新数值等价 | 同初始化、同replay及RNG，对比官方 `FlashSACAgent.update`；Actor/Q/target/α全部参数和BN buffer逐元素完全一致 |
| 64环境真实smoke | 100向量步、6,400真实转移、4次Q/2次Actor更新；第101步发送前中止，未完成1024步 |
| 强制timeout | 第3步env0纯truncated，旧目标保存、终态历史与reset历史分离 |
| 受控掉落 | 第5步env0移出工作空间，仅用于接口审计，真实terminated；不计自然策略效果 |
| 拒绝命令 | env2、joint10（中指PIP），最大目标变化2.46890165°，阈值2.4°；命令未送给Newton |
| 失败后checkpoint | `checkpoints/interrupted/`完整保存，`failure_state.pt`保存被拒绝动作和历史；不恢复该run继续训练 |
| 独立seed166 | 64个首完整回合，32,000转移；0成功、0掉落、64timeout、五项违规均0；平均角误差2.30010 rad |
| 独立seed167 | 同样64回合，32,000转移；0成功、0掉落、64timeout、五项违规均0；平均角误差2.13156 rad |
| 视频 | 独立seed166，1环境、500帧、25fps、20秒、960×720；全帧解码通过，零掉落/违规；查看首帧确认手与50mm黄色骰子可见 |

评估命令上限640步，但每槽首回合都在500步timeout，故在第500步结束；不是把640步标为完成。评估result中的 `q_updates=4/actor_updates=2` 是检查点累计训练计数，本次评估新增更新为0。两个seed检查点hash均未变化；后一个评估及最终视频还显式核对温度/奖励统计未变。

这只是初始化附近的闭环验证：两次Actor更新后的均值策略仍接近HOME，0成功不构成收敛结果。受控掉落使smoke有1次掉落与1次早期timeout，是带标签的接口干预，不能用来估计自然掉落率。

## 失败分析与修复记录

第一次启动发现脚本命名 `isaaclab.py` 遮蔽真实包，改为 `cli.py`。第二次统一入口发现共享环境中已注册的 `revo3_walnut_rl` 不在当前搜索路径，只添加其已有本地路径以完成官方任务发现；没有导入其训练配置或切换任务。

第三次启动首次真实跑到100步并拒绝跳变，但故障保存错误调用了仅适用于物理失败的snapshot入口，缺少 `physics_failure` 字段。修复为分别保存“命令尚未下发”和“实际物理故障”，使用新run-id第四次验证，仍在同一步触发同一阈值并成功保存现场。没有为了完成预算降低噪声、改学习率或放宽限速。

第一次独立录像因launcher的 `--viz none` 覆盖了配置而在0步退出；修复为record模式显式选择headless `newton_gl`，第二次完整500帧。所有失败日志和目录保留。

被拒绝关节分解：当前均值目标相对上次实际命令变化+0.49065°，本次样本相对当前均值偏移+1.97825°，合计+2.46890°。前一命令也含探索；这只是代数分解，不是噪声的因果消融。它再次说明0.5°级随机探索也不能保证1024环境长期无尾部跳变。

## 证据路径

- 有效smoke：`/home/brainco/reorientation/training/runs/flashsac_hist25_s0_smoke_20261106_04/`；`result.json`、`boundary_audits.json`、`failure_state.pt`、`checkpoints/interrupted/`。
- 两个独立评估：`/home/brainco/reorientation/training/runs/flashsac_hist25_s0_eval166_20261106/`、`flashsac_hist25_s0_eval167_20261106/`。
- [20秒视频](/home/brainco/reorientation/training/runs/flashsac_hist25_s0_video166_20261106_02/rollout.mp4)。对应目录含result、初帧、源码快照。
- 本目录 `unit_tests.txt`、`regression_tests.txt`、`video_validation.json`、`rejection_analysis.json`、`validation.json`。

## 复现入口

从仓库根运行，output_dir必须是新路径，已有目录拒绝覆盖。

```bash
bash scripts/flashsac/run.sh test
bash scripts/flashsac/run.sh train --output_dir /absolute/new/run \
  --num_envs 64 --steps 1024 --audit_boundaries
bash scripts/flashsac/run.sh play --checkpoint /absolute/run/checkpoints/interrupted \
  --output_dir /absolute/new/eval --num_envs 64 --steps 640 --seed 167
bash scripts/flashsac/run.sh play --checkpoint /absolute/run/checkpoints/interrupted \
  --output_dir /absolute/new/video --num_envs 1 --steps 500 --seed 166 --record
```

`--audit_boundaries`只允许≤64环境接口检查。默认smoke replay100k，正式1M replay需显式 `--capacity 1000000`；本轮没有1024环境1M性能/显存验收，不能把64环境吞吐外推为正式预算。`--save_replay`为可选大文件保存；缺物理/传感器完整状态时依然不能逐位resume。

## 下一步边界

S0保持现有全范围绝对目标合同；保护生效意味着这轮不能继续发送被拒绝命令。已询问是否采用S1：仍下发绝对位置，但将每步可达目标限制在上次目标±2.4°并同步Actor/TD/replay映射。这会改变动作合同，不作为默认修补偷偷加入；本轮尚未实现或启用S1。

等待该选择前不启动1M或长训、不关闭发送前保护。后续还需完成P2全预算、P3正式容量和多种子性能验证，以及训练resume、3-step有效折扣和吞吐优化。没有证明迁移完成了任务学习，也没有据此否定FlashSAC。
