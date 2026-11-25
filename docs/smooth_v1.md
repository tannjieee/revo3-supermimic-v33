# RSLSphere65 Smooth-v1 实现与验证

2026-10-30，独立于已停止的原 SuperMimic 球体任务。
新任务 ID：`Isaac-ViserDex-Repose-Revo3-SuperMimic-RSLSphere65-Smooth-v1`。

## 保护与部署

本地：`/Users/kezhang/Visual-Tactile/experiments/revo3_supermimic_rslsphere65_smooth_20261030/repository`。
22021服务器：`/home/zhangke/viserDex_supermimic_rslsphere65_smooth_20261030/repository`。
源码、环境YAML、运行目录、日志、检查点、编译缓存和guard状态均使用新目录。
运行时仅复用原有锁定环境的解释器，不写共享包或仿真引擎。

原版666个源码文件逐文件SHA-256匹配保护清单。
原版运行目录的latest和独立冻结的19234轮完整checkpoint均匹配：
`81a1d6bfe9b5128489d12b3f648a7b4a6a7ba6bc564953e4a6e70a4503a94db3`。
原版配置文件hash匹配，原guard状态仍为stopped。
可用 `scripts/supermimic/verify_original_unchanged.py` 重复执行只读校验。

## 奖励与梯度

具体权重及公式见主README。新增项在官方RewardManager使用独立ManagerTermBase实现，
保存额外一帧上一动作，compute一次后更新，reset时仅清空指定环境。
官方action_manager已有current和prev_action，二者与Adapter中已有raw action history独立交叉核对。

世界模型的state已经包含4帧原始动作（切片76:97、97:118分别为前一次、前两次动作）。
二阶项直接对这些动作求差，保持沿8步想象轨迹的完整可微梯度；
没有detach当前动作或想象动作历史，也没有扩大Actor的观测。

真实单步审计在pre-reset recorder中捕获二阶项的未加权值，
再独立计算 `sum((current - 2*previous + previous_previous)^2)` 并核对。
同时核对含新权重的总奖励、EMA实际目标、传感器观测和几何FK。
终止步先计算本回合奖励，再清空新回合历史。

## 已完成的初步验证

- 57项CPU测试通过（10.02秒），覆盖奖励数值、梯度、部分环境reset、原模型训练回归及v13检查点恢复。
- 单卡16环境，3轮WM/Actor/Critic联合更新，加32步独立留出轨迹，进程exit0。
- 强制一半环境超时后继续训练；共9次重置，总奖励最大误差3.72529e-9，EMA误差1.19209e-7。
- 实际碰撞体为半径0.0325m的球，碰撞尺寸误差0；FK位置最大误差3.64148e-7m。
- 新版v13的训练、评估、Actor导出与目标姿态视频入口已同步版本检查。

双卡完整更新、接触压力测试及正式启动结果见同目录部署证据与后续实测记录。
初步测试不代表策略已收敛，也不代表动作平滑程度已经通过长期对照评估。

## 保留的原设置

正式任务由start_smooth.py使用两张卡、每卡2048环境从头训练。
保留500轮世界模型预热、1000轮Actor学习率渐增、原网络/批量/UTD、BF16、
20Hz、8步监督/想象、同样的10×124带噪观测、域随机化和终止/目标逻辑。
真实动作仍采用关节位置映射和EMA；本改动没有新增关节速度硬限幅。

`scripts/supermimic/verify_smooth_run.py`会对正式任务的实际config.json与原版逐项比较，
要求已有训练配置仅EMA和一阶动作惩罚权重变化，另外三个新增配置字段必须符合本任务的值。
它还验证双卡进程存活、任务ID/路径、诊断开关关闭、未加载旧权重、观测/噪声/几何及关节顺序一致。

## 双卡实测及正式任务启动

补充CPU双进程Gloo回归通过，总计58项测试通过；新二阶项最大误差的跨卡聚合也已覆盖。

双卡各2048环境完成3轮完整联合更新，每卡优化器计数WM30、Actor3、Critic60。
两卡最终模型摘要均为：
`68a0b410c839a9a92f4829a236b5d9c9cca29e2e29f786c95dd7eff7ef65a50a`。
每卡二阶项独立核对最大误差0；真实总奖励误差最大9.53674e-7，EMA误差1.19209e-7。
两卡共记录2076次重置。首次更新包含139.26秒编译，后两次纯网络更新分别约1.55/1.56秒；
这不是正式训练完整一轮的耗时，正式任务另有采样和500轮WM预热。

双卡每环境768步接触压力测试通过：零动作、全正限位、全负限位、混合限位、
反向混合限位、交替限位、随机动作、再次全正限位，各96步。
每一步检查overflow标志、奖励和所有观测的有限性。
总碰撞候选峰值2939999（折算每环境约1435.55），最大约束行225，
仍使用原容量nconmax1536/njmax512；没有容量溢出，未更改求解器。

正式任务已使用GPU5/GPU7（按UUID选择）、各2048环境从头启动；
代码提交为服务器`585d7afa8e5cb5d45db95b2a353ad30fe960ecaf`。
Guard PID94047，独立运行目录和状态文件见主README。
实际启动信息在`../evidence/launch.json`；验证汇总在`../evidence/validation_summary.json`。
正式任务不复用任何诊断模型，不启用smoke/check_joint_update/forced_reset_audit。
后续进度以服务器实际metrics和guard状态为准。

正式任务核验快照：iteration5，phase=wm_warmup，两卡训练PID94112/94113。
实际config逐项对照通过，噪声/观测/几何和关节顺序一致，诊断开关全部关闭。
启动后再次核验原版666个源码文件、配置和19234轮权重，全部保持一致。
证据见`../evidence/production_verification.json`和`../evidence/original_protection_after.json`。
