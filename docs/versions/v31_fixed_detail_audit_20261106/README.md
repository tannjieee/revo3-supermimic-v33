# 固定初态的细节复核

本轮修复恢复边界视频漏录、resume显式目标位置静默忽略，并收紧固定托举profile和重力方向校验。
新manual-convex直接入口默认25 Hz HOME初始化；默认目标位置写入训练及Actor-only导出metadata，
与保持不变的网络手中心原点明确区分。固定pose、网络目标和物理参数不变，检查点版本仍31。

| 验证 | 结果 |
|---|---|
| 相关回归 | 109 passed；训练侧另77 passed，有重叠不相加 |
| 局部reset真实探针 | 16环境70步、16次reset，55D状态精确还原，672延迟缓冲槽清零 |
| 真实TD | 1120转移，1008个8步窗口，独立手算最大误差5.96e-8 |
| fresh入口 | 省略revision，64环境2轮，实际25 Hz，WM20/Actor0/Critic40 |
| 恢复 | 第2轮先补录500帧，再预热并更新到3轮，WM30/Actor0/Critic60 |
| 不同目标位置覆盖 | 实际CLI提前拒绝，训练目录未创建 |
| 恢复与导出独立核验 | 边界快照除metadata外逐位一致；导出32组history与12步API动作差异为0 |

恢复录像生成在新运行目录；父运行即使已有视频，也重新记录该恢复边界。
源检查点不修改；边界快照在normalizer更新前保存，并记录当前已验证的环境metadata。
生产间隔500不变，实验间隔2。没有启动正式长训练或真机，也不将这些检查视为WM/策略质量认证。

详见[详细MD第21节](../../revo3_network_v30_zh.md#21-固定初态的细节复核与接口修复2026-11-06)。
证据根目录：`/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/`。

- [训练与真实TD独立审查](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/training/README.md)
- [恢复、录像和目标位置复现](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/lifecycle/README.md)
- [固定profile与索引边界](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/reset/README.md)
- [真实动作后reset探针](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/root/reset_response.json)
- [109项回归日志](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/root/regression_final.log)
- [恢复边界视频](/home/brainco/reorientation/training/diagnostics/fixed_detail_audit_20261106/resume/videos/iteration_000002/rollout.mp4)

来源哈希与完整验证见[validation.json](validation.json)。独立终端日志格式改动继续保留，不混入本次提交。
