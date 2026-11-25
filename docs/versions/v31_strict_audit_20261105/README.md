# v31 第二轮独立严格复核

本轮修复四项可复现问题：tanh饱和日志使用了探索后的动作、TD窗口可超过初始Replay、
多卡成本统计用错归约方式、视频轨迹没有区分终止转移与reset后画面状态。
网络/检查点版本仍31，默认训练目标、探索过程和物理配置不变。
完整解释见[详细MD第19节](../../revo3_network_v30_zh.md#19-v31-第二轮独立严格审计2026-11-05)。

| 项目 | 结果 |
|---|---|
| 合并回归 | 174 passed，含两rank CPU/Gloo与CUDA Graph/eager |
| 实际预检 | 64环境2轮，完整网络/batch，WM20、Actor0、Critic40，正常退出 |
| 自动视频 | 500帧，25 fps，20秒，0掉落1超时，完整解码 |
| 视频reset字段对齐 | 可见姿态世界变换残差最大0.00000864°；reset前后约142.06°差异已分开保存 |
| 正确tanh遥测 | HOME理论0.92296761，训练0.92296767，流审0.92296773；覆盖2048/2048环境步 |
| 几何独立判据 | 32例全144凸包线性规划首次相交一致；4096例实际等价分布0查询失败 |
| 闭环梯度链路 | 固定随机数12方向中央差分均同号，最大绝对差1.57e-6 |

后两项均为有限样本检查；梯度链路正确不证明WM物理响应正确。现有HOME预检也不证明学会旋转。
全部历史失败证据保留，没有启动正式长训练或硬件。

仍未完成的要求：出生只保证接触手部。4096例中61.8%最近支持凸包为掌体、15.6%为DIP，
12.7%被抬高超过20 mm；23.1%不满足μ=.7的单接触竖直合力条件。该静力筛查不是掉落率。
WM响应方向、bootstrap主导和五头共同失败风险仍需在修正后的50 mm/100 g分布上优化和验证。

证据根目录：`/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/`。

- [训练链及饱和复现](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/training/README.md)
- [独立凸几何与支撑筛查](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/geometry/README.md)
- [生命周期/多卡/旧视频复现](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/lifecycle/README.md)
- [预检独立数值核验](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/root/preflight_verification.json)
- [174项测试日志](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/root/regression.log)
- [跨超时重置的视频](/home/brainco/reorientation/training/diagnostics/strict_audit_20261105/preflight/videos/iteration_000002/rollout.mp4)

独立终端日志美化改动原本已在工作区，继续保留；仅配套调整全局违规计数的百分比分母。
主代码改动及验证源码哈希见[validation.json](validation.json)。
