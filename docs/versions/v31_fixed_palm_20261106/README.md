# 固定匹配的 HOME 与掌内方块初态

按用户要求，新建50 mm／100 g训练固定手形、方块相对手基座的位置及朝向，关节和对象速度归零。
目标旋转继续随机，质量、摩擦、PD参数的启动随机化保留。初始化不搜索接触、不预推进仿真，
不使用抓取库或额外支撑力。由三个掌体凸包与两个MCP根部被动承托，PIP/DIP保持开放。

| 实际验证 | 结果 |
|---|---|
| 1024环境、seed42、550步 | 零掉落，最大漂移0.321 mm、转角0.831°；2560次出生/reset读回通过 |
| 64环境、seed43/44/45，各550步 | 全部零掉落，最大漂移0.327 mm、转角0.708°以内 |
| 局部reset | 所选环境回到固定零速初态并更换目标；未选环境55维物理状态和目标逐位不变 |
| 物理域参数 | 末尾从真实模型重读，与启动值完全一致 |
| 独立float32几何回读 | 1024世界、两套逆变换，2048次无穿透且均有3个掌体接触 |
| 单关节动作空间 | 10个PIP/DIP分别±10°扫描，共410个合法样本无对象穿透 |
| 回归 | 67 passed |
| 完整入口与周期视频 | 64环境2轮，WM20/Actor0/Critic40；500帧、20秒、零掉落、1超时 |

检验门槛一直为零掉落、22秒内漂移<5 mm／转角<5°。最初候选虽零掉落，漂移约7 mm，已拒绝；
其他失败候选、YAML和报告序列化修复前日志均保留。没有调高阈值或修改solver/摩擦让其通过。

详细位置、XYZW、HOME顺序、恢复边界与命令见[详细MD第20节](../../revo3_network_v30_zh.md#20-固定-home固定掌内初态2026-11-06)。
新reset合同名`fixed_palm_v1`，检查点网络版本仍31。历史正确随机接触v2按原配置恢复，
旧错误reset继续拒绝；固定初态拒绝grasp-bank覆盖和未经验证的70 mm对象。

本轮验证有限HOME承托及初始化链路，不证明学会旋转；WM响应和bootstrap问题仍需继续优化。
正式10000轮训练、真机均未启动。用户启动脚本为`/home/brainco/reorientation/training/start_v31.sh`，
正式视频仍每500轮录制；本轮测试为每2轮触发。

证据目录：`/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/`。

- [1024环境报告](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/runtime/final_1024_seed42.json)
- [局部reset与物理参数重读](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/runtime/final_partial_64_seed45.json)
- [独立几何说明](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/geometry/README.md)
- [回归日志](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/contract/regression.log)
- [初始画面](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/preflight/videos/iteration_000002/initial_view.png)
- [20秒视频](/home/brainco/reorientation/training/diagnostics/fixed_palm_reset_20261105/preflight/videos/iteration_000002/rollout.mp4)

验证快照、来源哈希及限制见[validation.json](validation.json)。独立终端日志美化改动继续保留，未混入本次提交。
