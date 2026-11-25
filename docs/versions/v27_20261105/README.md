# 当前算法基线：v27 姿态保持与五项代价

记录日期：2026-11-05。此版本保存当前算法实现和已启动训练的参数，供后续比较、
回退和继续开发使用；不表示策略已收敛或满足真实硬件约束。

- 分支：`codex/algorithm-v27-orientation-costs-20261105`
- 标签：`algorithm-v27-orientation-costs-20261105`
- 父版本：`000c760`（原 `codex/yellow-die-rgbd-play-predictor-20261102` 分支）
- 网络协议：`hardware_center25_v3`；检查点协议：27。

## 本版行为

Actor 为 `[1024,512,256]`，WM 为 `[2048,2048,2048]`，Critic 为 `[512,256,128]`，
末层包含 1 个奖励值和 5 个代价值。关节按小指到拇指排列，拇指下发顺序为
MCP/PIP/DIP/CMP/CMR；编码器上下限对齐固定 URDF，网络物理缩放统一。
几何特征和 delay-ready 输入关闭，物体位置在网络入口移到固定手中坐标系。

姿态到达是正向奖励；位置跟踪关闭。五项代价为关节速度、目标变化速度、
物体角速度、物体线速度、掉落。物体速度代价接近目标姿态时平滑增强；
连续稳定 13 步（0.52 秒）用于成功统计。具体输入输出、公式及兼容协议见
[网络详细说明](../../revo3_network_v27_zh.md)。

训练为 1024 环境、10000 轮、seed 42；从手中初始化物体，不使用抓握库。
控制/PD/积分为 25/200/400 Hz，γ=0.99，先进行 500 轮 WM 预热。
GPU Replay、WM CUDA Graph 启用，每 100 轮保存检查点。
电机力矩限幅为 1.28 N·m，声明速度为 100 rpm；当前求解器不将该字段作为
真实关节硬限速。速度代价阈值分别为实际关节 90°/s 和目标变化 60°/s。

## 版本内的运行资料

- `config.json`、`env.yaml`：本次运行的完整配置原样快照，含本机路径。
- `launch.json`：实际启动参数、来源作业和首次失败记录。
- `workspace_run.sh.txt`：当时的工作区启动器原文；这是审计快照，不能直接在本目录运行。
- `startup_validation.json`：第 919 轮的启动验证快照，不是实时训练状态。
- `source_match.json`：与启动时源码哈希清单逐项比较的结果。
- `runtime_versions.json`：实际运行时包版本；包版本外另有下述本地补丁。
- `unit_test_summary.txt`：该版本已有的 159 项测试结果。
- `epa_capacity_check.json`、[EPA 诊断](epa_diagnosis.md)：容量修改依据和验证边界。

当前运行名为 `orientation_costs_epa64_1024_10000_20261105_162115`。
日志和权重位于工作区 `training/runs/`、`training/jobs/`，不作为本版 Git 内容。
工作区的 `training/current_job/git_version.json` 将保存提交与标签对应关系。
原仓库已有的历史模型文件保持原状，本次不新增模型二进制。

## 环境及资产恢复

遵循仓库 [Newton 迁移文档](../../newton_migration.md) 固定的官方工作区与启动路径。
需要 Isaac Sim 6.2.0.0、Isaac Lab v3.0.0 官方工作区、Newton 1.6.1 和 Warp 1.17.0。
`launch.json` 依赖原工作区的 `training/run.sh`；移机时应先恢复工作区布局和依赖，
按快照校对路径，再选择新的输出目录，不能直接覆盖已运行目录。

当前安装的 MJWarp 还需要仓库 `patches/mujoco_warp_epa_horizon_64.patch`：
将 EPA 边临时容量 24 改为 64。原/新文件 SHA256 在
`config/mjwarp_capacity_patch.json`，启动时会验证补丁并将它写入运行合同。
不要对不匹配的上游文件盲目应用补丁，不要在训练进程运行时替换依赖。

右手资产固定为 BrainCo 描述库提交 `2a98621de70d59d182b8075214cebb96fd5375d3`，
源 USD SHA256 为 `737c42704c4e464b9da372142683f206ff43cbc1d9aacb5e25d05a4c7ce1914f`。
工作区路径和审计记录保存在运行配置；资产生成入口为
`scripts/preprocessing/prepare_revo3_newton.py --manual_convex_usd ...`。
`generated/revo3_manual_convex/` 只有依赖工作区相对路径的引用层和审计报告，
已设为忽略，应在恢复环境后通过上述入口重建。辨识参数 JSON、参考 URDF、
SDK 限位快照、训练限位和输出限幅均在本版中跟踪。

## 验证范围

已有 159 项单元/回归测试通过；4 环境短验证和正式 1024 环境启动均完成。
原启动第 107 轮的 EPA 缓存溢出后，扩容并使用同一任务、同一种子重新开始，
已进入联合训练；所保存第 900 轮检查点全部浮点张量为有限值。
这不保证整个训练过程不会再发生物理异常，也不证明此前机械越界已根治。
完整启动验证包含早期成功率与损失，不能将启动成功解读为训练成功。

本次只建立本地 Git 版本。后续算法修改应继续提交到分支，已保存的标签作为固定基线。
