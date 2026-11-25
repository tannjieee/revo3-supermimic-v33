# 2026-11-05 训练启动：EPA horizon 容量诊断

第一次正式启动 `orientation_costs_v27_1024_10000_20261105_161219`
在完成第 107 轮后、world 34 的真实采集中触发 `EPA_HORIZON`（bit 256），
健康检查中止训练；Actor/Critic 尚未更新。现场保存在该 run 的
`physics_failure.json` 和 `physics_failure_state.pt`，未删除失败记录。

## 原因和改动

安装的 MJWarp 在 `collision_convex.py` 中用 `MJ_MAX_EPAHORIZON` 分配 EPA
边界边临时数组。它原为 24，与 `nconmax`、`njmax`、约束求解迭代预算分别独立；
增加后几项不能扩大这个数组。诊断表明溢出发生在凸包距离/穿透求解中，
不能归因于 WM 预测，也不是本次机械限位违规的报告。

将安装依赖 `mujoco_warp/_src/types.py` 的常量由 24 改成 64。
这是明确记录的本地依赖补丁，不是未修改的上游安装；包版本保持固定。
每个 scratch pair 增加 160 字节。补丁不改变碰撞几何、物理参数、
求解阈值、初始随机化或健康检查。版本化副本位于训练仓库的
`patches/mujoco_warp_epa_horizon_64.patch`，本目录也保留补丁和原文件备份。

- 原文件 SHA256：`a49625a8994db0d3b52dd34f98adac7f70882d473108c4e4793e4830e81ea71c`
- 补丁后 SHA256：`12e16a7157dee7dbb561d23e69f0468ec694e0b0e968aec7f138391b747f3d23`
- 配置记录：仓库 `config/mjwarp_capacity_patch.json`。
- 启动时 `runtime_capacity.capacity_manifest()` 检查值、已导入模块的一致性和文件哈希；
  该记录进入 context manifest，检查点恢复不得静默跨越这个运行合同。

如需恢复原依赖，应先停止使用该运行时的进程，确认当前文件是上述补丁后哈希，
再恢复 `types.py.original` 并重启进程；不能在运行中替换常量。
恢复后旧运行合同与新补丁检查点不再一致，不应绕过检查继续训练。

## 验证及边界

`check_saved_contacts.py` 使用上游 `_geom_dist` 的显式 `horizon_size` 参数，
在保存的 40 个控制边界状态上比较 24/64 两种容量。528 组接触距离完全相同，
两组均无溢出，见 `saved_contact_capacity_check.json`。
这些是静态距离查询，使用已审计几何；不涉及质量或电机力矩。
失败的中间积分子步没有保存，因此这不是失败子步的精确重放证明。

补丁后 159 项测试通过，见 `training/validation_epa_capacity_tests.log`。
以相同 seed 42、1024 环境和任务配置重新从头启动
`orientation_costs_epa64_1024_10000_20261105_162115`，已越过原失败轮次，
完成 500 轮 WM 预热并进入联合训练。启动验证快照保存在该 job 的
`startup_validation.json`。这只是已运行区间的证据，不能保证未来永不溢出，
也不能视为之前机械越界问题已经解决；健康检查仍会在异常时保存现场并退出。

本次训练使用姿态奖励和五项代价，不跟踪目标位置；无抓握库，物体从手中初始化。
目标为 10000 轮，每 100 轮保存检查点，TensorBoard 为 `http://127.0.0.1:6009`。
