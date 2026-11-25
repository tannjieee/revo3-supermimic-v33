# 官方 Newton 迁移方式复核（2026-10-20）

结论：本项目通过官方 Isaac Lab v3.0.0 的启动器、物理预设、资产工厂、URDF 转换器和统一训练入口运行 Newton + MJWarp。Revo3 的任务、奖励、资产与容量属于项目定制；遵循官方迁移方式不等于照搬某个官方任务的全部默认数值。

## 发布版和源码来源

- 从 GitHub 远端查询到的 `v3.0.0` 标签为 `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`，与两台服务器的官方工作区 HEAD 一致，官方工作区的已跟踪文件无修改。
- Isaac Sim 6.2.0.0、Newton 1.6.1、Warp 1.17.0、MuJoCo / MuJoCo Warp 3.12.0、Torch 2.11.0+cu130、RSL-RL 5.5.1。
- 两台独立环境分别对 Newton 749、MuJoCo Warp 73、Warp 455 个 Python 文件校验安装包 RECORD，合计 1277 个，均无差异。
- 22031 是代码主目录，GPU 验证在 22020 空闲卡执行。项目 452 个源码、脚本和随包资产文件按 SHA-256 对照一致；最后的参数修改后再次核验。

## 运行时核对

项目配置采用 `PresetCfg`、`NewtonCfg(solver_cfg=MJWarpSolverCfg(...))`；自定义运行脚本调用 `launch_simulation`，训练采用 `isaaclab train`。
实际手部实现为 `isaaclab_newton.assets.articulation.articulation.Articulation`，物体为 `isaaclab_newton.assets.rigid_object.rigid_object.RigidObject`。
实际物理管理器是 `NewtonMJWarpManager`，求解器是官方 Newton 的 `SolverMuJoCo`。
这正是官方支持的 Newton + MJWarp 路径；无窗口状态训练的可视化器、相机列表均为空，启用 Kit 可视化时使用 Isaac Sim。

资产通过官方 `UrdfConverter`，启用 `run_asset_transformer` 和 `run_multi_physics_conversion`。
重新读取生成的 USD，得到 21 个活动关节、22 个刚体、82 个碰撞体；质量、惯量及质心检查通过，没有把碰撞 API 挂在非几何节点上，没有缺失依赖。
运行时所有 83 个几何体的 `condim=3`，具有法向和切向摩擦。

21 个受控关节通过官方导入形成 42 个 P / D 子执行器，合计力矩限制位于关节级 `jnt_actfrcrange`，为 ±0.6 N·m。实际力矩读回断言通过，armature 为 0.001。
不能仅看每个子执行器的 `actuator_forcelimited` 判断合计关节限力是否生效；上游正是按关节合计 P+D 的方式限力。

## 本轮发现并修复的问题

第一次将混合动作检查延长到 1200 步时，读取到 `overflow` 字段的 1024 位。上游枚举定义为 `LS_ITERATIONS`，代表 50 次线搜索上限触顶；接触数和约束数均未达到容量上限。
保留检查断言，通过官方公开参数将 `ls_iterations` 从 50 调到 100，随后同规模 1200 步检查通过，所有读取到的标志为 0。
该修复没有修改 Newton / Warp / Isaac Lab 源码，也保留了碰撞几何、自碰撞、两个子步和 CUDA Graph。

当前任务默认值：`nconmax=640`、`njmax=1024`、`ls_iterations=100`、`iterations=100`、`tolerance=1e-6`、`update_data_interval=1`、`debug_mode=True`。

## 本次实测

| 检查 | 结果 |
|---|---|
| 31 上最终配置的 CPU 测试 | 85 通过、2 跳过 |
| CUDA pose metrics / TF32 | 4 通过 |
| 资产机械审计与依赖检查 | 通过 |
| 64 环境 / 360 步，官方后端与动作观测契约 | 通过，标志 0 |
| 2048 环境 / 1200 步混合动作及重置 | 通过，标志 0，峰值 14218 MiB（13.9 GiB） |
| 4 环境 / 120 步，关节合计限力与摩擦模型 | 通过 |
| 默认 Newton 的 3DGS、手深度、投影和选择性重置 | 通过，2 环境 / 12 步 |
| 最终默认配置的官方 PPO 入口 | 2048 环境 / 3 次更新通过，峰值 14352 MiB（14.0 GiB），保存检查点 |
| Ruff、差异格式、wheel 内最终参数 | 通过 |

1200 步检查中宽相候选总数峰值 907663，小于共享容量 1310720；单世界约束行峰值 474，小于容量 1024。
计数在策略步边界读取，另检查运行日志；本次数据不是未运行过的状态分布的最大值保证。
无参数覆盖的训练命令会自行选中 Newton，而无需额外指定 `physics=newton_mjwarp`。

```bash
bash scripts/run_newton.sh isaaclab train \
  --task Isaac-ViserDex-Repose-Revo3 --viz none --device cuda:0 \
  --num_envs 2048 --max_iterations 3 --logger tensorboard \
  env.commands.object_pose.debug_vis=false
```

显存用 nvidia-smi 按进程及子进程采样。诊断脚本读取上游内部数组以检查标志和映射，单独保存于审计目录。
原始失败案例与修复后的成功案例都保留在 22020 的 `/home/zhangke/viserdex-newton-validation-20261020/official-recheck/`。
31 的记录目录为 `/home/zhangke/viserdex-newton-migration-20261020/official-recheck/`。

这是短程执行和迁移契约验证。完整策略尚未训练收敛，位姿 CNN 精度、其他物体、Allegro、扫描重建和真机行为没有在本轮验证。MJWarp 不执行原配置的硬关节速度限制，随机动作实测速度会超过 10 rad/s。

依据：[官方迁移指南](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)、[Newton 资产准备](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/prepare_asset_for_newton.rst)、[MJWarp 调优](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/solver_tuning_mjwarp.rst)。
