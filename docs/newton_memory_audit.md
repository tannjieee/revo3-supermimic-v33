# Revo3 Newton 显存审计（2026-10-20）

初始迁移将防溢出的保守容量 `nconmax=1024 / njmax=4096` 留作默认，尚未按实际负载收紧。
22.4 GiB 是该配置的实测值，不能当作完成调优后的显存需求。
本次依据官方求解器调优流程，分开测量容量、运行时缓冲区和 CUDA Graph 的开销。

## 官方接口核对

- 启动和配置：`launch_simulation`、`PresetCfg`、`NewtonCfg(solver_cfg=MJWarpSolverCfg(...))`；训练直接调用统一 `isaaclab train`。
- 运行时实际物理类为 `isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager`，求解器为官方 Newton 的 `SolverMuJoCo`。
- 实测 `nworld=2048`、单世界 `nv=27`、`ngeom=83`，包括手部 82 个碰撞体和物体碰撞体。
- 状态训练任务的可视化器列表、传感器列表均为空，显存高的主要来源是物理碰撞工作区。
- 机器人资产使用官方 `UrdfConverter` 的 asset transformer / multi-physics 转换。官方 Isaac Lab 源码未修改。

一次性诊断脚本通过只读 `_solver` 字段统计求解器内部数组；上游没有公开的 solver 访问器。该脚本单独保存在审计目录，生产入口仍通过上述公开配置构建和派发。

官方 [调优指南](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/solver_tuning_mjwarp.rst)
要求按重置、动作及随机化分布实测容量，稳定后逐项降低过量预留。
官方 [Allegro 配置](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/allegro_hand/allegro_hand_common.py)
采用更小容量，但其碰撞模型不同，不能直接把 70 / 80 搬到这只手上。

## 分项对照

以下 64 环境检查均执行 360 步混合动作与多次重置，退出 0，读取的溢出标记为 0。

| 配置 | nconmax | njmax | CUDA Graph | 进程显存峰值 |
|---|---:|---:|---|---:|
| 原始基线 | 1024 | 4096 | 开 | 1328 MiB |
| 只收紧约束容量 | 1024 | 1024 | 开 | 1264 MiB |
| 只减半接触容量 | 512 | 4096 | 开 | 1008 MiB |
| 诊断用关闭 Graph | 1024 | 4096 | 关 | 1006 MiB |

关闭 Graph 仅用于辨别来源。最终配置保留 Graph、2 个子步和原来的碰撞模型。

## 主要分配来源

安装的 MuJoCo Warp `collision_convex.py` 在 `ccd()` 内分配 EPA 临时数组。
`naccdmax` 默认等于 `nworld × nconmax`。当前 `ccd_iterations=35`，其顶点、索引、面、投影、法向模长与边界数组合计约为：

```text
每容量项字节数 = 16 × (10 + 2 × 35) + 20 × (6 + 5 × 35) + 4 × 24
               = 4996 bytes
```

这个 `ccd_iterations` 是凸体碰撞算法的迭代上限，不是连续碰撞检测开关。
图捕获的两个子步各保留一组该工作区：

- 原配置：`2 × 2048 × 1024 × 4996`，约 **19.52 GiB**。
- 新配置：`2 × 2048 × 640 × 4996`，约 **12.20 GiB**。

这是依据分配源码计算的工作区大小，不是完整 CUDA 内存清单。
它与实测吻合：新配置在 reset 后为 1634 MiB，运行并捕获 Graph 后为 14218 MiB，增加 12584 MiB；上述两份工作区计算为 12490 MiB。
新配置可枚举的持久 Warp 数组合计约 873 MiB；另有 CUDA 上下文、Graph、内存池预留及对齐开销。
Torch 峰值只有约 9.49 MiB，无法代表 Warp / MuJoCo / CUDA Graph 的内存。

## 2048 环境修正与验证

默认已改为 `nconmax=640 / njmax=1024`，保留 `debug_mode=True`。

后续 1200 步复核发现 50 次线搜索上限曾触发 `LS_ITERATIONS` 标志；这不是接触或约束容量溢出。
按官方调优接口将 `ls_iterations=100` 写入默认后，2048 环境 / 1200 步检查通过，标志为 0，显存仍为 14218 MiB。

| 配置 | 步数 | 进程显存峰值 | 结果 |
|---|---:|---:|---|
| 初始迁移，1024 / 4096 | 120 | 22954 MiB（22.4 GiB） | 通过 |
| 审计后，640 / 1024 | 600 | 14218 MiB（13.9 GiB） | 通过 |
| 官方 PPO，2048 环境，640 / 1024 | 2 次更新 / 98304 环境步 | 14352 MiB（14.0 GiB） | 通过并保存检查点 |

同样的 2048 环境，观测峰值减少 **8736 MiB，约 38%**。
600 步中接触总数峰值为 765722，宽相候选总数峰值为 900173，均小于共享容量 1310720；宽相平均约 440 对/世界。
单世界约束行峰值为 462，小于 1024；重置、选择性重置和观测/动作契约检查通过，溢出标记始终为 0。
计数在策略步边界读取，同时检查全部运行日志中的容量警告；它不是对未运行过的策略和状态分布的最大值保证。

PPO 检查使用 `bash scripts/run_newton.sh isaaclab train`，包含实际网络和 rollout 缓冲区，容量警告为 0；其采样覆盖 uv 启动的 Python 子进程。
源码差异格式检查、Ruff 及更新后的 wheel 构建通过，wheel 内的容量已核验为 640 / 1024。

各案例使用独立进程，显存由 `nvidia-smi` 每 0.5 秒采样当前进程；训练脚本的 uv 子进程也纳入采样。
64 环境的诊断期间其他 GPU 作业仍在运行，故不据此比较吞吐速度。
原始记录在 22020 的 `/home/zhangke/viserdex-newton-validation-20261020/memory-audit/`；22031 留存审计报告和副本。

这是容量调优和运行链路检查，尚未证明策略收敛或真机接触准确度。扩大重置、物体或策略分布后仍需监控容量警告。
