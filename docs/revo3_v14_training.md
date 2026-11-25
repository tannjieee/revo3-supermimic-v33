# V14 凸碰撞体接入原老师训练（2026-10-21）

> 2026-10-21 状态更新：按用户新指令已停止 V14 凸包训练，已恢复原 SDF 8192 环境训练。V14 资产和可选任务保留，以下为接入记录。

当前任务为 `Isaac-ViserDex-Repose-Revo3-V14`。采用选定的 `v14_mcp_touch_restored_20261013` 几何，93 个凸碰撞体、21 个活动关节；不使用 SDF。老师网络、PPO、观测与动作顺序、奖励、XY 目标姿态采样、重置、随机化、初始抓握姿态与驱动参数继承现有 Revo3 老师任务。

训练专用资产 `assets/generated/revo3_v14_training/revo3_v14_training.usda` 引用官方导入的 V14 多物理 USD。它和展示用的 `make_v14_convex_cfg()` 分开：展示配置的原 V14 SysID 驱动和零姿态不带入训练。所有刚体坐标、关节轴、局部帧与限位在构建时逐项核对；质量、COM、主惯量、主轴、Newton 显式完整惯量及被动关节属性采用旧老师资产的精确值。93 块碰撞网格的点数组和 V14 导入结果完全一致。原始源模型、SDF 资产和引擎源码均未修改。

生成训练资产：

```bash
bash scripts/run_newton.sh scripts/preprocessing/prepare_revo3_newton.py --v14_training --viz none
```

启动 4096 环境：

```bash
bash scripts/train_revo3_v14.sh
# 继续旧策略时追加 --checkpoint /absolute/path/model_100.pt
```

仍使用官方 `launch_simulation`、统一 `isaaclab train` 和 `physics=newton_mjwarp`。凸包采用 `use_mujoco_contacts=True`，去掉 SDF 所需的 Newton 三角形/SDF 碰撞管线。物理步长 1/120 s、每步 2 个求解子步、策略 60 Hz 保持不变。积分器 implicitfast、elliptic 摩擦锥、求解迭代 100、线搜索 100、GJK/EPA 35、update_data_interval=1 和调试诊断保持原值，512 环境压力测试保持 CUDA 图开启；4096 训练为避免 CUDA 图捕获 OOM，使用官方 `env.sim.physics.use_cuda_graph=false` 执行模式，未改变物理方程。

容量按官方 `solver_tuning_mjwarp.rst` 测量设置：`nconmax=768`、`njmax=512`。在 512 环境、512 步保持/零/随机混合动作及多次重置中，共享候选对计数最大 281941 / 393216（约 72%）；实际接触总数最大 214143，采样的单环境最大接触数 464，最大约束行数 174 / 512。随机化和观测噪声保持启用，未发现容量溢出、截断、迭代触顶或非有限状态。前置 64 环境、384 步检查也通过。容量会随策略分布改变，训练监督器保留运行时容量/数值诊断，遇到错误停止。

失败诊断单独保留：原接触容量 1024 在 8192 环境发生 EPA 缓冲分配 OOM，在 4096 环境发生 CUDA 图捕获 OOM；768/512 容量下的 4096 CUDA 图捕获同样 OOM；试验容量 256 被初始化自动增加且仍发生 broadphase overflow，判为不合格，未用于训练。没有通过删除碰撞体、关闭随机化或降低 GJK/EPA 精度解决显存问题。

旧 8192 SDF 训练按用户指令停止，日志最后完成第 181 轮；最近的已保存 checkpoint 为第 100 轮，保持原样。新训练在 22021 的物理 GPU 3、4096 环境恢复该 checkpoint 的策略、优化器和训练轮次；按原总目标 5000 轮配置剩余 4900 轮。由于环境数按用户要求减半，每轮 rollout 样本数也由 196608 改为 98304（24 步/环境）。

运行目录：
`/home/zhangke/viserdex-newton-validation-20261020/training-runs/revo3_v14_convex_4096_20261021/`

该目录保存启动命令、源码哈希、原配置副本、精确参数对照、压力测试、GPU 采样、训练日志与失败尝试。主项目和训练资产也同步到 22031 的 `/home/zhangke/viserDex_revo3`。这是接入与启动验证，不代表长程策略训练已完成。
