# Revo3 SuperMimic 两卡同步训练

**2026-10-24：已按用户许可改到22021的GPU1、GPU5，正式任务从第0轮开始，两卡共同训练一个模型，每卡2048环境、合计4096，控制频率30Hz。** 未传入`--resume`，两个rank的`parent_checkpoint`均为null，首轮优化计数为WM10 / Actor0 / Critic0；随机初始化后的模型摘要相同。保持原配置的前500轮WM预训练，之后进入Actor/Critic联合更新。

15:22北京时间已核查到100轮，并在CPU重新加载正式任务的首个`latest.pt`：iteration100，WM1000 / Actor0 / Critic0，parent_checkpoint=null。三网络及归一化buffer的落盘摘要为`881c1c89dbe6d8de4536448352669054bf25934e4f8cd6de1dec4f47b0471f21`，与保存时两rank一致性检查的摘要相同。最近50轮WM更新时间中位数0.547秒、整轮1.385秒，WM阶段每卡14706/14705 MiB；此阶段尚不能评价策略控制效果。[正式运行证据](evidence/supermimic_training_20261024/multigpu_scratch_22021.json)。

独立部署路径为 `/home/zhangke/viserDex_supermimic_2gpu_20261024`，运行时代码提交 `82a69c4`，100个代码/配置文件SHA-256与本地一致。正式目录为 `outputs/supermimic_revo3_2x2048_30hz_bf16_scratch_20261024`，状态文件为 `outputs/supermimic_2gpu_training_active.json`，启动脚本为 `outputs/launch_multigpu_scratch.sh`。rank0/1 PID为175860/175861；tmux会话为`revo3_sm_2gpu_scratch_20261024`。

22020在准备期间被其他用户的8卡任务占满，因此其guard拒绝启动；未干预该任务。[22020历史测试证据](evidence/supermimic_training_20261024/multigpu_setup_22020.json)。用户授权换服务器后，在22021重新执行通信及仿真容量验证。首次部署链接到缺少kornia的基础运行时，在导入阶段退出；已改为复用原项目的`/home/zhangke/viserDex_revo3/.venv-newton`，未安装或替换共享环境中的依赖。

官方版本仍为Isaac Lab v3.0.0 / Isaac Sim6.2.0.0 / Newton1.6.1 / Warp1.17.0 / Torch2.11.0+cu130，保留30Hz、物理120Hz、现有资产、观测、噪声和三网络架构。旧22021/GPU0单卡任务保持独立运行。

## 分布式语义

- 遵循官方 [多卡流程](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/features/multi_gpu.rst) 的 torchrun 一卡一进程、官方 launcher 分配设备和每rank独立仿真。该入口仍是本项目自定义 SuperMimic，不把算法切换为PPO或蒸馏。
- 两个rank各2048真实环境，各自持有1000向量步CPU replay；网络参数、归一化统计、Adam状态及优化次数保持同步。rank种子为42、43，模拟器及采样随机流独立。
- 每卡沿用WM batch2048、Actor batch4096、k8/H32、每轮10/1/20次WM/Actor/Critic更新；因此全局WM batch4096、Actor batch8192。学习率不随卡数倍增。每卡采集仍为交替2/3向量步（UTD0.4），每轮全局新增transition数量加倍。
- 每个逻辑批次完成backward后，将梯度全局求平均，再进行原梯度裁剪与Adam更新。这与安装的官方RSL-RL `reduce_parameters` 所用的“backward后归约梯度”顺序一致；自定义WM冻结反传路径不套用PPO实现。
- 实际数据更新的四个RunningNorm通过全局一阶/二阶矩合并，各rank获得相同均值、方差和样本总数。Critic的有效时间步数可能不同，因此分母按全局有效数量加权，不能简单平均不等长批次的均值。
- rank0保存一份共同checkpoint；rank1仅写自己的配置、指标和验证结果。全局成功数和结束episode数先求和，再计算目标数/episode。保存checkpoint时对三套完整模型及归一化buffer计算SHA-256，要求两rank完全一致。
- 代码支持从单卡checkpoint恢复模型、Adam、计数和调度器；恢复时仿真和replay重新预热，分布式随机流按rank重新设置。此次正式任务按用户最新要求从零训练，不使用该恢复入口。

## 已完成测试

22020 CPU测试：21项通过（8.12秒），包含不等长分片的归一化与合并批次比较、梯度先归约再裁剪/Adam与合并批次比较、三网络连续同步更新、checkpoint读写恢复、全局指标聚合和共同停止信号。

22020的GPU1/2及22021的GPU1/5均通过NCCL实测。22021使用正式项目运行时再次验证，无需通信兼容选项。两rank均执行WM3 / Actor3 / Critic18次优化，模型摘要均为 `eefee35ee6549b8c9d69a2c4b2a75dd353b64947a7e8a15893a512448cb06321`。测试入口：`scripts/tests/check_supermimic_nccl.py`。

22021完整容量验证使用旧11500轮checkpoint执行两轮真实联合更新，两个rank各2048环境，未启用smoke或缩减网络批次。正常退出码0，11502轮checkpoint可在CPU重新加载；两rank三套网络及归一化buffer的摘要与落盘权重一致：`8828464e298904cf2dccbc7b7737974accc214833dcb2a9cc0e656a239abca5a`。最终优化次数为WM115020 / Actor11002 / Critic220040；第二轮网络更新0.943秒，整卡峰值24538/24537 MiB。日志没有OOM、数值错误或碰撞容量溢出；奖励最大误差1.91e-6。此测试权重只用于容量检查，随后正式任务按用户要求从零启动。短测不能证明长期收敛。

[完整容量验证证据](evidence/supermimic_training_20261024/multigpu_validation_22021.json)同时保存两rank配置、验证输出、checkpoint摘要和GPU峰值。

## 启动方式

```bash
CUDA_VISIBLE_DEVICES=GPU-214e9d42-152c-d199-5172-5d2c4860886f,GPU-15a68363-36f8-da93-fc93-4cbc5d5313df \
ISAACLAB_PATH=/home/zhangke/IsaacLab-v3.0.0 OMP_NUM_THREADS=4 \
bash scripts/run_newton.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 scripts/supermimic/train.py \
  --distributed --num_envs 2048 --viz none --seed 42 \
  --env_config /home/zhangke/viserDex_revo3/outputs/student_dagger_start_20261022/frozen_teacher/params/env.yaml \
  --output_dir <新运行目录> \
  --amp bf16 --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 32768 \
  --iterations 320000 --save_interval 2000 --latest_interval 100
```

`run_guard.py` 的 `--gpu` 可重复两次，启动前检查两张卡，并分别记录显存峰值与rank PID。正常停止主rank后，通过集体同步让两rank在同一轮结束并保存共同权重。
