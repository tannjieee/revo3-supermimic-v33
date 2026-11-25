# RTX 5090 训练预设

> 此文保留旧 Sim 6.0 / beta PhysX 基线的安装或测量记录。当前 v3.0.0 Newton 安装、默认值和启动命令以 [Newton 迁移说明](newton_migration.md) 为准。

`scripts/train_revo3_5090.sh` 提供在 RTX 5090 32 GB 上测量过的 PPO 配置。
历史验证机器使用 Core Ultra 9 285K（24 核）和 62 GiB 内存。
不同硬件应先以较少环境做 smoke check，再选择容量。

## 使用

```bash
bash scripts/train_revo3_5090.sh --run_name my_teacher
# 小规模执行检查
bash scripts/train_revo3_5090.sh --num_envs 64 --max_iterations 5 --run_name my_smoke
```

默认 8192 环境、24 步 rollout、5000 轮，PPO 5 epochs / 16 minibatches。
物理步长 1/120 秒、decimation 为 2，控制为 60 Hz。
该预设保留任务的奖励、动作、观测、自碰撞和求解器精度，覆盖以下运行参数：

- PhysX CPU 线程为 8。
- 关闭未使用的机器人/物体接触报告和目标调试标记；物理碰撞仍启用。
  新增 ContactSensor 时，应重新开启对应资产的报告。
- `gpu_max_rigid_patch_count=1048576`。
- 每 100 轮保存一次检查点。

普通 `scripts/rsl_rl/train.py` 使用任务默认配置，不会自动套用这个预设。

## 历史短程测量（2026-10-13）

同机、相同 seed、8192 环境和 PPO 设置，匹配前 4 轮：

| 指标 | 原配置 | 预设 |
| --- | ---: | ---: |
| 平均耗时 / 轮 | 28.375 s | 26.035 s |
| GPU 进程占用采样 | 15,710 MiB | 14,750 MiB |
| CPU RSS 采样 | 28.42 GiB | 26.74 GiB |

短测轮耗时下降 8.25%，吞吐提高约 9.0%；不代表长期吞吐或达到同一成功率的训练时间。
4 轮共 786,432 条 transition，未观察到 PhysX 缓冲区溢出；两组初始检查点的 76 个张量相同。
原始日志和检查点不随代码分发。

## 保存、停止与分析

训练打印 `Checkpoint control ready` 后支持以下信号：

```bash
# TRAINER_PID 为当前运行新版 train.py 的 Python 进程 PID。
kill -USR1 "$TRAINER_PID"  # 当前轮完成后保存并继续
kill -TERM "$TRAINER_PID" # 当前轮完成后保存并停止
```

Ctrl+C 也会等待当前轮完成后保存。信号处理函数仅设置请求标志，检查点在优化器更新完成后写入。

```bash
# 生成分段耗时与资源记录
bash scripts/train_revo3_5090.sh --max_iterations 4 --profile --run_name my_profile

# 从相同手型、相同物体的检查点恢复
bash scripts/train_revo3_5090.sh --resume True \
  --load_run my_teacher --checkpoint model_100.pt --run_name my_teacher_resume
```

PPO 的 `max_iterations` 表示本次继续执行的轮数；学生蒸馏入口的同名参数表示累计更新上限，
详见 [蒸馏说明](revo3_distillation.md)。计时为嵌套主机调用耗时，不额外执行 CUDA 同步，
不能当作 GPU kernel 计时。
