# Revo3 SuperMimic V33 (In-Hand Dexterous Reorientation)

基于可微世界模型（SuperMimic 范式）的 **BrainCo Revo3 21-DoF 灵巧手手内物体重定向（In-Hand Reorientation）** V33 完整基线代码库。本仓库整理自经过生产验证的纯净 V33 冻结源码快照，并完整打包了 Revo3 右手 144 凸包资产、58 mm / 50 mm 骰子仿真资产、配置文件、单元测试与启动脚本，便于直接复现与后续分阶段优化。

## 核心特性（V33 基线）

- **控制与软限位协议**（[configs/supermimic_v33/control.yaml](configs/supermimic_v33/control.yaml)）：
  - 21 关节按名称配置软限位，扣除任意允许零位偏置后严格位于 URDF/USD 机械限位内；
  - 每个环境、每个关节在 reset 时独立采样编码器零偏 $\delta \in [-1^\circ, 1^\circ]$；
  - 动作经 `tanh` 仿射映射到软限位后执行 EMA 滤波（$\alpha=0.51$ @ 25 Hz）及可选目标速率限制（如 `100°/s`）。
- **网络结构**（[source/viserdex/viserdex/supermimic/networks.py](source/viserdex/viserdex/supermimic/networks.py)）：
  - **Actor**：`75×3=225 → 1024 → 512 → 256 → 21`（ELU + 末层 `tanh`），输入包含归一化关节角、编码器跟踪误差、上一原始动作、物体/目标位置及相对旋转 6D；
  - **World Model**：`149 → 2048×3 → 54 物理增量 + 22 手物接触 logits`（LayerNorm + SiLU），支持 `--wm_cuda_graph` 捕获加速；
  - **独立失败头**：`57 → 64 → 64 → 1`，使用独立 Adam 优化器在真实与 `pred.detach()` 状态上监督 `terminated`；
  - **Critic**：`200 → 512 → 256 → 128 → 6`（ELU，六头线性值输出，按想象 $\lambda$-return 拟合）。
- **物理栈与资产**：
  - Isaac Lab v3.0.0 + Newton / MuJoCo Warp（`physics=newton_mjwarp`，25 Hz 控制周期）；
  - 自包含 Revo3 右手 144 凸包 USD 资产（`assets/revo3_d405/vendor/brainco-description/revo3_system/optimized/right_20261027/revo3_right.usd`）与 58 mm / 50 mm 骰子资产（`assets/rounded_die_58mm/`、`assets/rounded_die_50mm_47g/`）。

## 目录结构

```text
.
├── assets/                   # Revo3 144-凸包右手 USD、58mm/50mm 骰子等完整仿真资产
├── config/                   # URDF 限位参考、SDK/训练软限位、MJWarp 容量补丁等
├── configs/supermimic_v33/   # V33 控制与边界配置（control.yaml / control_target100.yaml / control_no_target_home.yaml）
├── docs/                     # V33 设计文档、GPU 优化分析与 SuperMimic 官方实现对照文档
├── generated/                # 经审计的 Revo3 manual-convex 引用层与 mechanical-audit.json
├── scripts/
│   ├── launch/               # V33 一键启动脚本（run.sh / run_v33.sh / run_v33_no_target_home.sh）
│   ├── preprocessing/        # 资产预处理与审计脚本
│   └── supermimic/           # 训练（train.py）、录屏（record_manual.py）、导出（export_direct_actor.py）等
├── source/viserdex/          # viserdex 核心 Python 包（supermimic 算法、资产、仿真与推理）
└── tests/                    # V33 单元测试与回归测试
```

## 快速开始

### 1. 环境准备

依赖固定版本的 Isaac Sim 6.2 / Isaac Lab v3.0.0 / Newton 1.6.1 / Warp 1.17.0。可通过环境变量指定本机路径：

```bash
export ISAACLAB_DIR=/path/to/IsaacLab-v3.0.0
export VISERDEX_ENV_PREFIX=/path/to/envs/viserdex-newton
```

### 2. 校验与重建资产引用层（首次在新路径运行推荐）

```bash
bash scripts/launch/run.sh prepare
```

### 3. 运行单元测试

```bash
bash scripts/launch/run.sh test
```

### 4. 启动训练

```bash
# V34 Phase 0+1 优化版（27D 速度增量解析积分 + Bryson 物理容差 + 10% WM Holdout +
# 75/25 均匀失败采样 + 确定性想象 + Pre-slew target_rate + Critic V_max/LayerNorm/GELU +
# linear_fine 强梯度朝向奖励 + gamma=0.98/lam=0.9025）
bash scripts/launch/run_v34_opt.sh --output_dir ./runs/v34_opt_p1 --iterations 30000

# V34 快速 GPU smoke 验证
bash scripts/launch/run_v34_opt.sh --output_dir ./runs/v34_smoke --iterations 5 --smoke

# V33 基线（target100 / no_target_home）
bash scripts/launch/run_v33.sh --output_dir ./runs/v33_target100 --iterations 30000
bash scripts/launch/run_v33_no_target_home.sh --output_dir ./runs/v33_no_target_home --iterations 30000
```

## 相关文档

- V33 完整接口与网络说明：[docs/revo3_supermimic_v33_zh.md](docs/revo3_supermimic_v33_zh.md)
- V33 训练趋势与 WM CUDA Graph 优化：[docs/v33_training_gpu_20261107_zh.md](docs/v33_training_gpu_20261107_zh.md)
- SuperMimic 官方开源与本地 V33 对照审计：[docs/supermimic_official_comparison_20261111_zh.md](docs/supermimic_official_comparison_20261111_zh.md)
- Dex-Supermimic 开源参考与优化分析：[docs/dex_supermimic_reference_plan_20261106_zh.md](docs/dex_supermimic_reference_plan_20261106_zh.md)
