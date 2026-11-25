# Revo3 移植说明

本次将 ViserDex 的默认训练、回放和在线位姿估计入口切换到 BrainCo Revo3 右手。
任务仍为原来的手内物体重定向，奖励、目标姿态生成和物体数据选择方式保持一致。
代码与资产移植后，在远端工作站进行仿真与短程训练验证；验证范围和结果记录在文末。

## 资产

- 来源：https://github.com/BrainCoTech/brainco-description
- 分支：`main`；获取日期：2026-10-13。
- 提交：`f197d9d720b1b106a3298e2438ff07e055bf15ec`；上游发布标记：`20261009`。
- 位置：`source/viserdex/viserdex/assets/brainco/`。
- 范围：完整 Revo3 左右手、普通版和 MX Touch 版的 URDF、MJCF、USD、网格和纹理。
  未引入与手部移植无关的 RevoArm、RevoTron、Revo2 系统，也未复制上游工具脚本。
- 模型：`revo3_system/usd/revo3_right.usd`。普通版省略触觉点标记，保留实体手的几何。
  运行时使用 `assets/revo3_right_instanceable.usda`，通过 `revo3_right_isaaclab.usda`
  保留物理兼容配置并引用上游资产。
- `UPSTREAM.json` 保存提交来源及逐文件 SHA-256；上游资产内容未改写。

资产放在可随 Python 包分发的 `assets/brainco` 下，不受物体数据目录 `assets/data` 的忽略规则影响。

独立 USD 配置层将掌部凸分解的 `hullVertexLimit` 从 128 修正为 PhysX 支持的 64，
将 `errorPercentage` 从 0.01 改为 0.02，避免单精度表示略小于允许下限而触发参数警告。
上游文件和校验清单保留原样。

### 几何实例共享

`revo3_right_instanceable.usda` 将 139 个网格按刚体的视觉/碰撞分组，设为 44 个实例引用。
`revo3_right_geometry.usda` 保存这些分组及材质的引用，不复制顶点数组。不同环境共享
USD 几何原型，但 22 个刚体和 21 个转动关节仍独立；关节动作、重置、质量/摩擦随机化
继续作用于各环境的物理对象。未简化顶点、碰撞形状或材质，也未改变训练参数。
原来指向机器人全局材质的绑定被重定向到分组内的 `__materials` 引用，避免引用丢失
跨子树材质关系。独立遍历的机器人节点从 874 个降到 139 个；包含实例代理的完整几何不变。

上游更新后可用安装了 `usd-core` 的 Python 重新生成配置层：

```bash
python scripts/preprocessing/make_revo3_instanceable.py
```

生成器会拒绝包含刚体、关节或无法安全重定位的外部关系的共享子树。资产测试逐项比较原模型和实例模型的
属性与关系，并验证两个机器人共用原型且刚体变换独立；仿真测试另外检查 Isaac Lab
复制环境后仍保留实例共享。

2026-10-13 在同一台 RTX 5090 工作站上，用相同的 RSLCube、8192 环境、种子 42 和
默认 PPO 参数比较，结果如下。计时取相同的前几轮，排除首轮预热；短样本不代表长期吞吐。

| 指标 | 原资产 | 整组几何实例共享 |
| --- | ---: | ---: |
| 训练进程 RSS | 55.7 GiB | 28.1 GiB（减少 49.5%） |
| 场景创建 | 11.84 s | 8.48 s |
| 物理初始化 | 257.26 s | 128.97 s |
| PPO 每轮中位耗时 | 28.79 s | 28.03 s |
| 显存占用 | 约 15.7 GiB | 约 15.7 GiB |

主要收益是主机内存和启动时间，持续采样速度只小幅变化。资产检查 17 项通过，
4 环境/120 步的仿真检查和 60 帧 RTX 回放通过，未简化碰撞或改变动作/观测。
原训练的日志和检查点保留；优化后的正式训练重新从默认种子开始，仍为 8192 环境、5000 轮，
远端 run 为 `logs/rsl_rl/viserdex_revo3_sim6_ppo/2026-10-13_22-08-31`。
对比原始记录在 `logs/revo3_validation/20261013/instancing/comparison.json`。

## 动作和关节顺序

使用上游模型的 **21 个独立转动关节**，动作维数为 **21**。
转动关节之间没有 mimic 耦合；固定装饰件/触觉件上的 mimic 元数据不增加动作维度。
这里是模型的仿真控制接口，未额外推定真机协议的耦合规则。

`assets/revo3_spec.py` 中的 `REVO3_JOINT_NAMES` 是统一顺序来源：

| 下标 | 手指 | 关节顺序 |
| --- | --- | --- |
| 0–4 | 拇指 | CMP、CMR、MCP、PIP、DIP |
| 5–8 | 食指 | MPR、MCP、PIP、DIP |
| 9–12 | 中指 | MPR、MCP、PIP、DIP |
| 13–16 | 无名指 | MPR、MCP、PIP、DIP |
| 17–20 | 小指 | MPR、MCP、PIP、DIP |

完整名称遵循 `right_<finger>_<joint>_joint`。动作与关节观测均启用 `preserve_order=True`，
不依赖 USD 或 PhysX 自动枚举的顺序。

Isaac Lab 3.0 的原生 EMA 动作实现支持 `preserve_order`。此前为 Lab 2.3.2 编写的
排序兼容层已移除；运行测试检查限位映射、EMA 状态、位置目标和观测的顺序一致。

沿用任务的 EMA 位置动作：输入裁剪到 `[-1, 1]`，逐关节映射到模型限位，
再按 `target = 0.95 * mapped_action + 0.05 * previous_target` 平滑并限幅。
`-1` 对应该关节下限，`+1` 对应上限，`0` 对应限位中点。

## 观测布局

| 观测组 | 组成（顺序固定） | 维数 |
| --- | --- | --- |
| `policy` | 归一化关节位置 21 + 目标位置/四元数 7 + 上次动作 21 | 49 |
| `privileged` | 关节速度 21 + 物体位置 3 + 四元数 4 + 线速度 3 + 角速度 3 + 目标旋转差 4 | 38 |
| PPO actor / critic 输入 | 沿用原任务，拼接 `policy` 与 `privileged` | 87 |

关节位置按各自限位归一化；关节速度沿用 `0.2` 缩放。图像位姿估计的输出由物体关键点数量决定，
不随手的关节数改变。旧 Allegro 策略的输入/输出维数不同，需要重新训练 Revo3 策略。

## 初始位姿与驱动

`assets/revo3.py` 将手根固定在 `(0.12, 0, 0.51)`，绕世界 Y 轴旋转 `-90°`，
使模型局部 `+X` 抓握面向上、局部 `+Z` 手指方向朝世界 `-X`。
物体仍位于原任务的 `(0.02, 0, 0.565)` 工作区，保留原相机工作区。

初始关节姿态单独为 Revo3 指定：拇指设置对掌，四指轻微弯曲，侧摆置零。
重置时在默认姿态附近添加 `±0.03 rad` 扰动，裁剪到关节限位，速度清零。

上游 USD 的驱动增益为零，URDF 中 effort/velocity 为零占位。
移植配置显式提供仿真驱动初值：刚度 `7.0`、阻尼 `0.22`、力矩上限 `0.6`、速度上限 `10.0`，
启用自碰撞、重力和固定根节点。这些是待后续调试的仿真参数，不代表真机标定值。

## 策略与视觉流程

Revo3 策略日志目录独立为 `logs/rsl_rl/viserdex_revo3_sim6_ppo/`。
物体配置中的 `revo3_rollout_policy_run_name` 初始为 `None`；训练新策略后填写对应 run，
或向位姿估计脚本传入 `--policy_run <run_name>`。未配置时给出明确提示，不自动加载旧手检查点。

```bash
python scripts/rsl_rl/train.py --task Isaac-ViserDex-Repose-Revo3-v0 --logger tensorboard
python scripts/rsl_rl/play.py --task Isaac-ViserDex-Repose-Revo3-Play-v0 --load_run <run_name>
python scripts/pose_estimator/train_pose_estimator.py \
  --task Isaac-ViserDex-Repose-Revo3-Estimator-v0 --policy_run <run_name>
```

位姿估计及 Eval、Camera、NoAug 变体均已接入 Revo3。原 `masking_data.pt` 记录的是 Allegro 关节与轮廓，
因此 Revo3 配置设为 `masking_data=None`，不再使用该表。现通过动态多网格射线相机跟踪 Revo3
21 个指节及手掌，以手部深度和物体深度比较生成遮挡；Estimator、Eval、Camera、NoAug 均启用。
采集、审计和有遮挡训练入口见 [位姿训练说明](revo3_pose_training.md)。
相机内外参和真实采集数据仍来自原项目，真机迁移时需要单独标定、采集。

物体 USD、Gaussian Splats、关键点仍需按主 README 下载原物体数据集。
现有检查脚本保留供后续使用：`tests/test_revo3_assets.py` 和 `scripts/tests/smoke_revo3.py`。
这些检查用于确认代码、资产和运行接口；短程训练不代表已经获得有效的重定向策略。

## 当前验证

当前代码已迁移到 Isaac Sim 6.0.1 / Isaac Lab 3.0，结果见 [Sim 6 迁移说明](isaacsim6_migration.md)。
此前的 5.1 诊断、短程训练和 Storm 处理保存在 [历史记录](revo3_sim51_history.md)。
