# 每500轮录制与新版50 mm / 100 g骰子

2026-11-05。用户要求继续检查优化点、后续每500轮录制视频，随后指定使用新版50 mm骰子和100 g质量。
已实现并完成有限验证；没有启动正式1024环境/10000轮训练，没有连接真机。

- 新manual-convex运行默认 `die50`，完整更新500/1000/1500…轮后自动录制。
- 默认单环境500帧、25 fps、20秒、seed43。独立进程加载该轮检查点和保存的环境，训练同步等待。
- 旧播放错误使用SolverBudgetBinding且遗漏typed schema恢复，现已修正；播放不再额外改变接触容量。
- 记录中的 `.partial.mp4` 只在编码完成后改名；父进程核对摘要、帧数和模型哈希。失败留下日志并报错，检查点先保存。
- 新资产保留原造型与200顶点碰撞拓扑，视觉与碰撞均乘5/7。用户指定0.1 kg；惯量按质量和尺寸平方重算。
- 新旧模型用环境/geometry合同区分；旧70 mm恢复保持原对象，不能静默改成50 mm。新对象尚无训练完成的Actor。

完整说明：[网络详细文档第15、16节](../../revo3_network_v30_zh.md#15-每-500-轮自动录制策略视频)。
资产说明：[50 mm资产](../../../assets/rounded_die_50mm/README.md)。

## 实测

| 验证 | 结果 |
|---|---|
| 旧70 mm检查点1000→1001，视频周期1001、保存周期100 | 自动保存1001并录制，500帧全部解码 |
| 50 mm/100 g，64环境从头2轮，视频周期2、保存周期100 | 自动保存2并录制，500帧全部解码 |
| 视频格式 | H.264，1280×960，25 fps，20秒 |
| 录制前后完整checkpoint内容比较 | 全部相等，包含CPU/CUDA RNG与优化器 |
| 单次20秒视频墙钟耗时 | 约31.8–31.9秒 |
| 新对象单环境初始化视频 | 0次掉落、0次成功、1次20秒超时 |
| 新对象HOME Actor，seed44、64环境×500步 | 2次掉落、62次超时、0次成功、0次关节速度/目标速率违规 |
| 运行时凸包/质量/惯量与健康审计 | 通过；无溢出/机械健康失败 |
| 新旧完整环境/config比较 | 仅物体USD与目标标记USD路径变化；算法Config完全一致 |
| 104项相关测试 | 通过；另修正历史任务名陈旧测试断言 |

其中周期2/1001只用于有限功能测试，生产默认仍为500。
两轮新对象训练仅更新WM20次、Critic40次，Actor0次；视频不是已训练策略的效果演示。
多GPU同步路径未实机测试；没有进行PhysX对照或打印物体惯量测量。
固定seed不保证跨进程物理轨迹逐位相同。单段视频不能代替独立多环境评估。

## 优化判断

已修复录制加载/环境复用和新对象物理参数不一致的问题。50 mm/100 g HOME短评估提供了新的静态支撑基线，
但仍未学会旋转到达。WM动作响应方向/幅值、真实长期尾值、共同失败风险和速度通道有效尺度仍需在新对象数据上验证。
此前70 mm模型的响应/梯度数据保留为历史证据，不移植其WM-only候选或统计量，也不宣称这些学习问题已经解决。

## 证据与复现

完整产物根目录：`/home/brainco/reorientation/training/diagnostics/periodic_video_20261105`。
新视频：`die50_100g_preflight/videos/iteration_000002/rollout.mp4`。
临时21.87 g预检被100 g设定取代，`die50_preflight/DISCARDED.txt`标明作废，原临时USD另行归档。

```bash
# 在仓库中，104项相关检查
PYTHONPATH=source/viserdex /home/brainco/reorientation/runtime/policy/bin/python -m pytest -q \
  tests/test_die50_asset.py tests/test_periodic_video.py tests/test_absolute_policy.py \
  tests/test_stable_values.py tests/test_response_experiment.py tests/test_revised_contract.py \
  tests/test_rounded_die.py tests/test_manual_convex_policy.py

# 实际新对象预检命令；复现时替换输出路径，不能覆盖原结果
bash /home/brainco/reorientation/training/run.sh train \
  --num_envs 64 --iterations 2 --seed 42 \
  --actor_cost_weight 0.05 --critic_td_steps 8 --joint_velocity_norm_floor_rad_s 1.5707963267948966 \
  --adaptive_replay --wm_microbatch 2048 --actor_microbatch 4096 --critic_microbatch 8192 \
  --wm_cuda_graph --replay_storage cuda --video_interval 2 --video_steps 500 \
  --save_interval 100 --latest_interval 100 \
  --output_dir /home/brainco/reorientation/training/diagnostics/periodic_video_20261105/die50_100g_preflight
```

[validation.json](validation.json)保存范围、限制与源码哈希。
[环境差异](environment_diff.json)、[视频/状态验证](die50_100g_validation.json)、
[HOME评估](die50_100g_home_eval.json)、[训练预检](newton_training_preflight.json)、
[历史检查点续训录制](resume_hook_validation.json)均来自实际运行。
