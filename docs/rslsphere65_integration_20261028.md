# Revo3 RSLSphere65 接入与真实 Newton 预览（2026-10-28）

本次完成独立球体环境接入、有限仿真验证和原生 NewtonGL 截图。没有启动正式训练，没有加载旧方块策略。截图是保持初始关节姿态的物理仿真，并不表示球体旋转策略已经学会。

服务器 22031 新目录：`/home/zhangke/viserDex_supermimic_rslsphere65_20261028`。
独立 Git 分支：`codex/rslsphere65`。源快照提交 `70e75ecd041f45bb33c07be10713d912081222df` 包含旧项目当时的未提交源码，未复制旧 outputs/evidence/.git/虚拟环境。随后私有复制生成机器人资产和球体资产。没有切换父工作区分支。

本地交付目录：`/Users/kezhang/Visual-Tactile/experiments/revo3_supermimic_rslsphere65_20261028/`。本地包含基线归档、修改源码/测试、运行入口、日志、配置、JSON 证据和截图；Linux CUDA 运行环境仍在服务器。`baseline.tar.gz` 是最初源码快照归档，不能替代修改后的文件。

最终原始截图：`artifacts/final_preview_v2/sphere_step_0080.png`，1280×960，20 Hz 下第 80 个控制步（4 秒），原生 NewtonGL、headless、USD 视觉网格与纹理，未合成或裁剪。相机 eye=(-0.22,-0.24,0.78)，lookat=(0.02,0,0.565)。同目录保留其他时点、配置和验证结果。

## 实际修改

- `geometry_asset.py` 从实际活动 USD 碰撞体提取 Sphere 的球心、半径和均匀缩放，验证米制/Z-up，拒绝椭球变换及多碰撞体。保留方块解析分支用于旧算法单测；新 RSLSphere65 任务在 Adapter 中明确拒绝方块和非 65 mm 球体。
- `geometry.py` 增加精确球面有符号间隙 `||x-c||-r` 与外法向 `(x-c)/||x-c||`，中心采用确定的 +X 法向，避免除零。最近点等价于 `x-gap*normal`。保留原 30 个手部采样点和 300 维特征（每点相对位置3、相对速度3、间隙1、法向3），几何监督仍是原五组误差及原权重。特征不另行积分。
- 物体坐标相对速度继续使用 `R^-1(v_hand-v_object-omega_object × offset)`。球的几何距离旋转对称，但完整四元数、纹理朝向及角速度仍保留；WM 仍预测关节 q/qd、物体位置/旋转/线速度/角速度。
- `simulation.py` 使用独立任务 `Isaac-ViserDex-Repose-Revo3-SuperMimic-RSLSphere65-v1`。核验 Newton 实际加载的唯一碰撞体类型、半径、局部球心，核验实际随机化质量对应的 `0.4mr²` 球惯量。特权 context 继续读取实时质量、惯量、材质、驱动参数和关节限位，总维数548，物体材质仅统计物理碰撞体，视觉网格不混入。
- `objects.py` 注册独立 USD、GLB 和高斯路径；显式标记不支持当前相机位姿估计流程，没有虚构关键点或加载旧相机模型。PLY 保持原始 SH3、log scales、opacity logits、WXYZ 存储，未做额外缩放或轴变换。
- 新增 `run_sphere.sh`、`preview_sphere.py`、`validate_sphere.sh` 和球体测试。使用新源码路径优先导入，只读复用既有固定 Python 依赖，未执行安装、修改 editable/.pth 或共享依赖。缓存限定在新目录。

环境规范化差异审核 `evidence/semantic_diff.json` 证明：相对于旧 env20，动力学/奖励/事件/目标/观测合同仅改变 object USD 路径和 robot 私有副本路径。碰撞容量属于运行设置，设置为1536/512。初始中心仍为(0.02,0,0.565)，保留原位置/旋转/关节随机化、关节顺序、噪声协议、动作滤波和奖励。

Actor 仍为10×124观测历史；WM/Critic 仍使用特权状态与几何。网络宽度、WM监督窗口8、Actor展开8、WM更新10/轮、Actor1/轮、Critic20/轮保留。源 `Config.wm_only_iters=500`、Actor学习率1000轮渐增保持原值。唯一一轮联合更新诊断通过原有 `--check_joint_update` 临时把预热设为0，以执行所有网络更新；不是正式训练配置。

## 实测结果

运行遵循原固定官方 IsaacLab v3 Newton 工作流：`SimulationRuntime` → 官方 preset/`launch_simulation` → ManagerBasedRLEnv。IsaacLab仓库HEAD为 `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`。实际包版本：Isaac Sim6.2.0.0、Newton1.6.1、Warp1.17.0；此固定发行工作区的 isaaclab 包元数据版本为28.0.0（与原入口的精确版本检查一致）。后端为 `NewtonMJWarpManager` / `newton_mjwarp`，物理dt=1/120，2内部子步，decimation=6，控制20Hz，回合20秒。

先读取了源 AGENTS.md、迁移说明和固定发布工作区的资产准备/MJWarp调优文档。启动前 GPU1 为661MiB、0%利用率，只有必须保留的查看器PID1900441；所有新GPU进程均限定物理GPU1，未停止其他进程。

| 验证 | 结果与证据 |
|---|---|
| CPU解析/几何/算法回归 | 最终66项通过，含6项新增球体检查；`evidence/unit_tests_final.log`。检查内外距离/中心/最近点、法向、gradcheck、四元数坐标变换、旋转相对速度有限差分、监督梯度与实际USD30点提取。 |
| 单环境连续物理 | 441控制步，混合保持/零/随机动作，3次重置，无非有限值或容量溢出；`artifacts/preview_v2/validation.json`（服务器保留），完整日志在`evidence/preview_v2.log`。 |
| 16环境物理 | 每环境441步，合计7056次环境步，50次重置，强制超时路径通过，溢出标志始终为0；`artifacts/stability16_v1/validation.json`。候选峰值14297和接触峰值9979为16环境合计，约束行峰值186为每环境，不能把总量与单环境nconmax直接比较。 |
| 球质量/惯量 | USD质量0.07kg，解析r=0.0325m，I=2.9575e-5 kg·m²；原0.8–1.2随机化保留。16环境实测质量0.0589582–0.0835094kg，惯量与实时质量的0.4mr²最大误差3.64e-12 kg·m²。 |
| 真实碰撞体 | Newton类型3=SPHERE，唯一物理形状，scale=(0.0324999988,0,0)，半径误差0，球心核验通过；视觉Mesh没有CollisionAPI。 |
| 完整网络联合更新 | 最终16环境、从零初始化、1轮，WM/Actor/Critic=10/1/20实际优化步；保留生产批次2048/4096和8步展开，BF16 MLP、FP32 eager几何、微批512/512/4096。网络更新约12.38秒，WM/Actor/Critic loss=1.66190/-0.208956/0.00396175，梯度有限；`artifacts/full_update_final/`。诊断还完成32步留出预测，误差不表示已收敛。 |
| 状态/奖励协议 | 联合更新最大FK位置误差4.69e-7m、奖励误差5.59e-9、观测重建4.77e-7；成功边界判定分歧0，终止前采集与重置分离通过。 |
| 最终源码截图复核 | `artifacts/final_preview_v2/validation.json`，81步（80步保持+1步强制超时），再次通过球形、球心、惯量、路径与状态核验；第80步截图已在本地使用view_image实看。 |

原高度的固定seed42单环境在20步后稳定，80步球心约(-0.001821,0.002145,0.547577)m，线速度约6e-6m/s；160步保持时仍稳定。全场接触距离最小值从最初约-1.42mm收敛至约-0.191mm；这是包含手自碰撞的诊断下界，不是独立球—手最大穿透证明。16环境混合动作也无容量/数值异常，因此没有凭外观修改生成高度、奖励或求解器参数。随机动作阶段存在掉落，属未训练环境的实际行为。

OpenGL/CUDA互操作报告 `invalid device ordinal` 并自动回退到Warp数组与OpenGL缓冲区的拷贝传输。最终原生图片有效，物理和数值断言通过；不宣称零警告或零拷贝。早期预览入口曾修正CLI字段名及preset解析前访问solver配置的顺序问题，另一次回归检查修正了球体单位检查对旧方块合成测试的影响；失败日志保留，成功结论使用上表最终结果。

## 入口（在31服务器执行）

先重新只读检查GPU占用，再选择有余量的GPU。下面的1是本次验证时的空闲物理GPU编号，不能假定未来一直空闲。

```bash
cd /home/zhangke/viserDex_supermimic_rslsphere65_20261028
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sphere.sh scripts/supermimic/preview_sphere.py \
  --viz newton_gl --device cuda:0 --num_envs 1 --steps 80 \
  --output_dir artifacts/my_preview
# 只执行一轮完整联合更新的受限诊断入口；输出目录必须不存在。
CUDA_VISIBLE_DEVICES=1 bash scripts/validate_sphere.sh artifacts/my_one_update
```

`run_sphere.sh` 选择独立球体注册及固定依赖，预览默认绑定`evidence/env_sphere20.yaml`。运行时 `viserdex.__file__` 与几何模块路径均核验为新目录；Python解释器本身来自旧固定依赖环境，只读复用。

新高斯SHA-256为`39c1e2c801bfa38eda4d2cba5027709a56acf6e01f47a1cab8cda3a36a93b6da`，GLB为`32318829419a97770424eb1327607d40cfbd1d40d2b99a2674fd12bb913b9377`，USD为`13e2b4db5eea05b11c671852c01efeecc27bb86993648c19f9289a2e76435623`。完整复制资产哈希见`evidence/asset_hashes.json`，环境合同为`3150f1bb722c4cfbb5d5f9958e07510dce2aed37da26efd7dc7da2984c86f6c5`，几何摘要为`8c6ff62c6ae2f98c10249b2275f87c368d9a405655be1cc45b97a95f91acfaa8`。

## 保护和未验证范围

`evidence/protection_audit.json` 对旧实验810个非缓存、非.git、非符号链接的常规文件逐项比较前后SHA-256，变化为0，包含旧源码、日志、运行配置和最终权重。旧`outputs/revo3_horizon20_2x2048_scratch_seed42/latest.pt`仍为`bad3092ec1d151d870d7a3b412abf02d8d34d2aca4d9dc1e987a578e06855d38`。PID1900441从Oct21启动的原进程仍在；未关闭或更改SSH master socket，未访问21服务器，未改其学生训练。新验证进程结束后GPU1恢复661MiB/0%状态。

本次没有验证双卡4096环境或长程接触峰值、长期学习收敛、球体旋转策略效果、真机、相机训练/3DGS渲染或编译几何的生产执行性能。摩擦和质量为仿真配置值，不是实测材料标定。诊断`latest.pt`仅是新球体一轮检查产物，带`check_joint_update`和`performance_diagnostic`元数据；不能作为已训练策略。旧方块权重未加载，正式训练未启动。
