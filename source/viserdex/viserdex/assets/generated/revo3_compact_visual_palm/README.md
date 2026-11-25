# Revo3 少碰撞块版本：271 → 116

2026-10-21，服务器 22031，项目 `/home/zhangke/viserDex_revo3`。

## 已完成的改变

在上一版 visual 尺寸掌面和已裁剪拇指根部的基础上，只合并掌面的凸块，全手从 **271 块降到 116 块，减少 57.2%**。仍使用 Newton 凸包碰撞，不使用 SDF。

| 部位 | 之前 | 现在 |
|---|---:|---:|
| 掌面硬壳 | 64 | 10 |
| 掌面橡胶 | 64 | 13 |
| 掌面触觉层 | 64 | 14 |
| base / LED | 2 | 2 |
| 拇指和其他手指 | 77 | 77 |
| 全手 | **271** | **116** |

掌面的三层合计由 192 块降至 37 块。其余 **79 个碰撞体的实际运行时顶点与上一版完全一致，最大误差为 0**，包括已经裁剪好的全部拇指根部；没有进一步削减、删除拇指碰撞体。原 visual、22 个刚体的质量/惯量、21 个关节及其限位和驱动参数均保持一致。原始 URDF/USD 和正在使用的 SDF 训练配置未覆盖。

## 合并约束

逐层合并凸块，同时限制外轮廓变化，并保护拇指 CMR/MCP 在根部关节范围内扫过的三维空间。保护空间来自上一版实际编译的模型：43×45 个根部姿态、0.5 mm 体素采样，保留约 147 万个采样点。采样用于筛选合并候选，最终再用实际碰撞模型独立验证。

最初只保护二维轮廓的候选填充了三维内部空隙，根部扫描失败；该候选已作废，归档在 `rejected_naive_merge/`。自动裁剪产生的大量根部删除也一并作废，未安装或用于训练。最终版本保留了上一版根部全部顶点与裁剪数据。

实际 Newton 加载后的尺寸测量：

| 掌面层 | 外包围尺寸最大偏差（相对原 visual） | YZ 投影额外面积 | YZ 投影缺失面积 |
|---|---:|---:|---:|
| 硬壳 | 0.01459 mm | 0.4568% | 0.0067% |
| 橡胶 | 0.00069 mm | 0.6629% | 0.0023% |
| 触觉层 | 0.00087 mm | 0.6568% | 0.0036% |

外尺寸与拇指缺口保留。合并会让部分曲面和内部空腔变成更保守的凸包近似；不能视为逐点几何完全等价。上表只衡量尺寸和指定投影轮廓，三维运动间隙由下述接触测试补充验证。

## 验证结果

官方 Isaac Lab v3.0.0、Isaac Sim 6.2.0.0、Newton 1.6.1、Warp 1.17.0；使用项目公开预处理入口 `scripts/preprocessing/prepare_revo3_newton.py`、`launch_simulation` 和 `physics=newton_mjwarp`，未修改引擎或自碰撞过滤规则。`palm_visual_` 与 `root_trim_` 保留完整凸包顶点，其余原碰撞体继续原来的导入处理。

- 密集根部静态扫描：CMP 0～1.92 rad、CMR -0.4224～1.4277 rad，191×193＝**36,863 个姿态**，MCP/PIP/DIP 为 0；根部与掌面负距离有效接触为 **0**，无额外根部裁剪。
- 实际电机动作：**25 秒、3000 物理步、120 Hz、每步 2 个子步**。初始状态之后通过 PD 目标驱动，没有逐帧瞬移关节；测试对象移离，其他手指目标为 0。
- 前 15 秒 CMP/CMR 往返动作：拇指相关有效自接触为 0；全部 25 秒 CMR/MCP 根部有效接触也为 0。CMP 实际达到 **109.870°**，CMR 实际范围 **-24.193°～81.782°**。
- 完全屈指仍保留 DIP 指尖触掌，对指动作仍保留指尖与无名指接触；动态最大穿入约 **0.903 mm / 0.580 mm**。这不是所有手指任意组合姿态无碰撞的证明。
- 使用相同动作轨迹，记录到的求解器接触数峰值由上一版 **1599 降至 753**。这是接触数统计，不代表已测得同等幅度的速度或显存改善。
- 完整动作和截图日志无容量溢出、NaN 或 traceback。验证容量为 `nconmax=3072`、`njmax=4096`；它们只经过单环境验证，不是批量训练容量保证。
- 22 个质量 API、21 个转动关节的属性比较通过；包中 URDF 在删除 collision 并按文件内容归一化 mesh/texture 引用后，与原始 URDF 的其他内容完全一致。

未测试本资产的 4096/8192 环境训练速度、显存和收敛；训练配置未切换。

## 使用与文件

独立安装的 USD：

`source/viserdex/viserdex/assets/generated/revo3_compact_visual_palm/revo3_right_compact_v4/revo3_right_compact_v4.usda`

完整资产包 `revo3_compact_bundle.tar.gz` 包含：

- `input/urdf/revo3_right_compact_v4.urdf`：相对路径引用，包含所需 mesh 与纹理。
- `converted/`：USD、payload、几何和纹理。
- `source_reference/`：原始输入参考。
- `capture/`、`motion/`：实际 Newton GL 截图与电机动作视频。
- `validation/`：密集扫描、动作、力学参数、尺寸审计、失败候选原因和日志。
- `reproduce/`：合并脚本、上一版分解输入、根部保护点云和当前凸块。`simplify_palm.py` 可复用提供的点云重建合并结果；其余部分脚本保留服务器路径，迁移时需调整。
- `manifest.json`：文件 SHA256 清单和安装路径。

在项目官方环境中，用 `scripts/run_newton.sh` 包装以下公开预处理入口：

```bash
scripts/preprocessing/prepare_revo3_newton.py \
  --convex_urdf <bundle>/input/urdf/revo3_right_compact_v4.urdf \
  --usd_dir <output> \
  --exact_hull_prefix palm_visual_ --exact_hull_prefix root_trim_ \
  --convex_joint_stiffness 7 --convex_joint_damping 0.22 \
  --viz none --device cuda:0
```

原始 URDF SHA256：`c4eae42fddedfd02b6a2d15699fc21a3173f5ae3d5a164b585178a307f6173ca`。
