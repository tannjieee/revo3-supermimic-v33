# RSLSphere65：65 mm 六向彩色圆形特征球

当前设计：青绿色底色 RGB (63,149,157)，六向圆形色块半径角 38°，相邻色块净角距 14°。
方向与 65 mm 打印贴纸一致：+X=绿色 E，-X=红色 T，+Y=浅蓝 H，-Y=深蓝 R，+Z=紫色 S，-Z=黄色 L。

## 文件与参数

- `RSLSphere65.usda`：推荐的仿真入口，与 `textures/` 一起使用。标准 USD Physics 刚体、质量、球碰撞、接触材质；纹理用相对路径。
- `RSLSphere65.usdz`：打包了模型和纹理的单文件 USD 版本。
- `RSLSphere65.glb`：内嵌纹理的视觉模型；不包含物理参数。坐标保留 USD 的 Z-up，通用 GLB 查看器可能需要调整朝向。
- `preview.png`：设计预览，不是 Isaac Sim 渲染截图。
- `build_asset.py`：用 Python + NumPy + Pillow 从 `source_textures/` 重建 USDA、GLB 和纹理。USDZ 需另用 OpenUSD 打包。
- `asset.json` / `validation.json`：参数和静态验证结果。

单位：米；直径 0.065 m；质量 0.070 kg（用户选定，并非实测）。原点位于球心，无额外缩放。
视觉网格 49,152 个三角形；物理碰撞仅使用一个 `Sphere`，半径 0.0325 m。
惯量假设均匀实心球，Ixx=Iyy=Izz=0.000029575 kg·m²。若实物是薄壳或中空球，应按实际质量分布调整惯量。
静/动摩擦初值 0.8/0.6，恢复系数 0；这些数值未经过实物标定，任务随机化也可能覆盖它们。

## 在 Isaac Sim 中导入

将 `RSLSphere65.usda` 作为 USD reference 加入场景；保留整个 textures 目录。
资产已有刚体和碰撞 API，不必再对六个视觉 Mesh 添加碰撞。按米导入，缩放为 (1,1,1)。
给场景添加物理环境/地面，并将球心初始位置设置到地面上方至少 0.0325 m。
若使用 USDZ，则纹理已随文件打包。

## 接到本仓库的教师任务

本资产没有覆盖默认 RSLCube，也没有把缺少的 3DGS、关键点数据伪装成已有数据。
状态观测教师任务可通过覆盖物体与目标预览 USD 路径接入。以下命令需先准备 README 中的官方运行环境，
是建议的短程验证入口，本机没有执行。请从仓库根目录运行：

```bash
SPHERE_USD="$PWD/outputs/assets/RSLSphere65/RSLSphere65.usda"
bash scripts/run_newton.sh isaaclab train --rl_library rsl_rl \
  --task Isaac-ViserDex-Repose-Revo3-V1 \
  --num_envs 4 --max_iterations 2 --viz none --device cuda:0 \
  physics=newton_mjwarp \
  "env.scene.object.spawn.usd_path=$SPHERE_USD" \
  "env.commands.object_pose.goal_pose_visualizer_cfg.markers.goal.usd_path=$SPHERE_USD"
```

将资产移到别处后同步修改 SPHERE_USD。保持默认物理预设、任务奖励和域随机化。
运行后需核对后端实际导入的碰撞形状为球、半径 0.0325 m、质量约 0.07 kg（考虑任务随机化），
并确认无接触/约束容量警告。原方块的初始抓握与位置未针对球调优，需要检查穿透和稳定接触。
这只是物体替换，并非已训练好的转球策略。

## 验证范围

通过：OpenUSD 26.11 解析、USD/USDZ 依赖解析、刚体/质量/球形碰撞 schema、视觉网格半径与材质绑定。
解析器装在临时目录，没有修改项目锁定的仿真依赖。
本机未配置 Isaac Lab / Isaac Sim / Newton 运行时，未验证实时碰撞、重力下落、接触稳定性或训练。
未提供球的 3DGS / 位姿预测器 / 学生策略；旧方块权重不可据此视为适配球体。
照片中的凹槽、打印接缝和表面粗糙细节未建模；这是理想光滑球体。

## 纹理来源

颜色/字母源自 ETH Zurich RSL 的 leggedrobotics/ViserDexSplats，沿用本仓库标注的 CC BY-NC 4.0。
来源：https://huggingface.co/datasets/leggedrobotics/ViserDexSplats
SOURCE.json 为原始数据来源清单；其中摘要对应 source_textures 中的原始纹理，不对应重新烘焙的 textures。
本资产修改了几何、纹理布局和底色；没有更改原始源图。非商业与署名要求仍适用。
