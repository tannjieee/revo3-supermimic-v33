# RSLSphere65 Mesh → 3D Gaussian Splat

从用户提供的 `RSLSphere65.zip` 中的 `RSLSphere65/RSLSphere65.glb` 拟合。
保留原始 mesh、六组纹理、USD 与源数据记录，不执行附件中的构建或训练命令。
高斯来自该 mesh 的合成多视角图像，不包含实拍反射、表面接缝或实物扫描细节。

## 使用结果

- `rslsphere65_gaussians.ply`：50,000 个高斯，三阶球谐，符合 ViserDex 的 PLY 字段约定。
- `rslsphere65_gaussians.npz`：同一套高斯的 NumPy 参数。
- `RSLSphere65/RSLSphere65.glb`：原始视觉 mesh，与高斯对齐。
- `RSLSphere65/RSLSphere65.usda` / `.usdz`：原始物理资产，保持原样。
- `mesh_vs_gaussians.mp4`：左 mesh、右高斯的环绕对比。
- `six_axis_comparison.png`：六个坐标方向的 mesh / 高斯对照，检查字母和朝向。
- `final_comparison.png`：六个未参与训练视角，依次为 mesh、高斯、绝对误差×4。
- `metrics.json`、`evaluation/final_per_view.json`：测试指标。
- `viserdex_verification.json`：真实 ViserDex 加载方法的独立执行和渲染验证。
- `geometry_orientation_verification.json`：原坐标、六向结果和高斯中心半径分布。
- `asset_manifest.json`：输入 SHA256、单位、坐标变换、训练数据设置。
- `reconstruct.py`、`verify_viserdex.py`、`diagnostics.py`：完整生成与验证代码。

原压缩包内 `RSLSphere65/README.md` 与 `asset.json` 保留了生成高斯之前的记录；其中“没有高斯数据”描述的是原始包，不是本次输出状态。以本文件和本次验证结果为准。

## 尺寸与坐标

不做尺寸归一化、重定位或轴变换。米单位，Z-up，球心在原点，mesh 直径 0.065 m。
高斯和原 mesh/USD 使用相同坐标；加载原 USD 后，高斯应复用同一刚体位姿。
无需对高斯增加 65 倍、1000 倍缩放或额外绕轴旋转。

六个方向的设计对应关系沿用输入资产：+X=E（绿），-X=T（红），+Y=H（浅蓝），-Y=R（深蓝），+Z=S（紫），-Z=L（黄）。
这是外观表示。碰撞与质量仍使用原 USD 中的球体物理模型；高斯中心和椭球尺寸不等价于碰撞边界。

## 参数约定

| NPZ 字段 | 形状 | 含义 |
| --- | --- | --- |
| means | 50000×3 | 高斯中心，米 |
| scales | 50000×3 | 各主轴标准差，已 exp 激活，米 |
| quaternions_wxyz | 50000×4 | 单位四元数，wxyz |
| opacities | 50000 | 已 sigmoid 激活的不透明度 |
| sh_coefficients | 50000×16×3 | 三阶球谐系数 |
| log_scales | 50000×3 | 尺度的自然对数 |
| opacity_logits | 50000 | 不透明度的 logit |

PLY 字段：`x/y/z`、`nx/ny/nz`（零）、`f_dc_0..2`、`f_rest_0..44`、`opacity`、`scale_0..2`、`rot_0..3`。
其中 `opacity` 保存 logit，`scale` 保存 log 尺度；`f_rest` 按 RGB 通道分别排列，每通道 15 个系数。不要把这些原始字段直接当成已激活参数。

```python
import numpy as np
g = np.load('rslsphere65_gaussians.npz')
positions = g['means']
scales_m = g['scales']
rotations_wxyz = g['quaternions_wxyz']
opacity = g['opacities']
sh = g['sh_coefficients']
```

## 生成与验证范围

分别保留六个 mesh 的纹理材质和节点变换；不合并为单材质。
在球面分布渲染 180 张 512×512 RGBA 图，150 张训练、30 张测试，相机内外参精确已知。
渲染使用纹理/基础颜色的 FLAT 模式，没有额外真实环境光照或镜面反射监督。
按 mesh 表面积采样 50,000 个高斯，以面法向初始化各向异性尺度和旋转，以对应纹理初始化颜色。
用 gsplat 优化位置、尺度、旋转、不透明度和 SH，共 12,000 步；随机背景，颜色 L1、alpha L1 和弱位置先验；没有动态增密。
测试 PSNR 使用白底；前景由 mesh alpha > 0.5 定义，轮廓 IoU 以两者 alpha > 0.5 计算。指标是测试视角的算术均值。

30 个测试视角的结果：整图 PSNR 34.19 dB，物体前景 PSNR 31.32 dB，轮廓 IoU 0.99724。
初始化时对应为 22.39 dB、19.92 dB、0.98071。
PLY 保存后重读的最大渲染绝对差为 4.17e-7；实际 ViserDex 加载方法读出的参数与导出参数一致。
六个正轴视角单独检查，前景 PSNR 为 31.24–34.50 dB，字母方向保留。

在服务器上执行了真实 ViserDex 的 `_load_gaussian_splat_model` 函数（通过 AST 独立提取，避免启动 Isaac Lab），并用加载参数渲染验证。尚未测试完整 Isaac Lab 环境或视觉策略训练；未生成位姿网络、关键点和颜色聚类文件。
合成视角拟合质量不代表真实相机效果或实机控制成功率。

## 原服务器复现

目录：`/home/zhangke/mesh_to_gaussians_rslsphere65_20261028`，SSH 端口 22031，物理 GPU 1。
使用上一实验的独立 Python 环境；没有修改共享 ViserDex 包或原始资产。

```bash
cd /home/zhangke/mesh_to_gaussians_rslsphere65_20261028
PYTHON=/home/zhangke/mesh_to_gaussians_avocado_20261028/venv/bin/python
CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 "$PYTHON" -u reconstruct.py --steps 12000
CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 "$PYTHON" verify_viserdex.py
CUDA_VISIBLE_DEVICES=1 EGL_DEVICE_ID=1 "$PYTHON" diagnostics.py
```

这些命令会重写生成结果。换机器需要可用的 PyTorch/CUDA、gsplat、numpy、trimesh、scipy、pillow、pyrender、PyOpenGL 及 accelerate 3.1.10、plyfile、imageio 和 imageio-ffmpeg。
脚本优先复用本服务器对应 PyTorch/CUDA 的已编译 gsplat 缓存；在没有该缓存的机器上，使用常规 gsplat 安装/编译路径。
原服务器另保留 `images/`、`dataset.npz`、`checkpoint.pt` 和初始化文件。

## 来源

用户提供的 `SOURCE.json` 和原 README 记录纹理源自 `leggedrobotics/ViserDexSplats`，沿用其标注的 CC BY-NC 4.0。原来源、修改记录和署名随输入资产一起保留。
