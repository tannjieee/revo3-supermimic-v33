# 多 GPU 独立物体训练

> 此文保留旧 Sim 6.0 / beta PhysX 基线的安装或测量记录。当前 v3.0.0 Newton 安装、默认值和启动命令以 [Newton 迁移说明](newton_migration.md) 为准。

使用 `scripts/train_revo3_object.sh` 在不同 GPU 上各训练一个物体。
每个进程通过 `VISERDEX_OBJECT` 选择资产，控制和渲染设备绑定同一张 GPU。
这里是相互独立的训练任务，不进行分布式梯度同步。

## 准备

- 按 [安装说明](isaacsim6_migration.md) 配置 Sim 6 环境。
- 从仓库根目录运行；`run_sim6.sh` 支持 `.venv-sim6/bin/python` 或 `VISERDEX_PYTHON`。
- 使用 `scripts/download_data.py` 下载相应物体数据。
- `requirements/dsw-runtime-freeze.txt` 仅为历史运行环境清单，包含其他项目依赖；不要将它作为新环境安装入口。

## 启动

下面两条命令在两个终端分别运行：

```bash
bash scripts/train_revo3_object.sh Duck 0
bash scripts/train_revo3_object.sh Globe 1
```

默认各使用 8192 个环境、5000 轮，每 100 轮保存一次。确保每张卡的显存及整机内存足够。
可在命令末尾覆盖训练参数，例如先做小规模检查：

```bash
bash scripts/train_revo3_object.sh Duck 0 --num_envs 64 --max_iterations 5
```

检查点位于 `logs/rsl_rl/viserdex_revo3_sim6_ppo/<object>_gpu<index>_<timestamp>/`。
日志、运行环境与下载的数据均被 Git 忽略。后续位姿训练和蒸馏必须使用同一物体的检查点。

同一个项目中的多进程可以独立选择不同物体；运行期间不要修改共享资产或源代码。
