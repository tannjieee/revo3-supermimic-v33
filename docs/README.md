# Documentation

[当前版本、权重与复现命令](../README.md) · [评估记录](evaluation/)

本发布分支包含最新老师、学生和预测器权重，以及 compact 非 SDF 手和 RSLCube。
安装、权重准备和当前已验证命令以主 README 为准；下表部分文档保留历史实验。

| Guide | Contents |
| --- | --- |
| [Newton 官方 v3.0.0](newton_migration.md) | Current runtime, official migration workflow and historical validation |
| [Isaac Sim 6 setup](isaacsim6_migration.md) | Version baseline, CUDA toolchain, dependency conflicts, Docker, migration checks |
| [Revo3 integration](revo3_migration.md) | Asset provenance, geometry sharing, joint order, actions, observations |
| [Pose training](revo3_pose_training.md) | Dynamic hand occlusion, dataset collection/audit, training and evaluation |
| [Student distillation](revo3_distillation.md) | Recurrent model, observation noise, DAgger, resume/export, RGB playback |
| [Object scanning](object_scanning.md) | D435i capture, metric reconstruction, object crop, 3DGS and asset import |
| [Training performance](training_performance.md) | RTX 5090 preset and checkpoint controls |
| [Independent GPU jobs](dsw_training.md) | Train different objects on separate GPUs |
| [Object preprocessing](../scripts/preprocessing_instructions.md) | Keypoints, splats, clustering and visualization |
| [Development](../CONTRIBUTING.md) | CPU tests, formatting, contribution workflow |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Code/data/robot asset terms |
| [Sim 5.1 history](revo3_sim51_history.md) | Previous compatibility baseline |

Historical validation results describe the recorded experiments and their limits.
The three released final checkpoints and selected evaluation reports are bundled;
large datasets and complete training logs remain external.
