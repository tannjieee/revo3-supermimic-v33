# Isaac Sim 5.1 历史兼容说明

当前支持基线为 Isaac Sim 6.0.1 / Isaac Lab 3.0；请使用
[现行安装说明](isaacsim6_migration.md)。本页仅保留早期适配的技术背景。

早期 Revo3 移植使用 Isaac Sim 5.1 / Isaac Lab 2.3.2。该版本曾需要动作关节
排序兼容层，并尝试用 Storm 显示副本处理特定驱动下的 RTX 初始化问题。
这些兼容代码已经移除，不是当前训练和回放入口。

迁移到 Sim 6 后，四元数约定采用 XYZW，RSL-RL 采用新的 actor/critic 配置与
检查点格式。Sim 5.1 的教师检查点不能直接作为当前任务检查点；应在
`viserdex_revo3_sim6_ppo` 命名空间重新训练。原始 3DGS PLY 的 WXYZ 文件格式保留。

旧实验的日志与模型不随代码分发。早期短程训练、显示诊断和资产测试仅用于确认
当时的运行链路，不作为当前环境兼容性或真机性能的证据。
