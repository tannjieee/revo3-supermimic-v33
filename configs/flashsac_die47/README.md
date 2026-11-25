# FlashSAC 当前任务绑定

`env.yaml`逐字复制47g v32运行的环境；`task.json`保留同一运行的物理/观测/奖励/风险/噪声配置。
其中旧WM训练超参数只由复用Config解析，不参与FlashSAC更新。实际算法配置由
`viserdex.flashsac.agent.configuration()`解析并写入每个run的 `config.json`。

当前只验证1-step、FP32 eager、S0绝对位置＋发送前拒绝。Actor810D、Q观测963D。
单GPU64环境默认用于小闭环；1024环境/1M replay尚未验收。不要将此前20Hz149D实验
的配置或checkpoint复制进这里。真实验证见 `docs/versions/flashsac_hist25_20261106/README.md`。
