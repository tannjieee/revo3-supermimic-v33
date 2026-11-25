# 冻结 WM 特征实验与 ContactSDF 接入建议（2026-11-06）

状态：四组特征对照已完成；ContactSDF 尚未实现或训练。没有提升任何实验权重为生产策略。

## 1. 结论与适用范围

在固定 model1500、同一训练数据和200次更新预算下，仅加入现有几何代理、执行器特征或两者，
尚未显示稳定的8步物体旋转响应收益。三个相对baseline的方向改善区间均跨零。
这不能否定完整 ContactNets / ContactSDF，也不能证明网络容量不足。
测试仅11个旋转响应超过重复噪声阈值，后续必须补充可辨激励，避免用不可辨样本判断方向错误。

原50 mm/100 g骰子、fixed_palm_v1、随机目标、Newton MJWarp物理与随机化保持。
当前实验冻结Actor、Critic和target Critic，因此没有解决bootstrap主导，也没有新的策略成功率结论。

## 2. 可复现协议

证据根目录：`/home/brainco/reorientation/training/diagnostics/contact_wm_features_20261106`。
源检查点：`../wm_bootstrap_rootcause_20261106/current_v31_train/model_1500.pt`，SHA256
`78f40f3031f5402ff6b4eb1ac3a070d72020f937411b5540733f98d840e788e2`。

- 事前方案为证据根目录`plan.json`。真实采样seed50/51训练、52验证、53测试，每seed128域。
  训练2048起点/256域，验证512起点/128域，测试512起点/128域；同域不同时间起点不是独立环境。
- 每起点保存完整模拟缓冲/RNG快照，恢复后4个零扰动、2个正扰动、2个负扰动分支。
  单关节±0.5/1/2°，保持1或5控制步，总时域8步。基准动作固定为起点当前策略输出，属于开环响应。
  reset后后缀不进入训练，终止转移可进入训练；响应评估排除终止端点。
- 4组baseline/geometry/actuator/both，共用405维新特征槽和2048×405零初始化输入投影。
  初始预测与原WM逐位一致；名义参数数目相同，有效特征容量不同。
- 几何300维来自30个代理点：21个link包围盒中心、5个DIP表面点、4个掌面点。
  它们不是完整实际接触表面。支持面max也不是物体外部所有位置的精确欧氏距离。
- 执行器105维包括当前/旧目标误差、带偏置和力矩限幅的起点PD力矩、按15/40 ms延迟加权的力矩。
  delay_ready=0使用当前目标；此项只是固定起点近似，不是准确的多tick动力学积分。
- 新特征统计只由真实训练分支估计后冻结；原input/delta统计冻结。想象时几何由预测状态重算。
  不打开生产全局geometry开关，不改变Critic输入，也不增加几何损失。
- 原多步物理/任务损失，未加响应损失。batch64起点×3分支，Adam 1e-5，梯度裁剪1，200次更新。
  61/62仅改变minibatch顺序，不能称两个独立端到端训练seed。
- 验证0/50/100/200只作描述；事前固定最终200，完成全部8次拟合后才打开测试数据。
  Actor/Critic/target和原归一化逐组hash/逐位检查，新权重标为WM-only、不可恢复策略。

真实采集启动脚本`run_all_collect.sh`、`run_collect.sh`均保存在证据根目录。
离线复现（在仓库根目录；新输出目录须不存在）：

```bash
/home/brainco/reorientation/runtime/policy/bin/python scripts/supermimic/experiment_contact_features.py \
  --experiment /home/brainco/reorientation/training/diagnostics/contact_wm_features_20261106 \
  --output /home/brainco/reorientation/training/diagnostics/contact_wm_features_20261106/reproduction_new
/home/brainco/reorientation/runtime/policy/bin/python scripts/supermimic/analyze_contact_features.py \
  --experiment /home/brainco/reorientation/training/diagnostics/contact_wm_features_20261106 \
  --fits /home/brainco/reorientation/training/diagnostics/contact_wm_features_20261106/reproduction_new
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/brainco/reorientation/runtime/policy/bin/python -m pytest -q tests/test_contact_feature_experiment.py
```

## 3. 独立测试结果

合法且8步存活382起点；旋转可辨仅11起点/11域。可辨阈值为响应幅值>
10×max(重复噪声,数值floor)，噪声取6对零扰动和2对同号重复差分的最大值。
下表是可辨子集，幅值为预测/真实旋转差分范数比的中位数。

| 组 | seed61旋转余弦 | seed62旋转余弦 | 幅值比61/62 | 静止误报61/62 |
|---|---:|---:|---:|---:|
| baseline | .2241 | .2689 | .285/.364 | 52/55、53/55 |
| geometry | .1727 | .2670 | .331/.370 | 52/55、52/55 |
| actuator | .2536 | .2549 | .294/.323 | 51/55、51/55 |
| both | .1998 | .2705 | .284/.312 | 51/55、52/55 |

相对同seed baseline，两采样顺序平均旋转余弦差及物理域cluster bootstrap 95%区间：
geometry −.02665 [−.07528,.01966]；actuator +.00775 [−.06322,.07592]；
both −.01134 [−.12394,.10157]。这些区间只描述已采样域，不包含独立训练不确定性。

冻结原WM在本批测试旋转余弦.3407，幅值比.343，静止误报52/55。
不能拿.3407与旧审计−.0375直接比较：本轮采样seed、幅值、保持时间及可辨子集不同。
各组旋转相对向量误差约.768–1.785；预测零响应基线为1，幅值及误报均未解决。
最大单个真实旋转响应占总平方能量34.4%，同时提供去掉最大样本的敏感性分析。
完整h1/2/4/8、幅值、pulse/hold5、六物理组指标在`feature_ablation/result.json`；
聚类区间及对照在`comparison.json`、`comparison.md`。

## 4. 核验与失败记录

四个真实采集运行及8次WM拟合均正常退出；采集模型hash和优化器计数不变，拟合冻结断言通过。
初始预测max误差0；独立PD偏置/符号/限幅oracle max差5.96e-8，float32容差内。

保留以下失败与修复：

1. 初版collector把Adapter.step四返回值当两个解包，首个seed50采集失败；修复后从头采集。
   失败工件在`failed_initial_collector_unpack/`，未进入结果。
2. 初版PD oracle要求逐位一致，标量1.28与FP32张量舍入差导致失败；改用1e-7绝对/2e-7相对容差。
3. pytest自动加载无关ROS插件缺lark，使用禁用插件自动加载的隔离命令。
4. 四项合同测试揭示“所有环境已结束”时empty.split仍产出空chunk，使新特征统计出现NaN。
   修复为跳过空索引并检查最终统计有限。四项测试随后全部通过。
   实际训练统计49080条全部有限；用修复代码重算与原mean/variance/count逐位一致，
   因此原拟合结果保持有效，无需重训。详见`feature_statistics_fix_verification.json`。

没有修改生产RunningNorm或学习算法；用户先前console修改独立保留。

## 5. ContactSDF应怎样接入（设计，尚未执行）

原论文用几何SDF得到接触信息，再用速度空间SDF近似接触约束下的运动更新；
它包含动力学结构，而非仅拼接距离特征。其准动态假设适合慢操作，Allegro掌内实验主要是yaw目标。
本项目全SO(3)目标与当前高速失败轨迹需要额外验证，不能直接外推论文效果。
参考：[论文v2](https://arxiv.org/html/2408.09612v2)、[作者项目](https://yangwen-1102.github.io/contactsdf.github.io/)。

建议接口：`next_physical = contact_wm(state, absolute_target, context)`，保留现有55维物理输出、
Actor历史、目标、last_action、delay_ready及终止逻辑。Newton继续产生真实标签。
独立Torch模块先作为实验替代WM，后续才考虑进入策略训练。

### 5.1 几何与控制合同优先

- 复用已核验Revo3 FK及真实50 mm骰子碰撞凸包，不替换为理想长方体或旧70 mm几何。
  接触候选从真实手指/掌面外表面采样，排除内部凸包面；不同指间空隙必须保留。
  掌面候选必须覆盖现有托举支撑区，不能只保留几个中心点。
- 模块物理计算使用m、rad、s、kg、N、Nm。神经网络归一化只在残差网络边界使用。
  不将RunningNorm后的q/速度、×2输入缩放或旧单位带入接触公式。
- 从当前21维绝对编码器目标转换成内部几何位置误差：
  `u = target_encoder + encoder_bias - q_geometric`。这是内部变量，不改变Actor绝对位置协议。
  延迟未就绪使用当前指令；就绪后准确处理旧/新指令生效时刻、PD阻尼及1.28 Nm限幅。
  首先对齐25 Hz的40 ms控制步、200 Hz PD和15 ms延迟；加权力矩不能冒充子步积分。
- 广义速度包含对象6维及手21维，共27维；物理姿态积分保持当前XYZW及既有角速度坐标系。
  质量名义值100 g，训练域使用实际context；转动惯量从真实资产读取，不能只由边长和质量猜定。

### 5.2 原模型与适配模型分开做

首先建立准动态ContactSDF基线，用来量化原假设的上限。另立一个含速度状态、PD时序和有效惯性的
适配模型；若增加惯性项或自由运动预测，必须明确是混合模型，不能声称忠实复现原论文。

推荐结构：已知执行器/自由运动项 → 多接触约束 → 可微速度更新 → SO(3)积分。
残差网络先零初始化并限制容量，优先校正自由运动/有效参数；不要让任意姿态残差绕开接触约束。
接触同时影响手与物体，不能只对物体姿态加修正；固定掌座的外部反力允许系统总动量不守恒。

先固定已知几何与物理域参数；有效阻尼/惯性修正和两个平滑尺度采用正值参数化及有界范围。
物理真实质量与学习到的“有效准动态参数”分开记录，禁止用极端质量/摩擦吞掉执行器时序错误。
每个想象步由预测状态重算接触，不能把当前帧接触当成未来8步标签。

### 5.3 官方源码的具体迁移检查

已读取作者仓库commit `3a21edf8e51788f72a7d77259af4dd584d5ec43f`；只作为研究参考，未安装旧环境。
来源：[动力学](https://github.com/asu-iris/ContactSDF/blob/3a21edf8e51788f72a7d77259af4dd584d5ec43f/dynamics/joint_lse_dynamics.py)、
[接触检测](https://github.com/asu-iris/ContactSDF/blob/3a21edf8e51788f72a7d77259af4dd584d5ec43f/contact/sim_joint_lse_contact_detection.py)。

静态核查发现必须显式处理的差异：WXYZ/body角速度与当前合同不同；四元数Euler积分的归一化被注释；
速度SDF外层用硬max；接触筛选使用NumPy排序/阈值；部分法向通过带平移的齐次点变换。
后者会使方向依赖世界平移，应使用纯旋转。以上是源码核查，未运行作者完整控制器来量化影响。
另有接触距离×0.5、Jacobian行归一化常数和容量截断等实现约定，不能不加说明地移植。

我们需要平移/旋转等变性测试、法向和Jacobian有限差分、推动/分离方向测试、无接触自由运动测试、
掌内静止承托测试、空候选和容量溢出检查。自动微分需穿过接触点、法向及参数；硬筛选的梯度边界
要单独量化。几何和速度空间的平滑尺度分开校准；增加候选数会改变logsumexp偏差，需做密度敏感性检查。

### 5.4 下一组实验与准入

先补充HOME及低速接触区的成对真实分支，保留自然分布测试；训练可提高接触有效样本比例，
但必须报告选择规则和覆盖率。现有seed53已用于分析，新结构调参后应再留全新测试域。

建议同数据预算比较：原MLP；MLP+响应监督；准动态ContactSDF+响应监督；
含执行器时序/惯性的混合ContactSDF+响应监督。必要时追加混合模型无响应损失，隔离监督贡献。
当前200步特征实验只提供小预算基线；新训练预算应预注册，不能一边看最终测试一边选择。

响应监督比较同起点正/负动作的预测差与真实差，旋转使用统一参考坐标的SO(3)差分。
按各物理组训练尺度归一化，并用重复噪声定义可辨掩码；对真实静止分支另加虚假响应惩罚。
同时保留绝对状态损失，避免只学差分而丢失轨迹。

验收同时报告h1/2/4/8方向、幅值、向量误差、静止误报、穿透、速度/力矩范围、
自然分布和低速接触分布、域聚类置信区间、时间/显存。自动微分与模型自身有限差分相符
只证明实现梯度一致；仍需用真实模拟±动作验证方向。多接触近似也不保证严格无穿透或精确摩擦力。

只有WM方向及幅值跨新域改善后才进入Actor试训，且bootstrap独立做固定策略真实回报校准、
TD/EMA冷启动对照及有界尾值梯度实验。ContactSDF不能自动修正Critic尾值或改变gamma=.99、H8的长尾结构。
