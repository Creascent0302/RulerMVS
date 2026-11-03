# V6 源码结构审计：V8 的三个小改动候选

日期：2026-11-03。范围：真实 `overnight_v6_v7_mvs_20261103/train.py:41` 的 V6 入口，沿 V6→V5→V2TrainingModel→PretrainingModel→CalibratedDecoder 跟踪。已读 research_memory 顶部、162–173、221–223、246、247、252、254；未改生产代码、权重、训练配置或 research_memory，未使用 GPU。`audit_math.py` 已实际以 CPU 运行，产物 `audit_math.json`；这是构造和代数验证，不是真实精度实验。

结论：优先诊断 source/null 的合法源数量依赖；随后用对数深度导数替换带内部长度单位的敏感度。范围归一位置编码与固定内部候选步距构成第三类表示依赖，但应在独立训练分支验证，不能直接切换旧权重后声称架构优劣。没有发现可直接解释全部外测过浅误差的确定实现 bug。

## 真实前向与维度

| 环节 | 实际张量 / 运算 | 代码 |
|---|---|---|
| RGB / matching | 主图 256×448，matching 512×896，后一支只是共同 RGB 插值 | overnight 配置 `configs/v6.json` |
| 编码 | 4 层 ViT 各 768×16×28，adapter 各到 128；matching FPN 4/8/16 各 64 通道 | `jar/pretraining/encoders.py`；`jar/pretraining/model.py:123` |
| 粗候选 | 96 个 logZ 刻度，32×56；reference/source L2 归一化 | `jar/pretraining/model.py:118,131,201` |
| 每源 pair | 64 product + 64 abs-difference + 8 group correlation + 3 geometry = 139，MLP 139→64→64 | `jar/pretraining/model.py:18–100` |
| 跨源 | 4 头，每头 16 维；source 与单个 null 在源轴竞争 | `jar/pretraining/model.py:85–100` |
| 候选入口 | pooled64 + null1 + log-information1 → evidence64；加位置 MLP；候选 query 读取 4 层单目 bank；沿 C=64 做 LayerNorm | `jar/pretraining/model.py:245–264` |
| 粗上下文 | CandidateScaleSpace 保留全部 D=96，空间两次降采样，不跨深度降采样 | `jar/models/ruler_spatial.py:44–82` |
| 粗读出 | score head + 4×带 null 衰减的 weighted cosine；有界连续偏移 | `jar/models/ruler_direct.py:371–386` |
| 子射线 | 同深度格四邻域概率运输后选 4 峰，每峰 3 点；fine 为 12×64×112 | `jar/models/ruler_proposals.py:85–124` |
| 细判断 | 再投影、同 source/evidence/position/score；继承粗特征；V6 父族意见池、族内均值，winner 特征引导凸上采样 | `research/rulermvs_v6_20261101/model.py:34–91`；`handoff.py:85–149` |

CandidateChannelNorm 只沿每个候选的通道归一化，**没有**跨候选、跨像素全局统计污染（`initial_surface_memory.py:39–45`）。CandidateScaleSpace 的池化除以有效 mask 质量。Matching BN 已固定 eval，不能把“BN 统计训练/测试混用”作为本轮新原因。候选 valid 采用参考图有效性而不是至少一源有效，这让无匹配位置仍输出单目先验；它是当前完整覆盖契约，不能通过删掉这些像素改善评价。

## 1. 源数量改变匹配与单目之间的先验配比

**确定的结构事实。** `jar/pretraining/model.py:91–100` 直接执行

\[
w_0=\frac{\exp\ell_0}{\exp\ell_0+\sum_{v=1}^{m}\exp\ell_v}.
\]

没有合法源数校正。各源 logits 相同时，复制已有证据会改变 null、pooled、information 和 `model.py:262–263` 的 cosine。最终 `ruler_direct.py:373` 将该 cosine 再乘 4。这不是 LayerNorm 能完全消除的统一幅度变化：null、metric、位置/单目支路和评分旁路并非同倍缩放。

**已实际 CPU 验证。** 调用当前 CandidateViewAttention，使同一归一描述子、同一几何的 source/null logits 都为 0：

| 相同有效源份数 | null | pooled 各通道幅度 | 4cos 旁路 |
|---:|---:|---:|---:|
| 1 | 0.5 | 0.5 | 2.0 |
| 3 | 0.25 | 0.75 | 3.0 |
| 7 | 0.125 | 0.875 | 3.5 |

因此两图更依赖单目/位置先验、八图有更强匹配幅度，是可成立的架构效应；**尚未证明它导致 254 的精度差或过浅方向**。真正增加独立正确视图可以提高可信度，不能把所有视图数量效应都当错误。

**最小可插拔改法。** `null += beta * log(max(m,1)/m0)`，m 按每候选 `source_valid.sum(view)`，m0=3 保留常规四图完整有效位置原函数。beta=0 为旧模型，beta=1 为 log-mean-exp 源池；也可只训一个 beta，但必须先固定其实验语义。m=0 仍由唯一 null 承担全部质量。无新增投影、特征采样、卷积或源值张量；额外为一次源 mask 归约和一个候选标量 log/add，实际耗时未测。

这保证整套源集合等倍复制时总证据幅度不变，**不**保证任意复制多源集合中某一源都不变，也不是遮挡认证。与162/163不同：不在深度轴减 logmeanexp，不改变绝对 cosine，也不对截断 fine 候选重新定义背景；与164不同：不加可见性 loss。

**建议诊断顺序。** 在 TRAIN/VAL 固定图像和参考，比较真正 N=2/4/8 与同一已有源集合复制到对应 N；记录 coarse logits、null、cosine、coarse 深度、top4 oracle、fine/末端，先区分“新证据收益”和“纯数量增益”。再固定权重只改 beta=1 做机制 probe；即便有收益也须与同预算 control、后续适配分开报告。不要用 ETH/Tanks 反复选 m0/beta。

## 2. 绝对敏感度与内部长度单位绑定；用对数深度导数修正

**确定事实，且并非首次发现。** `jar/pretraining/model.py:98,246` 使用

\[
I=\sum_v\bar w_v\|g_v\|^2,\qquad \mathrm{metric}=\log(1+I),\qquad g_v=\partial u_v/\partial\tilde Z.
\]

外参平移与候选都除以域尺度 s（V5 `model.py:83–85`，PretrainingModel `model.py:312–317`）。保持物理场景不变但 s→a s 时，\(\tilde Z\to\tilde Z/a\)、\(g\to ag\)，I→a²I。pair 几何里的 \(\|g\|h\) 则保持不变。246 和252已记录这一性质；246的 scale=.5/2未恢复旧 courtyard，不能宣传换一个单位常数即可解决泛化。新247已经有真实单位增强，同样不能把“增加单位增强”当新V8。

**已实际 CPU 验证。** s=.5/1/2 时相同物理 pair 的 cell span 恒为.5，旧 I=.5/2/8、log1p(I)=.4055/1.0986/2.1972；\(\tilde Z^2 I\) 恒为200。

**最小改法。** 替换同一个66维 evidence 输入中的 metric 为

\[
I_{\log Z}=\tilde Z^2 I=\sum_v\bar w_v\|\partial u_v/\partial\log Z\|^2.
\]

这同时保留同一物理候选的图像视差杠杆，不依赖内部长度单位；平行双目时 \(Z\,|g|=fB/Z\)。与 \(h^2 I\) 相比，它不随候选格宽/粗细区间变动而直接重定义量纲。代价是候选标量逐元素乘法，无新参数和新投影。应在 `_tokens_from_source` 接入，以保持 source API 与其它调用方兼容。

**限制。** 这只修复敏感度通道，不保证整网尺度等变：`model.py:197` 的绝对 J 阈值，`ruler_proposals.py:99,115–116` 的固定内部峰距/步长限幅，仍会依赖单位。若要严格单位契约，分别改为无量纲 JZ 阈值、logZ 峰分离与相对深度限幅；这些会改变候选供给，应单独实验。旧模型对 I 的数值分布已经适配，直接替换是输入变换 probe，不能由其零样本升降认定新架构训练后的优劣。

**建议实验。** 先在不改变候选/权重的固定 coarse 输入上，检查 s=.25/.5/1/2/4 的 evidence/score 差异、实际 valid 翻转和 NMS变化。然后单独以 logZ-information 适配，与同预算旧函数 control 比；它是相对246的关键条件变化。旧223同时关闭所有融合几何导致9/11域退化，不能用它否定只换量纲的本候选。

## 3. 同一物理深度的语义依赖搜索范围；候选峰距也依赖内部刻度

**确定事实。** `jar/pretraining/model.py:248–250` 将

\[
\eta=(\log Z-\log n)/\log(f/n),\quad [\eta,\sin\pi\eta,\cos\pi\eta]
\]

经 MLP 加入候选特征，且位于单目 bank query 之前。10m 在 [.1,100] 中为 η=2/3，在 [1,100] 或 [.1,1000] 中为η=1/2。即使投影证据相同，单目读取与评分也得到不同候选语义。范围变化本来会改变离散候选和投影跨度，但这里又额外引入了“范围分位数作为深度先验”的路径。该作用能提高域内拟合，是否导致真实域浅偏必须诊断。

同文件 `set_range:118–121` 每次建立 log格，然而 `ruler_proposals.py:99` 使用固定内部绝对0.1的峰排斥。CPU计算 [.1,100] 的96格时，near≈.1处0.1范围排除10格，≈1处3格，≈10/50处仅自身1格；四个top峰在远处可能来自同一个宽峰的相邻格。没有证据证明它已经丢掉本轮真实GT峰。不会因全局96格oracle好就推断top4覆盖也好。

**先做无新增模块的因果隔离。** 固定完全相同候选、half_width、source结果和所有其它特征，只改变位置编码采用的 n/f；比较 coarse 学习分数和 cosine旁路分数、top4与真实GT最近候选的 rank。该测试隔离位置语义，区别于直接重设range造成候选、FPN路由和半径同时变动。再对固定 coarse logits 单独比较绝对0.1排斥与预先固定 logZ/格数排斥，记录选峰多样性和top4 oracle。

**可插拔方向。** 保持3维位置 MLP形状，另试不依赖搜索端点的几何坐标，例如 \(\xi=\log(Z/B_*)\)、[ξ,sinξ,cosξ]，\(B_*\)为有效相机基线的预定义无标签标量；或者用已有各源JZ构造候选可观测性坐标。前者需要明确纯旋转、不同view子集时基线归约及无基线fallback；后者可能丢掉绝对候选位置语义。**两者尚未实现或验证，不建议直接叠加进第一版。** 单纯固定近远界为另一常数没有消除问题；删除位置分支则是有损消融，不是严格修正。

## 建议的 V8 最小实验边界

先保留 V6 所有编码器、96/12候选预算、粗细空间模块、父族交接和渲染。使用独立研究 wrapper、显式开关与原state严格加载：A=source/null源数校正；B=logZ-information；A+B最后验证，位置/峰距暂只诊断。增加的是数个候选标量运算，参数可以保持不变；“近零开销”是计算量预期，必须独占同卡完整前向实测后才能写时延结论。

真实 TRAIN/VAL 需要同时观察 frozen、同预算control、各新函数；阶段指标与完整最终指标分开。父族gate与族内mean不是本轮首要改动：170–173/197–223已证明其局部收益与纠错代价，246粗层已错且254仅有末端统计，无法靠重新加一个gate解释全部泛化差距。若新证据表明 top4 候选缺失或 weak-parallax 族内均值是主要瓶颈，再改变对应关键条件，而不是继续堆 V7 重融合。

本审计没有执行真实TRAIN/VAL、精度、时延或显存实验；所有这些均为未运行。结构可疑性不等于性能根因，更不等于已实现泛化升级。
