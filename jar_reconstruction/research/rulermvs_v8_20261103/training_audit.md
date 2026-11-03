# V8 训练审计与建议（2026-11-03，只读审计）

已阅读项目 AGENTS、research_memory 当前共识及 247/249/254。本文核对实际代码、配置、已有日志和 TRAIN 清单；没有训练、GPU 前向、改权重、下载或访问新的 TEST。下列改进均为待验证方案，不能记为实验收益。没有修改生产代码或 research_memory。

## 结论

优先从 249 的新 V6 公共 `best.pt` 初始化，保留 V6 原件，并设置同起点、同预算的 V6 继续训练对照。当前没有证据需要更换通用预训练，更不应加载 Hero。最高优先级是把尺度增强语义、训练域权重和验证门槛写清，再用较小学习率续训；不建议照搬从零预训练的 1+2+6 轮解冻流程。

确定问题与待验证方向必须区分：`absolute_information=false` 当前不起作用是确定的配置接口问题；单位增强会改变细候选物理规则也是确定的实现事实，但它是否损害泛化尚未测试；域重配、冻结保护、图像增强的收益都未运行。

## 1. 实际数据曝光

配置为七域 TRAIN 64198、VAL 7957。实际每轮 84672 曝光、2646 更新；9 轮 23814 更新、762048 曝光。每个 optimizer update 内所有 rank、累积微批均属于同一域。`balanced_prefix` 仅使短前缀接近最终比例，**不是域等权**。

| 域 | TRAIN 不同组 | 当前每轮曝光 | 当前份额 | 建议候选每轮曝光 | 建议份额 |
|---|---:|---:|---:|---:|---:|
| TartanAir | 12862 | 12864 | 15.19% | 12864 | 15.19% |
| MatrixCity | 4185 | 4192 | 4.95% | 9920 | 11.72% |
| Hypersim | 5791 | 8704 | 10.28% | 12736 | 15.04% |
| MVS-Synth | 9476 | 9504 | 11.22% | 9504 | 11.22% |
| VKITTI2 | 9500 | 14272 | 16.86% | 12704 | 15.00% |
| BlendedMVS | 15184 | 24320 | 28.72% | 15200 | 17.95% |
| DynamicReplica | 7200 | 10816 | 12.77% | 11744 | 13.87% |
| 合计 | 64198 | 84672 | 100% | 84672 | 100% |

建议份额是一个可检验的候选，不是最优比例。保留所有域每轮全覆盖，总预算不增，把 Blended 的主要重复曝光转给占比较低的 Matrix/Hyp，Dynamic 不减。对应每轮更新数为 `[402,310,398,297,397,475,367]`，使用整数曝光配额最清楚；若复用倍率接口，须检查 `ceil(n*m)` 与 32 补齐后的实数配额，避免浮点多补一个批次。

理由来自训练协议：完整 VAL 域间等权，训练 Matrix 却只占 4.95%，Blended 占 28.72%；这不是测试调参的证明。风险也明确：Matrix TRAIN 只有 1 个 `sequence` 标识/4185 组，增加重复不能制造新地图；因此不推荐简单七域完全等权并继续压缩 Blended。清单统计的 sequence 数：Tartan14、Matrix1、Hyp61、Synth96、VK30、Blended95、Dynamic24；这是清单分组数量，不认证独立物理场景数。

建议先保留原比例作为架构主对照，单独检验上述重配；若只跑一个 V8，应披露训练比例改变，不能将总收益归因于新模块。ETH3D 与 Tanks 维持 0 训练曝光，不加入训练、教师缓存或选模。

证据：`research/overnight_v6_v7_mvs_20261103/configs/v6.json`、`v6/events.jsonl` initialized 记录；`research/v2_training_20261028/sampling.py:28`。

## 2. 范围、单位及候选语义

### 已正确执行的部分

- `own_training_domain` 在每个训练微批执行：50% 用 TRAIN 固定 near/far，50% 用本微批有效参考 GT 的 min/max 向外扩；50% 将内部 depth_scale 乘 loguniform(.5,2)。VAL 不调用它。
- 相机位移、粗候选 near/far、监督深度同时除以 scale，输出乘回 scale。粗格由 `CalibratedDecoder.set_range` 重新构造 **logspace**，不是配置 `transition=16` 所暗示的 hybrid 刻度；固定 near/far 时，单位扰动后的粗格物理深度相同。
- per-view 亮度 gain 同时用于 rgb/matching_rgb。equal_pixels 随后由已增强主 RGB 插值得到 matching_rgb，因此不存在这两支路看见不同亮度图的问题。K/T 没有被亮度增强改动。

### 确定的语义变化

`jar/models/ruler_proposals.py:99,115` 的峰间隔 `.1`、细步最小 `.005`、最大 `100` 直接按**内部单位**使用。若 s=depth_scale，它们对应物理阈值 `.1s/.005s/100s`；当 jitter 乘 f 时物理阈值也乘 f。单位扰动不只是重新表示同一候选集合，它还改变峰抑制及细候选截断。

主干 `information=Σw||J||²`（`jar/pretraining/model.py:98,246`）随内部单位变化约按 f² 缩放；pair MLP 的 `||J||*cell_half_width` 理论上保持无量纲，但信息通道不保持。`CalibratedDecoder._tokens_from_source` 只检查 `use_geometry`，**没有检查 `absolute_information`**，因此只把该配置设 false 不构成有效消融。当前 V6 配置为 true，所以这不是原训练意外关闭或打开的问题。

`jar/pretraining/losses.py:76` 的相对项使用 `gt.clamp_min(.1)`，该 `.1` 也是内部单位；扰动使其物理分母下限从 `.1s` 变为 `.1sf`。低深度区的损失权重随增强变化。最终 log 误差不受这一单位变化影响。

### 推荐处理

不要在暖起点续训中直接把全部阈值改成“米制常数”并声称无损修复：旧 V6 本就依赖各域 nominal_scale。若要隔离纯单位增强，可传入 `nominal_depth_scale`/`unit_jitter_factor`，将峰/细步阈值和 loss denominator floor 除以 f，使 f=1 时严格保留旧候选规则；其收益应与原增强控制对照。主干绝对信息可单独保留作增强压力，不把这种实验叫完整尺度不变；若改成无量纲信息，是表示改动，应单独消融。

先加入不训练的合成契约检查：相同 RGB/K/T 与固定物理 near/far，f=.5/1/2 时核对物理粗格、峰抑制、细候选边界、相对误差权重；再在固定 TRAIN/VAL 样本报告变化，不访问 ETH/Tanks。当前已有 test_common_data 只检验 dict 范围/原 batch 不变，没有检验完整候选语义。

当前范围取 B8 **微批** GT 的联合 extrema，并由 rank + micro update 决定随机数。改为 8 卡 B4 或 4 卡 B8 会改变 GT 范围分布/种子，即使 globalB32 相同也不是相同增强曝光。做严格同预算架构对比时，双方必须使用相同 world/batch/accum 配置；若要跨 GPU 数完全一致，需要新定义全局样本键/全局范围协议。

## 3. 损失、梯度与冻结核查

- 没有 Hero 教师，也没有在线蒸馏。V6 objective 用真实 GT；`group_targets` 有 no_grad；`fine_actual_regret` 的候选位置 detach；transport 返回 parent probability.detach；handoff 质量统计和父分布 detach。未发现教师/标签梯度泄漏。
- V6 group CE 和 group risk **替换**原 fine CE/risk（通过加差值），不是再叠一份同权重 fine loss。group_depth 标签按实际族内读出误差选择；final loss 仍向被选族内连续深度求导。
- 训练 mask 包含有效 GT、主图 mask、view mask、有限正深度；stage loss 加 ref_valid。公共没有额外 eval_range。有效标签不因预测不确定度被删除。所有候选在参考有效处都允许存在，这是有意允许无源区由上下文补全，不能将它误报为“未屏蔽遮挡”的实现错误。
- BN running stats 确实始终 eval（`encoders.py:69`）。stage2/3 matching backbone 解冻时 BN affine 参数仍可训练；两者不矛盾。VAL 广播 rank0 buffer 并恢复原模式、buffer、RNG，未发现旧混合 BN 验证错误再现。
- foundation 第1轮冻结、第2–3轮末4块、第4–9轮末8块；共享 norm 一直冻结。bridge 在配置顶层 detach=true，但**所有实际 stage 的 bridge_detach_backbone=false 覆盖它**，所以 matching 解冻后同时收到匹配与 bridge→ViT 方向梯度。应在报告中按有效 stage 值说明。
- 不建议把 `fine_readout_weight` 从0直接加回：历史203/相关条目已无通用净收益，本轮没有改变足以支持重试的损失条件。

## 4. 训练增益优先级与有限预算方案

1. **暖起点与稳定优化。** 使用249 V6 best（第9轮），V8 新层恒等/零初始化。若新增层导致分布变化，可先 128–256 更新只训练新增层，随后共同低学习率适配；已有 decoder/bridge/fpn 以 1e-5 起，matching 2e-6，DA 末4块 1e-7 是保守候选，不是经实验选择的最优值。BN stats 和 shared norm 继续冻结。新增层可用较高LR，但不提高所有成熟参数的LR。原始 V6 保留作为固定参照。
2. **先隔离配比与单位规则。** 同起点、同样本键的 V6 续训控制与 V8 主候选均先使用当前比例/增强；重配和纯单位增强另列一项训练条件控制。这样能判断收益来自模块、曝光还是单位分布变化。若预算只允许两路，优先 V6 continuation vs V8，而不是同时引入四项训练变化后宣称结构收益。
3. **图像增强留作第二优先级。** 当前只有独立 ±15% gain、参考轮换，没有 hue/gamma/noise/blur 或几何 crop/flip。可测试以所有视图共享弱色彩/gamma变化为主体、保留小幅逐源 exposure；不加空间不一致的逐视图 crop。几何缩放/裁剪必须同步更新两套 RGB、K、mask 与深度采样，不能只变图。增强不增加推理开销，但收益未测试。
4. **视图数量泛化。** 当前主动丢源关闭，N2域占约29.63%、N4域70.37%，没有 N3/N8训练。V8自研可单列小概率合法源子集训练（只从真实存在的相机采样，保留参考），但它改变247共同训练条件；不要复制视图伪造8图，也不要为此混入 ETH/Tanks。
5. **保护旧精度。** 首先用低LR和全部七域回放；如仍遗忘，再考虑冻结V6的 TRAIN-only excess-error 正则。教师必须独立eval/no_grad，同一增强RGB但用明确的nominal单位，输出转物理单位后比较；不可复用学生更新后的编码冒充教师。此项增加训练成本、不增加推理成本，且并不保证泛化，应是后续控制而非默认堆入。

原 V6 完整 VAL 风险 e3/e6/e9=.474504/.450808/.448344；e6→e9仅改善约0.55%，Hyp full MAE .544316→.572270、VK 2.738855→2.768813反而上升，不能由末轮best断言继续大LR长训全域有益。e7子集风险不能与e9完整风险直接比较。

既有每20更新抽样日志还显示明显梯度裁剪：stage3 Matrix 的37个记录全被clip1裁剪，裁剪前norm中位8.655；Hyp74个全裁剪、中位3.635；Blended231个中98.7%裁剪、中位6.091；VK139个中58.3%裁剪、中位1.101。这是已记录更新的统计，不是全部更新统计，也不能等同各域有效学习率。V8应记录loss分项与参数组norm，先判断是否某分支支配梯度；不据此直接放大clip阈值或断言裁剪是泛化问题原因。

建议短筛使用固定约1024更新与七域固定VAL子集，候选预选后再跑完整7957 VAL；完整续训可事先定3轮/7938更新，首次/每轮完整验证。由全域风险选一个checkpoint，同时报告各域MAE/RMSE/AbsRel及距离段。若声称“保V6”，应事先设逐域允许退化阈值并检查全部域，不能仅用宏平均掩盖某域退步；失败则保留V6，V8如实作为候选结果。

## 5. 多卡、存储与断点

- 训练主对照可用两路各4卡、每卡B8、accum1，均globalB32；它和原2卡B8 accum2的范围增强轨迹不同，不能称逐步复现。8卡单路B4 accum1也保持globalB32，但微批GT联合范围改变更大。正式前需要实际解冻阶段GPU容量冒烟。本审计环境 `nvidia-smi` 不能连接驱动，未认证当前GPU空闲/容量；247既有B8 BF16 stage3证据约20.16GiB是历史实测。
- 保留 BF16前向、FP32几何与loss、TF32off和grad clip1。现runner硬编码clip1，改配置的grad_clip不会实际生效；当前配置=1所以现训练一致。
- 从 V6 迁移应使用 **新运行的 initial_model_checkpoint**，不能把不同V8架构/新配置伪作resume。`best.pt`没有optimizer；`last.pt`含optimizer/scheduler、每rank RNG、update、exposures。严格故障恢复只用新运行last，同world/配置/manifest；既有runner已覆盖累积更新边界与跨stage恢复。
- 现runner的新运行 `best_risk=1e30`，**不会自动把暖起点作为本轮候选或保证优于V6**。应在首次更新前做共同协议完整VAL，保存step0参照；新best与“是否可替换V6”分开判定，避免把全程退步的最优新checkpoint误称保精度升级。
- 原runner在每个stage重建AdamW并清空动量，阶段1/2/3 LR分别重新起始；暖起点不要无意重用该从零流程。建议一段连续余弦、明确短warmup；有解冻时显式新增参数组或记录重建决定。现 warmup 只在epoch==0生效，后续stage不自动warmup。
- checkpoint每400更新原子覆盖last，加epoch端last，best仅完整VAL更新，final保留末步。新训练可取每200–400更新，不逐步存全量权重。当前V6 last/best/final大小约.879/.367/.367GiB；两路新实验常驻约3.23GiB，不含源归档与临时写入，建议预留至少8GiB实验增量。
- 数据盘当前只读检查约234GiB余量。沿用200GiB底线、208GiB配置安全线及现atomic_save额外4GiB临时余量。输出仍放数据盘已有软链接体系，不在项目内复制大checkpoint或每样本教师缓存。
- 冻结配置、manifest SHA、初始化checkpoint SHA、源码快照与launch_plan；每次实验完成/失败/停止由主任务追加research_memory。
- 本轮无需下载；DAv2-B与ImageNet ResNet18通用文件已在本地。V6只加载DAv2的 `pretrained.*` encoder、没有加载其DPT任务头；matching使用普通ImageNet ResNet18。以后如需资源，只经服务器eno5，清除所有代理并记录源URL/SHA，不能走本地代理。

## 证据入口

- `research/overnight_v6_v7_mvs_20261103/{train.py,common_data.py,configs/v6.json}`
- `research/rulermvs_v6_20261101/{model.py,objective.py,training.py,handoff.py}`
- `research/rulermvs_v5_20261031/{model.py,bridge.py}`
- `research/v2_training_20261028/{training.py,sampling.py,objectives.py}`
- `jar/pretraining/{data.py,model.py,encoders.py,losses.py,comparison.py,runner.py}`
- `jar/models/{ruler_direct.py,ruler_proposals.py}`
- `runs/active/overnight_v6_v7_mvs_20261103/v6/events.jsonl`
