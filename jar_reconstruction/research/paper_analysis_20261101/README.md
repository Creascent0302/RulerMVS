# V6 论文分析与可复现绘图

本目录只分析已经完成的统计，不运行模型，不重新选择权重，不从历史运行补填当前缺失的精度或速度。

```bash
official_methods/.venv/bin/python research/paper_analysis_20261101/analyze.py \
  --config research/paper_analysis_20261101/config.json
official_methods/.venv/bin/python research/paper_analysis_20261101/test_analysis.py
```

参数均在 JSON 中修改。默认读取 `full_training/evaluation_recovery`，输出通过既有 `runs/active` 软链接写入数据盘的 `full_training/paper_analysis`。可在评测尚未全部完成时运行：只生成 `completed=true` 的完整组，明确列出缺项；同一命令可在评测结束后重跑更新图表。`config_legacy_val.json` 单独保存原自动评测已完成的公共 VAL，不与恢复后的正式 TEST 混在一起。

每张图同时导出 PDF、SVG、PNG；图中文字使用英文便于投稿，输出 `图表说明.md` 为中文。`analysis_manifest.json` 保存输入统计、manifest、逐帧分片和分析源码的 SHA256。

| 文件 | 科研问题与解释边界 |
|---|---|
| `training_full_val` | 每个阶段是否改善完整验证集；仅使用 `full=true` 的记录，星号标验证风险最小轮。不同域曲线相对该域第一轮归一，不混合原始单位。 |
| `domain_relative_errors` | V6 相对每个对照在哪些域改善或退化。改善率为 `(对照−V6)/对照`，正值更好；负结果全部保留。 |
| `distance_relative_errors` | 改善集中在近、中还是远距离。距离分区完全继承原评测定义，不自行改变阈值。 |
| `boundary_relative_errors` | GT 深度变化带和平滑区分别受益多少。变化带不等于小障碍物实例，也不证明实例召回改善。 |
| `stages_stride8` | 粗、细、最终输出在同一个 stride8 真值网格比较，判断粗面改善是否被后级保留；不是全分辨率最终 MAE。 |
| `paired_sequence_intervals` | 同一数据上配对比较，并按场景/轨迹整组 bootstrap 2000 次；避免把相关像素或连续帧当独立样本。 |
| `accuracy_latency` | 仅在本轮同组测速完成、权重与输入设置核验一致后生成。报告真实计时视图数；网络与适配耗时不等于整车端到端耗时。 |
| `mechanism_*` | 可选读取固定 VAL 机制探针：粗对细错条件率、连续候选覆盖、选族遗憾。推理关模块只证明当前权重的依赖，不代替重训消融。 |

`depth_metrics.csv`、`stage_metrics.csv`、`surface_metrics.csv` 保留完整指标、单位与覆盖率；`depth_table.tex` 可用于 LaTeX（需要 `booktabs`）。`paired_frame_differences.csv` 保存逐帧配对差，便于后续定位退化帧。

统计必须保留的边界：

- 当前 MAE/RMSE 采用预测有效像素作分母，所以表中保留 `coverage`、`gt_pixels`、`predicted_pixels`。若预测覆盖不全，脚本不对该组声称同像素配对置信区间。
- VKITTI2 同一个 Scene 的天气和相机角度变体合成一个统计组；Hypersim 同一场景的多个相机也合并。不能将这些变体当多个独立场景。
- 一个独立场景只给描述性点估计、不画伪置信区间；少于五个场景的区间很不稳定。区间描述现有场景采样的不确定性，不证明任意外域泛化或训练种子稳定性。
- MAE 改善可能伴随 AbsRel、远距或边界退化；本脚本同时保留这些情况，不按结果裁掉数据域。
- 公共域已经用于训练，现有跨域测试不自动等于未见数据集的零样本泛化。各方法预训练、额外训练预算不同，主表不能替代同初始化、同训练预算的因果消融。
- `scene_unit` 不是米，不能与公制域直接平均 MAE。表面指标保持其原采样和阈值，并未重新做场景级点云融合。
- 快速编译版速度与普通执行的完整精度应分开记录；少量验证帧的近似数值保持不能自动等同完整 TEST 精度不变。

已实际运行的四个 CPU 数学测试覆盖：已知固定误差比例的场景 bootstrap、单场景禁止 CI、缺预测覆盖跳过配对 CI，以及同场景天气/视角不独立计数。没有 GPU 测试依赖。

补充分析入口：

```bash
# 首遍/细化共同TEST、可见表面，以及真正完成后的同卡速度对照。
official_methods/.venv/bin/python research/paper_evidence_20261101/firstpass_report.py
# 只在三seed交接与单seed几何重训汇总完成后生成对应图。
official_methods/.venv/bin/python research/paper_analysis_20261101/retraining.py
```

`retraining.py` 分开读取完整 VAL 与固定16帧/域的机制子集。误差条是三个训练 seed 的样本标准差，不是95%置信区间；单seed几何消融不画伪误差条。所有消融均从共同任务权重适配、固定末步，不能表述为从零训练或全预算架构上限对照。首遍报告只有在 `paper_fast_test` 实际完成完整 TEST 并核验同部署选项、权重、清单与 GT 分母后，才把 fast 精度加入精度—速度联合图；否则只报告 fast 的独立速度。
