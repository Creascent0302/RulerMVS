# 标尺三种用途：独立同预算重训消融

这组实验回答一个具体问题：**在已经具备V5表征的候选族模型中，标定导数进入源证据融合、尺度读取、细候选半径，分别是否提供净收益？** 不更改精确投影，不更改原始RGB和K/T，不读取TEST。

## 设计及实现对应

| 组 | 唯一改动 | 保留内容 | 应该检验的假设 |
|---|---|---|---|
| `control` | 无 | 全部标尺用途、V6父族交接 | 共同起点经过相同追加训练的基准 |
| `no_source_geometry` | `use_geometry=false` | 精确投影、几何有效掩码、FPN路由、半径控制、深度位置编码 | 源注意力是否需要知道候选移动对各源的像素影响，而非仅依赖外观 |
| `no_scale_routing` | `scale_routing=false` | 全部FPN编码、源几何条件、候选预算、半径控制 | 根据候选投影跨度选择读图尺度是否优于始终读取stride4 |
| `no_jacobian_width` | `fine_step_pixels=0` | 候选族数、每族三点、粗刻度宽度、上下限、精确重投影 | 一个特征像素对应的深度步长是否比只依据粗刻度宽度更适合搜索 |

三项使用主模型中已有的开关，没有另写近似模型。`geometry.py`只在构造时修改配置并保存在checkpoint，模型类、权重键和计算实现不变。结构相关代码为：

- `jar/pretraining/model.py::CandidateViewAttention.forward`：源pair中的二维导数方向、`log(1+|g|h)`投影足迹；`CalibratedDecoder._tokens_from_source`中的加权导数平方信息。第一组把这些额外几何条件置零，但深度位置编码不动。
- `CalibratedDecoder._observe`：从横向/纵向/深度投影跨度选择FPN读取层。关闭后将层位置固定为0，即stride4。其他层仍计算和读取后乘零，**不是删掉多尺度编码，也不是速度优化实现**。
- `jar/models/ruler_proposals.py::transport_proposals`：原半径为粗刻度半宽和`fine_step_pixels/|g|`的较大者，再按原限制裁剪。将像素步长设为0后仅剩刻度半宽；仍保留四族×三点和连续偏移，实际点仍精确投影。

256×448共同RGB下，粗候选为96×32×56，细候选为12×64×112。各组保持同候选数量、通道数、训练参数集合及目标。关闭几何输入可能让某些权重列梯度为零，但不删除参数或追加其他结构补偿。

## 固定训练协议

| 项目 | 公共组 | CARLA组 |
|---|---|---|
| 共同起点 | 同一个V5公共best | 同一个V5 CARLA best |
| 训练数据 | 七域各固定128组，按物理sequence分组均衡抽取 | 固定256个上参考训练组 |
| 轮数/批量 | 2轮，batch=2 | 2轮，batch=2 |
| 每组更新/曝光 | 896 / 1792 | 256 / 512 |
| 初始化与随机性 | 四组同seed20261102；相同样本和增强顺序 | 同左 |
| 可训练部分 | decoder，包括V6候选族门控；编码器和CNN桥冻结 | 同左 |
| 学习率 | 原decoder 1e−5；新门控1e−3 | 同左 |
| 权重选择 | 固定最后一次更新，不以VAL/TEST选best | 同左 |
| 固定诊断VAL | 每域16组，共112组 | 每参考16组，共64组 |
| 后续完整VAL | 7957组公共VAL | 411组四参考共同VAL |

公共和CARLA是两个独立适配实验，不串联。四参考VAL不代表四个独立物理轨迹。监督目标使用相同V6族目标，源几何消融不混入目标函数变化。与核心交接实验的`parent`、seed20261102采用同预算，可以对照control的一致性。

这属于**预训练任务权重起点的适配消融**。V5初始权重本身已经学过几何结构，896/256步不保证每个删除分支充分收敛；若变体明显退化，不能仅据此宣称它从零训练也必然无效。当前几何组只有一个seed，不能用核心交接三seed的误差条替它；主表需明确单seed限制。

## 实际执行状态

已实际完成CPU四项测试：control逐参数/输出与原V6一致；融合/路由开关保持相同投影导数和掩码；去掉Jacobian半径保持粗输出而改变细候选；四组均通过有限梯度、缺源NaN屏蔽、checkpoint恢复和标签隔离。CPU使用真实ResNet与小ViT替身，不能据此宣称完整权重GPU训练已成功。

八组训练和完整VAL现已全部完成。每组都有真实`completed.json`、`geometry_audit.json`及`full_val.json`；汇总`analysis/summary.json`的`complete_training`与`complete_full_val`均为true。统一队列最终`paper_handoff_ablation/queue_completed.json`为completed、`queue.exit_code.txt`为0；`queue_progress.json`是最后一轮进度快照，不作为最终状态。完整7957+411 VAL通过共享冻结编码完成，1463次额外完整forward逐位核对通过；该共享过程不计入模型运行速度。

完整VAL结论：取消源融合几何使9/11评测域MAE退化；取消路由的MAE变化不到1%且方向混合；取消Jacobian扩宽使8/11域MAE、10/11域AbsRel改善，但8/11域RMSE和7/9非空远距域MAE退化。Dynamic最终MAE从0.107013降为0.077514，仍不能据此全局切换。**本轮保留冻结默认模型，不把某一指标改善冒充全面升级。** 详细表格、oracle分解和数学解释已增补`docs/paper_experiment_plan_20261101.md`的E4。

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction

# 已通过，不需要为准备训练重复执行。
official_methods/.venv/bin/python -m unittest research.paper_ablation_20261102.test_geometry -v

# 独立启动一个训练组；通过CUDA_VISIBLE_DEVICES选择物理卡。
CUDA_VISIBLE_DEVICES=0 official_methods/.venv/bin/python \
  research/paper_ablation_20261102/geometry_train.py \
  --config research/paper_ablation_20261102/configs/geometry_public_control.json
```

每种消融都有自己的配置和目录，不覆盖V5/V6发布权重。所有训练/预测大产物经现有软链接进入：

```text
runs/active/rulermvs_v6_20261101/full_training/paper_geometry_ablation/
  plan.json
  training/public/<组名>/
  training/carla/<组名>/
```

`initial.json`、`final.json`由共享`handoff_evaluate.py`记录固定VAL的逐帧raw统计、最终深度、粗细误差、候选oracle、破坏/纠正率。`completed.json`记录实际更新、曝光、训练时间、梯度和恢复检查；`geometry_audit.json`记录运行源码及初始权重hash。只有二者均成功才能标成整组完成。完整VAL继续使用同一固定末步权重，不能看完结果又挑另一checkpoint。

完整VAL文件为每组目录下`full_val.json`，共享逐帧raw分片位置在其`datasets.<域>.frame_logs`中。CPU汇总入口为`geometry_summary.py`，生成`analysis/metrics.csv`、`summary.json`和`结果表.md`。完整VAL只额外统计深度/阶段/距离区间，候选oracle分解仅来自初始/末步固定16VAL，不可混用。

## 如何解释结果

1. 先看全图MAE、RMSE、AbsRel，再看近中远、粗细层与错误转移；没有GT的桶记N/A，不把scene_unit原始MAE跨域相加。
2. 若关某用途后平均变好，保留该事实并研究是否存在错误尺度条件、失配或冗余；不能为了维护标尺理念只报告有利域。
3. 若粗层改善但最终退化，结合相同候选供给下的选择损失与上采样误差解释，不把最终数字当唯一因果依据。
4. 若差距很小，先看种子/场景变动再说必要性；单seed小差异没有显著性保证。
5. 本消融把计算预算保持一致用于归因，不直接证明删掉某计算能提速。若要部署删分支，必须另做输出保持与独占计时验证。

实际同seed重复提供了额外边界：本组control与交接实验的parent(seed20261102)初始完整状态、初始输出和样本schedule完全相同，但未强制CUDA训练确定性，第二步已出现约1e−6梯度范数差异，末步公共VAL存在小差异。例如MatrixCity MAE相差约0.00515m、Dynamic约0.000794 scene_unit，CARLA仅约1e−5m量级。它不是正式置信区间或最坏误差界，但提醒不能把同量级单seed变化解释为稳定结构收益。完整证据为`paper_handoff_ablation/control_repeat_audit.json`。
