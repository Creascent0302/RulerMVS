# 四路短训的独立只读代码审查

2026-11-03。审查对象为已经启动的 `train.py`、`objective.py`、`data.py` 和四个 `configs/*.json`。没有改动运行源码/配置、没有使用GPU、没有读取新TEST。结论是未发现需要停止当前四路训练的实质DDP、范围或损失mask错误。

## 已核对的实现

- 四路完全共用seed、TRAIN/VAL manifests、domain倍率、2卡×B8×accum2=globalB32、256更新、阶段冻结/学习率/余弦与warmup。递归配置对比确认 `v8_units_count` 相 `v8_units` 除输出/GPU位置外只多 `v8.source_count_strength=1`。
- `train.py:342–367` 各rank同域、每两个微批一次同步更新；前一微批no_sync，loss除以accumulation，使用同一AdamW/clip规则。batch/micro标识在恢复后由global_step与micro位置重建。DDP `find_unused_parameters=True`，source checkpoint为non-reentrant；未发现漏同步或多更新。
- `train.py:232–269,333–338` resume核对配置、manifest、world size、architecture，按stage恢复optimizer及每rank RNG；DataLoader独立generator不消费模型主随机流。warm start严格加载完整参数，不加载旧optimizer。只接受完整新run的last恢复。
- `data.py:143–222` 范围函数只在训练入口调用，VAL标志在读取targets前返回。有效参考GT需同时满足GT mask、image mask、view mask及正有限深度。nominal/broad/padding都在必要时向外扩展以覆盖本微批GT，不静默丢超范围标签；非法极值明确失败。
- generalization三路按相同样本键生成RGB增强、同一micro/domain RNG生成范围分支与单位factor。因此它们之间有同增强对照。各rank采用同一随机分支，但GT extrema仍各自来自本rank B8；不能宣称换world/batch后增强也逐步相同。
- source dropout只删源，不删参考或GT；参考有效mask不受该操作影响。correspondence loss通过原view_mask排除已drop源。随机局部遮挡保留相机可用性，这是增强设计，不是证明物理可见性。
- `objective.py:15–27` mask由按位与新建，没有别名修改targets。新相对项以差值替换原系数.2的相对项，没有再叠一份；其内部floor `.1*nominal/scale` 恢复物理分母下限 `.1*nominal`。当前fine_readout_weight=0；若非零主动报错，避免漏修第二个floor。
- VAL实际经RestartHooks以FP32评估，输出unit_scale与domain尺度转换一致；各rank指标求和后汇总；验证上下文广播rank0 buffers并恢复原mode/buffer/RNG。训练micro平均loss与最终像素池化指标不是同一统计，但四路沿用相同训练归约。

## 结果解释时必须保留的三个边界

1. **V8_units是联合单位方案。** 相V6_augmentation同时改信息通道、源资格阈值、候选物理阈值和相对loss floor。只有V8_units_count−V8_units可直接对应单个源数校正；不能把前一个差值全部归因于information。
2. **physical_thresholds改变nominal候选规则。** 当前实现以原输入单位固定阈值，不是仅除掉jitter factor。没有单位jitter时，Synth原peak_separation=.1×26.625=2.6625、fine_max=100×26.625=2662.5，V8分别为.1和100；Tartan原peak_separation=.02484375变为.1。这是显式配置的表示/候选改变，符合当前联合方案，但有独立性能影响。若后续需要“只消除单位增强的随机候选变化”，须另设nominal-preserving阈值分支，不能改写本轮历史。
3. **当前短训不自动保V6。** initial_validation_limit与validation_limit都为64/域，不会生成完整VAL best/guarded资格；按固定256末步final比较是合理的短筛协议，但不能宣称已完整验证或按全VAL保证不退化。后续应按主任务既定方案完整VAL确认后再决定长训。

## 建议

保持当前四路运行不变，将固定初值和固定末步、近中远/逐域指标一起解释。没有发现必须紧急修复的代码问题。以上结论为静态与配置审查，不替代真实DDP退出码、训练有限性或完整VAL验收。
