# 项目内的其他 baseline

这里的 CasMVSNet、Census 和 ZNCC，与 [`official_methods/methods/`](../../official_methods/methods/README.md) 中的五种方法同属对照方法。两组沿用各自已有的训练入口和输出适配，不要求所有方法换成同一个预测头。

| 方法 | 实现与说明 | 性质 |
|---|---|---|
| CasMVSNet | [cas_mvsnet/](cas_mvsnet/README.md) | 作者三阶段网络和项目适配 |
| Soft Census | [census/](census/README.md) | 不需训练的工程对照 |
| ZNCC | [zncc/](zncc/README.md) | 不需训练的工程对照 |

`classical.py` 共享两种传统方法的平面扫描、候选细化和一致性检查；各自目录的 `Adapter` 只固定匹配函数。Census/ZNCC 不冒充任何论文完整实现，也没有可训练网络。`jar.baselines.ClassicalMVS` 和 `build_baseline` 保留旧导入。

每个方法的 `configs/` 通过相对软链引用既有 CARLA 配置，历史实验没有搬动或复制。CasMVSNet 源码实体在 `cas_mvsnet/upstream/`，旧 `third_party/cas_mvsnet/` 是相对软链；`jar/official_casmvs.py` 仅重导出公开接口，兼容已有代码和旧 checkpoint。

```bash
official_methods/.venv/bin/python -m pytest tests/test_baseline_packaging.py tests/test_baselines.py tests/test_official_casmvs.py -q
```

目录重组没有改变网络、匹配计算、状态键或历史实验参数；算法效果仍以实际实验报告为准。
