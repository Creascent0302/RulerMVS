# ZNCC 工程对照

`Adapter` 固定使用零均值归一化互相关（ZNCC）。它在局部 RGB 窗口中减去各通道均值，再用向量点积除以两边的长度来衡量相似度；候选深度上的整个窗口都会按照相机标定重投影。

共同的分块扫描、局部细化、视图聚合和预测一致性逻辑见 [../classical.py](../classical.py)。这是无学习参数的工程对照，不声称是某篇完整传统 MVS 论文的复现。低纹理、重复纹理、斜面和遮挡可能导致拒绝或错误匹配，误差与覆盖率必须一起报告。

```python
from jar.baselines.zncc import Adapter

model = Adapter(config)
prediction = model(inputs)
```

`configs/model.json`、`configs/dev_complete.json` 和 `configs/test_complete.json` 通过相对软链使用既有 CARLA 配置，修改原文件或软链入口修改的是同一份配置。

```bash
official_methods/.venv/bin/python tools/carla_compare.py --config jar/baselines/zncc/configs/dev_complete.json --device cuda:0 --output runs/zncc_new_eval
official_methods/.venv/bin/python -m pytest tests/test_baselines.py tests/test_baseline_packaging.py -q
```

从项目根目录执行，可选择 CPU/CUDA。它不需要训练 checkpoint；输出目录仍建议指向数据盘。
