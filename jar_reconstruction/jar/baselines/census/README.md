# Soft Census 工程对照

`Adapter` 固定使用 Soft Census 匹配，全部几何扫描逻辑在 [../classical.py](../classical.py) 中。它将局部灰度与中心灰度的差经 `tanh` 编码，在真实标定下对整个窗口进行候选深度重投影，并进行局部细化及预测之间的一致性检查。

这是无需训练的传统方法工程对照，不是某篇 Census+SGM 论文或完整算法包的官方复现。没有学习参数，也不需要训练 checkpoint。平面窗口假设可能损伤斜面、细物体和遮挡边界，评价需要同时报告误差和有效覆盖率。

```python
from jar.baselines.census import Adapter

model = Adapter(config)
prediction = model(inputs)
```

`configs/model.json`、`configs/dev_complete.json` 和 `configs/test_complete.json` 软链接到现有 CARLA 对照配置。修改参数直接编辑这些文件即可；不会产生第二份不一致的配置。

```bash
official_methods/.venv/bin/python tools/carla_compare.py --config jar/baselines/census/configs/dev_complete.json --device cuda:0 --output runs/census_new_eval
official_methods/.venv/bin/python -m pytest tests/test_baselines.py tests/test_baseline_packaging.py -q
```

从项目根目录执行，推理支持 CPU/CUDA；完整合成验证不代表真实数据精度。结果目录应使用数据盘软链。
