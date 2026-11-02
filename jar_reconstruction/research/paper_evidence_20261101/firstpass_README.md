# MVSAnywhere 首遍补充对照

目的：同时对比原生首遍和一次范围细化版，避免只选择较慢的 baseline 作为速度对手。原主评测配置、结果和模型架构均不修改。

已完成CPU配置准备、哈希核验、公共/CARLA完整GPU测试及独占同卡计时，`completed.json=true`、`finish.exit_code.txt=0`。以下保留复现命令；复现时应使用新的数据盘输出目录，不覆盖本轮证据：

```bash
# 只准备：不读RGB/深度，不使用GPU
official_methods/.venv/bin/python research/paper_evidence_20261101/firstpass.py \
  --config research/paper_evidence_20261101/firstpass.json --prepare

# 公共TEST → CARLA TEST → 独占GPU0同卡计时
official_methods/.venv/bin/python research/paper_evidence_20261101/firstpass.py \
  --config research/paper_evidence_20261101/firstpass.json
```

输出位于数据盘软链接 `runs/active/rulermvs_v6_20261101/full_training/paper_firstpass/`：

- `public/full_test/`：11970组公共TEST，真实双目域不复制视图。
- `carla_new/full_test/`：174完整同步四目帧×4参考，共696组。
- `speed/results.json`：固定完整四目VAL的第0、41、81组，不读取标签；同一5090交替计时，FP32、TF32关闭、共同位置编码缓存，预热10次、每模式每帧40次测量。
- `plan.json`：固定配置、权重、TEST/测速清单与源码哈希；执行前后核验。首遍官方权重仍为主比较的Hero，不做训练或权重替换。

首遍只将官方 `num_refinement_steps` 从1改为0。图像、标定、参考槽、TRAIN范围、GT掩码、指标与表面采样参数继承恢复评测，不能因首遍误差更大再改变范围。

速度分别报告：V6 eager、V6显式fast、MVSAnywhere首遍、MVSAnywhere细化的单参考与实际串行四参考；V6额外报告实际共享四参考。`methods` 是网络+输入适配，`methods_with_resampling` 额外包括共同256×448输出重采样。四参考不是单次耗时乘4。V6共享输出必须与同路径串行输出逐位一致。多个模型共驻，不能将此进程显存当方法独占显存；IO、H2D、点云融合不在计时内。

历史首遍结果进行了核查：公共TEST清单与权重相同，但seed、评测入口和公共指标源码hash不同；旧CARLA帧集合也不同。因此没有直接复制历史数字，保留历史证据并在新协议下重测。最初CPU准备版本保留为 `plan_prepare_v1.json`；最终计划加入共同输出重采样耗时统计，已实际执行。

最终同卡含共同输出处理的中位数：V6 eager/fast单参考33.081/27.486ms、共享四参考114.098/92.314ms；MVS首遍/细化单参考21.914/37.029ms、串行四参考87.520/148.068ms。V6 fast仍慢于首遍，不能仅展示细化版以声称最快。完整深度和可见表面比较、逐帧配对分布、CSV与LaTeX见`paper_firstpass/analysis/`，fast精度另通过`fast_test.py`完整测试核验。
