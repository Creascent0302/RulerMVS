# 初始表面重选：诊断、方法和验证

这是一个研究中的可选分支，尚不能宣称具有论文级新颖性或跨域性能。完整模型仍使用原来的四目编码、候选场和两轮精确重投影细化；新增分支只决定细化从哪个表面出发。

## 为什么先改表面选择

在已有 DEV25 帧的 stride 4 有效射线上，旧初值误差超过 4 米的 1471 条射线中，1453 条（98.8%）在已有候选集合里存在距真值 1 米以内的候选。这说明当前主要故障是**有候选但排序/压缩错误**，并非先要增加更多深度格点或更大的图像编码器。旧候选场的学习残差会大幅改动原始匹配排序；与此同时，稠密解码器在变成一个连续深度值后丢失了其他可能表面的位置。这里的百分比只是这批射线的诊断，不是总体召回率。

简单地把最强匹配候选替换旧初值，在 TRAIN66 上恶化；有些旧初值虽不准，原细化器本来可以修正它。这个失败促使我们保留“沿用旧初值”作为明确的第一种动作，并学习**什么时候值得换面**。

这一判断与 [MVSAnywhere](https://openaccess.thecvf.com/content/CVPR2025/html/Izquierdo_MVSAnywhere_Zero-Shot_Multi-View_Stereo_CVPR_2025_paper.html) 保留多目匹配与单目先验共同解释的思想相容；[PatchmatchNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_PatchmatchNet_Learned_Multi-View_Patchmatch_Stereo_CVPR_2021_paper.pdf) 和 [GoMVS](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_GoMVS_Geometrically_Consistent_Cost_Aggregation_for_Multi-View_Stereo_CVPR_2024_paper.html) 也表明候选传播/空间聚合必须尊重表面几何。[CHOSEN](https://arxiv.org/html/2404.02225v2) 已提出学习式候选排序，所以“给几个深度候选学习打分”**不是本研究的首创点**。当前可研究的区别是：复用我们已有的标定候选场，保留原解码结果为可拒绝换面的动作，读取各源视角的单独匹配曲线，并在尽量小的额外计算量下嵌入原细化流程。

## 具体计算

每条参考射线只保留三个候选：原解码初值、候选场的最高峰，以及与最高峰有实际米制间隔的次高峰。候选不是通过深度平均得到的。每种候选读取同一逆深度格点上的候选场评分、原预训练匹配评分和逐源视角匹配信息；另加入与原初值及局部邻域深度的差异。缺失、投影无效的源视图由掩码过滤。至少有两台源相机在该候选处**可合法采样**，才允许换离旧初值。这个条件不证明它们无遮挡或匹配正确；具体匹配可靠性仍由评分头学习，未建立严格的物理可见性证明。

一个共享的 2185 参数小头先逐候选投影到 24 维，再用两层 3×3 深度可分离卷积读取局部证据，输出三个风险分数。胜出候选作为原有标定细化器的初始深度。图像编码器、候选场和细化轮数都没有增加。模型读取的仅是 RGB、标定及由它们产生的推理证据；监督深度只进入训练损失。训练使用真实候选相对于真值的有界米制风险和选峰交叉熵。因为硬选峰不可微，单靠最终深度损失不能训练这个小头，配置必须启用 `loss.dense_surface_recovery`。

实现见 [`jar/models/spatial_surface_recovery.py`](../../jar/models/spatial_surface_recovery.py)、[`jar/models/dense_decoder.py`](../../jar/models/dense_decoder.py) 和 [`jar/dense_losses.py`](../../jar/dense_losses.py)。`candidate_field.stride` 必须为 4，保证候选、原初值和训练标签位于完全相同的像素格点。参数和功能开关在 [`surface_recovery_recommended.json`](../../configs/candidate_revision_20261018/surface_recovery_recommended.json)。

已实际运行的单帧无标签推理命令如下；输出的 NPZ 中 `dense_depth` 为 `[4,256,448]`，只请求上相机参考时其有效深度像素为 114,688 个。`infer.py` 不读取标签，所以该命令的指标栏为空是预期行为。

```bash
CUDA_VISIBLE_DEVICES=0 official_methods/.venv/bin/python infer.py \
  --config configs/candidate_revision_20261018/surface_recovery_recommended.json \
  --checkpoint runs/candidate_revision_20261018/surface_recovery_recommended.pt \
  --device cuda:0 --split test --limit 1 \
  --output runs/candidate_revision_20261018/surface_recovery_infer_smoke
```

## 已完成的实验

选择头在 CARLA TRAIN66 帧的缓存候选上训练 300 步，未用 DEV 真值更新权重。离线同射线初值 MAE：TRAIN66 从 0.779 米到 0.745 米，DEV25 从 2.429 米到 1.424 米。DEV25 上原模型与研究性前向接线使用相同输入、权重和 2,087,873 个有效评测像素；最终深度 MAE 从 1.012 米到 0.922 米，24–80 米组从 2.388 米到 2.127 米。全部 25 帧都有改善。前向接线实验摘要在 [`spatial_gate_full/summary.json`](../../runs/candidate_revision_20261018/spatial_gate_full/summary.json)。这些 DEV 帧曾用于方法开发，因此仅能说明开发集有效。

目前 CPU 集成及相关回归测试 51 项通过，覆盖 2/3/4 台相机、无源视图、无 GT 推理、梯度和严格 checkpoint 恢复。真实 CARLA 的一次 GPU 训练优化步骤成功，只有新增分支的参数更新；训练配置见 [`surface_recovery_training_smoke.json`](../../configs/candidate_revision_20261018/surface_recovery_training_smoke.json)。自包含推理 checkpoint 位于 [`surface_recovery_recommended.pt`](../../runs/candidate_revision_20261018/surface_recovery_recommended.pt)。研究脚本换面与正式模型在同一真实帧的最终深度逐像素相同，最大差值为 0。真实四目只保留两台相机时，正式 GPU 推理仍为有限值且拒绝换面。这不是长期端到端训练结果。

同帧交替、CUDA 同步的 FP32 计时显示，选择器让完整模型前向中位时间从 82.3 增至 85.5 毫秒；逐次配对增量中位数约 3.1 毫秒。该实验是已驻留 GPU 的输入，不包括图像采集、解码与输出搬运。特征构造约 2–3 毫秒，小头约 0.3 毫秒。顺序先测旧模型再测新模型得到更大的差值，受到 GPU 运行状态漂移影响，因此以配对计时为主要新增开销证据。详细原始计时见 [`surface_recovery_paired_cache.json`](../../runs/candidate_revision_20261018/surface_recovery_paired_cache.json)。完整 TEST200 的顺序测量为旧 108.2、新 130.8 毫秒均值，不具备同帧交替控制；完整车端 10Hz 仍未验证。

优化时发现相同候选在不同证据曲线上重复计算七次逆深度查找。现在一次计算后复用索引，五个真实帧的正式输出逐像素完全一致；同帧配对的特征构造计时从平均约 3.50 降至 2.80 毫秒。整网时间波动较大，不能据此声称确定的整网加速。BF16 两帧尝试出现非零精度差异且这台机器短测试速度反而下降，所以推荐配置继续用 FP32。优化对照见 [`surface_recovery_index_optimization_after.json`](../../runs/candidate_revision_20261018/surface_recovery_index_optimization_after.json)。

冻结模型后一次性评价独立轨迹 TEST200，双方 15,196,011 个 GT 像素、覆盖率均为 100%，输入张量的逐帧哈希一致。结果如下，完整分段和逐帧记录见 [`test200_surface_recovery_frozen/summary.json`](../../runs/candidate_revision_20261018/test200_surface_recovery_frozen/summary.json)：

| 指标 | 原推荐模型 | 新增初始表面选择 |
|---|---:|---:|
| stride 4 初值 MAE | 3.246 m | **2.729 m** |
| 最终全图 MAE | 2.396 m | **2.262 m** |
| 最终 24–80 m MAE | 7.034 m | **6.776 m** |
| 最终 20 cm 内比例 | 33.19% | **36.61%** |

200 帧初值全部改善，最终 177 帧改善、23 帧退化，最大单帧退化约 0.011 米。远处 24–80 米仅占 TEST 有效像素约 17.6%，却贡献原模型总绝对误差的一半以上。训练与 DEV 来自 `captured_data (1)`，TEST 来自独立序列 `captured_data`，因此跨轨迹误差远大于 DEV 是更重要的事实。已有 MVSAnywhere CARLA 微调记录在相同 TEST manifest 和 GT 评测像素下为 1.211 米，当前自研 2.262 米仍明显落后；其图像分辨率、预训练和训练预算不同，不能把此差额全归因于表面选择器。

TEST 的远距比例比 DEV 更低（17.6% 对 25.2%），却在各个距离段都更难；不能把跨序列差异简单解释为“TEST 远物体更多”。现有 manifest 没有地图或世界位姿，因此不能进一步断言两序列来自不同地图。MVSAnywhere 既有时间记录是四参考轮换后约 432 毫秒，自研只输出上相机一张深度；两者的耗时不能当成相同输出任务的速度排名。

我们还检查了是否需要再加入一个原始 Hero 匹配峰：在未参与选择器训练的 DEV25 候选缓存上，即使把它当作可由真值挑选的第四候选，1 米 oracle 覆盖率仅增加 1.57 个百分点；在现有三个候选距真值都超过 4 米的位置，Hero 最高峰距真值小于 1 米的只有 58 个像素。增量主要在远处，但不足以单独解决大错峰，因此本轮保留三候选结构，不为小幅 oracle 上界增加推理和训练成本。原始诊断见 [`initial_surface_hero_omission_20261018.json`](../../runs/candidate_revision_20261018/initial_surface_hero_omission_20261018.json)。

旧方法的一些坏初值原细化器也能修复，因此只报告初值 MAE 不足以证明最终收益。在目前 896 像素宽的匹配图上，焦距约 448 像素、参考上目与源目中心相距约 1.8 米；一个 40 米处的表面移动 0.5 米，只造成至多约 0.25 像素的视差变化，80 米处约 0.06 像素。这解释了远距亚米级精度仍受分辨率、基线、标定和纹理噪声限制；当前结果尚未达到中远距 0.5 米目标。CARLA 原始深度文件的生成脚本缺失，米制标签定义仍有待与采集端核实。
