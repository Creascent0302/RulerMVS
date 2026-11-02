# 可选几何后处理插件 MetricPolish

> 历史版本说明：保留设计与诊断依据；旧缓存和部分旧命令已归档。日常运行使用[当前配置](../../configs/current/README.md)，恢复方法见[整理记录](project_cleanup_20261019.md)。

该插件接收其他模型已经预测好的深度，用四目图像中的匹配证据做局部修正。官方模型、自研主体和插件相互独立；启用插件不会改写官方网络结构、权重或原生输出。

当前结果属于 **MVSAnywhere 完整网络 + 高分辨率匹配特征 + 几何插件的组合系统**，不能称为自研网络独立超过官方模型，也不是相同计算预算的比较。

插件也已实际接入自研模型：完整DEV25平均误差1.102226→1.029638米，但20厘米内比例基本不变、边界略退。不同接法、解释与运行配置见自研接入对照（历史内容已归档，见[整理记录](project_cleanup_20261019.md)）。该实验不调用完整MVSAnywhere。

## 输入、输出与实现边界

实现位于 `jar/metric_polish.py`。插件只从自研 checkpoint 严格提取 `pretrained_matching.*` 和 `dense_decoder.refinement.*`，本次权重分别包含 37 和 11 项状态。不创建或执行自研 CNN、粗候选网络、稠密解码主体。

`forward(inputs, initial_depth, reference_indices)` 的参数为：

- `inputs`：项目现有 `rgb`、`K`、`T_cam_from_rig`，以及可选 `view_mask`、`image_mask`。还需提供同一原图独立缩放的 `matching_rgb` 和 `matching_image_mask`。默认场景图像为 256×448，匹配图像为 512×896。
- `initial_depth`：`[B,R,H,W]`，单位米，表示相机光轴方向的深度；必须已经按标定映射到输入 RGB 像素格点。
- `reference_indices`：R 个相机槽位，例如 `[2]` 或 `[0,1,2,3]`，顺序与深度第二维对应。

插件不接受或读取 `targets`。缺失相机通过 `view_mask` 屏蔽，不复制图像补齐视图。支持批量 B 和缺少参考/源相机的情况；缺失参考的 `valid` 为假。无源证据不会产生有效更新。非有限或非正的初始预测返回无效掩码和有限占位值，不能当作重建结果。

输出保留：

- `depth`、`valid`、`reference_indices`：修正后深度及参考图像预测有效性。
- `before_anchors`、`after_anchors`、`anchor_valid`：原图 `4*j` 格点上的细化前后深度。
- `inverse_residual`：锚点上的原始逆深度修正量。
- `passes`：每轮的门控、米制搜索步长、源视图支持数、深度更新量。

`valid` 不代表已通过多视图确认的真实表面。插件不输出 free、occupied 或 unknown 标签，不能把这些深度预测直接当作占据真值。未知初值须用 0 或 NaN 表示。若原模型提供 `depth_valid`，调用入口先把无效位置置 0，不能因为插值得到有限正值就重新宣布有效。

当前默认算法固定为零语义 query、观测证据门控、倍率 1、九个精确投影候选、两轮最大跨度 12/4 米。零 query 仍保留旧线性层偏置及深度条件分支。候选间距使用原 stride4 几何尺子。最终只把逆深度修正插值到原始完整深度图，保留初始图像中的细节；零修正逐值保持原预测。所有参数冻结，但初始深度仍可反向传播。

配置可选 `residual_filter: "mean3"`，默认 `"none"`。mean3 只对逆深度修正做横向、纵向三点平均；各方向端点保持原值。它不平滑初始深度，不使用大矩阵拟合。该开关与默认结果分别记录，不能套用先前仿射拟合实验的指标。

## 四个阶段的尺寸与计算

1. **先取得任务模型的完整预测。** 官方网络输入四张 `[B,4,3,256,448]` RGB 和米制标定，轮换参考视角，深度映射到 RGB 格点后为 `[B,4,256,448]`。这是外部初值，不是标签。
2. **只提取额外匹配描述子。** 同一组原始图像独立缩放成 `[B,4,3,512,896]`，冻结 Hero 编码器输出 `[B,4,16,128,224]`。不能把低分辨率图像放大假装增加信息。投影采样显式同步两种图像尺寸及像素中心。
3. **只在 stride4 锚点细化。** 每个参考视角有 `64×112=7168` 个锚点，初始深度为 `[B,R,64,112]`。每点的语义 query 是 64 维零向量；两轮分别允许最大 12 米和 4 米的局部窗口。每轮九个候选均通过真实标定精确投影，读取相机中的描述子，再按证据决定更新。按 4096 点分块，不为每个像素写 Python 循环。
4. **运输修正量，保留初始图像细节。** 先求锚点上的逆深度修正 `[B,R,64,112]`。可选 mean3 只平滑这张修正图，再按 `4*j` 对齐插值到 `[B,R,256,448]`，加回原始完整预测的逆深度。这里没有重新平均初始前后景深度。边界仍可能受到邻域修正扩散影响，不能保证小物体完全恢复。

局部门控为 0 的像素可能接收相邻锚点运输来的修正，这是邻域借用的深度假设，不是该像素得到了直接多源验证。全片缺源、纯旋转或无纹理导致所有修正为 0 时，则整片逐值保留初始深度，包含 1 米和 100 米这类搜索范围外初值。

## 使用方式

参数通过 JSON 控制。实际推理示例配置为：

`configs/module_research_20261015_r2/metric_polish_infer_example.json`

```bash
official_methods/.venv/bin/python tools/apply_metric_polish.py \
  --config configs/module_research_20261015_r2/metric_polish_infer_example.json \
  --device cuda:0
```

该入口在加载阶段设置 `load_targets=False`，不打开监督文件。每个 `.pt` 分别保存 `native`、`native_on_image_grid`、`metric_polish`，不会悄悄把插件输出写成官方结果。结果目录通过现有 `runs/module_research_20261015_r2` 软链接存放到数据盘。

在其他代码中也可直接使用：

```python
from jar.metric_polish import MetricPolish

plugin = MetricPolish(checkpoint_path, {"residual_filter": "mean3"}).to(device).eval()
result = plugin(inputs, depth_on_rgb_grid, reference_indices=[0, 1, 2, 3])
```

## 实际验证记录

修复版 DEV25 使用 2,087,873 个共同 GT 像素，所有预测覆盖率为 100%。raw 与 mean3 来自同一次细化 forward：

| 输出 | MAE/米 | RMSE/米 | 20 厘米准确率 |
|---|---:|---:|---:|
| 官方原始预测 | 0.938387 | 2.288040 | 38.3141% |
| 官方仅固定裁剪 2–80 米 | 0.938342 | 2.287799 | 38.3141% |
| 零 query 插件，逆深度修正读出 | 0.861863 | 2.322114 | 46.4320% |
| 零 query 插件，mean3 修正读出 | **0.764464** | **2.187421** | **51.4528%** |

未平滑 raw 版的 RMSE 退步；最终显式启用 mean3 的组合在本次 DEV25 的 MAE、RMSE、20 厘米准确率及近中远/边界 MAE 都优于该低分辨率官方预测。这只是当前数据划分与固定权重的结果。固定范围裁剪对照只改善 0.000045 米，不能把它的贡献说成几何匹配收益。

最终逐帧记录位于 `runs/module_research_20261015_r2/metric_polish_corrected_dev25`。修复前 raw、mean3 及范围对照分别保留在 `metric_polish_dev25`、`metric_polish_mean3_dev25`、`metric_polish_controls_dev25`。直接逐全分辨率像素做独立细化的三帧负结果保留在 `metric_polish`，没有加入生产插件。

## 两项接口修复与历史等价性

- 原细化核心即使没有证据、门控为 0，也无条件裁剪到 2–80 米。插件外层现在只在门控为正时接受核心更新，无证据时逐值保留当前初值。原核心文件没有修改，已有研究历史保留。该修复同时阻止未观测天空的裁剪修正通过空间平滑扩散到其他位置。
- 推理入口保留原模型 `depth_valid`，先把声明无效的初值置 0，防止有限正数插值重新激活未知预测。原生输出仍独立保存。

`metric_polish_module_validation` 的历史逐值等价是在修复前完成。修复后有意不再要求无证据位置等于旧错误裁剪；修复版本标记为 `no_evidence_identity_v1`。新验收检查正门控位置的原子更新与原核心逐值相同，以及零门控位置保留输入。修复前后完整 DEV25 已分别运行，最终报告使用修复版字节哈希和指标，不覆盖旧记录。

以下命令已实际执行：

```bash
official_methods/.venv/bin/python -m pytest -q \
  tests/test_metric_polish.py tests/test_metric_polish_module.py

official_methods/.venv/bin/python tools/validate_metric_polish.py \
  --config configs/module_research_20261015_r2/metric_polish_corrected_validation.json \
  --device cuda:0
```

修复前真实三帧 0/12/24：模块与原实验全图张量逐值相同，各距离分组指标逐字段相同。GPU 初始深度梯度有限且非零；插件所有参数冻结，没有参数梯度。修复后 CPU 11 项测试通过，覆盖批量 B=2、缺源/缺参考、禁止读取监督、零残差恒等、投影与尺度采样，以及缺源/纯旋转/无纹理 × none/mean3 × 超范围 1/100 米初值的无证据恒等。

修复版 GPU 验收中，第一轮 6310 个正门控锚点的核心输出逐值保持，858 个零门控锚点逐值保持初值；初始深度梯度有限，99.2379% 像素非零。历史三帧 GT 指标仍相等，但未评价区域的全图预测有意发生变化，不能继续声称修复后与旧输出全图逐值相等。

固定 DEV0、3 次预热和 10 次重复的修复版计时：官方四参考网络加适配器平均 742.49 毫秒；四视图描述子加四参考插件平均 91.68 毫秒；真正串联执行的完整四参考流水线平均 837.71 毫秒，峰值已分配显存约 1340.64 MiB。完整流水线直接测量，未把两个独立时间相加。此基准使用默认 none 读出，不冒充 mean3 的精确时延。

修复前较低共享负载的一次完整流水线实测为 576.21 毫秒；另一次单参考插件为 24.85 毫秒，不能当作完整场景速度。全部原始样本保存在对应 validation 目录。服务器其他 GPU 有并行任务，CPU/PCIe 未严格隔离，不能挑选快的一次当作稳定速度。

最终 `apply_metric_polish.py` 示例已实际运行 mean3 四参考无标签推理：输出 `[1,4,256,448]` 有限深度，参考 2 与修复版 DEV25 第 0 帧逐值相同，无效原生预测被重新激活的数量为 0。输出为 `metric_polish_corrected_infer_example/frame_00000.pt`，实际位置在数据盘。

尚未验证真实道路泛化、其他数据集、更多训练种子或严格隔离的速度。没有长训练，也没有使用 TEST 调参。
