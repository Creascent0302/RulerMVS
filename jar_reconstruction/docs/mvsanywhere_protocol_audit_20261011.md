# MVSAnywhere 与自研方法的比较口径审计

> 历史版本说明：保留设计与诊断依据；旧缓存和部分旧命令已归档。日常运行使用[当前配置](../configs/current/README.md)，恢复方法见[整理记录](history/project_cleanup_20261019.md)。

本次核对使用 DEV25、已有固定 checkpoint 和官方实际代码；没有读取 TEST，没有修改网络、原结果、过滤阈值或标定。结论是：**目前没有发现足以解释米级差距的 Z-depth / 射线距离混淆、像素中心错位或推理标签泄漏；但原来的密集图与稀疏点投影不是同一个输出任务，不能直接用两者条件 MAE 判定架构高低。** 相同射线查询的补充实验仍显示 MVSAnywhere 明显更准确。

原始采集深度、相机标定和公共指标公式由另两项独立审计负责；本文不重复把那些检查写成自己运行的结论。

## 1. 实际经过的代码与监督边界

MVSAnywhere 路径：

```mermaid
flowchart LR
  A[共同四目 RGB / K / rig→camera] --> B[NianticAdapter：半像素转换、三个真实源视图]
  B --> C[官方 Hero：ResNet 匹配、metadata MLP、图像与代价 ViT、DPT]
  C --> D[官方预测范围细化，再次解码]
  D --> E[原生米制 Z-depth]
  E --> F[按 depth_K 映射共同图像]
  F --> G[参考相机2的 native_depth]
  G --> H[离线 GT 范围与 ROI 评价]
```

- [适配器](../official_methods/methods/niantic_common.py) 的 `forward(inputs)` 不接收 targets；官方当前图和源图字典只有归一化 RGB、内外参等输入。没有历史帧，也没有复制四目成六目。
- [MVSAnywhere 包装](../official_methods/methods/mvsanywhere/adapter.py) 使用 `view_agnostic_mlp_feature_volume`、ResNet 匹配、DINOv2 ViT-B 图像编码、ViT 代价编码与 DPT 解码，不是 dot product 消融模型。
- 训练配置只用有深度标签的参考槽 `[2]`；源视图是另外三张同步 RGB。没有因另外三目缺少 GT 而将当前真实训练损失额外除以四。四参考 `[0,1,2,3]` 只在推理评测时轮换，计算开销计入原完整场景推理。
- 训练专用范围扰动可以使用训练 GT；eval 的 `call_native` 强制 `estimate_depth_range=True`，不采用训练范围字段。该范围是算法的训练设定，须披露，但不是测试标签泄漏。
- adapter 对缺相机或任意无效源图区域明确报错。因此 baseline 的 DEV25 完整帧与自研 DEV54 缺视图结果必须分表；不支持缺相机不能写成其缺相机精度为零，也不能复制图像补齐后比较。

## 2. 深度单位与像素中心：代码与实际解析验证

官方 [BackprojectDepth](../official_methods/upstream/mvsanywhere/src/mvsanywhere/utils/geometry_utils.py) 使用 `arange + 0.5` 网格，并计算 `depth × inv(K) × pixel`；射线没有单位化，其第三坐标等于输入深度。因此原生输出是**相机光轴 Z-depth**，不是欧氏射线距离。外参平移以米输入，输出尺度也是米，不需要 GT 尺度对齐。

项目像素中心采用整数。适配器 `native_intrinsics` 先对 K 主点加 `0.5`，再按原生尺度缩放；`depth_intrinsics` 最后减回 `0.5`。这样同一物理射线在两种坐标记法下相等。官方投影采样使用 `2 × uv / size - 1`、`align_corners=False`，因为这里 uv 已经是半像素坐标，不应再次加 `0.5`。

本轮读取 DEV 第0帧四目真实 K/T、**不加载 targets**，调用官方 BackprojectDepth 与项目整数中心解析式逐像素比较。合成数值深度为 2–80 米斜坡，不是网络输出，也不用于报告精度：

| 检查 | 实测最大差异 |
|---|---:|
| 四目官方反投影与整数中心反投影的相机坐标 | 0.00001526 米 |
| 官方反投影第三坐标与输入 Z-depth | 0.00000763 米 |
| 同分辨率深度映射，2–80 米平滑斜坡 | 0.00000763 米 |
| 同分辨率深度映射，每像素交替 2/80 米的极端棋盘 | 0.00238037 米 |

最后一行是浮点矩阵求逆和双线性采样舍入的放大测试；平均差为 0.00085015 米。它提示“同尺寸映射”并非位级恒等，但远小于本轮米级性能差距。当前没有为了消除这点数值差改动旧后处理或历史报告。

MVSAnywhere 的预测尺度为 `1.0`，原生和共同网格均为 256×448。SimpleRecon 的 `0.5` 尺度则需要真实重采样，不能据此推广为所有 baseline 都没有插值效应。RGB 公共缩放是 `bilinear, align_corners=False, antialias=True`；K 同步半像素缩放。公共标签为 `nearest-exact`，标签重采样的边界敏感性由原始数据审计单独报告。

复跑命令及报告：

```bash
official_methods/.venv/bin/python \
  runs/carla_vggt_20261011/diagnostics/native_mvs_geometry_audit/audit_native_mvs_geometry.py
```

[脚本（历史归档）](../runs/cleanup_20261019/reference_assets/carla_vggt_20261011/diagnostics/native_mvs_geometry_audit/audit_native_mvs_geometry.py)、[实际输出](../runs/carla_vggt_20261011/diagnostics/native_mvs_geometry_audit/report.json)。这次解析核对不等于重新执行完整 baseline 的训练反向或 native/adapter 网络等价性测试；那些已有测试代码在 `official_methods/tests/niantic_smoke.py`，本轮没有重跑，不另记通过。

## 3. 内部深度范围不同，但共同评价范围相同

官方 [sr_depth_model.py](../official_methods/upstream/mvsanywhere/src/mvsanywhere/experiment_modules/sr_depth_model.py) 的 `get_heuristic_depth_range` 根据相机基线、焦距、图宽估计初始范围；`refine_depth_range` 根据自身预测的分位数和边界饱和比例更新范围。本配置额外细化一次。官方 [cost_volume.py](../official_methods/upstream/mvsanywhere/src/mvsanywhere/modules/cost_volume.py) 采用对数间距深度平面，DPT 头通过范围内的对数深度参数化得到米制深度。

DEV 第0帧输入标定对应的初始范围为：

| 参考槽 | 已知基线得出的初始范围（米） |
|---|---:|
| 左 0 / 右 1 | 0.9354–150.0000 |
| 上 2 | 0.9014–93.5414 |
| 下 3 | 0.9014–108.9725 |

因此配置里的 `min_depth=2, max_depth=80` **不能解释为 MVSAnywhere eval 的硬候选范围**。共同评价确实只用 GT 在 2–80 米且进入固定 rig ROI 的像素；预测超出这个范围仍计入误差，不会删除。自研模型内部搜索范围是 2–80 米，两者内部参数化和计算预算不同。这是保留原方法带来的明确差异，不是隐藏 GT 输入。若未来比较“相同内部范围”，应另建消融配置，不能悄悄改原生算法或重写旧结果。

## 4. 输出有效性、稀疏点投影与遮挡

官方 adapter 交出的 `depth_valid` 为有限正深度及图像有效性。它没有把 `overall_mask_bhw` 当拒绝掩码，也没有人为生成置信度。原生 overall mask 只是成本体投影覆盖诊断，并非最终深度的遮挡可见性证明，更不是全局空间 free/unknown 真值。密集图像先验允许模型在弱匹配处继续预测。

自研最终输出同时有 `valid`、`state` 与来源参考相机信息。当前有三类输出：

- `coarse_depth_diagnostic`：中间粗深度，不是最终重建。
- `unfused_reference_query_depth`：仅参考相机2自己的末级射线查询，保留其 `reference_depth`。不把其他相机可见背景强行解释为参考相机的第一表面。
- `final_point_splat`：所有来源相机的最终有效表面点投到参考相机，四舍五入到单像素，取最小 Z；不补洞。

第三种输出有真实的表示局限：如果前景没有在某个像素生成点，其他视角看到的背景可能投进该空洞。最小 Z 缓冲只能在已存在点之间处理遮挡，不能凭稀疏点证明参考视角无遮挡。因此 final_point_splat 的背景深度误差，不一定全是原始射线估计错误；它也可能来自稀疏采样、跨相机遮挡和融合。不能用 GT 删掉这些点来提高成绩，应单独改进后处理并保留原输出诊断。

`native_depth` 与 final_point_splat 的 MAE 分母覆盖区域不同。比较必须同时报告覆盖率、固定 GT 分母成功率和表示定义，不能只比较条件 MAE。占据与 free/unknown 也不能由 dense depth 能力列表伪造为该网络原生输出。

## 5. 相同查询的实际复核：差距仍存在

为排除密集与稀疏位置不一致，本轮 GPU1 将 field200 最后阶段、参考槽2的全部查询作为共同位置，再运行固定 MVSAnywhere checkpoint。查询由 field200 输入预测选取，GT 只决定共同有效评价分母。工具没有按自研是否接受先筛掉失败查询。

- 25 帧共 6400 个查询，3680 个有范围及 ROI 内有效标签；坐标全部为整数，无重复。
- 因此 baseline 的 `round` 最近邻取值与自研的连续 `reference_depth` 在**这次相同整数射线**上可直接比较，没有半像素采样差。若未来产生小数 UV，现工具的最近邻会引入额外偏移，需要另外明确共同射线采样协议。
- 两个 checkpoint SHA 均核对；baseline 实际配置完整等于原报告保存配置。没有修改权重、后处理或阈值。

| 相同查询结果 | field200 | MVSAnywhere step1000 |
|---|---:|---:|
| 有效 GT 查询 | 3680 | 3680 |
| 接受查询 | 3622 | 3680 |
| 拒绝的有效 GT 查询 | 58 | 0 |
| 接受查询 MAE（米） | 4.054784 | 1.007420 |
| 接受查询 RMSE（米） | 9.044108 | 2.325205 |
| 误差严格小于 0.2 米的查询 | 823 | 1105 |
| 以全部 3680 查询为分母的成功率 | 22.3641% | 30.0272% |

自研 MAE 的有效分母为 3622，baseline 为 3680，不能说两者 MAE 已完全同掩码；这里同时列出拒绝数及严格相同分母成功率。后者也显示差距。另从已归档逐查询数组独立用 NumPy 复算双方共同接受的 3622 条查询：field / baseline 的 MAE 为 4.054784 / 1.007041 米，RMSE 为 9.044108 / 2.332415 米，严格小于 0.2 米的命中数为 823 / 1083。因此这 58 条拒绝并不能解释主要差距。该共同接受交集诊断不得替代原固定分母结果，数值见 [common_accepted_query_audit.json](../runs/carla_vggt_20261011/diagnostics/native_mvs_geometry_audit/common_accepted_query_audit.json)。

原始逐查询数组、独立双精度复算及 SHA 核对在 [field_vs_mvsanywhere（历史归档）](../runs/cleanup_20261019/reference_assets/carla_vggt_20261011/diagnostics/field_vs_mvsanywhere/README.md)。双精度逐点复算与原指标小于 1e-8 米的差只来自数值累计方式。这组位置由自研模型选择，不是均匀全图测试；它不能代表小障碍物实例指标，也不是论文的官方测试协议。

## 6. 目前能与不能归因的事情

同一四目、同一分割、同一图像大小、同一米制范围与同位置诊断，排除了部分明显评测混淆。它支持“当前固定的 field200 在这些查询上仍明显落后于这个强 baseline”。不能据此独立证明某个模块在充分训练后必然更差。

两者训练与容量不对等：MVSAnywhere 使用完整官方 Hero 预训练，再在 TRAIN66 微调 1000 步，FP32；自研仅引入冻结的官方匹配 ResNet 子树，其他模块延续本项目较小网络的训练链，再进行本轮 200 步 BF16 新优化器对照。自研配置来自 `feature_context_200` 继承链，不能把这次 200 步当从零训练的总预算。图像先验、代价上下文、参数量、预训练数据和训练目标同时变化，不能把差距全部归因于一个 attention 写法。

更可靠的下一步是：固定同一个父 checkpoint、训练采样、步数、精度和 optimizer，对自研候选模块做 control/field 配对训练；用相同参考射线、固定分母和拒绝率验证，再用独立新轨迹做最终冻结评估。与 MVSAnywhere 的比较继续标注完整预训练及密集输出条件。当前没有可支持“超越 baseline”或“小障碍物全面提升”的证据。

## 7. 单帧开销复核

已在 GPU1 用同一 DEV 第0帧、仅 inputs，strict 加载 control200/field200；FP32 禁 TF32，各预热3次，8轮正反顺序交替。结果与限制见 [配对计时（历史归档）](../runs/cleanup_20261019/reference_assets/carla_vggt_20261011/diagnostics/paired_timing/README.md)。forward 中位 1283.93 / 1300.54 ms，包含 `prediction_streams` 后为 1288.35 / 1305.20 ms；主机同期共享 GPU2/3 训练资源，波动很大，不能认作稳定的全场景速度差。

field 仅多 17126 个参数，但本次调用的 PyTorch allocated 峰值增量从 399.33 MiB 增至 989.58 MiB，8次均相同。候选上下文显式保留深度维度和邻居 key/value 激活，与这个开销方向一致；未做算子级 profile，不将全部耗时差归因于某个算子。两模型共驻，不能把总峰值当成单模型独占显存。
