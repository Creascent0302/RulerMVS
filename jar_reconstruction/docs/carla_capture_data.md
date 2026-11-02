# 当前 CARLA 数据整理与缺失输入使用说明

本次仅使用 `stereo_left / stereo_right / stereo_up / stereo_down` 四台广角。`tele_left / tele_right` 不参与输入、完整帧定义或训练。原始数据只读，没有移动、重命名或复制图片；生成的是指向原文件的清单。

**2026-10-26 新采集补充：** 下文现有清单和数量均指旧的两条采集，不能直接用于新 `captured_data0/1/2/3`。新数据每台广角各有独立深度，原始转换器尚需适配。参考采集脚本的解码和前三条9帧跨相机核验共同支持 `depth_type=optical_z`、`depth_encoding=metric_npy`、`depth_scale=1.0`；不要再按射线距离做余弦校正。近1000m饱和值需排除，四相机标签不能互相复制。参考脚本的雷达距离/速度列读取反了，因此其保存的radar.npy不能直接按正确XYZ读取；不影响上述图像深度结论。第四条尚未完成上传完整性与深度几何核验。详见[研究记录126](research_memory.md#实验-126-参考采集脚本与新四目深度交叉核验采用米制光轴深度独立发现雷达列序错误)及[几何证据](../runs/active/carla_collector_audit_20261026/geometry/results.json)。

## 1. 整理结果

数据根目录是 `/home/liuzonghao/3Dscene/data/CARLA_data`。处理结果位于项目中的 [data/carla_wide4](../data/carla_wide4/report.json)。

| 轨迹与用途 | 索引总帧数 | 四目完整 | 缺广角图像但至少有一台 | 完全没有广角 | 可直接进行多视角深度监督的候选 |
|---|---:|---:|---:|---:|---:|
| `captured_data (1)`，训练 | 398 | 98 | 296 | 4 | 319 |
| `captured_data`，测试 | 456 | 200 | 251 | 5 | 350 |

“完整”有两个独立定义：`camera_completeness` 表示四台广角是否齐全；`sensor_completeness` 还要求 depth 和 radar 同帧存在。当前两组数据里恰好四目完整帧也都有 depth/radar，这不意味着代码把这两个概念混成一个。

有效相机数量的分布：

| 用途 | 1 台 | 2 台 | 3 台 | 4 台 |
|---|---:|---:|---:|---:|
| 训练轨迹 | 25 | 82 | 189 | 98 |
| 测试轨迹 | 27 | 73 | 151 | 200 |

训练候选 319 帧包含 98 个完整帧和 221 个缺相机帧。不是只用完整帧训练。深度类型已通过同帧雷达对照作经验确认，监督入口已启用，精度限制见第 3 节。

### 为什么没有直接扫描所有文件作为样本

`captured_data` 中混有另一条轨迹的旧文件。在排除长焦后，有 **1934 个文件不属于本目录的同步索引**；抽查六个重复广角文件，内容 SHA-256 与 `captured_data (1)` 中对应文件相同。它们保留在原目录，但不进入此轨迹清单，详见 [未索引文件清单](../data/carla_wide4/test_unindexed_files.json)。

整理以各自 `sync_index.json` 的真实 frame ID 为准，并核对文件名和实际存在性。不使用前后邻帧补图，不把同一相机的图复制到缺失槽，不再按原 `matched_frames.json` 的八传感器齐全标准筛选四目数据。

按整条轨迹划分训练和测试，避免相邻帧随机拆分。按用户当前安排，先只做训练和测试，`val_manifest` 保留空字符串，后续再采验证数据。训练器不依赖验证集：预先固定步数，使用最后一步 checkpoint 测试，不利用测试指标早停或挑选权重。当前配置 1000 步只是开发起点，尚未执行充分训练。这只是序列隔离：现有文件没有地图 ID、路线 ID 或车辆世界位姿，尚不能证明两条轨迹没有空间重叠。

## 2. 每种清单怎么用

每个清单都是 JSONL：一行对应一帧。它只保存路径、标定和可用性，不重复保存图像。

| 文件后缀，前缀为 `train_` 或 `test_` | 用途 |
|---|---|
| `all.jsonl` | 至少一台广角存在的全部帧；完整和缺失混合，适合统一推理 |
| `complete.jsonl` | 四台广角均存在，用于完整输入单独测试 |
| `partial.jsonl` | 仅 1–3 台广角存在，用于缺失输入单独测试 |
| `supervised.jsonl` | 至少两台 RGB，其中有 stereo_up，并且同帧 depth 存在；监督训练候选 |
| `inference_only.jsonl` | 不满足上述直接监督条件，仍保留用于推理 |
| `single_view.jsonl` | 只有一台广角；检验退化行为，不能视作可靠的多视角米制重建 |
| `sensor_complete.jsonl` | 四广角加 depth/radar 齐全，长焦不参与定义 |
| `no_rgb.jsonl` | 只有其他传感器等、没有广角；仅作库存记录，不可传给 RGB 模型加载器 |

一帧的稳定槽位始终是 `[stereo_left, stereo_right, stereo_up, stereo_down]`。例如缺少右相机时，第二个槽是零值占位，且 `view_mask / camera_valid=false`、`image_mask=false`。这只是保持张量形状，并不是有效的黑色观测，也不计入支持视角数。

`timestamp` 原始文件没有保存，清单写 `null` 并注明 `sync_basis=frame_id`、`timestamp_source=not_recorded`。没有用 frame ID 冒充秒数。

## 3. 深度类型的经验核查与剩余误差

2026-10-08 已比较两条轨迹的同帧 radar/depth。结果支持 **`depth_type=optical_z`、数值为米量级**，即沿相机朝向测量的深度，不是从相机中心斜着量到物体的距离。当前清单已经启用这一解释，`depth_scale=1`。`supervision_ready=true` 表示允许基于这个有证据的解释训练，不代表采集解码代码已经确认，也不代表标签达到毫米级精度。

对照步骤与约束：

1. 只匹配同一个 CARLA frame ID，使用保存的外参转换两个传感器中心。雷达高度 1 米，深度相机高度 2.5 米，不能直接拿雷达距离与同一个像素的数值相减。
2. 雷达 NPY 并非官方原始四列格式。前三列的长度与测距不符，静止时经常全零；经验上它们与射线方向平行，第四列为距离。对非零前三列归一化、消除速度符号后乘第四列，得到候选三维点。这种格式解释有投影一致性支持，但未找到采集脚本确认列的原始语义。
3. 转到相机坐标后，分别计算光轴深度与光心到点的距离，并投影到深度图。图外点、无法恢复方向的零向量及局部深度跳变大于 2.5% 的位置不参与比较。两种假设使用完全相同的点；不按匹配误差筛选“正确点”，也不在附近搜索最接近的深度值。
4. 分别按原 K 与半像素偏移检查敏感性。半像素检查和拟合比例仅用于诊断，**没有修改 K 或标签缩放**。测试轨迹的对照仅用于确认传感器格式，没有使用网络预测，也未据此调模型超参数。

按当前清单 K 的实际结果：

| 项目 | 训练轨迹 | 测试轨迹 |
|---|---:|---:|
| 同帧 radar/depth 数量 | 391 | 446 |
| 参与两种假设比较的点数 | 164681 | 186457 |
| 光轴深度解释：中位相对误差 | 0.4433% | 0.4781% |
| 射线距离解释：中位相对误差 | 7.7647% | 11.1445% |
| 光轴深度解释：中位绝对误差 | 0.02855 米 | 0.03361 米 |
| 光轴深度解释：相对误差小于 2% 的比例 | 91.19% | 63.80% |

这张表是**传感器一致性检查，不是模型性能**。测试轨迹仍有较多离群点；碰撞几何、遮挡和渲染表面差异都可能产生不一致，不能把全部雷达返回当精确表面真值。

现有深度为 `float32[1080,1920]`，最大值约 1005.9959。半像素约定下，光轴解释的中位比例偏差约 0.594%；诊断拟合系数约 0.99410，但原 RGB24 字节及解码脚本缺失，无法唯一确定成因。**当前保留原数值，不按最大值归一化，也不自动应用拟合系数；大于等于 1000 的值继续作为无效标签屏蔽。** 后续有脚本时应复核解码和像素中心约定，厘米级指标需考虑这项标签误差。

可复现代码是 [audit_carla_depth.py](../tools/audit_carla_depth.py)，完整统计在 [训练轨迹报告](../data/carla_wide4/depth_audit.json) 与 [测试轨迹报告](../data/carla_wide4/depth_audit_test.json)。清单中的 `depth_evidence` 保存报告路径、摘要和 `capture_decode_formula_verified=false`；整理器核对报告与原标定、同步索引的摘要，避免误用旧证据。未提供明确深度类型的新数据仍默认 `unverified`，不会自动放行。

[CARLA 官方传感器说明](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera) 给出了原始 RGB24 到米制深度的规则及雷达原始字段，但不能证明当前 NPY 的保存公式。相机旋转转换根据 [官方 Transform::GetMatrix](https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/geom/Transform.h) 的约定实现。

当前标定中，独立 depth 相机与 **stereo_up** 的位置、姿态、K、分辨率相同。因此只把该深度用作 stereo_up 的直接像素监督，没有把同一张深度图复制到左右、下方图像。如果 stereo_up RGB 缺失，即使独立 depth 存在，该帧仍列入库存和推理清单，但暂不从它生成其他视角的深度真值：未经可见性验证的重投影可能给遮挡区域制造错误标签。

雷达路径和标定保留在 manifest 的 `auxiliary_sensors`，不作为模型输入，也不把雷达数组当成激光雷达点云或稠密占据真值。本次网络仍是 RGB＋已知标定输入。

## 4. 训练和推理接口的修改

数据接口复用现有 [MultifocalFrameDataset](../jar/multifocal/data.py)。虽然文件名含 multifocal，这里只启用四目协议，长焦分支关闭。新增 [整理工具](../tools/prepare_carla_capture.py) 和 [几何缩放模块](../jar/data_transforms.py)。

主要处理规则：

1. **有 RGB、没深度标签：** 输入仍可参与多视角匹配；缺少的标签使用独立的监督掩码。
2. **参考图有深度、源图没深度：** 参考深度仍能监督条件深度分布，但不把未知可见性训练成 `no-match`。新增损失名是 `coarse_depth_class_partial`，不直接监督 no-match 通道。
3. **整个训练窗口没有监督：** 跳过优化器更新并记录 `no_supervision`，不增加优化步数；避免 AdamW 在零监督时仍衰减权重。如果整个完整 epoch 都没有监督，则明确报错，不进入无限空训练。
4. **多卡中部分卡没有标签：** 保持一致的分布式反向通信；只要全局窗口有监督就可以更新。实际进行了这一组合测试。
5. **缺失源相机：** 不参与投影匹配、跨视图支持计数或空闲证据。单目能够运行，但深度证据不足时保留 unknown。
6. **测试缺少标签：** 仍保存预测和 unknown；不伪造深度误差或 F-score。

缺相机训练可能帮助适应传感器失效，但不保证泛化一定更好。无标签图像也不会因为送入训练器就自动产生学习信号；本次没有新增自监督学习算法。默认训练清单使用可监督候选，避免大量空训练窗口。

当前只保留 MF-ET-JAR，四目模式使用其基础解码路径。官方 baseline 的缺失视图支持受各自原生网络限制，不能把自研模型测试解释成所有 baseline 都支持缺相机。

## 5. 图像大小和标定

原始图像为 1920×1080。新增开发配置显式使用 `data.image_size=[270,480]`（高、宽），并同步更新 K；原文件没有被缩放覆盖。

- RGB 使用抗混叠双线性缩放。
- 深度、有效掩码和类别使用不混合前后表面的采样方式。
- 像素中心转换为 `u_new=(u_old+0.5)*sx-0.5`，垂直方向同理。
- 射线距离转换到光轴深度前，先在原始米制标签上排除饱和/无效值，防止天空饱和值因投影缩短后变成有效深度。
- 坐标统一为右手 rig：x 前、y 左、z 上；CV 相机：x 右、y 下、z 前。平移和表面点为米，内参为当前图像像素。

例如原 `fx=960,cx=960`，缩放四分之一后为 `fx=240,cx=239.625`。代码没有简单地把所有 K 元素除以四。

## 6. 可直接运行的命令

在 `/home/liuzonghao/3Dscene/jar_reconstruction` 下执行。本次使用现有 `official_methods/.venv/bin/python`，没有修改驱动或环境。

先对全部完整/缺失帧做推理：

```bash
official_methods/.venv/bin/python infer.py \
  --config configs/carla/mf_et_jar_wide4.json \
  --checkpoint checkpoints/wide4/et_v4_seed17.pt \
  --device cuda --limit 8 --output runs/carla_wide4/infer
```

已有权重来自之前的合成场景训练，尚未在这批 CARLA 数据上训练，不能把这些输出当成已验证精度。

只测四目完整输入，使用 `configs/carla/mf_et_jar_wide4_complete.json`；只测缺相机输入，使用 `configs/carla/mf_et_jar_wide4_partial.json`。它们的模型参数完全相同，只更换测试清单。默认 `allow_missing=true` 已由当前任务明确启用；关闭它时，缺相机清单会报错。

复现只读传感器核查，再按已经检查过的结果重新整理清单：

```bash
official_methods/.venv/bin/python tools/audit_carla_depth.py \
  --root /home/liuzonghao/3Dscene/data/CARLA_data \
  --sequence 'captured_data (1)' --output data/carla_wide4/depth_audit.json

official_methods/.venv/bin/python tools/audit_carla_depth.py \
  --root /home/liuzonghao/3Dscene/data/CARLA_data \
  --sequence captured_data --output data/carla_wide4/depth_audit_test.json

official_methods/.venv/bin/python tools/prepare_carla_capture.py \
  --root /home/liuzonghao/3Dscene/data/CARLA_data \
  --output data/carla_wide4 \
  --train-sequence 'captured_data (1)' --test-sequence captured_data \
  --depth-type optical_z --depth-scale 1 \
  --depth-evidence data/carla_wide4/depth_audit.json data/carla_wide4/depth_audit_test.json
```

这一命令只适用于本次已经检查的采集数据。工具默认仍为 `unverified`，不会给后续未知数据自动套用结论。重新生成仅更新清单，不修改原始 NPY。诊断报告中的拟合值不自动成为 `depth_scale`。

当前可运行真实数据训练和评测。若只做两步短测，先把所选 JSON 的 `train.max_steps` 改为 `2`，再运行：

```bash
official_methods/.venv/bin/python train.py \
  --config configs/carla/mf_et_jar_wide4.json \
  --init-checkpoint checkpoints/wide4/et_v4_seed17.pt \
  --device cuda --output runs/carla_wide4/train_smoke

official_methods/.venv/bin/python eval.py \
  --config configs/carla/mf_et_jar_wide4.json --split test \
  --checkpoint runs/carla_wide4/train_smoke/last.pt \
  --device cuda --output runs/carla_wide4/eval
```

注意 `--split test`：当前没有独立验证轨迹，默认 `val` 入口会因路径为空报错。`--resume` 用于相同数据合同的断点续训；从旧权重开始新的真实数据实验用 `--init-checkpoint`。

开发配置保留旧 ET 的 2–24 米光轴深度搜索，不声称覆盖整个远距场景。改搜索范围时需要同时修改候选采样、权重兼容与评测范围，而不能只改变数据路径。

## 7. 部分标签评测规则

CARLA 配置启用 `eval.observed_only=true`：只在有 GT 首次命中/已观测空闲证据的区域计算表面精度，未标注相机视野不自动算错；深度指标按配置 ROI 裁定标签范围。unknown 比例仍统计模型输出，不通过删除 unknown 美化指标。这与原先拥有多视图标签的评测范围不同，结果不能直接混比。

推理/评测 JSON 额外包含 `by_camera_completeness` 和 `by_valid_camera_count`，分别统计完整/缺失以及 1/2/3/4 有效视角。没有标签的帧仅提供推理统计；不能从这些帧推导真实几何精度。

## 8. 验证与未完成事项

运行 `python -m pytest tests/test_carla_capture.py tests/test_ablation_controls.py -q` 检查相机缺失、尺寸变换、标签隔离和主动禁用。分布式空监督处理由 `tests/ddp_worker.py` 验证；这些测试不能代替真实精度评测。

本次实际运行 `official_methods/.venv/bin/python -m pytest tests/test_carla_depth_audit.py tests/test_carla_capture.py tests/test_data_losses.py tests/test_multifocal_data.py -q`：**40 passed，3 subtests passed**，包括解析平面的两种深度假设、零方向雷达排除、传感器外参转换、证据摘要及原有数据/损失测试。

以下为配置接口精简前实际运行的历史命令（RTX 5090，BF16，319 帧训练清单）。现版本重跑需删除 `--steps`/`--warmup`，改在 JSON 中设置 `train.max_steps=2` 和 `eval.warmup=1`，并使用新的输出目录。

```bash
official_methods/.venv/bin/python train.py \
  --config configs/carla/mf_et_jar_wide4.json \
  --init-checkpoint checkpoints/wide4/et_v4_seed17.pt \
  --device cuda:0 --steps 2 --output runs/carla_depth_audit_20261008/train_smoke
```

日志和 checkpoint 位于该输出目录，损失与梯度均有限。这只是训练管线检查，两个不同样本的损失不能用来判断是否收敛。原清单备份在 `/tmp/carla_depth_before_20261008_e_nl8n40`。

随后实际加载该 checkpoint 执行 GPU 评测：

```bash
official_methods/.venv/bin/python eval.py \
  --config configs/carla/mf_et_jar_wide4.json --split test \
  --checkpoint runs/carla_depth_audit_20261008/train_smoke/last.pt \
  --device cuda:0 --limit 8 --warmup 1 \
  --output runs/carla_depth_audit_20261008/eval_with_labels_smoke
```

8 帧覆盖 1/2/3/4 个有效相机，其中 2 帧具有可对齐深度；权重完整加载，无缺失或多余参数。无标签帧没有生成深度精度数值；单目帧保留 unknown。有标签帧已产生有限指标，但当前稀疏预测只覆盖约 0.135% 的有效标签像素，且仅训练两步，**不能将该次冒烟输出作为充分训练后的性能结论**。更早的两帧检查只覆盖无标签单目，保留在 `eval_smoke/`。

仍需采集脚本确认原解码公式和系统偏差，未来补充独立验证轨迹、地图/路线信息，并完成充分训练、精度与八卡吞吐实验。暂缺验证集不再阻塞当前训练；训练使用缺失数据也不会自动保证泛化提升，需要与完整帧训练对照。
