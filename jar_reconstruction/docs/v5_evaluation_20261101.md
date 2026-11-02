# V5正式测试与独立baseline训练验收（2026-11-01）

生成时间：2026-11-01 09:46:46 北京时间

V5训练及原先排队的六组完整TEST已结束，退出码0；本次核验134个源码/权重/清单文件，并重新补测真实四参考速度、174帧完整四目共同子集和PatchmatchNet官方权重迁移。测试未用于选权重。

V5相对V2改善了CARLA上相机，但尚未全面超过MVSAnywhere。它对近距和平滑区域的改善更明显，中距与遮挡边界仍是短板；下相机在共同像素模式下还不及V2。公共数据不能跨单位计算一个总米制MAE。

## 共同四目CARLA测试：174帧，256×448共同RGB信息

只输出上相机深度，范围2–80米；所有方法GT掩码与帧完全相同。V2/V5匹配分支仍运行512×896张量，但这里只能从共同256×448图像上采样，不能读取额外高分辨率信息。速度单列计时来自固定完整VAL首/中/末三帧；不使用精度评测中多GPU并行的耗时。

| 方法 | MAE/m ↓ | RMSE/m ↓ | AbsRel ↓ | 有效预测覆盖率 | F-score@0.1m ↑ | 单参考耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| V5（VAL-best） | 2.15650 | 5.88845 | 0.17216 | 100.00000% | 0.43120 | 32.66 ms |
| V2（20261029） | 2.72795 | 6.24501 | 0.27576 | 100.00000% | 0.32892 | 31.62 ms |
| MVSAnywhere 一次细化 | 1.93100 | 6.05562 | 0.11282 | 100.00000% | 0.45044 | 37.06 ms |
| PatchmatchNet 官方DTU | 5.11394 | 10.74118 | 0.81559 | 99.99506% | 0.22289 | 未独占补测 |
| Fast-MVSNet 官方DTU | 25.13075 | 30.99935 | 4.36863 | 100.00000% | 0.02785 | 未独占补测 |

PatchmatchNet存在少量无效预测，表中MAE/RMSE是有效预测条件误差，不能写成100%GT全覆盖误差；无效部分计入覆盖率和阈值准确率失败。DTU发布权重没有本轮CARLA适配，不能把此表解释为各架构在同样充分训练后的上限。MVSAnywhere完整任务预训练与V5训练经历也不同。

## 完整181帧CARLA上相机，与历史结果严格对应

**native**：自研匹配RGB512×896，baseline RGB256×448；不属于相同信息量。

| 方法 | 全范围MAE/m | 2–10m MAE | 10–40m MAE | 40–80m MAE | 全范围RMSE/m | F-score@0.1m |
| --- | --- | --- | --- | --- | --- | --- |
| V2（20261029） | 2.66457 | 1.70340 | 2.82147 | 16.81856 | 6.17108 | 0.33509 |
| V5（VAL-best） | 2.13076 | 0.92440 | 2.92106 | 15.65436 | 5.85987 | 0.43866 |
| MVSAnywhere 一次细化 | 1.93503 | 0.45525 | 2.64398 | 20.38539 | 6.06756 | 0.44878 |

**equal_pixels**：相同RGB信息，保留已训练网络的张量尺寸。

| 方法 | 全范围MAE/m | 2–10m MAE | 10–40m MAE | 40–80m MAE | 全范围RMSE/m | F-score@0.1m |
| --- | --- | --- | --- | --- | --- | --- |
| V2（20261029） | 2.72234 | 1.77937 | 2.85510 | 16.75968 | 6.24912 | 0.32879 |
| V5（VAL-best） | 2.16227 | 0.99538 | 2.88632 | 15.53206 | 5.90684 | 0.42964 |
| MVSAnywhere 一次细化 | 1.93503 | 0.45525 | 2.64398 | 20.38539 | 6.06756 | 0.44878 |

## 初始候选与后续解码

下面全部在共同stride8格点计算，不能与全分辨率最终MAE混用；coarse是粗分支输出，不是GT oracle。

| 方法 | coarse MAE/m | fine MAE/m | final stride8 MAE/m |
| --- | --- | --- | --- |
| V2（20261029） | 2.97574 | 2.99149 | 2.71908 |
| V5（VAL-best） | 2.33562 | 2.42055 | 2.15798 |
| MVSAnywhere 一次细化 | 不定义为同一中间阶段 | 不定义为同一中间阶段 | 2.01733 |

V5粗层2.97574→2.33562米、最终同网格2.71908→2.15798米，新增连接的改善确实传到了输出。但V5细层2.42055米仍差于粗层2.33562米，说明后续候选判断还有损失；这不等于细层在每个像素都无益，也不是删除细层的因果证据。V5近距改善而中距略退，不能概括为各距离全部提升。

## 七个公共TEST

采用VAL选出的public/best.pt；七域总11970组，VKITTI2和DynamicReplica为真实双目，其余为静态多位置四视图，并非全部同步四目。scene_unit没有认证为米。表内为MAE / RMSE，各域不相互平均。

**native**

| 数据集 | 单位 | 组数 | V2 | V5 | MVSAnywhere细化 |
| --- | --- | --- | --- | --- | --- |
| TartanAir_forest | meters | 848 | 2.5002 / 6.7358 | 2.4987 / 6.7421 | 2.6240 / 6.7650 |
| MatrixCity_street | meters | 160 | 7.4280 / 36.9056 | 7.4242 / 36.9506 | 10.6170 / 44.1089 |
| Hypersim_subset | meters | 194 | 0.4288 / 0.9137 | 0.4324 / 0.9211 | 0.2235 / 0.5643 |
| MVS_Synth | scene_unit | 1183 | 101.6544 / 760.6599 | 101.4617 / 760.1086 | 257.1828 / 1031.4373 |
| VKITTI2 | meters | 8370 | 5.1599 / 22.5018 | 5.1579 / 22.4551 | 5.2243 / 20.1923 |
| BlendedMVS | scene_unit | 915 | 0.7814 / 2.7432 | 0.7758 / 2.7207 | 0.7997 / 2.6491 |
| DynamicReplica_valid | scene_unit | 300 | 0.0947 / 0.1782 | 0.0934 / 0.1758 | 0.0765 / 0.1264 |

**equal_pixels**

| 数据集 | 单位 | 组数 | V2 | V5 | MVSAnywhere细化 |
| --- | --- | --- | --- | --- | --- |
| TartanAir_forest | meters | 848 | 2.5175 / 6.7534 | 2.5152 / 6.7539 | 2.6240 / 6.7650 |
| MatrixCity_street | meters | 160 | 7.4321 / 36.9049 | 7.4550 / 37.0228 | 10.6170 / 44.1089 |
| Hypersim_subset | meters | 194 | 0.4330 / 0.9151 | 0.4359 / 0.9211 | 0.2235 / 0.5643 |
| MVS_Synth | scene_unit | 1183 | 102.6411 / 774.1841 | 102.3792 / 771.9706 | 257.1828 / 1031.4373 |
| VKITTI2 | meters | 8370 | 5.1706 / 22.4534 | 5.1653 / 22.4026 | 5.2243 / 20.1923 |
| BlendedMVS | scene_unit | 915 | 0.8262 / 2.8661 | 0.8233 / 2.8679 | 0.7997 / 2.6491 |
| DynamicReplica_valid | scene_unit | 300 | 0.0953 / 0.1785 | 0.0941 / 0.1761 | 0.0765 / 0.1264 |

## 新CARLA其他参考相机

以下为各参考完整有效TEST，各行帧数不同，不能把单参考时间乘四当作实测四参考时间。

| 模式 | 参考相机 | 帧数 | V2 MAE/m | V5 MAE/m | MVS细化 MAE/m |
| --- | --- | --- | --- | --- | --- |
| native | CARLA_new_stereo_left | 184 | 4.75713 | 4.42662 | 3.04770 |
| native | CARLA_new_stereo_right | 183 | 4.56321 | 4.23072 | 3.06563 |
| native | CARLA_new_stereo_down | 183 | 5.38252 | 5.29457 | 5.91913 |
| equal_pixels | CARLA_new_stereo_left | 184 | 5.21423 | 4.87568 | 3.04770 |
| equal_pixels | CARLA_new_stereo_right | 183 | 4.97086 | 4.58524 | 3.06563 |
| equal_pixels | CARLA_new_stereo_down | 183 | 5.84167 | 5.99314 | 5.91913 |

## 实际速度与适用范围

5090单卡、FP32、TF32关闭，同样的位置编码缓存，三帧各10次预热和40次随机顺序交替计时。输入在GPU上；计入适配与全部网络调用，不含IO、CPU图像预处理/H2D和三维融合，因此不是车端完整系统速度。四参考共享入口与逐参考入口的depth/valid已逐位验证一致。

| 模式 | 方法 | 单参考/ms | 顺序四参考/ms | 可用四参考入口/ms | 四参考Hz |
| --- | --- | --- | --- | --- | --- |
| native | V2 | 31.49 | 125.29 | 106.63 | 9.38 |
| native | V5 | 32.50 | 129.70 | 110.90 | 9.02 |
| native | MVSAnywhere一次细化 | 37.00 | 148.00 | 148.00 | 6.76 |
| native | MVSAnywhere首遍 | 21.84 | 87.30 | 87.30 | 11.45 |
| equal_pixels | V2 | 31.62 | 125.77 | 106.96 | 9.35 |
| equal_pixels | V5 | 32.66 | 130.27 | 111.25 | 8.99 |
| equal_pixels | MVSAnywhere一次细化 | 37.06 | 148.28 | 148.28 | 6.74 |
| equal_pixels | MVSAnywhere首遍 | 21.87 | 87.44 | 87.44 | 11.44 |

MVSAnywhere首遍只用于速度边界参考，本报告精度主表对应一次细化，不能拿首遍耗时配细化精度。V5比一次细化快，但仍比MVS首遍慢，四参考未达到10Hz。这里是当前实现的速度比较，baseline没有本轮新加跨参考缓存；不宣称其理论最快实现。

## 权重选择与训练经历

复核已完成101608次原始预测、6组TEST，134个文件哈希一致。公共best与final参数不同；CARLA best与final参数逐张量完全相同，文件哈希不同来自checkpoint容器，不能据此当两套模型。

公共best在第一阶段第1轮（1670次更新）选中，后续2+6轮没有刷新VAL最佳；完整公共训练总15030次更新，但报告的public-best不是末轮模型。CARLA best在第二阶段第18轮、总384次更新选中。V5继承独立V2，不能把额外训练带来的全部变化归因于137856参数的CNN→ViT连接；本轮没有全预算V2继续训练对照。旧V2正式权重及默认选择保留。

## 新baseline独立训练与检查

PatchmatchNet和DiffMVS已分别在5090双卡NCCL、真实四目256×448 CARLA上通过2次优化更新、有限梯度及checkpoint保存。CPU本轮23项既有测试、4项新增验证/恢复测试和DiffMVS 6项原生图测试通过。核心网络和损失未替换。新增验证逻辑推理时只传inputs，保存并恢复训练随机状态/BN模式；best/final/last分开保存。

本轮先导训练只使用新CARLA完整403/82/174划分、上参考与其余三源。每方法384次更新，四卡×每卡2×累积4名义全局batch32，轮末短批次及DDP补齐的实际曝光写日志。PatchmatchNet官方DTU初始化、Adam 1e-4迁移，20/28轮衰减；DiffMVS随机初始化、原生AdamW/OneCycle及梯度裁剪2。两者不是与V5等预训练预算实验，也不能据此确认方法性能上限。PatchmatchNet保留原生SmoothL1，但DTU毫米与CARLA米的物理拐点不同，已登记为训练适配差异。

完整VAL逐像素MAE用于选优，无效预测按预定78米惩罚而非删除GT；此选择分数单独记录，不当作实际MAE。测试集不选择超参数或权重。测试会在训练完成后自动运行，同一174帧分别加载best与final。

- patchmatchnet：已启动，尚未完成；快照step=287，曝光=8920，rank0训练loss=10.71664（非精度指标）。目录`runs/active/v5_comparison_20261101/patchmatchnet_adapt`。
- diffmvs：已启动，尚未完成；快照step=288，曝光=8952，rank0训练loss=0.19721（非精度指标）。目录`runs/active/v5_comparison_20261101/diffmvs_adapt`。

独立启动（先确认所选GPU空闲，避免与当前tmux重复运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash official_methods/methods/patchmatchnet/train.sh \
  --config official_methods/methods/patchmatchnet/configs/train_carla_20261101.json \
  --output runs/active/你的新实验目录

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash official_methods/methods/diffmvs/train.sh \
  --config official_methods/methods/diffmvs/configs/train_carla_20261101.json \
  --output runs/active/另一个新实验目录
```

续训追加`--checkpoint 原目录/last.pt`，保持配置和卡数；推理/评测用`bash official_methods/scripts/run.sh 方法名 infer|evaluate --config 配置 --checkpoint 权重 --output 目录`。不带`--synthetic`不会回退合成数据。

## 证据与未完成事项

全部大文件位于`runs/active`到数据盘的既有软链接。新训练保存前检查数据盘至少200GiB，不删除旧权重。

- V5完整指标/分区/表面/原速度：`runs/active/rulermvs_v5_20261031/evaluation/{summary.json,depth_metrics.csv,stage_metrics.csv,surface_metrics.csv,accuracy_speed.csv}`。
- 本次补测/核验：`runs/active/v5_comparison_20261101/{v5_audit.json,four_reference_native.json,four_reference_equal_pixels.json,carla_complete_equal_pixels/full_test/summary.json}`。
- 新原生训练代码：各方法目录`train.sh`；公共训练控制在`official_methods/cli.py`和`training_validation.py`，网络仍在各自官方目录。
- 其他基线没有在此轮重新训练；MVSFormer++缺官方任务权重、SimpleRecon固定视图MLP需要独立训练。占据模型不同任务不混入本深度排名。
- 没有车端硬件测试、端到端传感/融合测速、重复多种训练种子或等总预训练数据预算试验；无统计显著性或论文原指标复现声明。

## PatchmatchNet官方权重完整公共TEST补测

仅发布DTU权重迁移，不是已经在七域充分训练的baseline。

| 数据集 | 单位 | 组数 | MAE | RMSE | 覆盖率 |
| --- | --- | --- | --- | --- | --- |
| TartanAir_forest | meters | 848 | 6.75469 | 10.59121 | 0.998447 |
| MatrixCity_street | meters | 160 | 67.50317 | 88.59241 | 0.997595 |
| Hypersim_subset | meters | 194 | 1.24897 | 1.91392 | 0.999592 |
| MVS_Synth | scene_unit | 1183 | 270.92052 | 1096.14737 | 0.995817 |
| VKITTI2 | meters | 8370 | 11.92680 | 37.77545 | 0.999647 |
| BlendedMVS | scene_unit | 915 | 28.29490 | 43.40967 | 0.995673 |
| DynamicReplica_valid | scene_unit | 300 | 0.30662 | 0.64590 | 0.999997 |
