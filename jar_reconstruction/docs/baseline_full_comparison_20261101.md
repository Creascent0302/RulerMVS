# 新baseline训练完成与完整对比（2026-11-01）

两种新baseline均完成384次更新、11940次实际样本曝光和自动best/final TEST。原训练结束时间均约为北京时间11月1日09:50。本次不再训练、不依据TEST改模型，而是补齐统一输入、四参考、随机重复及速度/显存对比。

## 上参考共同主表

174帧相同完整四目，256×448共同RGB信息，2–80米，同一GT掩码、FP32/TF32关闭；V2/V5匹配张量仍为512×896，但只能上采样共同主图。三个新随机种子只是推理重复，不能替代多次独立训练。

| 方法 | MAE/m | RMSE/m | AbsRel | 覆盖率 | F-score@0.1m | 网络+适配/ms | 含格点转换/ms | 独占峰值/GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 2.72795 | 6.24501 | 0.27576 | 100.00000% | 0.32892 | 31.40 | 31.46 | 1.043 |
| V5 | 2.15650 | 5.88845 | 0.17216 | 100.00000% | 0.43120 | 32.38 | 32.44 | 1.043 |
| MVSAnywhere一次细化 | 1.93100 | 6.05562 | 0.11282 | 100.00000% | 0.45044 | 36.81 | 36.86 | 1.769 |
| PatchmatchNet CARLA best | 2.91824 | 7.52665 | 0.33593 | 99.99414% | 0.31470 | 15.17 | 15.73 | 0.109 |
| DiffMVS CARLA best | 3.50577 | 8.73938 | 0.22711 | 100.00000% | 0.29758 | 25.03 | 25.60 | 0.096 |

误差是有效预测条件指标，覆盖率不足时不能称全GT MAE；缺失预测仍计入覆盖率/阈值失败，不静默删帧。表面F-score仅评价参考视图可见表面，不是全场景占据。

速度在单卡5090无其他GPU任务时测量，固定VAL首中末3帧、10次预热/40次交替计时；单模型显存另行独立驻留测量。GPU常驻输入，不含IO/H2D/点云融合；不能当作车端完整系统延迟。

## 实际四参考速度

| 方法 | 逐参考四次/ms | 当前可用四参考入口/ms | 四参考Hz |
| --- | --- | --- | --- |
| V2 | 125.44 | 106.62 | 9.38 |
| V5 | 129.82 | 110.75 | 9.03 |
| MVSAnywhere一次细化 | 147.79 | 147.79 | 6.77 |
| PatchmatchNet CARLA best | 59.87 | 59.87 | 16.70 |
| DiffMVS CARLA best | 99.83 | 99.83 | 10.02 |

V2/V5共享入口与逐参考depth/valid逐位一致；其余使用原生逐参考调用。四参考时间为实际执行而非单参考乘四，此列未包含格点转换。没有给baseline添加新的共享编码优化，不把当前实现速度当其理论最快速度。

## 每个相机与距离段

| 参考 | 方法 | 帧数 | 全范围MAE | 2–10m | 10–40m | 40–80m |
| --- | --- | --- | --- | --- | --- | --- |
| CARLA_new_upper | V5 | 174 | 2.15650 | 0.99262 | 2.87997 | 15.42411 |
| CARLA_new_upper | V2 | 174 | 2.72795 | 1.78964 | 2.85717 | 16.65056 |
| CARLA_new_upper | MVSAnywhere一次细化 | 174 | 1.93100 | 0.45431 | 2.63662 | 20.27294 |
| CARLA_new_upper | PatchmatchNet CARLA best | 174 | 2.91824 | 1.87885 | 3.04172 | 18.47945 |
| CARLA_new_upper | DiffMVS CARLA best | 174 | 3.50577 | 0.89538 | 5.82165 | 28.33646 |
| CARLA_new_stereo_left | V5 | 174 | 4.88658 | 3.35352 | 6.51156 | 15.19950 |
| CARLA_new_stereo_left | V2 | 174 | 5.23160 | 4.41375 | 4.93680 | 17.37800 |
| CARLA_new_stereo_left | MVSAnywhere一次细化 | 174 | 3.02206 | 0.99502 | 3.33377 | 27.16490 |
| CARLA_new_stereo_left | PatchmatchNet CARLA best | 174 | 7.79630 | 7.23069 | 5.58794 | 27.66178 |
| CARLA_new_stereo_left | DiffMVS CARLA best | 174 | 6.68004 | 3.15654 | 9.52981 | 35.44504 |
| CARLA_new_stereo_right | V5 | 174 | 4.55875 | 3.37549 | 5.49715 | 14.55811 |
| CARLA_new_stereo_right | V2 | 174 | 4.95554 | 4.36904 | 4.26457 | 17.13437 |
| CARLA_new_stereo_right | MVSAnywhere一次细化 | 174 | 3.03387 | 1.02847 | 3.24014 | 28.62806 |
| CARLA_new_stereo_right | PatchmatchNet CARLA best | 174 | 8.04520 | 8.27382 | 5.00438 | 23.97719 |
| CARLA_new_stereo_right | DiffMVS CARLA best | 174 | 5.83448 | 3.03250 | 7.53565 | 32.76802 |
| CARLA_new_stereo_down | V5 | 174 | 6.01523 | 6.10921 | 4.04160 | 14.68129 |
| CARLA_new_stereo_down | V2 | 174 | 5.85911 | 5.90009 | 3.38734 | 17.12192 |
| CARLA_new_stereo_down | MVSAnywhere一次细化 | 174 | 5.91791 | 3.10157 | 3.93198 | 30.17748 |
| CARLA_new_stereo_down | PatchmatchNet CARLA best | 174 | 6.51690 | 5.43521 | 4.09781 | 23.52842 |
| CARLA_new_stereo_down | DiffMVS CARLA best | 174 | 9.34589 | 4.91962 | 9.06610 | 34.27481 |

两新增baseline此次仅以上相机作为训练参考；左/右/下属于额外的跨参考迁移。V2/V5有参考轮换训练，因此这些副表不能作为相同训练任务下的架构能力结论。

## 随机重复与best/final

| 方法 | seed1701 MAE | seed2701 | seed3701 | 均值±样本标准差/m |
| --- | --- | --- | --- | --- |
| patchmatchnet | 2.91824 | 2.91862 | 2.91137 | 2.91608 ± 0.00408 |
| diffmvs | 3.50577 | 3.49788 | 3.50698 | 3.50354 ± 0.00494 |

保留官方随机候选/扩散噪声；每帧用RGB+K+T+参考号+seed确定随机流，既不读GT，也不依赖模型加载顺序或分片GPU。PatchmatchNet best/final参数相同，同随机流下MAE/RMSE在float32容差内一致，不宣称GPU数值末位逐位相同；先前约0.0015米差别是随机性，不是权重更优。

| 保存权重 | 完整VAL MAE/m | 缺失惩罚VAL分数 | TEST MAE/m | TEST RMSE/m |
| --- | --- | --- | --- | --- |
| PatchmatchNet CARLA best | 1.28082 | 1.28163 | 2.91824 | 7.52665 |
| PatchmatchNet final | 1.28082 | 1.28163 | 2.91824 | 7.52665 |
| DiffMVS CARLA best | 3.64572 | 3.64572 | 3.50577 | 8.73938 |
| DiffMVS final | 4.04709 | 4.04709 | 3.56670 | 9.07063 |

复核发现旧四卡验证混用了各rank的BN运行统计，而实际checkpoint保存rank0统计。因此历史best只能称旧规则选出的保存点，不保证是所有历史训练迭代在部署模式下的最佳点。已修复后续验证：临时广播rank0缓冲量，验证后恢复各rank训练状态；现表逐个实际checkpoint独立验证，best/final全部保留，不用TEST选择。该问题影响验证选模，不会把GT泄漏进推理，也不使此前固定checkpoint的TEST输出失效。

固定原来的采样/RNG、真实四卡单独隔离BN影响：

| 方法 | 旧混合BN VAL分数 | 实际rank0模型 VAL分数 |
| --- | --- | --- |
| patchmatchnet | 1.30090 | 1.28729 |
| diffmvs | 3.70152 | 3.64989 |

## 公平性与训练成熟度

PatchmatchNet从官方DTU权重迁移；DiffMVS无官方任务权重，从头只训练当前403帧CARLA。V2/V5还经过七域公共预训练，MVSAnywhere使用官方多域预训练。当前是相同测试输入与指标下的模型状态比较，不是等数据、等预训练预算的架构上限对比。尤其不能把DiffMVS先导训练较差当作论文方法无效。完整训练预算和原生单位/损失差异沿用194条记录；本次没有改任何网络结构。

## 公共跨域测试

| 数据集 | 单位 | CARLA适配权重 | 组数 | MAE | RMSE | 覆盖率 |
| --- | --- | --- | --- | --- | --- | --- |
| TartanAir_forest | meters | PatchmatchNet CARLA best | 848 | 6.89721 | 10.53018 | 0.998269 |
| TartanAir_forest | meters | DiffMVS CARLA best | 848 | 13.49878 | 43.15741 | 1.000000 |
| MatrixCity_street | meters | PatchmatchNet CARLA best | 160 | 50.87096 | 76.00191 | 0.997599 |
| MatrixCity_street | meters | DiffMVS CARLA best | 160 | 53.27984 | 153.15394 | 1.000000 |
| Hypersim_subset | meters | PatchmatchNet CARLA best | 194 | 1.24698 | 1.82299 | 0.999399 |
| Hypersim_subset | meters | DiffMVS CARLA best | 194 | 4.12526 | 6.13668 | 1.000000 |
| MVS_Synth | scene_unit | PatchmatchNet CARLA best | 1183 | 302.45315 | 1098.11990 | 0.997263 |
| MVS_Synth | scene_unit | DiffMVS CARLA best | 1183 | 1633.46606 | 6043.38532 | 1.000000 |
| VKITTI2 | meters | PatchmatchNet CARLA best | 8370 | 14.28350 | 39.51008 | 0.999579 |
| VKITTI2 | meters | DiffMVS CARLA best | 8370 | 45.81945 | 103.44094 | 1.000000 |
| BlendedMVS | scene_unit | PatchmatchNet CARLA best | 915 | 27.84031 | 42.40352 | 0.996603 |
| BlendedMVS | scene_unit | DiffMVS CARLA best | 915 | 43.24887 | 83.66838 | 1.000000 |
| DynamicReplica_valid | scene_unit | PatchmatchNet CARLA best | 300 | 0.32313 | 0.71224 | 0.999996 |
| DynamicReplica_valid | scene_unit | DiffMVS CARLA best | 300 | 1.13839 | 1.79695 | 1.000000 |

七域11970组全部完成，使用真实N=2/4，不复制视图；scene_unit不写成米。这里测CARLA适配后的域外迁移，不能冒充这些baseline已完成七域共同预训练。

## 文件与复现

配置/入口：`research/baseline_full_comparison_20261101/`；队列为`baseline_full_compare_20261101`。完整结果、逐帧分片、权重/源码/清单hash、时间分布、深度CSV与验证审计均在`runs/active/baseline_full_comparison_20261101/`数据盘软链接下。训练权重保留在`runs/active/v5_comparison_20261101/{patchmatchnet_adapt,diffmvs_adapt}/training/`。

新增回归包含随机流不受加载顺序影响、不同seed保留随机变化、双进程Gloo验证使用同一BN并恢复本地状态；连同既有训练保存/恢复测试共7项通过。初次Gloo受沙箱回环限制，已在允许本机通信环境实测；另修正随机初始化网络必定100%覆盖的错误测试前提，未因此改变网络。
