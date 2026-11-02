# V6论文证据脚本索引

本目录保存独立诊断与图表入口，不改已冻结的主模型或正式训练配置。中文实验设计见`../../docs/paper_experiment_plan_20261101.md`；持续经验只记录到`../../docs/research_memory.md`，这里不另建平行研发历史。

以下状态来自已存在结果的完成标志核对。大产物统一位于现有数据盘软链接：

```text
runs/active/rulermvs_v6_20261101/full_training/
```

## 1. 已运行内容与边界

| 脚本、配置 | 实际内容 | 产物子目录 | 当前状态 |
|---|---|---|---|
| `mechanism.py`、`mechanism.json` | 固定11评测域×16VAL；V5/V6以及同V6权重的无父信息/逐点读出 | `paper_mechanism/results.json` | 完成，176组；推理依赖诊断，未重训 |
| `robustness.py`、`robustness.json` | 固定16组CARLA上参考VAL；11条件×4方法 | `paper_robustness/` | 完成，704次预测；全部分母一致，清洁输入复核通过 |
| `range_probe.py`、`range_probe.json` | 176VAL原始输出与共同TRAIN范围裁剪；记录MVS逐遍实际范围 | `paper_range_probe/` | 完成；运行版本有单独审计说明 |
| `range_report.py` | 从范围JSON生成CSV、中文结论、范围图 | `paper_range_probe/analysis/` | CPU已运行；不再推理 |
| `firstpass.py`、`firstpass.json`、`firstpass_carla.sh` | 与主比较保持同协议的MVSAnywhere首遍 | `paper_firstpass/public/full_test/`、`carla_new/full_test/` | 两组完整TEST均完成，7公共域11970组与4参考696组 |
| `firstpass_speed.py` | 同卡首遍/细化/V6、单参考/四参考计时 | `paper_firstpass/speed/` | 完成，每模式120次；含共同输出处理的V6 fast四参考92.314ms，MVS首遍87.520ms |
| `qualitative.py`、`qualitative.json` | 固定CARLA TEST索引87的同帧四参考，可视化V5/V6/MVS细化 | `paper_qualitative/` | 完成4参考；重推指标与对应完整TEST一致 |
| `split_audit.py` | 实际TRAIN、VAL、TEST全部参考及源图路径/样本ID交叉审计 | `paper_split_audit/` | CPU完成；零路径交叉不等于无近重复或无预训练暴露 |
| `prepare_eth3d.py`、`check_eth3d_photometric.py`、`eth3d_report.py` | 官方真实双目27对，8方法、all/nocc掩码 | `paper_eth3d/` | 完成432预测；预登记外部测试，V6不优于强baseline |
| `physics.py`、`physics.json`、`physics_audit.py`、`physics_report.py` | 单位、物理尺度、坐标、标定压力与原生置信度 | `paper_physics/` | 完成6336前向；528清洁条件与完整VAL核对一致 |
| `fast_test.py`、`fast_test.json`、`fast_test.sh` | 快速执行完整12666参考TEST精度 | `paper_fast_test/` | 完成12666参考TEST及逐帧/编译日志核对，不借普通执行精度 |

同预算五组交接×三seed及标尺消融独立封装在`../paper_ablation_20261102/`，不在本目录偷偷改模型。38组训练及全部完整VAL均已完成，158992次模型指标预测，详见其队列日志和最终汇总。

这些结果不构成多个独立训练种子。四个CARLA参考来自同一物理采集，公共tuple包含双目/静态多视角/相邻时间选帧，不应称全部为同步四目。`firstpass_README.md`是此前准备阶段说明，运行状态以本索引及输出JSON为准。

## 2. 三个直接指导论文的结论

**候选保持与纠错存在冲突。** 同V6权重下，引入父分布相对只做族读出，11/11评测域降低粗正确点的破坏率，却也11/11降低粗错误点的纠正率；最终MAE仅7/11域更好。完整族交接相对同权重逐点读出11/11改善。脚本断言各V6分支的粗输出和连续细候选完全相同，排除了这些位置不同造成的混淆，但训练仍针对完整结构，不能替代同初始化、同预算重训消融。这里的“域”是评测分组，不是独立场景。

**V6仍存在坏源短板。** 固定16VAL中V6全11条件MAE优于V5，但全11条件均差于MVSAnywhere首遍和细化。双目子集平均相对清洁四目，V6退化68.27%、MVS细化11.27%；重噪声0.15分别退化15.15%/0.86%。保留轻噪声仍有收益，强坏源有时比直接删源更差，不能简单一律保留或一律丢弃。尚未通过attention统计或重训证明原因。

**远距差异不是此次预测越过TRAIN上界所致。** 176VAL共同裁剪对任何远距指标均无影响；MVS仅CARLA下参考67个有效像素低于TRAIN下界，裁剪full MAE仅改变约4.54e−7m。但Tartan最后细化范围出现过度收窄风险：15/16帧上界低于40m，其中14帧存在远距GT，其远距像素占子集远距GT的89.44%。这不等于已经证明范围收缩是全部误差原因，需用首遍/细化配对继续辨析。

详细数值、失败解释和同预算重训矩阵见实验计划；范围可视化不改正式主表。没有新增架构在这些脚本里被偷偷训练，也未将oracle当预测。

## 3. 运行入口

以下是与各脚本对应的复运行入口，**不是要求把已完成实验重新跑一遍**。GPU诊断通常拒绝覆盖非空输出；需要复现时复制JSON并设置新的数据盘输出目录，再运行。配置中的GPU索引需按实际空闲设备选择。统一从项目根目录执行：

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction

# 已完成的交接诊断，读取固定VAL索引。
official_methods/.venv/bin/python research/paper_evidence_20261101/mechanism.py \
  --config research/paper_evidence_20261101/mechanism.json

# 已完成的坏源/缺源测试，11条件而非11组独立场景。
official_methods/.venv/bin/python research/paper_evidence_20261101/robustness.py \
  --config research/paper_evidence_20261101/robustness.json

# 范围脚本的CPU检查与配置准备；不启动GPU推理。
official_methods/.venv/bin/python research/paper_evidence_20261101/range_probe.py --self-test
official_methods/.venv/bin/python research/paper_evidence_20261101/range_probe.py \
  --config research/paper_evidence_20261101/range_probe.json --prepare

# 已完成的范围推理入口。
official_methods/.venv/bin/python research/paper_evidence_20261101/range_probe.py \
  --config research/paper_evidence_20261101/range_probe.json

# 仅CPU生成范围CSV、分析和图。
official_methods/.venv/bin/python research/paper_evidence_20261101/range_report.py \
  --results runs/active/rulermvs_v6_20261101/full_training/paper_range_probe/results.json

# 固定定性样例；不是挑选本方法获胜区域。
official_methods/.venv/bin/python research/paper_evidence_20261101/qualitative.py \
  --config research/paper_evidence_20261101/qualitative.json

# CPU清单路径交叉审计。
official_methods/.venv/bin/python research/paper_evidence_20261101/split_audit.py
```

首遍补测先由`firstpass.py --prepare`冻结配置/hash；公共与CARLA两组实际评测分别使用生成配置执行`research/rulermvs_v6_20261101/evaluation.py`。独立CARLA脚本额外提供进程锁与退出码记录：

```bash
official_methods/.venv/bin/python research/paper_evidence_20261101/firstpass.py \
  --config research/paper_evidence_20261101/firstpass.json --prepare

official_methods/.venv/bin/python research/rulermvs_v6_20261101/evaluation.py \
  --config runs/active/rulermvs_v6_20261101/full_training/paper_firstpass/configs/public_test.json

bash research/paper_evidence_20261101/firstpass_carla.sh

# 独占计时入口；核对完成状态后再决定是否需要运行。
official_methods/.venv/bin/python research/paper_evidence_20261101/firstpass_speed.py \
  --config research/paper_evidence_20261101/firstpass.json
```

模型已在本地，不需要为了这些诊断联网下载。实际调度使用离线模式与服务器网络包装时，以对应`console.log`、`public_test.log`、`carla_new_test.log`为准；上述命令仅给出可复现的Python入口，不声称所有进程的外层shell原文均已保存。

## 4. 范围脚本运行版本审计

`range_probe.py`正在GPU运行时，磁盘源码增加了两个协议hash元数据字段并删除了未使用的`import copy`；正在运行的Python仍执行旧版本。结果没有新增两个字段，但结束时读取的源码hash来自较新的磁盘文件。

因此保留原`results.json`不改，另外保存：

- `source_executed.py`：根据这唯一补丁还原的实际执行版本快照；不是当时预先留存的启动快照，不可冒充其来源。SHA256为`6efb0d2afe0f54811e148deea5b2381994609ff2111f0893f511c4696d2ff17a`。
- `audit_note.json`：记录两版hash、补丁范围、重建依据以及当前权重hash仍一致。当前源码SHA256为`e74707c3511a1e25ef5b85a01f5fd160bb0721b5ca6d1ffe4c49dd25f8d93a24`。
- `source_executed.py`位于输出目录，仅作审计；不要直接从该目录执行，因为相对项目根目录定位不同。复现实验用本目录的冻结脚本，新旧差异仅为元数据及未使用import，不改变前向、权重、掩码或指标。

本次更正只说明可确认的版本差异，不能把事后hash复核说成全运行链启动时快照。

## 5. 必须保留的限制

1. 既有TEST已经多次参与研发分析。新方法选择只在TRAIN/VAL，最终独立验证需新增封存物理场景；仅把旧TEST改名并不能消除观察历史。
2. `mechanism.py`的推理开关不等于重新训练；三种子短预算适配消融与完整VAL已另行完成，仍不能称从零充分训练。训练波动与测试场景置信区间分别报告。
3. `range_probe.py`共同裁剪是诊断后处理，不属于MVSAnywhere官方原生网络，也不是修改V6架构；GT没有参与推理范围设置。
4. `robustness.py`只有16个同轨迹VAL参考帧，没有证明跨地图或真实传感器噪声鲁棒性。标定扰动由独立`physics.py`完成，不能与图像噪声混为一类。
5. 定性图统一色条饱和仅用于显示，指标不截断。灰色缺GT不是零误差或空闲空间。
6. `split_audit.py`仅比较规范化文件路径和ID，未完成图像内容近重复、地图物理重叠或通用/baseline预训练数据交叉审计。
7. `MVS_Synth`、`BlendedMVS`、`DynamicReplica_valid`沿用当前`scene_unit`，不得跨域平均原始MAE再标为米。
8. 网络计时若未包含IO、CPU解析、H2D和点云融合，应明确标注；四参考真实执行时间不能用单参考乘四代替，也不能直接等同车端系统频率。

## 6. 新增外域与物理实验的复运行

ETH3D已下载、处理并完成推理，不需要再次下载。图表可直接从原始统计重生成：

```bash
official_methods/.venv/bin/python research/paper_evidence_20261101/eth3d_report.py
official_methods/.venv/bin/python research/paper_evidence_20261101/physics_audit.py
official_methods/.venv/bin/python research/paper_evidence_20261101/physics_report.py \
  --result runs/active/rulermvs_v6_20261101/full_training/paper_physics/results.json
```

需要复现GPU实验时，复制配置并设置新的数据盘输出；`physics.py --config <新配置>`拒绝覆盖已有结果。ETH3D推理使用`research/rulermvs_v6_20261101/evaluation.py --config <复制后的paper_eth3d/evaluation.json>`。几何转换核对与首次失败均保留，不能以重生成报告覆盖历史失败记录。
