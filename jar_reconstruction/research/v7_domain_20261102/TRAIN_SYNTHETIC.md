# 同预算再生成数据对照

`train_synthetic.py`是第二轮训练的独立包装，进程内显式替换`train.MixedDataset`，复用原`train_round2`模型工厂、保持损失、优化器、验证和checkpoint检查。原`train.py`、`model.py`、`model_round2.py`与运行中的对照代码不修改。

数据加载先执行原MixedDataset的参考相机轮换/标定同步重排以及原增益/缺源增强，再对TRAIN且epoch≥0的样本用独立seed实施25%双线性短基线再生成。随后原训练入口继续执行相同灰度和真实源子集增强。VAL、负epoch的审计/推理样本完全绕过再生成。新增随机流由seed、epoch、样本index和域index确定，不依赖worker调度，也不消耗模型随机流。

配置仅由文件控制，位于`configs_synthetic/`。相对于`configs_round2/coordinate_*`，变化为输出目录、指定GPU和`synthetic_options`，模型、初始权重、预算、seed、可训练参数、原外观增强保持一致：

| 条件 | 可训练部分 | 保持约束 | 公共预算 | CARLA预算 |
|---|---|---|---|---|
| coordinate_new_only | 新context_prior | 无 | 七域各128 TRAIN×2轮，896更新 | 256 TRAIN×2轮，256更新 |
| coordinate_joint_keep | decoder | 原有单边保持约束 | 同上 | 同上 |

公共与CARLA各从对应V6权重独立初始化；脚本按public→carla先后运行，**不是把public短训权重传给CARLA**。每组只保留固定末步，无TEST、无按VAL挑best。

```bash
# 单阶段独立启动
official_methods/.venv/bin/python research/v7_domain_20261102/train_synthetic.py \
  --config research/v7_domain_20261102/configs_synthetic/coordinate_new_only_public.json

# 已实际启动的两个tmux队列
bash research/v7_domain_20261102/run_synthetic_group.sh new_only
bash research/v7_domain_20261102/run_synthetic_group.sh joint_keep
```

实际产物通过`runs/active/`软链接进入数据盘，目录`v7_domain_20261102/pilots_synthetic/coordinate_<条件>/`。每个worker逐曝光记录合成抽签、实际active、源覆盖、基线与相机数；完成后`synthetic_audit.json`核对记录与预登记sampler全部相符。数据没有长缓存，保留必要日志、图片和权重。

CPU包装测试3项通过；两种条件各已执行GPU0两步训练，均有有限梯度、非零新头梯度、冻结参数不变、checkpoint逐位重载及推理targets隔离。每次smoke的4个曝光为2个合成、2个原样本，真实执行了合成路径。完整pilot于上述检查通过后在GPU5/6的tmux启动；完成状态须查看`queue_completed.json`和`queue.exit_code.txt`，启动不等于已产生性能改进。

## 本轮实际完成结果

两组公共→CARLA队列均已退出0，每个公共阶段896更新/1792曝光，每个CARLA阶段256更新/512曝光。四阶段checkpoint、推理targets隔离、冻结参数、独立RNG、sampler实际曝光审计均通过；与对应无合成对照的初始指标最大漂移0。公共实际439次合成、1353次原样本，CARLA117次合成、395次原样本；合成源的平均有效覆盖分别72.71%、67.15%。这些是输入孔洞保留后的真实覆盖，不作填洞提高统计。

固定每域/参考16VAL中，联合训练＋保持约束下，再生成相同预算无再生成的CARLA上/左/右/下全图MAE变化为−3.615%/−0.282%/−2.830%/−0.652%；实际值为1.024883/1.116527/1.510300/1.291301米。公共7域有6域全图MAE较好，但BlendedMVS变差3.749%；DynamicReplica仍比冻结V6差，因此**没有满足全域保持条件**。

CARLA上参考粗→细阶段相同预算变化为−4.414%/−3.796%，其全图改善主要在平滑区和近距；边界带只改善0.741%。左参考粗层反退1.071%，不能笼统称所有初始候选改善。左/右/下近距比同预算无再生成分别变差2.917%/1.907%/4.921%，左远距亦变差0.461%；需要看完整指标，不能只挑全图MAE。

仅训练新先验的合成组在公共7域仅3域胜同预算对照，不能认为换数据＋冻结旧解码即可取得通用升级。这些是短预算适配，完整训练可能改变结论，但当前证据不足以直接替换V6。下一步只将联合版本作为完整VAL待确认候选，不读取ETH/TEST挑选训练参数。

汇总入口：`analyze_synthetic.py`；结果位于`runs/active/v7_domain_20261102/synthetic_analysis/`，含2112行分阶段/分区/多指标CSV、中文对照表和协议一致性审计。全部大产物在数据盘，旧V6和默认模型保持。

解释边界：这项干预同时改变基线、源数量和有效覆盖，是一套再生成策略的对照，不是单独基线变化的因果隔离。严格有效性使森林图缺失较多，须结合真实VAL判断是否有益。模型推理依然只有RGB与标定，绝不从GT生成测试输入。
