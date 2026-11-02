# 五种官方方法：2026-10-11 CARLA 实测

本轮完成 Fast-MVSNet、SimpleRecon Hero 四目版、MVSAnywhere Hero 的真实训练与独立轨迹测试。OctreeOcc 和 GaussianFormer-2 仍有明确阻塞，未填入假指标。**这不是原论文数据集精度复现，也不是同预训练、同训练算力的架构排名。**

所有 checkpoint、公开权重缓存及逐帧结果均写入 `runs/carla_research_20261011`，其实际位置为 `/data/liuzonghao_data/3Dscene_runs/carla_research_20261011`。机器可读汇总：[official_baseline_summary.json](../../runs/carla_research_20261011/baselines/official_baseline_summary.json)。

## 数据与评价口径

固定同步四目 stereo_left / stereo_right / stereo_up / stereo_down，不使用 tele_left / tele_right，不增加历史帧。训练 66 个完整帧、开发 25 个完整帧；同训练轨迹中间隔离 40 个 frame ID。测试使用另一轨迹的 200 个完整帧。开发集不是独立地图。只有 stereo_up（槽 2）有深度真值，因此训练参考固定为槽 2，评估虽轮换四个参考并计全部成本，但精度只对槽 2 的原始标签负责。

输入 256×448，RGB 缩放同时更新内参；标签用 nearest-exact。统一评价光轴深度 2–80 米，rig 范围 x=[0,80]、y=[-45,45]、z=[-2,20] 米。GT 与预测分离；测试范围来自配置。Fast/SimpleRecon 使用配置匹配范围；MVSAnywhere 保留原生标定启发式范围和预测驱动细化。

MAE / RMSE / AbsRel 在有效预测与固定 GT 的交集上计算，并同时保留覆盖率；“0.2米成功率”以全部固定 GT 像素为分母，拒绝预测不能获得虚高分。δ1 是预测/真值及其倒数较大者小于 1.25 的比例，同样用全部 GT 作分母。原生低分辨率深度仅按其内参映射回共同光栅，无几何过滤。

## 冻结配置后的 test200

| 方法 | 初始化与CARLA预算 | MAE(m)↓ | RMSE(m)↓ | AbsRel↓ | 覆盖率 | 全GT 0.2米成功率↑ | δ1↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Fast-MVSNet | 官方 DTU + CARLA 1000 步 | 3.3098 | 8.4627 | 0.3852 | 99.982% | 17.69% | 77.80% |
| SimpleRecon Hero 四目 | ImageNet 双骨干 + CARLA 3000 步 | 3.9022 | 7.6266 | 0.2567 | 100.000% | 25.15% | 57.75% |
| MVSAnywhere Hero | 官方完整权重，CARLA 0 步 | 1.4806 | 4.5066 | 0.0955 | 100.000% | 34.06% | 94.24% |
| MVSAnywhere Hero | 官方完整权重 + CARLA 1000 步 | 1.2109 | 3.7714 | 0.0586 | 100.000% | 49.66% | 95.32% |

每个表内条件只运行一次完整 test200；没有使用 test 结果回调参数。官方未微调 MVSAnywhere 与微调条件分列，后者不能称零样本。以上差异没有多种子置信区间。

## 开发集上的判断与保留项

- Fast-MVSNet：DTU 权重直接迁移的 dev MAE 为 31.8462m；原生微调 1000 步后，96 候选为 1.4817m。只增加测试候选到 128/192，MAE 分别变成 1.5040/1.5689m，而 0.2米成功率从 14.94% 上升到 20.96%/27.06%。这是取舍，未证明全面改善，因此默认保留 96。
- SimpleRecon：八目 Hero 的元数据 MLP 与四目输入宽度不兼容，未裁剪或补零权重。加载官方 ImageNet EfficientNetV2-S 和抗混叠 ResNet18，四目元数据/成本编码/解码器从头训练。1000 步 dev MAE 5.0578m；确认原生输出没有深度截断、标定及范围正确后，只延长原优化器到 3000 步，MAE 降到 3.7148m、0.2米成功率从 4.35% 到 36.62%，故保留 3000。远距 24–80m 仍为 12.5275m；这提示短训和数据覆盖不足，不能据此下结论说原论文方法本身不行。
- MVSAnywhere：完整官方 Hero 严格加载，dev MAE 1.1010m；按原生学习率组和损失微调 1000 步后为 0.8748m，0.2米成功率 28.65%→47.64%。只在 dev 冻结微调条件后运行 test。

三种方法均是单卡、FP32、batch1、累计1、seed17。Fast 保留 RMSprop、4个epoch的传播预热及后续GN；SimpleRecon 保留 AdamW；MVSAnywhere 保留匹配/编码/解码三个学习率组。没有换预测头、改网络层、用普通 MVSNet 代替 Fast-MVSNet，也没有移除 Hero 元数据模块。SimpleRecon 缺源图深度，因此原生多视图损失的空监督项为0；MVSAnywhere 无天空标签，天空监督为0。

## 同卡串行的速度和显存

为避免把并行任务期间的时间波动说成算法加速，另用 GPU4、同一真实训练帧，逐模型串行预热5次、统计20次。没有读取标签或重新计算精度。网络列包含四参考；MVSAnywhere 每参考两轮完整网络，共8轮，其他模型共4次参考调用。pipeline 再包含槽2的共同深度光栅转换，不含读盘与GT评价。显存是 PyTorch 峰值 allocated（包括模型/输入），不是 nvidia-smi 的reserved/驱动占用。

| 条件 | 原生网络(ms) | 含适配forward(ms) | pipeline(ms) | 峰值MiB |
|---|---:|---:|---:|---:|
| Fast-MVSNet / 官方 DTU + CARLA 1000 步 | 39.53 | 40.93 | 41.49 | 350.98 |
| SimpleRecon Hero 四目 / ImageNet 双骨干 + CARLA 3000 步 | 194.18 | 200.24 | 200.83 | 326.59 |
| MVSAnywhere Hero / 官方完整权重，CARLA 0 步 | 434.59 | 440.93 | 441.52 | 956.39 |
| MVSAnywhere Hero / 官方完整权重 + CARLA 1000 步 | 435.43 | 442.00 | 442.62 | 956.39 |

MVSAnywhere 两份权重的同条件耗时基本一致；完整test中原权重744ms与微调432ms的差异不能解释为微调加速。完整test原始逐帧时间仍保留。单帧串行计时也不代表所有分辨率和全部场景。

## 五种方法的未完成项

- OctreeOcc：固定官方提交存在配置重复关键字、缺失 temporal_self_attention 文件、配置类名未注册；RGB→离线八叉树语义初始化先验的训练/权重/生成流程没有完整发布，且论文和配置时序机制不同。本轮没有用近似网络填补，没有完整训练或测试指标。
- GaussianFormer-2：当前数据无语义占据标签，也无完整初始化器/FCOS3D或官方整网权重。本地历史 Prob-256 报告还记录了重复稀疏索引导致的同输入不一致；本轮未复测其整网。单个视角深度只支持已观测free/hit/unknown，不能伪造17类语义或把遮挡后方当free。没有生成语义IoU。
- 许可证：Fast-MVSNet、OctreeOcc 为 MIT；两种 Niantic 模型含非商业限制；GaussianFormer固定版本的顶层LICENSE为空，不能替它宣称MIT许可。具体commit、文件级问题见 [状态记录](carla_status_20261011.json)。

三个深度官方网络均要求完整四目，本轮不伪装缺视角输入。缺相机实验应与自研支持缺相机的模型单独分表。数据无小障碍物实例或完整3D真值，深度边界分组不等于小障碍物召回，未声称完整3D精度或原论文复现。

## 已执行验证与命令

实际通过：数据/共享接口29项；Niantic合同3项（含2个方法子测试）；三种网络真实GPU前后向与优化；全部训练每步检查损失/梯度有限；共享checkpoint严格重新加载；Niantic真实帧原生与适配器逐位一致、改变标签不改变eval；固定官方源码审计。本轮没有重新运行多卡DDP或两种占据整网，不能把已有入口当作本轮通过。

工作目录为项目 `jar_reconstruction`。下列是本轮实际使用的入口；已有结果目录需保留，重新实验应另取输出目录。所有参数改 JSON，命令行仅指定配置、设备、输出和必要的续训权重。

```bash
official_methods/.venv/bin/python -m pytest official_methods/tests/test_manifest.py official_methods/tests/test_shared.py -q
official_methods/.venv/bin/python -m pytest official_methods/tests/test_niantic_contract.py -q
python official_methods/scripts/fetch_upstream.py --audit --report /tmp/official_source_audit_oct11.json

# 每个 launch 脚本记录已实际执行的 GPU、数据盘缓存和完整训练命令
# Fast: runs/carla_research_20261011/baselines/fast_mvsnet/launch_train.sh
# SR:   runs/carla_research_20261011/baselines/simplerecon/launch_train.sh
# SR续训: runs/carla_research_20261011/baselines/simplerecon/launch_resume3000.sh
# MVS:  runs/carla_research_20261011/baselines/mvsanywhere/launch_train.sh

CUDA_VISIBLE_DEVICES=5 official_methods/.venv/bin/python tools/carla_compare.py --config configs/carla_comparison/simplerecon_test.json --device cuda:0 --output runs/carla_research_20261011/baselines/simplerecon/test_step3000
CUDA_VISIBLE_DEVICES=6 official_methods/.venv/bin/python tools/carla_compare.py --config configs/carla_comparison/mvsanywhere_test.json --device cuda:0 --output runs/carla_research_20261011/baselines/mvsanywhere/test_step1000
CUDA_VISIBLE_DEVICES=4 official_methods/.venv/bin/python official_methods/tools/benchmark_carla_latency.py --config configs/carla_comparison/official_latency.json --device cuda:0 --output runs/carla_research_20261011/baselines/latency_serial_gpu4.json
```

官方预训练来源和SHA256、已记录的manifest及checkpoint哈希（早期dev未保存的字段如实留空）、完整分距离指标与原始耗时，都在机器可读汇总及对应 comparison.json 中。MVS完整权重另核对官方MD5；SimpleRecon使用正确的 antialiased ResNet18 finetune 权重，没有把未用的普通版本混作初始化。
