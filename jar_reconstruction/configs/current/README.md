# 当前版本、权重和推理入口

参数直接在 JSON 中修改。版本与 SHA256 见 [rulermvs_versions.json](rulermvs_versions.json)。V2/V3 结构、原发布备份、所有正式训练结果均保留；当前整理不改变默认模型，也不改变网络参数。

| 配置 | 权重与用途 |
|---|---|
| `infer_public.json` | 历史推荐 V2，公共数据 |
| `infer_carla.json` | 历史推荐 V2，旧 CARLA 微调权重；数据入口已指向新 CARLA，不等于权重在新数据训练 |
| `infer_v2_latest_public.json` | 2026-10-29 全量新训 V2，公共域 best |
| `infer_v2_latest_carla.json` | 2026-10-29 新 CARLA 微调 V2，best |
| `infer_v3_public.json` | V3 连续面片研究版，未通过升级门槛 |
| `infer_public_fast.json` / `infer_carla_fast.json` | 历史推荐权重的可选浮点重排快速路径，不能承诺逐像素完全相同 |
| `carla_data.json` | 新 CARLA 数据清单、物理范围、真实四相机及缺失掩码 |

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  official_methods/.venv/bin/python tools/infer_public.py \
  --config configs/current/infer_v2_latest_carla.json
```

`checkpoint` 决定模型结构与参数，`data_config` 只修改数据。`indices` 选择样本，`dataset`/`split` 选择数据域与划分，`output` 必须是新目录。推理不读取深度/语义标签；参考相机缺失时不会偷偷改用别的参考。

主图为 256×448、匹配图为 512×896，输出一个参考视图的深度。默认启用已经验证的局部卷积布局整理，其他快速重排关闭。输入相机数取自实际数据，不能将双目复制为四目。

正式发布备份在 `runs/model_releases/RulerMVS-V2/` 和 `RulerMVS-V3/`；最新正式训练在 `runs/active/v2_restart_20261029/{public,carla_new}/`，保留 best、final、last 和阶段检查点。last 是恢复点，best 是按 VAL 选择，不能互相替代命名。公共与 CARLA 权重分别保留。

2026-10-19 的 `model/base_model/dev/test/train_reader/runtime/video/retest_original` 等历史配置已从当前目录移除，其文本保存在[清理前源码快照](../../runs/active/project_cleanup_20261031/source_before_cleanup.zip)。旧版数据和模型兼容代码仍因调用关系保留，不将它们误称为当前 V2/V3。

2026-10-31新增独立研究版本 **RulerMVS-V5**：在20261029 V2上增添共享CNN到ViT的可拆卸连接，原V2默认不改。逐层结构、短训证据及独立训练/无标签推理入口见[新模块说明](../../research/rulermvs_v5_20261031/README.md)。完整训练产物位于`runs/active/rulermvs_v5_20261031/`，正式训练结束前不把短训权重称为新推荐。V5使用独立`rulermvs_v5`标识，不能直接交给只识别旧V2/V3的入口；使用该目录的`infer.py`及JSON配置，`bridge_enabled`控制连接开关。

2026-11-01新增独立实验版本 **RulerMVS-V6**：保留 V5 编码和几何取证，在细候选处分开“选择候选族”和“族内连续定位”，用可撤回的父概率交接及相应训练目标适配。结构见 [model.py](../../research/rulermvs_v6_20261101/model.py)、[计算图](../../research/rulermvs_v6_20261101/architecture.mmd)；实际结果与限制见 [研发记录 197 起](../../docs/research_memory.md)。已完成同预算短训、完整 VAL 以及八卡全模型训练/恢复冒烟；北京时间19:28启动独立全量训练，尚未完成本轮新 TEST，不替换默认 V2/V5，也不宣称全面优于 baseline。

V6 独立入口只接受 `rulermvs_v6` 权重，参数在 [configs](../../research/rulermvs_v6_20261101/configs/) 的 JSON 中设置。`infer_carla_eager.json` / `infer_carla_fast.json` 分别采用普通执行与可选局部编译；快速路径会改变浮点运算顺序，首次编译时间不计入稳态耗时。公共域对应 `infer_public_*`；`pilot_*`、`round2_*` 是已完成实验的精确配置，重新运行前须指定新的数据盘输出目录。

```bash
official_methods/.venv/bin/python research/rulermvs_v6_20261101/infer.py \
  --config research/rulermvs_v6_20261101/configs/infer_carla_fast.json
```

V6 权重、验证结果与编译缓存均位于 `runs/active/rulermvs_v6_20261101/` 的数据盘软链接下。新增模块可通过 checkpoint 配置的 `handoff.enabled` 关闭；`handoff.mode` 和 `within_group` 用于读出消融。关闭模块仍使用当前 checkpoint 的其余参数，不等同于重新加载原始 V5 权重。

`handoff.upsample_feature` 支持 `winner`（当前默认）和 `group_mean`（研究对照）。后者已完成固定权重与上采样头适配测试，尚无通用收益，因此默认关闭；它的延迟尚未单独测试。`mask_*` 配置只训练原上采样头，用于复查这一结论。

V6全量训练采用 [公共训练配置](../../research/rulermvs_v6_20261101/configs/train_public.json) 和 [新CARLA配置](../../research/rulermvs_v6_20261101/configs/train_carla_new.json)，详细曝光量和设计依据见 [训练计划](../../research/rulermvs_v6_20261101/configs/training_plan.json)。公共七域42598组、2+3+4轮，从已验证的V6短训权重继续训练；新CARLA497组、8+24轮，从本轮公共best迁移。每卡batch4，共八卡；训练、VAL统一共同RGB256×448，匹配输入仅由这些像素上采样。按阶段解冻，保留独立学习率、可见对应监督、缺相机掩码和参考轮换；不读取TEST进行权重选择。

本轮会话与输出：

```bash
tmux attach -t rulermvs_v6_full_20261101
# 只读进度，也可不连接会话：
tail -n 5 runs/active/rulermvs_v6_20261101/full_training/public/events.jsonl
```

`full_training/public/`与`full_training/carla_new/`分别保存本轮`best.pt`（完整VAL选优）、`final.pt`（训练末步）、`last.pt`（优化器和各rank随机状态恢复）。整个目录经既有软链接写入数据盘，保存前检查至少200GiB空闲。`source_snapshot/`和评测计划已冻结，运行中不要修改对应源码/训练配置。

[run_full.sh](../../research/rulermvs_v6_20261101/run_full.sh)按公共训练→CARLA训练→完整评测执行，已有恢复点时自动恢复；已运行时不要重复启动。评测对照V2/V4/V5/V6的VAL-best及V6末步，并比较MVSAnywhere细化、PatchmatchNet和DiffMVS；V3因只有公共权重单列迁移对照。完整精度采用eager，共同像素/深度范围/掩码/固定数据划分；部署编译测速和176帧数值漂移另外记录，不将编译速度冒充精确保持输出。最终权重性能和速度以此次训练后的结果为准。
