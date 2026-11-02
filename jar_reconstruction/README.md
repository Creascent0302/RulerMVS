# RulerMVS：V2 / V3 与独立 baseline

当前主线是 `jar/pretraining/model.py` 的 `PretrainingModel`。**V2** 使用标尺候选匹配和凸上采样；**V3** 使用连续局部面片和逐像素选面。V3 未通过升级标准，保留为研究对照。2026-10-29 新训 V2 与历史推荐分别保存，不自动用新权重覆盖旧推荐。

模型仅接收 RGB、标定和有效掩码。公共域使用真实的两目或四目协议；CARLA 使用四个广角相机，支持缺失源图像，暂不使用 tele。单参考深度的耗时不能当作四参考完整场景耗时。

## 项目目录

| 目录 | 内容 |
|---|---|
| `jar/pretraining/` | 当前编码器、V2/V3 模型、损失、多域加载、训练、推理与公平评测 |
| `jar/models/` | 候选、标尺、空间交互等共享模块；部分历史命名仍被当前代码调用 |
| `jar/` 其余文件 | 标定几何、数据接口、运行工具及历史 checkpoint 兼容实现 |
| `official_methods/methods/` | 六种独立官方 baseline；原始代码、适配层、许可证和依赖未改动 |
| `configs/current/` | 当前版本登记、推理与 CARLA 数据设置 |
| `configs/pretraining/` | 正式多域训练配置及通用初始化设置 |
| `research/` | 独立研究、正式新训和全量评测；各目录用途见 [研究索引](research/README.md) |
| `tools/` | 保留的训练、推理、数据整理与评测入口，见 [工具索引](tools/README.md) |
| `tests/` | 当前模型、几何、数据和兼容入口测试 |
| `docs/` | 当前架构、数据、结果和单一研发记录；旧设计集中在 `docs/history/` |
| `runs/model_releases/` | 受保护的 V2/V3 发布权重及源码备份，软链接到数据盘 |
| `runs/active/` | 训练、指标、最新实验和清理审计，软链接到数据盘 |
| `../data/` | 原始公共数据与 CARLA 数据的软链接，未清理数据内容 |

2026-10-31 已删除过期实验缓存、未采用原型权重、历史专项脚本和配置，保留原始数据、正式模型、baseline、指标/日志和最近两轮完整实验。[清理记录](docs/project_cleanup_20261031.md)列出清单和验证。旧文档中的历史脚本或缓存可能已清理，不能把历史命令当当前入口；需要追溯时可从清理前源码归档恢复。

## 环境与运行

已安装环境为 `official_methods/.venv`；占据 baseline 另有 `.venv_occ`。依赖说明见[官方方法](official_methods/README.md)，本轮没有重新安装环境或变更 CUDA。以下从项目目录执行：

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction

# 默认保留的历史推荐 V2；参数在 JSON 中修改。
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  official_methods/.venv/bin/python tools/infer_public.py \
  --config configs/current/infer_public.json

# 最新全量训练 V2：公共 / 新 CARLA 各自使用独立权重。
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  official_methods/.venv/bin/python tools/infer_public.py \
  --config configs/current/infer_v2_latest_public.json

# V3 使用同一推理入口，结构严格从 checkpoint 恢复。
tools/with_server_network.sh env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  official_methods/.venv/bin/python tools/infer_public.py \
  --config configs/current/infer_v3_public.json
```

CARLA 分别选 `infer_carla.json` 或 `infer_v2_latest_carla.json`。入口不会读取深度标签，输出到配置的 `output`；非空输出目录会报错，重新测试前应在配置中设置新目录。详细版本、权重 SHA 和适用数据见[当前配置](configs/current/README.md)。

## 训练与评测

最近一轮八卡全量训练入口是 `research/v2_restart_20261029/run_full.sh`，配置在同目录 `configs/train_public.json`、`train_carla_new.json`，训练实现为 `training.py`。这是已完成实验的原始配置；**要重新训练必须复制配置并设置新的输出目录**，不要直接覆盖本轮证据。通用 DA-V2 与 ImageNet 初始化权重仍在数据盘，不依赖 Hero 任务权重。

基础训练入口为 `tools/train_public.py --config ...`，八卡启动脚本为 `scripts/pretrain_public_8gpu.sh`。不要混用不同训练器的配置。最新完整评测和近距统计入口见 `research/v2_retest_20261030/`；共享方法比较为 `tools/compare_public_pretrained.py`，速度比较为 `tools/benchmark_public_speed.py`。已有结果目录不应覆盖。

- [V2/V3 完整比较及适用条件](docs/latest_training_evaluation_20261028.md)
- [V2 固定权重评测协议](docs/v2_evaluation_20261029.md)
- [最新 V2 完整 TEST 结果](runs/active/v2_retest_20261030/report.json)
- [最新职责融合试验](research/role_fusion_20261030/README.md)
- [最新局部 Flow 可行性试验](research/ruler_flow_20261030/README.md)
- [单一实验记忆：实际结果、失败原因与重试条件](docs/research_memory.md)

米制与 `scene_unit` 域分开比较；验证集开发实验不作为测试集成绩。测试协议、分辨率、参考相机、缓存、精度模式与后处理均须匹配。

## 测试与数据说明

```bash
# CPU 单元/接口测试；CUDA 测试在没有可用 GPU 时跳过。
OMP_NUM_THREADS=2 official_methods/.venv/bin/python -m pytest tests -q

# 最新局部 Flow 独立实验的测试。
OMP_NUM_THREADS=2 official_methods/.venv/bin/python -m pytest research/ruler_flow_20261030 -q
```

数据接入见[七公共域使用说明](docs/public_datasets_pretraining_20261024.md)、[CARLA 说明](docs/carla_capture_data.md)、[数据契约](docs/data_contract.md)。新 CARLA 的准备脚本为 `tools/prepare_carla_new_20261028.py`，已生成清单保留在 `runs/active/v2_training_20261028/carla_prepared/`。采集脚本 `../sensor_data_collector(5).py` 保留，用于核对深度定义。
