# 2026-10-31 项目清理

本轮按用户要求整理项目，保留 V2/V3 权重、有用测试证据及独立 baseline。同一工作区另有 `ruler_flow_v2_20261031` 正在进行的新研究，本轮清理排除其代码和结果，没有停止该任务。没有改变 V2/V3 的网络计算、学习参数、默认权重、数据划分或 baseline 算法。

## 保留内容

- `runs/model_releases/` 中四套正式发布备份：V2 公共推荐、CARLA 推荐、读出 VAL 最佳备选，以及 V3 面片最终版，含 SHA 和测试时源码。
- `public_pretrain_no_hero_v2`、`public_retrain_latest_20261025` 的正式训练/微调权重与恢复点；`near_mid_20261027` 的正式 V2 对照和 V3 训练权重。
- `v2_training_20261028/{public,carla_new}` 与 `v2_restart_20261029/{public,carla_new}` 的正式权重、阶段检查点、best/final/last；新 CARLA 准备清单完整保留。
- 原始公共/CARLA 数据、通用 DA-V2/ImageNet 初始化、baseline 训练/官方权重、六个官方方法封装、上游源码和安装环境。
- 全部历史指标 JSON/CSV、训练日志、配置与已有源码快照；最近两轮 `role_fusion_20261030` 和 `ruler_flow_20261030` 的完整实验目录，包括权重、冒烟结果和诊断。
- 单一研发记录 `research_memory.md`。旧结论未改写为成功，失败研究历史继续可查。

## 清理内容

从原 191.38 GiB 的项目运行产物中删除约 **131.64 GiB**，主要为旧特征缓存、未采用原型权重和早期冒烟/恢复检查副本。典型缓存包括 `public_refinement_20261025/features` 约 27.10 GiB、`architecture_screen_20261026/features` 约 10.41 GiB。

删除清单中的 **78 个历史工具文件、338 份历史配置、139 个历史测试/夹具文件**。进一步检查静态依赖闭包与当前源码字符串引用后，移除 **17 个无当前调用方的旧原型模块**。没有按日期粗删 `jar/models`：旧命名但仍被 V2/V3、baseline 或新实验调用的共享模块继续保留。

源码整理前已生成约 **4.61 MiB** 的压缩快照，包含 1417 个项目自有源码/配置/文档等文件，保留原相对路径，不重复打包官方上游和虚拟环境。被删除的源码均在压缩包中核对过。**大型旧权重和缓存不备份**，不能把源码快照理解为可直接恢复原来的数值实验。

27 个历史文档/图示集中到 `docs/history/` 并重算相对 Markdown 链接。根目录、主项目、配置、工具和研究目录增加或重写简明索引。当前配置目录只保留 V2/V3 推理、数据和版本登记；新加 `infer_v2_latest_public.json`、`infer_v2_latest_carla.json` 方便选择最新正式权重，原默认不变。

## 验证及限制

实际结果和命令记录集中在[数据盘审计目录](../runs/active/project_cleanup_20261031/)。

- 首次合并多个研究目录运行 pytest 时，出现重复 `test_runner.py` 名称和相对导入的收集错误；没有执行为通过。已在 `pyproject.toml` 设置 importlib 导入及 namespace package 解析，避免各独立研究测试相互覆盖。
- 最终合并回归 **396 项通过、3 项跳过，51 个子测试通过**（26.12 秒）。此前分次 348/48 是相同测试的分组执行，不累加。详见 `final_validation.json` 及 `final_tests.log`。
- 六套已登记 V2/V3 权重分别通过正式无标签入口运行真实 TartanAir/CARLA 单帧 GPU 推理。输出均为 `[1,1,256,448]`、有限正米制深度，见 `gpu_inference_verification.json`。它只验证运行入口，没有重测完整精度或延迟。
- 清理前后权重和保留核心源码按 SHA256 核对；17 个明确归档的孤立原型单独登记，不冒充仍在原路径保留。见 `protected_sha256.json`（原始集合）、`retained_sha256.json`（最终保留集合）、`final_validation.json`。
- 检查数据/运行/官方别名软链接和当前索引链接。没有删除原始数据，也没有重训模型；未重新执行所有官方模型的完整 GPU 前后向、DDP 或完整 TEST。

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  official_methods/.venv/bin/python -m pytest -q \
  tests research/ruler_flow_20261030 research/role_fusion_20261030 \
  research/v2_restart_20261029 research/v2_training_20261028 \
  official_methods/tests/test_method_packaging.py \
  official_methods/tests/test_manifest.py official_methods/tests/test_shared.py \
  official_methods/tests/test_mvsanywhere_audit.py --disable-warnings
```

## 清单和历史追溯

审计根目录：`runs/active/project_cleanup_20261031/`，实际在 `/data/liuzonghao_data/3Dscene_runs/active_research/project_cleanup_20261031/`。

- `deleted_files.json`：逐文件删除清单及原因。
- `jar_run_plan.json`：大型产物保护范围及删除计划。
- `jar_code_plan.json`、`removed_orphan_modules.json`：源码清理决策。
- `source_before_cleanup.zip`：清理前项目自有源码、配置和文档快照。
- `document_moves.json`：旧文档/图示迁移映射。
- `final_validation.json`：最终校验、测试与实际磁盘空间。

历史文档引用的旧缓存或原型权重可能已删除；旧专项工具和配置可以在上述 ZIP 按原路径查找。不要重新创建空权重占位，也不要把缺失历史证据解释为新的模型结果。以后大产物继续写入现有数据盘软链接。
