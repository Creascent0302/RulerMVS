# 工具入口

从 `jar_reconstruction/` 运行。参数优先修改配置文件，命令行只选择配置。

| 工作 | 入口 |
|---|---|
| V2/V3 无标签推理 | `infer_public.py` |
| 基础多域训练 | `train_public.py`；最新训练器另见 `research/v2_restart_20261029/` |
| 统一精度评测 | `compare_public_pretrained.py`、`evaluate_public_depth.py` |
| 同卡速度/显存 | `benchmark_public_speed.py`、`profile_public_inference.py` |
| 完整结果汇总 | `report_full_test.py` |
| 新 CARLA 解析与清单 | `prepare_carla_new_20261028.py` |
| 公共数据转换 | `prepare_hypersim*`、`prepare_tartanair.py`、`prepare_matrixcity.py`、`prepare_vkitti2.py`、`prepare_mvssynth.py`、`prepare_blendedmvs.py`、`prepare_dynamicreplica.py` |
| 固定划分/注册 | `prepare_public_training.py`、`prepare_pretraining_registry.py` |
| 数据下载（服务器网络） | `with_server_network.sh` 包裹 `download_*`；不更改交互网络 |
| 小规模架构比较 | `screen_public_architecture.py`、`refine_public_candidates.py` |
| 数据/几何检查 | `smoke_public_multiview.py`、`smoke_outdoor_datasets.py`、`audit_*` |

剩余部分旧工具被以上入口或研究代码导入，因此保留。不要凭文件名认为它是当前模型入口。2026-10-31 删除的历史专项工具可从数据盘源码快照追溯；缓存重建需要相应原始配置，旧缓存路径不保证仍有内容。
