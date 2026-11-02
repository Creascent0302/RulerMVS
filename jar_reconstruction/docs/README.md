# 文档索引

研发前先读 [research_memory.md](research_memory.md)。它保留历次假设、结果、失败原因和重试条件；本轮整理没有改写历史结果。

## 当前模型与操作

- [项目入口与目录结构](../README.md)
- [V2 / V3 权重和推理配置](../configs/current/README.md)
- [V2 详细架构，供 AI 讨论](RulerMVS_V2_模型说明_供AI讨论.md)
- [V2 数据流与分层尺寸](ruler_direct_v2_dataflow.md)，[可编辑计算图](diagrams/自研RulerDirectV2_数据流_20261026.drawio)
- [V4 可编辑计算图（总图、匹配融合、Flow三页）](diagrams/V4_架构计算图_20261031.drawio)，[总图SVG](diagrams/V4_架构计算图_20261031_01.svg)
- [独立 baseline](../official_methods/README.md)
- [研究代码索引](../research/README.md)
- [2026-10-31 整理记录](project_cleanup_20261031.md)

## 结果和实验依据

- [V4 完整训练、两套 TEST 与基线测速](v4_evaluation_20261031.md)，对应研发记录184–186；有局部收益但未替换V2
- [最新 V2 完整 TEST 原始汇总](../runs/active/v2_retest_20261030/report.json)，对应研发记录 159 及 160–165
- [V2 共同评测协议](v2_evaluation_20261029.md)
- [V2 / V3 完整比较](latest_training_evaluation_20261028.md)
- [前轮公共数据和旧 CARLA 完整对照](full_test_20261026.md)
- [最新按职责融合试验](../research/role_fusion_20261030/README.md)，研发记录 170–173
- [最新局部 Flow 试验](../research/ruler_flow_20261030/README.md)，研发记录 174–178
- [MVSAnywhere 分支消融及强特征的利用](mvsanywhere_branch_ablation_20261019.md)
- [完整初值接入的历史诊断](mvsanywhere_initial_swap_20261019.md)

不同版本、训练预算、参考视角、数据划分和单位的结果不能混表。历史报告保留原日期及适用条件；“最新”以单一研发记忆顶部和对应原始 JSON 为准。

## 数据与复现

- [公共数据的本地布局、解析、划分和预训练](public_datasets_pretraining_20261024.md)
- [数据集下载教程](mvsanywhere_dataset_download_20261023.md)
- [CARLA 缺帧、深度和清单](carla_capture_data.md)
- [共享输入输出契约](data_contract.md)
- [指标计算审计](evaluation_audit_20261011.md)
- [MVSAnywhere 官方协议审计](mvsanywhere_protocol_audit_20261011.md)
- [历史设计与图示索引](history/README.md)
