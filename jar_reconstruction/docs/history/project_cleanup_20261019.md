# 项目整理记录（更新至 2026-10-20）

已按用户进一步授权，永久删除无用旧实验产物。当前核心代码、原始 CARLA 数据、最新一两轮测试及必要权重保持不变。日常使用从[当前配置](../../configs/current/README.md)和[文档索引](../README.md)进入。

## 本轮实际删除

- 永久删除 `cleanup_20261019/retired_artifacts/` 和 `local_retired/` 中的旧缓存、重复检查点、早期预测输出、停用原型和旧独立环境。
- 清理清单包含 **11,520 个文件**，逻辑大小约 **101.75 GiB**；执行前后观测到磁盘可用空间增加约 **101.78 GiB**。文件逻辑大小与磁盘分配空间略有区别。
- 先移出保留了 9 个仍被文档引用的历史图表和核查文件，共约 9.5 MiB；它们位于 `reference_assets/`，文档链接已同步更新。
- 删除后核对了 **570 个受保护文件**的状态；核心代码、配置及关键权重另核对 SHA256，均保持不变。

## 保留内容

- MF-ET-JAR 当前核心代码、实际接入主流程的独立消融开关、可选多焦距模块和几何校正插件。
- 最新原始特征读取实验的训练缓存、模型、DEV/TEST 指标与逐帧记录、计时、对照视频；MVSAnywhere 支路消融和初值替换的近期结果。
- 九候选基础模型、条件读取、静态读取及同预算旧头对照，以及当前 Fast-MVSNet、SimpleRecon、MVSAnywhere、CasMVSNet 和所需预训练权重。此次重点核对了 8 份独立权重文件。
- 六种官方方法的独立源码、适配器、许可证、有效运行环境；原始数据及数据划分清单。
- 当前测试相关代码、必要配置、数据与指标说明。近期结果仍保留在原路径，不需要重新生成。

## 日常入口与存储

| 路径 | 用途 |
|---|---|
| `configs/current/` | 当前模型、基础对照、DEV、TEST、读取模块训练、计时和视频，共 7 份 JSON |
| `runs/active/` | 新实验输出，软链接到数据盘 `active_research/` |
| `runs/candidate_revision_20261018/` | 近期模型、指标、缓存与视频 |
| `runs/cleanup_20261019/source_before_cleanup.tar.gz` | 清理前自研代码、配置、测试和文档备份，约 3 MiB |
| `runs/cleanup_20261019/history_records.tar.gz` | 历次实验文字记录的压缩备份，约 51 MiB；不含旧大缓存和权重 |
| `runs/cleanup_20261019/reference_assets/` | 9 个仍被文档引用的历史核查文件和图表 |
| `runs/cleanup_20261019/permanent_deletion_result_20261020.json` | 本轮实际删除及保留核对结果 |
| `runs/cleanup_20261019/permanent_deletion_plan_20261020.json` | 本轮逐文件清单、受保护文件及哈希 |

原始数据通过 `/home/liuzonghao/3Dscene/data/CARLA_data` 访问，未纳入删除范围。旧大缓存和废弃权重已永久删除，无法再从归档恢复，需要重新运行代码生成。清理前源码仍可按需提取单个文件，例如：

```bash
tar -xzf runs/cleanup_20261019/source_before_cleanup.tar.gz \
  tools/probe_candidate_information.py
```

## 上一轮代码整理和验证（2026-10-19）

工具脚本 157→48，JSON 配置 523→140，文档 68→21，日常入口集中到 `configs/current/`。旧一次性脚本、过期配置和文档、未接入主线的 19 个研究原型及对应测试已移出；保留的运行模块没有因清理而简化算法。

当时先完整归档旧产物，等待大范围永久删除的明确授权。本轮收到授权后删除了上述两个旧产物目录；10月19日的归档日志仅表示当时状态，以10月20日的实际删除记录为准。

- CPU 完整回归：**750 项通过、5 项跳过，266 个子测试通过**。日志见 [cpu_tests.log](../../runs/active/cleanup_verification/cpu_tests.log)。
- GPU：RTX 5090 上七组已有权重实际加载并完成前向，输出有限；覆盖自研三个版本、Fast-MVSNet、SimpleRecon 和 MVSAnywhere 两种运行配置。
- 当时恢复了必要配置，并在测试中为历史快照补齐固定 `coarse.stride=16` 元数据；预测张量继续逐位比较，没有重写预测快照。
- 本轮只删除已退出运行路径的产物，没有修改模型或运行配置，因此未重复训练或重新报告精度和正式计时。本轮文件核对结果见[实际删除报告](../../runs/cleanup_20261019/permanent_deletion_result_20261020.json)。
