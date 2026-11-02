# 将 MVSAnywhere 深度用作自研模型初值：实际对照（2026-10-19）

本实验回答“直接用 MVSAnywhere 的预测深度替换自研模型初值，初值和最终深度会改善多少”。这里调用了**完整官方 Hero 网络**（图像编码、候选代价体、ViT 与深度解码器），将其上相机深度映射到同一图像网格，再在自研模型 stride-4 细化入口替换深度；自研模型的编码器、候选头、细化器与稠密输出头参数均冻结。没有把单独的 DPT 头权重硬接到不兼容的自研特征上，也没有将组合成绩冒充自研模型单独推理。官方 Hero 的结构见[官方仓库](https://github.com/nianticlabs/mvsanywhere)。

输入均是同一帧四张 RGB 和已知标定，只评价上相机（物理相机索引 2）的光轴深度。官方网络每次只算该参考视角；自研模型仍读取四目，但快速路径只解码上相机。两个预测都不接收 GT；真值在全部前向之后按同一 2–80 米掩码评价。`DEV25` 与训练集来自同一序列并隔离 40 帧；`TEST200` 是另一条 CARLA 轨迹，但此前已用于研究诊断，不能再视作全新盲测。CARLA 原始深度采集脚本尚未取回，标签绝对尺度仍需复核。

## 四种输入及指标

“初值”在同一 stride-4 真值射线上评价；“最终”在同一全分辨率真值射线上评价。下表都是 MAE，单位米，数值越低越好。“官方单独”保留 MVSAnywhere 的全分辨率输出；“组合最终”只用其 stride-4 样本作为自研细化的起点，再执行自研后端。

| 划分与权重 | 官方输出方式 | 自研原初值 | 替换用的官方初值 | 自研原最终 | 官方单独 | 组合最终 |
|---|---|---:|---:|---:|---:|---:|
| DEV25，原始 Hero 预训练 | 首遍 | 1.122 | 1.076 | 0.914 | 1.076 | **0.851** |
| DEV25，CARLA 微调 | 首遍 | 1.122 | 0.965 | 0.914 | 0.963 | **0.817** |
| DEV25，CARLA 微调 | 再做一次范围重估 | 1.122 | 0.938 | 0.914 | 0.938 | **0.817** |
| TEST200，原始 Hero 预训练 | 首遍 | 2.519 | 1.460 | 2.221 | 1.476 | **1.407** |
| TEST200，CARLA 微调 | 首遍 | 2.519 | **1.106** | 2.221 | **1.120** | 1.131 |
| TEST200，CARLA 微调 | 再做一次范围重估 | 2.519 | 1.132 | 2.221 | 1.150 | 1.181 |

原始 Hero 权重为 `runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt`，无 CARLA 微调；CARLA 微调权重为 `runs/mvsanywhere_audit_20261013/finetune_verified/last.pt`，该版本来自项目中已经核查的官方 Hero 训练路径。首遍与范围重估使用**相同权重**，只切换官方推理时的第二次深度范围搜索。

在另一条轨迹的 200 帧上，相比当前自研模型，CARLA 微调首遍的官方初值使初值 MAE 从 **2.519 降至 1.106 米**，组合最终从 **2.221 降至 1.131 米**。这是约 **1.413 米的初值改善**和约 **1.089 米的最终改善**，但组合比单独使用官方的 1.120 米略差 0.011 米。原始 Hero 不微调时，组合最终为 1.407 米；可见官方预训练本身有帮助，CARLA 微调还有较大收益。不能据此把 MVSAnywhere 的任务先验计入自研模型的独立能力。

## 为什么开发集组合更好，另一轨迹却没超过官方

先把官方全分辨率深度降采样到 stride 4，再插值回原图，就会丢失细节。TEST200 微调首遍版本的官方全图 MAE 是 **1.120 米**；仅做这一步降采样/逆深度插值就变为 **1.152 米**。自研细化及全图头将其改善到 **1.131 米**，所以不能把相对官方原图的 0.011 米差距全部归咎于细化器。在 24–80 米，官方全图、降采样插值、组合最终分别为 **4.119、4.294、4.368 米**：远处细节损失和错误修正仍然突出。近、中距离组合反而优于官方。

DEV25 上组合最终 0.817 米，优于官方 0.963 米；另一轨迹上组合不再优于官方。这说明当前后端对不同初值分布和场景变化的修正不稳定，尤其是远距离。官方首遍在 TEST200 比自身范围重估后的结果更好（1.120 对 1.150 米），但本实验不能单凭这一差异证明范围重估的具体故障原因。

另测了一个**独立后处理诊断**：保留官方原图深度，只把自研 stride-4 细化得到的逆深度残差插值回原图。微调首遍的 TEST200 MAE 为 **1.092 米**，比官方单独的 1.120 米略低，但 RMSE 从 **3.463 升至 3.750 米**，说明大误差尾部变重；它不属于自研原生输出，也不应直接用作推荐结果。

## 时间与实际边界

同一张 RTX 5090 上逐帧 GPU 同步的诊断中位数（未做正式预热，排除磁盘解码）：微调首遍官方约 **63 毫秒**，后接自研模型约 **71 毫秒**，顺序组合约 **134 毫秒**；再做一次范围重估时官方约 **112 毫秒**，组合约 **183 毫秒**。这已经超过目前约 100 毫秒一次更新的研究目标，还未计入车端输入与后处理。组合也重复运行两套大网络，不能被描述为“只加了一个轻量预测头”。

当前更值得研究的是离线借用官方预测监督自研候选与可靠性判断，或在自研模型里设计能保留全分辨率初值细节的残差传递；两者都需要独立训练与验证，**本实验没有证明这些轻量方案能达到表中精度**。直接替换的价值是确定可达方向和瓶颈，而非得到满足实时目标的最终架构。

## 代码、配置与可复核结果

实验脚本是 [`tools/probe_depth_initialization.py`](../tools/probe_depth_initialization.py)。本轮将其参考相机索引转换为单参考解码器的局部槽位，旧四参考行为保留；对应 4 项 CPU 测试通过。配置在 `configs/candidate_revision_20261018/mvsanywhere_initial_swap_*.json`，权重、逐帧结果与汇总均经 `runs/candidate_revision_20261018` 软链接存到数据盘。关键结果目录：

- `runs/candidate_revision_20261018/mvsanywhere_initial_swap/dev_complete_current_verified/`
- `runs/candidate_revision_20261018/mvsanywhere_initial_swap/test_complete_firstpass/`
- `runs/candidate_revision_20261018/mvsanywhere_initial_swap/test_complete_current_verified/`
- `runs/candidate_revision_20261018/mvsanywhere_initial_swap/test_complete_pretrained_firstpass/`

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
official_methods/.venv/bin/python -m pytest -q tests/test_depth_initialization_probe.py
official_methods/.venv/bin/python tools/probe_depth_initialization.py \
  --config configs/candidate_revision_20261018/mvsanywhere_initial_swap_firstpass_test_complete.json \
  --device cuda:0
```

结果目录若已有 `frames.jsonl` 会拒绝覆盖。两份早期 TEST 汇总文件（`test_complete_current_verified`、`test_complete_firstpass`）的注释仍沿用旧脚本中的“没有运行 TEST”字样；其 `experiment.split=test_complete`、`frame_indices=0..199` 和实际 200 行逐帧记录确认执行的是 TEST200。该注释已在脚本中修正，本报告据实际配置与逐帧记录说明划分。
