# 各 baseline 的独立目录

每种方法拥有自己的适配器、官方源码、配置和说明，共享训练、数据与评价工具仍在 `official_methods/`。目录独立不等于算法间共享预测头。

项目还有独立封装的 [CasMVSNet、Census 和 ZNCC](../../jar/baselines/README.md)，它们同属 baseline，沿用项目早期的共同评价入口。Census/ZNCC 是工程对照，不冒充官方论文复现。

| 方法 | 目录说明 | 原始模块 |
|---|---|---|
| Fast-MVSNet | [使用说明](fast_mvsnet/README.md) | 稀疏深度、学习传播、Gauss–Newton |
| SimpleRecon Hero | [使用说明](simplerecon/README.md) | 元数据匹配、二维代价编码、UNet++ |
| MVSAnywhere Hero | [使用说明](mvsanywhere/README.md) | 视图无关匹配、双 ViT-B、DPT |
| MVSFormer++ | [使用说明](mvsformerpp/README.md) | DINOv2、特征交互FMT、粗代价CVT与多级细化 |
| PatchmatchNet | [使用说明](patchmatchnet/README.md) | 论文指定版本的候选传播与自适应评价 |
| DiffMVS / CasDiffMVS | [使用说明](diffmvs/README.md) | 置信度控制扩散与循环局部匹配 |
| OctreeOcc | [使用说明与发布阻塞](octreeocc/README.md) | 官方源码不完整，不能当作已跑通模型 |
| GaussianFormer-2 | [使用说明与数值限制](gaussianformer2/README.md) | Prob-256 分布初始化和概率高斯叠加 |

`niantic_common.py` 只共享 SimpleRecon 和 MVSAnywhere 的输入输出适配，网络分别在各自 `upstream/` 中。DINOv2、Pointcept 和编译环境是公共依赖，不复制进每个方法。

旧的 `upstream/<name>`、`configs/<name>_*.json` 和对应补丁路径均为相对软链，历史命令、已保存配置和权重仍能找到同一份文件。Python 导入继续使用 `official_methods.methods.<name>`；具体实现现在位于其 `adapter.py`。

目录兼容性测试不需要 GPU：

```bash
official_methods/.venv/bin/python -m pytest official_methods/tests/test_method_packaging.py -q
```
