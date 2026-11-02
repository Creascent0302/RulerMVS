# CasMVSNet 作者代码与适配边界

来源：[alibaba/cascade-stereo](https://github.com/alibaba/cascade-stereo)，CVPR 2020 论文《Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching》的作者实现。具体版本和原文件 SHA256 在 [provenance.json](provenance.json)，许可保留在 [LICENSE](LICENSE)。仅下载两个网络源文件和许可证，没有下载训练数据或权重。

本地兼容改动：

1. 删除 `module.py` 中未使用的 `local_pcd` 导入与修改导入路径的语句，避免依赖作者的整套工具目录。
2. 在单应变换采样处显式设置 `align_corners=True`。作者把坐标除以 `width-1` / `height-1`，对应旧 PyTorch 默认行为；当前默认值已经改变。单位变换的像素恒等测试覆盖此修正。

`jar/official_casmvs.py` 保留作者的三阶段特征网络、方差代价体、三维卷积和深度回归。适配器轮换各台有效相机作为参考相机；内参按实际特征步长缩放，外参不改变。训练沿用各阶段深度 SmoothL1 监督思路。空标签跳过，正常图像要求高宽为32的倍数。当前整个作者网络用 FP32 运行，没有悄悄套用未验证的混合精度优化。

为接入本项目，深度输出额外经过预测深度之间的一致性检查，再按固定、与标签无关的顺序限制表面点数量。尺度、法线、unknown 和空间评估来自适配器，并非 CasMVSNet 原论文的完整后处理。输出的 `dense_depth` 保留原始深度，便于同时报告过滤前误差与过滤后覆盖率。

该对比属于“作者网络在本项目合成数据上从头训练”。它不使用四个研究模型的共同骨干，不能当作同骨干消融；少量训练不能视为复现论文公开榜单成绩。官方室内/室外数据集的训练、融合参数和最终评分流程尚未复现。
