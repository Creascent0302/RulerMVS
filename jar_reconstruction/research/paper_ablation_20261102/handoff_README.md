# 候选交接：同起点、同预算、三随机种子实验

问题是：V6的收益来自候选族读出、粗层证据继承、空间自适应门，还是训练目标的改变？此前完整V5/V6训练经历不同，直接比较不能分离这些原因；只在训练好的V6上关模块也不能代替重新训练。

本实验保留生产模型与正式TEST权重，所有变体从同一个V5权重开始。公共和CARLA实验分别从对应V5的`best.pt`开始，彼此独立，不继承本轮V6训练历史。训练与验证没有TEST路径。

| 组别 | 细候选读出 | 粗层证据交接 | 新门参数 | 训练目标 |
|---|---|---|---:|---|
| v5 | 原点赢家 | 无 | 0 | 原点目标 |
| marginal | 先选族，再族内均值 | 无 | 0 | 族目标 |
| constant | 先选族，再族内均值 | 可学习空间常数门 | 1 | 族目标 |
| parent | 先选族，再族内均值 | 质量统计决定的空间自适应门 | 129 | 族目标 |
| parent_oldpoint | 先选族，再族内均值 | 同上 | 129 | 原点目标 |

常数门与自适应门均从`sigmoid(-2)`开始，已通过初始输出逐位相等的CPU测试。常数门在所有像素、所有图像共享一个可学习标量，不接收质量统计；它不是从VAL挑选的固定最佳门值。候选族不等同于物理实例，同一表面可能属于多个族。

比较`constant→parent`只改变门是否依赖质量；`parent_oldpoint→parent`只改变目标。`v5→parent_oldpoint`保留同一个点目标，读出与交接一起改变。`v5→marginal`同时改变读出和目标，不能把差异全归因于其中一个。

训练预算预登记为：公共每域抽取128个真实TRAIN样本、2轮、batch=2，共896更新、1792曝光；CARLA抽取256个TRAIN样本、2轮，共256更新、512曝光。样本按序列平衡，三种子为20261102/03/04；同种子全部变体使用完全相同的样本、顺序和逐样本增强。CARLA清单名含上参考，但原`MixedDataset`训练时按`(index+epoch)`在有深度的相机间轮换参考，不能描述为只训练上相机。

编码器、匹配ResNet与CNN-token桥全部冻结；原解码器学习率`1e-5`，新门`1e-3`，AdamW，梯度范数裁剪1。与此前448/128更新先导实验相比，本轮更新预算翻倍，补齐常数门、旧目标控制和三种子重复。它仍是冻结编码器的有限预算适配消融，不是从零训练或充分训练后的模型上限证明。

每组保存固定末步，先在预登记的每域16个VAL样本检查候选供给、粗好→细坏、粗坏→细好与全图误差，再对每个末步权重评估完整公共7957或CARLA411组VAL。全部标签只在前向后评分。完整VAL不用于选择seed、epoch或默认模型。表格以每seed完整GT像素池化误差计算均值和样本标准差，另给配对seed差值；不把不同数据域的米和scene_unit混合平均。

入口：

```bash
official_methods/.venv/bin/python research/paper_ablation_20261102/handoff_prepare.py
official_methods/.venv/bin/python research/paper_ablation_20261102/handoff_queue.py --config runs/active/rulermvs_v6_20261101/full_training/paper_handoff_ablation/queue.json
official_methods/.venv/bin/python research/paper_ablation_20261102/handoff_summary.py
```

队列须等正式主评测及MVS首遍独占测速成功结束、GPU空闲后才运行；先执行两个新增工厂的真实GPU两步预检，再执行正式训练，最后完整VAL。每卡一个任务，GPU0–6；GPU7留给独立物理验证，不记录混跑速度。独立几何消融也由同一队列调度，防止抢卡。

完整VAL按帧分成七份。在公共或CARLA各自的19个末步模型中，冻结编码器参数及全部buffer逐项hash相同、图像/数据协议相同时，每帧仅编码一次，再分别运行19个解码器。每域每分片首帧，19个完整模型各自重新编码并前向，其粗/细/最终深度及有效掩码须与共享路径逐位一致；同时检查所有编码张量没有被解码器原地修改。两阶段不互相共享，源码另有快照。此复用仅节约离线评测，不能当作任何单模型速度；无法通过检查会报错，不能静默使用近似结果。

权重、逐帧统计、原始累加和日志都在数据盘软链接`runs/active/rulermvs_v6_20261101/full_training/paper_handoff_ablation/`。每组`completed.json`记录训练步数、梯度、冻结参数和落盘恢复；`handoff_audit.json`记录来源；`full_val.json`完整后才计为完成。结果以实际文件为准，本说明不预设任何一组获胜。

额外审计见`control_repeat_audit.json`：几何control与同seed交接parent的完整初始状态、数据顺序、初始输出相同，但第2步反向已出现微小数值差异，随后公共域存在小幅末步差异。未强制确定性CUDA，因此不承诺同seed训练逐位复现；三seed标准差也不是测试集泛化置信区间。另见`common_support_addendum.json`，在发现粗好条件率的分母可能随训练改变后、完整VAL开始前，追加双方粗层均好/均坏交集上的成对诊断。它不改变训练、主指标或权重选择。
