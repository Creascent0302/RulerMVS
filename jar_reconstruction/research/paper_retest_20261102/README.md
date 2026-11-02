# V6论文主实验重测

复用已冻结V6及baseline权重、完整TEST清单和原评价器。模型及训练源码不改；V6快速路径独立重新计算深度、覆盖率、阶段和可见表面。SimpleRecon旧四图权重不参加双目；无任务权重的MVSFormer++不作精度排名。

所有结果保存在`runs/active/paper_retest_20261102`的既有数据盘软链接下。`prepare.py`只在首次运行时执行，以免覆盖本轮冻结配置；执行参数保存在该目录的`configs/*.json`和`plan.json`。

在项目根目录执行：

```bash
official_methods/.venv/bin/python research/paper_retest_20261102/prepare.py
official_methods/.venv/bin/python research/paper_retest_20261102/evaluate.py \
  --config runs/active/paper_retest_20261102/configs/public4.json --smoke
bash research/paper_retest_20261102/run.sh
```

`run.sh`先完成八卡精度评估，再逐域独立进程在同一GPU上交替测速，最后生成`docs/paper_results_consolidated_20261102.md`和数据盘CSV/PDF/SVG。任务失败时保留日志，不填完整结果。重启可复用本轮已完成且协议一致的分片，不复用历史轮次结果。

耗时使用无标签TEST输入、各域等间隔16组、首组预热10次、每组10次随机交替；GPU常驻完整深度输出与热文件缓存下含IO/H2D/D2H的数据管线分开。各模型独立驻留测显存，CARLA四参考实际执行，自研共享编码与串行输出另作核对。网络耗时不含评测专用逐帧随机流哈希，但保留Patchmatch/Diff原生随机计算。全部精度评测仍固定逐帧随机流。

当前进展与正式完成情况以`accuracy/*/full_test/summary.json`、`speed/*.json`、`delivery.json`和`queue.exit_code.txt`为准。源数据重复曝光和训练预算差异沿用`docs/mvsanywhere_dataset_comparison_20261102.md`，不把同输入系统比较写成同预算架构胜出。
