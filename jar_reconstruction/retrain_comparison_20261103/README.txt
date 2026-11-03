本轮三模型重训的共同测试与主表（2026-11-03）

不改训练结构和权重。固定三模型第9轮完整VAL-best，官方Hero另列外部参考。
复用之前11970公共组、CARLA174帧×四参考及ETH27对×两掩码的清单/GT规则。
七公共域输入范围与内部尺度采用新TRAIN统计；不是按TEST拟合。
本轮CARLA全部使用公共权重，不能把旧CARLA适配模型数字混进本表。

主表每格只有MAE、RMSE、ms。精度执行与计时均为相同快速部署路径：
V6/V7复用投影/局部编译；两套MVS均用fast cost volume、位置缓存和主深度头。
权重严格加载后才准备推理优化。FP32、TF32off、同RGB256×448、真实N2/N4。
没有以不同预测范围裁掉GT，缺失预测覆盖率另存且不用于美化误差。

图横轴为NMAE=Σ|预测−GT|/ΣGT，显示百分比；等于MAE/有效GT平均深度。
改变整个数据集的长度单位不改变这个值；它不等于逐像素AbsRel，也不消除
不同场景、视差、材质等难度差异。不存在跨数据集的全局Pareto排名。
纵轴为同卡完整深度输出中位ms，方法按颜色、数据集按符号。
外圈标记该数据集内非支配点，灰线只在同一数据集内连接前沿。
不挑选有利点；七公共域全部展示，CARLA固定上参考，ETH固定ALL掩码。
其他参考和掩码仍在完整主表。四参考实际运行时间另列，不能混为单参考。

运行（项目jar_reconstruction目录）：
official_methods/.venv/bin/python -m pytest -q research/retrain_comparison_20261103/test_metrics.py
official_methods/.venv/bin/python research/retrain_comparison_20261103/prepare.py
bash research/retrain_comparison_20261103/run.sh

prepare只首次运行，拒绝覆盖冻结配置；失败保留日志，可重启run.sh继续同协议分片。
模型、图、原始统计均由独立代码封装。精度复用原指标和GPU分片调度，跳过用户
此次未要求的表面最近邻/阶段统计，减少无关计算，不改变深度定义或GT掩码。
计时复用之前同卡交替协议：16组固定TEST无标签输入，每组10重复，首组10预热；
记录网络/共同格点输出/含IO管线三个范围，主表只取共同格点输出。
所有大文件在runs/active/overnight_v6_v7_mvs_20261103/comparison数据盘软链接内。
完成必须以delivery.json及queue.exit_code.txt=0为准，脚本存在不代表测试通过。
