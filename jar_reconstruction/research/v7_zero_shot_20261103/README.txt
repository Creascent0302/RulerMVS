V7零样本泛化实验，2026-11-03

问题：共同预算训练后的V7能否迁移到任务训练未包含的真实图像数据？
对照：固定V6/V7/重训MVSAnywhere公共VAL-best，加官方Hero细化外部参考。
权重均不更新。所有方法使用同一2/4/8张图、已知标定、256×448 RGB、
FP32/TF32off及0.1–100米输入搜索范围；不从测试GT取范围，不做尺度对齐。
四视图为主实验，2/8视图检验观测数量变化。自研训练仅有2/4视图。

ETH3D DSLR：官方MVSAnywhere版RMVD全部13场景104组。courtyard历史访问；
delivery_area/electro/facade/playground/terrace/terrains有同名历史双目场景。
kicker/meadow/office/pipes/relief/relief_2作为此前未分析场景组单列。
这里的zero-shot是任务训练未用该数据集，不等于研究者从未看过同名环境。
不认证DepthAnything/DINO/ImageNet等通用预训练的完整资产曝光。

Tanks and Temples：RMVD官方Barn/Church/Courthouse/Ignatius四场景69组。
原数据网站称其train split，但本项目只用于外测。只取官方配组前N幅，
不使用其可能由真值生成的depth_range。两数据集均保留实际原尺寸有效GT。

指标：主表场景等权MAE/RMSE/AbsRel；另存像素池化、delta1、coverage、
0–5/5–20/20米以上分层。不裁掉模型搜索范围外GT，不按模型输出筛GT。
RMSE为各场景RMSE平均，和全像素池化RMSE明确分开。V7−V6的配对区间按
场景簇bootstrap20000次，seed20261103；relief/relief_2合并抽样。
区间不覆盖训练seed变动，不把数百万像素当独立样本。

网络与存储：根目录/data/liuzonghao_data/dataset/V7_zero_shot_20261103，
3Dscene/data/V7_zero_shot_20261103及runs/active/v7_zero_shot_20261103软链接。
全部下载在tmux：curl -q、清除所有代理环境、--proxy ''、--noproxy '*'、
--interface eno5，校验local_ip=172.16.2.4，无代理回退。日志network.jsonl。
ETH逐场景下载/CRC提取，完成后仅清除本脚本临时归档；保留来源SHA和准备数据。
Tanks使用服务器HTTP Range读取ZIP目标成员，验证Content-Range和成员CRC。
全程保留200GiB，既有数据和权重不删除。

准备：
official_methods/.venv/bin/python research/v7_zero_shot_20261103/prepare.py --freeze
在tmux运行prepare.py，以及probe_tanks.py完成后运行prepare_tanks.py。
CPU验证：
official_methods/.venv/bin/python -m pytest -q research/v7_zero_shot_20261103/test_evaluation.py
历史场景接口检查：configure.py --smoke；evaluate.py --config results/smoke_config.json。
正式队列：在tmux运行run.sh。run.py按固定场景清单等待已准备的数据，三GPU
分别负责2/4/8视图，权重常驻。全部精度结束后再用GPU0独占计时。

结果：results/analysis/零样本泛化结果.md、results.json、per_scene.csv。
旧同数据家族留出结果从251同权重测试中重聚合，单独存supplemental文件，
不冒充本次新域结果。主报告不选择性移除效果差的场景。

队列完整结束后，在项目目录运行：
official_methods/.venv/bin/python research/v7_zero_shot_20261103/verify.py
official_methods/.venv/bin/python research/v7_zero_shot_20261103/audit_depth_extent.py
official_methods/.venv/bin/python research/v7_zero_shot_20261103/diagnose.py
official_methods/.venv/bin/python research/v7_zero_shot_20261103/plot.py
后两项只读取固定预测的统计结果，生成距离段/误差方向分解以及PNG/PDF/SVG。
verification.json核对2076次预测、原始GT分母、嵌套输入、权重/源码SHA和实际下载连接。
本轮归档慢时的临时并行调度保存于results/preparation_scheduling.json，核心适配未改。
新获取的数据不加入训练清单。本实验衡量深度零样本泛化，不代表动态场景、标定噪声
或完整车端三维系统已经通过验证。

来源：
https://github.com/lmb-freiburg/robustmvd/tree/master/rmvd/data
https://www.eth3d.net/datasets
官方ETH去畸变脚本和配组复用已有protocol目录，SHA记录于plan。
这是共同输入外测，不等同完整五域RMVD官方排行榜或论文原分辨率成绩。
