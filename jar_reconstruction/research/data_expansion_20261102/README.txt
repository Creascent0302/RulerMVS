2026-11-02 数据扩容与未见域测试

目的
增加真正的新场景/序列，不用重复采样伪装组数。官方八图 tuple 行数按
MVSAnywhere 5bd49bba0992aa56060e1aa3bb40b2150b04726f 审计：
Hypersim 45352 → 最少4536组，计划5500以上；
TartanAir 92111 → 最少9212组，计划10000以上；
DynamicReplica 70093 → 最少7010组，计划7200以上。
其余四个已接入公共域的原TRAIN数量已经超过对应公开tuple行数的十分之一。
这不是原论文训练预算的十分之一：本项目静态场景每组四图，动态场景只用同步
双目；不以跨时间动态物体伪装同步静态多视图。也不声称已覆盖SAIL等未接入域。

存储与网络
数据都在 /data/liuzonghao_data/dataset 下，3Dscene/data 下有同名软链接：
Hypersim_expanded、TartanAir_expanded、DynamicReplica_train、ETH3D_rmvd_subset。
审计、日志、训练配置在 public_expansion_20261102，已有数据和V6权重不覆盖。
direct_http.py 清除子进程代理，curl -q 禁用curlrc，显式 --proxy '' --noproxy '*'
并绑定 eno5。每次检查 local_ip=172.16.2.4，拒绝localhost。无代理回退。
当前对话代理不改变。下载必须通过tmux启动。network.jsonl记录实际连接。
新下载采用HTTP Range读取ZIP目标成员，核验Content-Range、长度和ZIP CRC；
没有下载整个ZIP时不声称验证了整包SHA。Dynamic官方metadata已验证整包SHA。
全部下载结束后还会重新核对落盘成员CRC，覆盖续传及切换镜像之前留下的文件。
可重启跳过此前原子落盘的已校验文件。保留至少200GiB，不删除用户旧数据。

数据处理
Hypersim：仅官方训练名单中新场景，提取tonemap RGB、深度和相机详情；把射线
距离转换为光轴Z深度，沿用已经验证的配组、姿态与像素中心转换。旧VAL/TEST不变。
TartanAir：增加office等环境，原始RGB/深度/分割/pose均保留。只用官方TRAIN轨迹，
按相机位姿和视锥重叠构造四图，每参考帧至多一组；选组不读GT。
DynamicReplica：读取官方真实train分包及完整相机注释，按sequence/frame/camera
严格同步左右目；PNG为float16的位模式，不是整数深度图。共享加载器新增显式
encoding并实时解码，避免复制全量float32标签。保守沿用scene_unit单位声明。
旧的600组来自官方valid，不再混入这版新TRAIN；其余旧VAL/TEST仍按序列隔离。
为了避免复制，扩容清单中的旧样本仍引用Hypersim_subset/TartanAir_forest等原目录，
VAL/TEST也保留原路径，因此这些旧数据目录不能直接删除。
配置name保留旧域ID以兼容分析代码，display_name和root标明新内容；因此
DynamicReplica_valid这个兼容ID在新训练配置中指向真正的DynamicReplica_train。

未见测试的两层定义
unseen_test_registry.json：主要外测为ETH3D、CARLA，以及新courtyard子集；这些
数据域不在Hero公开任务训练八域内。若自研使用CARLA微调权重，CARLA应称目标域
留出，不称双方都zero-shot；基线一旦也用CARLA微调，同样必须改变报告名称。
heldout_scene_registry.json：同数据集家族的补充场景留出，不是全新数据域。
旧VKITTI2 Scene20和Tartan测试轨迹全部在Hero公开训练名单，排除；MVS-Synth
只剩0117的98组。其余Matrix160、Hypersim194、Blended915、Dynamic300，共1667组。
历史测试文件/结果不删除；Blended官方pair名单缺失，公开扫描名单不是权重训练
日志。DINO/DepthAnything等通用预训练资产曝光无法排除，不作绝对未见承诺。
可执行配置为public_expansion_20261102/evaluation_configs/{ETH3D,CARLA}.json，
使用research/rulermvs_v6_20261101/evaluation.py --config对应文件运行。此时权重
仍是保留的旧V6；未来新训练完成需在文件中明确改weights，不能混用新旧结果。
旧评测配置为历史复现保留，新外测使用本目录清单。

RMVD小规模外测
固定先选courtyard，再下载官方三个素材包和MVSAnywhere官方配组pickle。
rmvd_subset.py运行RMVD官方脚本去畸变，检查生成K与官方一致、投影往返一致；
输出38张深度，保留原图。取官方8组、每组前4/8幅实际图像，不复制图像。
加载RGB使用256x448；真值不下采样，预测双线性放回原尺寸后计分，全部有限正GT
作为分母。固定0.1–100m输入搜索范围、scale=1，不由TEST深度确定范围。
evaluate_rmvd_subset.py固定V6 public best与Hero细化权重，推理后才加载GT。
这是单场景共同输入子集，不是完整RMVD官方协议、官方排行榜成绩或训练后效果。

运行（在jar_reconstruction目录）
本轮使用 official_methods/.venv/bin/python；Hypersim需要h5py，隔离安装在此目录
.deps，不修改已有环境。其它任务不需要新依赖。
任务tmux：expand_hypersim_20261102、expand_tartan_20261102、expand_dynamic_20261102。
查看状态：data软链接的public_expansion_20261102/{hypersim,tartan,dynamic}_status.json。
注意state=running不等于ready，progress.json只计已生成清单的完整有效组。

CPU验证：
official_methods/.venv/bin/python -m pytest -q research/data_expansion_20261102/test_data_expansion.py

GPU接口检查（实际已通过，三类新增TRAIN各一组，无优化器更新）：
official_methods/.venv/bin/python research/data_expansion_20261102/smoke_models.py
结果在gpu_interface_smoke.json，包含V6解码器有限非零梯度和Hero有限输出。

汇总/发布：
official_methods/.venv/bin/python research/data_expansion_20261102/build_registry.py --finalize
该命令只有三个任务全部ready才发布。逐域核对最低组数、TRAIN/VAL序列划分、
共享加载器首中末真实样本、有限输入和标签隔离，并仅以TRAIN统计新范围/尺度。
自动等待器finish_when_ready.py失败即退出，不在失败时伪造READY或自动启动训练。

新训练配方
最终生成的train_v6_expanded.json复用独立V6架构，从保留的正式public best继续
训练；不使用Hero权重。Hypersim和Dynamic有限重放1.5倍、Blended1.6倍，VKITTI2
维持原1.5倍，其余1倍。
新增的场景覆盖替代旧Hypersim8倍/Dynamic4倍重复曝光；该配比待VAL验证，尚无
效果提升证据。新增TRAIN决定范围和归一尺度；TEST不得参与选范围或模型选优。
输入深度范围取旧TRAIN范围与扩容TRAIN分位范围的并集，避免室内样本增多后把
旧远距离支持范围缩小；不因新真实测试的结果改变范围。
弱域依据是已有完整VAL：Hypersim/Blended/Dynamic的V6全图MAE相Hero细化约
高69%/38%/12%；以各域内部相对误差判断，不把米与scene_unit混合排名。
原始证据与固定倍率保存到replay_basis.json；没有按新courtyard测试结果调倍率。
初稿Blended1.2倍会因其它数据扩容而稀释其总份额，因此发布前改为1.6倍。
sampler_audit.json实际构建旧/新八卡采样计划，检查三个弱域的真实曝光份额均增加。
全部达标后才可运行下面命令；本轮没有自动启动：
official_methods/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 research/rulermvs_v6_20261101/training.py --config /data/liuzonghao_data/dataset/public_expansion_20261102/train_v6_expanded.json
权重输出到同一数据盘的future_training，旧V6权重保持。

本轮最终执行结果
Hypersim 5791、TartanAir 12862、DynamicReplica 7200组，七域总TRAIN64198；
VAL7957不变。83623个归档成员CRC、21组公共加载器检查、144次Dynamic几何检查
通过。最终配置下三类新增TRAIN均通过V6损失/反向与Hero推理，无优化器更新。
CPU测试4 passed。正式8卡采样计划每轮84672曝光、2646更新；不是已执行训练。
Hyp/Dynamic/Blended实际份额约为10.28%/12.77%/28.72%，均高于旧配方。
READY.json已生成。全部下载和处理任务完成，剩余空间约248.8GiB。
RMVD courtyard固定子集的实际结果在ETH3D_rmvd_subset/evaluation_results.json：
四视图V6/Hero细化 MAE 6.731918/0.184041米；八视图6.476813/0.170498米。
这是旧权重的新外测，不能称扩容训练效果或完整RMVD官方分数。
