MVSAnywhere 共同数据重训适配（2026-11-03，北京时间）

目的
保留MVSAnywhere论文主模型结构，在本项目扩容后的相同TRAIN/VAL上训练。
不加载mvsanywhere_hero.ckpt或任何Hero任务子树；“Hero”这里只指官方主架构。
主训练循环由本轮父目录统一入口负责，本目录只封装原生模型、损失和优化器。
旧官方预训练baseline、旧V6及其数据评测结果均不改动。

实际复用的官方代码
固定上游提交5bd49bba0992aa56060e1aa3bb40b2150b04726f。
official_methods/methods/mvsanywhere/upstream/src/mvsanywhere 下：
experiment_modules/sr_depth_model.py：DepthModel、compute_losses、推理范围细化。
modules/view_agnostic_feature_volume.py：带几何metadata的视图独立评分/加权。
modules/vit_modules.py：参考ViT、cost-volume patchifier、多层单/多目融合ViT。
modules/depth_anything_blocks.py：四尺度DPT输出头。
modules/networks.py：原生抗混叠ResNet18匹配头，16维匹配描述子。
数据转换复用既有NianticAdapter，包含整数像素中心至官方半像素中心的转换。

通用初始化与随机初始化的边界
1. 参考ViT-B：本地原始DepthAnythingV2 Base的pretrained.*子树，严格加载。
2. DPT公共层：同一DAv2文件中depth_head.*，按官方排除output_conv；
   多尺度深度/天空输出层重新初始化，不把新输出头说成预训练完整深度头。
3. 候选融合ViT-B：官方纯DINOv2-B权重，严格加载，非DA或Hero替代。
4. 匹配stem和layer1：官方antialiased_cnns ResNet18 LPF4 ImageNet权重；
   不是torchvision普通ResNet18权重。其余匹配描述头随机初始化。
5. metadata MLP、cost-volume patchifier、融合投影及输出头随机初始化。

新增资产路径为/data/liuzonghao_data/dataset/public_expansion_20261102/mvs_initialization/。
dinov2_vitb14_pretrain.pth：346378731字节，SHA256
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73。
resnet18_lpf4_finetune-8cc58f59.pth：46910513字节，SHA256
8cc58f59a3015f2a102d5c06dcdcfc7699af6df198fae8ea8bc2a2226a699ec7。
ResNet完整SHA匹配官方文件名前缀；DINO记录本次TLS官方源下载的完整哈希，
不伪称已与独立发布的完整摘要比对。两者均通过严格state_dict装载。

所有请求均显式禁代理并绑定服务器eno5，未使用本地隧道；逐请求证据在
public_expansion_20261102/network.jsonl。DINO最初在独立exec下载完成；
慢速ResNet整包任务中止后转入tmux mvs_assets_20261103，16块下载并合并校验。
下载不会触发模型构造联网，后续训练构造只读上述本地文件。

训练与评测接口
MVSAnywhereTrainingModel(config, initialize=True)
config.initialization需要foundation、dinov2、matching_antialiased三个路径。
initialize=False构造随机结构，随后由公共checkpoint严格恢复全部参数。
forward(inputs, domain, reference=0, targets=targets)：
  train时显式传targets，返回depth、valid、loss、loss_terms、unit_scale=1。
  eval时完全忽略targets，默认用事先固定的TRAIN深度范围；可选原生启发式范围。
  num_views根据实际2/4视图更新，不复制图片。缺相机或任意无效图像区域显式报错。
共同数据加载器必须关闭原先逐样本source dropout，不能将零图当成真实源图。
configure_optimizer(total_steps)：三参数组AdamW，与官方分组相同；
  matching/metadata=1e-4、参考ViT=5e-6、融合ViT/DPT=5e-5，weight_decay=1e-4。
  schedule=official保留源码70000/80000里程碑及70000步线性衰减。
  schedule=budget_scaled按本轮实际预算缩放，末期matching降十倍，
  两个ViT相关组线性衰减至初始的0.001。这是短预算协议适配，不是145k步复现。
DDP需要find_unused_parameters=True，官方本身也如此；保留未使用原生参数。

保留的训练算法与必要适配
直接调用原生compute_losses，含四尺度log深度、逆深度梯度、法线，以及
有真实天空标注时的sky BCE。当前共同接口无sky标签时按官方NaN未知规则跳过，
不把无效深度伪装成天空标签。官方Tartan/SAIL/MVS-Synth可具有sky标注，
因此这项监督覆盖与完整官方采集配置不完全相同，需在论文中披露。
训练范围按参考GT最小/最大值分别乘U[0.5,1]和U[1,2]，沿用官方增强；
eval绝不由targets确定范围。官方图像颜色增强、几何翻转和内部编码翻转保留。
共同输入256x448、实际N2/N4和本地比例/轮数，区别于官方480x640、八图及完整预算。
这些是同数据条件下的重新训练实验，不能称原论文精度已复现。

本轮实际检查
运行（jar_reconstruction目录）：
official_methods/.venv/bin/python research/overnight_v6_v7_mvs_20261103/mvs_baseline/smoke.py
七个真实TRAIN域各B2，Hyp/Tartan显式使用新增样本，Dynamic使用新官方train。
网络bf16、损失及法线显式FP32，五个主要分支梯度均有限且非零；
实际7次optimizer更新仅在烟雾实例。
eval改targets输出逐位不变；包装与原生直接调用输出逐位一致；
临时checkpoint保存/严格重载后输出逐位一致，临时checkpoint随后删除。
证据mvs_initialization/smoke.json；这些单样本损失不是精度或收敛成绩。

official_methods/.venv/bin/python research/overnight_v6_v7_mvs_20261103/mvs_baseline/profile_training.py
GPU6，B8四视图，网络bf16/损失FP32，10次完整训练步含clip=1/AdamW；
预热3步后中位0.257秒，峰值allocated10.92GiB、reserved11.63GiB。
排除IO、DDP与验证，不直接当整夜耗时。
证据mvs_initialization/profile_b8.json。初版脚本名profile.py遮蔽Python标准库，
首次导入失败；更名profile_training.py后真实重跑通过，不把失败写为成功。
早期将法线计算也置于bf16时，虽梯度有限，但部分分支norm达1e5至1e7；
该版本不采用。显式FP32损失后重跑七域norm约68至282，B8十步总norm约36至63。
保留原生数学表达，仅修正混合精度边界；原0.218秒为未修正版，不作为正式吞吐。
正式训练须记录梯度并作配置中的裁剪。

包装目录原命名mvsanywhere会在直接执行父目录train.py时遮蔽官方顶层包；
现已更名mvs_baseline，并模拟该脚本搜索路径验证原生mvsanywhere.modules导入正确。
此更名不改模型参数、损失或计算；root统一入口的DDP测试单独记录。

未在本目录执行长训、完整测试或多卡测试；这些由共同runner实际执行后另行报告。
