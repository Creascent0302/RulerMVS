# V8：围绕粗候选域外泛化的一级来源调研

日期：2026-11-03。范围：官方论文、官方实现与本地实际执行源码。已读经验文档顶部及246/247/254，并核对223、231–239的相关边界。本文件没有新训练、数据下载、性能试验或模型改动；下述收益均是待验证假设。三个方向都保持V6的编码器次数、96粗候选、四族及细化次数。训练侧方向推理增量为零；逐点归一方向预期低于1ms，但尚未计时。

## 先排除已做过的建议

- 247的新V6已用七域扩容数据，并真正执行50%登记范围、50%参考GT向外扩展（near乘0.5–1，far乘1–2），另有50%内部单位0.5–2倍扰动。不能把再开启这些开关写成V8新条件。
- `jar/pretraining/encoders.py:69` 已在训练时将ResNet全部BatchNorm置于eval；`jar/pretraining/model.py` 的实际观测路径有L2特征归一。简单“冻结BN”和“增加cosine/L2匹配”没有填补新缺口。
- 共同重训的自研增强是每个视图独立亮度gain 0.85–1.15；不是完整ColorJitter，更不是随机焦距/视场/分辨率增强。见`research/overnight_v6_v7_mvs_20261103/common_data.py:23`。
- 254范围内GT全覆盖只能排除边界截断，不能排除候选轴占位、密度或评分分布偏移；246旧权重的粗层错误也不能直接外推成新V6全部场景的唯一原因。
- 231–238的合成几何短训冻结了编码器，有VAL收益而没有可靠TEST升级。V8若再用合成/一致性，需要改变编码可训练性、清洁输入保持或数据覆盖，不能只放大旧短训。

## 优先1：随机化表面在候选轴上的位置，打破深度分布捷径

**直接证据。** RobustMVD将不同域的相关体统计偏移作为问题，采用保持投影不变的三维尺度增强；其表4a在该论文的双目设置中，ETH3D的rel从42.0降至19.4，ScanNet从81.9降至8.5。它不是简单固定区间内随机乘一个系数，而是追踪对数深度直方图，将当前样本中位深度移向已曝光最少的深度箱。论文还将此与普通空间/光度增强区分。[论文§4.2、表4](https://arxiv.org/html/2209.06681v1#S4.SS2)，[官方Scale3DEqualizedBatch](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/batch_transforms.py#L33)。

**与247的关键区别。** 现有内部单位扰动同时缩放位移、候选与范围，候选的归一log坐标不变；因此不能充分打乱网络学到的“哪几个深度箱通常有表面”。现有GT边界外扩只覆盖有限的log边距。应把“单位等变”与“候选占位随机化”作为两个因素分别登记。

**V6可实施方案。** 保留一部分原始登记范围；另一部分对TRAIN样本的log近/远边距作更宽且有界的独立采样，使GT中位数在归一log候选轴的位置覆盖多个区段。候选数不变，所有监督仍须在范围内。若采用RobustMVD式物理缩放，必须同时缩放全部相机平移、GT及所有深度单位量，而且不能随后把搜索范围也等比例搬走、从而抵消占位变化。建议先记录每域训练GT占位直方图、粗赢家和oracle的直方图，验证增强确实改变了目标分布。

**保护与风险。** 太宽范围会浪费96个格并增加重复结构错误；不能把它称为无代价泛化。用TRAIN oracle量化候选离散化代价，约束范围跨度，保留清洁样本，并同时看粗赢家遗憾、过浅误差量及远距RMSE。RobustMVD源码会屏蔽固定范围外标签，本项目不应直接复制这种删监督行为。DDP若用在线直方图，要全局同步或预计算可恢复调度；随机log边距方案更简单。训练额外计算小，推理**0ms**。

**MVSAnywhere交叉支持。** 其训练也随机扩GT范围；官方加载器实际是min乘0.5–1、max乘1–2。论文警告精确GT范围会使网络依赖整个候选区间被占满。该证据支持研究范围分布，但本项目已用同式扰动，不能仅照搬为改进。[官方加载器](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/datasets/generic_mvs_dataset.py#L613)。

## 优先2：保持通用图像特征，充分训练跨外观的匹配分支

**直接证据。** MVSFormer++的附录比较冻结DINOv2、LoRA和全量微调；作者报告全量微调比冻结差，LoRA改进有限，最后选冻结DINOv2。这个结果来自其有限DTU训练，不能当作“任何数据量下冻结必优”的定理。它的官方`vit_forward`用`torch.no_grad()`提取冻结特征，而旁路仍训练。[论文附录A.1及表8](https://arxiv.org/html/2401.11673v1#A1.SS1)，[官方vit_forward](https://github.com/maybeLx/MVSFormerPlusPlus/blob/bdd980cfbae9a863b5489e2ed973612b192d11a5/models/networks/DINOv2_mvsformer_model.py#L55)。

**当前的真实差异。** 247后六轮解冻DA末8块。建议将“全程冻结DA或只最后少量块低学习率训练”作为独立因素，matching CNN/FPN、bridge和候选评分仍训练；这不同于231–238冻结全部编码后只调小头。若从通用预训练开始，可真正保留通用DA特征；若从新V6任务权重续训，冻结只能防止继续漂移，不能声称恢复原通用特征。不能不作适配就静默把V6的DA子树换回原权重。

**具体增强。** MVSAnywhere的`CustomColorJitter(0.2,0.2,0.2,0.2)`在训练step对参考和每个源分别调用，Kornia采用`same_on_batch=False`，涵盖亮度、对比度、饱和度与色调。[step源码](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/experiment_modules/sr_depth_model.py#L761)，[增强实现](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/utils/augmentation_utils.py)。可先采用该量级的视图独立色彩变化，辅以少量源图降采样再恢复、轻模糊；后两项是针对传感器/分辨率偏移的本项目假设，不能写成MVSAnywhere原配方。

RobustMVD采用跨视图共享色彩增强、几何正确resize/crop以及**只遮挡源图**的Eraser；其色彩参数很强，不宜原封不动套到已有预训练特征。源遮挡应按图像面积设定，不能把原384×768上的250–500像素块照搬到256×448。[官方增强参数](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/augmentations.py)，[SpatialAugmentation/ColorJitter/Eraser](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/transforms.py)。所有几何增强必须同步RGB、标签mask和K；主干RGB与matching RGB应来自同一增强后图像，避免引入两套信息。

**保护与风险。** 独立色彩或遮挡过强会教网络忽视多目而依赖单目，恰好可能加重过浅偏置。先保留足够clean样本及至少一个可用源，量化匹配正确深度与错误深度的相关间隔、null质量和粗候选排序。冻结策略和增强应分开消融；更丰富的增强没有增加新场景几何/物体，不能替代新数据。训练图像变换有小开销，冻结DA可减少反传；推理**0ms**。

## 优先3：用对数深度敏感度替代带单位的粗评分信息，并训练等变一致性

**直接证据。** MVSAnywhere显式归一位移与候选深度元数据；原代码的深度项是`(d-min_depth)/max_depth`，并非想当然的`(d-min)/(max-min)`。MVSFormer++的三维位置编码也按当前视锥边界归一。[官方元数据代码](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/modules/view_agnostic_feature_volume.py#L187)，[官方深度元数据](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/modules/view_agnostic_feature_volume.py#L350)，[官方3D位置归一](https://github.com/maybeLx/MVSFormerPlusPlus/blob/bdd980cfbae9a863b5489e2ed973612b192d11a5/models/position_encoding.py#L138)。

**V6实际可定位的问题。** `jar/pretraining/model.py:98`实际返回\(I=\sum_s w_s\|\partial\pi_s/\partial d\|^2\)，`jar/models/ruler_direct.py:342`直接将`log1p(I)`输入粗候选证据。虽然pair内部已有无单位的方向和\(\|g\|h\)足迹，这一额外通道仍随深度单位改变。

**最小改动假设。** 用\(I_{\log d}=d^2I=\sum_s w_s\|\partial\pi_s/\partial\log d\|^2\)替代该通道，保留原方向、足迹和候选数。若单位变换\(d'=ad,t'=at\)，则\(g'=g/a\)，从而\(d'^2I'=d^2I\)。它表征相同比例深度变化引发的图像运动，与局部候选宽度不捆绑；比仅乘候选半宽平方更容易将范围宽度与几何条件分离。这是本项目推导，不是上述论文已验证的模块。

**一致性训练。** 对一小部分TRAIN微批保留同RGB、同物理相机，生成两个内部单位表示；共享编码结果、分别解码，比较恢复物理单位后的粗log深度/候选概率。这里的真实新条件是显式成对约束，而非247的一次随机单位采样。若两次使用不同范围，应把概率映射到共同log深度坐标后再比较，不能直接逐箱KL；不可靠/遮挡区域不能强制跨参考伪标签一致。此方案避免额外推理分支，训练额外开销主要来自第二次decoder。

**保护与风险。** 换统计后旧decoder输入分布会变，不能期待直接换公式即可免训提升；建议独立V8支路和同预算适配。峰分离常量、细搜索上下限、纯旋转判定中的`g²>1e-10`及误差项仍可能依赖单位，未同步处理前只可称“单位不变证据通道”，不能称全网严格尺度等变。MVSAnywhere附录承认未归一元数据在部分米制基准可更优，归一主要保护任意尺度，不能保证ETH/Tanks增益。[论文附录D](https://arxiv.org/html/2503.22430v1#A4)。新增逐点乘法、无新参数，预期**<1ms，未实测**；速度须按254同卡交替协议确认。

## 取舍与验收建议

先做训练条件清楚的最小对照：V6同预算续训；仅候选占位增强；仅外观增强；仅DA冻结；仅无单位信息。通过后才组合，不用组合收益冒充每个因素因果。初筛只用TRAIN/既有VAL，固定清单记录粗候选oracle、正确峰保留率、最终MAE/RMSE/AbsRel及过浅误差量；已有ETH/Tanks只是最终固定外测，不能用来调强度。跨域匹配统计建议看GT最近候选对hard negatives的margin与候选轴位置，单看最终MAE不能识别训练捷径。

本轮不推荐继续增加全局ViT或候选重融合层；V7已验证有场景收益而不满足总体目标。也不建议以额外测试时迭代、TTA或第二模型投票满足精度目标，它们不符合推理基本不变的约束。尚未验证“哪一种足以大幅缩小V6与同预算MVSAnywhere的真实域差距”。

检索说明：research-lookup技能用于限定一级来源与证据边界；本机没有parallel-cli，实际使用内置网页工具阅读官方HTML/源码，未启动外部付费研究任务。官方PDF体积导致读取失败时采用对应arXiv全文HTML；没有声称阅读全文PDF。RobustMVD的master链接对应本次检索可见源码，MVSAnywhere/MVSFormer++链接固定到本地已审计提交。
