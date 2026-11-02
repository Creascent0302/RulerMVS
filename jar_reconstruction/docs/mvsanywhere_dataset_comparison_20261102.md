# MVSAnywhere 与当前项目的数据内容和协议对照

日期：2026-11-02。范围为官方公开源码、配组名单和当前 V6 实际训练配置/JSONL 的静态核查；没有启动训练、下载或新的精度测试。

## 1. 证据与边界

- 官方固定提交：`5bd49bba0992aa56060e1aa3bb40b2150b04726f`，位于 `official_methods/methods/mvsanywhere/upstream`。关键源码从 `git show HEAD:路径` 读取，避免混入本项目兼容补丁。
- 官方 tuple 文件按 Git blob SHA-1 与 HEAD tree 核对一致；部分 blob 不在本地对象库，`git show` 会尝试远程读取并失败，随后直接读取工作区文件并校验 blob，而非把网络失败当文件不存在。
- 当前训练配方：`research/rulermvs_v6_20261101/configs/train_public.json`，不使用早期设计配置代替实际训练配置。
- 逐域计数、场景交集及清单路径：[机器可读核查结果](../research/mvsanywhere_dataset_audit_20261102.json)。交集按标识符计算，未做图像内容哈希或基础预训练资产追溯；公开训练名单不等于发布 checkpoint 的逐样本训练日志。
- 官方材料：[README](https://github.com/nianticlabs/mvsanywhere)、[论文](https://arxiv.org/html/2503.22430v1)、[测试入口](https://github.com/nianticlabs/mvsanywhere/blob/5bd49bba0992aa56060e1aa3bb40b2150b04726f/src/mvsanywhere/test_rmvd.py)。

## 2. 直接多视图训练数据

tuple 是一个参考图及其源图的组合，不等于独立图片、独立场景或 SGD 更新次数。

| 数据 | 官方已发布八图训练 tuple 行数 | 本项目 TRAIN / VAL / TEST | 内容差异 |
|---|---:|---:|---|
| Hypersim | 45,352 | 597 / 199 / 194 | 官方广泛室内覆盖；本地仅12个场景，按8/2/2场景划分 |
| TartanAir | 92,111 | 3,056 / 710 / 848 | 官方多环境、多难度轨迹；本地仅 seasonsforest/Easy 的10条轨迹，按7/1/2划分 |
| BlendedMVG | 97,106（配组文件有疑点，见下） | 15,184 / 1,719 / 915 | 官方要求 BlendedMVS、MVS+、MVS++；本地基础版113场景，95/11/7划分 |
| MatrixCity | 40,105 | 4,185 / 475 / 160 | 官方 big_city 六块航拍及四片街道；本地 small_city 的 street/outside |
| VKITTI2 | 40,158 | 9,500 / 3,390 / 8,370 | 官方五条路线全部天气/视角变体，Camera_0时序八图；本地三条训练、一条验证、一条测试，同步双目 |
| DynamicReplica | 70,093 | 600 / 300 / 300 | 官方 train 多序列左相机时序八图；本地来自原 valid 的四条序列，2/1/1划分，同步左右目 |
| MVS-Synth | 3,219 | 9,476 / 1,164 / 1,183 | 同为 GTAV 960×540来源；本地120序列按96/12/12划分，自建四图组合更密集 |
| SAIL-VOS 3D | 21,088 | 0 / 0 / 0 | 本地没有该训练源 |
| 合计 | 409,232 | 42,598 / 7,957 / 11,970 | 约9.61倍的配组清单量；不是9.61倍独立图片或确定的训练曝光量 |

官方论文将这八套直接训练源描述为合成渲染数据，其中 Blended 使用真实纹理。不能因为其真实 ETH3D 效果好，就推断它直接在 ETH3D 上训练过。前置 Depth Anything/DINO/ImageNet 训练另算，不包含在本表。

**Blended 不能做过度结论：** `blendedmvg_default_train.yaml` 指向 `_pair.txt`，当前公开树没有对应 `train_pair.txt`。公开八图文件包含388个场景键，与声明训练扫描列表495个键不同；我们的95个TRAIN及11个VAL场景全部出现在声明训练列表，却不出现在这份八图文件。7个TEST场景恰好是官方声明的val列表。因此97,106仅是已发布八图文件行数，不能认证为发布Hero的完整实际Blended训练输入，也不能因八图文件零交集就宣布我们的TRAIN/VAL未被Hero见过。

MVS-Synth本地训练tuple比官方多，说明“所有域都是我们训练样本太少”不成立。增加同一序列的配组密度，也不等于增加新场景的多样性。

## 3. 测试场景与官方训练名单交集

本表不代表当前自研训练/测试发生交叉，而是核查“项目TEST对官方预训练baseline是否未见”。参考帧交集包含它在官方训练tuple中作为参考图或源图任一角色出现；不是监督标签曝光的等价证明。

| 本项目 TEST | 官方公开训练名单交集 |
|---|---|
| VKITTI2 | Scene20的10种变体全部在官方训练tuple中；8,370个本地参考帧标识中8,365个出现在官方训练参考/源图集合。官方只读Camera_0，与我们参考相机一致 |
| TartanAir | P010/P011两条测试轨迹均在官方训练tuple；848个参考帧中804个标识出现 |
| MVS-Synth | 12个测试序列中11个出现；1,183个参考帧中565个标识出现 |
| Hypersim | 两个测试场景不在官方训练tuple场景集合；另检查了去除cam编号后的场景级交集，仍为0 |
| MatrixCity | 本地small_city不在官方big_city训练tuple集合 |
| DynamicReplica | 本地测试序列来自原valid，不在官方train tuple集合；不保证背景资产也独立 |
| BlendedMVS | 七个TEST场景属于官方声明val列表；上节配置/配组不一致仍须披露 |

没有发布权重的完整数据日志，以上只可称公开名单曝光证据，不能升级为 checkpoint 逐图训练证明。基础预训练覆盖也未在此审计。VKITTI2占本项目公共TEST组数约69.9%，所以简单按组混合的总平均会受它强烈主导；各域应单列，跨单位不合并米制MAE。

## 4. 官方测试并非我们当前七域测试

官方 `test_rmvd.py` 使用 Robust Multi-View Depth Benchmark：KITTI、ScanNet、ETH3D、DTU、Tanks and Temples；开启refinement，最多7个源图，`alignment=None`。输入高×宽：KITTI480×1280，其余480×640。另有论文中的网格/点云协议，不应与本项目深度MAE混用。

本地公共七域主要是直接训练来源的自定义留出集，另加新CARLA和真实ETH3D双目外测。**VKITTI2不是KITTI。** 项目对同一输入、范围、掩码运行双方模型的比较可以有效，但不是官方RMVD指标复现。

**ETH3D版本尤其不同：** RMVD下载脚本使用courtyard、delivery_area、electro、facade、kicker、meadow、office、pipes、playground、relief、relief_2、terrace、terrains共13个场景的DSLR彩色图、去畸变数据和深度。本地是`two_view_training.7z`及GT，27对低分辨率灰度校正双目，全部用于项目外测；all/nocc是同一批图的两个掩码，不能算54个独立场景。我们的深度评测也不是ETH官方视差榜单。下载脚本证据：https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/scripts/download_eth3d.sh 。

窄基线/弱视差和灰度图会改变匹配问题，不能从当前双目负结果直接推导原论文多视图ETH协议下同样的差距，也不能将两套结果并表排名。

## 5. 同名数据经过训练器之后也不同

1. 官方`GenericMVSDataset`默认8图，首图读GT，其余读图像和相机；源图按位姿距离排序。VKITTI只读Camera_0，DynamicReplica只索引left，不是我们使用的同步双目。时序动态物体与同步双目的几何违背情况不同。
2. 官方`train.py`把八域`ConcatDataset`后shuffle，不能描述成八域等概率。本轮V6使用逐域调度及倍率：Tartan1、Matrix1、Hypersim8、Synth1、VK1.5、Blended1、Dynamic4；倍数增加曝光，不增加新场景。
3. 官方Hero训练配置640×480；当前V6共用RGB256×448。V6的512×896匹配张量在equal_pixels模式来自共同RGB放大，不是更多原始信息。官方每图像素数约2.68倍，但项目同尺寸baseline比较不因此自动失效。
4. 官方训练由参考GT提取范围并随机扩宽；部署走无标签范围估计与可选再预测。当前项目范围/尺度策略不同，须保持测试不从GT指定范围，不能拿带GT范围的内置验证当部署效果。
5. 官方README示例使用ScanNet v2做可选验证，而非八域任务训练成员。本项目用自己的多域VAL挑权重，选择目标也不同。
6. 当前V6为V5/V2基础上的继续训练，不能把本轮轮数当完整学习历史；baseline的任务预训练和公共初始化也需分开登记。
7. Hypersim距离转光轴深度、VK厘米转米等转换已在本项目适配层处理；MVS-Synth、Blended及当前Dynamic契约仍标scene_unit。论文把Dynamic位姿列为metric，这与项目保守单位合同不同，未在本次静态审计中擅自改成米。

## 6. 对研究结论的影响

事实：官方覆盖更广、配组更多、原生视图数与尺寸不同；项目部分TEST与其公开训练输入存在交集；真实外测ETH不是论文同版数据。

合理假设：更广的室内、材质、视角、动态与尺度覆盖能帮助其真实域泛化，而本项目森林/道路的集中训练可能形成针对性优势。清单规模本身不能证明性能差距原因，更不能排除网络候选判断、训练目标、范围适应的作用。官方附录也有训练数据控制对比，不支持将其全部优势归因于数据量。

后续建议分开三种问题：①现有checkpoint在共同输入下的系统性能；②相同数据、初始化类别及预算下的架构能力；③按官方RMVD来源和协议验证跨域泛化。保留当前测试作实际使用比较，但标记公开训练名单重合；若建立新的独立最终测试，应预先固定并避免持续用它调参。当前这些公开TEST已反复参与研究分析，不宜称从未查看过的最终盲测。

本次没有改数据划分、评测程序、模型或训练状态，也没有产生新精度指标。
