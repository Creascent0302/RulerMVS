# 当前数据适配的baseline筛选、复现检查与实验协议

日期：2026-10-31。本轮目的是建立可信的对照实现和当前数据入口，**没有新增完整TEST精度排名，没有复现原论文官方排行榜数字**。V5八卡训练未停止、未修改；本轮网络检查使用CPU，不把有训练负载时的耗时当作速度比较。

## 1. 推荐六种主对照

当前问题是“RGB＋已知K/T的多视图深度”，主要数据提供深度监督，因此先选原生预测深度的网络。不是把单目、未知位姿重建、语义占据数字混在同一表中。

| 方法与官方资料 | 保留的算法 | 为什么值得比较 | 当前权重条件 |
|---|---|---|---|
| [MVSAnywhere，CVPR2025](https://github.com/nianticlabs/mvsanywhere) | Hero的ResNet匹配、metadata、Depth Anything先验、融合ViT、DPT和范围细化 | 当前强跨域基线，检验自研对强先验与几何证据的利用 | 官方Hero完整权重已获；首遍/细化分开报告 |
| [MVSFormer++，ICLR2024](https://github.com/maybeLx/MVSFormerPlusPlus) | DINOv2-B、特征交互FMT、粗候选CVT、四级深度估计 | 检验标尺候选相对强特征与候选注意力的收益和效率 | 官方任务权重仍未获；本轮完整结构测试为随机初始化 |
| [SimpleRecon Hero，ECCV2022](https://github.com/nianticlabs/simplerecon) | 图像先验、ResNet匹配、metadata MLP、2D代价编码与UNet++ | 与“不使用3D代价正则”的几何元数据路线对比 | 原Hero为八目，不能把其固定宽度MLP直接载入四目；需要四目训练权重 |
| [Fast-MVSNet，CVPR2020](https://github.com/svip-lab/FastMVSNet) | 稀疏深度、学习传播、Gauss–Newton细化 | 经典速度/连续定位对照，检验先选面再定位是否有益 | 作者DTU权重已严格加载 |
| [PatchmatchNet，CVPR2021](https://github.com/FangjinhuaWang/PatchmatchNet) | 随机逆深度候选、学习传播、自适应评价、RGB细化 | 直接比较候选搜索与局部传播效率；不是只比较大模型 | 作者指定论文提交及原始DTU权重已接入 |
| [DiffMVS / CasDiffMVS，TPAMI2025](https://github.com/cvg/diffmvs) | 置信度控制的扩散、循环局部候选、凸上采样 | 新增较新的轻量对手，挑战自研的精度/速度取舍 | 官方Drive权重直连超时；两个完整变体已跑通，暂不报精度 |

推荐主表先覆盖前五类中的可训练/有权重版本，并保留DiffMVS作为较新的效率对照。六种均已有独立文件夹、真实网络、原生损失和配置；“结构可运行”不等于“发布权重全部到齐”。

源码提交与本轮独立审计：

| 方法 | 固定commit | 主要证据 |
|---|---|---|
| Fast-MVSNet | `ccb686dda2717613c67d8a289dfe7b2aeb60e2fd` | `official_methods/methods/fast_mvsnet/audit/source_audit_20261031.json` |
| SimpleRecon | `113b825084cc62166c6f5584d526650928127317` | `official_methods/methods/simplerecon/audit_20261031.md` |
| MVSAnywhere | `5bd49bba0992aa56060e1aa3bb40b2150b04726f` | `official_methods/methods/mvsanywhere/audit_20261031.md` |
| MVSFormer++ | `bdd980cfbae9a863b5489e2ed973612b192d11a5` | `official_methods/methods/mvsformerpp/audit/source_audit_20261031.json` |
| PatchmatchNet | `c6d2dce962be8109779afcdb3b79880d35ff9c83` | `official_methods/methods/patchmatchnet/source_manifest.json` |
| DiffMVS | `cd10d5c282a9cabd45a2f64598cd2b990b408d35` | `official_methods/methods/diffmvs/source_manifest.json` |

Fast/Patchmatch为MIT；MVSFormer++/DiffMVS为Apache-2.0；两Niantic方法保留其原许可证，包含非商业限制。各独立目录保存许可证，不因本项目封装而改变。

## 2. 广泛筛选后的补充与排除理由

[MVSAnywhere正文及补充](https://arxiv.org/html/2503.22430v1)本来就区分是否给位姿、是否给逐图范围、是否单目、是否使用GT尺度对齐。不能把其不同分组的论文数字直接搬到我们的米制主表。

- **[Robust MVD，3DV2022](https://github.com/lmb-freiburg/robustmvd)**：下一优先级最高的泛化补充。它本身是已标定多目深度网络，尺度增强与跨域评测贴近本项目。注意“Robust MVD网络”与其支持很多外部方法的“benchmark框架”不是同一件事。本轮未新增实现。
- **[IterMVS，CVPR2022](https://github.com/FangjinhuaWang/IterMVS)**：用GRU维护深度分布信息，适合预算—候选质量对照。官方MIT、有DTU/Blended权重；本轮未新增实现。[Effi-MVS](https://github.com/bdwsq1996/Effi-MVS)也有迭代效率价值，但本轮未确认明确许可证，优先级低于DiffMVS。
- **[CasMVSNet，CVPR2020](https://github.com/alibaba/cascade-stereo)**：本地 `jar/baselines/cas_mvsnet/` 已有作者完整三阶段核心，不是假网络。但只有核心源码，旧适配器还带额外过滤/点云逻辑；现有11MB权重为旧CARLA训练1000步，并非作者DTU权重。配置和README测试路径已过时，本轮只审计、未迁移，不列入已验证主表。将来应比较未过滤 `dense_depth`，不得用置信mask删GT。
- **[FoundationStereo](https://github.com/NVlabs/FoundationStereo) / [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)**：适合另设双目专项，特别是VKITTI2。需要真实标定的水平极线校正，视差转换成深度；CARLA汇聚相机必须先校正。若四目拆成多对，要计入校正、逐对网络、融合时间，不能冒充原生四目模型。本轮未集成。
- **[VGGT](https://github.com/facebookresearch/vggt)、[MASt3R](https://github.com/naver/mast3r)**：适合独立的通用重建或“匹配＋标定三角化”对照。VGGT原生从图像预测相机/几何，未直接使用我们提供的K/T；MASt3R加三角化是额外管线。不能通过GT深度做逐帧尺度拟合后与直接公制输出排名。本轮未新增实现。
- **OctreeOcc、GaussianFormer-2**：保留原目录，暂不计入深度主表。前者公开代码仍有核心模块/初始化缺失，后者有重复稀疏索引导致的重现性问题；而且其任务需要占据/语义标签和对应评价空间。不能只因仓库存在就称已忠实复现全部流程。

## 3. 本轮实际核查及修复

**已有四种没有被替换为自写近似网络。** 本轮重新比对固定官方源码、原生模块和适配输出：

- MVSAnywhere：两个场景各四参考，慢路径144个原生张量、官方快速代价体路径另144个张量逐位一致；密集/天空/稀疏监督的损失与梯度一致。完整201,670,426参数模型CPU前后向通过。
- SimpleRecon：独立原始Git源码与四目适配器80个原生张量逐位一致；密集监督损失/梯度一致。稀疏监督下原版NaN、本地跳过无监督项属于已披露扩展。修复完整checkpoint恢复仍依赖外部初始化权重的问题。完整49,125,157参数CPU前后向通过。
- Fast-MVSNet：主网络/传播/GN/损失语义相同，主要兼容补丁是旧版采样的 `align_corners=True`。两训练阶段、优化、checkpoint、四参考、原生一致性全部复测通过。
- MVSFormer++：105个官方文件103个逐字节相同，其余为已有空监督及batch索引修复；独立原仓库与适配器26个张量逐位一致。补齐官方测试脚本同样执行的旧 `pe_dict` 缓存过滤，真实缺权重仍严格报错。9项CPU测试通过。随机权重有非正深度，valid如实标记，没有截断美化。

**新增两个完整官方方法：**

- PatchmatchNet：221,925参数，作者明确要求的论文版本及原始权重；唯一源码兼容补丁是两处设备索引改为通用device。6项CPU测试覆盖N=2/3/4、全部核心组梯度、损失一致性、像素尺度、checkpoint、标签隔离。统一CLI训练、推理、评测和CPU/Gloo两进程两步训练通过。
- DiffMVS/CasDiffMVS：754,246/925,435参数，全部37个运行相关文件验证Git blob；唯一源码补丁删除导入时未使用的CUDA热身。6项测试覆盖两变体×N=2/3/4，原生前向/损失/反向/重载/缺源剔除。公共CLI训练、无标签推理、两进程CPU/Gloo通过。扩散训练需要GT构造噪声，仅训练态通过独立targets传入，eval显式拒绝targets。

新旧六方法均在**八域各一个真实VAL样例、256×448输入**完成无标签CPU前向，共48组。SimpleRecon在N2和N4分别构造相应参数形状；这不是同一权重跨N。所有结果都只检验程序和数据接口，未计算正式精度。共享数据、单位、目录兼容及Patchmatch测试合计54项通过。

训练公平性细节：Patchmatch原DTU使用毫米，原生SmoothL1转折点1相当于1毫米，米制迁移会变成1米；共同监督范围的裁剪也不同于原DTUmask。当前保留公式并披露，不声称完全一致的物理训练目标。后续同预算训练应预先固定这些选择。原论文的多尺度增强、视图选取、训练预算和重建后处理，也不能由统一CLI自动视为已复现。

## 4. 与当前数据的对应

`prepare.py`复用冻结的已有TRAIN范围和划分，仅根据真实RGB完整性产生六方法共同清单，未用GT值挑输入、调范围、选TEST样本。所有较大的清单、日志、权重均在数据盘软链接 `runs/active/baseline_audit_20261031/`。

| 数据域 | 实际视图 | 几何单位 | 共同train / val / test组数 | 任务属性 |
|---|---:|---|---:|---|
| TartanAir_forest | 4 | 米 | 3056 / 710 / 848 | 静态场景多位置；不是同步四目采集 |
| MatrixCity_street | 4 | 米 | 4185 / 475 / 160 | 静态城市多视角 |
| Hypersim_subset | 4 | 米 | 597 / 199 / 194 | 室内静态几何 |
| MVS_Synth | 4 | scene_unit | 9476 / 1164 / 1183 | 合成城市；不擅自把单位称米 |
| VKITTI2 | 2 | 米 | 9500 / 3390 / 8370 | 同步双目，不能复制成四目 |
| BlendedMVS | 4 | scene_unit | 15184 / 1719 / 915 | 多视角场景单位 |
| DynamicReplica_valid | 2 | scene_unit | 600 / 300 / 300 | 动态双目；本项目对下载的原验证子集内部划分 |
| 新CARLA上参考 | 4 | 米 | 403 / 82 / 174 | 同步四目完整RGB子集，不使用tele |

新CARLA原清单为497/111/181，完整四目过滤分别排除94/29/7组，排除ID完整记录在 `suite/preparation.json`。**此前181帧自研结果不能直接与这里174帧baseline结果混比。** 缺图实验应另开分组；能合法改变N的方法剔除失效源图，不能为不支持的方法补零图。SimpleRecon固定N的MLP需要独立训练版本。

当前统一评测新增显式scene_unit支持：结果字段为 `MAE_scene_unit`、`RMSE_scene_unit` 等，表面阈值必须用 `surface_thresholds_scene_unit` 指定；不把非米制数字写成米。既有米制默认数值和字段保持不变。

## 5. 接下来怎样做公平而有解释力的比较

建议保留两张独立主表：**发布权重跨域迁移**、**共同训练数据及预算后的结果**。自研使用多域训练、别人只用DTU发布权重时，差异不能归因于架构。MVSAnywhere公开预训练含部分本地域/路线，不能都叫双方未见过的零样本测试。

共同锁定：真实图像集合、参考号、256×448像素信息、TRAIN固定搜索/评测范围、光轴Z定义、TEST划分、GT有效掩码。自研native高分辨率匹配与只用共同RGB上采样的equal-pixels分别报告。不得做逐帧GT尺度对齐，也不删除远处、无重叠、低置信或失败预测来压低误差。

报告MAE/RMSE/AbsRel、预测覆盖率、全部GT分母下的阈值命中率；当前MAE/RMSE只对有效预测计算，覆盖率不足时不能单独据此排序。表面指标只对应同一参考可见深度反投影，不冒充全场景完整重建；距离/单位/采样和阈值全部列明。

速度在训练结束、同一张空闲卡上交替测：相同精度、batch1、预热和同步、缓存设置；同时报告网络本体、完整管线、峰值显存。单参考和完整四参考分开，后者必须计入所有调用与后处理。Patchmatch和DiffMVS保留随机数机制，精度实验要登记seed并检查多seed波动。本轮不生成新GPU测速结果。

能检验自研优势的实验，而非提前宣称优势：

| 假设 | 最有针对性的对手 | 应观察的证据 |
|---|---|---|
| 标尺信息提升候选供给 | PatchmatchNet、MVSFormer++、DiffMVS | 同一像素网格、可观测范围内的粗深度误差；可获取原生候选时再比较候选覆盖，不能用最终深度假装候选 |
| 对候选判断更有效 | MVSAnywhere、SimpleRecon | 粗层/细层/最终误差，边界、纹理弱、弱重叠分组；不只看平均MAE |
| 较少计算达到相近精度 | PatchmatchNet、Fast-MVSNet、DiffMVS | 精度—延迟—显存曲线与完整四参考开销；小参数量不自动等于快 |
| 尺度与跨域更稳 | MVSAnywhere，后续Robust MVD | 已知平移缩放的等变测试、域留一验证、近中远分组；场景单位分别统计 |
| 对坏源图更稳 | 支持真实可变N的模型 | 同一帧受控失效/遮挡/噪声前后的退化曲线，参考相机和GT分母固定 |

证据只有在同训练条件对照及对应消融后，才能支持“标尺设计带来收益”的因果结论；当前代码检查还不能证明自研全面胜出。

## 6. 运行入口与尚未完成的项

从 `jar_reconstruction/` 执行：

```bash
# 重建当前数据的完整输入共同子集；不修改原清单。
official_methods/.venv/bin/python research/baseline_audit_20261031/prepare.py

# 某方法的八域真实VAL接口检查：这里允许显式随机结构测试，不作精度排名。
official_methods/.venv/bin/python research/baseline_audit_20261031/real_smoke.py --method patchmatchnet

# 正式配置示例：已有官方Patchmatch权重；可改为其他方法生成的配置。
official_methods/.venv/bin/python -m official_methods.cli evaluate \
  --config runs/active/baseline_audit_20261031/suite/configs/patchmatchnet/CARLA_new_upper.json \
  --split test --device cuda --output runs/active/baseline_audit_20261031/patchmatchnet/full_test
```

最后这条完整TEST命令**未执行**；先与自研锁定相同174帧配置，并等待空闲GPU。训练使用相同配置的 `train` 命令，单卡 `--device cuda`，多卡在前面加 `python -m torch.distributed.run --standalone --nproc_per_node=8`。SimpleRecon不同N需独立训练；MVSFormer++和DiffMVS需填写任务权重，或显式允许从头训练，不能误用随机模型跑正式排名。

尚未完成：新方法RTX5090/NCCL测试、MVSFormer++/DiffMVS官方任务权重下载、六种共同数据完整训练、正式全TEST精度/速度、原论文原数据指标复现。服务器直连OneDrive/Google Drive超时已记录，未借用户本地SSH代理绕过。

各方法详细模块图、原生损失、源码修改和单项命令位于 `official_methods/methods/<method>/README.md`。本轮机器可读证据索引位于数据盘 `baseline_audit_20261031/execution_summary.json`。
