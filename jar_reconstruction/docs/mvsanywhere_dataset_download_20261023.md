# MVSAnywhere 训练数据下载与后处理教程

原始核查日期：2026 年 10 月 23 日。适用项目：`/home/liuzonghao/3Dscene/jar_reconstruction`。2026 年 10 月 24 日已完成下表七类数据的指定范围下载、转换与真实模型接入检查，数据盘剩余约415GiB，超过用户要求保留的300GiB。没有启动正式训练。机器可读的路径、数量与空间快照见[数据目录](../configs/public_datasets/catalog.json)。

## 当前已经可以使用的数据

全部物理目录都在 `/data/liuzonghao_data/dataset/`，与 `CARLA_data` 同级；项目 `/home/liuzonghao/3Dscene/data/` 下提供同名软链接。表中空间为实际磁盘块占用的近似值，包含原包、转换数据及验证记录。

| 目录 | 已下载范围 | 视图与单位 | train / val / test组数 | 占用 |
|---|---|---|---:|---:|
| `TartanAir_forest` | 10条森林轨迹，4629帧 | 静态多位置4图，米 | 3056 / 710 / 848 | 15.4GiB |
| `MatrixCity_street` | small_city室外街景，6215帧 | 静态多位置4图，米 | 4250 / 475 / 160 | 41.8GiB |
| `Hypersim_subset` | 12完整场景，1003帧 | 静态多位置4图，米 | 597 / 199 / 194 | 37.8GiB |
| `MVS_Synth` | GTAV_540完整120场景，12000帧 | 静态多位置4图，scene_unit | 9476 / 1164 / 1183 | 42.9GiB |
| `VKITTI2` | 2.0.3完整RGB、depth、classSeg、textgt，42520图 | 真实同步2图，米 | 9500 / 3390 / 8370 | 91.7GiB |
| `BlendedMVS` | v1.0.0基础版113场景，17819帧 | 静态多位置4图，scene_unit | 15184 / 1719 / 915 | 96.2GiB |
| `DynamicReplica_valid` | 4条完整原validation序列，2400图 | 真实同步2图，scene_unit | — / 1200 / — | 33.2GiB |

`scene_unit`表示原始深度与相机平移使用一致数值单位，但尚无足够依据把它称为米。DynamicReplica的论文描述了物理基线，文件数值也与之相符；仍保留保守单位声明，直到原始数值的米制定义被直接核实。禁止随意乘比例制造米制真值。森林、城市、Hypersim、MVS-Synth、Blended四图来自不同拍摄位置，不是同步物理四相机。VKITTI另保存17976组原生跨时间四图tuple，单独标明动态与非同步，没有将其混入同步双目清单。

MatrixCity可训练参考标签清单为 `manifests/train_labeled.jsonl`（4185组）；全train保留4250组，包括65个原始参考深度全部无效的样本。DynamicReplica保留原validation身份，训练/测试路径为空；如后续要在这个子集做开发训练，必须另行按完整序列划分并明确它不是官方train。Blended测试使用官方7个validation场景，新的val从官方训练场景整场景留出；这不排除baseline的预训练暴露。

**空间取舍：** 没有继续下载BlendedMVS+、++（分别约87.49/85.91GB压缩，处理后更大），也没有下载TB级全量Hypersim/TartanAir/MatrixCity/DynamicReplica。SAIL-VOS 3D需要向作者申请原始深度，且MVSAnywhere还要求额外`rage_matrices`；当前未取得完整公开输入，明确未下载。旧的Hypersim单场景样例仍保留。

### 下载链路和完整性

大文件下载全部在专用tmux socket `dataset-direct` 中执行：清除大小写代理变量，`curl -q --proxy '' --noproxy '*' --interface eno5`，忽略个人curl配置并绑定服务器网卡。实际下载记录的本地地址均为`172.16.2.4`，没有连接本地电脑的SSH代理`127.0.0.1:17898`，也没有修改交互代理或SSH。Hypersim来自魔搭；其余部分下载会访问境外CDN，但使用的是服务器网络。

Blended官方GitHub下载较慢，使用明确记录的公开转发源`gh-proxy.com`加速。完整末卷SHA256与官方API下载逐字节等价，两段Range也一致；部分节点不支持续传，遇curl33时从零重下该未完成分卷，不拼接HTTP200正文。16分卷合并后71842成员CRC全部通过，原分卷保留，冗余合并ZIP已删除。其余数据校验各来源的SHA256；VKITTI另核对官网MD5，DynamicReplica另核对作者公布SHA。各目录`download_evidence.json`保存URL、实际IP、字节数与哈希。

下载和转换工具新增300GiB剩余空间检查；不足则停止并保留已有文件，不自动删除其他数据。查看会话：

```bash
tmux -L dataset-direct list-sessions
df -h /data
```

### 接入方法与实际验证

配置均在 `configs/public_datasets/`：每套有 `*_own.json`、`*_mvsanywhere.json`，以及下载、转换与烟测配置。参数直接改JSON。自研公共入口使用`data.protocol=calibrated_multiview`，保留真实相机数；baseline通过`data.num_views`显式选择2或4。旧CARLA四目默认没有放宽。公共图像主图256×448、自研匹配图512×896都从原始RGB分别缩放，K同步更新；图像无效区域读取与缩小策略已统一。只有RGB和标定进入预测。

自研、MVSAnywhere已实际通过新增五套数据的GPU前向，移除targets后预测完全相同。Hypersim、MVS-Synth、Blended另外通过Fast-MVSNet和SimpleRecon前向。对四套有train的数据各执行了自研和Hero的一步训练，临时更新丢弃；DynamicReplica没有拟合原validation。53项相关CPU测试及独立单位化深度指标自检通过。MVSFormer++未配置完整任务权重而未运行；占据模型尚缺同定义的体积可见性/占据标签，不能把深度或前景mask当成完整占据真值。

Blended初次训练烟测因固定首样本在旧2–80范围没有GT而明确失败，原记录位于`smoke/train`。仅将监督范围显式改为0.001–10原场景单位后，两模型反向与更新通过，记录位于`smoke/train_explicit_supervision`。这没有改变自研旧2–80候选格，**不表示新域范围适配或精度已解决**。当前`*_own.json`是同架构接入模板；正式训练须单独制定候选范围、单位尺度和损失归一化，不能直接把CARLA参数套到所有域。验证集、测试集不能用于选取这些参数。

自研正式推理入口示例（已实际执行，结果存数据盘）：

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
tools/with_server_network.sh official_methods/.venv/bin/python infer.py \
  --config configs/public_datasets/vkitti2_own.json \
  --checkpoint runs/active/multiview_20261024/full_rgb_final/scale_transport_complete.pt \
  --device cuda:0 --split test --limit 1 \
  --output /data/liuzonghao_data/dataset/VKITTI2/interface_infer/own
```

MVSAnywhere正式入口也已在MVS-Synth执行：

```bash
tools/with_server_network.sh official_methods/.venv/bin/python -m official_methods.cli infer \
  --config configs/public_datasets/mvssynth_mvsanywhere.json \
  --checkpoint runs/mvsanywhere_audit_20261013/finetune_verified/last.pt \
  --device cuda:0 --split test --max-batches 1 \
  --output /data/liuzonghao_data/dataset/MVS_Synth/interface_infer/mvsanywhere
```

非米制数据在原CARLA米制evaluate入口会明确拒绝。改用独立工具`tools/evaluate_public_depth.py`，输出MAE/RMSE及明确`geometry_units`、AbsRel、delta、覆盖率；参考图/范围/样本数由配置控制。低分辨率输出有depth_K时按真实内参映射，没有时要求显式格点约定，不混淆网格。该工具已在MVS-Synth单样本运行两模型自检，不是完整测试集排名；已有结果拒绝覆盖，需要修改`evaluation_output`。完整正式精度、速度、泛化训练均未运行。

```bash
official_methods/.venv/bin/python tools/evaluate_public_depth.py --self-test
# 先复制并修改JSON里的evaluation_output、evaluation.limit与显式depth_range：
official_methods/.venv/bin/python tools/evaluate_public_depth.py \
  --config configs/public_datasets/mvssynth_depth_eval_check.json
```

### 特殊几何修正和审计位置

Hypersim射线距离转光轴深度，场景单位换算只作用相机平移，原depth_meters不重复缩放。`ai_011_004/cam_01/frame0`原基底为2.5I，首次严格旋转检查失败；后续证明正均匀缩放不改变针孔射线方向，且独立世界位置标签验证约毫米差，于是明确除去共同尺度并记录，未做任意SVD修正。全1003帧深度转换独立核对最大差5.43e-6米。[Hypersim审计](../../data/Hypersim_subset/audit.json)

MVS-Synth采用作者960×540原K与整数中心，不照抄MVSAnywhere读取器的810宽假设；反射坐标统一转为合法右手rig。Blended保留PFM翻转与原K；半像素诊断差异很小，不能说几何检查唯一确定了亚像素约定。DynamicReplica半精度PNG按位解码，不按厘米或普通16位整数读；相机与作者PyTorch3D转换逐项一致。VKITTI保存厘米uint16深度，加载乘0.01，65535饱和无效。[MVS-Synth审计](../../data/MVS_Synth/validation.json)、[Blended审计](../../data/BlendedMVS/processed/interface_audit.json)、[Dynamic审计](../../data/DynamicReplica_valid/prepared/validation.json)、[VKITTI审计](../../data/VKITTI2/logs/public_loader_check.json)

下文保留原始来源调研和之前的小规模下载记录，不能把旧记录的范围当成上述最终目录的现状。

MVSAnywhere 的直接训练数据为八套：Hypersim、TartanAir、BlendedMVG、MatrixCity、Virtual KITTI 2、Dynamic Replica、MVS-Synth、SAIL-VOS 3D。ScanNet v2 在官方训练示例中作为验证数据。模型还继承 Depth Anything V2、DINOv2 与 ResNet 的预训练知识；下载八套数据不等于重新构建这些前置预训练过程。[论文表 1 与 Training data](https://arxiv.org/html/2503.22430v1)、[官方训练说明](https://github.com/nianticlabs/mvsanywhere#-training)。

**国内下载结论：Hypersim 找到了可列出完整原始包的魔搭镜像；MatrixCity 有作者提供的百度网盘和 OpenXLab。其他数据提供官方来源及条件镜像，但不能保证八套数据都能全程经国内服务器取得。** 本次 HF-Mirror 的若干大文件 HEAD 请求先跳转到 Hugging Face，再跳到 `us.aws.cdn.hf.co`；把网址改成镜像域名并不保证避免跨境。SAIL-VOS 3D 需要申请，现有公开说明也未包含当前加载器要求的全部附加文件。

| 直接训练集 | 应下载的版本和内容 | 国内资源状态 |
|---|---|---|
| Hypersim | 原始场景 ZIP，包含 RGB、距离、位姿及场景元数据 | 魔搭 OpenDataLab/Hypersim，样例最终为国内 CDN |
| TartanAir | V1，Easy/Hard，左目 RGB、depth、seg 和 pose | 官方 HF 仓库可经 HF-Mirror 尝试；本次链路跨境 |
| BlendedMVG | BlendedMVS、BlendedMVS+、BlendedMVS++ 三个低分辨率集合 | 未确认覆盖全部三集合的国内原始镜像；官方 Release 可下载 |
| MatrixCity | big_city、big_city_depth、big_city_depth_float32 | 作者百度网盘、OpenXLab；也提供 HF |
| Virtual KITTI 2 | 2.0.3，rgb、depth、classSegmentation、textgt | 有社区 HF 原始归档镜像；国内直连未确认 |
| Dynamic Replica | 原生 train；RGB、depth 和 frame annotations | 有社区 HF 包，但未确认完整兼容的国内训练镜像 |
| MVS-Synth | GTAV_540.tar.gz，960×540 | 作者 HF 仓库；HF-Mirror 本次重定向境外 |
| SAIL-VOS 3D | RGB、原始深度、camera、额外投影矩阵 | 需要向作者申请；未确认授权的国内公开镜像 |

## 1 准备磁盘和工具

### 已完成的最小可用样例下载

2026 年 10 月 24 日，在独立 tmux `dataset-direct / hypersim-sample` 下载魔搭镜像的 **Hypersim `ai_011_010` 完整场景**。选择它是因为包小且使用标准针孔投影；更小的 `ai_015_007` 带倾斜相机投影，需要额外适配。该场景含 9 张原始 1024×768 RGB、9 张原始米制距离及相机位姿，不是整个 Hypersim 数据集。

- 包大小 112,248,512 字节；实测传输 4.357557 秒，约 25.8 MB/s。
- curl 同时禁用代理、忽略用户 curl 配置、绑定 `eno5`。实际服务器 `172.16.2.4` 直接连接 CDN `58.205.209.139`，没有经过 `127.0.0.1:17898`。原交互代理与原 tmux 会话未修改。
- SHA256 `31d8a62a155bb2ea48923fbf1b3945394b764191dbc22df542d71a215f81ca75` 与镜像清单相同，ZIP CRC 校验通过。
- 物理目录 `/data/liuzonghao_data/dataset/Hypersim_sample`，与 CARLA_data 同级；项目软链接 [`../../data/Hypersim_sample`](../../data/Hypersim_sample)。原包、解压数据、转换深度、日志、预览及依赖共约 261 MiB，均在数据盘。
- [下载证据](../../data/Hypersim_sample/download_evidence.json)、[下载日志](../../data/Hypersim_sample/logs/download.log)、[数据与几何检查](../../data/Hypersim_sample/sample_summary.json)、[两模型推理检查](../../data/Hypersim_sample/smoke/summary.json)、[预览](../../data/Hypersim_sample/smoke/preview.jpg)。

已将 9 帧射线距离转为独立 float32 光轴深度。仅根据标定选取真实帧 82、89、81、83，参考槽位 2 对应帧 81，三源基线约 0.79–1.35 米；生成 `manifests/smoke.jsonl`，复用公共数据加载器。样本标注 `static_multiview_tuple` 和 `synchronized_capture=false`，各槽位保存原始 `source_frame_id`。同一静态场景的不同位置图像可以做多视角重建，但不能称为 CARLA 式同步四相机。

自研 `scale_transport` 与 MVSAnywhere 首遍使用现有本地权重，GPU 实测输出均为有限正深度 `256×448`；不加载 targets 时输入与预测逐元素不变。这里只验证程序和数据接入。场景在 MVSAnywhere 官方训练清单中，且现自研沿用 CARLA 的 2–80 米搜索范围，不能据此报告未见场景泛化成绩或公平精度排名；预览中自研的明显室内深度偏差仍需后续单独研究。其他 baseline 未在本次逐一运行；同一深度 manifest 可复用其公共接口，占据方法还需要独立构造可见性和占据标签。

复查已完成的 tmux 任务（保留退出后的窗口）：

```bash
tmux -L dataset-direct attach-session -t hypersim-sample
```

复跑数据转换与真实输入烟测，不访问网络、不重新下载权重：

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
tools/with_server_network.sh bash --noprofile --norc \
  /data/liuzonghao_data/dataset/Hypersim_sample/tools/validate_sample.sh
```

对应独立工具为 [直连下载](../tools/download_hypersim_sample.sh)、[数据转换](../tools/prepare_hypersim_sample.py)、[真实输入烟测](../tools/smoke_public_multiview.py)。没有改模型、原 CARLA 配置或训练流程。

### 已完成的两套室外子集（2026-10-24）

优先使用 **TartanAir 森林子集**研究野外匹配；MatrixCity 补充道路、建筑和较远距离。来源分别为作者的 [TartanAir V1 仓库](https://huggingface.co/datasets/theairlabcmu/tartanair) 和 [MatrixCity 仓库](https://huggingface.co/datasets/BoDai/MatrixCity)。下载的是以下完整选定子集，不是两个仓库的全量数据。

| 数据 | 实际内容 | 传输大小（十进制） | 四视图组 train / val / test | 当前磁盘占用 |
|---|---|---:|---:|---:|
| TartanAir V1 | `seasonsforest/Easy`，10条完整轨迹；4629张RGB、深度、分割及位姿 | 5.053 GB | 3056 / 710 / 848 | 约16 GiB |
| MatrixCity | `small_city/street/outside`，官方train与test；6215张RGB、深度及位姿 | 9.999 GB | 4250 / 475 / 160 | 约42 GiB |

数据与 CARLA 同级存放，原包、原始图像、转换深度、模型预测和日志都在数据盘：

```text
/data/liuzonghao_data/dataset/TartanAir_forest
/data/liuzonghao_data/dataset/MatrixCity_street

/home/liuzonghao/3Dscene/data/TartanAir_forest -> 上述数据盘目录
/home/liuzonghao/3Dscene/data/MatrixCity_street -> 上述数据盘目录
```

**网络与完整性已实际检查。** 两项下载均在独立 tmux 服务器 `dataset-direct` 中完成，curl 忽略个人配置、显式禁用代理、绑定服务器 `eno5`；每个下载的本地地址都是 `172.16.2.4`。下载大包最终直连 CDN `13.33.183.70`，没有通过本机回环地址 `127.0.0.1:17898` 的 SSH 代理。这里的“服务器直连”不等于全程国内 CDN。没有修改交互代理、SSH 或原 tmux 会话。

七个大归档的 SHA256 均与作者仓库清单匹配，ZIP 完整解压并通过 CRC，TAR 完整解压。三个 MatrixCity 相机 JSON 又逐一核对了官方 Git blob 哈希，全部相同；通过此核对后将其 SHA256 固定到下载配置，防止后续源文件变化。证据：[森林下载](../../data/TartanAir_forest/download_evidence.json)、[城市下载](../../data/MatrixCity_street/download_evidence.json)、[相机JSON核验](../../data/MatrixCity_street/metadata_integrity.json)。

**已执行的后处理：**

- 两套数据的全部10844个原始RGB/深度/位姿记录均检查了帧对应、完整图像解码、深度尺寸与格式、外参旋转；并非只读文件头。原始文件没有被覆盖。
- TartanAir 保留原始 float32 米制光轴深度，转换 NED 位姿到 CV 相机和项目 rig 坐标；统一整数像素中心内参为 `fx=fy=320, cx=319.5, cy=239.5`，与原加载器半整数像素射线一致。非有限、非正及超过1000米的深度保留无效掩码，不当作可见空闲空间。原生分割保留，尚未擅自映射为共同语义类别。
- MatrixCity 的 EXR 光轴深度从厘米除以100，保存 float32 米制 NPY；天空等饱和值标为无效。只恢复相机旋转矩阵中的缩放，平移保持米制；转换 OpenGL/CV 约定并同步像素中心。1000×1000图像不离线裁切。
- 四视图组仅依靠位姿、基线和固定深度探针的视锥重叠构造，不用真实深度挑容易样本。四个视图均为不同真实图像、不同相机中心。森林要求任意两相机至少相距0.1米，排除了15个无法满足条件的参考帧；此前仅约束参考—源会选入几乎重合的源，本次已修正。
- 森林按整条轨迹划分：train为P001/P002/P003/P004/P005/P007/P008，val为P009，test为P010/P011。城市在官方train的4725个允许帧中按空间区域分出4250/475；另外945个被官方过滤的原始帧保留但不参与组装。官方test的545帧中160个参考可以组成四视图，385个不足三源，原图仍全部保留。
- 全量检查三份清单的原图路径互不交叉。MatrixCity不同官方split可能重复数字帧号，标识必须包含目录/split，不能只比较裸帧号。

最终清单和完整审计：

| 数据 | 根目录下的清单 | 全量审计 |
|---|---|---|
| TartanAir | `prepared/manifests/{train,val,test}.jsonl` | [转换汇总](../../data/TartanAir_forest/prepared/summary.json)、[独立复核](../../data/TartanAir_forest/delivery_audit.json) |
| MatrixCity | `manifests/{train,val,test}.jsonl` | [转换汇总](../../data/MatrixCity_street/processed/preparation_summary.json)、[独立复核](../../data/MatrixCity_street/delivery_audit.json) |

城市另提供 `manifests/train_labeled.jsonl`，含4185组，推荐给要求参考图有深度监督的训练入口。它仅从完整4250组train中排除65个参考原始深度完全无有效像素的组；不重选源视角、不更改原train/val/test清单，也不按测试标签筛选样本。[排除帧与规则审计](../../data/MatrixCity_street/processed/train/labeled_manifest_audit.json)已保存。这里“有标签”不保证每种模型自定范围内都有标签，原生有效掩码仍须保留。

**接入现有方法时的配置：** 把对应配置中的 `data.root`、`train_manifest`、`val_manifest`、`test_manifest` 指向上表。清单相机ID统一为 `view_0, view_1, view_2, view_3`；若自研配置使用 `camera` 分组，把 `camera.wide_ids` 同样设为这四项，`mode` 为 `wide4`，不要沿用 CARLA 的 `stereo_up` 等名称。参考槽位仍是索引2，但这里表示选定的参考图，不表示物理上相机。实际 K/T 随每组读取。自研正式训练加载器、基础加载器和 baseline 加载器已在两套数据每个split首/中/尾样本比对，RGB、K/T、标签和匹配RGB一致。

**已运行模型检查：** 两套数据的train/test各固定首/中/尾三组，共12个四视图输入，每个输入运行当前自研 `scale_transport`、MVSAnywhere Hero 首遍、Fast-MVSNet、SimpleRecon Hero，48个模型—样本组合全部产生有限正深度；移除targets后重复预测差为0。自研与MVSAnywhere分别在两套train上执行了一次真实损失、反向传播和优化更新，损失与梯度有限；临时更新随进程丢弃，原权重未改。Fast-MVSNet和SimpleRecon本轮只测推理。MVSFormer++未配置完整本地任务权重，本轮没有运行；占据基线也未验证，深度数据不能直接冒充完整占据或语义占据真值。

Fast-MVSNet 首次重复前向出现最大0.673米差异，已追到原生三维转置卷积的约`7.15e-7`特征扰动被GN细化放大。仅在该模型的检查中开启 cuDNN 确定性执行，所有阶段连续三次重复差均0；网络与原生GN没有修改。首次失败日志仍保留。下载脚本最初还遇到系统Python3.8不支持`Path.is_relative_to`，已改为兼容路径检查并恢复完成，不把首次失败记为通过。

结果与预览：[森林train](../../data/TartanAir_forest/smoke/train/summary.json)、[森林test](../../data/TartanAir_forest/smoke/test/summary.json)、[森林预览](../../data/TartanAir_forest/smoke/test/preview_000424.jpg)、[城市train](../../data/MatrixCity_street/smoke/train/summary.json)、[城市test](../../data/MatrixCity_street/smoke/test/summary.json)、[城市预览](../../data/MatrixCity_street/smoke/test/preview_000080.jpg)。灰度深度预览统一显示0–80米，黑色GT包含无效区，不表示0米真实表面。

复跑命令如下，均从项目目录执行；配置中的设备和路径可以直接修改：

```bash
cd /home/liuzonghao/3Dscene/jar_reconstruction
# 数据转换：读取已下载文件，不联网。
official_methods/.venv/bin/python tools/prepare_tartanair.py \
  --config configs/public_datasets/tartanair_forest_prepare.json
official_methods/.venv/bin/python tools/prepare_matrixcity.py \
  --config configs/public_datasets/matrixcity_street_prepare.json

# 真实权重检查：工具禁止Python IP连接，缺权重时直接报错。
official_methods/.venv/bin/python tools/smoke_outdoor_datasets.py \
  --config configs/public_datasets/tartanair_forest_smoke_test.json
official_methods/.venv/bin/python tools/smoke_outdoor_datasets.py \
  --config configs/public_datasets/matrixcity_street_smoke_test.json
# 对应 *_smoke_train.json 另含自研/Hero各一次训练检查。

# 查看已完成窗口，Ctrl+B、D退出查看，不结束tmux。
tmux -L dataset-direct attach-session -t tartanair-forest
tmux -L dataset-direct attach-session -t matrixcity-street
```

如需重新检查下载，用以下命令新建tmux会话；已有正确文件会校验后复用，直连失败不会切回本地代理。`download_public_dataset.py` 本身要求处于tmux中，并强制关闭代理、绑定网卡：

```bash
tmux -L dataset-direct new-session -s forest-recheck \
  '/home/liuzonghao/3Dscene/jar_reconstruction/tools/with_server_network.sh /usr/bin/python3 /home/liuzonghao/3Dscene/jar_reconstruction/tools/download_public_dataset.py --config /home/liuzonghao/3Dscene/jar_reconstruction/configs/public_datasets/tartanair_forest_download.json'
# 城市使用同目录 matrixcity_street_download.json 和不同session名。
```

**使用边界：** 两套合成场景均按静态多视图假设组织，清单明确 `synchronized_capture=false`、保存原始 `source_frame_id`；这不是对风动植被、粒子等内容绝对静止的保证。适合公共多视图预训练和室外接入研究，不能称为同步物理四目，也不能与CARLA协议混算一张排行榜。TartanAir是MVSAnywhere直接预训练来源；MatrixCity此次用small_city，官方直接训练说明用big_city，但其他预训练暴露未完全排除，不能笼统宣称模型未见这类数据。

MatrixCity官方test的参考—源距离中位数约55.8米，与train/val的约5米差别很大。事后重投影抽检中，train/val容差内比例约90.9%/90.8%，test仅约46.8%（包含遮挡，容差为`max(0.1米,1%源深度)`）；不能把test全部称为高质量近基线匹配。可将其单独作为大基线压力测试，正式比较另固定统一协议。城市train还有65个参考没有有效深度，保留图像输入和空标签掩码，不能将其计为零误差。无效标签与空间unknown不等于free。

本次自研实际匹配输入为512×896，Hero包装器读取256×448主图，Fast/SimpleRecon保留原生128×224深度输出；这里验证各自管线，不提供公平精度/速度结论。未来训练、统一范围的完整评测、真实户外泛化与占据标签适配仍需另做。

### 服务器直连下载与 SSH 交互分开

2026年10月24日实测可行。当前`.bashrc`将HTTP/HTTPS/SOCKS代理指向`127.0.0.1:17898`，通过SSH RemoteForward使用本地电脑；服务器自身另有`eno5`网卡和默认出口。只清除下载进程的代理即可同时保留当前交互代理，无需改`.bashrc`、SSH隧道或路由。

项目提供[仅作用于子进程的直连助手](../tools/with_server_network.sh)。实际对照为：现有代理读取小README连接127.0.0.1；直连同文件从服务器172.16.2.4连接魔搭公网地址。独立tmux内已成功直连Hypersim国内CDN并读取4096字节，HTTP206且ZIP文件头正确；探针会话已自动退出，原tmux会话及交互代理保持原状。[实际核查结果](download_assets/mvsanywhere_20261023/network_probe_20261024.json)。

后续真正下载时，使用独立tmux服务器和不读启动配置的新shell：

```bash
tmux -L dataset-direct new-session -s datasets \
  '/home/liuzonghao/3Dscene/jar_reconstruction/tools/with_server_network.sh bash --noprofile --norc'
```

这里的助手会清除大小写HTTP/HTTPS/ALL/FTP代理，并设置`NO_PROXY=*`与`no_proxy=*`。`--noprofile --norc`防止新shell再次从`.bashrc`恢复代理。按`Ctrl+B`后再按`D`离开，下载继续；重新查看用：

```bash
tmux -L dataset-direct attach-session -t datasets
```

进入上述shell后再运行本教程的下载命令；aria2增加`--no-conf`以忽略可能存在的用户代理配置。普通tmux、新窗口或普通SSH终端可能继承原代理，不能仅凭“已经用了tmux”认定走服务器网络。新开下载窗口也应使用同一助手。助手不会自动回退到本地代理；若直连某个源失败，应改用其他源或等待网络处理。

前次探针中魔搭与清华源直连成功；HF-Mirror相关文件HEAD最终到`cas-bridge.xethub.hf.co`并返回200，但未读取该大文件正文；GitHub直连5秒连接超时，不能保证该源可用。服务器直连境外CDN仍使用服务器出口，不等于消耗本地电脑代理流量。前次只运行小文件/4KB探针；后续 112 MB 完整场景下载结果见上节，尚未测试 TB 级持续传输和断点续传。

下面是待执行的命令。归档、解压后的数据、下载缓存都放数据盘；教程本身只在项目中保存小文件。

```bash
export PROJECT=/home/liuzonghao/3Dscene/jar_reconstruction
export D=/data/liuzonghao_data/dataset/MVSAnywhere_pretrain
export R="$PROJECT/official_methods/upstream/mvsanywhere"

mkdir -p "$D/archives" "$D/configs" "$D/cache" "$D/tools"
# 仅在目标尚不存在时建立软链接；不要覆盖已有路径。
ln -s "$D" /home/liuzonghao/3Dscene/data/MVSAnywhere_pretrain
```

使用独立的 Python 3.10 或 3.11 下载环境，避免改动已有训练环境。以下示例使用项目现有 Python 创建环境：

```bash
"$PROJECT/official_methods/.venv/bin/python" -m venv "$D/tools/download_env"
source "$D/tools/download_env/bin/activate"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'huggingface_hub==0.35.3' numpy h5py
```

同时需要 `curl`、`unzip`、`tar`；Hypersim 批量下载建议使用 `aria2c`。这是数据下载环境，运行官方训练仍使用相应方法环境。

```bash
export HF_HOME="$D/cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

本教程固定 0.35.3，使用 `hf download --repo-type dataset` 语法；不要混用其他版本的 `--type`。CLI 会保留断点缓存，失败后重新运行原命令。`HF_HUB_DISABLE_XET=1` 只切换传输实现，不保证数据位于国内。[HF-Mirror 使用说明](https://hf-mirror.com/)。

本机现有 `hf` 下载一个小 README 的实际检查失败于元数据获取，报 `FileMetadataError/LocalEntryNotFoundError`，因此后文 HF 命令没有标成“本机已下载通过”。可先尝试一个小文件；如果失败，检查最终下载域名，再使用 `curl -fL -C -` 的直链方式。大文件未完整下载，持续速度也未实测。

```bash
curl -IL --max-time 30 \
  https://hf-mirror.com/datasets/phuang17/MVS-Synth/resolve/main/GTAV_540.tar.gz
```

若 Location 或最终地址指向境外，且你的网络严格禁止跨境，这条路线不能满足要求。不要仅凭镜像站首页能打开就开始几百 GB 的任务。

## 2 Hypersim 国内原始包与光轴深度转换

入口：[魔搭 OpenDataLab/Hypersim](https://modelscope.cn/datasets/OpenDataLab/Hypersim)、[Apple 官方资料](https://github.com/apple-aiml-research/ml-hypersim)。

本次完整分页清单中有 **457 个场景 ZIP，约 1.914 TB**，与当前 Apple 下载脚本列出的 457 个文件名一致；MVSAnywhere 当前 TRAIN 清单需要 303 个场景，这些场景均在镜像中。文件名覆盖核查不等于已经验证每个压缩包内容或整包哈希。单个 `ai_001_001.zip` 的 HEAD 为 200，最终域名为 `cdn-lfs-cn-1.modelscope.cn`。

已为本项目准备两份 aria2 清单，包含镜像提供的 SHA256 校验信息：

- [只取官方训练清单所需的 303 个场景](download_assets/mvsanywhere_20261023/hypersim_train_303.aria2)
- [全部 457 个场景](download_assets/mvsanywhere_20261023/hypersim_all_457.aria2)

先选其中一份，不要两个都重复执行：

```bash
mkdir -p "$D/archives/hypersim" "$D/hypersim/raw"
aria2c --continue=true --max-concurrent-downloads=2 \
  --split=4 --max-connection-per-server=4 \
  --dir="$D/archives/hypersim" \
  --input-file="$PROJECT/docs/download_assets/mvsanywhere_20261023/hypersim_train_303.aria2"

for archive in "$D/archives/hypersim"/*.zip; do
  unzip -n "$archive" -d "$D/hypersim/raw"
done
```

也可以先用单包测试。下面命令会下载约 2.25 GB，不是小文件探针：

```bash
curl -fL -C - \
  https://modelscope.cn/datasets/OpenDataLab/Hypersim/resolve/master/raw/ai_001_001.zip \
  -o "$D/archives/hypersim/ai_001_001.zip"
```

解压后的必要结构：

```text
hypersim/raw/ai_001_001/
  _detail/cam_00/camera_keyframe_positions.hdf5
  _detail/cam_00/camera_keyframe_orientations.hdf5
  _detail/metadata_scene.csv
  images/scene_cam_00_final_preview/frame.0000.tonemap.jpg
  images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
```

原始 `depth_meters.hdf5` 是相机到像素对应三维点的**射线距离**，不能直接作为 Z-depth。需要按每个场景的相机射线方向转换，不要统一假设一个焦距。Apple 的部分场景使用非标准相机参数。[官方相机参数说明](https://github.com/apple-aiml-research/ml-hypersim/blob/main/contrib/mikeroberts3000/README.md)。

已保存官方相机参数 CSV 的本地副本：[metadata_camera_parameters.csv](download_assets/mvsanywhere_20261023/metadata_camera_parameters.csv)。来源为 Apple 的 `contrib/mikeroberts3000/metadata_camera_parameters.csv`；SHA256 在同目录 `source_audit.json` 中。此文件需要放到加载器实际查找的位置：

```bash
mkdir -p "$D/hypersim_preprocessed"
cp "$PROJECT/docs/download_assets/mvsanywhere_20261023/metadata_camera_parameters.csv" \
   "$D/hypersim_preprocessed/metadata_camera_parameters.csv"
```

项目新增的纯数据转换工具只依赖 NumPy/h5py，可绕开官方脚本因公共导入而引入的 PyTorch3D 依赖。先转换一帧：

```bash
python "$PROJECT/tools/prepare_hypersim_planar.py" \
  --raw-root "$D/hypersim/raw" \
  --metadata "$D/hypersim_preprocessed/metadata_camera_parameters.csv" \
  --split "$R/data_splits/hypersim/bd_split/train_files_bd.json" \
  --limit 1
```

确认路径、读写正常后，去掉 `--limit 1` 转换训练清单。已有输出会跳过。输出为：

```text
MVSAnywhere_pretrain/hypersim_preprocessed/ai_001_001/images/
  scene_cam_00_geometry_hdf5/frame.0000.depth_meters_planar.hdf5
```

**配置的 dataset_path 仍指向 `$D/hypersim/raw`，不是 planar 输出目录。** 官方加载器通过 raw 的父目录名称推导输出路径。转换工具按官方习惯保存 float16 HDF5，原始距离文件保留不动。该工具仅做数据格式转换，不改变模型。

若使用官方转换脚本，应从仓库根目录使用实际存在的脚本路径，并以 `--config_file` 合并单数据集选项：

```bash
cd "$R"
PYTHONPATH=src python scripts/data_scripts/generate_hypersim_planar_depths.py \
  --config_file "$D/configs/hypersim.yaml" --num_workers 8
```

这里需先完成第 10 节的配置复制。README 的旧 `./data_scripts/...` 路径和本机源码不一致；当前 `--data_config_file` 会生成数据集列表，而这个旧脚本访问扁平 `opts.dataset`，不能直接照搬。原脚本 `--help` 在项目方法环境中实际因缺 PyTorch3D 失败；纯数据工具的检查结果见最后一节。

## 3 TartanAir 选择 V1 与左目完整模态

入口：[官方工具仓库](https://github.com/castacks/tartanair_tools)、[作者 HF 原始数据仓库](https://huggingface.co/datasets/theairlabcmu/tartanair)。

MVSAnywhere 的当前加载器对应 **V1 的 Easy/Hard 目录格式**。不要用新发布的 V2 或仅包含 48×48 图像的轻量版本替代。

```bash
hf download theairlabcmu/tartanair --repo-type dataset \
  --include '*image_left.zip' '*depth_left.zip' '*seg_left.zip' \
  --local-dir "$D/archives/tartanair"
```

当前仓库这些左目归档共约 213.7 GB，覆盖本地 TRAIN 清单要求的 16 个环境；这是下载文件清单的计算，不是解压后体积。位姿可能随归档提供，解压后必须确认 `pose_left.txt`；不能因为没有单独 pose ZIP 就认为不需要位姿。

```bash
mkdir -p "$D/tartanair"
python - "$D/archives/tartanair" "$D/tartanair" <<'PY'
from pathlib import Path
import sys
import zipfile
source, target = map(Path, sys.argv[1:])
for archive in sorted(source.rglob('*.zip')):
    with zipfile.ZipFile(archive) as reader:
        reader.extractall(target)
PY
```

必要结构为 `环境/Easy或Hard/Pxxx/{image_left,depth_left,seg_left,pose_left.txt}`。检查 ZIP 内顶层路径是否已有环境名，避免多加一层同名目录。官方 V1 使用 `000000_left.png`、`000000_left_depth.npy`、`000000_left_seg.npy` 等命名；保留原名，不要展平帧序列。

官方当前下载器也支持 `--huggingface`，可在作者工具仓库运行：

```bash
python download_training.py --output-dir "$D/tartanair" \
  --rgb --depth --seg --only-left --unzip --huggingface
```

该命令需要工具目录内的 URL 清单及依赖。MVSAnywhere README 中 TartanAir 下载说明误指到了 Hypersim 仓库，应使用上面的 TartanAir 官方仓库。原始 `.npy` 深度按加载器读取；不要当作 uint16 PNG 再除 1000。seg 被用于有效区域/天空处理，不能只下载 RGB 和 depth。

## 4 BlendedMVG 必须包含三个低分辨率集合

入口：[官方 BlendedMVS 仓库](https://github.com/YoYo000/BlendedMVS)。MVSAnywhere 使用的是 **BlendedMVS + BlendedMVS+ + BlendedMVS++**，均为 768×576 低分辨率数据。仅下载基础 BlendedMVS 会漏掉扩展训练场景。

目前核实的官方 Release：

| 集合 | Release | 文件 | 当前资产总量 |
|---|---|---|---:|
| BlendedMVS | v1.0.0 | BlendedMVS.z01–z15 与 BlendedMVS.zip | 29.53 GB |
| BlendedMVS+ | v1.0.1 | BlendedMVS1.z01–z42 与 BlendedMVS1.zip | 87.49 GB |
| BlendedMVS++ | v1.0.2 | BlendedMVS2.z01–z42 与 BlendedMVS2.zip | 85.91 GB |

实际 API 资产数量已核查。官方 README 仍把基础集合写成单个 ZIP；只取 `.zip` 文件会漏掉分卷。[基础集合下载说明问题](https://github.com/YoYo000/BlendedMVS/issues/48)。这些 Release 属于境外来源；未找到可确认覆盖三个原始集合的纯国内镜像。

```bash
mkdir -p "$D/archives/blendedmvg" "$D/blendedmvg"
cd "$D/archives/blendedmvg"
for spec in 'v1.0.0 BlendedMVS 15' 'v1.0.1 BlendedMVS1 42' 'v1.0.2 BlendedMVS2 42'; do
  read -r tag name count <<< "$spec"
  for part in $(seq -f '%02g' 1 "$count"); do
    curl -fL -C - "https://github.com/YoYo000/BlendedMVS/releases/download/$tag/$name.z$part" \
      -o "$name.z$part" || break 2
  done
  curl -fL -C - "https://github.com/YoYo000/BlendedMVS/releases/download/$tag/$name.zip" \
    -o "$name.zip" || break
done
```

分卷 ZIP 应先合并，不能把每个 `.z01` 单独解压：

```bash
for name in BlendedMVS BlendedMVS1 BlendedMVS2; do
  zip -s 0 "$name.zip" --out "$name.joined.zip"
  unzip -n "$name.joined.zip" -d "$D/blendedmvg"
done
```

需要 `zip` 工具；合并文件额外占磁盘。解压后三套的**场景目录**应放在同一根目录，场景中保留 `blended_images/*.jpg`、`rendered_depth_maps/*.pfm`、`cams/*_cam.txt` 和配对信息。先看 `unzip -l` 的根目录，不能在未知目录层级时盲目 `--strip-components`。

条件镜像：[Gofinge/DenseHybrid](https://huggingface.co/datasets/Gofinge/DenseHybrid/tree/main) 的文件树存在三个 Blended 子集。它是社区重打包，未逐场景核对原始帧 ID、分辨率、相机和 PFM 内容；本教程不将其列为已经验证的原论文复现来源。普通 BlendedMVS 的国内资源也不自动覆盖两个扩展集。

## 5 MatrixCity 优先作者百度网盘

[作者下载说明](https://github.com/city-super/MatrixCity#data-download) 提供：

- [百度网盘](https://pan.baidu.com/s/187P0e5p1hz9t5mgdJXjL1g)，提取码 **hqnn**。
- [OpenXLab](https://openxlab.org.cn/datasets/bdaibdai/MatrixCity)。
- [作者 HF 仓库](https://huggingface.co/datasets/BoDai/MatrixCity/tree/main)。

百度和 OpenXLab 链接由作者发布；本次没有登录网盘完成实际文件下载。选择 `big_city`、`big_city_depth`、`big_city_depth_float32`。不用把整个 2.55 TB 的 HF 仓库全部下载。上述三个目录当前资产合计约 **1.027 TB**，包含该目录下的 train/test；若只训练，可再按官方训练清单选择所需块。

条件镜像命令：

```bash
hf download BoDai/MatrixCity --repo-type dataset \
  --include 'big_city/*' 'big_city_depth/*' 'big_city_depth_float32/*' \
  --local-dir "$D/matrix_city"
```

普通 `.tar` 在所在目录中解压。遇到 `big_high_block_1.tar00` 这类连续切割的 tar，先按编号拼回去：

```bash
cd "$D/matrix_city/big_city/aerial/train"
cat big_high_block_1.tar[0-9][0-9] > big_high_block_1.joined.tar
tar -tf big_high_block_1.joined.tar | head
# 确认成员根路径正确后再执行：
tar -xf big_high_block_1.joined.tar
```

各目录、各块分别处理，不要把所有 `.tar00` 混合拼接。解压后保留 `transforms.json`，使数据根下能找到 `big_city/aerial/train/块名/transforms.json` 等路径。

RGB 为 PNG，深度为 EXR、Z-depth，原始单位厘米。**航拍使用 float32 深度，街景读取普通 depth 目录**；这正是当前官方加载器的分支逻辑。加载器会除以 100 转成米，提前改文件再除一次会错。MatrixCity 官方说明了 float16 EXR 在高空距离的饱和问题；不要为了省空间统一转 float16。[深度说明](https://github.com/city-super/MatrixCity#depth-and-normal-map-note)。

## 6 Virtual KITTI 2 下载四种归档并合并目录

[官方页面](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)。需要 **2.0.3** 版本：

```text
vkitti_2.0.3_rgb.tar
vkitti_2.0.3_depth.tar
vkitti_2.0.3_classSegmentation.tar
vkitti_2.0.3_textgt.tar.gz
```

本次查到 [nhatchung/Virtual_KITTI2](https://huggingface.co/datasets/nhatchung/Virtual_KITTI2/tree/main) 社区镜像包含四个原名归档，总计约 16.72 GB；没有验证其与官方逐文件哈希一致，HF-Mirror 本次该链接 HEAD 超时。

```bash
hf download nhatchung/Virtual_KITTI2 --repo-type dataset \
  --include 'vkitti_2.0.3_rgb.tar' 'vkitti_2.0.3_depth.tar' \
            'vkitti_2.0.3_classSegmentation.tar' 'vkitti_2.0.3_textgt.tar.gz' \
  --local-dir "$D/archives/vkitti2"
```

官方直链也可使用；不能保证国内下载速度：

```bash
mkdir -p "$D/archives/vkitti2"
cd "$D/archives/vkitti2"
for file in vkitti_2.0.3_rgb.tar vkitti_2.0.3_depth.tar \
            vkitti_2.0.3_classSegmentation.tar vkitti_2.0.3_textgt.tar; do
  curl -fL -C - "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/$file" -o "$file"
done
```

官网当前 textgt 链接是 `.tar`，上述社区镜像文件名是 `.tar.gz`，按实际归档格式解压，不要只改扩展名。三种普通 `.tar` 使用 `tar -xf`；镜像 textgt 使用 `tar -xzf`。

各包解压到同一个 `$D/vkitti2`；若包内各自还有归档名称顶层目录，则合并其中的 `Scene*` 子目录。最终应能同时找到：

```text
vkitti2/Scene01/clone/
  intrinsic.txt
  extrinsic.txt
  frames/rgb/Camera_0/rgb_00000.jpg
  frames/depth/Camera_0/depth_00000.png
  frames/classSegmentation/Camera_0/classgt_00000.png
```

深度是 uint16 PNG、厘米单位，官方加载器除 100；保留 classSegmentation 以满足当前加载器。不要下载 Virtual KITTI 1.3.1 冒充 V2；不少国内搜索结果实际是旧版本。

## 7 Dynamic Replica 通过官方授权入口取训练集

[官方项目页面](https://dynamic-stereo.github.io/)、[官方代码](https://github.com/facebookresearch/dynamic_stereo#download-the-dynamic-replica-dataset)。

1. 在项目页面 Data 入口阅读并接受数据协议，取得 `links.json`。
2. 准备官方 `dynamic_stereo` 仓库和下载脚本依赖，将 JSON 放到自己的下载工具目录。
3. 只选 `train` 下载；验证/测试按实际需要另取。

```bash
# 在已经准备好的官方 dynamic_stereo 仓库根目录执行。
python scripts/download_dynamic_replica.py \
  --link_list_file "$D/tools/dynamic_replica_links.json" \
  --download_folder "$D/dynamic_replica" \
  --download_splits train
```

下载器会校验并解压归档。官方列出的 train 全模态解压后约 **1.8 TB**，不是只含训练必需模态的大小；光流、轨迹等都在全量中。先确认 RGB、depth 和元数据读取正常，再考虑不保留不使用的光流、轨迹等；不要删除 `frame_annotations_train.jgz`。

当前 MVSAnywhere 加载器从 `dataset_path/train/frame_annotations_train.jgz` 读取注释，以注释中的相对路径定位图片和深度，并只选左相机。请按实际解压结果确定 dataset_path，勿重复多套一层 `train`。

**深度 PNG 保存的是 float16 的位模式**，应将 uint16 内存重新解释为 float16，再转 float32；直接除以 100 或 1000 都不对。使用原加载器即可完成该步骤，通常不需要先批量转 NPY。

关于条件镜像：

- `ZhengGuangze/DynamicReplica_raw` 本次清单只有 `test_*.zip`、`valid_*.zip`，不能补齐训练集。
- `geyongtao/dynamic_replica` 本次看到了 `train_001.zip` 到 `train_020.zip`，但未看到独立 annotation 包/完整说明，未解包核验，不能承诺兼容。
- `DynamicReplica_vlbm`、WAI、CUT3R 等处理版改变过目录、分辨率、深度或相机精度，需要额外适配，不能冒充原始数据。

因此本教程将官方授权包作为复现主来源；没有确认可用的纯国内完整原生镜像。

## 8 MVS-Synth 下载作者发布的 540p 版本

[数据集主页](https://phuang17.github.io/DeepMVS/mvs-synth.html)、[作者仓库 phuang17/MVS-Synth](https://huggingface.co/datasets/phuang17/MVS-Synth)。

只需 `GTAV_540.tar.gz`，分辨率 960×540。当前文件实际为 **10,492,893,238 字节，约 10.49 GB**；旧页面标注 9 GB，不宜据此判断文件损坏。

```bash
mkdir -p "$D/archives/mvssynth" "$D/mvssynth"
hf download phuang17/MVS-Synth GTAV_540.tar.gz --repo-type dataset \
  --local-dir "$D/archives/mvssynth"
```

CLI 元数据请求失败但镜像直链可达时：

```bash
curl -fL -C - \
  https://hf-mirror.com/datasets/phuang17/MVS-Synth/resolve/main/GTAV_540.tar.gz \
  -o "$D/archives/mvssynth/GTAV_540.tar.gz"

tar -tzf "$D/archives/mvssynth/GTAV_540.tar.gz" | head
# 检查存在 GTAV_540 顶层后，解压到上一级。
tar -xzf "$D/archives/mvssynth/GTAV_540.tar.gz" -C "$D/mvssynth"
```

配置指向 `$D/mvssynth/GTAV_540`。必要目录是 `0000/images/*.png`、`0000/depths/*.exr`、`0000/poses/*.json`。EXR 中的 `inf` 包括天空等无限远情况；不要变成普通有效深度。相机 extrinsic 是 world→camera，部分旋转的行列式为 -1，坐标转换应沿用对应加载器。无需为学习深度而先把 depth 取倒数改成 disparity。

本次 HEAD 最终到了境外 CDN，因此这个 HF-Mirror 入口只能作为条件下载路线。

## 9 SAIL-VOS 3D 要申请深度与额外矩阵

[官方网站](https://sailvos.web.illinois.edu/_site/_site/index.html)、[数据 README](https://sailvos.web.illinois.edu/_site/_site/dataset_readme.html)。

按官网联系作者，说明用于 MVSAnywhere 的非商业多视图深度研究。申请时明确需要：

- 训练序列的 RGB 图像。
- 每帧 `camera/*.yaml` 相机信息。
- `depth/*.npy` 原始深度，官网说明此项需额外申请。
- 与 MVSAnywhere 加载器兼容的 `rage_matrices/*.npz`，其中包含 `P`、`P_inv`。

最后一项是代码核查发现的要求：官方普通说明仅列出相机 YAML，而当前 MVSAnywhere 的内参读取和 NDC 深度转换都要用 RAGE 投影矩阵。仓库未见这些矩阵的生成脚本。不能把缺少它的普通下载包直接写成“解压即可训练”；需要数据作者或 MVSAnywhere 作者补充这部分预处理资源。

```text
sailvos3d/序列/
  images/000000.bmp
  camera/000000.yaml
  depth/000000.npy
  rage_matrices/000000.npz
```

原始 depth 在当前加载器中按 NDC 深度解码，不是拿 NPY 数值直接当米。下载脚本和链接依作者授权提供，不能在取得前编造通用直链。官网要求非商业研究用途、购买 GTA V 并禁止再次分发；没有确认授权的国内公开镜像。

## 10 配置路径与公共预处理

保持官方源代码和数据划分，复制数据配置到自己的数据目录后，仅修改 dataset_path。下面代码不会修改官方配置：

```bash
python - "$R" "$D" <<'PY'
from pathlib import Path
import re
import sys
repo, data = map(Path, sys.argv[1:])
paths = {
    'hypersim': 'hypersim/raw',
    'tartanair': 'tartanair',
    'blendedmvg': 'blendedmvg',
    'matrix_city': 'matrix_city',
    'vkitti': 'vkitti2',
    'dynamic_replica': 'dynamic_replica',
    'mvssynth': 'mvssynth/GTAV_540',
    'sailvos3d': 'sailvos3d',
}
for name, folder in paths.items():
    source = repo / 'configs/data' / name / f'{name}_default_train.yaml'
    text = re.sub(r'^dataset_path:.*$', f'dataset_path: {data / folder}',
                  source.read_text(), flags=re.MULTILINE)
    (data / 'configs' / f'{name}.yaml').write_text(text)
PY
```

`tuple_info_file_location` 和 split 文件继续沿用仓库 `data_splits/`；从 `$R` 运行官方脚本，使相对路径正确。作者已经提供 tuple 清单，一开始不需要重跑所有配对生成。若删帧或只下载部分场景，必须产生独立子集清单并记录，不要静默跳过缺失项后仍称完整复现。

公开八套数据多数是相机序列或一般多视图，**不是八套同步四目阵列数据**。其作用是多视图预训练；接入 CARLA 同步四目时另行适配源视图数量，不能复制图片凑数，或把时间相邻帧改称同一时刻。

单位与格式注意：

| 数据 | 原始深度/尺度重点 | 处理方式 |
|---|---|---|
| Hypersim | 射线距离，米；位姿存在场景资产单位 | 距离转光轴深度；位姿用官方场景尺度 |
| TartanAir V1 | `.npy` 深度与 `pose_left.txt` | 原加载器解码，保持命名和坐标约定 |
| BlendedMVG | `.pfm`，论文不将该数据视作绝对米制位姿 | 保持相机与深度同尺度，不猜一个米制常数 |
| MatrixCity | EXR，厘米，Z-depth | 加载器除 100；航拍 float32 |
| VKITTI2 | PNG uint16，厘米，Z-depth | 加载器除 100；保留天空/无效掩码 |
| Dynamic Replica | PNG 内容为 float16 位模式 | reinterpret uint16→float16，再转 float32 |
| MVS-Synth | EXR、无限远与特殊坐标约定 | 保留官方处理；论文不将其归为绝对米制位姿 |
| SAIL-VOS 3D | 原始 NDC 深度 | 依 `P_inv` 解码为相机深度 |

几何标定与深度必须同时处于一致单位。原始分辨率应保留，训练时缩放由加载器同步更新内参；不要离线随意裁图后沿用旧 K。深度可视化 RGB 图不是监督标签。

## 11 验证集、预训练权重与下载后检查

ScanNet v2 在官方训练示例中用于验证，不列入上述八套训练集。[官方提取教程](https://github.com/nianticlabs/mvsanywhere/tree/main/scripts/data_scripts/scannet_wrangling_scripts)。下载需遵循 ScanNet 申请；国内平台上的第三方副本也应核对授权及版本。使用官方 `reader.py` 从 `.sens` 提取 RGB、depth、pose、intrinsic；只下 mesh 不能验证深度网络。

```bash
cd "$R/scripts/data_scripts/scannet_wrangling_scripts"
python reader.py \
  --scans_folder "$D/scannet/scans" \
  --output_path "$D/scannet/scans" \
  --scan_list_file splits/scannetv2_val.txt \
  --num_workers 8 --export_poses --export_depth_images \
  --export_color_images --export_intrinsics
```

另外，官方从头训练命令需要 Depth Anything V2 **Base 相对深度权重**。不要误选 Small/Large 或 metric depth 版本。[官方权重说明](https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models)。

```bash
hf download depth-anything/Depth-Anything-V2-Base depth_anything_v2_vitb.pth \
  --local-dir "$D/pretrained"
```

若当前已有完整官方 MVSAnywhere checkpoint，可以直接用其所含编码器状态；不必为微调而重新下载所有前置预训练数据。

正式训练前按顺序检查：

1. 下载工具退出成功，远端大小/提供的哈希相符。自己第一次生成的 SHA256 只能作为以后副本比对基准，不能证明原下载正确。
2. 检查 ZIP/tar 完整性，保留下载清单和来源版本。分卷必须完整。
3. 随机看三个场景，确认 RGB、深度、相机和帧 ID 能一一对应。
4. 按原加载器读一个 tuple，查看深度值域、有效比例和变换方向；使用已知标定检查同一静态点在另一视图的投影。不要把独立采样的随机帧当作对应关系。
5. 先加载单个训练样本并运行一次前后向，再启动多卡训练。缺失 RAGE 矩阵、位姿或 annotations 时明确报错，不能自动伪造。
6. 为当前自研模型接入时，保持 inputs/targets 分离；训练/验证/测试按场景或轨迹划分。不能用测试真值深度确定推理搜索范围。

## 12 原始调研阶段的核查记录（2026-10-23）

本节保留首次调研时的状态。10月24日已经完成的下载、真实转换及模型检查见第1节，不再用以下“未下载”描述当前状态。

- 已读取论文和官方训练清单；核对八个数据集加载器。线上 main commit 为 `5bd49bba0992aa56060e1aa3bb40b2150b04726f`，本地项目源码需以项目封装记录为准。
- 已列出 Hypersim 全部 457 场景包，文件名与 Apple 下载脚本一致，并确认当前 TRAIN 所需 303 场景在镜像中。
- 已核查 TartanAir 所需环境覆盖、MatrixCity 三目录、VKITTI2 四包、MVS-Synth 原始包大小，以及 BlendedMVG 三组分卷的数量。
- Hypersim 样例下载 HEAD 为国内 CDN；若干 HF-Mirror 大文件实际重定向境外，VKITTI2 镜像 HEAD 超时。没有据此保证用户所在网络的持续速度。
- 当前项目 `hf` 小文件 CLI 检查失败，官方 Hypersim 转换脚本公共导入因 PyTorch3D 缺失失败。失败记录保留，没有写为下载成功。
- 纯数据工具实际完成三项检查：四个真实相机场景参数下的射线因子对照官方函数，最大差1.19e-7；已知平面与NaN的合成HDF5转换；跳过已有文件与临时写入完成后的原子替换。项目原环境缺少h5py，验证时仅在/tmp独立依赖目录安装，没有修改训练环境。结果见 [核查记录](download_assets/mvsanywhere_20261023/source_audit.json)。未下载真实 Hypersim 场景运行转换，未完成任何数据集的全量解压或官方训练验收。

后续实际选择了 Hypersim 单场景、TartanAir 森林和 MatrixCity 街景子集。MVS-Synth 的特殊标定以及 VKITTI2 动态双目与当前四视图协议的差异需要另行处理，本次没有下载它们。子集接入不代表完整数据集或论文精度复现。
