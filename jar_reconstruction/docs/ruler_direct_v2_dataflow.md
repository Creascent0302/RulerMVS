# 当前自研架构的数据流计算图

绘制日期：北京时间2026-10-26。对应当前 `PretrainingModel` / `ruler_direct_v2`，采用 [train_latest_20261025.json](../configs/pretraining/train_latest_20261025.json) 的实际结构。它已实现通用初始化、编码器继续训练和标尺候选解码；不能再按旧的冻结Hero编码器版本解释。本次只核对和画图，没有修改模型或开展性能实验。

## 按顺序打开四页

完整可编辑文件：[四页draw.io计算图](diagrams/自研RulerDirectV2_数据流_20261026.drawio)。在VS Code的draw.io编辑器或diagrams.net打开，底部切换页面。

| 页 | 主要内容 | 矢量图 | 图片 |
|---|---|---|---|
| 1 | 输入、两条编码支路、粗候选、细候选、深度输出的总数据流 | [总图SVG](diagrams/自研RulerDirectV2_数据流_20261026_01.svg) | [总图PNG](../runs/active/architecture_20261026/自研RulerDirectV2_数据流_20261026_01.png) |
| 2 | ResNet各层、FPN、ViT每块结构及特征尺寸 | [编码器SVG](diagrams/自研RulerDirectV2_数据流_20261026_02.svg) | [编码器PNG](../runs/active/architecture_20261026/自研RulerDirectV2_数据流_20261026_02.png) |
| 3 | 标尺尺度路由、源视图注意力、无匹配槽位、四层上下文注意力 | [匹配融合SVG](diagrams/自研RulerDirectV2_数据流_20261026_03.svg) | [匹配融合PNG](../runs/active/architecture_20261026/自研RulerDirectV2_数据流_20261026_03.png) |
| 4 | 候选空间交互、96到12候选传递、共享输出头、上采样与独立监督 | [解码SVG](diagrams/自研RulerDirectV2_数据流_20261026_04.svg) | [解码PNG](../runs/active/architecture_20261026/自研RulerDirectV2_数据流_20261026_04.png) |

SVG和draw.io保存在代码目录；PNG、PDF及形状核对记录在数据盘已有软链接 [architecture_20261026](../runs/active/architecture_20261026) 下。

## 先用文字理解主线

输入的每张图片有两种尺寸。较大的512×896图片进入共享ResNet与FPN，用来寻找对应位置、保留纹理和小物体。256×448主图中的本次参考图进入ViT，提供对道路、物体结构和大范围场景的理解。

在32×56的参考网格上，每个位置先放96个可能深度。对每个深度假设，模型用标定计算它应该出现在其他相机的什么像素，再读取特征。标尺决定应该用多细的特征尺度，以及后续局部搜索的深度步长；它本身不能证明这个深度正确。

模型先比较各源相机的证据，再让候选读取参考图ViT的四层上下文，并与附近像素、附近深度格交换信息。这时仍然保留96个候选，没有先输出一个唯一深度再盲目细化。

到64×112细网格时，先传递完整的96项概率，再选择4个峰，每峰生成3个局部候选。12个候选都重新投影、读取源图。细层还继承粗层特征，随后选择一个候选并给出有界连续修正。最后根据图像特征预测邻域组合权重，把深度恢复为256×448。

## 模块与源码逐一对应

形状中的B是batch、N是相机数、C是通道、D是候选数。FPN的stride以512×896匹配图为基准；粗/细查询的stride以256×448主图为基准，两者不能混用。

| 顺序 | 实际模块 | 结构与输出 | 作用 | 源码 |
|---|---|---|---|---|
| 1 | `PretrainingModel.encode` | 主图/匹配图、掩码、K/T；外参平移除以域尺度s | 统一输入和工作单位 | [model.py](../jar/pretraining/model.py) |
| 2a | `MatchingPyramid` | ResNet-18四组各2个BasicBlock；FPN输出64通道的F4、F8、F16 | 全部相机共享纹理定位特征 | [encoders.py](../jar/pretraining/encoders.py) |
| 2b | `TrainableFoundation` | DA-V2 ViT-B，12块；第3/6/9/12块各768×16×28 | 本次参考图的多层上下文 | [encoders.py](../jar/pretraining/encoders.py) |
| 3 | `set_range`、`depth_lattice` | 由每域near/far构造96个对数间隔候选 | 覆盖给定搜索范围 | [model.py](../jar/pretraining/model.py) |
| 4 | `CalibratedDecoder._observe` | 反投影、精确投影、J与射线导数、FPN路由、特征归一化 | 按候选位置和尺度收集证据 | [model.py](../jar/pretraining/model.py) |
| 5 | `CandidateViewAttention` | 139→64→64成对MLP；4头注意力，每头16维；null槽位 | 聚合源证据并允许不采用源信息 | [model.py](../jar/pretraining/model.py) |
| 6 | `_tokens_from_source` | 66→64证据卷积；3→64→64位置编码；768→128四层适配及候选查询读取 | 结合匹配、几何与场景理解 | [model.py](../jar/pretraining/model.py) |
| 7 | `CandidateScaleSpace` | 三个空间尺度，各DW3D3×3×3→GELU→PW1×1×1；保留96深度格 | 让周边位置一起判断粗表面 | [ruler_spatial.py](../jar/models/ruler_spatial.py) |
| 8 | `_prediction` | score及offset两个64→1卷积；硬选最高分候选并在区间内连续修正 | 粗/细共享的预测规则 | [ruler_direct.py](../jar/models/ruler_direct.py) |
| 9 | `SubrayTransport`、`transport_proposals` | 同刻度概率的空间传递；选4峰；标尺步长；每峰3点 | 把粗证据转成更密集位置的细候选 | [ruler_proposals.py](../jar/models/ruler_proposals.py) |
| 10 | 细层 `_observe`、`_tokens`、`fine_inherit` | 重新取证；粗特征64→64投影后相加，再通道LayerNorm | 允许新证据修正粗阶段判断 | [model.py](../jar/pretraining/model.py)、[ruler_direct.py](../jar/models/ruler_direct.py) |
| 11 | `_upsample` | 64→64的3×3卷积及64→144的1×1卷积；9邻域×16子像素权重 | 学习式4倍上采样 | [ruler_direct.py](../jar/models/ruler_direct.py) |
| 12 | 导出与单位恢复 | 深度乘s；保存K/T及深度、置信度、掩码和支持数 | 恢复数据域原单位并封装输出 | [infer_public.py](../tools/infer_public.py) |
| 训练专用 | `depth_objective`、`visible_correspondence` | 全图深度、粗细分类/定位/风险、GT可见对应损失 | 更新编码器和解码器，不参与推理输入 | [losses.py](../jar/pretraining/losses.py) |

## 当前实现中需要准确理解的细节

- **是单参考前向。** CNN对全部相机槽位执行，缺图内容由掩码置零；ViT只处理本次参考图。参数共享不代表一次前向已经同时输出四个参考深度。当前 `forward` 每次调用 `encode`，不能假定轮换四参考会自动复用上次编码结果。
- **两种注意力的对象不同。** 源注意力的query来自参考像素，深度条件通过源采样位置和标尺进入key/value及可靠性。上下文注意力的query来自候选token，在四个ViT层之间选择；它不是另一次源相机注意力。
- **尺度路由不是省掉全部未选层采样。** 当前代码采样F4/F8/F16三层，再以最多两个非零权重混合。参考锚点始终取F4，源点按尺度混合；不能把计划中的对称多尺度读取当已实现行为。
- **粗层和细层并不完全相同。** 两层共享证据与上下文模块、评分和偏移头；只有粗层执行候选空间三维卷积，细层加继承特征后直接预测。不能在图中为细层增加不存在的Transformer或3D卷积堆叠。
- **尺度s来自每域固定训练配置。** 当前没有实现根据本帧相机基线自适应计算s。工作尺度变换也不会把`scene_unit`数据认证成米。候选间距和步长配置中的`_m`名字来自历史代码，在当前归一化模型中应按工作单位理解。
- **置信度与支持数不是认证。** confidence由候选softmax及上采样产生，support是几何上可采样的源图数。当前没有后端可见性认证，输出unknown全true、observed全false、state全2；不把该接口解释成已经学习了free/surface/unknown占据分类。
- **当前无匹配项没有独立监督。** null通过深度等任务损失间接学习。可见对应损失直接训练匹配描述子，不能写成已经对null概率进行了独立真值校准。
- **训练第一阶段仍训练深度解码器。** 当前阶段1冻结ViT，但全部七域仍参加深度训练；只有对应辅助项按域开关。它不是只用六域做纯匹配预训练。阶段2解冻ViT末4块与末端LN，阶段3/4全部解冻。
- **启用的速度优化没有增加模型层。** `compact_sources`删除恒被屏蔽的参考源计算；`spatial_layout`改变三维卷积输入的内存布局。`factorized_fusion`、`hoist_reference`、`compact_pair`在本图对应稳妥配置中关闭。

## 核对方法与参数规模

本次严格加载最新重训公共权重 `runs/active/public_retrain_latest_20261025/pretrained.pt`，用TartanAir的VAL第0组、N=4、参考索引2运行一次CPU前向，加载器设置 `load_targets=False`。实际hook确认：FPN为三张64维图、ViT四张768×16×28、粗层96×32×56、细层12×64×112、上采样mask为144×64×112、最终深度为1×1×256×448；输出有限，unknown/observed合同与源码一致。这不是精度或速度测评，没有使用VAL选择任何新参数。

| 部分 | 实际参数量 |
|---|---:|
| 通用上下文编码器 | 86,580,480 |
| ResNet及匹配金字塔 | 11,252,992 |
| 标尺候选解码器 | 513,562 |
| 总计 | **98,347,034** |

原始形状记录与源码哈希见 [shape_audit.json](../runs/active/architecture_20261026/shape_audit.json) 和 [diagram_sources.json](../runs/active/architecture_20261026/diagram_sources.json)。图中文字边界、节点/连线引用和四页XML均已检查，并逐页查看PNG。draw.io编辑器中的实际打开仍取决于本地插件；本轮未使用GUI编辑器打开。

复现图形导出，在主项目目录运行：

```bash
MPLCONFIGDIR=/tmp/jar_architecture_mpl \
  official_methods/.venv/bin/python tools/render_pretraining_architecture.py
```

脚本核对配置后再导出。如果模型的通道数、候选数或关键开关变化，需要重新核对图中文字和源码，不应直接沿用旧图。
