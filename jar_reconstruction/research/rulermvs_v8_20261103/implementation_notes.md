# V8 实现与验证边界

2026-11-03。独立文件 `model.py`、`test_model.py`。未改 V6、公共模型源码、原权重或经验文档。训练/GPU/真实精度由主任务另行调度。

## 接口

```python
from research.rulermvs_v8_20261103.model import V8Model
model = V8Model(config, initialize=False)
model.load_state_dict(v6_checkpoint["model"], strict=True)
```

V8 与 V6 的全部 state keys/shapes、参数数量完全相同，无新增可训练参数。原 `forward(inputs, domain, reference=0)`、`set_stage`、`encode`、`forward_references` 与 optimizer 分组保持可用。`build_from_checkpoint(saved, options=None)` 也可严格加载 V6/V8。

顶层配置：

```json
{
  "architecture": "rulermvs_v8",
  "v8": {
    "information_mode": "log_depth",
    "information_scale": 10.0,
    "source_count_strength": 0.0,
    "source_count_reference": 3.0,
    "relative_validity": true,
    "relative_validity_threshold": 0.00001,
    "physical_thresholds": true,
    "proposal_scale_reference": "nominal",
    "position_mode": "range"
  }
}
```

以上示例只打开单位表示修正，`source_count_strength=1` 则加源数校正。所有默认值均兼容旧模型：information_mode=legacy，source_count_strength=0，relative_validity=false，physical_thresholds=false；其它数字不会在关闭时改变原函数。proposal_scale_reference 默认 input，保留首轮 V8 实验；示例显式选择后续 nominal 协议。position_mode 目前只支持原 range，未知开关直接报错。

## 具体变化

1. log_depth 模式把 source 输入导数换成 Zg、半宽换成 h/Z，因此 pair 的投影跨度仍是 |g|h，导数方向与方向归一化 epsilon 都采用无量纲导数。信息变成 `sum(w*|Zg|²) / information_scale²`。只改变候选标量及2维导数，不增加投影/采样、卷积、候选数量。默认10只在旧内部Z=10处对齐旧量级，不估计场景尺度。
2. 源数修正逐候选加 `strength*log(max(m,1)/3)` 到4头null logit。三有效源时原函数保持，零有效源安全。单个源或整套源集合等倍复制时，strength=1的pooled/null/information保持；不同独立视图仍有不同值与logit，因此不是“图数无作用”。没有在深度候选轴归一化，也没有删除原cosine旁路。
3. relative_validity 将 source 资格改为原图像素下 `|g*Z| > threshold`。精确投影/FPN路由/采样沿用原表达，两种 projective_ruler 路径都支持。
4. physical_thresholds 每个 domain 调整 `peak_separation_m`、`fine_min_step_m`、`fine_max_step_m`。`proposal_scale_reference=input` 保留首轮 V8 的配置值除以 `domain.depth_scale`；阈值代表原输入长度单位，对于 scene_unit 数据不冒称真实米。`proposal_scale_reference=nominal` 则使用下节的旧 V6 nominal 候选契约。两者只改 decoder 私有 settings 副本，从初始配置值每次计算，既不累计除法也不污染 config/checkpoint。

source attention 的 enabled分支保留 hoist_reference 与 compact_pair，tokens继续调用原 factorized_fusion。全关直接走原V6函数，包含原浮点顺序。为不改公共源码，source attention 的局部forward与观察函数被独立覆盖；未复制整个解码器/渲染/候选交接。

## Nominal 候选尺度契约（同日追加修正）

设配置中的旧 V6 阈值为 c，未扰动域尺度为 s0，当前内部换算尺度为 s。nominal 模式采用 `c_internal = c * (s0 / s)`，所以输入单位中的阈值恒为 `c * s0`。计算先求比值，s=s0 时比值严格等于1，保留旧 V6 的原始数值。Synth s0=26.625 时，fine_max 输入单位上限恢复为2662.5，而不是首轮 input 模式的100；这不会额外扩大旧 V6 的 nominal 候选供给。

`domain.nominal_depth_scale` 声明 s0；训练的 `compose_training_domain` 已在 jitter 前保存此字段。推理没有此字段时，当前 `domain.depth_scale` 被视为未扰动 s0。调用者若改变内部尺度，必须保留原 nominal 字段；仅更改任意 domain dict 的 depth_scale 不构成纯单位变换契约。forward、encode、forward_references 均传递该字段。原输入长度单位自身变化时，s0 和 s 应同时作相应换算。

这只是三个候选阈值的校正，没有新增网络层，也没有修改信息、源数、候选评分或 epsilon。它不承诺完整网络在极端尺度或硬换峰处精确等变。旧 input 模式与关闭 physical_thresholds 的行为保持原样。

## 已执行 CPU 验证

```bash
CUDA_VISIBLE_DEVICES='' official_methods/.venv/bin/python -m unittest research.rulermvs_v8_20261103.test_model -v
```

首轮11项全部通过，最终运行耗时1.677秒；nominal 修正后14项全部通过，耗时2.449秒。使用真实ResNet与TinyViT替身、小图、小候选，未加载正式训练checkpoint或真实数据。覆盖：

- V6→V8严格state与全关所有阶段/完整输出逐位等价，参数数量相同。
- 非零4头source/null logits、不同候选mask、整套源复制不变性、三有效源保原、零有效源有限输出/梯度。
- 两种hoist/compact组合；factorized_fusion完整前向；torch.compile的CPU eager backend fullgraph。后者只验证图捕获和逻辑，**不是**CUDA/Inductor成功或时延证据。
- 固定物理输入、内部scale=.25/.5/1/2/4的标尺信息与源mask、候选/输出变换；源导数接近旧阈值和纯旋转；两种投影路径。
- 私有settings不污染config，来回换domain不累积阈值变化。
- nominal=26.625/.23671875、factor=.5/1/2 时阈值输入单位值恒定；nominal=26.625 且仅启用候选阈值修正时，完整 coarse/fine 输出与 V6 逐位相同；forward/encode/共享 references 显式 nominal 单位变换通过，缺省字段声明未扰动域，input 模式仍按原表达计算。
- 现有真实目标反向、source梯度非零有限、optimizer更新、set_stage、严格重载与共享多参考接口。

information_scale=1和10均通过同一单位元变换测试。这类测试只能验证任一固定数值归一常数不破坏代数，**不能**决定哪一常数精度更好；本轮10是事前数值约定，不是经过VAL/TEST拟合的结论。

第一次测试中两处过严容差失败；随后定位为FP32数值差，没有隐藏失败或宣称逐位单位等变。固定随机fixture的最大最终深度差约1.15e-5（恢复输入单位后），最大fine logit差8.98e-5；scale=.25时3个fine单点selected_index改变，但组读出/最终差仍在上述范围。不同投影表达的极小导数约1.4e-7差。最终测试显式允许这些数值容差，仍要求source_valid掩码相同。真实checkpoint的硬换峰可能放大微差，未在本文件测试。

## 保留的数值边界

没有宣称任意极端长度/视角下整个网络严格单位等变。未改的epsilon/截断包括：

| 位置 | 常数/作用 | 边界 |
|---|---|---|
| `jar/geometry.py:59,64`，`projective_ruler.py:40,45` | 源光轴深度clamp/合法性1e-4，当前内部单位 | 极近源光轴或极端单位换算仍可能改变投影资格；正常测试候选远离该界限 |
| `ruler_proposals.py:114` | 半像素步长分母g下限1e-6 | 常见范围先被物理max_step截断，但极端范围未保证不变 |
| `ruler_direct.py:413` | nearest路径同类分母/步长 | V6当前强制bilinear，未启用该路径 |
| source几何 | 方向归一化1e-6、跨度log1p后上限8 | log_depth模式用Zg，方向epsilon无内部长度单位；饱和仍有信息损失 |
| `geometry.unproject` | ray第三分量clamp1e-8 | 相机射线规范化；标准K第三行为(0,0,1)，不是深度尺度修正 |
| 归一化与概率 | 特征L2 epsilon1e-6，mask与概率1e-8/1e-12等 | 非长度单位；未改变 |
| 训练目标 | depth clamp1e-6、相对误差分母0.1 | 损失在极近内部深度下仍不是单位不变；本任务仅改模型，未改既有目标 |

范围位置编码、FPN投影跨度、96格log范围、父族交接和族内均值都保留。因此本实现只测试明确的量纲/源数机制，没有得到任何新的真实精度、零样本泛化、时延或显存结论。
