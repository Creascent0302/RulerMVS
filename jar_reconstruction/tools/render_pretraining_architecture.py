"""按当前 ruler_direct_v2 源码生成四页中文数据流图，不修改模型。"""

import hashlib
import html
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from render_current_architecture import COLORS, Page, save_visuals


ROOT = Path(__file__).resolve().parents[1]
STEM = "自研RulerDirectV2_数据流_20261026"


def heading(page, title, subtitle):
    page.node("title", title, 35, 15, page.width - 70, 65, "text", 35)
    page.node("subtitle", subtitle, 35, 80, page.width - 70, 55, "text", 21)


def box(page, name, label, x, y, width, height, color="blue", size=22):
    page.node(name, label, x, y, width, height, color, size)


def wire(page, source, target, points, color="blue", dashed=False):
    page.edge(source, target, color, dashed, points)


def overview():
    p = Page("01 按数据流阅读总图", 1800, 2310)
    heading(p, "当前自研架构：RulerDirect-V2", "PretrainingModel → CalibratedDecoder｜实线：推理数据流｜蓝：图像｜橙：标定｜紫：候选解码")
    box(p, "input", "① RGB与有效掩码：示例 B=1，N=4\n主图 [B,N,3,256,448]；匹配图 [B,N,3,512,896]\n两种尺寸分别从原始RGB缩放；targets不进入模型", 500, 155, 800, 135)
    box(p, "cnn", "②a MatchingPyramid：全部有效视图共享\nImageNet ResNet-18 → 四级侧投影 → 对齐FPN\n输出F4/F8/F16，各64维；负责纹理与定位\n编码器参与训练，非冻结Hero特征", 50, 365, 740, 180)
    box(p, "vit", "②b TrainableFoundation：本次参考图\n原始DA-V2 ViT-B／12块，取第3/6/9/12块\n四张 [B,768,16,28] 上下文图\n负责结构与场景理解；分阶段解冻", 1010, 365, 740, 180)
    box(p, "geometry", "独立几何输入\nK：主图内参\nT_cam_from_rig：阵列→相机\n每域固定 near / far / s\n平移与候选除以 s\n范围来自TRAIN配置，非本帧GT", 40, 620, 380, 240, "orange", 20)
    box(p, "context", "上下文准备／重复使用\n四个独立1×1卷积\n768 → 128 维\n按真实patch中心对齐\n粗层：32×56\n细层：64×112", 1380, 620, 380, 240, "blue", 21)
    box(p, "candidates", "③ 全范围粗候选：set_range + depth_lattice\n每个stride-8像素沿射线布置96个对数深度候选\n候选位置 [B,96,32,56]，采用工作单位", 500, 610, 800, 140, "purple")
    box(p, "observe", "④ _observe + CandidateViewAttention\n候选精确投影 → 标尺选择特征尺度 → 采样源特征\n成对特征MLP + 4头源视图注意力 + 无匹配槽位\n四目时只计算3个源；几何无效源被屏蔽", 500, 825, 800, 180, "purple")
    box(p, "fusion", "⑤ 候选解释：_tokens_from_source + CandidateScaleSpace\n66→64证据编码 + 深度位置编码 + 四层上下文读取\n在32×56／16×28／8×14三个空间尺度交换候选信息\n保留全部96个深度格；得到 [B,64,96,32,56]", 500, 1080, 800, 180, "purple", 21)
    box(p, "transport", "⑥ 粗预测 + SubrayTransport\n分数头与有界连续偏移 → 96项候选概率\n先传递全部概率到64×112细网格，再选4个峰\n每峰按标尺步长取3点；共12个候选，并继承粗特征", 500, 1335, 800, 185, "purple")
    box(p, "fine", "⑦ 细层重新取证与选面：_observe → _tokens → _prediction\n12个候选重新精确投影，复用④的注意力与⑤的上下文读取\n加上1×1×1投影后的继承特征，再做通道LayerNorm\n共享粗层分数/偏移头；取最高分候选，输出连续深度\n细层不再执行CandidateScaleSpace", 500, 1595, 800, 205, "purple", 21)
    box(p, "upsample", "⑧ _upsample：学习式4倍凸组合上采样\n选中候选特征 → Conv3×3 → GELU → Conv1×1\n预测9邻域×16子像素权重，将64×112恢复至256×448", 500, 1875, 800, 145, "purple", 21)
    box(p, "output", "⑨ 单参考深度输出 [B,1,256,448]\n导出时 depth × s，恢复该数据域的原始长度单位\n同时给出 confidence / valid / support\n当前 unknown=true，observed=false，state=2\n表示深度假设；尚未认证为已观测表面", 500, 2090, 800, 185, "green", 21)
    box(p, "roles", "当前相机角色\n所有相机共用CNN参数\n一次forward指定一个参考相机\nViT只处理该参考图\n请求其他参考要另做前向\n当前入口未自动共享多参考编码", 40, 1080, 380, 245, "gray", 20)
    box(p, "training", "训练标签另走监督支路\nGT深度与可见对应\n只进入损失函数\n不会成为推理输入\n完整训练流程见第4页", 40, 1600, 380, 220, "gray", 21)
    box(p, "shape", "尺寸读法\nB：一批图组数量\nN：输入相机数量\nC：特征通道数\nD：候选深度数量\nH×W：当前空间网格", 1380, 1080, 380, 245, "gray", 21)
    box(p, "disabled", "本图不含的历史模块\n无完整Hero融合ViT／DPT\n无旧树头、GN后端或tele输入\n场景token分支关闭\n独立细头与峰内平均关闭\n可选浮点重排加速关闭", 1380, 1600, 380, 245, "gray", 20)
    wire(p, "input", "cnn", [(690, 290), (690, 325), (420, 325), (420, 365)])
    wire(p, "input", "vit", [(1110, 290), (1110, 325), (1380, 325), (1380, 365)])
    wire(p, "cnn", "observe", [(790, 455), (820, 455), (820, 570), (460, 570), (460, 905), (500, 905)])
    wire(p, "cnn", "fine", [(790, 475), (840, 475), (840, 585), (445, 585), (445, 1700), (500, 1700)])
    wire(p, "vit", "context", [(1570, 545), (1570, 620)])
    wire(p, "context", "fusion", [(1380, 740), (1340, 740), (1340, 1170), (1300, 1170)])
    wire(p, "context", "fine", [(1380, 775), (1360, 775), (1360, 1700), (1300, 1700)])
    wire(p, "geometry", "candidates", [(420, 680), (500, 680)], "orange")
    wire(p, "geometry", "observe", [(420, 800), (430, 800), (430, 955), (500, 955)], "orange")
    for a, b in [("candidates", "observe"), ("observe", "fusion"), ("fusion", "transport"),
                 ("transport", "fine"), ("fine", "upsample"), ("upsample", "output")]:
        p.edge(a, b, "purple")
    return p


def encoders():
    p = Page("02 两条编码支路与每层尺寸", 2000, 1730)
    heading(p, "编码阶段：纹理支路与上下文支路", "结构来自当前 encoders.py｜下列形状按 B=1、N=4 展示；每台相机使用同一套权重")
    box(p, "match_input", "匹配图：合并B和N\n[4,3,512,896]\n掩码置零 → ImageNet归一化", 40, 170, 470, 140)
    box(p, "stem", "ResNet-18 stem\nConv7×7，stride2，3→64 → BN → ReLU\nMaxPool3×3，stride2\n[4,64,128,224]", 40, 380, 470, 175, size=21)
    labels = [
        ("c2", "layer1：2个BasicBlock\n通道64，首块stride1\n[4,64,128,224]", 625),
        ("c3", "layer2：2个BasicBlock\n64→128，首块stride2\n[4,128,64,112]", 840),
        ("c4", "layer3：2个BasicBlock\n128→256，首块stride2\n[4,256,32,56]", 1055),
        ("c5", "layer4：2个BasicBlock\n256→512，首块stride2\n[4,512,16,28]", 1270),
    ]
    previous = "stem"
    for key, label, y in labels:
        box(p, key, label, 40, y, 470, 145)
        p.edge(previous, key)
        previous = key
    box(p, "f4", "F4：侧投影64→64 + 对齐的上层结果\nDW Conv3×3 → PW Conv1×1 → GN8 → GELU\n输出 [B,N,64,128,224]，匹配图stride4", 620, 625, 610, 145, size=20)
    box(p, "f8", "F8：侧投影128→64 + 对齐的上层结果\nDW Conv3×3 → PW Conv1×1 → GN8 → GELU\n输出 [B,N,64,64,112]，匹配图stride8", 620, 840, 610, 145, size=20)
    box(p, "f16", "F16：侧投影256→64 + 对齐的上层结果\nDW Conv3×3 → PW Conv1×1 → GN8 → GELU\n输出 [B,N,64,32,56]，匹配图stride16", 620, 1055, 610, 145, size=20)
    box(p, "f32", "最深侧投影：Conv1×1，512→64\n[4,64,16,28]，匹配图stride32\n参与自顶向下融合，不作为第4张输出图", 620, 1270, 610, 145, size=20)
    p.edge("match_input", "stem")
    for left, right, y in [("c2", "f4", 697), ("c3", "f8", 912), ("c4", "f16", 1127), ("c5", "f32", 1342)]:
        wire(p, left, right, [(510, y), (620, y)])
    for a, b, start, end in [("f32", "f16", 1270, 1200), ("f16", "f8", 1055, 985), ("f8", "f4", 840, 770)]:
        wire(p, a, b, [(925, start), (925, end)])
    box(p, "cnn_note", "BasicBlock：Conv3×3 → BN → ReLU → Conv3×3 → BN → 加捷径 → ReLU\n通道/步长变化时捷径用1×1卷积；BN运行统计冻结，仿射参数可训练\nFPN向上传递的是平滑前特征；DW为逐通道卷积，PW为1×1通道混合\n匹配格点中心按标准ResNet推导，不套用旧Hero匹配支路的中心偏移", 40, 1500, 1190, 175, "gray", 21)
    box(p, "ref_input", "参考主图 [B,3,256,448]\n掩码置零 → ImageNet归一化\n按14/16缩放为 [B,3,224,392]", 1320, 170, 640, 140)
    box(p, "patch", "ViT-B patch嵌入\nConv14×14，stride14，3→768\n16×28＝448个patch + 1个CLS\n加位置编码 → [B,449,768]", 1320, 380, 640, 175)
    box(p, "vit_blocks", "12个Transformer Block，通道768\n每块：LayerNorm → 12头自注意力\nLayerScale → 加残差\nLayerNorm → Linear768→3072 → GELU\nLinear3072→768 → LayerScale → 加残差\n在第3、6、9、12块读取特征", 1320, 625, 640, 245)
    box(p, "four_raw", "共享末端LayerNorm → 丢弃CLS\n四份 [B,768,16,28]\n它们的空间尺寸相同，语义深度不同\n不是四张不同分辨率的图", 1320, 950, 640, 175)
    box(p, "adapters", "四个独立raw_adapters\n各Conv1×1：768→128\n四份 [B,128,16,28]\n粗/细阶段按主图patch中心7.5+16j对齐", 1320, 1210, 640, 175, size=21)
    box(p, "vit_note", "权重起点：原始Depth Anything V2 Base\n阶段1：冻结ViT；阶段2：末4块及末端LN可训\n阶段3/4：全ViT可训\n本支路没有执行DA-V2的DPT深度头\n可训练块启用激活重计算，只影响训练内存", 1320, 1470, 640, 205, "gray", 21)
    for a, b in [("ref_input", "patch"), ("patch", "vit_blocks"), ("vit_blocks", "four_raw"), ("four_raw", "adapters")]:
        p.edge(a, b)
    return p


def matching_fusion():
    p = Page("03 标尺匹配与两种注意力", 2200, 2050)
    heading(p, "匹配与融合：标尺管取证，图像管证据", "本页的模块在粗层与细层各调用一次并共享参数；粗D=96、32×56，细D=12、64×112")
    box(p, "start", "输入：候选深度Z、参考像素网格、K/T、视图/像素掩码、匹配FPN与四层ViT上下文\n所有候选均为待判断假设；投影在图内不等于无遮挡，也不等于存在真实表面", 40, 155, 2120, 110, "orange", 23)
    box(p, "unproject", "① unproject：反投影\n参考像素 + 候选Z → 阵列点X\n同时得到射线方向 dX/dZ\n候选区间半宽为 ΔZ\nZ是光轴深度，不是射线距离", 40, 355, 470, 205, "orange", 21)
    box(p, "project", "② project：精确重投影\nX → 每台源相机的像素u、v\n计算投影导数J\ng = J·dX/dZ\n屏蔽图外、负深度、缺图与退化源", 590, 355, 470, 205, "orange", 21)
    box(p, "route", "③ 标尺尺度路由\n计算横向/纵向/深度跨度的投影\n最大跨度p决定F4/F8/F16权重\n目标约4个特征像素\n实现采样三层，相邻最多两层权重非零", 1140, 355, 470, 205, "orange", 20)
    box(p, "samples", "④ grid_sample + L2归一化\n参考描述子固定读取F4\n源描述子按路由权重混合\n四目只算3个源，每源64通道\n源 [B,3,64,D,H,W]", 1690, 355, 470, 205, "blue", 21)
    for a, b in [("unproject", "project"), ("project", "route"), ("route", "samples")]:
        p.edge(a, b, "orange")
    wire(p, "start", "unproject", [(275, 265), (275, 355)], "orange")
    box(p, "reference", "⑤a 参考query与拒绝分支\nf_ref：64维\nLinear64→64：Q，拆成4头×16维\nLinear64→4：每头的null分数\nQ不随候选深度改变", 40, 720, 470, 210, "purple", 21)
    box(p, "keys", "⑤b 源key\n在当前候选投影处取f_src：64维\nLinear64→64：K，拆成4头×16维\n候选改变 → 取样位置与源key改变\n分数的外观部分：Q·K / 4", 590, 720, 470, 210, "purple", 21)
    box(p, "pair", "⑤c 成对描述：拼成139维\n逐通道乘积：64\n逐通道绝对差：64\n8组乘积求和：8\n标尺方向2 + log(1+|g|ΔZ)：1", 1140, 720, 470, 210, "purple", 21)
    box(p, "mlp", "⑤d pair + reliability\nMLP：139→64 → GELU →64\n输出64维value\nLinear64→4：可靠性分数\n它与Q·K共同决定各头源权重", 1690, 720, 470, 210, "purple", 21)
    box(p, "weights", "⑥ 在源视图维做masked softmax\n每头：3个源分数 + 1个null分数\n无效源的权重为0，null始终可选\n不把拒绝的概率重新分给剩余源\n权重用于value聚合和证据质量统计", 590, 1050, 470, 215, "purple", 20)
    box(p, "pooled", "⑦ 聚合得到三个量\npooled：4头value加权合并，64维\nnull_mass：4头无匹配权重均值，1维\ninformation：按源权重加权的 |g|²，取log1p后为1维\n合计66维；information是几何敏感度，不是纹理置信度", 1140, 1050, 1020, 215, "purple", 22)
    wire(p, "samples", "keys", [(1850, 560), (1850, 625), (825, 625), (825, 720)])
    wire(p, "samples", "pair", [(2020, 560), (2020, 650), (1375, 650), (1375, 720)])
    wire(p, "samples", "reference", [(1730, 560), (1730, 600), (275, 600), (275, 720)])
    p.edge("pair", "mlp", "purple")
    p.edge("keys", "weights", "purple")
    wire(p, "reference", "weights", [(510, 825), (550, 825), (550, 1157), (590, 1157)], "purple")
    wire(p, "mlp", "weights", [(1925, 930), (1925, 985), (900, 985), (900, 1050)], "purple")
    p.edge("weights", "pooled", "purple")
    wire(p, "mlp", "pooled", [(2080, 930), (2080, 1050)], "purple")
    box(p, "position", "⑧b 深度位置编码\n将 log Z 在 near/far 之间归一化为p\np、sin(πp)、cos(πp)共3维\nLinear3→64 → GELU → Linear64→64", 590, 1350, 470, 170, "orange", 20)
    box(p, "evidence", "⑧a evidence：Conv3D1×1×1 66→64 → GELU → Conv3D1×1×1 64→64\n加上⑧b深度位置编码，形成候选token [B,64,D,H,W]", 1140, 1350, 1020, 170, "purple", 22)
    p.edge("pooled", "evidence", "purple")
    p.edge("position", "evidence", "orange")
    box(p, "bank", "⑨ 四层参考图上下文 bank\n每层128维，按当前网格精确对齐\n[B,128,4,H,W]\n同一像素的四个层级作为key/value\n这一步是选编码层，不是选相机", 40, 1650, 600, 245, "blue", 22)
    box(p, "query", "⑩ 候选query读取上下文\nraw_query：Conv3D1×1×1 64→128\nquery与四层bank点积 / √128\nsoftmax沿4个编码层归一化\n不同候选可以选择不同层级", 760, 1650, 550, 245, "purple", 22)
    box(p, "fused", "⑪ 四层加权求和 → 128维\nraw_output：Conv3D1×1×1 128→64\n与原候选token相加 → 通道LayerNorm → 掩码\n粗输出 [B,64,96,32,56]\n细输出 [B,64,12,64,112]", 1430, 1650, 730, 245, "purple", 22)
    p.edge("bank", "query")
    p.edge("query", "fused", "purple")
    wire(p, "evidence", "query", [(1650, 1520), (1650, 1585), (1035, 1585), (1035, 1650)], "purple")
    box(p, "note", "所有候选中心精确重投影；J仅用于尺度、注意力几何条件和后续步长。无匹配槽位由最终任务间接学习，当前没有独立null监督头。", 40, 1940, 2120, 70, "text", 22)
    return p


def decoding():
    p = Page("04 空间交互 细候选 解码与监督", 2100, 2250)
    heading(p, "从候选到深度：先保留，再细格取证，最后连续定位", "主图256×448｜候选体维度 [B,C,D,H,W]｜粗细两级共享评分与偏移参数；细层不执行三维空间卷积")
    box(p, "space", "① CandidateScaleSpace：只在粗层执行\n输入 F0=[B,64,96,32,56]\n带掩码平均池化(1,2,2)：F1=16×28，F2=8×14\n三个尺度各做 DW Conv3D3×3×3 → GELU → PW Conv3D1×1×1\n低尺度消息逐级双线性返回，加到原尺度消息\n外部再做 F0 + 消息；96个深度格始终不降采样", 40, 175, 920, 270, "purple", 22)
    box(p, "heads", "② 粗 _prediction：两个1×1×1卷积头\nscore：64→1，再加 4×源加权余弦相似度\noffset：64→1，经tanh得到有界的区间内修正\nsoftmax在96个候选之间生成概率\n输出各候选连续深度与特征；最高分深度仅作中间诊断\n默认相同两个头还用于细层", 1120, 175, 940, 270, "purple", 22)
    p.edge("space", "heads", "purple")
    box(p, "transport", "③ SubrayTransport + transport_proposals：先传全部概率，再选峰\n在64×112细网格，对同一个深度刻度混合4个粗邻居的概率；连续深度/特征用条件权重继承\n在细网格选4个峰（当前仅0.1工作单位分离，不保证4个不同真实表面）\n每峰步长 = max(粗刻度半宽, 0.5 / 最大有效投影导数)，再限幅到[0.005,100]工作单位\n每峰取 -步长／中心／+步长，共12个候选；不对跨峰深度作一次全局平均", 40, 560, 2020, 250, "purple", 23)
    wire(p, "heads", "transport", [(1590, 445), (1590, 560)], "purple")
    wire(p, "space", "transport", [(500, 445), (500, 560)], "purple")
    box(p, "fine_observe", "④ 细层：12候选，每位置64维\n在64×112网格重新精确投影\n复用第3页的标尺路由、源注意力和四层上下文读取\n得到细层新证据 [B,64,12,64,112]\n不是简单把粗深度放大", 40, 940, 920, 220, "purple", 23)
    box(p, "inherit", "⑤ 继承粗层信息，但允许新证据修改\n只提取选中4个深度刻度对应的64维粗特征\n每峰复制给三个局部候选\nfine_inherit：Conv3D1×1×1 64→64\n加到细层新证据 → 通道LayerNorm → 参考有效掩码", 1120, 940, 940, 220, "purple", 22)
    wire(p, "transport", "fine_observe", [(500, 810), (500, 940)], "purple")
    wire(p, "transport", "inherit", [(1590, 810), (1590, 940)], "purple")
    p.edge("fine_observe", "inherit", "purple")
    box(p, "fine_heads", "⑥ 共享score / offset头：12个候选重新评分\n分数 = learned_score + 4×加权余弦相似度\n偏移受各候选局部区间限制\nargmax选最高分候选，取其连续修正后的Z\n深度与候选置信度 [B,1,64,112]", 40, 1280, 920, 220, "purple", 23)
    wire(p, "inherit", "fine_heads", [(1590, 1160), (1590, 1220), (500, 1220), (500, 1280)], "purple")
    box(p, "upsample", "⑦ 学习式凸组合上采样\n取胜出候选特征 [B,64,64,112]\nConv2D3×3 64→64 → GELU → Conv2D1×1 64→144\n144 = 3×3邻居 × 4×4子像素；对9邻居做masked softmax\n相同凸权重混合深度与confidence，恢复256×448", 1120, 1280, 940, 220, "purple", 22)
    p.edge("fine_heads", "upsample", "purple")
    box(p, "outputs", "⑧ 输出合同与单位恢复\ndepth：模型内为工作单位，导出时乘该域s\nvalid：参考RGB有效性；confidence：候选softmax及上采样结果\nsupport：最近细格胜出候选的几何可采样源数\nunknown全true／observed全false／state全2\n不直接输出法线、网格、占据或已认证表面\n参考有效但邻域没有深度时，用√(near×far)作先验回退", 40, 1650, 920, 320, "green", 21)
    wire(p, "upsample", "outputs", [(1590, 1500), (1590, 1570), (500, 1570), (500, 1650)], "green")
    box(p, "loss", "独立训练支路：depth_objective，不进入推理\n全图：log深度L1 + 0.2×相对误差 + 0.05×log梯度误差\n粗/细各：0.25×最近候选分类 + 0.2×可达区间定位\n　　　　 + 0.1×候选概率加权距离风险\n可见对应：双方GT深度认证，9个标尺扰动候选分类\n对应损失权重阶段1/2/3为0.2/0.1/0.05\nCARLA及MVS-Synth配置关闭对应项；无独立null/法线损失", 1120, 1650, 940, 320, "gray", 21)
    wire(p, "fine_heads", "loss", [(960, 1390), (1010, 1390), (1010, 1810), (1120, 1810)], "gray", True)
    box(p, "note", "当前启用的计算减法：compact_sources跳过恒无效的参考源；spatial_layout仅整理3D卷积内存布局，不新增网络层。\n当前关闭：场景节点、独立细头、峰内平均、候选相邻格排除及可选浮点重排加速；不把这些历史实验画进主线。\n本图对应train_latest_20261025.json及同构公共模型，绘制日期为北京时间2026-10-26。", 40, 2070, 2020, 130, "text", 23)
    return p


def save_editable(pages, destination):
    """固定连线端口，使draw.io与SVG中的折线路径一致。"""
    document = ET.Element("mxfile", host="app.diagrams.net", pages=str(len(pages)))
    for number, page in enumerate(pages):
        diagram = ET.SubElement(document, "diagram", id=f"ruler-v2-{number}", name=page.name)
        model = ET.SubElement(diagram, "mxGraphModel", page="1", pageWidth=str(page.width),
                              pageHeight=str(page.height), grid="1", gridSize="10")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for key, node in page.nodes.items():
            fill, stroke = COLORS[node["color"]]
            style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                     f"fontSize={node['size']};fontFamily=Microsoft YaHei;spacing=8;strokeWidth=2;")
            cell = ET.SubElement(root, "mxCell", id=key, parent="1", vertex="1", style=style,
                                 value="<br>".join(html.escape(line) for line in node["label"].splitlines()))
            ET.SubElement(cell, "mxGeometry", **{k: str(node[k]) for k in ("x", "y", "width", "height")},
                          attrib={"as": "geometry"})
        for index, edge in enumerate(page.edges):
            points = page.path(edge)
            source, target = page.nodes[edge["source"]], page.nodes[edge["target"]]
            ports = {
                "exitX": (points[0][0] - source["x"]) / source["width"],
                "exitY": (points[0][1] - source["y"]) / source["height"],
                "entryX": (points[-1][0] - target["x"]) / target["width"],
                "entryY": (points[-1][1] - target["y"]) / target["height"],
            }
            style = (f"edgeStyle=orthogonalEdgeStyle;endArrow=block;strokeColor={COLORS[edge['color']][1]};"
                     f"strokeWidth=2;dashed={int(edge['dashed'])};exitPerimeter=0;entryPerimeter=0;"
                     + "".join(f"{key}={value};" for key, value in ports.items()))
            cell = ET.SubElement(root, "mxCell", id=f"edge-{index}", parent="1", edge="1",
                                 source=edge["source"], target=edge["target"], style=style)
            geometry = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
            array = ET.SubElement(geometry, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))
    ET.indent(document)
    ET.ElementTree(document).write(destination, encoding="utf-8", xml_declaration=True)


def validate(pages, font):
    """检查节点、文本边界和引用；不代替最终图像的视觉检查。"""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    properties = FontProperties(fname=str(font))
    for page in pages:
        for key, node in page.nodes.items():
            assert 0 <= node["x"] and node["x"] + node["width"] <= page.width, (page.name, key)
            assert 0 <= node["y"] and node["y"] + node["height"] <= page.height, (page.name, key)
            lines = node["label"].splitlines()
            assert len(lines) * node["size"] * 1.5 <= node["height"] + 8, (page.name, key, "height")
            for line in lines:
                text = TextPath((0, 0), line, size=node["size"], prop=properties)
                assert text.get_extents().width < node["width"] - 12, (page.name, key, line, "width")
        for edge in page.edges:
            assert edge["source"] in page.nodes and edge["target"] in page.nodes


def main():
    config_file = ROOT / "configs/pretraining/train_latest_20261025.json"
    config = json.loads(config_file.read_text())
    required = {"channels": 64, "raw_channels": 128, "depth_bins": 96, "peak_count": 4,
                "local_kind": "scale_space", "proposal_transport": "bilinear", "compact_sources": True,
                "spatial_layout": True, "use_scene": False, "use_fine_offset": True, "scale_routing": True}
    assert all(config["model"][key] == value for key, value in required.items()), "模型变化后须重新核对图中结构"
    assert config["image_size"] == [256, 448] and config["matching_image_size"] == [512, 896]
    pages = [overview(), encoders(), matching_fusion(), decoding()]
    font = ROOT / "runs/stage_diagnosis_20261015/fonts/SourceHanSansSC-Regular.otf"
    validate(pages, font)
    destination = ROOT / "docs/diagrams"
    artifacts = ROOT / "runs/active/architecture_20261026"
    artifacts.mkdir(parents=True, exist_ok=True)
    save_editable(pages, destination / f"{STEM}.drawio")
    for index, page in enumerate(pages, 1):
        output = artifacts / f"{STEM}_{index:02d}"
        save_visuals(page, output, str(font))
        shutil.copyfile(output.with_suffix(".svg"), destination / output.with_suffix(".svg").name)
    sources = ["configs/pretraining/train_latest_20261025.json", "jar/pretraining/model.py", "jar/pretraining/encoders.py",
               "jar/models/ruler_direct.py", "jar/models/ruler_spatial.py", "jar/models/ruler_proposals.py", "jar/pretraining/losses.py"]
    provenance = {"date_beijing": "2026-10-26", "architecture": config["architecture"],
                  "pages": [page.name for page in pages], "model": config["model"],
                  "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    (artifacts / "diagram_sources.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    audit = Path("/tmp/architecture_shape_audit_20261026.json")
    if audit.exists():
        shutil.copyfile(audit, artifacts / "shape_audit.json")
    audit_code = Path("/tmp/audit_architecture_20261026.py")
    if audit_code.exists():
        shutil.copyfile(audit_code, artifacts / "shape_audit.py")
    print(destination / f"{STEM}.drawio")
    print(artifacts)


if __name__ == "__main__":
    main()
