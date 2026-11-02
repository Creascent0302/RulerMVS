"""依照计算依赖布局：分支、汇合、复用；导出可编辑draw.io和矢量预览。"""

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/diagrams/v6_baselines_20261102"
FONT = "Source Han Sans CN"
PALETTE = {
    "image": ("#EDF3FF", "#4B74B5"),
    "geometry": ("#FFF3DF", "#BC852C"),
    "compute": ("#F1EDFA", "#8264AA"),
    "core": ("#DDF4EF", "#07887E"),
    "output": ("#EAF6EB", "#478456"),
    "muted": ("#F5F6F8", "#8C95A3"),
}


def quote(value):
    return json.dumps(str(value), ensure_ascii=False)


class Flow:
    """节点尺寸随内容变化；Graphviz按依赖关系排布，而非固定行列表格。"""

    def __init__(self, name, title, subtitle, direction="TB"):
        self.name = name
        self.title = title
        self.subtitle = subtitle
        self.direction = direction
        self.nodes = {}
        self.edges = []
        self.ranks = []

    def node(self, key, label, kind="compute", shape="box"):
        self.nodes[key] = dict(label=label, kind=kind, shape=shape)
        return self

    def tensor(self, key, label, kind="image"):
        return self.node(key, label, kind, "box3d")

    def op(self, key, label, kind="compute"):
        return self.node(key, label, kind, "circle")

    def edge(self, source, target, label="", kind="compute", reuse=False, constraint=True):
        self.edges.append(dict(source=source, target=target, label=label,
                               kind=kind, reuse=reuse, constraint=constraint))
        return self

    def chain(self, *keys, kind="compute"):
        for source, target in zip(keys, keys[1:]):
            self.edge(source, target, kind=kind)
        return self

    def rank(self, *keys):
        self.ranks.append(keys)

    def dot(self):
        title = (f'<B>{html.escape(self.title)}</B><BR/>'
                 f'<FONT POINT-SIZE="15">{html.escape(self.subtitle)}</FONT>')
        lines = ["digraph model {",
                 f'graph [rankdir={self.direction}, splines=polyline, ranksep=0.44, '
                 f'nodesep=0.36, pad=0.25, bgcolor="white", outputorder=edgesfirst, '
                 f'fontname="{FONT}", fontsize=25, labelloc=t, labeljust=l, label=<{title}>];',
                 f'node [fontname="{FONT}", fontsize=19, fontcolor="#223447", '
                 'margin="0.16,0.10", penwidth=1.5];',
                 f'edge [fontname="{FONT}", fontsize=13, fontcolor="#4B596D", '
                 'arrowsize=0.7, penwidth=1.7];']
        for key, node in self.nodes.items():
            fill, stroke = PALETTE[node["kind"]]
            style = "rounded,filled" if node["shape"] == "box" else "filled"
            size = 21 if node["shape"] == "circle" else 19
            lines.append(f'{key} [label={quote(node["label"])}, shape={node["shape"]}, '
                         f'style="{style}", fillcolor="{fill}", color="{stroke}", '
                         f'fontsize={size}, penwidth={2.2 if node["kind"] == "core" else 1.5}];')
        for index, edge in enumerate(self.edges):
            assert edge["source"] in self.nodes and edge["target"] in self.nodes
            attrs = dict(id=f"edge_{index}", label=edge["label"],
                         color=PALETTE[edge["kind"]][1],
                         style="dashed" if edge["reuse"] else "solid",
                         constraint="true" if edge["constraint"] else "false")
            args = ",".join(f"{key}={quote(value)}" for key, value in attrs.items())
            lines.append(f'{edge["source"]} -> {edge["target"]} [{args}];')
        for keys in self.ranks:
            lines.append("{ rank=same; " + "; ".join(keys) + "; }")
        return "\n".join(lines + ["}"])


def v6_overview():
    g = Flow("V6-1 整体数据流", "RulerMVS-V6 · 从图像到候选，再到深度",
             "沿实线向下阅读；虚线表示已有特征/质量的复用。一次输出一张参考深度。")
    g.tensor("rgb", "多相机 RGB\nN×3×256×448")
    g.node("cnn", "共享 ResNet-18 + FPN\n同源图放大至512×896", "image")
    g.node("patch", "参考图 patch嵌入\n448个768维 token", "image")
    g.node("bridge", "CNN → token桥\nL1/L2局部子格 → 768维", "image")
    g.op("add", "+", "image")
    g.node("vit", "参考 ViT-B · 12层\n取第3/6/9/12层", "image")
    g.tensor("fpn", "匹配特征\nF4 / F8 / F16 · 各64维")
    g.tensor("bank", "四层上下文\n每层768→128维")
    g.tensor("geo", "K / T / 有效掩码", "geometry")
    g.tensor("bins", "全范围96个深度候选\n网格32×56", "geometry")
    g.node("match", "粗候选匹配\n精确投影 · 标尺条件注意力", "core")
    g.node("context", "候选查询四层上下文\n残差融合 → 64维")
    g.node("coarse", "三尺度候选空间交互\n评分 + 有界深度修正")
    g.node("transport", "概率传到细射线 → 选4峰\n每峰3点，共12个候选", "core")
    g.node("fine", "细候选重新投影匹配\n共享注意力/上下文/评分头")
    g.node("handoff", "粗细候选族交接\n先选族，再在族内定位", "core")
    g.node("up", "4倍凸组合上采样\n9邻居 × 16子像素权重")
    g.tensor("depth", "参考深度 · 256×448\n置信度 / 有效性 / 支持数", "output")
    g.edge("rgb", "cnn", "全部视图", "image")
    g.edge("rgb", "patch", "取参考图", "image")
    g.edge("cnn", "fpn", kind="image")
    g.edge("cnn", "bridge", "参考L1/L2", "image")
    g.edge("bridge", "add", kind="image")
    g.edge("patch", "add", kind="image")
    g.chain("add", "vit", "bank", kind="image")
    g.edge("fpn", "match", kind="image")
    g.edge("geo", "match", "投影与导数", "geometry")
    g.edge("bins", "match", kind="geometry")
    g.edge("match", "context", "候选证据")
    g.edge("bank", "context", kind="image")
    g.edge("context", "coarse", "64×96×32×56")
    g.edge("coarse", "transport", "96项概率 + 连续候选")
    g.edge("transport", "fine", "12候选 + 继承特征\n网格64×112")
    g.edge("fpn", "fine", "再次采样同一FPN", "image", reuse=True, constraint=False)
    g.edge("bank", "fine", "再次读取同一上下文", "image", reuse=True, constraint=False)
    g.edge("transport", "handoff", "父峰质量", "core", reuse=True)
    g.edge("fine", "handoff", "细族质量 + 族内深度")
    g.edge("handoff", "up", "所选族深度 + 特征", "core")
    g.edge("up", "depth", kind="output")
    g.rank("cnn", "patch")
    return g


def v6_encoder():
    g = Flow("V6-2 编码分支与汇合", "V6 · 局部CNN信息如何进入ViT",
             "两条编码分支在patch处相加；图像编码只做一次，结果由粗/细候选重复读取。")
    g.tensor("rgb", "多视图RGB\n256×448")
    g.node("cnn", "共享ResNet-18\nstem → L1 → L2 → L3 → L4", "image")
    g.node("fpn", "侧接1×1 + 自顶向下相加\nDW3×3 → PW1×1 → GN → GELU", "image")
    g.tensor("feature", "F4 / F8 / F16\n各64维；128×224 / 64×112 / 32×56")
    g.node("subcell", "取参考L1/L2，各投影到32维\n每patch采样2×2子格", "image")
    g.node("bridge", "256 → LN → 128 → GELU → 768\n末层零初始化；残差系数0.1", "image")
    g.node("patch", "参考RGB缩至224×392\nConv14×14 / stride14", "image")
    g.op("plus", "+", "image")
    g.node("vit", "加CLS与位置编码\n12层ViT-B · 768维 / 12头", "image")
    g.tensor("layers", "第3 / 6 / 9 / 12层\n各768×16×28")
    g.node("adapt", "四个独立1×1投影\n768 → 128", "image")
    g.tensor("bank", "四层上下文bank\n按patch中心对齐粗/细网格")
    g.tensor("query", "几何/外观候选证据\n64维")
    g.node("read", "64→128的候选查询\n对四层bank做点积softmax")
    g.node("fuse", "加权读取128维 → 64维\n与原候选残差相加 + LN")
    g.tensor("out", "粗/细候选特征", "output")
    g.edge("rgb", "cnn", "放大至512×896", "image")
    g.chain("cnn", "fpn", "feature", kind="image")
    g.edge("cnn", "subcell", "L1:64维 / L2:128维", "image")
    g.chain("subcell", "bridge", "plus", kind="image")
    g.edge("rgb", "patch", "参考图", "image")
    g.chain("patch", "plus", "vit", "layers", "adapt", "bank", kind="image")
    g.edge("bank", "read", "key / value", "image")
    g.edge("query", "read", "query")
    g.chain("read", "fuse", "out")
    g.edge("query", "fuse", "残差", reuse=True)
    return g


def v6_matching():
    g = Flow("V6-3 匹配内部数据流", "V6 · 图像证据与标定怎样共同决定匹配",
             "本模块由粗层96候选、细层12候选共享；所有中心均精确投影。")
    g.tensor("candidate", "像素射线 + 候选Z + K/T", "geometry")
    g.node("project", "反投影到3D → 精确重投影", "geometry")
    g.tensor("pyramid", "多视图FPN · F4/F8/F16")
    g.node("jac", "解析投影导数 g = ∂π/∂Z\n方向2维 + 区间位移1维", "geometry")
    g.node("sample", "投影足迹选择特征尺度\ngrid_sample + L2归一化", "geometry")
    g.tensor("pairs", "参考 / 源描述子\n各64维")
    g.node("qk", "Q/K投影64→64\n4头内容分数 Q·K / 4")
    g.node("pair", "乘积64 + 绝对差64 + 相关8\n拼入3维标尺 → 139维", "core")
    g.node("value", "成对MLP · 139→64→64", "core")
    g.node("reliable", "可靠性投影 · 64→4", "core")
    g.op("add", "+", "core")
    g.node("null", "参考特征 → 4个null分数")
    g.node("softmax", "源视图 + null的masked softmax\n屏蔽几何无效源", "core")
    g.node("pool", "按注意力汇聚\n64维value + null质量 + 几何敏感度", "core")
    g.node("embed", "66→64→64证据编码\n加入3→64→64的log深度编码")
    g.tensor("bank", "参考ViT四层上下文")
    g.node("read", "候选查询四层bank\n128维读取 → 64维残差 + LN")
    g.tensor("out", "64维候选特征\n送空间交互 / 细层评分", "output")
    g.chain("candidate", "project", kind="geometry")
    g.edge("project", "jac", kind="geometry")
    g.edge("project", "sample", "投影坐标与有效掩码", "geometry")
    g.edge("jac", "sample", "投影尺度", "geometry")
    g.edge("pyramid", "sample", kind="image")
    g.edge("sample", "pairs", kind="image")
    g.edge("pairs", "qk")
    g.edge("pairs", "pair", kind="core")
    g.edge("jac", "pair", "方向 + 敏感度", "geometry")
    g.edge("pair", "value", kind="core")
    g.chain("value", "reliable", "add", kind="core")
    g.edge("qk", "add")
    g.edge("pairs", "null", "参考描述子")
    g.edge("add", "softmax", kind="core")
    g.edge("null", "softmax")
    g.edge("softmax", "pool", "源权重 + null质量", "core")
    g.edge("value", "pool", "64维value", "core")
    g.edge("jac", "pool", "加权‖g‖²", "geometry", reuse=True, constraint=False)
    g.chain("pool", "embed", "read", "out")
    g.edge("bank", "read", kind="image")
    return g


def v6_handoff():
    g = Flow("V6-4 候选交接与读出", "V6 · 选哪一族，与族内在哪里，分别计算",
             "粗层信息跨过细匹配保留下来；所有GT仅进入图底部训练支路。")
    g.tensor("coarse", "96项粗概率 + 连续候选", "compute")
    g.node("transport", "同深度刻度概率传到细网格\n选4峰，每峰布置3个局部点", "core")
    g.node("fine", "12个细候选重新取证\n评分 s + 有界深度修正 z")
    g.tensor("parent", "所选父峰质量 p_c\n归一化 / 停止梯度", "core")
    g.node("mass", "每族汇总3点得分\n得到细层族质量 p_f", "core")
    g.node("local", "每族内部softmax\n3个连续深度加权 → μ", "compute")
    g.node("quality", "父/细熵、分歧、null等6个量\n6→16→1 → sigmoid", "core")
    g.node("mix", "粗细族质量融合\nq ∝ p_f^(1−α) · p_c^α", "core")
    g.node("select", "取质量最高的候选族\n深度 = 该族μ", "core")
    g.node("winner", "选中族内最高分点特征\n64维")
    g.node("mask", "Conv3×3 → GELU → Conv1×1\n预测9×16个上采样权重")
    g.node("up", "局部深度凸组合\n64×112 → 256×448")
    g.tensor("depth", "最终参考深度", "output")
    g.tensor("gt", "GT深度 + 有效掩码\n仅监督", "muted")
    g.node("target", "按各族实际μ的误差生成目标\n族分类 + 族风险；保留offset/全图损失", "muted")
    g.edge("coarse", "transport", kind="core")
    g.edge("transport", "parent", "父质量", "core")
    g.edge("transport", "fine", "候选 + 继承特征")
    g.edge("fine", "mass", "族间质量", "core")
    g.edge("fine", "local", "族内定位")
    g.edge("parent", "quality", kind="core")
    g.edge("mass", "quality", kind="core")
    g.edge("quality", "mix", "α · 129参数门", "core")
    g.edge("parent", "mix", "p_c", "core", reuse=True)
    g.edge("mass", "mix", "p_f", "core")
    g.edge("mix", "select", "argmax族编号", "core")
    g.edge("local", "select", "4个族内深度μ")
    g.edge("select", "winner", "族编号")
    g.chain("winner", "mask", "up")
    g.edge("select", "up", "选中深度", reuse=True)
    g.edge("up", "depth", kind="output")
    g.edge("depth", "target", "全图监督", "muted", reuse=True)
    g.edge("gt", "target", kind="muted", reuse=True)
    g.edge("local", "target", "训练读出对齐", "muted", reuse=True, constraint=False)
    return g


def mvs_overview():
    g = Flow("MVSA-1 整体数据流", "MVSAnywhere HERO · 多视图匹配与单目先验汇合",
             "参考ViT与匹配CNN并行；单目特征在Patchifier及第二个ViT多次注入。")
    g.tensor("rgb", "多视图RGB\nN×3×256×448")
    g.node("matching", "共享匹配CNN\n抗混叠ResNet-18 stem + layer1", "image")
    g.node("mono", "参考单目ViT-B · 12层\nDepth Anything V2初始化", "image")
    g.tensor("feat", "16维匹配特征\n64×112")
    g.tensor("shallow", "单目浅层特征\n第1 / 3层")
    g.tensor("deep", "单目深层特征\n第5 / 6 / 9 / 12层")
    g.tensor("calib", "K / T + 输入深度范围", "geometry")
    g.node("warp", "64个log深度候选\n精确warp + 逐源metadata", "geometry")
    g.node("mlp", "46→32→32→2的共享MLP\n匹配分数 + 源权重")
    g.tensor("cost", "源加权代价体\n64×64×112", "compute")
    g.node("patchify", "Cost Volume Patchifier\n卷积缩网格，融合浅层单目特征")
    g.node("fusion", "第二个ViT-B · 12层\n在第3/6/9/12块前加深层单目特征")
    g.node("dpt", "DPT多尺度重组与RefineNet\n四个融合层 → 连续深度logit")
    g.node("restore", "sigmoid映射到输入log深度范围")
    g.tensor("depth", "首遍深度 · 256×448", "output")
    g.node("refine", "可选：预测更新范围，复用图像特征\n再运行上述匹配 → 融合 → 解码", "muted")
    g.tensor("refined", "细化深度（可选）", "output")
    g.edge("rgb", "matching", "全部视图", "image")
    g.edge("rgb", "mono", "参考图", "image")
    g.edge("matching", "feat", kind="image")
    g.edge("mono", "shallow", kind="image")
    g.edge("mono", "deep", kind="image")
    g.edge("feat", "warp", kind="image")
    g.edge("calib", "warp", kind="geometry")
    g.chain("warp", "mlp", "cost", "patchify", "fusion", "dpt", "restore", "depth")
    g.edge("shallow", "patchify", "两级拼接", "image")
    g.edge("deep", "fusion", "四次残差注入", "image")
    g.edge("depth", "refine", "仅使用预测", "muted", reuse=True)
    g.edge("refine", "refined", kind="muted", reuse=True)
    g.rank("matching", "mono")
    return g


def mvs_detail():
    g = Flow("MVSA-2 匹配到深度", "MVSAnywhere · 匹配体如何变成深度",
             "所有维度以256×448共同输入说明；metadata MLP主模型，不是dot消融。")
    g.tensor("image", "参考/源匹配描述子\n各16维 + 内积1维")
    g.tensor("meta", "候选几何metadata · 13维\nmask / 深度 / 射线 / 夹角 / 位姿", "geometry")
    g.op("concat", "拼接")
    g.node("mlp", "每源共享MLP\n46→32→32→2")
    g.node("weight", "源权重logit → softmax")
    g.node("score", "每源候选匹配分数")
    g.op("pool", "Σ")
    g.tensor("volume", "64×64×112代价体", "compute")
    g.tensor("s1", "单目层1\n768→96，反卷积×4")
    g.tensor("s3", "单目层3\n768→192，反卷积×2")
    g.node("p1", "BasicBlock 64→128\n拼浅层96维 → 224→128")
    g.node("p2", "stride2 BasicBlock 128→256\n拼浅层192维 → 448→256")
    g.node("p3", "stride2 BasicBlock 256→768\n448个token + CLS / 位置")
    g.tensor("deep", "单目层5/6/9/12\n各Linear768→768 + ReLU")
    g.node("fusion", "融合ViT-B\n块3/6/9/12前相加，块后取输出")
    g.node("dpt", "四层重组为stride4/8/16/32\n96/192/384/768 → 统一128维")
    g.node("decode", "RefineNet4→1逐级融合\n输出头 → 连续log深度")
    g.tensor("out", "参考深度", "output")
    g.edge("image", "concat", kind="image")
    g.edge("meta", "concat", kind="geometry")
    g.edge("concat", "mlp", "46维")
    g.edge("mlp", "weight")
    g.edge("mlp", "score")
    g.edge("weight", "pool", "w")
    g.edge("score", "pool", "c")
    g.edge("pool", "volume", "Σ w·c")
    g.chain("volume", "p1", "p2", "p3", "fusion", "dpt", "decode", "out")
    g.edge("s1", "p1", kind="image")
    g.edge("s3", "p2", kind="image")
    g.edge("deep", "fusion", "分层注入", "image")
    return g


def former_overview():
    g = Flow("Former-1 整体数据流", "MVSFormer++ · 特征增强后，粗全局推理再逐级收窄",
             "两路图像编码汇合；每级重新投影取证，上一深度决定下一搜索区间。")
    g.tensor("rgb", "多视图RGB\n共同示例256×448")
    g.node("cnn", "共享CNN\n通道8→16→32→64", "image")
    g.node("dino", "冻结DINOv2-B\n第4 / 8 / 12层", "image")
    g.node("sva", "侧路视图注意力 SVA\n参考self / 源对参考cross")
    g.node("adapt", "768→256→128→64\n两次反卷积放大", "image")
    g.op("add", "+", "image")
    g.node("fpn", "FPN多尺度特征\nstride8 / 4 / 2 / 1", "image")
    g.node("fmt", "匹配特征Transformer FMT\nself / cross线性注意力，传播至细尺度")
    g.tensor("geo", "投影矩阵 + 近远范围", "geometry")
    g.node("coarse", "stride8 · 32个全范围候选\nwarp → 视图融合 → CVT全局推理")
    g.node("s2", "stride4 · 16个局部候选\nwarp → 视图融合 → 3D CNN")
    g.node("s3", "stride2 · 8个局部候选\nwarp → 视图融合 → 3D CNN")
    g.node("s4", "stride1 · 4个局部候选\nwarp → 视图融合 → 3D CNN")
    g.tensor("depth", "温度softmax → 深度期望\n256×448", "output")
    g.edge("rgb", "cnn", kind="image")
    g.edge("rgb", "dino", "全部视图", "image")
    g.chain("dino", "sva", "adapt", "add")
    g.edge("cnn", "add", "stride8特征", "image")
    g.chain("add", "fpn", "fmt")
    g.edge("cnn", "fpn", "浅层侧接", "image", reuse=True)
    g.edge("fmt", "coarse", "粗匹配特征")
    g.edge("geo", "coarse", "投影 / 3D位置", "geometry")
    g.edge("coarse", "s2", "深度 → 收窄区间")
    g.edge("s2", "s3", "深度 → 收窄区间")
    g.edge("s3", "s4", "深度 → 收窄区间")
    g.edge("s4", "depth", kind="output")
    for target, text in [("s2", "stride4"), ("s3", "stride2"), ("s4", "stride1")]:
        g.edge("fmt", target, text + "特征", "image", reuse=True, constraint=False)
    return g


def former_detail():
    g = Flow("Former-2 候选体内部", "MVSFormer++ · 粗候选之间如何全局协商",
             "CVT仅用于最粗一级；后续三个尺度采用3D CNN。")
    g.tensor("feat", "FMT增强的参考/源特征\n64维，32×56")
    g.tensor("hyp", "32个逆深度候选 + K/T", "geometry")
    g.node("warp", "精确warp → 8组相关\n每源8×32×32×56")
    g.node("weight", "候选概率熵 → 2D CNN\n每像素源权重")
    g.op("weighted", "Σ")
    g.node("pos", "视锥3D位置归一化\n正弦编码 + 投影", "geometry")
    g.op("add", "+")
    g.node("down", "Conv3D · kernel/stride=(2,4,4)\n8→64 + LayerNorm3D")
    g.tensor("token", "1792个候选token\n64×16×8×14", "compute")
    g.node("cvt", "6层候选体Transformer\n64维 / 4头 / 全局softmax注意力")
    g.node("up", "ConvTranspose3D · (2,4,4)\n64→8，恢复候选网格")
    g.node("score", "1×1×1评分 → softmax\n推理：温度加权深度期望")
    g.tensor("depth", "粗深度 → 下一尺度局部采样", "output")
    g.edge("feat", "warp", kind="image")
    g.edge("hyp", "warp", kind="geometry")
    g.edge("warp", "weight")
    g.edge("warp", "weighted", "相关体")
    g.edge("weight", "weighted", "归一权重")
    g.edge("hyp", "pos", kind="geometry")
    g.edge("pos", "add", kind="geometry")
    g.edge("weighted", "add", "聚合候选体")
    g.chain("add", "down", "token", "cvt", "up", "score", "depth")
    return g


def patch_overview():
    g = Flow("Patch-1 整体数据流", "PatchmatchNet · 局部提出候选，传播并评价",
             "特征逐级供给；深度从粗到细传递。每次迭代均对当前候选重新warp。")
    g.tensor("rgb", "多视图RGB\n共同示例256×448")
    g.node("cnn", "共享轻量CNN + FPN\n自顶向下融合", "image")
    g.tensor("f8", "stride8特征\n64×32×56")
    g.tensor("f4", "stride4特征\n32×64×112")
    g.tensor("f2", "stride2特征\n16×128×224")
    g.tensor("calib", "投影矩阵 + 深度范围", "geometry")
    g.node("init", "48个分层随机逆深度\n粗级首次不传播", "geometry")
    g.node("p3", "粗级PatchMatch · 2次\n第2次：16局部 + 16传播")
    g.node("p2", "中级PatchMatch · 2次\n8局部 + 8传播")
    g.node("p1", "细级PatchMatch · 1次\n8局部，无传播")
    g.node("refine", "RGB引导残差细化\n深度反卷积×2 + RGB卷积")
    g.tensor("out", "全分辨率参考深度\n256×448", "output")
    g.edge("rgb", "cnn", kind="image")
    for feature in ("f8", "f4", "f2"):
        g.edge("cnn", feature, kind="image")
    g.edge("calib", "init", kind="geometry")
    g.edge("init", "p3", kind="geometry")
    g.edge("f8", "p3", kind="image")
    g.edge("f4", "p2", kind="image")
    g.edge("f2", "p1", kind="image")
    g.edge("p3", "p2", "深度 / 视图权重 ×2")
    g.edge("p2", "p1", "深度 / 视图权重 ×2")
    g.edge("p1", "refine", "半分辨率深度")
    g.edge("rgb", "refine", "参考RGB", "image", reuse=True, constraint=False)
    g.edge("refine", "out", kind="output")
    g.rank("f8", "f4", "f2")
    return g


def patch_detail():
    g = Flow("Patch-2 单次更新", "PatchmatchNet · 一次候选更新怎样计算",
             "第一次使用48点随机初始化；图示后续一般迭代。最细末次无传播。")
    g.tensor("depth", "上次深度", "compute")
    g.tensor("ref", "参考特征", "image")
    g.tensor("src", "源特征 + 投影矩阵", "geometry")
    g.node("local", "局部逆深度扰动\n16 / 8 / 8个候选")
    g.node("offset", "参考特征 → 学习传播偏移\n16 / 8 / 0个邻居")
    g.node("prop", "采样邻居的中心深度")
    g.op("concat", "拼接")
    g.node("warp", "逐候选精确warp\n参考/源分组相关")
    g.node("view", "逐源加权融合\n粗级首次学习权重，后续复用")
    g.node("evalweight", "学习9个评价位置\n特征权重 × 深度权重")
    g.node("eval", "SimilarityNet点式卷积\n采样9邻域并加权评价")
    g.node("softmax", "候选softmax → 深度读出\n最细末次用逆深度索引读出")
    g.tensor("next", "更新深度\n继续迭代 / 进入下一尺度", "output")
    g.edge("depth", "local")
    g.edge("ref", "offset", kind="image")
    g.edge("offset", "prop")
    g.edge("depth", "prop", "邻居深度")
    g.edge("local", "concat")
    g.edge("prop", "concat")
    g.edge("concat", "warp", "局部 + 传播候选")
    g.edge("src", "warp", kind="geometry")
    g.edge("ref", "warp", kind="image")
    g.edge("warp", "view")
    g.edge("ref", "evalweight", "空间特征", "image", reuse=True)
    g.edge("concat", "evalweight", "候选深度", reuse=True)
    g.edge("view", "eval", "聚合相关体")
    g.edge("evalweight", "eval", "9邻域与权重")
    g.chain("eval", "softmax", "next")
    return g


def configure_fonts():
    cache = Path(tempfile.gettempdir()) / "rulermvs_flow_fonts"
    cache.mkdir(exist_ok=True)
    config = cache / "fonts.conf"
    fonts = ROOT / "runs/stage_diagnosis_20261015/fonts"
    config.write_text(f'<?xml version="1.0"?>\n<fontconfig>'
                      '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                      f'<dir>{html.escape(str(fonts))}</dir>'
                      f'<cachedir>{cache}</cachedir></fontconfig>')
    env = dict(os.environ, FONTCONFIG_FILE=str(config))
    subprocess.run(["fc-cache", "-f", str(fonts)], env=env, check=True, capture_output=True)
    matched = subprocess.check_output(["fc-match", FONT, "--format=%{family}"], env=env, text=True)
    assert "Source Han Sans" in matched, matched
    return env


def draw_text_width(layout):
    return max((entry["width"] for entry in layout if entry["op"] == "T"), default=20)


def export_drawio(graphs, layouts, destination):
    """把自动布局转换为原生mxCell；模块、算子、数据块、连线文字均可编辑。"""
    document = ET.Element("mxfile", host="app.diagrams.net", pages=str(len(graphs)))
    for page_index, (graph, layout) in enumerate(zip(graphs, layouts)):
        _, _, width, height = map(float, layout["bb"].split(","))
        diagram = ET.SubElement(document, "diagram", id=f"flow-{page_index}", name=graph.name)
        model = ET.SubElement(diagram, "mxGraphModel", page="1", pageWidth=str(width + 36),
                              pageHeight=str(height + 36), grid="0")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")

        def vertex(key, label, x, y, w, h, style):
            cell = ET.SubElement(root, "mxCell", id=key, parent="1", vertex="1",
                                 value=label, style=style)
            ET.SubElement(cell, "mxGeometry", x=str(x + 18), y=str(y + 18),
                          width=str(w), height=str(h), attrib={"as": "geometry"})

        base = f"html=1;fontFamily={FONT};fontColor=#223447;whiteSpace=wrap;spacing=6;"
        title = (f'<b style="font-size:25px">{html.escape(graph.title)}</b><br>'
                 f'<span style="font-size:15px">{html.escape(graph.subtitle)}</span>')
        vertex("heading", title, 0, 0, width, 64, base + "text;strokeColor=none;fillColor=none;align=left;")
        by_id = {obj["_gvid"]: obj for obj in layout["objects"]}
        geometry = {}
        for obj in layout["objects"]:
            key = obj["name"]
            if key not in graph.nodes:
                continue
            node = graph.nodes[key]
            x, y = map(float, obj["pos"].split(","))
            w, h = float(obj["width"]) * 72, float(obj["height"]) * 72
            x, y = x - w / 2, height - y - h / 2
            geometry[key] = (x, y, w, h)
            fill, stroke = PALETTE[node["kind"]]
            shape = {"box": "rounded=1;arcSize=12;", "circle": "ellipse;aspect=fixed;",
                     "box3d": "shape=cube;size=5;direction=south;"}[node["shape"]]
            style = (base + shape + f"fontSize={obj['fontsize']};fillColor={fill};strokeColor={stroke};"
                     f"strokeWidth={obj['penwidth']};")
            label = "<br>".join(html.escape(line) for line in node["label"].splitlines())
            vertex(key, label, x, y, w, h, style)
        for index, edge in enumerate(layout.get("edges", [])):
            source, target = by_id[edge["tail"]]["name"], by_id[edge["head"]]["name"]
            segments = [item["points"] for item in edge["_draw_"] if item["op"] == "b"]
            points = []
            for segment in segments:
                for x, y in segment:
                    point = (float(x), height - float(y))
                    if not points or point != points[-1]:
                        points.append(point)
            end = edge["pos"].split()[0].split(",")
            if end[0] == "e":
                points.append((float(end[1]), height - float(end[2])))
            sx, sy, sw, sh = geometry[source]
            tx, ty, tw, th = geometry[target]
            ports = dict(exitX=(points[0][0] - sx) / sw, exitY=(points[0][1] - sy) / sh,
                         entryX=(points[-1][0] - tx) / tw, entryY=(points[-1][1] - ty) / th)
            style = (f'edgeStyle=none;endArrow=block;endFill=1;strokeColor={edge["color"]};'
                     f'strokeWidth=1.7;dashed={int(edge.get("style") == "dashed")};'
                     'exitPerimeter=0;entryPerimeter=0;'
                     + "".join(f"{key}={value};" for key, value in ports.items()))
            cell = ET.SubElement(root, "mxCell", id=f"edge-{index}", parent="1", edge="1",
                                 source=source, target=target, style=style)
            geom = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
            array = ET.SubElement(geom, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(array, "mxPoint", x=str(x + 18), y=str(y + 18))
            if edge.get("label"):
                x, y = map(float, edge["lp"].split(","))
                w = draw_text_width(edge.get("_ldraw_", [])) + 16
                h = len(edge["label"].splitlines()) * 18 + 6
                value = "<br>".join(html.escape(line) for line in edge["label"].splitlines())
                vertex(f"label-{index}", value, x - w / 2, height - y - h / 2, w, h,
                       base + "text;fontSize=13;strokeColor=none;fillColor=white;spacing=2;")
    ET.indent(document)
    ET.ElementTree(document).write(destination, encoding="utf-8", xml_declaration=True)


def check_layout(graph, layout):
    objects = {obj["name"]: obj for obj in layout["objects"] if obj["name"] in graph.nodes}
    assert set(objects) == set(graph.nodes), graph.name
    for key, obj in objects.items():
        width = float(obj["width"]) * 72
        assert draw_text_width(obj.get("_ldraw_", [])) <= width + 1, (graph.name, key)
    assert len(layout.get("edges", [])) == len(graph.edges), graph.name


def source_files():
    own = [
        "research/rulermvs_v6_20261101/configs/train_public.json",
        "research/rulermvs_v6_20261101/model.py",
        "research/rulermvs_v6_20261101/handoff.py",
        "research/rulermvs_v6_20261101/objective.py",
        "research/rulermvs_v5_20261031/model.py",
        "research/rulermvs_v5_20261031/bridge.py",
        "jar/pretraining/encoders.py", "jar/pretraining/model.py",
        "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py", "jar/models/ruler_spatial.py",
    ]
    official = "official_methods/methods/"
    baseline = {
        "mvsanywhere": ["src/mvsanywhere/experiment_modules/sr_depth_model.py",
                        "src/mvsanywhere/modules/vit_modules.py", "src/mvsanywhere/modules/networks.py",
                        "src/mvsanywhere/modules/depth_anything_blocks.py",
                        "src/mvsanywhere/modules/view_agnostic_feature_volume.py"],
        "mvsformerpp": ["config/mvsformer++.json", "models/networks/DINOv2_mvsformer_model.py",
                        "models/module.py", "models/FMT.py", "models/cost_volume.py"],
        "patchmatchnet": ["models/net.py", "models/patchmatch.py"],
    }
    return own + [f"{official}{method}/upstream/{path}"
                  for method, paths in baseline.items() for path in paths]



def main():
    config = json.loads((ROOT / "research/rulermvs_v6_20261101/configs/train_public.json").read_text())
    assert config["model"]["depth_bins"] == 96 and config["model"]["peak_count"] == 4
    assert config["handoff"]["within_group"] == "mean"
    assert config["handoff"]["upsample_feature"] == "winner"
    groups = {
        "RulerMVS_V6": [v6_overview(), v6_encoder(), v6_matching(), v6_handoff()],
        "MVSAnywhere_HERO": [mvs_overview(), mvs_detail()],
        "MVSFormerPlusPlus": [former_overview(), former_detail()],
        "PatchmatchNet": [patch_overview(), patch_detail()],
    }
    env = configure_fonts()
    DEST.mkdir(parents=True, exist_ok=True)
    all_graphs, all_layouts = [], []
    for stem, graphs in groups.items():
        layouts = []
        for index, graph in enumerate(graphs, 1):
            output = DEST / f"{stem}_{index:02d}"
            dot = graph.dot().encode()
            layout = json.loads(subprocess.check_output(["dot", "-Tjson"], input=dot, env=env))
            check_layout(graph, layout)
            layouts.append(layout)
            for extension in ("svg", "png", "pdf"):
                subprocess.run(["dot", "-T" + extension, "-o", str(output.with_suffix("." + extension))],
                               input=dot, env=env, check=True, capture_output=True)
            print(f"完成 {graph.name}: {layout['bb']}", flush=True)
        export_drawio(graphs, layouts, DEST / f"{stem}.drawio")
        all_graphs.extend(graphs)
        all_layouts.extend(layouts)
    export_drawio(all_graphs, all_layouts, DEST / "V6与三种Baseline_计算图合集.drawio")
    for path in DEST.glob("*.drawio"):
        tree = ET.parse(path)
        for diagram in tree.findall("diagram"):
            cells = diagram.findall("./mxGraphModel/root/mxCell")
            ids = {cell.attrib["id"] for cell in cells}
            assert len(ids) == len(cells)
            for cell in cells:
                if cell.attrib.get("edge") == "1":
                    assert cell.attrib["source"] in ids and cell.attrib["target"] in ids
    manifest = {
        "date_beijing": "2026-11-02", "layout": "按数据依赖展开的分支、汇合、复用；替代旧行列布局",
        "groups": {name: [graph.name for graph in graphs] for name, graphs in groups.items()},
        "checks": "配置断言、布局节点与文字宽度、XML及引用检查；未运行新性能实验",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in source_files()},
    }
    (DEST / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"已重绘 {len(all_graphs)} 页：{DEST}")


if __name__ == "__main__":
    main()
