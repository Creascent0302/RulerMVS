"""从当前实验配置生成可编辑draw.io及同版SVG/PNG/PDF计算图。"""

import argparse
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": ("#EAF2FF", "#477BC2"),
    "purple": ("#F1ECFF", "#8665BC"),
    "orange": ("#FFF3DF", "#C48836"),
    "green": ("#E8F5EE", "#408766"),
    "gray": ("#F2F3F5", "#8D96A3"),
    "text": ("none", "none"),
}


class Page:
    def __init__(self, name, width=2200, height=1320):
        self.name, self.width, self.height = name, width, height
        self.nodes, self.edges = {}, []

    def node(self, key, label, x, y, width, height, color="blue", size=23):
        self.nodes[key] = dict(
            label=label, x=x, y=y, width=width, height=height, color=color, size=size
        )

    def edge(self, source, target, color="blue", dashed=False, points=None):
        self.edges.append(
            dict(source=source, target=target, color=color, dashed=dashed, points=points)
        )

    def path(self, edge):
        if edge["points"]:
            return edge["points"]
        source, target = self.nodes[edge["source"]], self.nodes[edge["target"]]
        sx, sy = source["x"] + source["width"] / 2, source["y"] + source["height"] / 2
        tx, ty = target["x"] + target["width"] / 2, target["y"] + target["height"] / 2
        if abs(tx - sx) < 100:
            return [(sx, source["y"] + source["height"]), (tx, target["y"])]
        middle = (source["x"] + source["width"] + target["x"]) / 2
        return [(source["x"] + source["width"], sy), (middle, sy), (middle, ty), (target["x"], ty)]


def pages(model):
    width = model["backbone_width"]
    feature = model["feature_dim"]
    blocks = model["deep_encoder"]["stage_blocks"]
    dense = model["dense_decoder"]
    hidden = dense["hidden_dim"]
    bins = model["depth_bins"]
    all_pages = []
    page = Page("01 当前启用的整体架构")
    page.node("title", "MF-ET-JAR：当前实际执行的四目稠密路径", 40, 20, 2120, 65, "text", 37)
    page.node(
        "sub",
        "对应 matching.json / 蓝：图像信息　紫：几何候选与解码　橙：固定标定　灰：本配置关闭",
        40,
        90,
        2120,
        45,
        "text",
        23,
    )
    page.node("input", "同步四目RGB\n[1,4,3,256,448]\n缺失视图与像素掩码", 40, 190, 235, 145)
    page.node(
        "calibration",
        "固定内参和外参\nrig：x前/y左/z上\n相机：x右/y下/z前\n不预测或修改标定",
        40,
        450,
        235,
        180,
        "orange",
    )
    page.node(
        "cnn", f"① 深层本地CNN\n五级共{sum(blocks)}个残差块\n保留边缘与局部纹理", 340, 180, 335, 155
    )
    page.node("fpn", f"多尺度FPN\nstride 4/8/16/32\n各尺度{feature}维特征", 750, 180, 290, 155)
    page.node(
        "vit",
        "② 冻结Hero图像ViT\n12层，取3/6/9/12层\n每层768维\n提供场景上下文",
        340,
        460,
        335,
        180,
    )
    page.node(
        "semantic",
        f"逐层适配768→{feature}\n精确对齐像素中心\n上下文送入解码器\n当前不加入粗匹配特征",
        750,
        460,
        290,
        180,
    )
    page.node(
        "matching",
        "③ 冻结Hero匹配编码器\n完整16维描述子\nstride4，独立匹配通路",
        340,
        760,
        335,
        145,
    )
    page.node(
        "evidence", "精确投影读取描述子\n相似度、分歧与几何\n学习可靠分数增量", 750, 760, 290, 145
    )
    page.node(
        "coarse",
        f"④ 全范围已标定匹配\nstride16：16×28位置\n{bins}个米制深度候选\n分组相关＋3D正则化\n三层代价上下文\n保留无匹配选项",
        1110,
        170,
        390,
        290,
        "purple",
    )
    page.node(
        "packet",
        f"⑤ 完整候选证据包\n分数/概率/有效性/位置\n{bins * 4 + 5}→{hidden}维\n两层空间Transformer",
        1110,
        550,
        390,
        190,
        "purple",
    )
    page.node(
        "dense",
        "⑥ 独立稠密解码头\n32→16→8→4→1\n融合CNN、语义、候选\n继承父层特征和深度编码\n每级输出连续深度\n首级学习深度，后级修正",
        1580,
        170,
        565,
        315,
        "purple",
    )
    page.node(
        "depth",
        "四视图完整深度＋置信度\n[1,4,256,448]，2–80米\n预测有效性单独输出",
        1580,
        600,
        565,
        145,
        "green",
    )
    page.node(
        "verify",
        "⑦ 独立多视图观测验证\n精确投影、纹理、视差\n图像与深度一致性\n输出支持数/observed/unknown",
        1580,
        835,
        565,
        180,
        "green",
    )
    page.node(
        "sparse",
        "可选：原分层稀疏表面头\n四级射线查询与连续定位\n当前关闭；参数仍保留",
        1110,
        920,
        390,
        160,
        "gray",
    )
    page.node(
        "optional",
        "当前关闭的独立开关\n跨层语义增强 / 跨相机交流 / 候选平面运输\n局部几何细化 / 稀疏头 / 长焦输入\n长焦支持原稀疏路径\n当前深编码＋稠密组合暂不支持长焦",
        40,
        1080,
        1000,
        180,
        "gray",
        21,
    )
    page.node(
        "note",
        "标签只进入训练损失与离线评价\n完整深度预测与已观测空间分别输出\n遮挡后方保持unknown",
        1090,
        1140,
        1050,
        130,
        "text",
        23,
    )
    for source, target in [
        ("input", "cnn"),
        ("input", "vit"),
        ("input", "matching"),
        ("cnn", "fpn"),
        ("vit", "semantic"),
        ("matching", "evidence"),
        ("fpn", "coarse"),
        ("coarse", "packet"),
        ("packet", "dense"),
        ("dense", "depth"),
        ("depth", "verify"),
    ]:
        page.edge(source, target, "purple" if source in ("coarse", "packet", "dense") else "blue")
    page.edge("evidence", "coarse", points=[(1040, 832), (1075, 832), (1075, 390), (1110, 390)])
    page.edge("semantic", "dense", points=[(1040, 550), (1530, 550), (1530, 340), (1580, 340)])
    page.edge(
        "fpn",
        "dense",
        points=[(1040, 255), (1065, 255), (1065, 135), (1550, 135), (1550, 215), (1580, 215)],
    )
    page.edge(
        "calibration",
        "coarse",
        "orange",
        points=[(275, 530), (305, 530), (305, 145), (1125, 145), (1125, 170)],
    )
    page.edge(
        "coarse",
        "sparse",
        "gray",
        True,
        points=[(1500, 430), (1535, 430), (1535, 1000), (1500, 1000)],
    )
    page.edge("fpn", "sparse", "gray", True, points=[(1040,285),(1055,285),(1055,1000),(1110,1000)])
    all_pages.append(page)

    page = Page("02 编码器与特征尺寸", 2200, 1160)
    page.node("title", "编码器：保住细节，并独立读取预训练场景信息", 40, 20, 2120, 65, "text", 36)
    page.node(
        "sub",
        "尺寸写作 [批次×相机, 通道, 高, 宽]；当前B=1、N=4，所有相机共享网络权重",
        40,
        90,
        2120,
        45,
        "text",
        23,
    )
    page.node("rgb", "RGB合并批次和相机\n[4,3,256,448]", 40, 205, 310, 105)
    stages = [
        (2, width, 128, 224),
        (4, width * 2, 64, 112),
        (8, width * 3, 32, 56),
        (16, width * 4, 16, 28),
        (32, width * 4, 8, 14),
    ]
    for index, (stride, channels, height, columns) in enumerate(stages):
        x = 390 + index * 350
        page.node(
            f"cnn{stride}",
            f"CNN阶段{index+1}\nstride{stride}，{blocks[index]}个块\n[4,{channels},{height},{columns}]",
            x,
            190,
            300,
            140,
        )
        page.edge("rgb" if index == 0 else f"cnn{stages[index-1][0]}", f"cnn{stride}")
        if stride > 2:
            page.node(
                f"fpn{stride}",
                f"1×1侧向映射＋对齐融合\n3×3平滑\n[4,{feature},{height},{columns}]",
                x,
                450,
                300,
                145,
            )
            page.edge(f"cnn{stride}", f"fpn{stride}")
    for coarse, fine in [(32, 16), (16, 8), (8, 4)]:
        page.edge(
            f"fpn{coarse}",
            f"fpn{fine}",
            points=[
                (page.nodes[f"fpn{coarse}"]["x"] + 150, 595),
                (page.nodes[f"fpn{coarse}"]["x"] + 150, 640),
                (page.nodes[f"fpn{fine}"]["x"] + 150, 640),
                (page.nodes[f"fpn{fine}"]["x"] + 150, 595),
            ],
        )
    page.node(
        "vit",
        "冻结Hero ViT-B\n同一图像取4个中间层\n每层[4,768,16,28]\npatch中心7.5＋16×i",
        40,
        760,
        360,
        185,
    )
    page.node(
        "adapt",
        f"分别投影768→{feature}\n按各CNN格点精确采样\n作为解码上下文\n冻结ViT每帧只计算一次",
        470,
        760,
        440,
        185,
    )
    page.node(
        "desc",
        "冻结的匹配描述子\n[1,4,16,64,112]\n中心1.5＋4×i\n不和CNN格点混用",
        1040,
        760,
        445,
        185,
    )
    page.node(
        "optional",
        "可选跨层语义融合\n各ViT层先768→128\n每个CNN位置选择语义层\n当前matching配置关闭",
        1590,
        760,
        540,
        185,
        "gray",
    )
    page.edge("vit", "adapt")
    page.node(
        "note",
        "CNN的中心为stride×i；图像缩放、ViT patch和官方匹配描述子采用不同采样中心，代码分别处理。",
        40,
        1020,
        2120,
        85,
        "text",
        25,
    )
    all_pages.append(page)

    page = Page("03 稠密解码与诊断位置", 2200, 1200)
    page.node(
        "title", "稠密解码：候选信息进入每一级，深度由粗到细修正", 40, 20, 2120, 65, "text", 36
    )
    page.node(
        "packet",
        f"完整{bins}候选＋无匹配\n{bins*4+5}通道证据包\n1×1投影＋残差块\n{hidden}维候选特征",
        45,
        185,
        325,
        170,
        "purple",
    )
    page.node(
        "attention",
        f"stride16空间Transformer\n{dense['context_layers']}层/{dense['context_heads']}头\n448位置/每相机\n位置含像素、射线与相机中心",
        45,
        450,
        325,
        170,
        "purple",
        21,
    )
    page.edge("packet", "attention", "purple")
    for index, stride in enumerate((32, 16, 8, 4, 1)):
        x = 430 + index * 350
        height, columns = 256 // stride, 448 // stride
        local_hidden = hidden if stride > 1 else dense["full_resolution_dim"]
        label = (
            f"stride{stride}解码\nCNN＋语义＋候选＋父特征\n{dense['blocks_per_stage']}个残差块/{local_hidden}维\n深度[1,4,{height},{columns}]"
            if stride > 1
            else f"完整分辨率解码\n继承stride4＋RGB细节\n{local_hidden}维轻量特征\n深度[1,4,256,448]"
        )
        page.node(f"stage{stride}", label, x, 450, 310, 185, "purple", 21)
        page.node(
            f"head{stride}",
            f"连续深度＋置信度头\n诊断点：解码stride{stride}",
            x,
            785,
            310,
            110,
            "green",
            21,
        )
        page.edge(f"stage{stride}", f"head{stride}", "green")
        if index:
            page.edge(f"stage{(32,16,8,4,1)[index-1]}", f"stage{stride}", "purple")
        else:
            page.edge("attention", f"stage{stride}", "purple")
    page.node(
        "evidence",
        "候选特征在每级对齐读取\nCNN和ViT上下文也按该级读取\n层间传递特征和有界深度编码",
        475,
        195,
        1175,
        145,
    )
    for stride in (32, 16, 8, 4):
        x = page.nodes[f"stage{stride}"]["x"] + 155
        page.edge("evidence", f"stage{stride}", points=[(x, 340), (x, 450)])
    page.node(
        "initial",
        "本配置：首级学习深度\n其后预测深度编码残差\n不直接继承粗深度均值",
        1730,
        190,
        385,
        160,
        "orange",
        21,
    )
    page.node(
        "diagnostic",
        "固定权重干预\n去语义 / 打平候选 / 去空间注意力 / 去RGB / 去代价上下文\n标签纠正候选排序：仅诊断上限，不能作为正常推理结果",
        430,
        990,
        1640,
        155,
        "gray",
        24,
    )
    page.node(
        "ruler",
        "可选局部标定细化\nstride4上9个精确投影候选\n当前关闭；梯度求解尚为独立实验",
        45,
        800,
        325,
        165,
        "gray",
        21,
    )
    all_pages.append(page)
    return all_pages


def save_drawio(all_pages, output):
    root = ET.Element("mxfile", host="app.diagrams.net", pages=str(len(all_pages)))
    for index, page in enumerate(all_pages):
        diagram = ET.SubElement(root, "diagram", id=f"mf-current-{index}", name=page.name)
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            page="1",
            pageWidth=str(page.width),
            pageHeight=str(page.height),
            grid="1",
            gridSize="10",
        )
        cells = ET.SubElement(model, "root")
        ET.SubElement(cells, "mxCell", id="0")
        ET.SubElement(cells, "mxCell", id="1", parent="0")
        for key, node in page.nodes.items():
            fill, stroke = COLORS[node["color"]]
            label = "<br>".join(html.escape(line) for line in node["label"].splitlines())
            style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize={node['size']};fontFamily=Microsoft YaHei;spacing=10;strokeWidth=2;"
            if node["color"] == "gray":
                style += "dashed=1;"
            cell = ET.SubElement(
                cells, "mxCell", id=key, parent="1", vertex="1", value=label, style=style
            )
            ET.SubElement(
                cell,
                "mxGeometry",
                x=str(node["x"]),
                y=str(node["y"]),
                width=str(node["width"]),
                height=str(node["height"]),
                attrib={"as": "geometry"},
            )
        for number, edge in enumerate(page.edges):
            _, stroke = COLORS[edge["color"]]
            cell = ET.SubElement(
                cells,
                "mxCell",
                id=f"edge-{number}",
                parent="1",
                edge="1",
                source=edge["source"],
                target=edge["target"],
                style=f"edgeStyle=orthogonalEdgeStyle;endArrow=block;strokeColor={stroke};strokeWidth=2;dashed={int(edge['dashed'])};",
            )
            geometry = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
            points = page.path(edge)
            array = ET.SubElement(geometry, "Array", attrib={"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def save_visuals(page, stem, font_path):
    font = FontProperties(fname=font_path)
    fig, axis = plt.subplots(figsize=(page.width / 100, page.height / 100))
    axis.set_xlim(0, page.width)
    axis.set_ylim(page.height, 0)
    axis.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page.width}" height="{page.height}" viewBox="0 0 {page.width} {page.height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#637A98"/></marker></defs>',
    ]
    for edge in page.edges:
        points = page.path(edge)
        color = COLORS[edge["color"]][1]
        for first, last in zip(points[:-2], points[1:-1]):
            axis.plot(
                [first[0], last[0]],
                [first[1], last[1]],
                color=color,
                linewidth=1.5,
                linestyle="--" if edge["dashed"] else "-",
            )
        axis.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=15,
                color=color,
                linewidth=1.5,
                linestyle="--" if edge["dashed"] else "-",
            )
        )
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 6"' if edge["dashed"] else ""
        svg.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"{dash}/>'
        )
    for node in page.nodes.values():
        x, y, width, height = node["x"], node["y"], node["width"], node["height"]
        fill, stroke = COLORS[node["color"]]
        if fill != "none":
            axis.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width,
                    height,
                    boxstyle="round,pad=0,rounding_size=12",
                    facecolor=fill,
                    edgecolor=stroke,
                    linewidth=1.4,
                    linestyle="--" if node["color"] == "gray" else "-",
                )
            )
            svg.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
        lines = node["label"].splitlines()
        line_height = node["size"] * 1.5
        start = y + height / 2 - (len(lines) - 1) * line_height / 2
        for index, line in enumerate(lines):
            axis.text(
                x + width / 2,
                start + index * line_height,
                line,
                ha="center",
                va="center",
                fontproperties=font,
                fontsize=node["size"] * 0.72,
                color="#22334A",
            )
            svg.append(
                f'<text x="{x+width/2}" y="{start+index*line_height}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" font-size="{node["size"]}" fill="#22334A">{html.escape(line)}</text>'
            )
    svg.append("</svg>")
    stem.with_suffix(".svg").write_text("\n".join(svg))
    fig.savefig(stem.with_suffix(".png"), dpi=130)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage_diagnosis/current.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    source_config = json.loads(Path(config["own"]["model_config"]).read_text())
    model = source_config["model"]
    # 固定图布局有明确适用范围，避免配置修改后导出一张不真实的旧图。
    inactive = ("feature_enrichment", "feature_exchange", "candidate_transport")
    matching_layout = (
        source_config["data"]["image_size"] == [256, 448]
        and source_config["camera"]["mode"] == "wide4"
        and not model["sparse_decoder"]["enabled"]
        and model["dense_decoder"]["enabled"]
        and model["deep_encoder"]["enabled"]
        and model["deep_encoder"]["foundation"]["enabled"]
        and not model["deep_encoder"]["foundation"].get("fuse_matching", False)
        and model["matching_evidence"]["enabled"]
        and model["cost_context"]["enabled"]
        and model["dense_decoder"]["depth_initialization"] == "learned"
        and not model["dense_decoder"]["calibrated_refinement"]["enabled"]
        and all(not model.get(name, {}).get("enabled", False) for name in inactive)
    )
    if not matching_layout:
        raise ValueError("这份布局对应四目256×448的matching稠密配置；结构开关变化后须更新布局。")
    all_pages = pages(model)
    options = config["diagrams"]
    destination, artifacts = Path(options["output"]), Path(options["artifacts"])
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "source_config.json").write_text(
        json.dumps(source_config, ensure_ascii=False, indent=2) + "\n"
    )
    save_drawio(all_pages, destination / "MF-ET-JAR_当前架构_20261015.drawio")
    for index, page in enumerate(all_pages):
        stem = artifacts / f"architecture_{index+1:02d}"
        save_visuals(page, stem, options["font"])
        (destination / f"MF-ET-JAR_当前架构_{index+1:02d}.svg").write_text(
            stem.with_suffix(".svg").read_text()
        )
    print(destination / "MF-ET-JAR_当前架构_20261015.drawio")
    print(artifacts)


if __name__ == "__main__":
    main()
