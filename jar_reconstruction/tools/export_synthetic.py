#!/usr/bin/env python3
"""导出带标定的解析渲染样例：结果是几何真值，不是模型预测。

在项目目录运行：
    python tools/export_synthetic.py --output artifacts/synthetic_demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jar.synthetic import SyntheticRigDataset


def depth_visualization(depth, valid, near, far):
    """使用固定米制色标，仅用于显示；原始深度另行完整导出。"""
    fraction = np.clip((depth - near) / (far - near), 0, 1)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    palette = np.array(
        [[230, 65, 45], [252, 179, 75], [130, 202, 128], [49, 155, 195], [50, 53, 140]],
        dtype=np.float32,
    )
    colour = np.stack([np.interp(fraction, stops, palette[:, c]) for c in range(3)], -1)
    colour[~valid] = 24
    return np.rint(colour).astype(np.uint8)


def resolve_chinese_font(explicit_path=None):
    """优先使用用户指定或已安装中文字体；缺失时安全回退为字段缩写。"""
    candidates = [explicit_path] if explicit_path else []
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            ImageFont.truetype(str(candidate), 14)
            return str(candidate)
        except OSError:
            continue
    return None


def font(size, path=None):
    try:
        return ImageFont.truetype(path or "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def export(
    output, cameras=4, height=160, width=256, seed=7, depth_near=2.0, depth_far=25.0, font_path=None
):
    if cameras < 2 or height < 16 or width < 16 or depth_near >= depth_far:
        raise ValueError("相机数必须至少为 2，图像尺寸至少为 16，且 depth_near < depth_far")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    sample = SyntheticRigDataset(
        length=1, image_size=(height, width), num_cameras=cameras, seed=seed
    )[0]
    inputs, targets = sample["inputs"], sample["targets"]
    arrays = {key: value.detach().cpu().numpy() for key, value in {**inputs, **targets}.items()}
    np.savez_compressed(output / "calibrated_ground_truth.npz", **arrays)

    palette = np.array(
        [[113, 139, 229], [130, 186, 113], [233, 150, 82], [203, 119, 190], [246, 214, 93]],
        dtype=np.uint8,
    )
    margin, label_width, header, cell_header = 12, 156, 72, 26
    total_width = label_width + cameras * (width + margin) + margin
    row_height = height + cell_header + margin
    canvas = Image.new("RGB", (total_width, header + 3 * row_height + 80), "#111827")
    draw = ImageDraw.Draw(canvas)
    chinese_font = resolve_chinese_font(font_path)
    title_font, body_font = font(18, chinese_font), font(14, chinese_font)
    # 无中文字体时不绘制缺字方框；图内保留字段缩写，完整中文说明在 JSON。
    title = "解析渲染真值：不是模型预测" if chinese_font else "GT / RGB / Z(m) / ID"
    subtitle = (
        "共用世界坐标纹理、已知相机标定、平面与方体的最近交点遮挡。"
        if chinese_font
        else "K / T_cam_from_rig / depth / normals / semantic"
    )
    draw.text((margin, 12), title, font=title_font, fill="#f9fafb")
    draw.text((margin, 42), subtitle, font=body_font, fill="#d1d5db")
    row_labels = ["RGB 图像", "光轴深度", "表面类别"] if chinese_font else ["RGB", "Z(m)", "ID"]
    for row, label in enumerate(row_labels):
        top = header + row * row_height
        draw.text(
            (margin, top + cell_header + height // 2 - 8), label, font=body_font, fill="#f9fafb"
        )
        for camera in range(cameras):
            left = label_width + camera * (width + margin)
            if row == 0:
                tile = (
                    np.rint(arrays["rgb"][camera].transpose(1, 2, 0) * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            elif row == 1:
                tile = depth_visualization(
                    arrays["depth"][camera], arrays["depth_valid"][camera], depth_near, depth_far
                )
            else:
                semantic = arrays["semantic"][camera]
                tile = palette[np.clip(semantic, 0, len(palette) - 1)]
                tile[semantic < 0] = 24
            draw.text(
                (left, top + 3),
                f"相机 {camera}" if chinese_font else f"V{camera}",
                font=body_font,
                fill="#d1d5db",
            )
            canvas.paste(Image.fromarray(tile), (left, top + cell_header))
    footer = header + 3 * row_height
    footnotes = (
        [
            f"显示色标：{depth_near:g}–{depth_far:g} m；仅显示时截断，NPZ 保留原始 float32 米制深度。",
            "类别：蓝=背景平面，绿=地面，橙/紫=方体，黄=细遮挡物。",
            "深度为光轴 Z，不是射线欧氏距离；rig 坐标为 x 前、y 左、z 上。",
        ]
        if chinese_font
        else [
            f"Z: [{depth_near:g}, {depth_far:g}] m / NPZ: float32",
            "ID: 0 / 1 / 2 / 3 / 4",
            "scene_info.json / GT",
        ]
    )
    for offset, note in zip([4, 28, 52], footnotes):
        draw.text((margin, footer + offset), note, font=body_font, fill="#d1d5db")
    canvas.save(output / "ground_truth_contact_sheet.png")

    info = {
        "artifact_type": "解析渲染几何真值",
        "model_predictions": False,
        "generator": "jar.synthetic.SyntheticRigDataset",
        "seed": seed,
        "camera_count": cameras,
        "image_shape": [height, width],
        "depth_units": "米",
        "depth_convention": "各 CV 相机的光轴深度 Z，不是射线欧氏距离",
        "rig_axes": "右手坐标系：x 向前、y 向左、z 向上",
        "camera_axes": "CV 坐标系：x 向右、y 向下、z 向前",
        "extrinsic_convention": "X_cam = T_cam_from_rig @ X_rig，使用齐次列向量",
        "pixel_centers": "整数像素中心；保存的 K 不包含联系图显示缩放",
        "input_keys": list(inputs),
        "target_keys": list(targets),
        "inference_warning": "本 NPZ 的 targets 仅作为标签，禁止放入模型 inputs。这些图像是解析渲染真值，不是模型预测。",
        "contact_sheet_labels": (
            "中文"
            if chinese_font
            else "系统无可用中文字体，图内使用字段缩写；完整中文说明见本 JSON。可用 --font 指定中文字体。"
        ),
        "contact_sheet_rows": ["第一行：RGB 图像", "第二行：米制光轴深度", "第三行：表面类别"],
        "contact_sheet_depth_display_range_m": [depth_near, depth_far],
        "geometry": "两个平面和三个不透明方体，其中包含细长竖直遮挡物；采用射线最近的正向交点。",
        "variation_limit": "修改种子只改变世界坐标中的解析纹理，不产生新的场景几何。",
        "semantic_labels": {
            "0": "背景平面",
            "1": "地面",
            "2": "方体 A",
            "3": "方体 B",
            "4": "细遮挡物",
            "-1": "无命中",
        },
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
    }
    (output / "scene_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示使用说明并退出")
    parser.add_argument("--output", default="artifacts/synthetic_demo", help="输出目录")
    parser.add_argument("--cameras", type=int, default=4, help="相机数量")
    parser.add_argument("--height", type=int, default=160, help="原始图像高度")
    parser.add_argument("--width", type=int, default=256, help="原始图像宽度")
    parser.add_argument("--seed", type=int, default=7, help="世界坐标纹理种子")
    parser.add_argument("--depth-near", type=float, default=2.0, help="深度显示色标下限，单位米")
    parser.add_argument("--depth-far", type=float, default=25.0, help="深度显示色标上限，单位米")
    parser.add_argument(
        "--font", default=None, help="可选中文字体路径；缺失时安全回退为图内字段缩写"
    )
    args = parser.parse_args()
    path = export(
        args.output,
        args.cameras,
        args.height,
        args.width,
        args.seed,
        args.depth_near,
        args.depth_far,
        args.font,
    )
    print(f"已导出带标定的解析渲染真值（不是模型预测）：{path}")


if __name__ == "__main__":
    main()
