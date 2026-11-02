"""将已有评测转成中文 HTML、对照图、分组 CSV 和点云；不运行或修改网络。"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.config import load_config
from jar.data import build_dataset, collate_frames
from jar.geometry import pixel_grid, unproject
from jar.losses import SURFACE, UNKNOWN, target_cloud
from jar.metrics import render_surface_depth


def summary_rows(report):
    groups = [("全部", report["summary"])]
    names = {"complete": "完整四目", "partial": "缺失相机"}
    groups += [
        (names.get(name, name), value)
        for name, value in report.get("by_camera_completeness", {}).items()
    ]
    groups += [
        (f"{name} 台有效相机", value)
        for name, value in report.get("by_valid_camera_count", {}).items()
    ]
    rows = []
    for name, group in groups:
        depth = group["depth_pixel_weighted"]
        metrics = group["metrics_frame_mean"]
        rows.append(
            {
                "分组": name,
                "帧数": group["frame_count"],
                "有深度标签帧数": sum(
                    int(r.get("has_depth_labels", 0) > 0)
                    for r in report["samples"]
                    if name == "全部"
                    or names.get(r["metadata"].get("camera_completeness")) == name
                    or f'{r["metadata"].get("valid_camera_count")} 台有效相机' == name
                ),
                "深度MAE_米": depth["mae_m"],
                "深度RMSE_米": depth["rmse_m"],
                "深度覆盖率": depth["coverage"],
                "表面F分数": metrics.get("surface_fscore"),
                "表面精确率": metrics.get("surface_precision"),
                "表面召回率": metrics.get("surface_recall"),
                "未知查询比例": metrics.get("unknown_fraction"),
                "单帧网络耗时_毫秒": group["latency_ms_mean"],
                "峰值显存_MiB": group["peak_cuda_memory_bytes"] / 2**20,
            }
        )
    return rows


def select_frames(records, count):
    """按输入完整性/标签可用性选第一帧，不按模型误差挑好看的结果。"""
    selected = []
    for complete, labeled in [("complete", True), ("partial", True), ("partial", False)]:
        for record in records:
            if (
                record["metadata"].get("camera_completeness") == complete
                and bool(record.get("has_depth_labels", 0)) == labeled
                and record not in selected
            ):
                selected.append(record)
                break
    selected += [record for record in records if record not in selected]
    return selected[:count]


def inside_roi(points, roi):
    bounds = points.new_tensor(roi)
    return torch.isfinite(points).all(-1) & (
        (points >= bounds[:, 0]) & (points <= bounds[:, 1])
    ).all(-1)


def comparison_maps(scene, batch, roi):
    """复用评测的单像素 z-buffer，不为可视化补洞或把未知像素当零误差。"""
    inputs, targets = batch["inputs"], batch["targets"]
    predicted, pred_valid = render_surface_depth(scene, inputs)
    gt = targets.get("depth", torch.zeros_like(predicted)[None])[0]
    valid = targets.get("depth_valid", torch.zeros_like(gt, dtype=torch.bool)[None])[0].clone()
    valid &= inputs["view_mask"][0, :, None, None] & inputs["image_mask"][0]
    grid = pixel_grid(*gt.shape[-2:])
    for camera in range(len(gt)):
        mask = valid[camera].clone()
        if mask.any():
            points, _ = unproject(
                grid[mask], gt[camera][mask], inputs["K"][0, camera],
                inputs["T_cam_from_rig"][0, camera],
            )
            valid[camera][mask] = inside_roi(points, roi)
    counts = valid.flatten(1).sum(-1)
    camera = int(counts.argmax()) if counts.any() else int(inputs["view_mask"][0].long().argmax())
    overlap = pred_valid[camera] & valid[camera]
    error = torch.full_like(gt[camera], float("nan"))
    error[overlap] = (predicted[camera][overlap] - gt[camera][overlap]).abs()
    return camera, gt[camera], valid[camera], predicted[camera], pred_valid[camera], error


def capped(points, limit=4096):
    if len(points) > limit:
        points = points[torch.linspace(0, len(points) - 1, limit).long()]
    return points.numpy()


def write_ply(path, points):
    """只导出确认的表面点；未知查询仍保留在原 NPZ 和 HTML 的独立图层。"""
    with path.open("w") as stream:
        stream.write(
            "ply\nformat ascii 1.0\ncomment rig x-forward y-left z-up, meters\n"
            f"element vertex {len(points)}\nproperty float x\nproperty float y\n"
            "property float z\nend_header\n"
        )
        np.savetxt(stream, points, fmt="%.6f")


def plot_frame(scene, batch, cfg, destination, error_max):
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent.resolve() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = batch["inputs"]
    roi = cfg["model"]["roi"]
    camera, gt, gt_valid, predicted, pred_valid, error = comparison_maps(scene, batch, roi)
    surface_mask = scene["valid"] & (scene["state"] == SURFACE) & inside_roi(scene["points"], roi)
    surface = scene["points"][surface_mask]
    unknown = scene["points"][(scene["state"] == UNKNOWN) & inside_roi(scene["points"], roi)]
    target, _ = target_cloud(inputs, batch["targets"])
    target = target[inside_roi(target, roi)]
    cloud = {"surface": capped(surface), "target": capped(target), "unknown": capped(unknown)}
    columns = max(4, inputs["rgb"].shape[1])
    fig, axes = plt.subplots(2, columns, figsize=(columns * 4, 6.5), layout="constrained")
    ids = batch["metadata"][0].get("camera_ids", [str(i) for i in range(columns)])
    for index, axis in enumerate(axes[0]):
        if index >= inputs["rgb"].shape[1]:
            axis.set_visible(False)
            continue
        axis.imshow(inputs["rgb"][0, index].permute(1, 2, 0).numpy())
        present = bool(inputs["view_mask"][0, index])
        axis.set_title(f'{ids[index]} | {"present" if present else "MISSING"}')
        axis.set_axis_off()
    lo, hi = cfg["model"]["near"], cfg["model"]["far"]
    image = axes[1, 0].imshow(
        np.ma.masked_where(~gt_valid.numpy(), gt.numpy()),
        vmin=lo, vmax=hi, cmap="viridis", interpolation="nearest",
    )
    gt_title = f"GT optical depth | {ids[camera]}" if gt_valid.any() else "NO DEPTH LABEL"
    axes[1, 0].set_title(gt_title)
    fig.colorbar(image, ax=axes[1, 0], label="meters", shrink=0.7, extend="both")
    for axis, values, valid, title, palette, maximum in [
        (axes[1, 1], predicted, pred_valid, "Predicted depth: enlarged sparse dots", "viridis", hi),
        (axes[1, 2], error, torch.isfinite(error), "Absolute error: labeled overlaps ONLY",
         "magma", error_max),
    ]:
        y, x = torch.where(valid)
        image = axis.scatter(
            x, y, c=values[valid], s=8, cmap=palette,
            vmin=lo if palette == "viridis" else 0, vmax=maximum,
        )
        axis.set_xlim(0, gt.shape[1])
        axis.set_ylim(gt.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="meters", shrink=0.7, extend="max")
    for axis in axes[1, :3]:
        axis.set_facecolor("#172033")
        axis.set_xticks([])
        axis.set_yticks([])
    axis = axes[1, 3]
    for key, color, size, label in [
        ("target", "#a0acb4", 1, "GT (sampled)"),
        ("surface", "#e46919", 15, "Predicted surface"),
    ]:
        points = cloud[key]
        axis.scatter(points[:, 1], points[:, 0], c=color, s=size, label=label)
    axis.set_xlim(roi[1][1], roi[1][0])
    axis.set_ylim(roi[0])
    axis.set_aspect("equal")
    axis.set_title("Top view: surface points, no completion")
    axis.set_xlabel("rig y: left (m)")
    axis.set_ylabel("rig x: forward (m)")
    axis.legend(fontsize=8)
    for axis in axes[1, 4:]:
        axis.set_visible(False)
    metadata = batch["metadata"][0]
    fig.suptitle(
        f"Frame {metadata['frame_id']} | predicted surface: {len(surface)} | "
        f"unknown queries: {int((scene['state'] == UNKNOWN).sum())}/{len(scene['state'])}\n"
        "Blank depth pixels are unpredicted/unlabeled, NOT free space. No hole filling."
    )
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    write_ply(destination.with_suffix(".ply"), surface.numpy())
    return cloud, camera, int(gt_valid.sum()), int((gt_valid & pred_valid).sum())


def display_value(key, value):
    if value is None:
        return "—"
    if key in ("深度覆盖率", "表面F分数", "表面精确率", "表面召回率", "未知查询比例"):
        return f"{value:.3%}"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def build_page(
    directory, output, max_frames=3, samples=None, split=None, summary_only=False, error_max=2
):
    report = json.loads((directory / "evaluation.json").read_text())
    cfg = load_config(directory / "config.json")
    output.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(report)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headings = "".join(f"<th>{html.escape(key)}</th>" for key in rows[0])
    cells = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(display_value(k, v))}</td>" for k, v in row.items())
        + "</tr>"
        for row in rows
    )
    records = report["samples"]
    if samples is not None:
        lookup = {r["sample"]: r for r in records}
        records = [lookup[index] for index in samples]
    else:
        records = select_frames(records, max_frames)
    visual_records = [r for r in records if r.get("prediction_file")] if not summary_only else []
    frames = []
    if visual_records:
        dataset = build_dataset(
            cfg, split or report.get("split", "test"),
            synthetic=report.get("synthetic", False), load_targets=True,
        )
        for record in visual_records:
            item = dataset[record["sample"]]
            for key in ("sequence", "frame_id"):
                if str(item["metadata"].get(key)) != str(record["metadata"].get(key)):
                    raise ValueError("清单与评测帧身份不一致，请确认 --split 与原测试配置")
            batch = collate_frames([item])
            with np.load(directory / record["prediction_file"], allow_pickle=False) as arrays:
                scene = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
            for key in ("K", "T_cam_from_rig", "view_mask", "image_mask"):
                torch.testing.assert_close(scene[key], batch["inputs"][key][0])
            path = output / f'frame_{record["sample"]:09d}.png'
            cloud, camera, labeled, overlap = plot_frame(scene, batch, cfg, path, error_max)
            label = (
                f'帧 {record["metadata"]["frame_id"]} · '
                f'{record["metadata"].get("valid_camera_count", "?")} 台相机'
            )
            frames.append({
                "label": label,
                "sample": record["sample"],
                "camera": item["metadata"].get("camera_ids", [str(i) for i in range(6)])[camera],
                "labeled": labeled,
                "overlap": overlap,
                "image": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
                "png": path.name,
                "ply": path.with_suffix(".ply").name,
                **{key: np.round(value, 5).tolist() for key, value in cloud.items()},
            })
    total = report["summary"]["frame_count"]
    data = {"frames": frames, "roi": cfg["model"]["roi"]}
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "tools/result_viewer.html").read_text()
    table = f"<thead><tr>{headings}</tr></thead><tbody>{cells}</tbody>"
    page = template.replace("__TABLE__", table)
    page = page.replace("__DATA__", serialized)
    page = page.replace("__RUN__", html.escape(str(directory)))
    page = page.replace("__COUNT__", str(total))
    page = page.replace("__THRESHOLD__", str(cfg["eval"]["fscore_threshold"]))
    page = page.replace("__ERROR_MAX__", str(error_max))
    (output / "index.html").write_text(page)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"可视化：{output / 'index.html'}；分组指标：{output / 'metrics.csv'}")
    if not frames:
        print("没有绘制预测：评测时加 --save-predictions 导出 NPZ，或取消 --summary-only。")
    return rows, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="默认在结果目录下创建 visualization")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--samples", type=int, nargs="+", help="指定报告中的 sample 序号，不是 CARLA frame ID")
    parser.add_argument("--split", choices=["test", "val"], help="旧报告未记录 split 时默认 test")
    parser.add_argument("--summary-only", action="store_true", help="只整理指标，不读取图像和标签")
    parser.add_argument("--error-max", type=float, default=2.0, help="误差图色标上限（米），不截断指标")
    args = parser.parse_args()
    if args.max_frames < 1 or args.error_max <= 0:
        parser.error("max-frames 和 error-max 必须为正数")
    torch.set_num_threads(2)
    build_page(
        args.result_dir, args.output or args.result_dir / "visualization",
        args.max_frames, args.samples, args.split, args.summary_only, args.error_max,
    )


if __name__ == "__main__":
    main()
