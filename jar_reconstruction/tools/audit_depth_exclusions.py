"""审计保存预测在超范围/无效深度像素上的表面输出；不修改旧指标或模型。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.config import load_config
from jar.data import build_dataset, collate_frames
from jar.geometry import pixel_grid, unproject
from jar.metrics import render_surface_depth


def depth_categories(depth, valid, roi_mask, near, far):
    return {
        "in_search_and_roi": valid & roi_mask & (depth >= near) & (depth <= far),
        "valid_beyond_search": valid & (depth > far),
        "valid_outside_roi": valid & ~roi_mask,
        "invalid_depth": ~valid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = Path(args.evaluation_dir)
    report = json.loads((folder / "evaluation.json").read_text())
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, "test", load_targets=True)
    manifest = {
        (f["sequence"], str(f["frame_id"])): f
        for f in map(json.loads, Path(cfg["data"]["test_manifest"]).read_text().splitlines())
    }
    totals = {}
    samples = []
    for record in report["samples"]:
        frame = manifest[(record["metadata"]["sequence"], str(record["metadata"]["frame_id"]))]
        cameras = [c for c in frame["cameras"] if "depth" in c.get("targets", {})]
        if not cameras:
            continue
        item = dataset[record["sample"]]
        batch = collate_frames([item])
        inputs, targets = batch["inputs"], batch["targets"]
        with np.load(folder / record["prediction_file"], allow_pickle=False) as arrays:
            scene = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
        for key in ("K", "T_cam_from_rig", "view_mask", "image_mask"):
            torch.testing.assert_close(scene[key], inputs[key][0])
        predicted_depth, predicted = render_surface_depth(scene, inputs)
        depth, valid = targets["depth"][0], targets["depth_valid"][0]
        grid = pixel_grid(*depth.shape[-2:])
        bounds = depth.new_tensor(cfg["model"]["roi"])
        row = {"sample": record["sample"], "frame_id": frame["frame_id"]}
        for camera in cameras:
            n = cfg["camera"]["wide_ids"].index(camera["camera_id"])
            points, _ = unproject(grid, depth[n], inputs["K"][0, n], inputs["T_cam_from_rig"][0, n])
            roi = ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all(-1)
            categories = depth_categories(
                depth[n], valid[n], roi, cfg["model"]["near"], cfg["model"]["far"]
            )
            # 原值仅用于诊断饱和代理，不将其变成天空/空闲标签。与现有深度标签分开统计。
            label = camera["targets"]
            if label.get("depth_encoding", "metric_npy") != "metric_npy":
                raise ValueError("原始饱和诊断目前仅支持 metric_npy")
            raw = torch.from_numpy(np.load(Path(cfg["data"]["root"]) / label["depth"])) * label.get(
                "depth_scale", 1.0
            )
            high = torch.isfinite(raw) & (raw >= label.get("max_valid_depth", 1000.0))
            categories["raw_at_or_above_valid_limit_proxy"] = F.interpolate(
                high[None, None].float(), size=depth.shape[-2:], mode="nearest-exact"
            )[0, 0].bool()
            categories["all_pixels_in_labeled_camera"] = torch.ones_like(valid[n])
            for name, mask in categories.items():
                mask = mask & inputs["image_mask"][0, n]
                count = {
                    "pixels": int(mask.sum()),
                    "predicted_pixels": int((mask & predicted[n]).sum()),
                }
                total = totals.setdefault(name, {"pixels": 0, "predicted_pixels": 0})
                for key, value in count.items():
                    total[key] += value
                    row[name + "_" + key] = value
        samples.append(row)
    result = {
        "evaluation_directory": str(folder.resolve()),
        "frames_with_raw_depth": len(samples),
        "scope": "仅具有原始深度文件的相机；类别可重叠；分母为单像素投影，不是三维点数",
        "sky_caveat": "大于深度有效上限只是饱和/无效代理，未获得语义与原始解码证明，不等价天空真值",
        "totals": totals,
        "samples": samples,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
