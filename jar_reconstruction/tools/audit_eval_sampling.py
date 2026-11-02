"""不运行模型：独立核对同查询评测的原始标签采样及平滑区重采样敏感性。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample_raw_depth(depth, uv, image_size):
    """独立 NumPy 实现半像素映射、最近邻与平滑区逆深度双线性插值。"""
    height, width = image_size
    old_h, old_w = depth.shape
    xy = (uv + 0.5) * np.array([old_w / width, old_h / height]) - 0.5
    # PyTorch CPU nearest-exact 使用 float32 scale；半像素平分点附近
    # 不能用 float64 的另一种舍入结果反过来断言原加载器采错。
    scale32 = np.array([old_w / width, old_h / height], dtype=np.float32)
    nearest_xy = np.floor((uv.astype(np.float32) + np.float32(0.5)) * scale32).astype(int)
    nearest = depth[nearest_xy[:, 1], nearest_xy[:, 0]]
    double_xy = np.floor(xy + 0.5).astype(int)
    double_nearest = depth[double_xy[:, 1], double_xy[:, 0]]
    lo = np.floor(xy).astype(int)
    hi = np.ceil(xy).astype(int)
    lo = np.clip(lo, [0, 0], [old_w - 1, old_h - 1])
    hi = np.clip(hi, [0, 0], [old_w - 1, old_h - 1])
    samples = np.stack([
        depth[lo[:, 1], lo[:, 0]], depth[lo[:, 1], hi[:, 0]],
        depth[hi[:, 1], lo[:, 0]], depth[hi[:, 1], hi[:, 0]],
    ], axis=1)
    dx, dy = (xy - np.floor(xy)).T
    weights = np.stack([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy], 1)
    supported = np.isfinite(samples).all(1) & (samples > 0).all(1) & (samples < 1000).all(1)
    minimum = samples.min(1)
    smooth = supported & (samples.max(1) - minimum <= 0.05 + 0.01 * minimum)
    safe = np.where(samples > 0, samples, 1.0)
    aligned = 1.0 / (weights / safe).sum(1)
    return nearest, aligned, smooth, double_nearest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    protocol = json.loads(Path(config["protocol"]).read_text())
    report_dir = Path(config["query_report_dir"])
    report = json.loads((report_dir / "summary.json").read_text())
    anchor = json.loads(Path(report["config"]["anchor_report"]).read_text())
    if anchor["experiment"]["split"] != "dev_complete" or anchor["protocol"] != protocol:
        raise ValueError("仅审计同协议 DEV25 报告。")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("请使用新目录。")
    rows = [json.loads(x) for x in (report_dir / "frames.jsonl").read_text().splitlines()]
    manifest = [json.loads(x) for x in Path(protocol["data"]["manifests"]["dev_complete"]).read_text().splitlines()]
    frames = {f"{f['sequence']}/{f['frame_id']}": f for f in manifest}
    statistics = {}
    differences = []
    matched, eligible, smooth_count = 0, 0, 0
    raw_max_difference = 0.0
    double_changed = 0
    double_max_difference = 0.0
    for row in rows:
        frame = frames[row["sample_id"]]
        camera = next(c for c in frame["cameras"] if c["camera_id"] == "stereo_up")
        label = camera["targets"]
        if label["depth_type"] != "optical_z" or label.get("depth_scale", 1) != 1:
            raise ValueError("本独立核查只支持当前 optical_z、scale=1 的原始标签合同。")
        raw = np.load(Path(protocol["data"]["root"]) / label["depth"]).astype(np.float64)
        uv = np.asarray(row["query_uv"], dtype=np.float64)
        if not np.array_equal(uv, np.round(uv)):
            raise ValueError("当前同查询报告应为整数锚点；亚像素需另行定义。")
        nearest, aligned, smooth, double_nearest = sample_raw_depth(
            raw, uv, protocol["data"]["image_size"]
        )
        truth = np.asarray(row["target_depth_m"], dtype=np.float64)
        mask = np.asarray(row["eligible_gt"], dtype=bool)
        error = np.abs(nearest[mask] - truth[mask])
        raw_max_difference = max(raw_max_difference, error.max(initial=0))
        matched += int((error < 1e-6).sum())
        eligible += int(mask.sum())
        double_error = np.abs(double_nearest[mask] - truth[mask])
        double_changed += int((double_error > 1e-6).sum())
        double_max_difference = max(double_max_difference, double_error.max(initial=0))
        selected = mask & smooth
        smooth_count += int(selected.sum())
        differences.extend(np.abs(nearest[selected] - aligned[selected]).tolist())
        for name, values in row["methods"].items():
            pred = np.asarray(values["depth_m"], dtype=np.float64)
            valid = np.asarray(values["accepted"], dtype=bool)
            accepted = selected & valid & np.isfinite(pred) & (pred > 0)
            counts = statistics.setdefault(name, {
                "accepted": 0, "nearest_error_sum": 0.0, "aligned_error_sum": 0.0,
                "nearest_hits_02": 0, "aligned_hits_02": 0,
            })
            counts["accepted"] += int(accepted.sum())
            for kind, target in (("nearest", nearest), ("aligned", aligned)):
                distance = np.abs(pred[accepted] - target[accepted])
                counts[kind + "_error_sum"] += float(distance.sum())
                counts[kind + "_hits_02"] += int((distance < 0.2).sum())
    for values in statistics.values():
        for kind in ("nearest", "aligned"):
            values[kind + "_MAE_m"] = values[kind + "_error_sum"] / max(values["accepted"], 1)
    delta = np.asarray(differences)
    result = {
        "config": config, "command": [sys.executable, *sys.argv], "frames": len(rows),
        "eligible_queries": eligible, "exact_raw_nearest_matches": matched,
        "max_raw_nearest_difference_m": raw_max_difference,
        "float64_halfway_rounding_sensitivity": {
            "changed_labels": double_changed, "max_difference_m": double_max_difference,
            "note": "独立double舍入仅作敏感性检查，正式GT仍是所有方法共用的CPU float32 nearest-exact。",
        },
        "smooth_eligible_queries": smooth_count,
        "smooth_target_resampling_difference_m": {
            "mean": float(delta.mean()), "median": float(np.median(delta)),
            "p95": float(np.quantile(delta, 0.95)), "max": float(delta.max()),
        },
        "metrics_on_same_smooth_subset": statistics,
        "limitations": [
            "原始深度编码和像素中心约定仍需采集脚本确认；本工具不认证原始真值。",
            "平滑阈值为已有预定义0.05米+1%局部最小深度；未根据模型结果调节。",
            "逆深度插值只作共同平滑子集敏感性检查，不改正式标签、不挑选有利协议。",
            "未运行新模型，未读取TEST；所有预测来自冻结DEV逐查询记录。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
