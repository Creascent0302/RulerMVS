"""从固定TEST分片只读统计配对逐帧差异，不运行模型或重新选择权重。"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"
EVALUATION = OUTPUT / "evaluation"


def read(path):
    return json.loads(Path(path).read_text())


def compare(frames, manifest, region):
    records = []
    sequences = defaultdict(lambda: {"frames": 0, "gt_pixels": 0,
                                      "old_abs_sum": 0., "new_abs_sum": 0.})
    for index, frame in sorted(frames.items()):
        old = frame["methods"]["original_v2"]["depth"][region]
        new = frame["methods"]["flow_best"]["depth"][region]
        if old["gt_pixels"] != new["gt_pixels"]:
            raise RuntimeError("逐帧GT分母不一致")
        count = old["gt_pixels"]
        if count == 0:
            continue
        if old["predicted_pixels"] != count or new["predicted_pixels"] != count:
            raise RuntimeError("存在非完整预测覆盖，不能按常规逐帧MAE统计")
        old_mae, new_mae = old["abs_sum"] / count, new["abs_sum"] / count
        sequence = str(manifest[index]["sequence"])
        records.append((new_mae - old_mae, old_mae, new_mae))
        total = sequences[sequence]
        total["frames"] += 1
        total["gt_pixels"] += count
        total["old_abs_sum"] += old["abs_sum"]
        total["new_abs_sum"] += new["abs_sum"]
    values = np.asarray(records, dtype=np.float64)
    if not len(values):
        return {"frames": 0}
    delta, old, new = values.T
    tolerance = 1e-9
    wins, losses = int((delta < -tolerance).sum()), int((delta > tolerance).sum())
    near_equal = int((np.abs(delta) <= tolerance).sum())
    rows = []
    for name, item in sorted(sequences.items()):
        old_mean = item["old_abs_sum"] / item["gt_pixels"]
        new_mean = item["new_abs_sum"] / item["gt_pixels"]
        rows.append({"sequence": name, "frames": item["frames"], "gt_pixels": item["gt_pixels"],
                     "old_mae": old_mean, "new_mae": new_mean,
                     "delta_new_minus_old": new_mean - old_mean,
                     "relative_change": new_mean / old_mean - 1. if old_mean > 0 else None})
    total_pixels = sum(item["gt_pixels"] for item in sequences.values())
    return {
        "frames": len(values), "wins": wins, "losses": losses, "ties": near_equal,
        "win_fraction": wins / len(values), "loss_fraction": losses / len(values),
        "tie_absolute_tolerance": tolerance,
        "delta_new_minus_old": {
            "mean_frame_weighted": float(delta.mean()), "median": float(np.median(delta)),
            "p10": float(np.quantile(delta, .1)), "p90": float(np.quantile(delta, .9)),
            "min": float(delta.min()), "max": float(delta.max())},
        "relative_improvement_over_one_percent_fraction": float((new < old * .99).mean()),
        "relative_degradation_over_one_percent_fraction": float((new > old * 1.01).mean()),
        "global_pixel_weighted_mae": {
            "old": sum(item["old_abs_sum"] for item in sequences.values()) / total_pixels,
            "new": sum(item["new_abs_sum"] for item in sequences.values()) / total_pixels},
        "sequence_count": len(rows),
        "improved_sequences": sum(row["delta_new_minus_old"] < -tolerance for row in rows),
        "worsened_sequences": sum(row["delta_new_minus_old"] > tolerance for row in rows),
        "sequences": rows,
    }


def main():
    report = {
        "completed": False, "comparison": "flow_best vs original_v2; native TEST",
        "delta_sign": "new MAE - old MAE; negative is improvement",
        "scope": "同帧配对描述统计，不重跑模型、不调参、不按TEST选权重；帧间相关，非统计显著性检验。",
        "regions": "full以及原已登记near；region定义沿用原分片，不改变阈值。",
        "datasets": {},
    }
    for group in ("public", "carla", "carla_other_refs"):
        config = read(EVALUATION / "configs" / f"{group}.json")
        if any(config["methods"][name].get("input_mode") != "native"
               for name in ("flow_best", "original_v2")):
            raise RuntimeError("本分析只接受已登记native输入")
        for domain in config["datasets"]:
            manifest = [json.loads(line) for line in Path(domain["manifests"]["test"]).read_text().splitlines()]
            frames = {}
            for filename in sorted((EVALUATION / group / "full_test" / domain["name"]).glob("part_*.json")):
                data = read(filename)
                if not data["completed"]:
                    raise RuntimeError(f"未完成分片 {filename}")
                for frame in data["frames"]:
                    if frame["index"] in frames:
                        raise RuntimeError("重复帧索引")
                    frames[frame["index"]] = frame
            if sorted(frames) != list(range(len(manifest))):
                raise RuntimeError(f"{domain['name']}帧与manifest未一一对应")
            result = {"units": domain["geometry_units"], "group": group,
                      **{region: compare(frames, manifest, region) for region in ("full", "near")}}
            # 与已有完整汇总再核对，防止写错分母或混入同名历史分片。
            summary = read(EVALUATION / group / "full_test" / "summary.json")
            for region in ("full", "near"):
                for label, method in (("old", "original_v2"), ("new", "flow_best")):
                    expected = summary["datasets"][domain["name"]]["methods"][method]["depth"][region]["MAE"]
                    actual = result[region]["global_pixel_weighted_mae"][label]
                    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                        raise RuntimeError("逐帧累加与全量汇总不符")
            report["datasets"][domain["name"]] = result
    report["completed"] = True
    destination = OUTPUT / "verification" / "per_frame_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, result in report["datasets"].items():
        row = result["full"]
        print(name, result["units"], "frames", row["frames"], "wins/losses", row["wins"], row["losses"],
              "win%", round(100 * row["win_fraction"], 2),
              "median/P90", row["delta_new_minus_old"]["median"], row["delta_new_minus_old"]["p90"],
              "seq", row["improved_sequences"], "/", row["sequence_count"])
    print(destination)


if __name__ == "__main__":
    main()
