"""从固定TEST逐帧统计重组近距比较；不重新推理、不改掩码或距离阈值。"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/active/v2_restart_20261029/evaluation"


def read(path):
    return json.loads(Path(path).read_text())


def total(values):
    result = defaultdict(float)
    for item in values:
        for name, value in item.items():
            result[name] += value
    return dict(result)


def metrics(value):
    count = value["predicted_pixels"]
    if not count:
        return {key: None for key in ("mae", "rmse", "absrel", "bias", "over_error_share")}
    return {"mae": value["abs_sum"] / count,
            "rmse": math.sqrt(value["sq_sum"] / count),
            "absrel": value["abs_rel_sum"] / count,
            "bias": value["signed_sum"] / count,
            "over_error_share": (value["abs_sum"] + value["signed_sum"]) / (2 * value["abs_sum"])
            if value["abs_sum"] else 0.}


def csv_write(path, records):
    names = list(dict.fromkeys(key for record in records for key in record))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def median(values):
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "runs/active/near_evidence_20261030"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    domain_rows, frame_rows, sequence_rows, details = [], [], [], {}
    for group in ("public", "carla", "carla_other_refs"):
        config = read(SOURCE / "configs" / (group + ".json"))
        summary = read(SOURCE / group / "full_test/summary.json")
        own = "v2_restart_public_best" if group == "public" else "v2_restart_carla_best"
        old = "v2_old_public" if group == "public" else "v2_previous_carla"
        baseline = "mvsanywhere_refined"
        for domain in config["datasets"]:
            name = domain["name"]
            manifests = [json.loads(line) for line in Path(domain["manifests"]["test"]).read_text().splitlines()]
            frames = []
            for part in sorted((SOURCE / group / "full_test" / name).glob("part_*.json")):
                frames.extend(read(part)["frames"])
            frames.sort(key=lambda row: row["index"])
            assert [row["index"] for row in frames] == list(range(len(manifests)))
            totals = {method: total(frame["methods"][method]["depth"]["near"] for frame in frames)
                      for method in (own, old, baseline)}
            near = {method: metrics(value) for method, value in totals.items()}
            full = summary["datasets"][name]["methods"]
            near_pixels = int(totals[own]["gt_pixels"])
            full_pixels = full[own]["depth"]["full"]["gt_pixels"]
            assert near_pixels == int(totals[baseline]["gt_pixels"])
            for method in totals:
                assert abs(near[method]["mae"] - full[method]["depth"]["near"]["MAE"]) < 1e-9
            row = {"dataset": name, "units": domain["geometry_units"],
                   "near_upper_exclusive": 10. if domain["geometry_units"] == "meters" else 5 * domain["depth_scale"],
                   "frames": len(frames), "near_gt_pixels": near_pixels,
                   "near_gt_share": near_pixels / full_pixels,
                   "near_own_error_share": totals[own]["abs_sum"] / (full[own]["depth"]["full"]["MAE"] * full_pixels),
                   "near_gap_contribution_to_full_mae": (totals[own]["abs_sum"] - totals[baseline]["abs_sum"]) / full_pixels,
                   "full_mae_gap": full[own]["depth"]["full"]["MAE"] - full[baseline]["depth"]["full"]["MAE"]}
            for label, method in (("new", own), ("old", old), ("baseline", baseline)):
                row.update({label + "_" + key: value for key, value in near[method].items()})
            local_rows = []
            sequences = defaultdict(list)
            for frame in frames:
                index = frame["index"]
                identity = manifests[index]
                sequence = identity.get("sequence", "unknown")
                a, b = [frame["methods"][method]["depth"]["near"] for method in (own, baseline)]
                if not a["gt_pixels"]:
                    continue
                point = {"dataset": name, "index": index, "sequence": sequence,
                         "frame_id": identity.get("frame_id"), "valid_views": frame["valid_views"],
                         "near_gt_pixels": a["gt_pixels"], "near_gt_share": a["gt_pixels"] / frame["methods"][own]["depth"]["full"]["gt_pixels"],
                         "gap_abs_sum": a["abs_sum"] - b["abs_sum"]}
                for label, values in (("new", a), ("baseline", b)):
                    point.update({label + "_" + key: value for key, value in metrics(values).items()})
                point["mae_gap"] = point["new_mae"] - point["baseline_mae"]
                local_rows.append(point)
                sequences[sequence].append(frame)
            good = [item for item in local_rows if item["near_gt_pixels"] >= 1024]
            positive = sorted((max(0., item["gap_abs_sum"]) for item in local_rows), reverse=True)
            error_mass = sorted((item["new_mae"] * item["near_gt_pixels"] for item in local_rows), reverse=True)
            row.update(near_nonempty_frames=len(local_rows), near_1024px_frames=len(good),
                       near_frame_win_fraction=sum(item["mae_gap"] < 0 for item in local_rows) / len(local_rows),
                       near_1024px_frame_win_fraction=sum(item["mae_gap"] < 0 for item in good) / len(good) if good else None,
                       near_frame_new_mae_median=median(item["new_mae"] for item in local_rows),
                       near_frame_baseline_mae_median=median(item["baseline_mae"] for item in local_rows))
            for fraction in (.01, .05, .1):
                count = max(1, math.ceil(len(local_rows) * fraction))
                suffix = str(round(fraction * 100)) + "pct_frames"
                row["positive_gap_share_top_" + suffix] = sum(positive[:count]) / sum(positive) if sum(positive) else 0.
                row["own_error_share_top_" + suffix] = sum(error_mass[:count]) / sum(error_mass) if sum(error_mass) else 0.
            for sequence, selected in sequences.items():
                a, b = [total(frame["methods"][method]["depth"]["near"] for frame in selected) for method in (own, baseline)]
                item = {"dataset": name, "sequence": sequence, "frames": len(selected),
                        "near_pixels": a["gt_pixels"],
                        "contribution_to_dataset_near_gap": (a["abs_sum"] - b["abs_sum"]) / near_pixels}
                for label, values in (("new", a), ("baseline", b)):
                    item.update({label + "_" + key: value for key, value in metrics(values).items()})
                sequence_rows.append(item)
            domain_rows.append(row)
            frame_rows.extend(local_rows)
            top = sorted(local_rows, key=lambda item: item["gap_abs_sum"], reverse=True)[:10]
            for item in top:
                record = manifests[item["index"]]
                camera = record["cameras"][domain["reference_index"]]
                item["reference_rgb"] = str(Path(domain["root"]) / camera["rgb"])
            details[name] = {"summary": row, "worst_positive_gap_frames": top,
                             "sequence_count_with_near_pixels": len(sequences)}
            print(name, "near MAE new/baseline", round(row["new_mae"], 6), round(row["baseline_mae"], 6),
                  "frame wins", round(row["near_frame_win_fraction"], 4),
                  "top10% positive gap", round(row["positive_gap_share_top_10pct_frames"], 4), flush=True)
    csv_write(output / "near_by_dataset.csv", domain_rows)
    csv_write(output / "near_by_sequence.csv", sequence_rows)
    csv_write(output / "near_by_frame.csv", frame_rows)
    report = {"completed": True, "source": str(SOURCE),
              "scope": "既有固定TEST结果重组，不改变GT/掩码/阈值，不构成新推理；near交叉边界/语义未由这些边际统计获得。",
              "near_definition": "meters: Z<10m；scene_unit: Z<TRAIN中位深度/2=5*depth_scale；CARLA额外已有2–80m有效范围。",
              "gap_concentration": "按逐帧近距绝对误差差的正部排序；分母是所有正部之和，不能解释为净差比例或像素失败占比。",
              "datasets": details}
    (output / "near_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
