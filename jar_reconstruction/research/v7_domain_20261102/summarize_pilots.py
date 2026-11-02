"""汇总已完成的固定末步短训，分别对比冻结V6和同增强的续训对照。"""

import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    groups = [f"{variant}_{augmentation}" for variant in ("control", "projective", "prior")
              for augmentation in ("original", "augment")]
    completed, pending, rows = {}, [], []
    for group in groups:
        for phase in ("public", "carla"):
            folder = root / group / phase
            if not (folder / "completed.json").exists():
                pending.append(f"{group}/{phase}")
                continue
            status = json.loads((folder / "completed.json").read_text())
            if not status.get("completed"):
                raise RuntimeError(f"无效完成记录：{folder}")
            initial = json.loads((folder / "initial.json").read_text())["datasets"]
            final = json.loads((folder / "final.json").read_text())["datasets"]
            completed[f"{group}/{phase}"] = status
            for domain, record in final.items():
                metrics = record["metrics"]
                stages = {"depth": metrics["depth"], **metrics["stages_stride8"]}
                before = initial[domain]["metrics"]
                starts = {"depth": before["depth"], **before["stages_stride8"]}
                for stage, regions in stages.items():
                    for region, values in regions.items():
                        if not values.get("gt_pixels", 0):
                            continue
                        row = {"group": group, "phase": phase, "dataset": domain,
                               "units": record["units"], "stage": stage, "region": region,
                               "pixels": values["gt_pixels"]}
                        for metric in ("MAE", "RMSE", "AbsRel"):
                            row[metric] = values[metric]
                            row[metric + "_vs_initial_pct"] = 100 * (
                                values[metric] / starts[stage][region][metric] - 1)
                        rows.append(row)
    lookup = {(r["group"], r["phase"], r["dataset"], r["stage"], r["region"]): r for r in rows}
    for row in rows:
        control = "control_" + row["group"].rsplit("_", 1)[1]
        key = (control, row["phase"], row["dataset"], row["stage"], row["region"])
        for metric in ("MAE", "RMSE", "AbsRel"):
            row[metric + "_vs_matched_control_pct"] = 100 * (
                row[metric] / lookup[key][metric] - 1) if key in lookup else None
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    if rows:
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {"completed": completed, "pending": pending, "all_complete": not pending,
              "scope": "固定小TRAIN/16VAL每域，同V6起点单seed短适配，不是TEST或全预算训练",
              "test_accessed": False, "metrics": rows}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    table = ["# V7 第一轮短训结果", "", result["scope"], "",
             "负的变化百分比表示误差降低；同增强对照隔离结构作用。", "",
             "| 组 | 域 | 单位 | MAE | RMSE | AbsRel | MAE相冻结V6 | MAE相同增强对照 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["stage"] != "depth" or row["region"] != "full":
            continue
        matched = row["MAE_vs_matched_control_pct"]
        matched = "待完成" if matched is None else f"{matched:+.2f}%"
        table.append(f"| {row['group']} | {row['dataset']} | {row['units']} | "
                     f"{row['MAE']:.6f} | {row['RMSE']:.6f} | {row['AbsRel']:.6f} | "
                     f"{row['MAE_vs_initial_pct']:+.2f}% | {matched} |")
    table.extend(["", "未完成项：" + ("、".join(pending) if pending else "无"), ""])
    (output / "短训结果.md").write_text("\n".join(table))
    print(json.dumps({"completed_runs": len(completed), "pending": pending, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/active/v7_domain_20261102/pilots_round1"))
    summarize(parser.parse_args().root)
