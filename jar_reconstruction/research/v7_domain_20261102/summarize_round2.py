"""第二轮只对比同训练条件的fixed/coordinate，并保留相冻结V6的变化。"""

import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    rows, pending, completed = [], [], {}
    for variant in ("fixed", "coordinate"):
        for condition in ("new_only", "joint_keep"):
            group = variant + "_" + condition
            for phase in ("public", "carla"):
                folder = root / group / phase
                if not (folder / "round2_audit.json").exists():
                    pending.append(f"{group}/{phase}")
                    continue
                audit = json.loads((folder / "round2_audit.json").read_text())
                if not audit["completed"]:
                    raise RuntimeError(f"未通过审核的结果: {folder}")
                completed[f"{group}/{phase}"] = json.loads((folder / "completed.json").read_text())
                initial = json.loads((folder / "initial.json").read_text())["datasets"]
                final = json.loads((folder / "final.json").read_text())["datasets"]
                for domain, record in final.items():
                    start, end = initial[domain]["metrics"], record["metrics"]
                    before = {"depth": start["depth"], **start["stages_stride8"]}
                    after = {"depth": end["depth"], **end["stages_stride8"]}
                    for stage, regions in after.items():
                        for region, values in regions.items():
                            if not values.get("gt_pixels", 0):
                                continue
                            row = {"variant": variant, "condition": condition, "phase": phase,
                                   "dataset": domain, "units": record["units"], "stage": stage,
                                   "region": region, "pixels": values["gt_pixels"]}
                            for metric in ("MAE", "RMSE", "AbsRel"):
                                row[metric] = values[metric]
                                row[metric + "_vs_v6_pct"] = 100 * (values[metric] / before[stage][region][metric] - 1)
                            rows.append(row)
    lookup = {(r["variant"], r["condition"], r["phase"], r["dataset"], r["stage"], r["region"]): r for r in rows}
    for row in rows:
        key = ("fixed", row["condition"], row["phase"], row["dataset"], row["stage"], row["region"])
        for metric in ("MAE", "RMSE", "AbsRel"):
            row[metric + "_vs_fixed_pct"] = 100 * (row[metric] / lookup[key][metric] - 1) if key in lookup else None
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    report = {"completed": completed, "pending": pending, "all_complete": not pending,
              "test_accessed": False, "metrics": rows,
              "scope": "同V6起点、小TRAIN与每域16VAL、固定末步单seed，不是完整TEST；两种训练条件非单因素loss消融"}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if rows:
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ["# 第二轮连续候选坐标短训", "", report["scope"], "",
            "|结构|训练条件|域|单位|MAE|RMSE|AbsRel|MAE相V6|MAE相同条件fixed|",
            "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["stage"] == "depth" and row["region"] == "full":
            difference = row["MAE_vs_fixed_pct"]
            difference = "未完成" if difference is None else f"{difference:+.2f}%"
            text.append(f"|{row['variant']}|{row['condition']}|{row['dataset']}|{row['units']}|"
                        f"{row['MAE']:.6f}|{row['RMSE']:.6f}|{row['AbsRel']:.6f}|"
                        f"{row['MAE_vs_v6_pct']:+.2f}%|{difference}|")
    text.extend(["", "未完成：" + ("、".join(pending) if pending else "无"), ""])
    (output / "短训结果.md").write_text("\n".join(text))
    print(json.dumps({"completed": len(completed), "pending": pending, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/active/v7_domain_20261102/pilots_round2"))
    summarize(parser.parse_args().root)
