"""同分母比较冻结V2、续训V2、四峰无更新、直接回归及Flow各积分步数。"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(output, arms=None, report="summary.json"):
    output = Path(output)
    output = output if output.is_absolute() else ROOT / output
    read = lambda file: json.loads(file.read_text())
    destination = output / report
    if destination.exists():
        raise FileExistsError(destination)
    rows, checks = [], {}
    for group in ("public", "carla"):
        control = output / f"{group}_control"
        common_plan = read(control / "plan.json")
        frozen = read(control / "initial.json")
        comparison = read(control / "final.json")
        results = {"frozen": frozen, "control": comparison,
                   "reduced4": read(output / f"{group}_direct/reduced4.json")}
        checks[group] = {}
        for arm in arms or ("control", "direct", "flow", "flow_no_direction"):
            folder = output / f"{group}_{arm}"
            plan, completed = read(folder / "plan.json"), read(folder / "completed.json")
            for key in ("schedule", "training", "validation"):
                if plan[key] != common_plan[key]:
                    raise RuntimeError(f"{group}/{arm}的{key}与control不同")
            if not completed["completed"]:
                raise RuntimeError(f"{group}/{arm}未完成")
            checks[group][arm] = completed
            for step in completed["evaluation_steps"]:
                label = arm if arm == "control" else f"{arm}_steps{step}"
                results[label] = read(folder / f"final_steps{step}.json")
        for split, domains in comparison.items():
            for name, domain in domains.items():
                for level in ("depth", "coarse", "fine", "final"):
                    def measures(result):
                        record = result[split][name]
                        return record["depth"] if level == "depth" else record["stages_stride8"][level]

                    base, start = measures(comparison), measures(frozen)
                    for region in base:
                        counts = {method: measures(result)[region]["gt_pixels"] for method, result in results.items()}
                        if len(set(counts.values())) != 1:
                            raise RuntimeError(f"{group}/{name}/{split}/{level}/{region}GT分母不同")
                        for method, result in results.items():
                            metrics = measures(result)[region]
                            for metric in ("MAE", "RMSE", "AbsRel", "coverage"):
                                value = metrics[metric]
                                baseline, original = base[region][metric], start[region][metric]
                                rows.append({
                                    "group": group, "method": method, "split": split, "dataset": name,
                                    "units": domain["units"], "frames": domain["frames"], "level": level,
                                    "region": region, "metric": metric, "value": value,
                                    "control": baseline, "frozen": original, "gt_pixels": metrics["gt_pixels"],
                                    "relative_to_control": value / baseline - 1 if value is not None and baseline else None,
                                    "relative_to_frozen": value / original - 1 if value is not None and original else None,
                                })
    checks["scope"] = "同预算固定末步、历史16VAL；depth是完整输出分辨率，final是stride8抽样。非完整TEST。"
    destination.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存{len(rows)}条指标", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/active/ruler_flow_20261030")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config:
        run(**json.loads((ROOT / args.config).read_text()))
    else:
        run(args.output)
