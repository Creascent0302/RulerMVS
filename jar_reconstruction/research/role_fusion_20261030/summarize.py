"""逐域与同预算control比较，不把不同单位的MAE混成一个数。"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(settings):
    output = ROOT / settings["output"]
    rows, summary = [], {}
    for group, arms in settings["groups"].items():
        runs = {}
        for name, relative in arms.items():
            folder = ROOT / relative
            runs[name] = {key: json.loads((folder / f"{key}.json").read_text())
                          for key in ("initial", "final", "completed", "plan", "settings")}
        control = runs["control"]
        checks = {}
        for arm, run in runs.items():
            same_plan = all(run["plan"][key] == control["plan"][key]
                            for key in ("schedule", "training", "validation"))
            same_initial = run["initial"] == control["initial"]
            if not same_plan:
                raise RuntimeError(f"{group}/{arm}与control的训练或验证清单不同")
            if run["settings"].get("initial_identity", True) and not same_initial:
                raise RuntimeError(f"{group}/{arm}声明恒等初始化但实际初始指标不同")
            checks[arm] = {"same_plan": same_plan, "same_initial_metrics": same_initial,
                           "completed": run["completed"]}
            for split, domains in run["final"].items():
                for domain_name, domain in domains.items():
                    for stage in ("depth", "coarse", "fine", "final"):
                        def read(result):
                            item = result[split][domain_name]
                            return item["depth"] if stage == "depth" else item["stages_stride8"][stage]

                        after = read(run["final"])
                        initial = read(run["initial"])
                        base = read(control["final"])
                        frozen = read(control["initial"])
                        for region in after:
                            for metric in ("MAE", "RMSE", "AbsRel"):
                                value, baseline, before = after[region][metric], base[region][metric], frozen[region][metric]
                                rows.append({
                                    "group": group, "arm": arm, "split": split,
                                    "dataset": domain_name, "units": domain["units"],
                                    "stage": stage, "region": region, "metric": metric,
                                    "value": value, "control": baseline, "frozen": before,
                                    "variant_initial": initial[region][metric],
                                    "relative_to_control": value / baseline - 1 if value is not None and baseline else None,
                                    "relative_to_frozen": value / before - 1 if value is not None and before else None,
                                    "relative_to_variant_initial": value / initial[region][metric] - 1
                                    if value is not None and initial[region][metric] else None,
                                })
        summary[group] = checks
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError(f"拒绝覆盖汇总：{output}")
    summary["scope"] = "固定短预算、历史已查看VAL；非完整TEST或完整重训。"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"保存{len(rows)}项逐域指标到 {output.with_suffix('.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads((ROOT / parser.parse_args().config).read_text()))
