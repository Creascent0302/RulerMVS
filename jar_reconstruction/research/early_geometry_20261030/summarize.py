"""按数据域汇总短实验，不跨米制与scene_unit混合MAE。"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/early_geometry_20261030"
ARMS = ("control", "appearance", "early", "late", "both")


def run(arms=ARMS, prefix=""):
    summary, rows = {}, []
    for group in ("public", "carla"):
        current = {}
        for arm in arms:
            folder = OUTPUT / f"{group}_{arm}"
            if not (folder / "completed.json").exists():
                continue
            current[arm] = {name: json.loads((folder / f"{name}.json").read_text())
                            for name in ("initial", "final", "completed", "plan")}
        if "control" not in current:
            continue
        control = current["control"]
        summary[group] = {"runs": {arm: values["completed"] for arm, values in current.items()},
                          "initial_matches_control": {}, "schedule_matches_control": {}}
        for arm, values in current.items():
            summary[group]["initial_matches_control"][arm] = values["initial"] == control["initial"]
            summary[group]["schedule_matches_control"][arm] = (
                values["plan"]["schedule"] == control["plan"]["schedule"]
                and values["plan"]["training"] == control["plan"]["training"]
                and values["plan"]["validation"] == control["plan"]["validation"])
            for partition, domains in values["final"].items():
                for name, domain in domains.items():
                    for level in ("depth", "coarse", "fine", "final"):
                        def level_values(payload):
                            value = payload[partition][name]
                            return value["depth"] if level == "depth" else value["stages_stride8"][level]
                        candidate = level_values(values["final"])
                        reference = level_values(control["final"])
                        frozen = level_values(control["initial"])
                        for region in ("full", "near", "mid", "far", "edge", "smooth"):
                            for metric in ("MAE", "RMSE", "AbsRel"):
                                after, base, before = candidate[region][metric], reference[region][metric], frozen[region][metric]
                                rows.append({"group": group, "arm": arm, "partition": partition,
                                             "dataset": name, "units": domain["units"], "level": level,
                                             "region": region, "metric": metric, "value": after,
                                             "control": base, "frozen": before,
                                             "relative_to_control": after / base - 1 if after is not None and base else None,
                                             "relative_to_frozen": after / before - 1 if after is not None and before else None})
    with (OUTPUT / f"{prefix}metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["scope"] = "短预算历史VAL对照；不是TEST或全量重训；训练时间不作推理延迟。"
    (OUTPUT / f"{prefix}summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({group: values.get("initial_matches_control") for group, values in summary.items()
                      if isinstance(values, dict)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
