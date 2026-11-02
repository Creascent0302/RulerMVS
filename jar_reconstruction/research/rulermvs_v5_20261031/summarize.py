"""汇总六组V5短训；核对同起点/同预算，不跨域混合米与scene_unit误差。"""

import argparse
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("control", "layer1", "layer12")
REGIONS = ("full", "near", "mid", "far")
METRICS = ("MAE", "RMSE", "AbsRel")


def read(filename):
    return json.loads(filename.read_text())


def relative(current, reference):
    return current / reference - 1 if current is not None and reference is not None and reference > 0 else None


def fields(domain):
    return {"depth": domain["depth"],
            **{name + "_stride8": values for name, values in domain["stages_stride8"].items()}}


def verify(records):
    control = records["control"]
    comparisons = {}
    for variant, record in records.items():
        checks = {key: record["plan"][key] == control["plan"][key]
                  for key in ("training", "validation", "schedule", "checkpoint_sha256")}
        checks["initial_exact"] = record["initial"] == control["initial"]
        keys = ("seed", "batch_size", "epochs", "train_samples_per_domain", "lr",
                "weight_decay", "augmentation", "data_config")
        checks["training_settings_equal"] = all(record["settings"].get(key) == control["settings"].get(key)
                                                 for key in keys)
        done = record["completed"]
        checks.update(completed=bool(done["completed"]), encoders_unchanged=bool(done["encoders_unchanged"]),
                      no_test=not done["test_accessed"],
                      checkpoint_checks=all(done["checkpoint_checks"].values()),
                      steps_match=done["steps"] == record["plan"]["schedule"]["steps"],
                      exposures_match=done["exposures"] == record["plan"]["schedule"]["exposures"])
        if not all(checks.values()):
            raise RuntimeError(f"{variant}对照协议核对失败：{checks}")
        comparisons[variant] = checks
    return comparisons


def summarize(folder):
    rows, summary = [], {"completed": True, "test_accessed": False, "phases": {},
                         "scope": "固定末步小预算VAL；depth为全分辨率，其他为stride8；无baseline或TEST胜出结论"}
    for phase in ("public", "carla_new"):
        records = {variant: {name: read(folder / "pilot" / (phase + "_" + variant) / (name + ".json"))
                             for name in ("settings", "plan", "initial", "final", "completed")}
                   for variant in VARIANTS}
        checks = verify(records)
        control = records["control"]
        phase_summary = {"checks": checks, "methods": {}}
        summary["phases"][phase] = phase_summary
        for variant, record in records.items():
            risks = {partition: {"frozen": [], "control": []} for partition in record["final"]}
            for partition, domains in record["final"].items():
                for domain, current in domains.items():
                    frozen = control["initial"][partition][domain]
                    continued = control["final"][partition][domain]
                    if current["units"] != frozen["units"] or current["frames"] != frozen["frames"]:
                        raise RuntimeError(f"{phase}/{domain}单位或帧数不同")
                    before, baseline = fields(frozen), fields(continued)
                    for stage, regions in fields(current).items():
                        for region in REGIONS:
                            value, old, compare = regions[region], before[stage][region], baseline[stage][region]
                            for count in ("gt_pixels", "predicted_pixels"):
                                if value[count] != old[count] or value[count] != compare[count]:
                                    raise RuntimeError(f"{phase}/{domain}/{stage}/{region}/{count}分母不同")
                            for metric in METRICS:
                                new, start, reference = value[metric], old[metric], compare[metric]
                                changes = {"frozen": relative(new, start), "control": relative(new, reference)}
                                rows.append({"phase": phase, "partition": partition, "dataset": domain,
                                             "units": current["units"], "method": variant, "stage": stage,
                                             "region": region, "metric": metric, "gt_pixels": value["gt_pixels"],
                                             "value": new, "frozen": start, "control": reference,
                                             "delta_vs_frozen": new - start if new is not None else None,
                                             "delta_vs_control": new - reference if new is not None else None,
                                             "relative_vs_frozen": changes["frozen"],
                                             "relative_vs_control": changes["control"]})
                                if stage == "depth" and region == "full":
                                    for name, change in changes.items():
                                        if change is not None:
                                            risks[partition][name].append(change)
            phase_summary["methods"][variant] = {
                "steps": record["completed"]["steps"], "training_seconds": record["completed"]["training_seconds"],
                "new_parameters": record["completed"]["new_parameters"],
                "macro_relative_full_risk": {partition: {name: statistics.mean(values) if values else None
                                                          for name, values in comparison.items()}
                                             for partition, comparison in risks.items()},
            }
    summary["auxiliary_risk_definition"] = "各域full MAE/RMSE/AbsRel相对变化均值；无量纲辅助描述，不混合原单位误差"
    summary["rows"] = rows
    for suffix in ("json", "csv"):
        if (folder / ("summary." + suffix)).exists():
            raise FileExistsError(folder / ("summary." + suffix))
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "phases": summary["phases"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "runs/active/rulermvs_v5_20261031")
    summarize(parser.parse_args().folder)
