"""汇总预登记末步短试；不选权重、不读取TEST、不将域间绝对单位混合。"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/near_evidence_20261030"
ARMS = ("control", "relative_evidence", "intervention_only")


def read(path):
    return json.loads(path.read_text())


def relative(value, reference):
    return value / reference - 1 if value is not None and reference is not None and reference > 0 else None


def main(arms=ARMS, prefix="pilot"):
    report, rows = {}, []
    for phase in ("public", "carla"):
        artifacts = {arm: {name: read(OUTPUT / f"{phase}_{arm}" / (name + ".json"))
                           for name in ("initial", "final", "completed", "plan")} for arm in arms}
        assert len({item["completed"]["schedule_sha256"] for item in artifacts.values()}) == 1
        if "intervention_only" in artifacts:
            assert artifacts["control"]["initial"] == artifacts["intervention_only"]["initial"]
        for arm, item in artifacts.items():
            done = item["completed"]
            assert done["completed"] and done["gradients_finite"]
            assert done["initial_encoder_sha256"] == done["final_encoder_sha256"]
            assert (OUTPUT / "logs" / f"{phase}_{arm}.exit_code.txt").read_text().strip() == "0"
        report[phase] = {"runs": {arm: item["completed"] for arm, item in artifacts.items()},
                         "corruption_eval_initial_identical": True if "intervention_only" in artifacts else None,
                         "comparisons": {}}
        for split in ("development", "confirmation"):
            split_rows = []
            for domain, control in artifacts["control"]["final"][split].items():
                frozen = artifacts["control"]["initial"][split][domain]
                for arm in arms:
                    candidate = artifacts[arm]["final"][split][domain]
                    initial = artifacts[arm]["initial"][split][domain]
                    for stage in ("dense", "coarse", "fine", "final_stride8"):
                        def stage_metrics(item):
                            return item["depth"] if stage == "dense" else item["stages_stride8"][stage.removesuffix("_stride8")]
                        for region, values in stage_metrics(candidate).items():
                            a, c, f = [stage_metrics(item)[region] for item in (initial, control, frozen)]
                            row = {"phase": phase, "split": split, "dataset": domain, "arm": arm,
                                   "stage": stage, "region": region, "units": candidate["units"],
                                   "frames": candidate["frames"], "gt_pixels": values["gt_pixels"]}
                            for metric in ("MAE", "RMSE", "AbsRel", "signed_bias", "delta_1", "coverage"):
                                row.update({metric: values[metric], "initial_" + metric: a[metric],
                                            "control_final_" + metric: c[metric], "frozen_latest_" + metric: f[metric]})
                                if metric in ("MAE", "RMSE", "AbsRel"):
                                    row["vs_control_" + metric] = relative(values[metric], c[metric])
                                    row["vs_frozen_" + metric] = relative(values[metric], f[metric])
                            rows.append(row)
                            if stage == "dense" and region in ("full", "near", "far"):
                                split_rows.append(row)
            comparison = {}
            for arm in arms[1:]:
                comparison[arm] = {}
                for region in ("full", "near", "far"):
                    selected = [row for row in split_rows if row["arm"] == arm and row["region"] == region and row["gt_pixels"]]
                    changes = [row["vs_control_MAE"] for row in selected]
                    comparison[arm][region] = {"domains": len(changes), "mae_wins_vs_control": sum(x < 0 for x in changes),
                                              "mean_relative_mae_change": sum(changes) / len(changes) if changes else None,
                                              "worst_relative_mae_change": max(changes) if changes else None}
            report[phase]["comparisons"][split] = comparison
    report["scope"] = "固定末步；公共896TRAIN曝光/448步，CARLA497曝光/249步。每域8dev+8confirmation来自既有VAL，非新盲测；初始冻结模型与同预算control分开。百分比的域均值不是跨单位MAE。"
    if prefix in ("round2", "round3"):
        report["scope"] += " 本轮development和confirmation均为已查看的复用VAL，不称新盲确认。"
    if prefix == "round3":
        report["visibility_training_descriptive"] = {}
        for phase in ("public", "carla"):
            grouped = {}
            for line in (OUTPUT / f"{phase}_visibility_veto/train.jsonl").read_text().splitlines():
                row = json.loads(line)
                grouped.setdefault(row["dataset"], []).append(row)
            domains = {}
            for domain, values in grouped.items():
                quarter = max(1, len(values) // 4)
                partitions = {"all": values, "first_quarter": values[:quarter], "last_quarter": values[-quarter:]}
                result = {"batches": len(values)}
                for partition, selected in partitions.items():
                    summary = {"batches": len(selected), "skipped_batches": sum(row["visibility"]["skipped"] for row in selected)}
                    for stage in ("coarse", "fine"):
                        observations = [row["visibility"]["stages"][stage] for row in selected if stage in row["visibility"]["stages"]]
                        if not observations:
                            continue
                        summary[stage] = {"mean_" + key: sum(item[key] for item in observations) / len(observations)
                                          for key in ("null_mass_mean", "occluded_mass_mean")}
                        summary[stage].update({"sum_" + key: sum(item[key] for item in observations)
                                               for key in ("positive_pairs", "occluded_pairs", "known_pairs", "supervised_rays", "reference_rays", "covered_rays")})
                    result[partition] = summary
                domains[domain] = result
            report["visibility_training_descriptive"][phase] = domains
        report["training_trend_caveat"] = "初末为不同TRAIN批次和参考视角；仅描述质量趋势，不能当同输入因果对照。stage mean是批次等权平均。"
    (OUTPUT / (prefix + "_summary.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (OUTPUT / (prefix + "_metrics.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({phase: report[phase]["comparisons"] for phase in ("public", "carla")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=("first", "coarse", "visibility"), default="first")
    args = parser.parse_args()
    if args.round == "first":
        main()
    elif args.round == "coarse":
        main(("control", "coarse_attention"), "round2")
    else:
        main(("control", "visibility_veto"), "round3")
