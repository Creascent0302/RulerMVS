"""从真实完成产物生成标尺消融表；CPU运行，不补齐或猜测缺失结果。"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/rulermvs_v6_20261101/full_training/paper_geometry_ablation"
SCOPES = {"initial_val": "initial.json", "final_val": "final.json", "full_val": "full_val.json"}


def read(path):
    return json.loads(path.read_text())


def records(report, scope):
    rows = {}
    for name, domain in report["datasets"].items():
        metrics = domain["metrics"]
        candidate = metrics.get("candidates_stride8", {}).get("full", {})
        coarse_good = candidate.get("coarse_good")
        correction = candidate.get("coarse_bad_fine_good")
        for region, value in metrics["depth"].items():
            if region not in ("full", "near", "mid", "far"):
                continue
            row = {"dataset": name, "units": domain["units"], "region": region,
                   "frames": domain["frames"] if isinstance(domain["frames"], int) else len(domain["frames"]),
                   **{key: value.get(key) for key in ("MAE", "RMSE", "AbsRel", "coverage", "gt_pixels")}}
            for stage in ("coarse", "fine", "final"):
                if scope == "full_val":
                    stage_metric = metrics[stage + "_stride8"][region]
                else:
                    stage_metric = metrics["stages_stride8"][stage][region]
                row[stage + "_stride8_MAE"] = stage_metric.get("MAE")
            if region == "full":
                row["coarse_correct_destroyed"] = candidate.get("coarse_good_destroyed_fraction")
                row["coarse_wrong_corrected"] = (
                    correction / (1 - coarse_good)
                    if correction is not None and coarse_good is not None and coarse_good < 1 else None)
                row["continuous_coverage_5pct"] = candidate.get("continuous_coverage_5pct")
                row["group_selection_regret"] = candidate.get("group_selection_regret")
                for key in ("sampling_candidate_oracle_MAE", "continuous_candidate_oracle_MAE",
                            "group_mean_oracle_MAE", "selected_group_mean_MAE",
                            "group_mean_coverage_5pct"):
                    row[key] = candidate.get(key)
            rows[(name, region)] = row
    return rows


def main():
    plan = read(OUTPUT / "plan.json")
    summary = {"complete_training": True, "complete_full_val": True,
               "test_accessed": False, "seed": plan["seed"], "phases": {}, "rows": [],
               "limitations": ["单seed，不能当多seed显著性结论。",
                               "共同V5任务权重暖启动，不能声称从零架构消融。",
                               "固定16VAL与完整VAL分别列示，均不用于挑选本组末步权重。",
                               "同seed控制重复存在CUDA训练数值差异；微小单seed效应不能当可靠收益。"]}
    repeated_control = OUTPUT.parent / "paper_handoff_ablation/control_repeat_audit.json"
    if repeated_control.exists():
        summary["control_repeat_audit"] = read(repeated_control)
    for phase, info in plan["phases"].items():
        statuses = {}
        for variant, job in info["jobs"].items():
            folder = Path(job["output"])
            complete = folder / "completed.json"
            audit = folder / "geometry_audit.json"
            valid = complete.exists() and audit.exists()
            status = {"training_complete": False, "full_val_complete": False}
            if valid:
                trained, source = read(complete), read(audit)
                valid = bool(trained["completed"] and source["completed"])
                if trained["schedule_sha256"] != info["schedule"]["sha256"]:
                    raise ValueError(f"{phase}/{variant}采样次序改变")
                if source["checkpoint_sha256"] != info["checkpoint_sha256"]:
                    raise ValueError(f"{phase}/{variant}初始权重改变")
                status.update(training_complete=valid, steps=trained["steps"],
                              exposures=trained["exposures"], seconds=trained["training_seconds"])
            full = folder / "full_val.json"
            status["full_val_complete"] = valid and full.exists() and read(full).get("completed", False)
            statuses[variant] = status
        summary["phases"][phase] = statuses
        summary["complete_training"] &= all(s["training_complete"] for s in statuses.values())
        summary["complete_full_val"] &= all(s["full_val_complete"] for s in statuses.values())
        for scope, filename in SCOPES.items():
            condition = "full_val_complete" if scope == "full_val" else "training_complete"
            if not all(s[condition] for s in statuses.values()):
                continue
            tables = {variant: records(read(Path(job["output"]) / filename), scope)
                      for variant, job in info["jobs"].items()}
            control = tables["control"]
            for variant, table in tables.items():
                if table.keys() != control.keys():
                    raise ValueError("各组评测域或距离分区不同")
                for key, row in table.items():
                    baseline = control[key]
                    if (row["gt_pixels"], row["frames"], row["units"]) != (
                            baseline["gt_pixels"], baseline["frames"], baseline["units"]):
                        raise ValueError("各组GT分母/样本数/单位不同")
                    row.update(phase=phase, scope=scope, variant=variant, seed=plan["seed"])
                    repeated = summary.get("control_repeat_audit", {}).get(phase, {})
                    if scope == "final_val":
                        row["observed_same_seed_control_repeat_MAE_delta"] = repeated.get(
                            "final_small_val_MAE_delta", {}).get(row["dataset"])
                    for metric in ("MAE", "RMSE", "AbsRel"):
                        a, b = row[metric], baseline[metric]
                        row[metric + "_change_vs_control_pct"] = (
                            (a / b - 1) * 100 if a is not None and b is not None and b > 0 else None)
                    summary["rows"].append(row)
    destination = OUTPUT / "analysis"
    destination.mkdir(exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["rows"]:
        columns = list(dict.fromkeys(key for row in summary["rows"] for key in row))
        with (destination / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary["rows"])
    lines = ["# 标尺用途同预算适配消融", "",
             "单seed20261102，固定最后一步，所有数据为VAL。负的相对变化表示比control更好。",
             "训练完整：" + str(summary["complete_training"]) + "；完整VAL齐备：" + str(summary["complete_full_val"]), ""]
    for phase in plan["phases"]:
        for scope in ("final_val", "full_val"):
            selected = [row for row in summary["rows"]
                        if row["phase"] == phase and row["scope"] == scope and row["region"] == "full"]
            if not selected:
                continue
            lines.extend([f"## {phase} / {scope}", "",
                          "|评测域|组|MAE|RMSE|AbsRel|MAE相对control|单位|",
                          "|---|---|---:|---:|---:|---:|---|"])
            for row in selected:
                values = [row[k] for k in ("MAE", "RMSE", "AbsRel", "MAE_change_vs_control_pct")]
                formatted = ["N/A" if value is None else f"{value:.6f}" for value in values]
                lines.append("|" + "|".join([row["dataset"], row["variant"], *formatted, row["units"]]) + "|")
            lines.append("")
    lines.extend(["## 解释边界", "", *["- " + item for item in summary["limitations"]]])
    lines.extend(["", "## 候选供给与读出分解", "",
                  "下表只来自初始/末步固定VAL的共同stride8格点；完整VAL未额外保存候选oracle，不能冒充全量分解。",
                  "采样oracle为未做连续偏移的离散候选；点oracle为连续偏移后的有限候选集合；族均值oracle为四个族内均值中离线最优者，均使用GT仅作诊断。",
                  "|阶段|数据域|组|采样oracle|点oracle|族均值oracle|实际fine|选族regret|",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in summary["rows"]:
        if row["scope"] == "full_val" or row["region"] != "full":
            continue
        keys = ("sampling_candidate_oracle_MAE", "continuous_candidate_oracle_MAE",
                "group_mean_oracle_MAE", "fine_stride8_MAE", "group_selection_regret")
        values = ["N/A" if row.get(key) is None else f"{row[key]:.6f}" for key in keys]
        lines.append("|" + "|".join([row["scope"], row["dataset"], row["variant"], *values]) + "|")
    (destination / "结果表.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete_training": summary["complete_training"],
                      "complete_full_val": summary["complete_full_val"],
                      "rows": len(summary["rows"]), "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
