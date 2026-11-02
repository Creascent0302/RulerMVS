"""汇总真实已完成的同预算短基线再生成对照，不执行模型或访问TEST。"""

import argparse
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def percentage(value, reference):
    if value is None or reference is None or reference == 0:
        return None
    return 100 * (value / reference - 1)


def summarize(root):
    rows = []
    audits = {}
    for condition in ("coordinate_new_only", "coordinate_joint_keep"):
        for phase in ("public", "carla"):
            current = root / "pilots_synthetic" / condition / phase
            control = root / "pilots_round2" / condition / phase
            synthesis = load(current / "synthetic_audit.json")
            completed = load(current / "completed.json")
            control_completed = load(control / "completed.json")
            assert synthesis["completed"] and not synthesis["test_accessed"]
            assert all(completed["checkpoint_checks"].values())
            assert all(control_completed["checkpoint_checks"].values())
            current_plan = load(current / "plan.json")
            control_plan = load(control / "plan.json")
            for key in ("checkpoint_sha256", "training", "validation", "schedule",
                        "test_accessed", "selection", "input_mode"):
                assert current_plan[key] == control_plan[key], (condition, phase, key)
            final = load(current / "final.json")
            baseline = load(control / "final.json")
            initial = load(control / "initial.json")
            own_initial = load(current / "initial.json")
            assert final["completed"] and baseline["completed"]
            assert not final["test_accessed"] and not baseline["test_accessed"]
            assert set(final["datasets"]) == set(baseline["datasets"])
            initial_max_drift = 0.
            for dataset, item in final["datasets"].items():
                old = baseline["datasets"][dataset]
                start = initial["datasets"][dataset]
                own_start = own_initial["datasets"][dataset]
                for key in ("units", "indices", "manifest_sha256"):
                    assert item[key] == old[key] == start[key] == own_start[key]
                paths = {"depth": item["metrics"]["depth"]}
                paths.update(item["metrics"]["stages_stride8"])
                for stage, groups in paths.items():
                    def metric_container(value):
                        metrics = value["metrics"]
                        return metrics["depth"] if stage == "depth" else metrics["stages_stride8"][stage]

                    for band, metrics in groups.items():
                        previous = metric_container(old)[band]
                        frozen = metric_container(start)[band]
                        own_frozen = metric_container(own_start)[band]
                        assert metrics["gt_pixels"] == previous["gt_pixels"] == frozen["gt_pixels"]
                        for metric in ("MAE", "RMSE", "AbsRel", "coverage"):
                            value, reference, untouched = metrics[metric], previous[metric], frozen[metric]
                            if untouched is not None:
                                initial_max_drift = max(initial_max_drift, abs(untouched - own_frozen[metric]))
                            rows.append({
                                "condition": condition, "phase": phase, "dataset": dataset,
                                "units": item["units"], "stage": stage, "band": band,
                                "metric": metric, "gt_pixels": metrics["gt_pixels"],
                                "frozen_V6": untouched, "same_budget_no_synthesis": reference,
                                "synthesis25": value,
                                "change_vs_no_synthesis_pct": percentage(value, reference),
                                "change_vs_frozen_V6_pct": percentage(value, untouched),
                            })
            audits[f"{condition}/{phase}"] = {
                "same_schedule_checkpoint_validation": True,
                "initial_max_metric_drift": initial_max_drift,
                "exposure": synthesis["exposure"],
                "checkpoint_checks": completed["checkpoint_checks"],
                "source_sha256": synthesis["source_sha256"],
            }
    return rows, audits


def run(root, output):
    rows, audits = summarize(root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "audit.json").write_text(json.dumps({
        "completed": True, "test_accessed": False, "audits": audits,
        "scope": "同一V6起点、同预算16VAL/分区；策略同时改变视差、相机数与覆盖，不隔离单一基线因素",
    }, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 短基线再生成：同预算验证结果", "",
             "全部数值来自固定末步、每域/参考16个预登记VAL；没有访问TEST或ETH。"
             "公共与CARLA分别从对应V6正式权重初始化，不能把本表当成全预算训练结论。", "",
             "同组相较无再生成仅改变TRAIN数据策略；合成同时改变视差、源数量、覆盖和插值纹理。"
             "不能将差值全部归因于短基线。", "",
             "| 训练条件 | 域/相机 | 单位 | 冻结V6 MAE | 同预算原数据 | 再生成25% | 相原数据变化 |", 
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if (row["stage"], row["band"], row["metric"]) != ("depth", "full", "MAE"):
            continue
        lines.append(f'| {row["condition"]} | {row["dataset"]} | {row["units"]} | '
                     f'{row["frozen_V6"]:.6f} | {row["same_budget_no_synthesis"]:.6f} | '
                     f'{row["synthesis25"]:.6f} | {row["change_vs_no_synthesis_pct"]:+.3f}% |')
    lines.extend(["", "## 实际训练曝光与覆盖", "",
                  "| 训练条件/阶段 | 总曝光 | 合成 | 原样本 | 合成源平均有效覆盖 |", 
                  "|---|---:|---:|---:|---:|"])
    for condition, audit in audits.items():
        item = audit["exposure"]
        total_coverage = sum(domain["coverage_sum_synthetic"] for domain in item["datasets"].values())
        mean_coverage = total_coverage / max(item["synthetic"], 1)
        lines.append(f'| {condition} | {item["exposures"]} | {item["synthetic"]} | '
                     f'{item["original"]} | {mean_coverage:.2%} |')
    lines.extend(["", "指标完整表见metrics.csv，含共同stride8粗/细/最终层、全分辨率输出，"
                  "以及近/中/远、边界带和平滑区域。不同单位不得直接相加平均。"
                  "checkpoint、曝光schedule、VAL索引与清洁起点核对见audit.json。", ""])
    (output / "对照结果.md").write_text("\n".join(lines))
    print(json.dumps({"completed": True, "rows": len(rows), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/active/v7_domain_20261102"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/active/v7_domain_20261102/synthetic_analysis"))
    args = parser.parse_args()
    run(args.root, args.output)
