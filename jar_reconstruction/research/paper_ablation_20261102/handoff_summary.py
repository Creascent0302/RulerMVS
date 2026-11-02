"""三seed配对汇总；分别报告固定机制VAL与完整VAL，不跨单位平均MAE。"""

import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import write_json


VARIANTS = ("v5", "marginal", "constant", "parent", "parent_oldpoint")
SEEDS = (20261102, 20261103, 20261104)
OUTPUT = ROOT / "runs/active/rulermvs_v6_20261101/full_training/paper_handoff_ablation"


def aggregate(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
            "values": values, "seeds": list(SEEDS), "n": len(values)}


def run():
    records = {}
    for phase in ("public", "carla"):
        for variant in VARIANTS:
            for seed in SEEDS:
                folder = OUTPUT / f"{phase}_{variant}_seed{seed}"
                records[phase, variant, seed] = {
                    key: json.loads((folder / filename).read_text())
                    for key, filename in (("completed", "completed.json"), ("plan", "plan.json"),
                                          ("initial", "initial.json"), ("mechanism", "final.json"),
                                          ("full", "full_val.json"), ("audit", "handoff_audit.json"))}
    checks = []
    for phase in ("public", "carla"):
        for seed in SEEDS:
            reference = records[phase, "v5", seed]
            for variant in VARIANTS:
                record = records[phase, variant, seed]
                for key in ("training", "validation", "schedule", "checkpoint_sha256"):
                    if record["plan"][key] != reference["plan"][key]:
                        raise ValueError(f"{phase}/{seed}/{variant}的{key}不相同")
                if not record["completed"]["frozen_encoder_bridge_unchanged"]:
                    raise ValueError("冻结编码器改变")
                if not all(record["completed"]["checkpoint_checks"].values()):
                    raise ValueError("权重恢复或标签隔离失败")
                for domain, initial in record["initial"]["datasets"].items():
                    old = reference["initial"]["datasets"][domain]
                    if initial["raw"]["stages_stride8"]["coarse"] != old["raw"]["stages_stride8"]["coarse"]:
                        raise ValueError(f"{phase}/{seed}/{variant}初始粗面不一致")
                for domain, values in record["full"]["datasets"].items():
                    old = reference["full"]["datasets"][domain]
                    if (values["manifest_sha256"], values["frames"]) != (old["manifest_sha256"], old["frames"]):
                        raise ValueError("完整VAL清单不一致")
                    for region in ("full", "near", "mid", "far"):
                        if values["raw"]["depth"][region]["gt_pixels"] != old["raw"]["depth"][region]["gt_pixels"]:
                            raise ValueError("完整VAL的GT分母不一致")
                checks.append({"phase": phase, "seed": seed, "variant": variant,
                               "schedule_sha256": record["completed"]["schedule_sha256"],
                               "steps": record["completed"]["steps"],
                               "exposures": record["completed"]["exposures"],
                               "trainable_parameters": record["completed"]["trainable_parameters"],
                               "gate_gradient_max": record["completed"]["gate_gradient_max"]})
    report = {"completed": True, "test_accessed": False, "seeds": list(SEEDS),
              "checks": checks, "datasets": {}, "notes": [
                  "误差先按每seed全部有效GT像素汇总，再对3seed作算术均值和样本标准差(ddof=1)。",
                  "相机/数据域分别报告，scene_unit不标成米，不跨单位平均MAE。",
                  "共同V5权重起点的冻结编码器适配；不是从零训练或全预算上限对照。",
                  "v5与parent_oldpoint用原点目标，其余三组用相同族目标；与v5的差异不能全部归因于门控。",
                  "constant是1参数可学习空间常数门；parent是129参数、读取质量统计的空间自适应门。",
                  "末步固定保存，完整VAL只评估不选epoch、seed或方法；未访问TEST。",
              ]}
    rows = []
    for phase in ("public", "carla"):
        names = records[phase, "v5", SEEDS[0]]["full"]["datasets"]
        for domain in names:
            reference = names[domain]
            item = {"phase": phase, "units": reference["units"], "frames": reference["frames"],
                    "full_validation": {}, "mechanism_validation": {}, "paired_candidate_intersections": {}}
            for variant in VARIANTS:
                full = {}
                for stage in ("depth", "coarse_stride8", "fine_stride8", "final_stride8"):
                    full[stage] = {}
                    for region in ("full", "near", "mid", "far"):
                        metrics = {}
                        for metric in ("MAE", "RMSE", "AbsRel", "coverage"):
                            values = [records[phase, variant, seed]["full"]["datasets"][domain]["metrics"][stage][region][metric]
                                      for seed in SEEDS]
                            if any(value is None for value in values):
                                metrics[metric] = None
                                continue
                            metrics[metric] = aggregate(values)
                            base = [records[phase, "v5", seed]["full"]["datasets"][domain]["metrics"][stage][region][metric]
                                    for seed in SEEDS]
                            metrics[metric]["paired_delta_vs_v5"] = aggregate([a - b for a, b in zip(values, base)])
                            rows.append({"dataset": domain, "units": reference["units"], "variant": variant,
                                         "stage": stage, "region": region, "metric": metric,
                                         "mean": metrics[metric]["mean"], "sample_std": metrics[metric]["sample_std"],
                                         **{str(seed): value for seed, value in zip(SEEDS, values)}})
                        full[stage][region] = metrics
                item["full_validation"][variant] = full
                mechanism = {}
                first = records[phase, variant, SEEDS[0]]["mechanism"]["datasets"][domain]
                for region, values in first["metrics"]["candidates_stride8"].items():
                    mechanism[region] = {}
                    for metric in values:
                        if metric == "pixels":
                            continue
                        data = [records[phase, variant, seed]["mechanism"]["datasets"][domain]["metrics"]["candidates_stride8"][region][metric]
                                for seed in SEEDS]
                        mechanism[region][metric] = aggregate(data) if all(v is not None for v in data) else None
                item["mechanism_validation"][variant] = mechanism
                intersections = {}
                for region in ("full", "near", "mid", "far"):
                    pairs = [records[phase, variant, seed]["full"]["datasets"][domain]["paired_candidates_stride8"][region]
                             for seed in SEEDS]
                    joint = {"raw_per_seed": {str(seed): pair["raw"] for seed, pair in zip(SEEDS, pairs)}}
                    for label in ("baseline_destroyed_on_common_good", "method_destroyed_on_common_good",
                                  "baseline_corrected_on_common_bad", "method_corrected_on_common_bad"):
                        values = [pair[label] for pair in pairs]
                        joint[label] = aggregate(values) if all(v is not None for v in values) else None
                    for label, method, baseline in (
                        ("destroyed_rate_paired_delta", "method_destroyed_on_common_good", "baseline_destroyed_on_common_good"),
                        ("corrected_rate_paired_delta", "method_corrected_on_common_bad", "baseline_corrected_on_common_bad"),
                    ):
                        values = [pair[method] - pair[baseline] if pair[method] is not None and pair[baseline] is not None else None
                                  for pair in pairs]
                        joint[label] = aggregate(values) if all(v is not None for v in values) else None
                    intersections[region] = joint
                item["paired_candidate_intersections"][variant] = intersections
            report["datasets"][domain] = item
    write_json(OUTPUT / "summary.json", report)
    with (OUTPUT / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 候选交接同预算三随机种子消融", "", "以下均为固定末步的完整VAL，均值±样本标准差；不用于TEST挑选。", "",
             "| 数据域 | 单位 | V5 | 族边缘化 | 常数门 | 自适应门 | 自适应门+旧目标 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, record in report["datasets"].items():
        values = [record["full_validation"][variant]["depth"]["full"]["MAE"] for variant in VARIANTS]
        cells = [f"{value['mean']:.4f}±{value['sample_std']:.4f}" for value in values]
        lines.append("| " + " | ".join([name, record["units"], *cells]) + " |")
    lines.extend(["", *["- " + note for note in report["notes"]], "", "全部逐seed、分段、阶段与覆盖率见metrics.csv与summary.json。"])
    (OUTPUT / "三种子消融结果.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": True, "domains": len(report["datasets"]), "rows": len(rows)}))


if __name__ == "__main__":
    run()
