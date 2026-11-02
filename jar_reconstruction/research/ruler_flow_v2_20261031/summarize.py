"""汇总固定末步短试；全图与共同stride8阶段指标分别记录。"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"


def read(filename):
    return json.loads(Path(filename).read_text())


def main():
    rows, summary, initial_checks = [], {}, {}
    for phase in ("public", "carla_new"):
        folders = {name: OUTPUT / "pilot" / (phase + "_" + name)
                   for name in ("control", "direct", "meanflow")}
        if not all((folder / "completed.json").exists() for folder in folders.values()):
            raise RuntimeError(f"{phase}的全部正式短试尚未结束")
        initial = read(folders["control"] / "initial.json")
        variants = {"frozen": initial}
        for name, folder in folders.items():
            initial_checks[phase + "_" + name] = read(folder / "initial.json") == initial
            variants[name] = read(folder / "final.json")
        if not all(initial_checks.values()):
            raise RuntimeError("新头零初始化没有保持原起点指标，需分析后再比较")
        summary[phase] = variants
        for method, partitions in variants.items():
            for partition, domains in partitions.items():
                for name, metrics in domains.items():
                    base = {"phase": phase, "partition": partition, "dataset": name,
                            "method": method, "units": metrics["units"], "frames": metrics["frames"]}
                    for region, values in metrics["depth"].items():
                        rows.append(dict(base, stage="full_resolution_depth", region=region, **values))
                    for stage, regions in metrics["stages_stride8"].items():
                        for region, values in regions.items():
                            rows.append(dict(base, stage="stride8_" + stage, region=region, **values))
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("phase", "partition", "dataset", "stage", "region"))
        groups.setdefault(key, set()).add(row["gt_pixels"])
    if any(len(counts) != 1 for counts in groups.values()):
        raise RuntimeError("共同GT分母不一致")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / "pilot_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"completed": True, "initial_identical": initial_checks,
              "common_gt_denominators": True, "groups": summary,
              "scope": "历史固定16VAL分8开发/8确认、固定末步；不是TEST，不做全域单位混合均值",
              "stage_definition": "depth为全图，coarse/fine/final均共同stride8抽样。"}
    (OUTPUT / "pilot_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for phase, methods in summary.items():
        print(phase, "确认8VAL，逐域 full MAE / near MAE")
        for domain in methods["frozen"]["confirmation"]:
            values = []
            for method in ("frozen", "control", "direct", "meanflow"):
                metric = methods[method]["confirmation"][domain]["depth"]
                values.append(f"{method}={metric['full']['MAE']:.6f}/{metric['near']['MAE']:.6f}")
            print(domain, "; ".join(values))


if __name__ == "__main__":
    main()
