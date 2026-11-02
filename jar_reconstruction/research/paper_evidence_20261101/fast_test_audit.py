"""完整快速TEST结束后，审计共同分母、真实指标漂移与编译日志。"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.paper_evidence_20261101.fast_test import read, verify
from research.paper_analysis_20261101 import analyze as plots


def main():
    base = ROOT / "runs/active/rulermvs_v6_20261101/full_training/paper_fast_test"
    if not read(base / "completed.json")["completed"] or (base / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("快速TEST没有真实成功结束")
    plan = read(base / "plan.json")
    verify(plan)
    rows = []
    frame_rows = []
    for phase, settings in plan["groups"].items():
        fast_folder = base / phase / "full_test"
        eager_folder = Path(settings["eager_folder"])
        fast, eager = read(fast_folder / "summary.json"), read(eager_folder / "summary.json")
        for name, domain in fast["datasets"].items():
            value = domain["methods"]["v6_fast"]
            reference = eager["datasets"][name]["methods"]["v6_best"]
            if value["frames"] != reference["frames"]:
                raise ValueError("fast/eager样本数不同")
            for scope in ("depth", "stages_stride8"):
                for region, metrics in value[scope].items():
                    other = reference[scope][region]
                    if metrics["gt_pixels"] != other["gt_pixels"]:
                        raise ValueError("fast/eager GT分母不相同")
                    for metric in ("MAE", "RMSE", "AbsRel", "coverage"):
                        a, b = metrics[metric], other[metric]
                        rows.append({"phase": phase, "dataset": name, "units": domain["units"],
                                     "scope": scope, "region": region, "metric": metric, "fast": a, "eager": b,
                                     "delta": None if a is None else a - b,
                                     "relative_change_percent": None if a is None or not b else 100 * (a / b - 1)})
            frames = {}
            fingerprints = []
            for folder, method in ((eager_folder, "v6_best"), (fast_folder, "v6_fast")):
                signs = {}
                for part in (folder / name).glob("part_*.json"):
                    for frame in read(part)["frames"]:
                        values = frame["methods"][method]["depth"]["full"]
                        frames.setdefault(frame["index"], {})[method] = values
                        if "input_sha256" in frame:
                            signs[frame["index"]] = frame["input_sha256"]
                fingerprints.append(signs)
            if not fingerprints[0] or fingerprints[0] != fingerprints[1]:
                raise ValueError("fast/eager首中末输入张量指纹不同")
            if len(frames) != value["frames"]:
                raise ValueError("逐帧统计不完整")
            for index, values in frames.items():
                a, b = values["v6_fast"], values["v6_best"]
                if a["gt_pixels"] != b["gt_pixels"]:
                    raise ValueError("逐帧GT分母不符")
                frame_rows.append({"phase": phase, "dataset": name, "index": index, "units": domain["units"],
                                   "fast_mae": a["abs_sum"] / a["predicted_pixels"] if a["predicted_pixels"] else None,
                                   "eager_mae": b["abs_sum"] / b["predicted_pixels"] if b["predicted_pixels"] else None,
                                   "fast_coverage": a["predicted_pixels"] / a["gt_pixels"] if a["gt_pixels"] else None,
                                   "eager_coverage": b["predicted_pixels"] / b["gt_pixels"] if b["gt_pixels"] else None})
    log = (base / "run.log").read_text()
    pattern = re.compile(r"fallback|falling back|\beager\b|cache_size_limit|recompile_limit|CacheLimitExceeded|WON.T CONVERT|graph.?break|Traceback|TorchRuntimeError", re.IGNORECASE)
    suspicious = [{"line": index, "text": line} for index, line in enumerate(log.splitlines(), 1) if pattern.search(line)]
    audit = {"completed": True, "full_test_reference_samples": len(frame_rows),
             "same_gt_denominators": True, "sampled_input_fingerprints_equal": True,
             "deployment_options": plan["fast"], "compilation_fallback_or_error_log_entries": suspicious,
             "no_fallback_notice_observed": not suspicious,
             "online_softmax_strategy_warnings": log.count("Online softmax is disabled on the fly"),
             "log_sha256": plots.sha(base / "run.log"), "source_sha256": plots.sha(__file__),
             "limits": "完整TEST误差漂移不是逐像素预测差；未做新测速。无回退日志不等于所有网络算子均被编译，部署只编译预先指定的两段。"}
    plots.csv_write(base / "fast_vs_eager_metrics.csv", rows)
    plots.csv_write(base / "fast_vs_eager_frames.csv", frame_rows)
    plots.write_json(base / "completion_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
