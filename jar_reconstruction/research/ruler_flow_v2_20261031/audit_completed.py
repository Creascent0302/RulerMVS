"""只读核验训练/TEST完整性、分母、权重和源码；不执行模型或修改权重。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def training_audit(base, phase):
    directory = base / phase
    events = records(directory / "events.jsonl")
    config = read(directory / "config.json")
    expected = 18370 if phase == "public" else 384
    check((directory / "exit_code.txt").read_text().strip() == "0", f"{phase}:退出异常")
    check(events[-1]["event"] == "complete", f"{phase}:没有结束事件")
    complete_stages = [row for row in events if row["event"] == "stage_complete"]
    check(len(complete_stages) == len(config["stages"]), f"{phase}:阶段缺失")
    full_validations = [row for row in events if row["event"] == "validation" and row["full"]]
    check(len(full_validations) == sum(s["max_epochs"] for s in config["stages"]), f"{phase}:完整VAL轮数不符")
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False, mmap=True)
    final = torch.load(directory / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    check(best["architecture"] == final["architecture"] == "ruler_flow_v2", "新架构标记不符")
    check(final["global_step"] == expected, f"{phase}:训练预算未完成")
    check(best["validation"]["split"] == "val" and best["validation"]["full"], "best非完整VAL选模")
    score = min(row["selection_risk"] for row in full_validations)
    check(math.isclose(score, best["selection_score"], rel_tol=0, abs_tol=1e-12), "best并非记录的最优完整VAL")
    check(best["config"] == final["config"] == config, "权重配置不一致")
    unequal = [name for name in best["model"] if not torch.equal(best["model"][name], final["model"][name])]
    return {"exit_code": 0, "start_utc": next(row["time_utc"] for row in events if row["event"] == "start"),
            "complete_utc": events[-1]["time_utc"], "final_step": final["global_step"],
            "best_step": best["global_step"], "best_stage": best["stage"], "best_epoch": best["epoch"],
            "full_validation_count": len(full_validations), "best_selection_score": score,
            "best_final_different_tensors": len(unequal), "best_final_identical": not unequal,
            "train_exposures": last["state"]["exposures"]}


def split_audit(domain):
    data = {split: records(path) for split, path in domain["manifests"].items()}
    keys = {split: {(str(row.get("sequence")), str(row["frame_id"])) for row in rows}
            for split, rows in data.items()}
    image_paths = {split: {str(camera["rgb"]) for row in rows for camera in row["cameras"] if camera.get("rgb")}
                   for split, rows in data.items()}
    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{a}_{b}"] = {"tuple_keys": len(keys[a] & keys[b]),
                                 "all_view_rgb_paths": len(image_paths[a] & image_paths[b])}
    return {"rows": {split: len(rows) for split, rows in data.items()}, "overlap": overlap,
            "test_sequence_count": len({row.get("sequence") for row in data["test"]}),
            "units": domain["geometry_units"], "views": domain["num_views"]}


def group_audit(base, group):
    directory = base / "evaluation" / group / "full_test"
    summary = read(directory / "summary.json")
    audit = read(directory / "audit.json")
    check(summary["completed"] and summary["split"] == "test", f"{group}:非完成TEST")
    protocol = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    check(protocol == summary["protocol"], f"{group}:协议hash不符")
    domains = {}
    frame_total = 0
    part_total = 0
    for domain in audit["config"]["datasets"]:
        name = domain["name"]
        expected = len(records(domain["manifests"]["test"]))
        parts = sorted((directory / name).glob("part_*.json"))
        indices = []
        for filename in parts:
            part = read(filename)
            check(part["completed"] and part["protocol"] == protocol, f"{filename}:失败或协议不符")
            for frame in part["frames"]:
                indices.append(frame["index"])
                methods = frame["methods"]
                check(set(methods) == set(audit["config"]["methods"]), f"{filename}:方法缺失")
                for region in ("full", "near", "mid", "far", "gradient_band", "smooth"):
                    stats = [value["depth"][region] for value in methods.values()]
                    check(len({value["gt_pixels"] for value in stats}) == 1, f"{filename}:GT分母不一致")
                    check(all(value["predicted_pixels"] == value["gt_pixels"] for value in stats),
                          f"{filename}:有拒绝预测的GT像素")
            for method, value in part["totals"].items():
                check(value["indices"] == [f["index"] for f in part["frames"]], f"{filename}:汇总索引不符")
        check(sorted(indices) == list(range(expected)), f"{name}:TEST有缺失或重复")
        check(all(value["frames"] == expected for value in summary["datasets"][name]["methods"].values()),
              f"{name}:汇总帧数不符")
        domains[name] = {"frames": expected, "parts": len(parts), "splits": split_audit(domain),
                         "all_gt_denominators_equal": True, "all_gt_covered": True}
        frame_total += expected
        part_total += len(parts)
    return {"completed": True, "frames": frame_total, "parts": part_total,
            "method_predictions": frame_total * len(audit["config"]["methods"]), "domains": domains}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = ROOT / "runs/active/ruler_flow_v2_20261031"
    check((base / "queue.exit_code.txt").read_text().strip() == "0", "队列非正常退出")
    plan = read(base / "evaluation/plan.json")
    hashes = {path: digest(ROOT / path) for path in plan["weight_sha256"]}
    check(hashes == plan["weight_sha256"], "评测后权重文件发生改变")
    launch = read(base / "launch_audit.json")
    source_changes = {}
    for name, expected in launch["source_sha256"].items():
        path = ROOT / name
        if not path.is_file():
            source_changes[name] = "missing"
        elif digest(path) != expected:
            source_changes[name] = "changed"
    critical = ["jar/pretraining/model.py", "jar/pretraining/encoders.py", "jar/pretraining/runner.py",
                "jar/pretraining/losses.py", "jar/pretraining/data.py", "jar/models/ruler_direct.py",
                "jar/models/ruler_proposals.py", "research/ruler_flow_v2_20261031/model.py",
                "research/ruler_flow_v2_20261031/field.py", "research/ruler_flow_v2_20261031/losses.py",
                "research/ruler_flow_v2_20261031/training.py"]
    check(not any(name in source_changes for name in critical), "关键模型/训练源码发生改变")
    evaluation_sources = plan["extension_source_sha256"]
    check(all(digest(ROOT / name) == expected for name, expected in evaluation_sources.items()),
          "评测新模型源码发生改变")
    report = {"completed": True, "queue_exit_code": 0,
              "training": {phase: training_audit(base, phase) for phase in ("public", "carla_new")},
              "test": {group: group_audit(base, group) for group in ("public", "carla", "carla_other_refs")},
              "weights_match_evaluation_record": True, "critical_source_unchanged": True,
              "source_changes_since_launch": source_changes,
              "protocol_limits": [
                  "自研matching输入512x896、主图256x448；官方baseline主图256x448，native并非等像素预算。",
                  "V4从自研V2热启动后增加训练预算，不是等预算纯架构实验。",
                  "MVS-Synth/BlendedMVS/DynamicReplica仅scene_unit，不合并成米制均值。",
                  "静态公共四图tuple不等于同步四相机；VKITTI/Dynamic为真实同步双目。",
                  "表面指标仅单参考可见GT区域，1024查询点对完整另一侧点集，非完整场景融合。",
                  "近距米制<10m，非米制<5*TRAINdepth_scale；CARLAGT限制2–80m。",
                  "baseline为官方Hero首遍+一次范围细化，使用共同TRAIN初始范围，不是原生zero-shot范围。",
                  "未认证官方baseline历史预训练与这些公开测试场景无重叠，不宣称完全未见域。",
              ]}
    write_json(args.output, report)
    print(json.dumps({"training": report["training"],
                      "test_counts": {k: {x: v[x] for x in ("frames", "parts", "method_predictions")}
                                      for k, v in report["test"].items()},
                      "source_changes": source_changes,
                      "split_overlaps": {n: d["splits"]["overlap"] for g in report["test"].values()
                                         for n, d in g["domains"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
