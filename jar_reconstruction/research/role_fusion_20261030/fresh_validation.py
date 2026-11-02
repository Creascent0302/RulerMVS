"""排除两批历史16VAL后的固定新索引验证；同序列，不宣称盲测新场景。"""

import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import seed_all
from research.near_evidence_20261030.pilot import evaluate, path, payload_hash
from research.role_fusion_20261030.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


def prepare(settings):
    group = settings["group"]
    runs = ROOT / "runs/active/role_fusion_20261030"
    control = runs / f"{group}_control"
    old_path = ROOT / f"runs/active/near_evidence_20261030/{group}_fresh_confirmation/plan.json"
    first_path = control / "plan.json"
    first = json.loads(first_path.read_text())["validation"]
    second = json.loads(old_path.read_text())["validation"]
    original = json.loads((control / "settings.json").read_text())
    source = json.loads(path(original["data_config"]).read_text())
    config = copy.deepcopy(source)
    config["datasets"] = copy.deepcopy(source.get("validation_datasets", source["datasets"]))
    config.pop("validation_datasets", None)
    methods = {"frozen": original["checkpoint"]}
    methods.update({name: str((runs / f"{group}_{name}/final.pt").relative_to(ROOT))
                    for name in ("control", "context", "carry", "both")})
    selected = {}
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
        filename = path(domain["manifests"]["val"])
        manifest_sha = digest(filename)
        name = domain["name"]
        if manifest_sha != first[name]["manifest_sha256"] or manifest_sha != second[name]["manifest_sha256"]:
            raise RuntimeError(f"{name}: 历史清单和当前VAL不同，不能静默只按索引排除")
        length = sum(bool(line.strip()) for line in filename.read_text().splitlines())
        excluded = set(first[name]["indices"]) | set(second[name]["indices"])
        remaining = [index for index in range(length) if index not in excluded]
        if not remaining:
            raise RuntimeError(f"{name}: 排除后没有剩余VAL")
        count = min(16, len(remaining))
        locations = torch.linspace(0, len(remaining) - 1, count).long().tolist()
        indices = [remaining[i] for i in locations]
        selected[name] = {
            "indices": indices, "development": [], "confirmation": indices,
            "excluded_first": first[name]["indices"], "excluded_second": second[name]["indices"],
            "excluded_union": sorted(excluded), "val_length": length,
            "remaining_count": len(remaining), "actual_count": count,
            "manifest_sha256": manifest_sha, "selection_sha256": payload_hash(indices),
        }
    plan = {
        "validation": selected, "methods": methods,
        "weight_sha256": {name: digest(path(filename)) for name, filename in methods.items()},
        "previous_plan_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (first_path, old_path)},
        "source_sha256": {name: digest(ROOT / name) for name in (
            "research/role_fusion_20261030/fresh_validation.py",
            "research/role_fusion_20261030/variants.py",
            "research/role_fusion_20261030/pilot.py",
            "research/near_evidence_20261030/pilot.py",
        )},
        "group": group, "test_accessed": False, "training_performed": False,
        "scope": "排除本轮16及实验163另16的VAL索引；仍来自既有VAL序列，相邻帧相关，不是新场景盲测。固定全部模型，不按结果改权重。",
    }
    output = path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    filename = output / "plan.json"
    if filename.exists():
        if json.loads(filename.read_text()) != plan:
            raise RuntimeError("已有fresh计划与当前配置/源码/权重不同，拒绝覆盖")
    else:
        write_json(filename, plan)
    return config, plan, output


def summarize(output, plan):
    results = {name: json.loads((output / f"{name}.json").read_text())["metrics"]["confirmation"]
               for name in plan["methods"]}
    rows = []
    for dataset in plan["validation"]:
        for level in ("depth", "coarse", "fine", "final"):
            def measures(method):
                record = results[method][dataset]
                return record["depth"] if level == "depth" else record["stages_stride8"][level]
            for region in measures("control"):
                counts = {method: measures(method)[region]["gt_pixels"] for method in results}
                if len(set(counts.values())) != 1:
                    raise RuntimeError(f"{dataset}/{level}/{region}: GT分母不一致")
                for metric in ("MAE", "RMSE", "AbsRel", "coverage"):
                    control = measures("control")[region][metric]
                    frozen = measures("frozen")[region][metric]
                    for method in results:
                        value = measures(method)[region][metric]
                        rows.append({
                            "dataset": dataset, "units": results[method][dataset]["units"],
                            "frames": results[method][dataset]["frames"], "method": method,
                            "level": level, "region": region, "metric": metric,
                            "value": value, "control": control, "frozen": frozen,
                            "relative_to_control": value / control - 1 if value is not None and control else None,
                            "relative_to_frozen": value / frozen - 1 if value is not None and frozen else None,
                        })
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"completed": True, "metrics": results,
                                          "scope": plan["scope"], "plan_sha256": digest(output / "plan.json")})


def gate_recorder(records):
    """只观察已有输出，不改变前向；统计有效参考格点，不按GT或误差筛选。"""
    def record(module, arguments, prediction):
        fine = prediction["fine"]
        if "handoff_gate" not in fine:
            return
        name = arguments[1]["name"]
        eta = fine["handoff_gate"][:, 0].detach()[fine["valid"].any(1)]
        if not eta.numel():
            return
        current = records.setdefault(name, {"count": 0, "sum": 0., "min": 1., "max": 0.})
        current["count"] += eta.numel()
        current["sum"] += float(eta.double().sum())
        current["min"] = min(current["min"], float(eta.min()))
        current["max"] = max(current["max"], float(eta.max()))
        current["mean"] = current["sum"] / current["count"]
    return record


def run(settings, prepare_only=False):
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    config, plan, output = prepare(settings)
    print(json.dumps({"prepared": settings["group"], "counts": {name: p["actual_count"]
                          for name, p in plan["validation"].items()}}, ensure_ascii=False), flush=True)
    if prepare_only:
        return
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset = MixedDataset(config, "val")
    for name, filename in plan["methods"].items():
        destination = output / f"{name}.json"
        if destination.exists():
            old = json.loads(destination.read_text())
            if old["plan_sha256"] != digest(output / "plan.json"):
                raise RuntimeError("已有结果不是本次固定计划")
            continue
        model, _ = load_model(filename, device)
        statistics = {}
        handle = model.register_forward_hook(gate_recorder(statistics))
        try:
            metrics = evaluate(model, dataset, plan["validation"], device)
        finally:
            handle.remove()
        write_json(destination, {"metrics": metrics, "weight_sha256": plan["weight_sha256"][name],
                                 "plan_sha256": digest(output / "plan.json"), "test_accessed": False})
        if statistics:
            write_json(output / f"{name}_gate_statistics.json", {
                "domains": statistics, "plan_sha256": digest(output / "plan.json"),
                "scope": "fine参考有效格点的eta，无GT筛选；只观察原模型输出，不新增前向。",
            })
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"completed": settings["group"], "method": name}, ensure_ascii=False), flush=True)
    summarize(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    run(json.loads(path(arguments.config).read_text()), arguments.prepare_only)
