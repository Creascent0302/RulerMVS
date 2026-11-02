"""固定三权重在未用于首轮诊断的互斥VAL帧上确认；不是新序列或TEST。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset
from jar.pretraining.model import PretrainingModel
from research.near_evidence_20261030.variants import install_variants
from jar.runtime import seed_all
from research.near_evidence_20261030.pilot import evaluate, path, payload_hash
from tools.smoke_outdoor_datasets import offline_only


def prepare(settings):
    output = path(settings["output"])
    source = json.loads(path(settings["data_config"]).read_text())
    source["datasets"] = source.get("validation_datasets", source["datasets"])
    source.pop("validation_datasets", None)
    for domain in source["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
    old_plan = json.loads(path(settings["old_plan"]).read_text())["validation"]
    validation = {}
    for domain in source["datasets"]:
        filename = path(domain["manifests"]["val"])
        length = sum(bool(line.strip()) for line in filename.read_text().splitlines())
        old = old_plan[domain["name"]]
        assert digest(filename) == old["manifest_sha256"]
        remaining = [index for index in range(length) if index not in old["indices"]]
        offsets = torch.linspace(0, len(remaining) - 1, min(16, len(remaining))).long().tolist()
        selected = [remaining[index] for index in offsets]
        assert not set(selected) & set(old["indices"])
        validation[domain["name"]] = {"indices": selected, "development": [], "confirmation": selected,
            "excluded_previous_indices": old["indices"], "manifest_sha256": digest(filename),
            "val_length": length, "selection_sha256": payload_hash(selected)}
    plan = {"validation": validation, "methods": settings["methods"],
            "weights_sha256": {name: digest(path(filename)) for name, filename in settings["methods"].items()},
            "source_sha256": {name: digest(ROOT / name) for name in (
                "research/near_evidence_20261030/fresh_confirmation.py", "research/near_evidence_20261030/pilot.py",
                "jar/pretraining/model.py", "research/near_evidence_20261030/ray_evidence.py",
                "research/near_evidence_20261030/variants.py")},
            "rule": "先排除首轮16VAL索引，再对剩余索引均匀floor取16；不读GT选择。",
            "scope": "同VAL序列内未用于此前诊断的小样本确认，不是独立新序列盲测，不是TEST。",
            "test_accessed": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        assert json.loads((output / "plan.json").read_text()) == plan
    else:
        write_json(output / "plan.json", plan)
        write_json(output / "settings.json", settings)
    return source, plan


def run(settings, source, plan):
    offline_only()
    output = path(settings["output"])
    if (output / "metrics.json").exists():
        raise FileExistsError("确认结果已存在，不覆盖")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(settings.get("seed", 20261030))
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    dataset = MixedDataset(source, "val")
    results = {}
    start = time.monotonic()
    for name, filename in settings["methods"].items():
        saved = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
        model = PretrainingModel(saved["config"], initialize=False)
        model.load_state_dict(saved["model"], strict=True)
        install_variants(model)
        model.requires_grad_(False).eval().to(device)
        install_position_cache(model.foundation.model)
        results[name] = evaluate(model, dataset, plan["validation"], device)["confirmation"]
        write_json(output / (name + ".json"), results[name])
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()
        print("确认完成：", name, flush=True)
    assert {name: digest(path(filename)) for name, filename in settings["methods"].items()} == plan["weights_sha256"]
    comparison = {}
    for domain in results["control"]:
        comparison[domain] = {}
        for region in ("full", "near", "mid", "far", "edge"):
            fixed = results["frozen"][domain]["depth"][region]
            control = results["control"][domain]["depth"][region]
            candidate = results["coarse_attention"][domain]["depth"][region]
            comparison[domain][region] = {"gt_pixels": control["gt_pixels"],
                "frozen_mae": fixed["MAE"], "control_mae": control["MAE"], "new_mae": candidate["MAE"],
                "relative_to_control": candidate["MAE"] / control["MAE"] - 1 if control["MAE"] else None}
    write_json(output / "metrics.json", {"completed": True, "methods": results, "comparison": comparison,
        "seconds": time.monotonic() - start, "scope": plan["scope"], "test_accessed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    settings = json.loads(path(args.config).read_text())
    source, plan = prepare(settings)
    print(json.dumps(plan["validation"], ensure_ascii=False, indent=2), flush=True)
    if not args.prepare_only:
        run(settings, source, plan)
