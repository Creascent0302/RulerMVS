"""只训练新增参数的结果，复用本轮已固定fresh清单，不重新选帧。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import seed_all
from research.near_evidence_20261030.pilot import evaluate, path
from research.role_fusion_20261030.fresh_validation import gate_recorder, summarize
from research.role_fusion_20261030.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


def run(settings):
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    group = settings["group"]
    study = ROOT / "runs/active/role_fusion_20261030"
    previous = study / f"fresh_{group}"
    previous_plan = json.loads((previous / "plan.json").read_text())
    training = json.loads((study / f"{group}_both_adapters/settings.json").read_text())
    source = json.loads(path(training["data_config"]).read_text())
    config = copy.deepcopy(source)
    config["datasets"] = copy.deepcopy(source.get("validation_datasets", source["datasets"]))
    config.pop("validation_datasets", None)
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
        if digest(path(domain["manifests"]["val"])) != previous_plan["validation"][domain["name"]]["manifest_sha256"]:
            raise RuntimeError("原fresh VAL manifest已经改变")
    methods = {name: previous_plan["methods"][name] for name in ("frozen", "control", "both")}
    methods["both_adapters"] = str((study / f"{group}_both_adapters/final.pt").relative_to(ROOT))
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖结果：{output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "validation": previous_plan["validation"], "methods": methods,
        "weight_sha256": {name: digest(path(filename)) for name, filename in methods.items()},
        "previous_plan_sha256": digest(previous / "plan.json"),
        "test_accessed": False, "training_performed": False,
        "scope": "沿用本轮已经查看的fresh VAL相同索引；不重新选样，不称盲测。旧三权重结果原样引用。",
    }
    write_json(output / "plan.json", plan)
    current_plan_sha = digest(output / "plan.json")
    for name in ("frozen", "control", "both"):
        filename = previous / f"{name}.json"
        record = json.loads(filename.read_text())
        if record["weight_sha256"] != plan["weight_sha256"][name]:
            raise RuntimeError(f"{name}原权重改变，不能引用旧结果")
        write_json(output / f"{name}.json", {
            **record, "plan_sha256": current_plan_sha,
            "reused_result": str(filename.relative_to(ROOT)), "reused_result_sha256": digest(filename),
        })
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    model, _ = load_model(methods["both_adapters"], device)
    dataset = MixedDataset(config, "val")
    statistics = {}
    handle = model.register_forward_hook(gate_recorder(statistics))
    try:
        metrics = evaluate(model, dataset, plan["validation"], device)
    finally:
        handle.remove()
    write_json(output / "both_adapters.json", {
        "metrics": metrics, "weight_sha256": plan["weight_sha256"]["both_adapters"],
        "plan_sha256": current_plan_sha, "test_accessed": False,
    })
    write_json(output / "both_adapters_gate_statistics.json", {"domains": statistics})
    summarize(output, plan)
    print(f"{group}：沿用原fresh计划完成新增权重评测", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
