"""短试后独占GPU比较完整RGB到深度延迟；不使用187的算子时间外推。"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.rulermvs_v5_20261031.pilot import load_model, path
from research.ruler_flow_v2_20261031.latency import shared_original
from tools.smoke_outdoor_datasets import offline_only


@torch.no_grad()
def run(config):
    offline_only()
    output = path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    seed_all(20261031)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    result = {"gpu": torch.cuda.get_device_name(), "dtype": "FP32", "tf32": False,
              "warmup": 10, "iterations": 40,
              "scope": "GPU常驻RGB到深度，含两编码器/桥/候选解码；不含读盘与CPU到GPU传输",
              "groups": {}}
    for group in config["groups"]:
        settings = json.loads(path(group["settings"]).read_text())
        data_config = json.loads(path(settings["data_config"]).read_text())
        dataset = MixedDataset(data_config, "val")
        domain = dataset.domains[0]
        inputs = move(collate_mixed([dataset[(0, 0, -1)]])["inputs"], device)
        models = {name: load_model(filename, device)[0] for name, filename in group["methods"].items()}
        calls = {name: (lambda model=model: model(inputs, domain)) for name, model in models.items()}
        if group.get("shared_references", False):
            refs = list(range(inputs["rgb"].shape[1]))
            for name, model in models.items():
                if hasattr(model, "forward_references"):
                    calls[name + "_shared_four"] = lambda model=model: model.forward_references(inputs, domain, refs)
                else:
                    calls[name + "_shared_four"] = lambda model=model: shared_original(model, inputs, domain, refs)
        for call in calls.values():
            for _ in range(result["warmup"]):
                call()
        values = {name: [] for name in calls}
        for iteration in range(result["iterations"]):
            names = list(calls)
            names = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in names:
                torch.cuda.synchronize()
                start = time.perf_counter()
                calls[name]()
                torch.cuda.synchronize()
                values[name].append((time.perf_counter() - start) * 1000)
        result["groups"][group["name"]] = {
            "methods": {name: {"median_ms": statistics.median(v), "p10_ms": sorted(v)[3],
                                "p90_ms": sorted(v)[35], "Hz": 1000 / statistics.median(v)}
                        for name, v in values.items()},
            "weight_sha256": {name: digest(path(filename)) for name, filename in group["methods"].items()},
            "views": inputs["rgb"].shape[1], "rgb_size": list(inputs["rgb"].shape[-2:]),
            "matching_size": list(inputs["matching_rgb"].shape[-2:]),
        }
        write_json(output, result)
        del models, calls, dataset, inputs
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
