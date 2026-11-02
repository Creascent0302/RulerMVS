"""同GPU交替计时，显式区分单参考与顺序所有参考，不复用GT或遗漏积分步数。"""

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.ruler_flow_20261030.pilot import install, load_model, path, set_inference_steps
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists():
        raise FileExistsError(output)
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    config = json.loads(path(settings["data_config"]).read_text())
    config["datasets"] = config.pop("validation_datasets", config["datasets"])
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
    dataset = MixedDataset(config, "val")
    models = {}
    for name, spec in settings["methods"].items():
        with torch.inference_mode(False):
            model, _ = load_model(spec["checkpoint"], device)
            if spec.get("variant"):
                install(model, spec["variant"])
                model.requires_grad_(False).eval().to(device)
            set_inference_steps(model, spec.get("steps", 1))
            models[name] = model
    rng = random.Random(20261030)
    results = {}
    for domain_name in settings["datasets"]:
        domain_index = next(i for i, domain in enumerate(dataset.domains) if domain["name"] == domain_name)
        domain = dataset.domains[domain_index]
        length = len(dataset.datasets[domain_index])
        indices = torch.linspace(0, length - 1, min(3, length)).long().tolist()
        results[domain_name] = {}
        for mode in settings.get("reference_modes", ["single", "all"]):
            durations = {name: [] for name in models}
            extra_memory = {name: [] for name in models}
            view_counts, calls = [], []
            for index in indices:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                inputs = batch["inputs"]
                references = [0] if mode == "single" else list(range(inputs["rgb"].shape[1]))
                calls.append(len(references))
                view_counts.append(int(inputs["view_mask"].sum()))

                def forward(model):
                    for reference in references:
                        prediction = model(inputs, domain, reference=reference)
                        del prediction

                for model in models.values():
                    for _ in range(settings.get("warmup", 10)):
                        forward(model)
                    if model.foundation.model._frozen_position_cache.key is None:
                        raise RuntimeError("冻结位置编码缓存未命中")
                torch.cuda.synchronize(device)
                for _ in range(settings.get("repeats", 40)):
                    order = list(models)
                    rng.shuffle(order)
                    for name in order:
                        torch.cuda.synchronize(device)
                        before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start = time.perf_counter()
                        forward(models[name])
                        torch.cuda.synchronize(device)
                        durations[name].append((time.perf_counter() - start) * 1000)
                        extra_memory[name].append((torch.cuda.max_memory_allocated(device) - before) / 2**30)
                del batch, inputs
            results[domain_name][mode] = {
                "indices": indices, "valid_views": view_counts, "network_calls_per_measurement": calls,
                "methods": {name: {
                    "median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)),
                    "p95_ms": float(np.percentile(times, 95)), "frequency_hz": 1000 / float(np.median(times)),
                    "additional_allocated_gib": max(extra_memory[name]), "measurements": len(times),
                } for name, times in durations.items()},
            }
            print(json.dumps({"completed": domain_name, "reference_mode": mode,
                              "median_ms": {name: value["median_ms"] for name, value
                                            in results[domain_name][mode]["methods"].items()}}), flush=True)
    write_json(output, {
        "completed": True, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "precision": "FP32, TF32 disabled", "results": results, "methods": settings["methods"],
        "weights_sha256": {name: digest(path(spec["checkpoint"])) for name, spec in settings["methods"].items()},
        "scope": "GPU常驻相同输入交替计时，single只参考0；all顺序所有参考，不是场景融合。不含读盘/传输/预处理。",
        "memory_scope": "额外allocated峰值，不包含同时常驻的模型参数，不是独占总显存。",
    })
    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
