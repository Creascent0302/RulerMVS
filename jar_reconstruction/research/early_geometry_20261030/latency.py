"""同GPU交替计时：已解码GPU输入到单参考深度，不包含读盘/传输。"""

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
from research.early_geometry_20261030.pilot import load_model, path
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有计时：{output}")
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    source = json.loads(path(settings["data_config"]).read_text())
    source["datasets"] = source.get("validation_datasets", source["datasets"])
    for domain in source["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
    source.pop("validation_datasets", None)
    dataset = MixedDataset(source, "val")
    models = {}
    for name, filename in settings["methods"].items():
        # 参数需有版本计数，避免 inference tensor 使冻结位置缓存静默失效。
        with torch.inference_mode(False):
            models[name], _ = load_model(filename, device)
    results = {}
    order_rng = random.Random(20261030)
    for domain_name in settings["datasets"]:
        domain_index = next(i for i, domain in enumerate(dataset.domains) if domain["name"] == domain_name)
        domain = dataset.domains[domain_index]
        length = len(dataset.datasets[domain_index])
        indices = torch.linspace(0, length - 1, min(3, length)).long().tolist()
        durations = {name: [] for name in models}
        extra_memory = {name: [] for name in models}
        valid_views = []
        for index in indices:
            batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
            inputs = batch["inputs"]
            valid_views.append(int(inputs["view_mask"].sum()))
            for model in models.values():
                for _ in range(settings.get("warmup", 10)):
                    model(inputs, domain, reference=0)
                if model.foundation.model._frozen_position_cache.key is None:
                    raise RuntimeError("位置缓存没有实际命中，拒绝记录不公平耗时")
            torch.cuda.synchronize(device)
            for repeat in range(settings.get("repeats", 40)):
                names = list(models)
                order_rng.shuffle(names)
                for name in names:
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    prediction = models[name](inputs, domain, reference=0)
                    torch.cuda.synchronize(device)
                    durations[name].append((time.perf_counter() - start) * 1000)
                    extra_memory[name].append((torch.cuda.max_memory_allocated(device) - before) / 2**30)
                    del prediction
            del batch, inputs
        results[domain_name] = {
            "indices": indices, "valid_views": valid_views,
            "methods": {name: {"median_ms": float(np.median(values)),
                                "mean_ms": float(np.mean(values)), "p95_ms": float(np.percentile(values, 95)),
                                "frequency_hz": 1000 / float(np.median(values)),
                                "additional_allocated_gib": max(extra_memory[name]),
                                "measurements": len(values)} for name, values in durations.items()},
        }
    write_json(output, {
        "completed": True, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "precision": "FP32, TF32 disabled", "results": results,
        "weights_sha256": {name: digest(path(filename)) for name, filename in settings["methods"].items()},
        "scope": "单参考网络本体，GPU常驻相同输入，交替计时与预热；不含读盘/预处理/传输/点云融合。",
        "memory_scope": "额外allocated峰值，不含同时常驻的各模型权重；不是单模型独占总显存。",
    })
    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
