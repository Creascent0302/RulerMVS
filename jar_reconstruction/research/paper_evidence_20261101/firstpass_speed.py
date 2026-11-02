"""V6与MVSAnywhere首遍/细化同卡交替计时；真实单参考、串行四参考与共享四参考。"""

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move, seed_all
from research.rulermvs_v6_20261101.evaluation import Models
from research.rulermvs_v6_20261101.speed_final import distribution, initialize, load
from tools.smoke_outdoor_datasets import fingerprint


def pack(native, reference):
    return {"depth": native["depth"] * native["unit_scale"], "valid": native["valid"], "reference_indices": [reference]}


def run(settings):
    initialize(settings)
    seed_all(settings["seed"])
    device = torch.device("cuda", settings["gpu"])
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(settings["data_config"]).read_text())
    domain = config["datasets"][0]
    if domain["num_views"] != 4:
        raise ValueError("本正式补充使用真实完整四目")
    dataset = build_dataset(dataset_config(domain, config), "val", load_targets=False)
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    models = Models(config, device, fixed_random_stream=False)
    for method in config["methods"]:
        models.get(method, domain)
    own = {"v6_best": models.get("v6_best", domain)}
    if settings["include_v6_fast"]:
        own["v6_fast"] = load(settings, "v6_fast")
    durations, with_resampling, fingerprints, equality, warmups = {}, {}, {}, [], {}
    rng = random.Random(settings["seed"])
    with torch.inference_mode():
        for index in indices:
            cpu = collate_frames([dataset[index]])["inputs"]
            if not bool(cpu["view_mask"].all()) or cpu["rgb"].shape[1] != 4:
                raise ValueError("预登记帧缺视图，拒绝悄悄换样本")
            fingerprints[str(index)] = fingerprint(cpu)
            inputs = move(cpu, device)

            def single(method, reference):
                if method == "v6_fast":
                    return pack(own[method](equal_pixel_inputs(inputs), domain, reference), reference)
                return models.forward(method, dict(domain, reference_index=reference), inputs)

            def serial(method):
                return {reference: single(method, reference) for reference in range(4)}

            def shared(method):
                native = own[method].forward_references(equal_pixel_inputs(inputs), domain, range(4))
                return {reference: pack(value, reference) for reference, value in native.items()}

            calls = {}
            for method in list(config["methods"]) + (["v6_fast"] if settings["include_v6_fast"] else []):
                calls[method + "_single"] = lambda method=method: single(method, domain["reference_index"])
                calls[method + "_serial_four"] = lambda method=method: serial(method)
            for method in own:
                calls[method + "_shared_four"] = lambda method=method: shared(method)

            def convert(name, result):
                if name.endswith("_single"):
                    return models.convert(result, inputs)
                return {reference: models.convert(value, inputs) for reference, value in result.items()}

            for name, call in calls.items():
                start = time.perf_counter()
                for _ in range(settings["warmup"]):
                    result = call()
                    converted = convert(name, result)
                    del result, converted
                torch.cuda.synchronize(device)
                warmups[f"{index}/{name}"] = time.perf_counter() - start
            for method in own:
                a, b = serial(method), shared(method)
                same = all(torch.equal(a[reference][field], b[reference][field]) for reference in range(4)
                           for field in ("depth", "valid"))
                if not same:
                    raise RuntimeError("真实四参考共享入口与同路径串行输出不一致")
                equality.append({"index": index, "method": method, "depth_valid_exact": True})
                del a, b
            for _ in range(settings["repeats"]):
                order = list(calls)
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    result = calls[name]()
                    torch.cuda.synchronize(device)
                    network_done = time.perf_counter()
                    converted = convert(name, result)
                    torch.cuda.synchronize(device)
                    conversion_done = time.perf_counter()
                    durations.setdefault(name, []).append((network_done - start) * 1000)
                    with_resampling.setdefault(name, []).append((conversion_done - start) * 1000)
                    del result, converted
            print(json.dumps({"frame_done": index, "methods": len(calls)}), flush=True)
            del inputs, cpu, calls
    report = {"completed": True, "settings": settings, "config": config,
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "precision": "FP32/TF32off",
              "split": "val", "load_targets": False, "actual_views": 4, "indices": indices,
              "input_sha256": fingerprints, "methods": {name: distribution(value) for name, value in durations.items()},
              "methods_with_resampling": {name: distribution(value) for name, value in with_resampling.items()},
              "samples_with_resampling_ms": with_resampling,
              "samples_ms": durations, "shared_vs_serial": equality, "warmup_including_compile_seconds": warmups,
              "checkpoint_sha256": {name: digest(value["weights"]) for name, value in config["methods"].items()},
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): digest(__file__)},
              "scope": "GPU常驻完整四目；单参考、实际串行四参考、自研共享四参考分别记录；methods含输入适配+网络，methods_with_resampling另含共同256x448输出重采样；不包括IO/H2D/点云融合。",
              "baseline": "官方首遍num_refinement_steps=0，一次范围细化=1；共同TRAIN范围与位置缓存，不增加未验证的跨参考共享。",
              "fast_limit": "V6 fast为同权重显式部署路径；本报告只有三帧推理检查，不替代fast完整TEST精度。",
              "memory_limit": "多个模型共驻，不报告方法独占显存；不得把单参考耗时乘四代替四参考实测。"}
    write_json(output / "results.json", report)
    models.loaded.clear()
    own.clear()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
