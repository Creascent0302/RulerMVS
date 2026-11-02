"""同卡同输入交替测速；双方均报告独立与共享匹配编码的多参考入口。"""

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
from research.ruler_flow_v2_20261031.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


@torch.no_grad()
def shared_original(model, inputs, domain, references):
    """原V2精确计算复用对照，不关闭或修改任何原模型层。"""
    batch, _, _, height, width = inputs["rgb"].shape
    scale = float(domain["depth_scale"])
    model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
    transformed = inputs["T_cam_from_rig"].clone().float()
    transformed[..., :3, 3] = transformed[..., :3, 3] / scale
    match_height, match_width = inputs["matching_rgb"].shape[-2:]
    matching_scale = inputs["K"].new_tensor((match_width / width, match_height / height))
    shared = {"pyramid": model.matching(inputs), "K": inputs["K"],
              "T_cam_from_rig": transformed, "view_mask": inputs["view_mask"],
              "image_mask": inputs["image_mask"], "image_size": (height, width),
              "matching_scale": matching_scale[None].expand(batch, -1)}
    output = {}
    for reference in references:
        valid = inputs["image_mask"][:, reference] & inputs["view_mask"][:, reference, None, None]
        image = torch.where(valid[:, None], inputs["rgb"][:, reference], 0.)
        work = {**shared, "raw_maps": model.foundation(image), "reference_index": reference}
        prediction = model.decoder(work)
        prediction.update(pyramid=None, unit_scale=scale)
        output[reference] = prediction
    return output


def compare_predictions(expected, actual):
    """对同权重串行/共享入口做逐张量核对，不比较不同训练权重。"""
    exact = True
    maximum = 0.
    fields = [(key,) for key in ("depth", "confidence", "support", "valid")]
    fields += [(stage, key) for stage in ("coarse", "fine")
               for key in ("depth", "scores", "probability", "sampling_depth",
                           "candidate_depth", "selected_index", "source_valid")]
    for reference in expected:
        for keys in fields:
            left, right = expected[reference], actual[reference]
            for key in keys:
                left, right = left[key], right[key]
            exact &= bool(torch.equal(left, right))
            error = float((left.float() - right.float()).abs().max())
            maximum = max(maximum, error)
            if not torch.allclose(left.float(), right.float(), rtol=0., atol=1e-6):
                raise RuntimeError(f"共享入口改变预测：reference={reference}/{keys}, max={error}")
    return {"all_checked_tensors_exact": exact, "max_absolute_difference": maximum,
            "references": list(expected), "checked_fields_per_reference": len(fields)}


def valid_samples(dataset, domain_index, device, count=3):
    """只按输入相机是否齐全选完整样本，不读取GT选择样例。"""
    length = len(dataset.datasets[domain_index])
    order = torch.linspace(0, length - 1, length).long().tolist()
    # 先从全序列均匀位置扫描；碰到缺RGB再向相邻位置补齐。
    preferred = torch.linspace(0, length - 1, min(count, length)).long().tolist()
    order = preferred + [index for index in order if index not in preferred]
    selected = []
    for index in order:
        batch = collate_mixed([dataset[(domain_index, index, -1)]])
        if not bool(batch["inputs"]["view_mask"].all()):
            continue
        selected.append((index, move(batch["inputs"], device)))
        if len(selected) == count:
            return selected
    raise RuntimeError("没有足够的完整RGB输入进行预登记计时")


@torch.inference_mode()
def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists():
        raise FileExistsError(output)
    seed_all(20261031)
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
    with torch.inference_mode(False):
        original, _ = load_model(settings["original_checkpoint"], device)
        updated, _ = load_model(settings["flow_checkpoint"], device)
    if original.architecture != "ruler_direct_v2" or updated.architecture != "ruler_flow_v2":
        raise ValueError("必须比较真实旧V2与独立Flow模型，不能关闭Flow伪装旧V2")
    root_files = [ROOT / item for item in (
        "jar/pretraining/model.py", "jar/pretraining/encoders.py", "jar/pretraining/data.py",
        "jar/pretraining/losses.py", "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py")]
    hashes_before = {str(file.relative_to(ROOT)): digest(file) for file in root_files}
    results = {}
    generator = random.Random(20261031)
    for name in settings["datasets"]:
        domain_index = next(i for i, domain in enumerate(dataset.domains) if domain["name"] == name)
        domain = dataset.domains[domain_index]
        samples = valid_samples(dataset, domain_index, device, settings.get("samples", 3))
        durations = {key: [] for key in ("original_single", "flow_single")}
        multiple = bool(settings.get("all_references", False))
        if multiple:
            durations.update({key: [] for key in (
                "original_serial", "flow_serial", "original_shared_matching", "flow_shared_matching")})
        comparisons = []
        for index, inputs in samples:
            references = list(range(inputs["rgb"].shape[1]))

            def serial(model):
                return {reference: model(inputs, domain, reference) for reference in references}

            methods = {
                "original_single": lambda: original(inputs, domain, 0),
                "flow_single": lambda: updated(inputs, domain, 0),
            }
            if multiple:
                methods.update({
                    "original_serial": lambda: serial(original),
                    "flow_serial": lambda: serial(updated),
                    "original_shared_matching": lambda: shared_original(original, inputs, domain, references),
                    "flow_shared_matching": lambda: updated.forward_references(inputs, domain, references),
                })
                comparisons.append({
                    "index": index,
                    "original": compare_predictions(serial(original), shared_original(original, inputs, domain, references)),
                    "flow": compare_predictions(serial(updated), updated.forward_references(inputs, domain, references)),
                })
            for forward in methods.values():
                for _ in range(settings.get("warmup", 10)):
                    forward()
            for model in (original, updated):
                if model.foundation.model._frozen_position_cache.key is None:
                    raise RuntimeError("位置编码缓存没有命中")
            torch.cuda.synchronize(device)
            for _ in range(settings.get("repeats", 40)):
                order = list(methods)
                generator.shuffle(order)
                for method in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    prediction = methods[method]()
                    torch.cuda.synchronize(device)
                    durations[method].append((time.perf_counter() - start) * 1000)
                    del prediction
        results[name] = {
            "indices": [index for index, _ in samples],
            "input_views": [int(inputs["view_mask"].sum()) for _, inputs in samples],
            "serial_shared_equality": comparisons,
            "methods": {method: {
                "median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)), "frequency_hz": 1000 / float(np.median(values)),
                "measurements": len(values),
            } for method, values in durations.items()},
        }
        print(json.dumps({"dataset": name, "medians": {key: value["median_ms"]
              for key, value in results[name]["methods"].items()}}), flush=True)
        del samples
    hashes_after = {str(file.relative_to(ROOT)): digest(file) for file in root_files}
    if hashes_before != hashes_after:
        raise RuntimeError("计时期间生产源代码发生变化")
    write_json(output, {
        "completed": True, "settings": settings, "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "precision": "FP32/TF32off",
        "results": results, "production_source_sha256": hashes_after,
        "checkpoint_sha256": {"original": digest(path(settings["original_checkpoint"])),
                               "flow": digest(path(settings["flow_checkpoint"]))},
        "scope": "GPU常驻RGB输入，完整网络；不含IO/传输/预处理/融合。serial与shared均实际输出全部参考。",
        "fairness": "原V2也提供相同匹配编码复用，不能将共享计算收益全部归于Flow。",
        "memory": "两个模型同时驻留，本次不报告容易混淆的独占峰值显存。",
    })
    del original, updated
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(json.loads(path(arguments.config).read_text()))
