"""五种推理配置在同卡、同VAL首中末输入上交替计时；不加载GT。"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, compact_valid_views, digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move, seed_all
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def summarize(values):
    ordered = sorted(values)
    middle = statistics.median(values)
    return {"median_ms": middle, "mean_ms": statistics.mean(values),
            "p90_ms": ordered[max(0, math.ceil(len(ordered) * .9) - 1)],
            "single_reference_hz": 1000 / middle, "calls": len(values)}


def applicable_methods(config, domain):
    return [name for name, settings in config['methods'].items()
            if settings.get('num_views', domain['num_views']) == domain['num_views']
            and (not settings.get('datasets') or domain['name'] in settings['datasets'])]


def prepare_model(models, name, settings, domain, inputs):
    """在普通张量上下文加载实际视图数实例，使冻结位置参数保有版本计数。"""
    actual = domain
    if settings.get('drop_missing_views') and not bool(inputs['view_mask'].all()):
        compact, reference = compact_valid_views(inputs, domain['reference_index'])
        actual = dict(domain, num_views=compact['rgb'].shape[1], reference_index=reference)
    with torch.inference_mode(False), torch.no_grad():
        return models.get(name, actual)


def isolated_resources(config, domain, cpu_inputs, device):
    """各方法独立驻留，报告输入+权重+激活显存，不把共驻模型算进去。"""
    reports = {}
    for name in applicable_methods(config, domain):
        gc.collect()
        torch.cuda.empty_cache()
        inputs = move(cpu_inputs, device)
        models = Models(config, device)
        prepare_model(models, name, config['methods'][name], domain, inputs)
        with torch.inference_mode():
            for _ in range(config['warmup']):
                prediction = models.forward(name, domain, inputs)
                converted = models.convert(prediction, inputs)
                del prediction, converted
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            prediction = models.forward(name, domain, inputs)
            converted = models.convert(prediction, inputs)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reports[name] = {
                'parameters': sum(p.numel() for model in models.loaded.values() for p in model.parameters()),
                'resident_allocated_gib': baseline / 2**30,
                'peak_allocated_gib': peak / 2**30,
                'additional_peak_gib': (peak - baseline) / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
                'scope': '单模型、batch1、输入+权重+前向及格点转换；allocator不包含CUDA驱动上下文。',
            }
            del prediction, converted
        models.loaded.clear()
        del models, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def measure_frame(config, models, domain, inputs, device):
    names = applicable_methods(config, domain)
    for name in names:
        prepare_model(models, name, config['methods'][name], domain, inputs)
    with torch.inference_mode():
        for _ in range(config["warmup"]):
            for name in names:
                output = models.forward(name, domain, inputs)
                converted = models.convert(output, inputs)
                del output, converted
        torch.cuda.synchronize(device)
        samples = {name: {"wall_network_ms": [], "cuda_network_ms": [],
                          "wall_with_resampling_ms": [], "resident_peak_allocated_bytes": []}
                   for name in names}
        for repeat in range(config["repetitions"]):
            # 每轮循环平移次序，让各配置均匀占据前/后测量位置。
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                stop_event = torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start_event.record()
                prediction = models.forward(name, domain, inputs)
                stop_event.record()
                torch.cuda.synchronize(device)
                network_done = time.perf_counter()
                converted = models.convert(prediction, inputs)
                torch.cuda.synchronize(device)
                conversion_done = time.perf_counter()
                values = samples[name]
                values["wall_network_ms"].append((network_done - begin) * 1000)
                values["cuda_network_ms"].append(start_event.elapsed_time(stop_event))
                values["wall_with_resampling_ms"].append((conversion_done - begin) * 1000)
                values["resident_peak_allocated_bytes"].append(torch.cuda.max_memory_allocated(device))
                del prediction, converted
    return {name: {"summary": {key: summarize(values) for key, values in record.items() if key.endswith("_ms")},
                   "samples": record, "memory_note": "所有已加载模型共驻，峰值不是该方法独占显存。"}
            for name, record in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    data_config = json.loads(Path(config["data_config"]).read_text())
    # 数据尺寸、TRAIN范围和真实视图数沿用既定协议；仅计时设置在本配置中。
    for key in ("datasets", "image_size", "matching_image_size", "training_config"):
        config[key] = data_config[key]
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已保存的比较：{output}")
    output.mkdir(parents=True)
    offline_only()
    torch.set_num_threads(config["cpu_threads"])
    torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda", config["gpu"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(config["seed"])
    paths = {config["own_weights"]} | {item["weights"] for item in config["methods"].values() if "weights" in item}
    sources = ["tools/benchmark_public_speed.py", "jar/pretraining/comparison.py",
               "jar/pretraining/model.py", "jar/pretraining/encoders.py",
               "jar/pretraining/inference.py", "jar/pretraining/fast_spatial.py",
               "jar/pretraining/fast_fusion.py"]
    sources += [str(path.relative_to(ROOT)) for path in (ROOT / "jar/pretraining").glob("*attention*.py")]
    audit = {"config": config, "checkpoint_sha256": {path: digest(path) for path in paths},
             "source_sha256": {path: digest(path) for path in sources},
             "val_manifest_sha256": {item["name"]: digest(item["manifests"]["val"]) for item in config["datasets"]},
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(device),
             "protocol": "仅VAL首中末3组；load_targets=False；不读取测试/深度标签。所有方法FP32、TF32关闭、位置缓存、同卡交替。",
             "scope": "batch1、真实N图输入、单参考深度；含输入适配；不含读盘/H2D/多参考/点云融合。",
             "budget_note": "主图256x448；自研原生匹配512x896。baseline使用原生匹配设置，不是同像素计算预算。",
             "memory_note": "缓存模型共驻；本次记录的allocator峰值禁止称为方法独占显存。"}
    audit["protocol_sha256"] = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    write_json(output / "audit.json", audit)
    write_json(output / "config.json", config)
    models = Models(config, device)
    report = {"completed": False, "protocol_sha256": audit["protocol_sha256"], "datasets": {}}
    for domain in config["datasets"]:
        dataset = build_dataset(dataset_config(domain, config), "val", load_targets=False)
        indices = list(dict.fromkeys((0, len(dataset) // 2, len(dataset) - 1)))
        frames = []
        for index in indices:
            cpu_inputs = collate_frames([dataset[index]])["inputs"]
            inputs = move(cpu_inputs, device)
            frame = {"index": index, "input_sha256": fingerprint(cpu_inputs),
                     "valid_views": int(inputs["view_mask"][0].sum()),
                     "rgb_shape": list(inputs["rgb"].shape),
                     "matching_rgb_shape": list(inputs["matching_rgb"].shape),
                     "methods": measure_frame(config, models, domain, inputs, device)}
            frames.append(frame)
            write_json(output / domain["name"] / f"frame_{index}.json", frame)
            print(json.dumps({"dataset": domain["name"], "index": index,
                              "wall_network_ms": {name: row["summary"]["wall_network_ms"]
                                                  for name, row in frame["methods"].items()}}, ensure_ascii=False), flush=True)
            del inputs, cpu_inputs
        summary = {}
        for method in applicable_methods(config, domain):
            summary[method] = {}
            for field in ("wall_network_ms", "cuda_network_ms", "wall_with_resampling_ms"):
                values = [value for frame in frames for value in frame["methods"][method]["samples"][field]]
                summary[method][field] = summarize(values)
        report["datasets"][domain["name"]] = {"num_views": domain["num_views"],
                                               "reference_index": domain["reference_index"],
                                               "frame_indices": indices, "summary": summary,
                                               "distribution_note": "3个固定VAL样本的重复调用，不是完整测试集延迟分布。"}
        if config.get('isolated_resources', False):
            models.loaded.clear()
            gc.collect()
            torch.cuda.empty_cache()
            cpu_inputs = collate_frames([dataset[indices[1]]])['inputs']
            report['datasets'][domain['name']]['isolated_resources'] = isolated_resources(
                config, domain, cpu_inputs, device)
            report['datasets'][domain['name']]['resource_valid_views'] = int(cpu_inputs['view_mask'][0].sum())
            del cpu_inputs
        write_json(output / "latency.json", report)
    report["completed"] = True
    write_json(output / "latency.json", report)
    print(f"完成：{output}", flush=True)


if __name__ == "__main__":
    main()
