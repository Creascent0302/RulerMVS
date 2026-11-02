"""同一真实DEV四目帧上的完整模型与原始特征读取器GPU计时。

JSON配置：protocol、split、frames、warmup、repeats、precision、threads、
seed、output、cache_root，以及variants中的{名称:{config,checkpoint}}。
baseline须存在；其余配置使用已导出的完整模型检查点。
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, read_json
from jar.data import collate_frames
from jar.runtime import move, precision_context, seed_all
from tools.benchmark_surface_recovery import summarize
from tools.carla_compare import file_sha256, make_model


@torch.inference_mode()
def measure(call, settings, device):
    """输入已驻GPU；同步墙钟为主，CUDA事件作为另一个计时口径。"""
    records = []
    with precision_context(device, settings["precision"]):
        for _ in range(settings["warmup"]):
            result = call()
            del result
        torch.cuda.synchronize(device)
        for _ in range(settings["repeats"]):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            resident = torch.cuda.memory_allocated(device)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            result = call()
            end.record()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            peak = torch.cuda.max_memory_allocated(device)
            records.append({
                "wall_ms": elapsed,
                "cuda_event_ms": begin.elapsed_time(end),
                "resident_allocated_MiB": resident / 2**20,
                "peak_allocated_MiB": peak / 2**20,
                "peak_increment_MiB": (peak - resident) / 2**20,
            })
            del result
    return records


def summarize_records(records):
    return {
        "wall_ms": summarize([row["wall_ms"] for row in records]),
        "cuda_event_ms": summarize([row["cuda_event_ms"] for row in records]),
        "peak_allocated_MiB": max(row["peak_allocated_MiB"] for row in records),
        "peak_increment_MiB": max(row["peak_increment_MiB"] for row in records),
    }


def load_frames(protocol, plan):
    dataset = comparison_dataset(protocol, plan["split"], load_targets=False)
    indices = plan["frames"]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("frames须为非空、不重复的帧索引")
    frames = []
    for index in indices:
        if not 0 <= index < len(dataset):
            raise ValueError("frame索引超出数据清单范围")
        frame = dataset[index]
        if frame["targets"]:
            raise AssertionError("完整模型计时不得加载监督标签")
        inputs = collate_frames([frame])["inputs"]
        if inputs["rgb"].shape[:2] != (1, 4) or not bool(inputs["view_mask"].all()):
            raise ValueError("本次基准只使用batch1、同一时刻完整四目")
        source = dataset.frames[index]
        frames.append({
            "frame_index": index,
            "sample_id": f"{source['sequence']}/{source['frame_id']}",
            "inputs": inputs,
        })
    return frames


@torch.inference_mode()
def benchmark_variant(name, variant, frames, plan, protocol, device):
    model, cfg = make_model(
        {"method": "mf_et_jar", "model_config": variant["config"]},
        variant["checkpoint"], device,
    )
    reference = protocol["evaluation"]["reference_index"]
    if cfg["model"].get("execution", {}).get("reference_indices") != [reference]:
        raise ValueError("所有变体须保持四目输入、只输出协议指定的上相机深度")
    if cfg["data"]["image_size"] != protocol["data"]["image_size"]:
        raise ValueError("模型与共同数据协议的图像尺寸不一致")
    load_report = model._checkpoint_load_report
    if load_report.get("missing_keys") or load_report.get("unexpected_keys"):
        raise ValueError("性能测试须使用完整且严格恢复的模型权重")
    reader = getattr(model.dense_decoder, "raw_feature_reader", None)
    result = {
        "config": variant["config"],
        "config_sha256": file_sha256(variant["config"]),
        "checkpoint": variant["checkpoint"],
        "checkpoint_sha256": file_sha256(variant["checkpoint"]),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "reader_parameters": sum(p.numel() for p in reader.parameters()) if reader else 0,
        "network_records": [],
        "prediction_checks": [],
    }
    for frame in frames:
        inputs = move(frame["inputs"], device)
        records = measure(lambda: model(inputs), plan, device)
        result["network_records"].extend(
            {"sample_id": frame["sample_id"], **row} for row in records
        )
        with precision_context(device, plan["precision"]):
            prediction = model(inputs)
        depth = prediction["dense_depth"][0, reference]
        valid = prediction["dense_valid"][0, reference]
        if not bool(torch.isfinite(depth).all()):
            raise FloatingPointError("完整模型输出深度包含非有限值")
        result["prediction_checks"].append({
            "sample_id": frame["sample_id"],
            "rgb_shape": list(inputs["rgb"].shape),
            "matching_rgb_shape": list(inputs["matching_rgb"].shape),
            "output_depth_shape": list(depth.shape),
            "valid_pixels": int(valid.sum()),
        })
        del depth, valid, prediction, inputs
    result["network_summary"] = summarize_records(result["network_records"])
    # 完整模型先独立计时。之后只保留reader，隔离其驻留内存与额外调用开销。
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result["reader_records"] = []
    if reader is not None:
        cache_root = Path(plan["cache_root"])
        paths = read_json(cache_root / "index.json")["sets"][plan["split"]]
        for frame in frames:
            path = cache_root / paths[frame["frame_index"]]
            entry = torch.load(path, map_location="cpu", weights_only=False)
            if entry["sample_id"] != frame["sample_id"]:
                raise ValueError("单模块缓存与完整模型计时帧不一致")
            fields = ("features", "valid", "raw_maps", "base_logits")
            cached = move({key: entry["inputs"][key] for key in fields}, device)
            records = measure(
                lambda: reader(*(cached[key] for key in fields)), plan, device
            )
            result["reader_records"].extend(
                {"sample_id": frame["sample_id"], "cache": str(path), **row}
                for row in records
            )
            del cached, entry
        result["reader_summary"] = summarize_records(result["reader_records"])
    else:
        result["reader_summary"] = None
    del reader
    gc.collect()
    torch.cuda.empty_cache()
    print(name, json.dumps(result["network_summary"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = read_json(args.config)
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError("已有性能记录，请使用新output: " + str(output))
    if plan["warmup"] < 1 or plan["repeats"] < 1:
        raise ValueError("warmup与repeats须为正整数")
    if "baseline" not in plan["variants"]:
        raise ValueError("variants须包含baseline，其他变体使用相同数据和计时设置")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本工具只测GPU，请指定可用CUDA设备")
    torch.cuda.set_device(device)
    torch.set_num_threads(plan.get("threads", 2))
    seed_all(plan.get("seed", 43))
    protocol = read_json(plan["protocol"])
    frames = load_frames(protocol, plan)
    report = {
        "plan": plan,
        "command": [sys.executable, *sys.argv],
        "script_sha256": file_sha256(__file__),
        "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][plan["split"]]),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": plan["precision"],
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "batch_size": 1,
        "input_cameras": 4,
        "output_reference": protocol["evaluation"]["reference_index"],
        "definitions": {
            "network": "RGB/K/T/掩码预驻GPU；常规model(inputs)，无hook、无缓存替代前向；包含内部选面、细化、稠密输出和几何验证。每次预热后前后CUDA同步，不含图像解码、文件I/O、搬运、模型加载或指标计算。",
            "reader": "相同帧已缓存推理输入预驻GPU，仅reader(features,valid,raw_maps,base_logits)，不读取targets；不含ViT编码、候选生成、原评分头或缓存读取。此数值不能直接代替完整模型增量。",
            "memory": "allocated峰值包含当前模型与当前输入；increment为超过调用前驻留分配量的峰值，均不包含其他进程、CUDA上下文或PyTorch预留缓存。",
            "comparison": "同一GPU顺序执行、同一真实帧与精度，完整四目仅上相机输出；少量帧微基准不能代表所有场景或车端部署速度。",
        },
        "variants": {},
    }
    names = ["baseline", *[name for name in plan["variants"] if name != "baseline"]]
    for name in names:
        report["variants"][name] = benchmark_variant(
            name, plan["variants"][name], frames, plan, protocol, device
        )
        baseline = report["variants"]["baseline"]["network_summary"]["wall_ms"]["median"]
        current = report["variants"][name]["network_summary"]["wall_ms"]["median"]
        report["variants"][name]["median_change_from_baseline_ms"] = current - baseline
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
