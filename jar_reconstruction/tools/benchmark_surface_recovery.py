"""无标签真实四目帧的生产模型 GPU 推理基准。参数只在 JSON 中配置。"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, read_json
from jar.data import collate_frames
from jar.runtime import move, seed_all
from tools.carla_compare import file_sha256, make_model


def synchronize(device):
    torch.cuda.synchronize(device)


def summarize(values):
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
        "std": statistics.pstdev(values),
    }


def raw_inputs(dataset, index):
    frame = dataset[index]
    if frame["targets"]:
        raise AssertionError("推理计时数据加载不得读取监督标签")
    return collate_frames([frame])["inputs"]


def copy_prediction(prediction, reference):
    """复制两个配置共同的常规上目输出，不包含训练诊断张量。"""
    depth = prediction["dense_depth"][0, reference].detach().cpu()
    valid = prediction["dense_valid"][0, reference].detach().cpu()
    return depth, valid


@torch.inference_mode()
def measured_call(model, device, stage, cpu_inputs, gpu_inputs, dataset, index, reference):
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if stage == "network_forward":
        prediction = model(gpu_inputs)
    else:
        source = cpu_inputs if stage == "cpu_tensor_pipeline" else raw_inputs(dataset, index)
        prediction = model(move(source, device))
        copy_prediction(prediction, reference)
    synchronize(device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    del prediction
    return {"index": index, "elapsed_ms": elapsed_ms, "peak_allocated_MiB": peak_mib}


def benchmark_variant(name, config_path, checkpoint, dataset, indices, settings, device):
    model, cfg = make_model(
        {"method": "mf_et_jar", "model_config": config_path}, checkpoint, device
    )
    load_report = model._checkpoint_load_report
    missing = load_report.get("missing_keys", [])
    if any(not key.startswith(("candidate_field.", "dense_decoder.surface_recovery.")) for key in missing):
        raise ValueError(f"{name} 的旧模型参数没有完整恢复：{missing}")
    output = {
        "config": config_path,
        "config_sha256": file_sha256(config_path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_missing_keys": missing,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "surface_recovery_weight": cfg["model"]["dense_decoder"].get("surface_recovery", {}).get("weights", ""),
        "records": {},
        "summary": {},
        "prediction_fingerprint": {},
    }
    if output["surface_recovery_weight"]:
        output["surface_recovery_weight_sha256"] = file_sha256(output["surface_recovery_weight"])
    reference = settings["reference_index"]
    stages = ("network_forward", "cpu_tensor_pipeline", "raw_frame_pipeline")
    for stage in stages:
        rows = []
        for index in indices:
            cpu_inputs = raw_inputs(dataset, index)
            if cpu_inputs["rgb"].shape[1] != 4 or not cpu_inputs["view_mask"].all():
                raise ValueError(f"{index} 不是完整四目帧")
            gpu_inputs = move(cpu_inputs, device) if stage == "network_forward" else None
            for _ in range(settings["warmup"]):
                measured_call(model, device, stage, cpu_inputs, gpu_inputs, dataset, index, reference)
            for _ in range(settings["repeats"]):
                rows.append(
                    measured_call(model, device, stage, cpu_inputs, gpu_inputs, dataset, index, reference)
                )
            if stage == "network_forward":
                with torch.inference_mode():
                    prediction = model(gpu_inputs)
                    depth, valid = copy_prediction(prediction, reference)
                    output["prediction_fingerprint"][str(index)] = {
                        "depth_shape": list(depth.shape),
                        "valid_pixels": int(valid.sum()),
                        "mean_valid_depth_m": float(depth[valid].mean()) if valid.any() else None,
                    }
            del gpu_inputs, cpu_inputs
        output["records"][stage] = rows
        output["summary"][stage] = {
            "latency_ms": summarize([row["elapsed_ms"] for row in rows]),
            "peak_allocated_MiB": max(row["peak_allocated_MiB"] for row in rows),
        }
        print(name, stage, output["summary"][stage], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = read_json(args.config)
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError(f"基准结果已存在，避免覆盖：{output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本脚本只测 GPU，请指定可用 CUDA 设备")
    torch.cuda.set_device(device)
    torch.set_num_threads(plan["threads"])
    seed_all(plan["seed"])
    protocol = read_json(plan["protocol"])
    dataset = comparison_dataset(protocol, plan["split"], load_targets=False)
    indices = plan["frames"]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("frame 索引超出数据清单长度")
    settings = {
        "reference_index": protocol["evaluation"]["reference_index"],
        "warmup": plan["warmup"],
        "repeats": plan["repeats"],
    }
    report = {
        "plan": plan,
        "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][plan["split"]]),
        "script_sha256": file_sha256(__file__),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "precision": "fp32",
        "batch_size": 1,
        "input_shape": [1, 4, 3, *protocol["data"]["image_size"]],
        "matching_input_shape": [1, 4, 3, *protocol["data"]["matching_image_size"]],
        "timing_definitions": {
            "network_forward": "RGB 与标定已驻留 GPU；仅 model(inputs)；每次前后 CUDA 同步。",
            "cpu_tensor_pipeline": "已解码 CPU 输入张量到 GPU，model(inputs)，上目深度/有效掩码回 CPU；不含文件读取。",
            "raw_frame_pipeline": "从 CARLA 原始 RGB 文件读取、校准缩放与 collate、GPU 搬运、model(inputs)、上目深度/有效掩码回 CPU；不含模型加载和写盘。",
        },
        "data_contract": "comparison_dataset(load_targets=False)，只读同步四目 RGB/K/T 和掩码，不读 GT 深度/语义。",
        "variants": {},
    }
    for name, variant in plan["variants"].items():
        report["variants"][name] = benchmark_variant(
            name, variant["config"], variant["checkpoint"], dataset, indices, settings, device
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
