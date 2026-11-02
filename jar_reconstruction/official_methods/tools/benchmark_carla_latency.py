"""同一 GPU、同一真实训练帧串行计时；不计算精度，不据此调整模型。"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, prediction_streams, read_json
from jar.runtime import move, seed_all
from official_methods.data import FourCameraDataset, collate
from tools.carla_compare import NativeTimer, make_model, resolve_device, synchronize


def benchmark(experiment, config, device):
    model, model_config = make_model(experiment, experiment.get("weights", ""), device)
    if "protocol" in config:
        protocol = read_json(config["protocol"])
        dataset = comparison_dataset(protocol, "train_complete", load_targets=False)
        source = dataset.frames[config["frame_index"]]
    else:
        dataset = FourCameraDataset(model_config, "train", load_targets=False)
        source = dataset.source.frames[config["frame_index"]]
    batch = collate([dataset[config["frame_index"]]])
    inputs = move(batch["inputs"], device)
    timer = NativeTimer(model, device)
    records = []
    with torch.inference_mode():
        for index in range(config["warmup"] + config["iterations"]):
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            prediction = model(inputs)
            reference_indices = prediction.get("reference_indices")
            synchronize(device)
            forward_ms = (time.perf_counter() - start) * 1000
            native_ms = timer.result()
            streams = prediction_streams(
                prediction, inputs, experiment["method"], model_config, 2
            )
            synchronize(device)
            pipeline_ms = (time.perf_counter() - start) * 1000
            peak = torch.cuda.max_memory_allocated(device)
            if index >= config["warmup"]:
                records.append(
                    dict(
                        native_ms=native_ms,
                        forward_ms=forward_ms,
                        pipeline_ms=pipeline_ms,
                        peak_allocated_bytes=peak,
                    )
                )
            del streams, prediction
    timer.close()
    summary = {
        key: {
            "mean": statistics.mean(row[key] for row in records),
            "median": statistics.median(row[key] for row in records),
        }
        for key in ("native_ms", "forward_ms", "pipeline_ms")
    }
    summary["peak_allocated_MiB"] = max(row["peak_allocated_bytes"] for row in records) / 2**20
    result = dict(
        experiment=experiment,
        sample_id=f"{source['sequence']}/{source['frame_id']}",
        image_size=list(inputs["rgb"].shape[-2:]),
        input_camera_count=int(inputs["view_mask"].sum()),
        reference_indices=reference_indices,
        reference_note="未显式返回reference_indices的方法记录为null；保留完整forward调用，不据输出形状伪造参考视图。",
        summary=summary,
        iterations=records,
    )
    del model, inputs, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("本工具用于同一 CUDA GPU 的计时与显存比较。")
    torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    seed_all(config["seed"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("请使用新报告路径，保留历史计时。")
    report = {
        "config": config,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "definition": "同一训练帧、FP32、batch1、四参考整场景；pipeline含共同槽2深度光栅转换。读取RGB和标定，不加载标签。显存为max_memory_allocated，包含模型与输入，不是nvidia-smi占用。",
        "results": {},
    }
    for name, path in config["experiments"].items():
        experiment = json.loads(Path(path).read_text())
        report["results"][name] = benchmark(experiment, config, device)
        print(name, json.dumps(report["results"][name]["summary"]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
