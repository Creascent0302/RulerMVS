"""真实 DEV 四目图像上的四模型轮转计时，不加载标签或 TEST。"""

import argparse
import copy
import json
from pathlib import Path
import platform
import shlex
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import read_json
from jar.models.metric_ruler_model import MetricRulerReconstructionModel
from jar.models.position_cache import install_position_cache
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move, seed_all
from tools.benchmark_raw_feature_reader import load_frames
from tools.carla_compare import file_sha256, make_model
from tools.verify_position_cache import measure_alternating


def position_caches(model):
    """先固定模块列表，原地安装只缓存位置插值的同一种控制器。"""
    result = []
    for name, module in tuple(model.named_modules()):
        required = ("pos_embed", "interpolate_pos_encoding", "patch_size",
                    "interpolate_offset", "interpolate_antialias")
        if all(hasattr(module, key) for key in required):
            result.append((name, install_position_cache(module)))
    if not result:
        raise ValueError("模型中没有支持本次统一位置缓存的视觉模块")
    return result


def enable(caches, value):
    for _, cache in caches:
        cache.clear()
        cache.enabled = value


def depth_from_output(output, kind, reference):
    # baseline 返回单参考 [B,1,H,W]；自研保留原相机槽位 [B,N,H,W]。
    return output["depth"][:, 0] if kind == "mvsanywhere" else output["dense_depth"][:, reference]


@torch.inference_mode()
def verify_cache(model, caches, frames, kind, reference):
    records = []
    previous = None
    changed = False
    for frame in frames:
        inputs = frame["inputs"]
        enable(caches, False)
        expected = depth_from_output(model(inputs), kind, reference).clone()
        enable(caches, True)
        cold = depth_from_output(model(inputs), kind, reference).clone()
        hot = depth_from_output(model(inputs), kind, reference).clone()
        if expected.shape != (1, 256, 448) or not bool(torch.isfinite(expected).all()):
            raise ValueError("各方法必须返回有限的上相机256×448米制深度")
        torch.testing.assert_close(cold, expected, rtol=0, atol=0)
        torch.testing.assert_close(hot, expected, rtol=0, atol=0)
        if previous is not None:
            changed |= not torch.equal(expected, previous)
        previous = expected
        records.append({"sample_id": frame["sample_id"],
                        "cold_max_difference_m": float((cold - expected).abs().max()),
                        "hot_max_difference_m": float((hot - expected).abs().max())})
    if not changed:
        raise RuntimeError("不同RGB帧深度完全相同，不能排除误缓存图像内容")
    used = []
    for name, cache in caches:
        if cache.value is not None and cache.value.is_inference():
            raise RuntimeError("位置缓存应持有普通张量，不能持有inference tensor")
        used.append({"module": name, "used": cache.value is not None,
                     "position_frozen": not cache.module.pos_embed.requires_grad,
                     "cached_shape": list(cache.value.shape) if cache.value is not None else None})
    return {"frames": records, "different_RGB_changes_depth": changed, "position_caches": used}


def summarize(samples):
    ordered = sorted(samples)
    median = statistics.median(samples)
    return {"median_ms": median, "mean_ms": statistics.mean(samples),
            "p90_ms": ordered[min(len(ordered) - 1, int(.9 * len(ordered)))],
            "reciprocal_median_Hz": 1000 / median, "sample_count": len(samples),
            "samples_ms": samples}


@torch.inference_mode()
def allocation_probe(models, inputs, device):
    """仅报告同驻留条件下单次调用的增量，不冒充独立模型总显存。"""
    rows = {}
    for name, model in models.items():
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = model(inputs)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        rows[name] = {"all_models_resident_MiB": resident / 2**20,
                      "call_peak_allocated_MiB": peak / 2**20,
                      "call_extra_allocated_MiB": max(0, peak - resident) / 2**20}
        del result
    return rows


def run(plan, device):
    if plan["split"] != "dev_complete":
        raise ValueError("本工具只允许DEV，不读取TEST或监督")
    output = Path(plan["output"])
    if (output / "summary.json").exists():
        raise FileExistsError("保留已有计时，请使用新的output")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), ROOT / "tools/verify_position_cache.py",
                    ROOT / "tools/benchmark_raw_feature_reader.py",
                    ROOT / "jar/models/position_cache.py", ROOT / "jar/models/shared_initializer.py",
                    ROOT / "jar/model.py", ROOT / "jar/models/dense_decoder.py",
                    ROOT / "jar/models/spatial_surface_recovery.py", ROOT / "jar/models/raw_feature_reader.py",
                    ROOT / "jar/models/metric_ruler_model.py", ROOT / "jar/models/metric_ruler_fine.py",
                    ROOT / "jar/models/metric_ruler_network.py", ROOT / "jar/models/metric_ruler_geometry_evidence.py"]
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths}
    protocol = read_json(plan["protocol"])
    reference = protocol["evaluation"]["reference_index"]
    frames = load_frames(protocol, plan)
    for frame in frames:
        frame["inputs"] = move(frame["inputs"], device)
    if len(frames) < 2 or torch.equal(frames[0]["inputs"]["rgb"], frames[1]["inputs"]["rgb"]):
        raise ValueError("缓存检查至少需要两张不同RGB帧")

    models, metadata = {}, {}
    base = plan["original"]
    original, cfg = make_model({"method": "mf_et_jar", "model_config": base["config"]},
                               base["checkpoint"], device)
    if original._checkpoint_load_report.get("missing_keys") or original._checkpoint_load_report.get("unexpected_keys"):
        raise ValueError("原模型必须严格恢复，不接受部分权重")
    models["original"] = original.eval()
    metadata["original"] = {**base, "kind": "self", "config": cfg}
    for name, checkpoint in plan["ruler_checkpoints"].items():
        model = MetricRulerReconstructionModel.from_checkpoint(checkpoint).to(device).eval()
        models[name] = model
        metadata[name] = {"checkpoint": checkpoint, "kind": "self", "config": model.cfg,
                          "ruler_settings": model.settings}
    initializer = copy.deepcopy(plan["initializer"])
    if any(initializer.get(key) for key in ("readout_weights", "fusion_weights", "decoder_weights")):
        raise ValueError("baseline只用原CARLA权重，不加载额外适配模块")
    initializer["cache_position_encoding"] = True
    models["mvsanywhere_firstpass"] = SharedPretrainedInitializer(initializer).to(device).eval()
    metadata["mvsanywhere_firstpass"] = {"kind": "mvsanywhere", "checkpoint": initializer["weights"],
                                        "initializer": initializer}
    caches = {name: position_caches(model) for name, model in models.items()}
    parity = {}
    for name, model in models.items():
        row = metadata[name]
        if row["kind"] == "self":
            cfg = row["config"]
            if cfg["model"]["execution"]["reference_indices"] != [reference]:
                raise ValueError("自研模型必须只输出同一个上相机参考")
            row["scene_image_size"] = cfg["data"]["image_size"]
            row["matching_image_size_used"] = cfg["data"].get("matching_image_size", cfg["data"]["image_size"])
        else:
            row["scene_image_size"] = [256, 448]
            row["matching_image_size_used"] = [256, 448]
        if row["scene_image_size"] != [256, 448]:
            raise ValueError("场景图像协议应固定256×448")
        row["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        row["checkpoint_sha256"] = file_sha256(row["checkpoint"])
        parity[name] = verify_cache(model, caches[name], frames, row["kind"], reference)
        print(json.dumps({"cache_parity": name, "max_difference_m": 0,
                          "different_RGB_changes_depth": True}, ensure_ascii=False), flush=True)

    times = {name: [] for name in models}
    per_frame = []
    torch.cuda.reset_peak_memory_stats(device)
    for frame in frames:
        measured = measure_alternating(models, frame["inputs"], device, plan["warmup"], plan["repeats"])
        per_frame.append({"sample_id": frame["sample_id"], "timing": measured})
        for name, row in measured.items():
            times[name].extend(row["samples_ms"])
        print(json.dumps({"timing_frame": frame["sample_id"],
                          "median_ms": {name: row["median_ms"] for name, row in measured.items()}},
                         ensure_ascii=False), flush=True)
    common_peak = torch.cuda.max_memory_allocated(device) / 2**20
    allocation = allocation_probe(models, frames[-1]["inputs"], device)
    for path in source_paths:
        if file_sha256(path) != source_hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError("计时期间源码发生变化，请独立重跑以保持记录可解释")
    report = {
        "plan": plan, "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "device": {"index": device.index, "name": torch.cuda.get_device_name(device),
                   "capability": list(torch.cuda.get_device_capability(device)),
                   "total_memory_MiB": torch.cuda.get_device_properties(device).total_memory / 2**20},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "precision": "FP32; matmul/cudnn TF32 disabled"},
        "frames": [{"sample_id": row["sample_id"], "frame_index": row["frame_index"],
                    "rgb_shape": list(row["inputs"]["rgb"].shape),
                    "matching_rgb_shape_available": list(row["inputs"]["matching_rgb"].shape)} for row in frames],
        "models": metadata, "position_cache_parity": parity,
        "timing": {name: summarize(values) for name, values in times.items()}, "per_frame": per_frame,
        "common_process_peak_allocated_MiB": common_peak, "call_allocation_probes": allocation,
        "source_sha256": source_hashes, "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][plan["split"]]),
        "notes": [
            "同GPU四模型同时驻留；每帧warmup后轮转先后顺序，输入已驻GPU。",
            "四目256×448只输出上参考图；自研匹配图512×896，baseline原生匹配图256×448，内部计算预算不同。",
            "baseline为完整MVSAnywhere首遍深度；自研包含候选、九候选选择、原标定细化和全图头。",
            "全部模型同开FrozenPositionCache，只缓存位置插值；cold/hot与off逐元素相等且不同RGB产生不同深度。",
            "共同峰值不是任何单模型显存；call_extra只表示多模型驻留时一次调用的额外分配。",
            "含网络和输入适配，不含读盘/H2D/外部点云融合；频率为网络中位延迟的倒数，不等于车端端到端频率。",
            "无GT/TEST访问，仅计时；不宣称预训练数据量或训练预算公平，不由DEV3速度推断泛化精度。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "experiment.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("median_ms", "reciprocal_median_Hz", "sample_count")}
                      for name, row in report["timing"].items()}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本工具需要CUDA，CPU时间不能替代GPU比较")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    seed_all(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(read_json(args.config), device)


if __name__ == "__main__":
    main()
