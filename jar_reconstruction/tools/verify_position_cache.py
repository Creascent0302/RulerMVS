"""GPU 验证冻结位置缓存的逐元素一致性，并公平交替计时初值器。"""

import argparse
import copy
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
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move, seed_all
from tools.carla_compare import file_sha256


def caches(model):
    native = model.native_model
    return [
        native.encoder.model._frozen_position_cache,
        native.cost_volume_net.model._frozen_position_cache,
    ]


def enable(model, value):
    for cache in caches(model):
        cache.clear()
        cache.enabled = value


@torch.inference_mode()
def measure_alternating(models, inputs, device, warmup, repeats):
    names = list(models)
    rows = {name: [] for name in names}
    for iteration in range(warmup + repeats):
        # 每轮轮转先后次序，避免一个模型总在最早或最后运行。
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            models[name](inputs)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            if iteration >= warmup:
                rows[name].append(elapsed)
    return {
        name: {
            "median_ms": statistics.median(times),
            "mean_ms": statistics.mean(times),
            "p90_ms": sorted(times)[min(len(times) - 1, int(len(times) * 0.9))],
            "samples_ms": times,
        }
        for name, times in rows.items()
    }


def run(plan, device):
    output = Path(plan["output"])
    if (output / "summary.json").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(plan["protocol"])
    dataset = comparison_dataset(protocol, "dev_complete", load_targets=False)
    inputs = [
        move(collate_frames([dataset[index]])["inputs"], device)
        for index in range(plan["parity_frames"])
    ]
    models = {}
    weight_hashes = {}
    parity = {}
    for variant in plan["variants"]:
        settings = copy.deepcopy(plan["initializer"])
        settings.update(variant["initializer"])
        settings["cache_position_encoding"] = True
        model = SharedPretrainedInitializer(settings).to(device).eval()
        if variant.get("fusion_weights"):
            saved = torch.load(variant["fusion_weights"], map_location=device, weights_only=True)
            model.native_model.cost_volume_net.load_state_dict(saved["model"], strict=True)
        models[variant["name"]] = model
        differences = []
        previous = None
        changed_output = False
        with torch.inference_mode():
            for frame in inputs:
                enable(model, False)
                expected = model(frame)["depth"]
                enable(model, True)
                cold = model(frame)["depth"]
                hot = model(frame)["depth"]
                torch.testing.assert_close(cold, expected, rtol=0, atol=0)
                torch.testing.assert_close(hot, expected, rtol=0, atol=0)
                if previous is not None:
                    changed_output = changed_output or not torch.equal(expected, previous)
                previous = expected
                differences.append(float((expected - hot).abs().amax()))
                if any(cache.value is None or cache.value.is_inference() for cache in caches(model)):
                    raise RuntimeError("缓存必须保存普通位置张量，不能是 inference tensor")
        if not changed_output:
            raise RuntimeError("两帧预测完全相同，需要检查是否误缓存图像相关结果")
        parity[variant["name"]] = {
            "max_depth_difference_m": max(differences),
            "frame_differences_m": differences,
            "different_images_produce_different_depth": changed_output,
            "state_dict_has_cache_fields": any("cache" in key for key in model.state_dict()),
            "cache_shapes": [list(cache.value.shape) for cache in caches(model)],
        }
        if parity[variant["name"]]["state_dict_has_cache_fields"]:
            raise RuntimeError("位置缓存不应进入 checkpoint state_dict")
        paths = [value for key, value in settings.items() if key.endswith("weights")]
        paths.append(variant.get("fusion_weights"))
        for path in paths:
            if path and path not in weight_hashes:
                weight_hashes[path] = file_sha256(path)
        print("parity", variant["name"], parity[variant["name"]], flush=True)

    results = {}
    for enabled in (False, True):
        for model in models.values():
            enable(model, enabled)
        name = "cached" if enabled else "uncached"
        results[name] = measure_alternating(
            models, inputs[0], device, plan["warmup"], plan["repeats"]
        )
        print(name, {key: value["median_ms"] for key, value in results[name].items()}, flush=True)
    report = {
        "plan": plan,
        "samples": [
            f"{dataset.frames[i]['sequence']}/{dataset.frames[i]['frame_id']}"
            for i in range(plan["parity_frames"])
        ],
        "device": torch.cuda.get_device_name(device),
        "precision": "FP32; TF32 disabled",
        "parity": parity,
        "timing": results,
        "cache_saved_fraction": {
            name: 1 - results["cached"][name]["median_ms"] / results["uncached"][name]["median_ms"]
            for name in models
        },
        "weights_sha256": weight_hashes,
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (Path(__file__), ROOT / "jar/models/position_cache.py",
                         ROOT / "jar/models/shared_initializer.py")
        },
        "notes": [
            "DEV前两帧；不读取监督或TEST。冷缓存与热缓存均逐元素等于原前向。",
            "固定四目256×448、上参考、首遍初值，无后端和范围二次细化。",
            "同一GPU、相同precision、每轮轮转模型顺序；所有模型同开或同关缓存。",
            "完整wall计时含输入适配和同步，不含读取/H2D/后处理。",
            "多个模型同时驻留显存；不将该实验总显存当作单模型显存。",
            "位置缓存属于公平工程优化，baseline同样受益，不能算作自研架构创新。",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "experiment.py").write_text(Path(__file__).read_text(), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all(43)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    run(read_json(args.config), device)


if __name__ == "__main__":
    main()
