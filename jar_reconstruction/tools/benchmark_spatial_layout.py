"""真实权重、无标签VAL输入下检查局部卷积布局优化与GPU反向。"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.models.ruler_spatial import CandidateScaleSpace
from jar.pretraining.comparison import write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.fast_spatial import install_spatial_layout
from jar.pretraining.model import PretrainingModel
from jar.runtime import move
from tools.smoke_outdoor_datasets import offline_only
from types import SimpleNamespace


def gpu_gradient_check(device):
    torch.manual_seed(13)
    original = CandidateScaleSpace(8).to(device)
    modified = copy.deepcopy(original)
    install_spatial_layout(SimpleNamespace(decoder=SimpleNamespace(local=modified)))
    inputs = torch.randn(2, 8, 7, 5, 6, device=device).contiguous(memory_format=torch.channels_last_3d)
    first = inputs.detach().requires_grad_()
    second = inputs.detach().clone().requires_grad_()
    valid = torch.rand(2, 7, 5, 6, device=device) > .2
    a, b = original(first, valid), modified(second, valid)
    weight = torch.randn_like(a)
    (a * weight).mean().backward()
    (b * weight).mean().backward()
    comparisons = [(a, b), (first.grad, second.grad)]
    comparisons.extend((x.grad, y.grad) for x, y in zip(original.parameters(), modified.parameters()))
    for x, y in comparisons:
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
    return {"passed": True, "forward_max_abs": (a - b).abs().max().item(),
            "input_gradient_max_abs": (first.grad - second.grad).abs().max().item(),
            "parameter_gradient_max_abs": max((x - y).abs().max().item() for x, y in comparisons[2:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(config["gpu"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", config["gpu"])
    gradient = gpu_gradient_check(device)
    saved = torch.load(config["checkpoint"], weights_only=False, map_location="cpu", mmap=True)
    original = PretrainingModel(saved["config"], initialize=False).to(device).eval().requires_grad_(False)
    original.load_state_dict(saved["model"], strict=True)
    modified = copy.deepcopy(original)
    install_spatial_layout(modified)
    for model in (original, modified):
        install_position_cache(model.foundation.model)
    data_settings = json.loads(Path(config["data_config"]).read_text())
    report = {"config": config, "gpu_gradients": gradient, "datasets": {}}
    with torch.inference_mode():
        for name in config["datasets"]:
            domain = next(d for d in data_settings["datasets"] if d["name"] == name)
            dataset = build_dataset(dataset_config(domain, data_settings), "val", load_targets=False)
            checks = []
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            for index in indices:
                inputs = move(collate_frames([dataset[index]])["inputs"], device)
                first = original(inputs, domain, domain["reference_index"])
                second = modified(inputs, domain, domain["reference_index"])
                checks.append({"index": index,
                               "final_max_abs_m": (first["depth"] - second["depth"]).abs().max().item() * domain["depth_scale"],
                               "coarse_max_abs_m": (first["coarse"]["depth"] - second["coarse"]["depth"]).abs().max().item() * domain["depth_scale"],
                               "fine_max_abs_m": (first["fine"]["depth"] - second["fine"]["depth"]).abs().max().item() * domain["depth_scale"],
                               "coarse_changed_winners": (first["coarse"]["selected_index"] != second["coarse"]["selected_index"]).sum().item(),
                               "fine_changed_winners": (first["fine"]["selected_index"] != second["fine"]["selected_index"]).sum().item()})
            inputs = move(collate_frames([dataset[len(dataset) // 2]])["inputs"], device)
            models = {"original": original, "contiguous_depthwise": modified}
            for _ in range(config["warmup"]):
                for model in models.values():
                    model(inputs, domain, domain["reference_index"])
            torch.cuda.synchronize()
            timings = {name: [] for name in models}
            for repeat in range(config["repetitions"]):
                names = list(models) if repeat % 2 == 0 else list(reversed(models))
                for method in names:
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    models[method](inputs, domain, domain["reference_index"])
                    stop.record()
                    torch.cuda.synchronize()
                    timings[method].append(start.elapsed_time(stop))
            report["datasets"][name] = {"split": "val", "load_targets": False, "checks": checks,
                                        "median_ms": {key: statistics.median(value) for key, value in timings.items()},
                                        "samples_ms": timings}
            write_json(output / "results.json", report)
            print(json.dumps({"dataset": name, **report["datasets"][name]}, ensure_ascii=False), flush=True)
    print("GPU forward/backward passed", flush=True)


if __name__ == "__main__":
    main()
