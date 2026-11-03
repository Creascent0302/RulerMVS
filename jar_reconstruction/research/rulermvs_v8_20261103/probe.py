"""固定 V6 权重的机制探针：仅 TRAIN/VAL，不用于宣称 V8 已训练效果。"""

import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import collate_mixed
from jar.runtime import move
from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset
from research.rulermvs_v6_20261101.inference import prepare_fast_inference
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v8_20261103.model import V8Model
from tools.smoke_outdoor_datasets import offline_only


VARIANTS = {
    "v6": None,
    "v8_disabled": {},
    "v8_units": {"information_mode": "log_depth", "information_scale": 10.,
                 "relative_validity": True, "physical_thresholds": True},
    "v8_units_count": {"information_mode": "log_depth", "information_scale": 10.,
                       "relative_validity": True, "physical_thresholds": True,
                       "source_count_strength": 1.},
}


def main():
    offline_only()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261103)
    output = ROOT / "runs/active/rulermvs_v8_20261103/diagnostics"
    saved = torch.load(ROOT / "runs/active/overnight_v6_v7_mvs_20261103/v6/best.pt",
                       map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(saved["config"])
    data = SharedDataset(config, "val")
    models = {}
    for name, options in VARIANTS.items():
        settings = copy.deepcopy(config)
        settings["v8"] = options or {}
        model = (V6Model if options is None else V8Model)(settings, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
        models[name] = model.cuda().eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for index, domain in enumerate(data.domains):
            batch = move(collate_mixed([data[(index, len(data.datasets[index]) // 2, -1)]]), "cuda")
            work = equal_pixel_inputs(batch["inputs"])
            truth = batch["targets"]["depth"][:, :1]
            valid = batch["targets"]["depth_valid"][:, :1] & work["image_mask"][:, :1]
            valid &= torch.isfinite(truth) & (truth > 0)
            for name, model in models.items():
                original = None
                for factor in (1., .5, 2.):
                    settings = dict(domain, depth_scale=domain["depth_scale"] * factor)
                    prediction = model(work, settings)
                    depth = prediction["depth"] * settings["depth_scale"]
                    coarse = prediction["coarse"]["depth"] * settings["depth_scale"]
                    if original is None:
                        original = (depth.clone(), coarse.clone())
                    row = dict(domain=domain["name"], method=name, factor=factor,
                               mae=float((depth - truth).abs()[valid].mean()),
                               unit_delta_mean=float((depth - original[0]).abs().mean()),
                               coarse_unit_delta_mean=float((coarse - original[1]).abs().mean()))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            if index == 1:
                timing_inputs, timing_domain = work, domain
    write_json(output / "unit_probe.json", dict(rows=rows, split="one_fixed_VAL_sample_per_domain",
                                                trained=False, variants=VARIANTS))
    # Same GPU, same input, alternating complete network forwards; compilation
    # and warmup excluded, IO/transfer excluded, FP32/TF32 off for all methods.
    for name, model in models.items():
        install_position_cache(model.foundation.model)
        prepare_fast_inference(model, dict(compile_source=True, compile_fusion=True,
                               compile_threads=1, cache_dir=str(output / "compile_cache" / name)))
    timings = {name: [] for name in models}
    before = {}
    with torch.inference_mode():
        for name, model in models.items():
            for _ in range(10):
                result = model(timing_inputs, timing_domain)
            before[name] = result["depth"].clone()
        for iteration in range(30):
            names = list(models)
            if iteration % 2:
                names.reverse()
            for name in names:
                torch.cuda.synchronize()
                start = time.perf_counter()
                models[name](timing_inputs, timing_domain)
                torch.cuda.synchronize()
                timings[name].append((time.perf_counter() - start) * 1000)
    result = {name: dict(median_ms=float(torch.tensor(values).median()), samples_ms=values)
              for name, values in timings.items()}
    result["disabled_max_depth_delta"] = float((before["v6"] - before["v8_disabled"]).abs().max())
    write_json(output / "timing_probe.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
