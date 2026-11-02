"""用既有七域 VAL 确认集核对推理优化；不训练、不访问 TEST。"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import add_statistics, digest, measure, metrics, write_json
from jar.pretraining.model import CalibratedDecoder
from jar.pretraining.inference import prepare_inference
from jar.runtime import move


@torch.no_grad()
def run(config):
    torch.set_num_threads(2)
    torch.cuda.set_device(config["gpu"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", config["gpu"])
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    state = {key.removeprefix("decoder."): value for key, value in saved["model"].items()
             if key.startswith("decoder.")}
    decoders = {}
    for name, options in config["variants"].items():
        decoder = CalibratedDecoder(dict(saved["config"]["model"])).to(device).eval()
        decoder.load_state_dict(state, strict=True)
        prepare_inference(SimpleNamespace(training=False, decoder=decoder), **options)
        decoders[name] = decoder
    index = json.loads(Path(config["cache"]).read_text())
    result = {"config": config, "checkpoint_sha256": digest(config["checkpoint"]),
              "cache_index_sha256": digest(config["cache"]), "test_accessed": False,
              "precision": "float32, TF32 disabled", "domains": {}}
    for domain in index["domains"]:
        paths = index["splits"]["confirmation"][domain["name"]]
        if config.get("samples_per_domain"):
            paths = paths[:config["samples_per_domain"]]
        totals = {name: {} for name in config["variants"]}
        deltas = {name: {"sum": 0., "max": 0., "pixels": 0, "fine_flips": 0,
                         "fine_pixels": 0, "validity_differences": 0,
                         "gt_valid_max": 0., "gt_valid_changed_over_01": 0,
                         "gt_valid_worse_over_01": 0, "gt_valid_better_over_01": 0,
                         "gt_valid_error_increase_max": 0., "gt_valid_pixels": 0}
                  for name in config["variants"] if name != "original"}
        for decoder in decoders.values():
            decoder.set_range(domain["near"] / domain["depth_scale"], domain["far"] / domain["depth_scale"])
        rows = []
        for path in paths:
            batch = move(torch.load(path, map_location="cpu", weights_only=False), device)
            predictions = {}
            row = {"sample": path}
            for name, decoder in decoders.items():
                output = decoder(batch["work"])
                output["depth"] = output["depth"] * domain["depth_scale"]
                predictions[name] = output
                stats = measure(output["depth"], output["valid"], batch, reference=0)
                add_statistics(totals[name], stats)
                row[name] = stats
            original = predictions["original"]
            truth = batch["targets"]["depth"][:, :1]
            gt_mask = batch["targets"]["depth_valid"][:, :1]
            gt_mask = gt_mask & batch["inputs"]["image_mask"][:, :1]
            gt_mask = gt_mask & batch["inputs"]["view_mask"][:, :1, None, None]
            gt_mask = gt_mask & torch.isfinite(truth) & (truth > 0)
            for name, summary in deltas.items():
                output = predictions[name]
                error = (output["depth"] - original["depth"]).abs()
                summary["sum"] += error.double().sum().item()
                summary["max"] = max(summary["max"], error.max().item())
                summary["pixels"] += error.numel()
                first = original["fine"]["scores"].argmax(1)
                second = output["fine"]["scores"].argmax(1)
                summary["fine_flips"] += (first != second).sum().item()
                summary["fine_pixels"] += first.numel()
                summary["validity_differences"] += (output["valid"] != original["valid"]).sum().item()
                eligible = error[gt_mask]
                change = ((output["depth"] - truth).abs() - (original["depth"] - truth).abs())[gt_mask]
                if eligible.numel():
                    summary["gt_valid_max"] = max(summary["gt_valid_max"], eligible.max().item())
                    summary["gt_valid_changed_over_01"] += (eligible > .1).sum().item()
                    summary["gt_valid_worse_over_01"] += (change > .1).sum().item()
                    summary["gt_valid_better_over_01"] += (change < -.1).sum().item()
                    summary["gt_valid_error_increase_max"] = max(summary["gt_valid_error_increase_max"], change.max().item())
                    summary["gt_valid_pixels"] += eligible.numel()
            rows.append(row)
        for summary in deltas.values():
            summary["mean"] = summary["sum"] / summary["pixels"]
        result["domains"][domain["name"]] = {
            "samples": len(paths), "metrics": {name: metrics(stats, domain["geometry_units"])
                                               for name, stats in totals.items()}, "differences": deltas}
        write_json(Path(config["output"]) / (domain["name"] + ".json"), rows)
        write_json(Path(config["output"]) / "validation.json", result)
        print(json.dumps({domain["name"]: result["domains"][domain["name"]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
