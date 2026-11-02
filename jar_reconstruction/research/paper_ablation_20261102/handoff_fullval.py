"""固定末步权重的完整VAL：全图/近中远与共同stride8阶段，不访问TEST。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, distance_limits, ground_truth
from research.paper_ablation_20261102.handoff_train import build_ablation
from tools.smoke_outdoor_datasets import offline_only


def finish(raw):
    return {stage: {region: depth_metrics(sums) for region, sums in regions.items()}
            for stage, regions in raw.items()}


@torch.inference_mode()
def run(settings):
    offline_only()
    seed_all(settings["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = settings["device"]
    torch.cuda.set_device(device)
    output = Path(settings["output"])
    audit = "geometry_audit.json" if "geometry_ablation" in settings else "handoff_audit.json"
    if not (output / audit).exists():
        raise RuntimeError("必须先完成本组训练及真实恢复核验")
    if (output / "full_val.json").exists():
        raise FileExistsError(output / "full_val.json")
    saved = torch.load(output / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    if "geometry_ablation" in settings:
        from research.paper_ablation_20261102.geometry import build_geometry_model
        model, config = build_geometry_model(saved, settings["geometry_ablation"])
    else:
        model, config = build_ablation(saved, settings["variant"])
    model.to(device).eval().requires_grad_(False)
    install_position_cache(model.foundation.model)
    config = copy.deepcopy(config)
    config["datasets"] = config.pop("validation_datasets", config["datasets"])
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
    dataset = MixedDataset(config, "val")
    report = {"completed": False, "split": "smoke_val" if settings.get("smoke") else "full_val", "test_accessed": False,
              "variant": settings.get("geometry_ablation", settings["variant"]), "seed": settings["seed"],
              "checkpoint_sha256": digest(output / "final.pt"), "datasets": {},
              "input_mode": "equal_pixels", "selection": "fixed_last_step"}
    plan = {domain["name"]: {"frames": min(2, len(dataset.datasets[i])) if settings.get("smoke") else len(dataset.datasets[i]),
                              "manifest_sha256": digest(domain["manifests"]["val"])}
            for i, domain in enumerate(dataset.domains)}
    report["plan"] = plan
    write_json(output / "full_val_plan.json", plan)
    for domain_index, domain in enumerate(dataset.domains):
        count = plan[domain["name"]]["frames"]
        keys = [(domain_index, index, -1) for index in range(count)]
        loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                            num_workers=2, pin_memory=True)
        local = dict(domain, reference_index=0)
        totals = {}
        frame_path = output / ("full_val_" + domain["name"] + ".jsonl")
        with frame_path.open("w") as stream:
            for index, batch in enumerate(loader):
                prediction = model(equal_pixel_inputs(move(batch["inputs"], device)), domain, reference=0)
                scale = float(prediction["unit_scale"])
                depth = prediction["depth"].cpu() * scale
                validity = prediction["valid"].cpu()
                stages = {"depth": depth, "coarse_stride8": prediction["coarse"]["depth"].cpu() * scale,
                          "fine_stride8": prediction["fine"]["depth"][..., ::2, ::2].cpu() * scale,
                          "final_stride8": depth[..., ::8, ::8]}
                del prediction
                truth, valid = ground_truth(batch, local)
                near, far = distance_limits(domain)
                masks = {"full": valid, "near": valid & (truth < near),
                         "mid": valid & (truth >= near) & (truth < far), "far": valid & (truth >= far)}
                raw = {}
                for stage, estimate in stages.items():
                    full = stage == "depth"
                    target = truth if full else truth[..., ::8, ::8]
                    good = validity if full else torch.isfinite(estimate) & (estimate > 0)
                    for region, mask in masks.items():
                        sums = depth_sums(estimate, good, target, mask if full else mask[..., ::8, ::8])
                        raw.setdefault(stage, {})[region] = sums
                        add_statistics(totals.setdefault(stage, {}).setdefault(region, {}), sums)
                stream.write(json.dumps({"index": index, "raw": raw}) + "\n")
                if (index + 1) % 200 == 0:
                    stream.flush()
                    print(json.dumps({"domain": domain["name"], "frames": index + 1, "total": count}), flush=True)
        report["datasets"][domain["name"]] = {
            "frames": count, "units": domain["geometry_units"], "raw": totals,
            "metrics": finish(totals), "manifest_sha256": plan[domain["name"]]["manifest_sha256"],
        }
        write_json(output / "full_val_progress.json", report)
    report["completed"] = True
    report["source_sha256"] = digest(__file__)
    write_json(output / "full_val.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
