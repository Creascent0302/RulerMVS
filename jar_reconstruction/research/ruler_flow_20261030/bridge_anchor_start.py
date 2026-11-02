"""最后一次机制probe：起点加权Flow的默认零噪声速度，不重训旧对照。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import parameter_hash, path
from research.ruler_flow_20261030.diagnostics import aggregate, summarize_maps, truth_masks
from research.ruler_flow_20261030.flow import bridge_targets
from research.ruler_flow_20261030.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run(settings):
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = path(settings["output"])
    plan = json.loads((output / "plan.json").read_text())
    reference_file = output / "bridge_zero_start.json"
    reference = json.loads(reference_file.read_text())
    destination = output / "bridge_anchor_start.json"
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(output).free / 2 ** 30 < 200:
        raise RuntimeError("数据盘剩余空间不足200GiB")
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    result = {"reference_sha256": digest(reference_file), "source_sha256": digest(Path(__file__)),
              "sigma": 0., "time": 0., "sign_target_threshold_q": 1e-4, "domains": {},
              "parameter_hashes": {}, "weight_sha256": {},
              "scope": "最后机制probe：原4域前2DEV，仅新flow_anchor各一次_read；旧zero/direct/flow按固定bridge_zero_start.json引用。所有窗口、GT责任、分母、目标统计均逐帧比对。不训练、不选权重，不以此代替完整精度。"}
    for group, group_plan in plan["groups"].items():
        training = json.loads((output.parent / f"{group}_flow_anchor/settings.json").read_text())
        source = json.loads(path(training["data_config"]).read_text())
        config = copy.deepcopy(source)
        config["datasets"] = [copy.deepcopy(domain) for domain in source.get("validation_datasets", source["datasets"])
                              if domain["name"] in group_plan["selection"]]
        config.pop("validation_datasets", None)
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
            if digest(path(domain["manifests"]["val"])) != group_plan["selection"][domain["name"]]["manifest_sha256"]:
                raise RuntimeError("VAL清单改变")
        checkpoint = output.parent / f"{group}_flow_anchor/final.pt"
        result["weight_sha256"][group] = digest(checkpoint)
        model, _ = load_model(checkpoint, device, 1)
        before = parameter_hash(dict(model.named_parameters()))
        dataset = MixedDataset(config, "val")
        for domain_index, domain in enumerate(dataset.domains):
            name, frames = domain["name"], []
            old_domain = reference["domains"][name]
            for index in group_plan["selection"][name]["stochastic_indices"]:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                scale = float(domain["depth_scale"])
                model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                context = model.decoder._coarse_context(model.encode(batch["inputs"], domain, reference=0))
                read = model.decoder._read(context, torch.zeros_like(context["anchor"]), 0.)
                truth, masks = truth_masks(batch, domain)
                bridge = bridge_targets(context["anchor_log"], context["radius"], context["low_state"], context["high_state"],
                                        truth[:, None].float() / scale, masks["full"][:, None], "flow", 0.,
                                        torch.Generator(device=device).manual_seed(20261030 + index))
                assigned = bridge["assigned"] & read["valid"]
                target, velocity = bridge["velocity"], read["velocity"]
                labels = {region: mask & assigned.any(1) for region, mask in masks.items()}
                directional = assigned & (target.abs() > 1e-4)
                sign_labels = {region: mask & directional.any(1) for region, mask in masks.items()}

                def reduce(value):
                    return torch.where(assigned, value, 0.).sum(1) / assigned.sum(1).clamp_min(1)

                errors = {"zero_velocity_MSE": reduce(target.square()), "target_velocity_MAE": reduce(target.abs()),
                          "flow_anchor_velocity_MSE": reduce((velocity - target).square()),
                          "flow_anchor_velocity_MAE": reduce((velocity - target).abs())}
                signs = {"flow_anchor_sign_accuracy": ((velocity * target > 0) & directional).sum(1) / directional.sum(1).clamp_min(1)}
                frame = {"index": index, "assigned_pixels": int(assigned.any(1).sum()),
                         "original_gt_pixels": int(masks["full"].sum()),
                         "velocity_errors": summarize_maps(errors, labels), "sign_accuracy": summarize_maps(signs, sign_labels)}
                old_frame = next(row for row in old_domain["frames"] if row["index"] == index)
                for region in ("near", "full", "edge"):
                    for part in ("velocity_errors", "sign_accuracy"):
                        if frame[part][region]["gt_pixels"] != old_frame[part][region]["gt_pixels"]:
                            raise RuntimeError("与旧诊断分母不同")
                    for metric in ("zero_velocity_MSE", "target_velocity_MAE"):
                        actual = frame["velocity_errors"][region]["sums"][metric]
                        expected = old_frame["velocity_errors"][region]["sums"][metric]
                        if abs(actual - expected) > 1e-8 * max(1., abs(expected)):
                            raise RuntimeError("与旧诊断速度目标统计不同")
                frames.append(frame)
            error = aggregate([frame["velocity_errors"] for frame in frames])
            signs = aggregate([frame["sign_accuracy"] for frame in frames])
            comparison = {}
            for region in error:
                old_error = old_domain["velocity_errors"][region]["metrics"]
                old_sign = old_domain["sign_accuracy"][region]["metrics"]
                comparison[region] = {"zero_MSE": old_error["zero_velocity_MSE"],
                    "direct_MSE": old_error["direct_velocity_MSE"], "flow_MSE": old_error["flow_velocity_MSE"],
                    "flow_anchor_MSE": error[region]["metrics"]["flow_anchor_velocity_MSE"],
                    "direct_sign_accuracy": old_sign["direct_sign_accuracy"], "flow_sign_accuracy": old_sign["flow_sign_accuracy"],
                    "flow_anchor_sign_accuracy": signs[region]["metrics"]["flow_anchor_sign_accuracy"]}
            result["domains"][name] = {"frames": frames, "velocity_errors": error, "sign_accuracy": signs,
                                       "comparison": comparison, "same_denominators_and_targets": True}
            print(json.dumps({"anchor_start_done": name}), flush=True)
        after = parameter_hash(dict(model.named_parameters()))
        if before != after:
            raise RuntimeError("诊断改变了参数")
        result["parameter_hashes"][group] = {"before": before, "after": after, "identical": True}
        del model, context, read
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    write_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
