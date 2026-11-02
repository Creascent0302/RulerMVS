"""固定VAL小样本的近距证据诊断；GT只在前向结束后的离线归因中使用。"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.geometry import pixel_grid, project, sample_maps, unproject
from jar.pretraining.comparison import Models, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions, distance_limits
from tools.benchmark_public_speed import prepare_model
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only

SOURCE = ROOT / "runs/active/v2_restart_20261029/evaluation/configs"
OUTPUT = ROOT / "runs/active/near_evidence_20261030/diagnostics"


def read(path):
    return json.loads(Path(path).read_text())


def capture_attention(storage):
    def hook(module, args, result):
        reference, sources, valid, derivative = args[:4]
        cosine = (sources * reference[:, None, :, None]).sum(2)
        storage.append({
            "similarity": (cosine * result["weights"]).sum(1).detach(),
            "null": result["null_mass"][:, 0].detach(),
            "gmax": torch.where(valid, derivative.square().sum(2).sqrt(), 0).amax(1).detach(),
        })
    return hook


def gather(value, index):
    return value.gather(1, index)[:, 0]


def add_group(destination, name, mask, values):
    item = destination.setdefault(name, {"pixels": 0, "sums": {}, "counts": {}})
    item["pixels"] += int(mask.sum())
    for key, value in values.items():
        usable = mask & torch.isfinite(value)
        item["sums"][key] = item["sums"].get(key, 0.) + float(value[usable].double().sum())
        item["counts"][key] = item["counts"].get(key, 0) + int(usable.sum())


def finish(groups):
    result = {}
    for key, value in groups.items():
        means = {field: total / value["counts"][field] if value["counts"][field] else None
                 for field, total in value["sums"].items()}
        result[key] = dict(value, means=means)
    return result


def analyze(inputs, targets, native, baseline, attention, domain):
    """所有阶段统一到主图整数stride8格点；几何统计使用外部物理单位。"""
    batch = {"inputs": inputs, "targets": targets}
    truth_full, valid_full = ground_truth(batch, domain)
    ref = domain["reference_index"]
    truth = truth_full[:, 0, ::8, ::8].float()
    valid = valid_full[:, 0, ::8, ::8]
    h, w = inputs["rgb"].shape[-2:]
    hc, wc = truth.shape[-2:]
    K, T = inputs["K"][0].float(), inputs["T_cam_from_rig"][0].float()
    grid = pixel_grid(hc, wc, stride=8, device=truth.device).reshape(-1, 2)
    safe_z = torch.where(valid, truth, 1).reshape(-1)
    world, rays = unproject(grid, safe_z, K[ref], T[ref])
    uv, camera_z, legal, J = project(world, K, T, (h, w), inputs["view_mask"][0], inputs["image_mask"][0])
    legal[ref] = False
    derivative = (J * rays[None, :, None]).sum(-1)
    legal &= torch.linalg.vector_norm(derivative, dim=-1) > 1e-5
    known = sample_maps(targets["depth_valid"][0, :, None].float(), uv, mode="nearest")[..., 0] > .5
    observed = sample_maps(targets["depth"][0, :, None].float(), uv, mode="nearest")[..., 0]
    known &= legal & torch.isfinite(observed) & (observed > 0)
    tolerance = .02 * camera_z.abs().clamp_min(1e-3)
    visible = known & ((observed - camera_z).abs() < tolerance)
    occluded = known & (observed < camera_z - tolerance)
    support = legal.sum(0).reshape(1, hc, wc)
    visible_count = visible.sum(0).reshape(1, hc, wc)
    known_count = known.sum(0).reshape(1, hc, wc)

    scale = native["unit_scale"]
    coarse = native["coarse"]
    fine = {key: value[..., ::2, ::2] for key, value in native["fine"].items()
            if torch.is_tensor(value) and value.ndim >= 3}
    coarse_z = coarse["sampling_depth"] * scale
    nearest = (coarse_z.log() - truth[:, None].clamp_min(1e-6).log()).abs().argmin(1, keepdim=True)
    chosen_z = gather(coarse_z, nearest).reshape(-1)
    sample_world, _ = unproject(grid, chosen_z, K[ref], T[ref])
    sample_uv, _, sample_legal, _ = project(sample_world, K, T, (h, w), inputs["view_mask"][0], inputs["image_mask"][0])
    ratio = uv.new_tensor((inputs["matching_rgb"].shape[-1] / w, inputs["matching_rgb"].shape[-2] / h))
    displacement = torch.linalg.vector_norm((sample_uv - uv) * ratio / 4, dim=-1)
    displacement = displacement.masked_fill(~(legal & sample_legal), torch.nan)
    step_error = torch.nanmedian(displacement, dim=0).values.reshape(1, hc, wc)

    final_z = native["depth"][:, 0, ::8, ::8] * scale
    base_z = baseline[:, 0, ::8, ::8]
    coarse_prediction = coarse["depth"][:, 0] * scale
    fine_prediction = fine["depth"][:, 0] * scale
    fine_candidates = fine["candidate_depth"] * scale
    fine_sampling = fine["sampling_depth"] * scale
    fine_lower, fine_upper = fine["lower"] * scale, fine["upper"] * scale
    oracle_actual = (fine_candidates - truth[:, None]).abs().amin(1)
    oracle_interval = torch.maximum(fine_lower - truth[:, None], truth[:, None] - fine_upper).clamp_min(0).amin(1)
    fine_index = (fine_sampling.clamp_min(1e-6).log() - truth[:, None].clamp_min(1e-6).log()).abs().argmin(1, keepdim=True)
    training_reachable = (truth >= gather(fine_lower, fine_index)) & (truth <= gather(fine_upper, fine_index))
    selected_support = gather(fine["support"], fine["selected_index"])
    nearest_coarse_support = gather(coarse["support"], nearest)
    nearest_null = gather(attention[0]["null"], nearest)
    selected_null = gather(attention[1]["null"][..., ::2, ::2], fine["selected_index"])
    nearest_similarity = gather(attention[0]["similarity"], nearest)
    selected_similarity = gather(attention[1]["similarity"][..., ::2, ::2], fine["selected_index"])
    coarse_oracle = (coarse["candidate_depth"] * scale - truth[:, None]).abs().amin(1)
    values = {
        "own_MAE": (final_z - truth).abs(), "own_sq_error": (final_z - truth).square(),
        "own_bias": final_z - truth, "baseline_MAE": (base_z - truth).abs(),
        "baseline_sq_error": (base_z - truth).square(), "baseline_bias": base_z - truth,
        "coarse_MAE": (coarse_prediction - truth).abs(), "coarse_bias": coarse_prediction - truth,
        "fine_MAE": (fine_prediction - truth).abs(), "fine_bias": fine_prediction - truth,
        "coarse_oracle_MAE": coarse_oracle, "fine_oracle_actual_MAE": oracle_actual,
        "fine_interval_oracle_MAE": oracle_interval, "fine_localization_gap": oracle_actual - oracle_interval,
        "fine_selection_regret": (fine_prediction - truth).abs() - oracle_actual,
        "fine_training_reachable": training_reachable.float(),
        "fine_any_interval_covers": (oracle_interval <= 1e-6).float(),
        "true_geometric_support": support.float(), "true_visible_support": visible_count.float(),
        "source_label_known": known_count.float(), "selected_support": selected_support.float(),
        "nearest_coarse_support": nearest_coarse_support.float(),
        "nearest_coarse_null": nearest_null, "selected_fine_null": selected_null,
        "nearest_coarse_similarity": nearest_similarity, "selected_fine_similarity": selected_similarity,
        "nearest_coarse_projection_error_F4": step_error,
        "gt_depth": truth, "prediction_depth": final_z,
        "selected_support_exceeds_true": (selected_support > support).float(),
        "overestimate_gt_25percent": (final_z > truth * 1.25).float(),
        "underestimate_gt_25percent": (final_z < truth * .75).float(),
    }
    near_limit, far_limit = distance_limits(domain)
    near = valid & (truth < near_limit)
    masks = {"all": valid, "near": near, "mid": valid & (truth >= near_limit) & (truth < far_limit),
             "far": valid & (truth >= far_limit)}
    for label, selected in (("0", support == 0), ("1", support == 1), ("2plus", support >= 2)):
        masks["near_geometric_support_" + label] = near & selected
    masks.update({
        "near_visible_any": near & (visible_count > 0),
        "near_known_but_no_visible": near & (known_count > 0) & (visible_count == 0),
        "near_no_known_source_label": near & (known_count == 0),
        "near_occluded_any": near & occluded.any(0).reshape(1, hc, wc),
        "near_selected_support_more_than_true": near & (selected_support > support),
        "near_selected_support_at_most_true": near & (selected_support <= support),
        "near_overestimate": near & (final_z > truth),
        "near_underestimate": near & (final_z <= truth),
        "near_error_over_25percent": near & ((final_z - truth).abs() > truth * .25),
        "near_overestimate_over_25percent": near & (final_z > truth * 1.25),
        "near_underestimate_over_25percent": near & (final_z < truth * .75),
    })
    full_regions = regions(truth_full, valid_full, domain)
    masks["near_edge"] = near & full_regions["gradient_band"][:, 0, ::8, ::8]
    masks["near_smooth"] = near & full_regions["smooth"][:, 0, ::8, ::8]
    yy = torch.arange(hc, device=truth.device)[None, :, None]
    for label, selected in (("top", yy < hc / 3), ("middle", (yy >= hc / 3) & (yy < hc * 2 / 3)), ("bottom", yy >= hc * 2 / 3)):
        masks["near_" + label] = near & selected
    for label, selected in (("0_to_0.5", step_error <= .5), ("0.5_to_1", (step_error > .5) & (step_error <= 1)),
                            ("1_to_2", (step_error > 1) & (step_error <= 2)), ("above_2", step_error > 2)):
        masks["near_projection_error_" + label] = near & selected
    return masks, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    offline_only()
    output.mkdir(parents=True, exist_ok=True)
    public, carla, others = [read(SOURCE / (name + ".json")) for name in ("public", "carla", "carla_other_refs")]
    jobs = [(d, public, "v2_restart_public_best") for d in public["datasets"]]
    jobs += [(d, c, "v2_restart_carla_best") for c in (carla, others) for d in c["datasets"]]
    priorities = {"CARLA_new_upper": 0, "Hypersim_subset": 1, "VKITTI2": 2} if args.split == "val" else {
        "CARLA_new_upper": 0, "Hypersim_subset": 1, "BlendedMVS": 2, "DynamicReplica_valid": 3}
    jobs.sort(key=lambda item: priorities.get(item[0]["name"], 3))
    device = configure(public, 1)
    report = {"completed": False, "split": args.split, "stride": 8,
              "protocol": "每域均匀16样本；VAL偶数位development、奇数confirmation；TEST仅固定模型历史结果解释，不选模/调参。GT只离线诊断。",
              "domains": {}}
    plan = {}
    for domain, config, own in jobs:
        dataset = build_dataset(dataset_config(domain, config), args.split, load_targets=True)
        indices = torch.linspace(0, len(dataset) - 1, min(16, len(dataset))).long().tolist()
        plan[domain["name"]] = {"indices": indices,
            ("development" if args.split == "val" else "diagnostic_even"): indices[::2],
            ("confirmation" if args.split == "val" else "diagnostic_odd"): indices[1::2]}
    write_json(output / "plan.json", plan)
    for domain, config, own in jobs:
        config = dict(config, methods={name: config["methods"][name] for name in (own, "mvsanywhere_refined")})
        models = Models(config, device)
        dataset = build_dataset(dataset_config(domain, config), args.split, load_targets=True)
        groups, dev, confirmation, frames = {}, {}, {}, []
        for number, index in enumerate(plan[domain["name"]]["indices"]):
            batch = move(collate_frames([dataset[index]]), device)
            inputs, targets = batch["inputs"], batch["targets"]
            net = prepare_model(models, own, config["methods"][own], domain, inputs)
            prepare_model(models, "mvsanywhere_refined", config["methods"]["mvsanywhere_refined"], domain, inputs)
            attention = []
            handle = net.decoder.source.register_forward_hook(capture_attention(attention))
            with torch.inference_mode():
                native = net(inputs, domain, reference=domain["reference_index"])
                handle.remove()
                baseline, _ = models.convert(models.forward("mvsanywhere_refined", domain, inputs), inputs)
                masks, values = analyze(inputs, targets, native, baseline, attention, domain)
                local = {}
                for name, mask in masks.items():
                    add_group(groups, name, mask, values)
                    add_group(dev if number % 2 == 0 else confirmation, name, mask, values)
                    add_group(local, name, mask, values)
            subset = ("development" if number % 2 == 0 else "confirmation") if args.split == "val" else "fixed_test_diagnostic"
            frames.append({"index": index, "subset": subset, "groups": finish(local)})
            print(json.dumps({"domain": domain["name"], "index": index, "near": finish(local)["near"]["means"]}, ensure_ascii=False), flush=True)
            del batch, inputs, targets, native, baseline, attention, masks, values
        result = {"units": domain["geometry_units"], "near_limit": distance_limits(domain)[0], "own_method": own,
                  "groups": finish(groups), "frames": frames}
        if args.split == "val":
            result.update(development=finish(dev), confirmation=finish(confirmation))
        report["domains"][domain["name"]] = result
        write_json(output / (domain["name"] + ".json"), result)
        write_json(output / "summary.json", report)
        models.loaded.clear()
        del models, net
        torch.cuda.empty_cache()
    report["completed"] = True
    write_json(output / "summary.json", report)


if __name__ == "__main__":
    main()
