"""固定 DEV 子集的四峰机制诊断与小噪声重复；不训练，不读 TEST。"""

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
from research.near_evidence_20261030.pilot import path, payload_hash
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from research.ruler_flow_20261030.pilot import install, load_model
from tools.smoke_outdoor_datasets import offline_only


STUDY = ROOT / "runs/active/ruler_flow_20261030"


def prepare(settings, complete=False):
    result, data = {}, {}
    for group, names in settings["domains"].items():
        training = json.loads((STUDY / f"{group}_flow/settings.json").read_text())
        old_plan = json.loads((STUDY / f"{group}_flow/plan.json").read_text())
        source = json.loads(path(training["data_config"]).read_text())
        config = copy.deepcopy(source)
        config["datasets"] = [copy.deepcopy(item) for item in source.get("validation_datasets", source["datasets"])
                              if item["name"] in names]
        config.pop("validation_datasets", None)
        selections = {}
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
            name = domain["name"]
            original = old_plan["validation"][name]
            if digest(path(domain["manifests"]["val"])) != original["manifest_sha256"]:
                raise RuntimeError(f"{name}: VAL清单改变")
            indices = original["development"][:4]
            selections[name] = {"indices": indices, "stochastic_indices": indices[:2],
                                "manifest_sha256": original["manifest_sha256"],
                                "selection_sha256": payload_hash(indices)}
        if set(selections) != set(names):
            raise RuntimeError("请求的域未全部找到")
        methods = {"frozen": {"checkpoint": training["checkpoint"], "steps": 1}}
        methods["reduced4"] = {"checkpoint": training["checkpoint"], "steps": 0,
                                "install_zero_head": True}
        for label, folder, steps in (("direct1", "direct", 1), ("flow1", "flow", 1),
                                     ("flow2", "flow", 2), ("flow4", "flow", 4),
                                     ("flow_no_direction1", "flow_no_direction", 1)):
            methods[label] = {"checkpoint": str((STUDY / f"{group}_{folder}/final.pt").relative_to(ROOT)),
                              "steps": steps}
        if complete:
            for info in methods.values():
                info["checkpoint_sha256"] = digest(path(info["checkpoint"]))
        data[group] = config
        result[group] = {"selection": selections, "methods": methods}
    plan = {
        "groups": result, "grid_stride": 4, "noise_sigma": .05, "noise_seeds": [0, 1, 2],
        "source_sha256": {name: digest(ROOT / name) for name in (
            "research/ruler_flow_20261030/diagnostics.py", "research/ruler_flow_20261030/variants.py",
            "research/ruler_flow_20261030/direction.py", "research/ruler_flow_20261030/pilot.py")},
        "test_accessed": False, "training_performed": False,
        "scope": "本轮原16VAL中development前4；已用于研发，不是独立确认或TEST。所有方法相同GT分母、共同stride4。噪声重复不是校准不确定性。",
    }
    output = path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free / 2 ** 30 < settings.get("minimum_free_gib", 200):
        raise RuntimeError("数据盘剩余空间不足200GiB")
    filename = output / ("plan.json" if complete else "prepared.json")
    if filename.exists() and json.loads(filename.read_text()) != plan:
        raise RuntimeError(f"拒绝覆盖不同计划：{filename}")
    write_json(filename, plan)
    return data, plan, output


def truth_masks(batch, domain):
    truth, valid = ground_truth(batch, dict(domain, reference_index=0))
    selected = regions(truth, valid, domain)
    masks = {name: selected[key][:, 0, ::4, ::4] for name, key in (
        ("full", "full"), ("near", "near"), ("edge", "gradient_band"))}
    return truth[:, 0, ::4, ::4].double(), masks


def summarize_maps(values, masks):
    result = {}
    for region, mask in masks.items():
        count = int(mask.sum())
        sums = {key: float(value.double()[mask].sum()) for key, value in values.items()}
        if any(not torch.isfinite(value[mask]).all() for value in values.values()):
            raise RuntimeError("固定GT分母内有非有限诊断值，不允许静默删除")
        result[region] = {"gt_pixels": count, "sums": sums,
                          "metrics": {key: total / count if count else None for key, total in sums.items()}}
    return result


def aggregate(rows):
    result = {}
    for row in rows:
        for region, record in row.items():
            merged = result.setdefault(region, {"gt_pixels": 0, "sums": {}})
            merged["gt_pixels"] += record["gt_pixels"]
            for key, total in record["sums"].items():
                merged["sums"][key] = merged["sums"].get(key, 0.) + total
    for merged in result.values():
        count = merged["gt_pixels"]
        merged["metrics"] = {key: total / count if count else None for key, total in merged["sums"].items()}
    return result


def analyze(native, domain, truth, masks, reference=None):
    stage = native["fine"]
    scale = float(native["unit_scale"])
    low = stage["lower"].double().clamp(domain["near"] / scale, domain["far"] / scale) * scale
    high = stage["upper"].double().clamp(domain["near"] / scale, domain["far"] / scale) * scale
    sampled, actual = stage["sampling_depth"].double() * scale, stage["candidate_depth"].double() * scale
    valid = stage["valid"].bool()
    selected = stage["selected_index"]
    if not bool(valid.any(1)[masks["full"]].all()):
        raise RuntimeError("GT合法点没有任何候选；不可静默排除此点")
    frozen = actual.shape[1] == 12
    union_low = low.unflatten(1, (4, 3)).amin(2) if frozen else low
    union_high = high.unflatten(1, (4, 3)).amax(2) if frozen else high
    group_valid = valid.unflatten(1, (4, 3)).any(2) if frozen else valid
    covered = ((truth[:, None] >= low) & (truth[:, None] <= high) & valid).any(1)
    union_covered = ((truth[:, None] >= union_low) & (truth[:, None] <= union_high) & group_valid).any(1)
    if not torch.equal(covered[masks["full"]], union_covered[masks["full"]]):
        raise RuntimeError("原组三点合法区间的联合存在空洞，不能假设单个连续窗口")
    comparison = None
    if reference is not None:
        expected_low, expected_high, expected_covered = [value.to(low.device) for value in reference]
        difference = torch.maximum((expected_low - union_low).abs(), (expected_high - union_high).abs())
        mismatch = (expected_covered != covered) & masks["full"]
        torch.testing.assert_close(union_low, expected_low, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(union_high, expected_high, rtol=2e-6, atol=2e-6)
        if bool(mismatch.any()):
            raise RuntimeError("新旧连续窗口GT覆盖不相等")
        comparison = {"gt_coverage_mismatch": int(mismatch.sum()), "max_bound_absolute_difference": float(difference.max())}
    sample_error = (sampled - truth[:, None]).abs().masked_fill(~valid, torch.inf)
    actual_error = (actual - truth[:, None]).abs().masked_fill(~valid, torch.inf)
    fine = stage["depth"][:, 0].double() * scale
    fine_error = (fine - truth).abs()
    anchor = stage.get("anchor_depth", stage["sampling_depth"]).double() * scale
    anchor_error = (anchor - truth[:, None]).abs()
    good_anchor = (anchor_error <= .05 * truth[:, None]) & valid
    good_actual = (actual_error <= .05 * truth[:, None]) & valid
    movement = (actual.clamp_min(1e-12).log() - anchor.clamp_min(1e-12).log()).abs()
    group_low = union_low.repeat_interleave(3, 1) if frozen else union_low
    group_high = union_high.repeat_interleave(3, 1) if frozen else union_high
    tolerance = 1e-5
    boundary = ((actual.clamp_min(1e-12).log() - group_low.clamp_min(1e-12).log()).abs() < tolerance)
    boundary |= ((actual.clamp_min(1e-12).log() - group_high.clamp_min(1e-12).log()).abs() < tolerance)
    domain_boundary = (actual / domain["near"] - 1).abs() < tolerance
    domain_boundary |= (actual / domain["far"] - 1).abs() < tolerance
    denominator = valid.sum(1).clamp_min(1)
    interval_distance = torch.maximum(low - truth[:, None], truth[:, None] - high).clamp_min(0).masked_fill(~valid, torch.inf)
    values = {
        "interval_gt_coverage": covered.double(), "interval_oracle_MAE": interval_distance.amin(1),
        "sampling_oracle_MAE": sample_error.amin(1), "continuous_oracle_MAE": actual_error.amin(1),
        "fine_MAE": fine_error, "fine_MSE": (fine - truth).square(),
        "selection_regret_MAE": fine_error - actual_error.amin(1),
        "selected_sampling_MAE": sample_error.gather(1, selected)[:, 0],
        "anchor_oracle_MAE": anchor_error.masked_fill(~valid, torch.inf).amin(1),
        "oracle_change_from_anchor": actual_error.amin(1) - anchor_error.masked_fill(~valid, torch.inf).amin(1),
        "mean_branch_error_change": torch.where(valid, actual_error - anchor_error, 0.).sum(1) / denominator,
        "selected_branch_error_change": (actual_error - anchor_error).gather(1, selected)[:, 0],
        "good_anchor_branch_count": good_anchor.sum(1).double(),
        "good_anchor_worsened_branch_count": (good_anchor & (actual_error > anchor_error + 1e-6)).sum(1).double(),
        "good_anchor_crossed_5pct_branch_count": (good_anchor & ~good_actual).sum(1).double(),
        "anchor_good_candidate_presence": good_anchor.any(1).double(),
        "all_good_anchor_candidates_lost": (good_anchor.any(1) & ~good_actual.any(1)).double(),
        "mean_abs_log_movement": (movement * valid).sum(1) / denominator,
        "selected_abs_log_movement": movement.gather(1, selected)[:, 0],
        "union_boundary_fraction": (boundary & valid).sum(1) / denominator,
        "domain_boundary_fraction": (domain_boundary & valid).sum(1) / denominator,
        "selected_union_boundary_fraction": boundary.gather(1, selected)[:, 0].double(),
        "final_stride4_MAE": (native["depth"][:, 0, ::4, ::4].double() * scale - truth).abs(),
    }
    reference_values = tuple(value.detach().cpu() for value in (union_low, union_high, covered))
    return summarize_maps(values, masks), reference_values, comparison


def restore(info, device):
    if info.get("install_zero_head"):
        model, _ = load_model(info["checkpoint"], device, 1)
        install(model, {"objective": "flow", "direction": True, "inference_steps": 0,
                        "hidden": 32, "sigma": .05})
        model.requires_grad_(False).eval()
        return model
    return load_model(info["checkpoint"], device, info["steps"])[0]


def noise_statistics(predictions, deterministic, truth, masks):
    values = torch.stack(predictions).double()
    mean = values.mean(0)
    maps = {"output_variance": (values - mean).square().mean(0),
            "mean_seed_MAE": (values - truth).abs().mean(0),
            "mean_abs_change_from_sigma0": (values - deterministic).abs().mean(0)}
    for i in range(3):
        maps[f"seed{i}_MAE"] = (values[i] - truth).abs()
    statistics = summarize_maps(maps, masks)
    pairs = torch.stack([(values[i] - values[j]).abs() for i, j in ((0, 1), (0, 2), (1, 2))])
    samples = {region: pairs[:, mask].flatten().cpu() for region, mask in masks.items()}
    return statistics, samples


@torch.inference_mode()
def run(settings, prepare_only=False):
    offline_only()
    if prepare_only:
        _, plan, output = prepare(settings)
        print(json.dumps({"prepared": str(output), "groups": plan["groups"]}, ensure_ascii=False), flush=True)
        return
    completion = STUDY / "logs/formal_queue.exit_code.txt"
    if not completion.exists() or completion.read_text().strip() != "0":
        raise RuntimeError("正式训练队列尚未成功结束，诊断GPU任务禁止启动")
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configs, plan, output = prepare(settings, complete=True)
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    result = {"domains": {}, "stochastic": {}, "plan_sha256": digest(output / "plan.json")}
    for group, config in configs.items():
        dataset = MixedDataset(config, "val")
        references, deterministic = {}, {}
        group_plan = plan["groups"][group]
        for method, info in group_plan["methods"].items():
            model = restore(info, device)
            for domain_index, domain in enumerate(dataset.domains):
                name = domain["name"]
                frames = []
                for index in group_plan["selection"][name]["indices"]:
                    batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                    native = model(batch["inputs"], domain, reference=0)
                    truth, masks = truth_masks(batch, domain)
                    key = (name, index)
                    stats, bounds, equality = analyze(native, domain, truth, masks, references.get(key))
                    if method == "frozen":
                        references[key] = bounds
                    if method in ("flow1", "flow2") and index in group_plan["selection"][name]["stochastic_indices"]:
                        scale = float(native["unit_scale"])
                        deterministic[(name, index, method)] = {
                            "fine": native["fine"]["depth"][:, 0].double().cpu() * scale,
                            "final": native["depth"][:, 0, ::4, ::4].double().cpu() * scale,
                        }
                    frames.append({"index": index, "candidate_count": native["fine"]["candidate_depth"].shape[1],
                                   "regions": stats, "interval_comparison": equality})
                record = {"units": domain["geometry_units"], "frames": frames,
                          "aggregate": aggregate([frame["regions"] for frame in frames])}
                result["domains"].setdefault(name, {})[method] = record
                write_json(output / f"{name}_{method}.json", record)
                print(json.dumps({"done": name, "method": method}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for method in ("flow1", "flow2"):
            info = group_plan["methods"][method]
            model, _ = load_model(info["checkpoint"], device, info["steps"])
            model.decoder.flow_options["inference_noise_sigma"] = .05
            for domain_index, domain in enumerate(dataset.domains):
                name = domain["name"]
                frames, quantiles = [], {"fine": {}, "final": {}}
                for index in group_plan["selection"][name]["stochastic_indices"]:
                    batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                    truth, masks = truth_masks(batch, domain)
                    predicted = {"fine": [], "final": []}
                    for seed in (0, 1, 2):
                        model.decoder.flow_options["inference_seed"] = seed
                        native = model(batch["inputs"], domain, reference=0)
                        scale = float(native["unit_scale"])
                        predicted["fine"].append(native["fine"]["depth"][:, 0] * scale)
                        predicted["final"].append(native["depth"][:, 0, ::4, ::4] * scale)
                    levels = {}
                    for level, predictions in predicted.items():
                        zero = deterministic[(name, index, method)][level].to(device)
                        stats, differences = noise_statistics(predictions, zero, truth, masks)
                        levels[level] = stats
                        for region, vector in differences.items():
                            quantiles[level].setdefault(region, []).append(vector)
                    frames.append({"index": index, "levels": levels})
                merged = {level: aggregate([frame["levels"][level] for frame in frames]) for level in ("fine", "final")}
                for level in merged:
                    for region, vectors in quantiles[level].items():
                        values = torch.cat(vectors)
                        merged[level][region]["pairwise_abs_diff_quantiles"] = {
                            str(q): float(torch.quantile(values, q)) if values.numel() else None for q in (.5, .95, .99, 1.)}
                record = {"units": domain["geometry_units"], "sigma": .05, "seeds": [0, 1, 2],
                          "frames": frames, "aggregate": merged,
                          "scope": "同输入三次初态噪声重复，报告全部seed，不选择最佳seed，不称校准不确定性。"}
                result["stochastic"].setdefault(name, {})[method] = record
                write_json(output / f"{name}_{method}_noise.json", record)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    for name, methods in result["domains"].items():
        for region in ("full", "near", "edge"):
            counts = [record["aggregate"][region]["gt_pixels"] for record in methods.values()]
            if len(set(counts)) != 1:
                raise RuntimeError(f"{name}/{region}: 方法间GT分母不一致")
    result["completed"] = True
    result["scope"] = plan["scope"]
    write_json(output / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    run(json.loads(path(arguments.config).read_text()), arguments.prepare_only)
