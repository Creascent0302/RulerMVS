"""只读TRAIN/CPU增强诊断；不创建模型、不改变活跃训练文件。

覆盖统计使用全部有效GT；96格oracle使用每组最多2048个确定性GT点。
oracle是事后最近粗采样格误差，不是网络预测、连续offset或细候选性能。
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from jar.pretraining.data import collate_mixed
from research.overnight_v6_v7_mvs_20261103.common_data import own_training_domain
from research.rulermvs_v8_20261103.data import SharedDataset, compose_training_domain
from tools.smoke_outdoor_datasets import offline_only


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accessed_settings(source):
    """从真实函数AST提取字面量配置键，避免维护第二份手工允许列表。"""
    tree = ast.parse(source)
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == "settings":
                key = node.args[0]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        if isinstance(node.func, ast.Name) and node.func.id in {"_interval", "_probability"}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                keys.add(node.args[1].value)
    return keys


def audit_configurations(paths):
    photo = accessed_settings(inspect.getsource(SharedDataset)) | {"enabled", "clean_probability"}
    modern = accessed_settings(inspect.getsource(compose_training_domain))
    legacy = accessed_settings(inspect.getsource(own_training_domain))
    records = []
    for path in paths:
        config = json.loads(path.read_text())
        recipe = config.get("training_recipe", "generalization")
        runtime_path = Path(config["output"]) / "config.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
        settings = config.get("range_augmentation", {})
        supported = legacy if recipe == "legacy" else modern
        unknown_range = sorted(set(settings) - supported)
        unknown_photo = sorted(set(config.get("photometric_augmentation", {})) - photo)
        records.append({
            "name": path.stem, "path": str(path), "sha256": sha256(path),
            "runtime_config": str(runtime_path), "runtime_config_present": runtime is not None,
            "runtime_config_matches": runtime == config if runtime is not None else None,
            "recipe": recipe, "range_settings": settings,
            "photometric_settings": config.get("photometric_augmentation", {}),
            "range_unknown_keys": unknown_range, "photometric_unknown_keys": unknown_photo,
            "new_photometric_active": recipe != "legacy" and bool(config.get("augmentation", True))
                and config.get("photometric_augmentation", {}).get("enabled", False),
            "inactive_by_recipe": ["photometric_augmentation"] if recipe == "legacy" else [],
            "inactive_reason": "legacy recipe uses old shared brightness; new photo block is not read"
                if recipe == "legacy" else None,
        })
        if unknown_range or unknown_photo:
            raise ValueError(f"{path.name} 有未被实现读取的增强配置键：{unknown_range}, {unknown_photo}")
        if runtime is not None and runtime != config:
            raise ValueError(f"{path.name} 与实际运行目录config.json不同")
    return {"records": records, "supported_new_photo_keys": sorted(photo),
            "supported_new_range_keys": sorted(modern), "supported_legacy_range_keys": sorted(legacy),
            "scope": "augmentation config blocks and training_recipe routing; not a complete model option audit"}


def reference_truth(batch):
    truth = batch["targets"]["depth"][:, 0].double()
    valid = batch["targets"]["depth_valid"][:, 0] & batch["inputs"]["image_mask"][:, 0]
    valid &= batch["inputs"]["view_mask"][:, 0, None, None] & torch.isfinite(truth) & (truth > 0)
    return [value[mask] for value, mask in zip(truth, valid)]


def quantization_metrics(depth, near, far, bins):
    log_near, span = math.log(near), math.log(far / near)
    position = (depth.log() - log_near) / span
    coordinate = position * (bins - 1)
    low_index = coordinate.floor().clamp(0, bins - 1)
    high_index = coordinate.ceil().clamp(0, bins - 1)
    low = (log_near + span * low_index / (bins - 1)).exp()
    high = (log_near + span * high_index / (bins - 1)).exp()
    relative = torch.minimum((low - depth).abs(), (high - depth).abs()) / depth
    log_error = torch.minimum((low.log() - depth.log()).abs(), (high.log() - depth.log()).abs())
    return {"mean_axis_position": float(position.mean()),
            "axis_position_p05": float(position.quantile(.05)),
            "axis_position_p95": float(position.quantile(.95)),
            "oracle_absrel_mean": float(relative.mean()),
            "oracle_absrel_p95": float(relative.quantile(.95)),
            "oracle_log_error_mean": float(log_error.mean()),
            "half_log_step_relative_bound": math.expm1(span / (2 * (bins - 1)))}


def aggregate(rows):
    keys = [key for key, value in rows[0].items() if isinstance(value, (float, int))]
    result = {key + "_mean": statistics.mean(row[key] for row in rows) for key in keys}
    result["mean_axis_position_std_across_frames_and_draws"] = statistics.pstdev(
        row["mean_axis_position"] for row in rows)
    result["full_gt_outside_pixels_total_across_draws"] = sum(row["outside_pixels"] for row in rows)
    result["per_frame_position_std_across_draws_mean"] = statistics.mean(
        statistics.pstdev(row["mean_axis_position"] for row in rows if row["sample_index"] == index)
        for index in sorted({row["sample_index"] for row in rows}))
    result["per_frame_log_span_fraction_std_across_draws_mean"] = statistics.mean(
        statistics.pstdev(row["gt_log_span_fraction"] for row in rows if row["sample_index"] == index)
        for index in sorted({row["sample_index"] for row in rows}))
    return result


def audit_domain(dataset, domain_index, legacy_config, modern_config, samples, draws, points, batch_sizes):
    domain = dataset.domains[domain_index]
    length = len(dataset.datasets[domain_index])
    indices = torch.linspace(0, length - 1, min(samples, length)).long().unique().tolist()
    items = [dataset[domain_index, index, 0] for index in indices]
    sample_records = []
    for index, item in zip(indices, items):
        truth = reference_truth(collate_mixed([item]))[0]
        if not len(truth):
            raise ValueError(f"{domain['name']} TRAIN index {index}无有效参考GT")
        sample_records.append({"index": index, "full_gt_pixels": len(truth),
                               "gt_min": float(truth.min()), "gt_max": float(truth.max()),
                               "oracle_gt_points": min(points, len(truth)),
                               "augmentation": item["metadata"]["v8_augmentation"]})
    summaries, raw = {}, []
    policies = {"old": (legacy_config, own_training_domain), "new": (modern_config, compose_training_domain)}
    for batch_size in batch_sizes:
        summaries[str(batch_size)] = {}
        for name, (config, transform) in policies.items():
            rows, modes, range_records = [], {}, []
            for start in range(0, len(items), batch_size):
                batch = collate_mixed(items[start:start + batch_size])
                full_truth = reference_truth(batch)
                subsampled = [value[torch.linspace(0, len(value) - 1, min(points, len(value))).long()]
                              for value in full_truth]
                for draw in range(draws):
                    changed = transform(batch, domain, config, draw, config["seed"])
                    near, far = float(changed["near"]), float(changed["far"])
                    mode = changed.get("range_augmentation_mode", "nominal" if
                        near == domain["near"] and far == domain["far"] else "gt_expansion")
                    modes[mode] = modes.get(mode, 0) + 1
                    range_records.append({"batch_begin": start, "draw": draw, "mode": mode,
                                          "near": near, "far": far,
                                          "unit_factor": changed["depth_scale"] / domain["depth_scale"]})
                    for offset, (truth, sampled) in enumerate(zip(full_truth, subsampled)):
                        row = quantization_metrics(sampled, near, far, int(config["model"]["depth_bins"]))
                        row.update(sample_index=indices[start + offset], draw=draw,
                                   gt_log_span_fraction=float((truth.max() / truth.min()).log()) / math.log(far / near),
                                   outside_pixels=int(((truth < near) | (truth > far)).sum()))
                        rows.append(row)
            summaries[str(batch_size)][name] = dict(summary=aggregate(rows), mode_counts=modes,
                                                    range_draws=range_records)
            raw.append({"batch_size": batch_size, "policy": name, "per_frame_draw_rows": rows})
        old, new = [summaries[str(batch_size)][p]["summary"] for p in ("old", "new")]
        summaries[str(batch_size)]["change_new_minus_old"] = {
            key: new[key] - old[key] for key in (
                "per_frame_position_std_across_draws_mean", "gt_log_span_fraction_mean",
                "oracle_absrel_mean_mean", "oracle_absrel_p95_mean")}
    return {"domain": domain["name"], "units": domain["geometry_units"], "train_count": length,
            "manifest": domain["manifests"]["train"], "manifest_sha256": sha256(domain["manifests"]["train"]),
            "samples": sample_records, "results_by_microbatch": summaries, "raw": raw}


def self_check():
    values = torch.logspace(0, 2, 96, dtype=torch.float64)
    result = quantization_metrics(values, 1., 100., 96)
    assert result["oracle_absrel_mean"] < 1e-14
    assert quantization_metrics(torch.tensor([.5, 200.]), 1., 100., 96)["oracle_absrel_mean"] == .75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).with_name("configs"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/active/rulermvs_v8_20261103/diagnostics/augmentation_audit.json")
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--oracle-points-per-sample", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.samples_per_domain <= 8:
        raise ValueError("本诊断每域只允许1–8个TRAIN组")
    offline_only()
    torch.set_num_threads(2)
    self_check()
    paths = sorted(args.config_dir.glob("*.json"))
    config_audit = audit_configurations(paths)
    legacy = json.loads((args.config_dir / "v6_control.json").read_text())
    modern = json.loads((args.config_dir / "v6_augmentation.json").read_text())
    if legacy["datasets"] != modern["datasets"]:
        raise ValueError("新旧配置训练清单或域协议不同")
    # 防止诊断未来误拿目标域配置；仅接受当前明确授权七域与TRAIN清单。
    expected = {"TartanAir_forest", "MatrixCity_street", "Hypersim_subset", "MVS_Synth",
                "VKITTI2", "BlendedMVS", "DynamicReplica_valid"}
    if {domain["name"] for domain in modern["datasets"]} != expected:
        raise ValueError("诊断只允许现有七域TRAIN")
    dataset = SharedDataset(modern, "train")
    watched = [Path(inspect.getfile(SharedDataset)), Path(inspect.getfile(compose_training_domain)),
               Path(inspect.getfile(own_training_domain)), *paths]
    before = {str(path): sha256(path) for path in watched}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "completed": False,
        "protocol": {"split": "TRAIN", "test_accessed": False, "val_accessed": False,
                     "gpu_used": False, "model_created": False, "network_disabled": True,
                     "sample_selection": "evenly spaced manifest indices; epoch0 reference rotation",
                     "max_unique_samples_per_domain": args.samples_per_domain,
                     "range_draws_per_batch": args.draws, "microbatch_sizes": [8, 4],
                     "range_seed_rank": 0, "full_gt_for_coverage": True,
                     "oracle_points_per_sample": args.oracle_points_per_sample,
                     "oracle_point_selection": "evenly spaced valid pixels in flattened reference GT",
                     "depth_bins": 96, "oracle_definition": "minimum abs(depth_grid-GT)/GT at 96 log-spaced physical depths",
                     "aggregation": "equal weight per selected frame and draw; not pixel pooled or independent scenes",
                     "interpretation": "TRAIN augmentation geometry diagnostic only; oracle is not attainable model accuracy",
                     "batch_effect": "B4 splits same eight samples; branch/padding seed fixed across batch sizes; GT extrema can differ"},
        "configuration_audit": config_audit, "source_sha256_before": before, "domains": [],
    }
    for index in range(len(dataset.domains)):
        row = audit_domain(dataset, index, legacy, modern, args.samples_per_domain, args.draws,
                           args.oracle_points_per_sample, [8, 4])
        result["domains"].append(row)
        print(json.dumps({"domain": row["domain"], "B8": row["results_by_microbatch"]["8"]["change_new_minus_old"]}), flush=True)
    after = {str(path): sha256(path) for path in watched}
    if after != before:
        raise RuntimeError("审计期间活跃增强源码或配置发生变化，不输出混合版本结论")
    result.update(completed=True, source_sha256_after=after, live_sources_unchanged=True)
    result["summary"] = {}
    for batch_size in ("8", "4"):
        rows = [domain["results_by_microbatch"][batch_size] for domain in result["domains"]]
        fields = ("per_frame_position_std_across_draws_mean", "gt_log_span_fraction_mean",
                  "oracle_absrel_mean_mean", "oracle_absrel_p95_mean")
        result["summary"][batch_size] = {
            "domain_equal_mean": {policy: {field: statistics.mean(row[policy]["summary"][field] for row in rows)
                                             for field in fields} for policy in ("old", "new")},
            "domains_with_greater_position_dispersion": sum(row["change_new_minus_old"][fields[0]] > 0 for row in rows),
            "new_full_gt_outside_pixels": sum(row["new"]["summary"]["full_gt_outside_pixels_total_across_draws"] for row in rows),
            "old_full_gt_outside_pixels": sum(row["old"]["summary"]["full_gt_outside_pixels_total_across_draws"] for row in rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
