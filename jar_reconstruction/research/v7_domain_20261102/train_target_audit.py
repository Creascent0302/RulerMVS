"""只读 TRAIN 审计：粗标签/连续候选、几何门与保持约束，禁止更新参数。

复用第二轮已曝光 TRAIN 的实际 batch 顺序与增强随机流；只抽取每域最早
若干曝光。基准只有冻结 V6，学生仅用于重算保持约束，不访问 TEST/VAL。
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.rulermvs_v6_20261101.objective import objective
from research.v7_domain_20261102.augmentation import augment_inputs
from research.v7_domain_20261102.model_round2 import build_model
from research.v7_domain_20261102.retention_round2 import retention_penalty
from tools.smoke_outdoor_datasets import offline_only


def truth_and_mask(batch, domain, scale):
    inputs, targets = batch["inputs"], batch["targets"]
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid = valid & inputs["view_mask"][:, :1, None, None]
    valid = valid & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid = valid & (truth * scale >= low) & (truth * scale <= high)
    return torch.where(valid, truth, 1.), valid


def candidate_maps(stage, truth, mask, scale):
    """log 距离决定标签；native 单位误差只用于诊断，不互换两者。"""
    lattice, actual = stage["sampling_depth"].float(), stage["candidate_depth"].float()
    legal = stage["valid"] & torch.isfinite(actual) & (actual > 0)
    sampling_error = (lattice.clamp_min(1e-6).log() - truth.log()).abs()
    actual_error = (actual.clamp_min(1e-6).log() - truth.log()).abs().masked_fill(~legal, torch.inf)
    grid_index = sampling_error.argmin(1, keepdim=True)
    actual_index = actual_error.argmin(1, keepdim=True)
    absolute = (actual - truth).abs() * scale
    grid_error = actual_error.gather(1, grid_index)
    best_error = actual_error.gather(1, actual_index)
    chosen_error = actual_error.gather(1, stage["selected_index"])
    grid_absolute = absolute.gather(1, grid_index)
    best_absolute = absolute.min(1, keepdim=True).values
    chosen_absolute = absolute.gather(1, stage["selected_index"])
    disagreement = grid_index != actual_index
    lower, upper = stage["lower"].gather(1, grid_index), stage["upper"].gather(1, grid_index)
    actual_lower, actual_upper = stage["lower"].gather(1, actual_index), stage["upper"].gather(1, actual_index)
    reachable = (truth >= lower) & (truth <= upper)
    actual_reachable = (truth >= actual_lower) & (truth <= actual_upper)
    values = {
        "label_disagreement": disagreement.float(),
        "label_distance_bins": (grid_index - actual_index).abs().float(),
        "label_jump_more_than_one": ((grid_index - actual_index).abs() > 1).float(),
        "nearest_grid_sampling_log_error": sampling_error.gather(1, grid_index),
        "nearest_grid_actual_log_error": grid_error,
        "actual_log_oracle_error": best_error,
        "grid_label_actual_log_regret": grid_error - best_error,
        "chosen_actual_log_error": chosen_error,
        "chosen_log_selection_regret": chosen_error - best_error,
        "nearest_grid_actual_native_error": grid_absolute,
        "actual_native_oracle_error": best_absolute,
        "grid_label_actual_native_regret": grid_absolute - best_absolute,
        "chosen_actual_native_error": chosen_absolute,
        "chosen_native_selection_regret": chosen_absolute - best_absolute,
        "native_oracle_of_sampling_grid": ((lattice - truth).abs() * scale).min(1, keepdim=True).values,
        "grid_label_unreachable": (~reachable).float(),
        "actual_best_label_unreachable": (~actual_reachable).float(),
        "true_outside_model_range": ((truth < stage["lower"].amin(1, keepdim=True)) |
                                     (truth > stage["upper"].amax(1, keepdim=True))).float(),
        "offset_fraction_abs": ((actual - lattice).abs() /
                                torch.where(actual >= lattice, stage["upper"] - lattice,
                                            lattice - stage["lower"]).clamp_min(1e-8)).mean(1, keepdim=True),
    }
    valid = mask & legal.any(1, keepdim=True)
    return values, valid, grid_index, actual_index


def quality_maps(source, candidates, ray_derivative):
    """核实实际运行类的 information 等于权重×投影导数平方，非GT可见性。"""
    null = source["null_mass"][:, 0].float()
    information = source["information"][:, 0].float()
    sensitivity = (source["weights"] * ray_derivative.square().sum(2)).sum(1).float()
    torch.testing.assert_close(information, sensitivity, atol=0, rtol=0)
    z_squared = candidates.float().square()
    current = null + (1 - null) * .25 / (.25 + information * z_squared)
    geometric = null + (1 - null) * .25 / (.25 + sensitivity * z_squared)
    return {"information_gate": current, "recomputed_gate": geometric,
            "gate_recompute_error": current - geometric, "null_mass": null,
            "gate_below_01": (current < .1).float(),
            "gate_above_05": (current > .5).float(),
            "gate_above_09": (current > .9).float(),
            "parallax_pixels": (sensitivity * z_squared).clamp_min(0).sqrt(),
            "information_logdepth": information * z_squared}


def add_maps(totals, maps, mask):
    count = int(mask.sum())
    if not count:
        return
    totals["count"] = totals.get("count", 0) + count
    for name, values in maps.items():
        values = values.expand_as(mask)[mask].double()
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"审计量非有限：{name}")
        sums = totals.setdefault(name, {"sum": 0., "sum_square": 0., "min": float("inf"), "max": -float("inf")})
        sums["sum"] += float(values.sum())
        sums["sum_square"] += float(values.square().sum())
        sums["min"] = min(sums["min"], float(values.min()))
        sums["max"] = max(sums["max"], float(values.max()))


def summarize(totals):
    if "count" in totals:
        count = totals["count"]
        return {"count": count, **{name: {"mean": item["sum"] / count,
                "rms": (item["sum_square"] / count) ** .5,
                "min": item["min"], "max": item["max"]}
                for name, item in totals.items() if name != "count"}}
    return {name: summarize(item) for name, item in totals.items()}


def retention_maps(prediction, baseline, truth, valid, scale):
    current = prediction["depth"].float().clamp_min(1e-6)
    previous = baseline["depth"].float().clamp_min(1e-6)
    current_log = (current.log() - truth.log()).abs()
    previous_log = (previous.log() - truth.log()).abs()
    native_difference = ((current - truth).abs() - (previous - truth).abs()) * scale
    difference = current_log - previous_log
    return {"current_log_error": current_log, "teacher_log_error": previous_log,
            "log_error_change": difference, "hinge_retention": F.relu(difference),
            "improvement_log": F.relu(-difference), "worse_fraction": (difference > 1e-6).float(),
            "worse_over_one_percent_fraction": (difference > .01).float(),
            "native_error_change": native_difference,
            "current_native_error": (current - truth).abs() * scale,
            "teacher_native_error": (previous - truth).abs() * scale}


def load_phase(settings, phase, device):
    base = Path(settings["pilot_root"])
    directory = base / "fixed_joint_keep" / phase
    pilot = json.loads((directory / "settings.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    saved = torch.load(pilot["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model, config = build_model(saved, "control")
    config = copy.deepcopy(config)
    data = json.loads(Path(pilot["data_config"]).read_text())
    for key in ("datasets", "image_size", "matching_image_size"):
        config[key] = data[key]
    config.pop("validation_datasets", None)
    for domain in config["datasets"]:
        domain["manifests"] = {"train": domain["manifests"]["train"]}
    config.update(seed=pilot["seed"], augmentation=True)
    students = {}
    for group, variant in (("fixed_joint_keep", "fixed"), ("coordinate_joint_keep", "coordinate")):
        filename = base / group / phase / "final.pt"
        record = torch.load(filename, map_location="cpu", weights_only=False, mmap=True)
        for name, value in saved["model"].items():
            if name.startswith(("foundation.", "matching.", "bridge.")):
                if not torch.equal(value, record["model"][name]):
                    raise RuntimeError("审计共享编码需要逐位一致的冻结编码器")
        student, _ = build_model(record, variant)
        students[group] = student.decoder.eval().requires_grad_(False).to(device)
        del student, record
    model.eval().requires_grad_(False).to(device)
    install_position_cache(model.foundation.model)
    return model, students, MixedDataset(config, "train"), plan, pilot


@torch.inference_mode()
def run(settings):
    offline_only()
    seed_all(20261102)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings.get("device", "cuda:7"))
    torch.cuda.set_device(device)
    output = Path(settings["output"])
    if not output.resolve().is_relative_to(Path("/data")) or output.exists():
        raise ValueError("审计输出必须是数据盘上未存在的新目录")
    output.mkdir(parents=True)
    write_json(output / "settings.json", settings)
    all_totals, provenance, checks = {}, {}, []
    for phase in settings.get("phases", ["public", "carla"]):
        model, students, dataset, plan, pilot = load_phase(settings, phase, device)
        captured = {}
        original = model.decoder._tokens

        def capture(self, prepared, observation, candidates, stride):
            result = original(prepared, observation, candidates, stride)
            if stride == 8:
                captured.update(raw=result[1], source=result[-1], derivative=observation["ray_derivative"])
            return result

        model.decoder._tokens = types.MethodType(capture, model.decoder)
        generator = torch.Generator().manual_seed(pilot["seed"] + 7153)
        limit = int(settings.get("public_count", 8) if phase == "public" else settings.get("carla_count", 16))
        counts = {domain["name"]: 0 for domain in dataset.domains}
        selected = []
        with (output / (phase + ".jsonl")).open("w") as log:
            for step, keys in enumerate(plan["schedule"]["batches"], 1):
                batch = collate_mixed([dataset[tuple(key)] for key in keys])
                domain = dataset.domains[int(batch["domain_index"])]
                name = domain["name"]
                if counts[name] >= limit:
                    # 只推进与真实源数有关的增强随机流；该批不做模型计算。
                    augment_inputs(batch["inputs"], generator=generator)
                    continue
                if len(keys) + counts[name] > limit:
                    raise ValueError("审计数量须为实际训练 batch 大小的整数倍")
                batch = move(batch, device)
                # 与实际训练保持同设备、同次序：先GPU双线性缩放，再增强。
                inputs = equal_pixel_inputs(batch["inputs"])
                augmented, aug_stats = augment_inputs(inputs, generator=generator)
                counts[name] += len(keys)
                selected.append({"step": step, "domain": name, "keys": keys,
                                 "grayscale": aug_stats["grayscale"].tolist(),
                                 "views": aug_stats["retained_views"].tolist()})
                targets = batch["targets"]
                scale = float(domain["depth_scale"])
                for mode, source_inputs in (("original_train_augmentation", inputs),
                                             ("v7_train_augmentation", augmented)):
                    source_inputs = move(source_inputs, device)
                    current_batch = {"inputs": source_inputs, "targets": targets}
                    truth, valid = truth_and_mask(current_batch, domain, scale)
                    model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                    work = model.encode(source_inputs, domain, reference=0)
                    baseline = model.decoder(work)
                    baseline["unit_scale"] = scale
                    baseline_loss = objective(baseline, source_inputs, targets, domain, model.config.get("loss", {}))
                    coarse = baseline["coarse"]
                    coarse_truth, coarse_valid = truth[..., ::8, ::8], valid[..., ::8, ::8]
                    maps, coarse_valid, grid_index, actual_index = candidate_maps(coarse, coarse_truth, coarse_valid, scale)
                    gates = quality_maps(captured["source"], coarse["sampling_depth"], captured["derivative"])
                    grid_gate = {key: value.gather(1, grid_index) for key, value in gates.items()}
                    source_count = coarse["support"].gather(1, grid_index)
                    masks = {"all": coarse_valid, "no_projectable_source": coarse_valid & (source_count == 0),
                             "weak_projectable": coarse_valid & (source_count > 0) & (grid_gate["parallax_pixels"] < .5),
                             "strong_projectable": coarse_valid & (source_count > 0) & (grid_gate["parallax_pixels"] >= .5),
                             "label_disagree": coarse_valid & (maps["label_disagreement"] > 0)}
                    destination = all_totals.setdefault(phase, {}).setdefault(name, {}).setdefault(mode, {})
                    row = {"step": step, "domain": name, "mode": mode, "coarse": {}, "retention": {}}
                    for region, mask in masks.items():
                        subtotal = destination.setdefault("coarse", {}).setdefault(region, {})
                        add_maps(subtotal, {**maps, **grid_gate}, mask)
                        local = {}
                        add_maps(local, {**maps, **grid_gate}, mask)
                        row["coarse"][region] = summarize(local)
                    all_candidates = coarse_valid.expand_as(coarse["valid"]) & coarse["valid"]
                    add_maps(destination.setdefault("gate_all_candidates", {}), gates, all_candidates)
                    for student_name, decoder in students.items():
                        decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                        prior = decoder.context_prior
                        # 不改任何实际预测，单独衡量同一 V6 raw/source 下的门衰减。
                        raw = captured["raw"]
                        zeros = torch.zeros_like(captured["source"]["information"])
                        ungated = prior(raw, zeros, captured["source"]["null_mass"], coarse["sampling_depth"])
                        old = ungated * gates["information_gate"]
                        norm_maps = {"ungated_logits": ungated, "original_gate_logits": old,
                                     "gate_attenuated_logits": old - ungated,
                                     "ungated_depth_centered": ungated - ungated.mean(1, keepdim=True),
                                     "gated_depth_centered": old - old.mean(1, keepdim=True),
                                     "baseline_depth_centered_scores": coarse["scores"] - coarse["scores"].mean(1, keepdim=True)}
                        add_maps(destination.setdefault("prior_norms_common_v6", {}).setdefault(student_name, {}),
                                 norm_maps, all_candidates)
                        prediction = decoder(work)
                        prediction["unit_scale"] = scale
                        retention = retention_maps(prediction, baseline, truth, valid, scale)
                        measured = retention_penalty(prediction, baseline, source_inputs, targets, domain, margin=0.)
                        expected = retention["hinge_retention"][valid].mean()
                        torch.testing.assert_close(measured, expected, atol=1e-7, rtol=1e-6)
                        losses = objective(prediction, source_inputs, targets, domain, model.config.get("loss", {}))
                        loss_maps = {"main_loss": losses["total"], "teacher_main_loss": baseline_loss["total"],
                                     "main_loss_change": losses["total"] - baseline_loss["total"],
                                     "retention": measured, "retention_main_ratio": measured / losses["total"].clamp_min(1e-6),
                                     "total_with_retention_change": losses["total"] + measured - baseline_loss["total"],
                                     "coarse_ce": losses["coarse_ce"], "teacher_coarse_ce": baseline_loss["coarse_ce"],
                                     "group_ce": losses["group_ce"], "teacher_group_ce": baseline_loss["group_ce"]}
                        add_maps(destination.setdefault("loss_per_batch", {}).setdefault(student_name, {}),
                                 loss_maps, torch.ones((), dtype=torch.bool, device=device))
                        subtotal = destination.setdefault("retention", {}).setdefault(student_name, {})
                        add_maps(subtotal, retention, valid)
                        local = {}
                        add_maps(local, retention, valid)
                        row["retention"][student_name] = summarize(local)
                        row["retention"][student_name]["loss_components"] = {key: float(value) for key, value in loss_maps.items()}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    checks.append({"phase": phase, "step": step, "mode": mode, "retention_matches_actual_function": True})
                print(json.dumps({"phase": phase, "counts": counts}), flush=True)
                if all(value >= limit for value in counts.values()):
                    break
        provenance[phase] = {"selected_exposures": selected, "counts": counts,
                             "base_checkpoint": pilot["checkpoint"], "base_sha256": digest(pilot["checkpoint"]),
                             "schedule_sha256": plan["schedule"]["sha256"],
                             "actual_source_class": type(model.decoder.source).__module__ + "." + type(model.decoder.source).__name__,
                             "units": {d["name"]: d["geometry_units"] for d in dataset.domains}}
        if not all(value >= limit for value in counts.values()):
            raise RuntimeError("未达到审计样本数")
        del model, students, dataset, captured, baseline, prediction, work
        torch.cuda.empty_cache()
    write_json(output / "completed.json", {"completed": True, "split": "train", "test_accessed": False,
        "no_updates": True, "source_sha256": digest(__file__), "settings": settings,
        "provenance": provenance, "checks": checks, "metrics": summarize(all_totals), "raw": all_totals,
        "limits": ["每域少量已曝光TRAIN，不代表完整TRAIN/VAL/TEST",
                   "original_train_augmentation仍含既有RGB gain与随机缺源，不是原始无增强图",
                   "学生仅用于保持损失复算；主候选诊断仅使用冻结V6",
                   "gate范数使用同一V6取证，不等同于改变gate后的整网精度",
                   "本审计复现实际batch大小/顺序/增强RNG，不回放训练过程中每步权重"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
