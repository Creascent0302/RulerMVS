"""七域小规模结构筛选：固定 VAL 分区、完整 RGB、逐域误差与独立交替计时。"""

import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.inference import prepare_inference
from jar.pretraining.model import PretrainingModel, load_pretraining_state
from jar.runtime import move
from tools.probe_public_stages import diagnose, sum_records, summarize


def rgb_paths(record):
    return {camera["rgb"] for camera in record.get("cameras", []) if camera.get("rgb")}


def balanced_sample(groups, count, seed):
    """先均衡序列，再在序列内均匀分段抽样，不集中抽取相邻的前几帧。"""
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    quota = {name: 0 for name in names}
    for _ in range(count):
        eligible = [name for name in names if quota[name] < len(groups[name])]
        if not eligible:
            raise ValueError(f"候选样本不足，无法抽取 {count} 组")
        name = min(eligible, key=lambda item: quota[item])
        quota[name] += 1
    selected = []
    for name in names:
        indices = groups[name]
        for part in range(quota[name]):
            start = part * len(indices) // quota[name]
            end = (part + 1) * len(indices) // quota[name]
            selected.append(indices[rng.randrange(start, end)])
    return sorted(selected)


def select_partition(records, development, confirmation, seed):
    """仅根据清单元数据分区。时间留块并去除跨分区共享的输入图像。"""
    groups = {}
    for index, record in enumerate(records):
        sequence = str(record.get("sequence", record.get("source_sequence", "single_sequence")))
        groups.setdefault(sequence, []).append(index)
    first, second = {}, {}
    rng = random.Random(seed)
    for sequence, indices in sorted(groups.items()):
        middle = len(indices) // 2
        gap = max(1, len(indices) // 20)
        left, right = indices[:max(0, middle - gap)], indices[middle + gap:]
        if rng.randrange(2):
            left, right = right, left
        first[sequence], second[sequence] = left, right
    development_indices = balanced_sample(first, development, seed)
    seen = set().union(*(rgb_paths(records[index]) for index in development_indices))
    second = {name: [index for index in indices if not rgb_paths(records[index]) & seen]
              for name, indices in second.items()}
    confirmation_indices = balanced_sample(second, confirmation, seed + 1)
    confirmation_images = set().union(*(rgb_paths(records[index]) for index in confirmation_indices))
    return {
        "development": development_indices,
        "confirmation": confirmation_indices,
        "sequence_count": len(groups),
        "shared_indices": sorted(set(development_indices) & set(confirmation_indices)),
        "shared_rgb_paths": sorted(seen & confirmation_images),
    }


def selection(config, training):
    result = {}
    for index, domain in enumerate(training["datasets"]):
        manifest = Path(domain["manifests"]["val"])
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        selected = select_partition(records, config["development_per_domain"],
                                    config["confirmation_per_domain"], config["seed"] + index)
        selected.update(manifest=str(manifest), manifest_sha256=digest(manifest),
                        manifest_frames=len(records))
        result[domain["name"]] = selected
    return result


def engineering_gate(domains, baseline, variants, tolerance):
    """阈值是预先固定的工程筛选标准，不能解释成统计显著性或保精度证明。"""
    decisions = {}
    for name in variants:
        if name == baseline:
            continue
        relative, failures = {}, []
        for domain, values in domains.items():
            base = values[baseline]["full"]
            candidate = values[name]["full"]
            deltas = {key: candidate[key] / max(base[key], 1e-12) - 1
                      for key in ("mae", "rmse", "abs_rel")}
            relative[domain] = deltas
            for key in ("mae", "rmse", "abs_rel"):
                if deltas[key] > tolerance["per_domain_relative_degradation"]:
                    failures.append(f"{domain}/{key}")
            for group in ("quartile_4", "edge_5px"):
                base_group = values[baseline].get("groups", {}).get(group)
                candidate_group = values[name].get("groups", {}).get(group)
                if not base_group or not candidate_group or not base_group["count"]:
                    continue
                for key in ("mae", "rmse", "abs_rel"):
                    change = candidate_group[key] / max(base_group[key], 1e-12) - 1
                    if change > tolerance["per_domain_relative_degradation"]:
                        failures.append(f"{domain}/{group}/{key}")
        macro = {key: statistics.mean(row[key] for row in relative.values())
                 for key in ("mae", "rmse", "abs_rel")}
        for key in ("mae", "rmse", "abs_rel"):
            if macro[key] > tolerance["macro_relative_degradation"]:
                failures.append("macro/" + key)
        decisions[name] = {
            "passes_engineering_screen": not failures,
            "failures": failures,
            "macro_relative_change": macro,
            "per_domain_relative_change": relative,
            "interpretation": "仅当前小样本筛选；通过不证明全量或重训后精度保持",
        }
    return decisions


def verify_common_protocol(training, data):
    """换完整权重时保持像素信息、单位和候选物理范围相同。"""
    for key in ("image_size", "matching_image_size"):
        if training[key] != data[key]:
            raise ValueError(f"完整权重的 {key} 与共同数据协议不同")
    domains = {domain["name"]: domain for domain in training["datasets"]}
    for domain in data["datasets"]:
        stored = domains.get(domain["name"])
        if stored is None:
            raise ValueError(f"完整权重缺少共同域 {domain['name']} 的训练范围记录")
        for key in ("near", "far", "depth_scale", "geometry_units", "num_views", "reference_index"):
            if stored[key] != domain[key]:
                raise ValueError(f"{domain['name']}/{key} 与共同数据协议不同")


def build_models(config, checkpoint, device, data=None):
    models = {}
    for name, variant in config["variants"].items():
        path = variant.get("checkpoint", config["checkpoint"])
        saved = checkpoint if path == config["checkpoint"] else torch.load(
            path, map_location="cpu", weights_only=False, mmap=True)
        verify_common_protocol(saved["config"], data or checkpoint["config"])
        training = copy.deepcopy(saved["config"])
        training["model"].update(variant.get("model_overrides", {}))
        model = PretrainingModel(training, initialize=False)
        load_pretraining_state(model, saved["model"])
        if variant.get("decoder_checkpoint"):
            trained = torch.load(variant["decoder_checkpoint"], map_location="cpu", weights_only=False)
            if trained["settings"] != training["model"]:
                raise ValueError(f"{name}: 短训解码器 settings 与当前结构配置不同")
            if digest(trained["plan"]["checkpoint"]) != digest(path):
                raise ValueError(f"{name}: 短训解码器必须使用同一冻结编码器 checkpoint")
            model.decoder.load_state_dict(trained["model"], strict=True)
        model.requires_grad_(False).to(device).eval()
        install_position_cache(model.foundation.model)
        prepare_inference(model, **variant.get("inference", {}))
        models[name] = model
    return models


def output_difference(prediction, baseline, truth, valid):
    difference = (prediction - baseline).abs()
    error_change = (prediction - truth).abs() - (baseline - truth).abs()
    relative_threshold = truth.abs() * .01
    return {
        "pixels": difference.numel(),
        "changed_pixels": int((difference != 0).sum()),
        "absolute_sum": float(difference.double().sum()),
        "absolute_max": float(difference.max()),
        "gt_pixels": int(valid.sum()),
        "gt_absolute_sum": float(difference[valid].double().sum()),
        "gt_absolute_max": float(difference[valid].max()),
        "gt_error_worse_over_1pct_depth": int(((error_change > relative_threshold) & valid).sum()),
        "gt_error_better_over_1pct_depth": int(((error_change < -relative_threshold) & valid).sum()),
    }


def merge_differences(records):
    result = {key: max(row[key] for row in records) if key.endswith("_max")
              else sum(row[key] for row in records) for key in records[0]}
    result["mean_absolute_change"] = result["absolute_sum"] / max(result["pixels"], 1)
    result["gt_mean_absolute_change"] = result["gt_absolute_sum"] / max(result["gt_pixels"], 1)
    return result


@torch.inference_mode()
def screen(config, training, choices, models, device, output):
    dataset = MixedDataset(training, "val")
    baseline = config["baseline"]
    names = [baseline] + [name for name in models if name != baseline]
    results, changes = {}, {}
    for domain_index, domain in enumerate(dataset.domains):
        name = domain["name"]
        indices = choices[name][config["partition"]]
        rows = {method: [] for method in names}
        delta = {method: [] for method in names if method != baseline}
        for index in indices:
            batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
            if domain.get("eval_range"):
                near, far = domain["eval_range"]
                depth = batch["targets"]["depth"]
                batch["targets"]["depth_valid"] &= (depth >= near) & (depth <= far)
            truth = batch["targets"]["depth"][:, :1]
            valid = batch["targets"]["depth_valid"][:, :1] & batch["inputs"]["image_mask"][:, :1]
            valid &= batch["inputs"]["view_mask"][:, :1, None, None]
            valid &= torch.isfinite(truth) & (truth > 0)
            if not bool(valid.any()):
                raise ValueError(f"{name}/{index} 没有可评估 GT，不能静默删样本")
            for method in names:
                captured = []
                handle = models[method].decoder.upsample_mask.register_forward_hook(
                    lambda module, args, result: captured.append(result))
                try:
                    prediction = models[method](batch["inputs"], domain, reference=0)
                finally:
                    handle.remove()
                if not bool(torch.isfinite(prediction["depth"]).all()):
                    raise ValueError(f"{name}/{index}/{method} 输出非有限，不能忽略困难像素")
                statistics_ = diagnose(prediction, captured.pop(), batch, domain)
                rows[method].append({"index": index, "statistics": statistics_})
                depth = prediction["depth"] * prediction["unit_scale"]
                if method == baseline:
                    base_depth = depth.clone()
                else:
                    delta[method].append(output_difference(depth, base_depth, truth, valid))
                del prediction, depth
        results[name] = {method: summarize(sum_records([row["statistics"] for row in values]))
                         for method, values in rows.items()}
        changes[name] = {method: merge_differences(values) for method, values in delta.items()}
        write_json(output / f"{name}.json", {
            "indices": indices, "geometry_units": domain["geometry_units"],
            "metrics": results[name], "output_changes": changes[name], "rows": rows,
        })
        print(json.dumps({"dataset": name, "full": {key: value["full"]
                         for key, value in results[name].items()}}, ensure_ascii=False), flush=True)
    return {"domains": results, "output_changes": changes,
            "engineering_gate": engineering_gate(results, baseline, names, config["tolerance"])}


@torch.inference_mode()
def timing(config, training, choices, models, device):
    # 仅推理，不运行损失、诊断 hook，也不按性能选择输入。
    dataset = MixedDataset(training, "val")
    report = {}
    names = list(models)
    for domain_index, domain in enumerate(dataset.domains):
        if domain["name"] not in config["timing"]["datasets"]:
            continue
        indices = choices[domain["name"]]["development"]
        selected = [indices[0], indices[len(indices) // 2], indices[-1]]
        durations = {name: [] for name in names}
        view_counts = []
        for index in selected:
            inputs = move(collate_mixed([dataset[(domain_index, index, -1)]])["inputs"], device)
            view_counts.append(int(inputs["view_mask"].sum()))
            for name in names:
                for _ in range(config["timing"]["warmup"]):
                    prediction = models[name](inputs, domain, reference=0)
                    del prediction
            torch.cuda.synchronize(device)
            for repeat in range(config["timing"]["repetitions"]):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in order:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    prediction = models[name](inputs, domain, reference=0)
                    torch.cuda.synchronize(device)
                    durations[name].append((time.perf_counter() - started) * 1000)
                    del prediction
        report[domain["name"]] = {
            name: {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values),
                   "p90_ms": sorted(values)[min(len(values) - 1, int(len(values) * .9))],
                   "samples_ms": values, "frames": selected, "valid_views_per_frame": view_counts}
            for name, values in durations.items()
        }
        print(json.dumps({"timing": domain["name"], "median_ms": {
            name: values["median_ms"] for name, values in report[domain["name"]].items()}},
            ensure_ascii=False), flush=True)
    return {"latency": report, "scope": "完整 RGB 至单参考深度，不含读盘/上传/外部融合；同卡交替计时"}


def run(config):
    if config.get("split", "val") != "val":
        raise ValueError("结构筛选仅允许 VAL；禁止用 TEST 选择结构")
    if config["partition"] not in ("development", "confirmation"):
        raise ValueError("partition 须为 development 或 confirmation")
    if config["mode"] not in ("select", "screen", "timing"):
        raise ValueError("mode 须为 select、screen 或 timing")
    if config["baseline"] not in config["variants"]:
        raise ValueError("baseline 必须对应一个 variant")
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出非空，请为新实验指定独立 output")
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    training = copy.deepcopy(checkpoint["config"])
    if config.get("data_config"):
        data = json.loads(Path(config["data_config"]).read_text())
        for key in ("image_size", "matching_image_size", "datasets"):
            if key in data:
                training[key] = data[key]
    training["augmentation"] = False
    choices = selection(config, training)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "selection.json", choices)
    audit = {"checkpoint_sha256": digest(config["checkpoint"]), "test_accessed": False,
             "variant_checkpoint_sha256": {
                 name: digest(variant.get("checkpoint", config["checkpoint"]))
                 for name, variant in config["variants"].items()},
             "decoder_sha256": {name: digest(variant["decoder_checkpoint"])
                                for name, variant in config["variants"].items()
                                if variant.get("decoder_checkpoint")},
             "sources": {name: digest(ROOT / name) for name in (
                 "tools/screen_public_architecture.py", "jar/pretraining/model.py",
                 "jar/pretraining/projective_ruler.py", "jar/pretraining/inference.py",
                 "jar/models/ruler_direct.py", "tools/probe_public_stages.py")},
             "notes": [
                 "development 与 confirmation 索引及实际 RGB 路径不重叠；两者均来自已有 VAL。",
                 "该 VAL 曾用于旧训练的权重选择，不是新的盲测集。",
                 "远距采用各帧有效 GT 最远四分位，仅用于诊断，不决定网络输入或候选范围。",
                 "候选 oracle 是事后诊断，不可部署；粗细网格不同，不直接相减作为因果收益。",
                 "工程容差不等于统计显著性；小规模通过不能保证重训或全数据性能。",
                 "几何、编码器或新参数的学习改动还须同预算旧结构续训对照；本工具仅筛选固定权重。",
             ]}
    write_json(output / "audit.json", audit)
    if config["mode"] == "select":
        return
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models = build_models(config, checkpoint, device, training)
    result = (screen(config, training, choices, models, device, output)
              if config["mode"] == "screen" else timing(config, training, choices, models, device))
    write_json(output / "summary.json", {"config": config, "audit": audit, **result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
