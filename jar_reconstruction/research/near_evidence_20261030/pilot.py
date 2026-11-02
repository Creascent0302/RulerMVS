"""固定TRAIN小预算与双分区VAL试验；只保存初始/末步，不读取TEST或挑best。"""

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import PretrainingModel
from research.near_evidence_20261030.variants import install_variants
from jar.pretraining.runner import atomic_save
from jar.runtime import move, seed_all
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.v2_training_20261028.objectives import fine_actual_regret
from research.near_evidence_20261030.visibility import capture_attention, auxiliary_loss
from tools.screen_public_architecture import balanced_sample
from tools.smoke_outdoor_datasets import offline_only


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parameter_hash(state, prefixes=None):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if prefixes is not None and not name.startswith(tuple(prefixes)):
            continue
        result.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def prepare(config, settings):
    selections, manifests, batches = {}, {}, []
    budget = int(settings.get("train_samples_per_domain", 128))
    batch_size = int(settings.get("batch_size", 2))
    epochs = int(settings.get("epochs", 1))
    if min(budget, batch_size, epochs) < 1:
        raise ValueError("TRAIN样本数、batch_size、epochs必须为正")
    for domain_index, domain in enumerate(config["datasets"]):
        filename = path(domain["manifests"]["train"])
        lines = [line for line in filename.read_text().splitlines() if line.strip()]
        groups = {}
        for index, line in enumerate(lines):
            row = json.loads(line)
            groups.setdefault(str(row.get("sequence", row.get("source_sequence", "single"))), []).append(index)
        selected = balanced_sample(groups, min(budget, len(lines)), settings["seed"] + domain_index)
        selections[domain["name"]] = {"indices": selected, "sequence_count": len(groups),
                                       "source_count": len(lines), "manifest_sha256": digest(filename)}
        manifests[domain["name"]] = "".join(lines[index] + "\n" for index in selected)
    for epoch in range(epochs):
        rng = random.Random(settings["seed"] + epoch)
        current = []
        for domain_index, domain in enumerate(config["datasets"]):
            indices = selections[domain["name"]]["indices"].copy()
            rng.shuffle(indices)
            current.extend([(domain_index, index, epoch) for index in indices[start:start + batch_size]]
                           for start in range(0, len(indices), batch_size))
        rng.shuffle(current)
        batches.extend(current)
    full_steps = len(batches)
    if "smoke_steps" in settings:
        if not settings.get("smoke", False) or not 1 <= int(settings["smoke_steps"]) <= full_steps:
            raise ValueError("只有显式smoke允许截断更新数")
        offset = int(settings.get("smoke_step_offset", 0))
        if not 0 <= offset < full_steps or offset + int(settings["smoke_steps"]) > full_steps:
            raise ValueError("smoke batch区间越界")
        batches = batches[offset:offset + int(settings["smoke_steps"])]
    validation = copy.deepcopy(config)
    validation["datasets"] = copy.deepcopy(config.get("validation_datasets", config["datasets"]))
    val_plan = {}
    for domain in validation["datasets"]:
        filename = path(domain["manifests"]["val"])
        length = sum(bool(line.strip()) for line in filename.read_text().splitlines())
        val_count = int(settings.get("smoke_validation_samples", 2)) if settings.get("smoke", False) else 16
        indices = torch.linspace(0, length - 1, min(val_count, length)).long().tolist()
        val_plan[domain["name"]] = {"indices": indices, "development": indices[::2],
                                   "confirmation": indices[1::2], "manifest_sha256": digest(filename)}
    schedule = {"batches": batches, "sha256": payload_hash(batches),
                "steps": len(batches), "full_steps": full_steps,
                "exposures": sum(len(batch) for batch in batches)}
    return selections, manifests, validation, val_plan, schedule


@torch.no_grad()
def evaluate(model, dataset, val_plan, device):
    model.eval()
    result = {"development": {}, "confirmation": {}}
    for domain_index, domain in enumerate(dataset.domains):
        local_domain = dict(domain, reference_index=0)  # MixedDataset已把指定参考重排到槽0。
        selected = val_plan[domain["name"]]
        records = {split: {"frames": 0, "depth": {}, "stages_stride8": {}} for split in result}
        keys = [(domain_index, index, -1) for index in selected["indices"]]
        loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed, num_workers=0)
        for index, batch in zip(selected["indices"], loader):
            batch = move(batch, device)
            inputs = batch["inputs"]
            native = model(inputs, domain, reference=0)
            scale = float(native["unit_scale"])
            estimate = native["depth"] * scale
            truth, valid = ground_truth(batch, local_domain)
            if not bool((torch.isfinite(estimate[valid]) & (estimate[valid] > 0)).all()):
                raise RuntimeError(f"{domain['name']}存在非有限或非正预测")
            masks = regions(truth, valid, domain)
            partition = "development" if index in selected["development"] else "confirmation"
            record = records[partition]
            record["frames"] += 1
            for region, mask in masks.items():
                label = "edge" if region == "gradient_band" else region
                sums = depth_sums(estimate, native["valid"], truth, mask)
                add_statistics(record["depth"].setdefault(label, {}), sums)
            stages = {"coarse": native["coarse"]["depth"] * scale,
                      "fine": native["fine"]["depth"][..., ::2, ::2] * scale,
                      "final": estimate[..., ::8, ::8]}
            for stage, value in stages.items():
                target = record["stages_stride8"].setdefault(stage, {})
                for region, mask in masks.items():
                    label = "edge" if region == "gradient_band" else region
                    sums = depth_sums(value, torch.isfinite(value) & (value > 0),
                                      truth[..., ::8, ::8], mask[..., ::8, ::8])
                    add_statistics(target.setdefault(label, {}), sums)
        for partition, values in records.items():
            result[partition][domain["name"]] = {
                "frames": values["frames"], "units": domain["geometry_units"],
                "depth": {name: depth_metrics(sums) for name, sums in values["depth"].items()},
                "stages_stride8": {stage: {name: depth_metrics(sums) for name, sums in regions_.items()}
                                   for stage, regions_ in values["stages_stride8"].items()},
            }
    return result


def train_mode(model):
    # root.training决定是否返回pyramid；冻结编码器与FPN始终eval，不更新统计/随机状态。
    model.eval()
    model.training = True
    model.decoder.train()


def run(settings):
    offline_only()
    settings = dict(seed=20261030, **settings) if "seed" not in settings else settings
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有试验：{output}")
    checkpoint = path(settings["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = json.loads(path(settings["data_config"]).read_text())
    config = copy.deepcopy(saved["config"])
    for key in ("datasets", "validation_datasets", "image_size", "matching_image_size"):
        if key in source:
            config[key] = copy.deepcopy(source[key])
    if "validation_datasets" not in source:
        config.pop("validation_datasets", None)
    config["model"] = merge(config["model"], settings.get("model_overrides", {}))
    config["loss"] = merge(config.get("loss", {}), settings.get("loss_overrides", {}))
    config.update(seed=settings["seed"], augmentation=settings.get("augmentation", True))
    # 只传TRAIN/VAL路径给数据接口，防止研究入口误访问TEST。
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {key: value for key, value in domain["manifests"].items() if key in ("train", "val")}
    selections, manifests, val_config, val_plan, schedule = prepare(config, settings)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "settings.json", settings)
    write_json(output / "plan.json", {"training": selections, "validation": val_plan, "schedule": schedule,
        "test_accessed": False, "checkpoint_sha256": digest(checkpoint), "smoke": settings.get("smoke", False),
        "source_sha256": {name: digest(ROOT / name) for name in (
            "research/near_evidence_20261030/pilot.py", "jar/pretraining/model.py", "jar/pretraining/data.py",
            "jar/pretraining/losses.py", "research/near_evidence_20261030/ray_evidence.py",
            "research/near_evidence_20261030/variants.py",
            "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py",
            "research/near_evidence_20261030/visibility.py")},
        "input_intervention": {"probability": config["model"].get("source_intervention_probability", 0.),
                               "seed": config["model"].get("source_intervention_seed", settings["seed"]),
                               "note": "只干预组额外改变训练匹配输入；不宣称增强完全相同。独立generator不消耗全局/dataaugmentation RNG。"}})
    for name, content in manifests.items():
        folder = output / "selected_train"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".jsonl")).write_text(content)
    if settings.get("prepare_only", False):
        return

    seed_all(settings["seed"])
    torch.set_num_threads(settings.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = PretrainingModel(config, initialize=False)
    model.load_state_dict(saved["model"], strict=True)  # 本研究仅允许不增加参数的架构设置。
    install_variants(model)
    initial_hash = parameter_hash(saved["model"])
    encoder_hash = parameter_hash(saved["model"], ("foundation.", "matching."))
    del saved
    model.requires_grad_(False).eval().to(device)
    prefixes = tuple(settings.get("trainable_prefixes", ["decoder."]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    install_position_cache(model.foundation.model)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("没有匹配到可训练参数")
    decay = settings.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in trainable if p.ndim > 1 and not name.endswith("bias")], "weight_decay": decay},
        {"params": [p for name, p in trainable if p.ndim <= 1 or name.endswith("bias")], "weight_decay": 0.},
    ], lr=settings.get("lr", 1e-5))
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(val_config, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(settings["seed"]))
    started = time.monotonic()
    write_json(output / "initial.json", evaluate(model, val_data, val_plan, device))
    training_start = time.monotonic()
    train_mode(model)
    gradient_max = 0.
    visibility_weight = float(settings.get("visibility_weight", 0.))
    visibility_totals = {"positive_pairs": 0, "occluded_pairs": 0, "known_pairs": 0,
                         "supervised_rays": 0, "skipped_steps": 0}
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            with capture_attention(model) if visibility_weight else nullcontext() as captured:
                prediction = model(batch["inputs"], domain, reference=0)
            terms = depth_objective(prediction, batch["inputs"], batch["targets"], domain, config["loss"],
                                    settings.get("correspondence_weight", 0.))
            if settings.get("actual_fine_risk", True):
                risk = fine_actual_regret(prediction, batch["inputs"], batch["targets"], domain)
                terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
            visibility_log = None
            if visibility_weight:
                evidence = auxiliary_loss(prediction, batch["inputs"], batch["targets"], domain, captured)
                terms["total"] = terms["total"] + visibility_weight * evidence["loss"]

                def logged(value):
                    if isinstance(value, dict):
                        return {key: logged(item) for key, item in value.items()}
                    return float(value.detach()) if torch.is_tensor(value) else value

                visibility_log = logged(evidence)
                for key in ("positive_pairs", "occluded_pairs", "known_pairs", "supervised_rays"):
                    visibility_totals[key] += evidence[key]
                visibility_totals["skipped_steps"] += int(evidence["skipped"])
                if settings.get("smoke_visibility_gradient", False):
                    if not settings.get("smoke", False) or not evidence["occluded_pairs"]:
                        raise RuntimeError("visibility CUDA梯度smoke需要明确负例")
                    gradients = torch.autograd.grad(evidence["loss"], tuple(model.decoder.source.parameters()),
                                                    retain_graph=True, allow_unused=True)
                    norms = [gradient.norm() for gradient in gradients if gradient is not None]
                    auxiliary_norm = torch.stack(norms).norm()
                    if not bool(torch.isfinite(auxiliary_norm)) or float(auxiliary_norm) <= 0:
                        raise RuntimeError("visibility辅助损失梯度非有限或全部为0")
                    visibility_log["auxiliary_source_gradient_norm"] = float(auxiliary_norm)
            if not bool(torch.isfinite(terms["total"])) or not int(terms["supervised_pixels"]):
                raise RuntimeError(f"step {step}: loss非有限或没有监督")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], settings.get("grad_clip", 1.), error_if_nonfinite=True)
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            row = {"step": step, "dataset": domain["name"], "loss": float(terms["total"].detach()), "gradient_norm": float(norm)}
            if visibility_log is not None:
                row["visibility"] = visibility_log
            log.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0 or step == schedule["steps"]:
                log.flush()
                print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - training_start
    write_json(output / "final.json", evaluate(model, val_data, val_plan, device))
    final_hash = parameter_hash(model.state_dict())
    final_encoder_hash = parameter_hash(model.state_dict(), ("foundation.", "matching."))
    encoders_frozen = not any(name.startswith(("foundation.", "matching.")) for name, _ in trainable)
    if encoders_frozen and encoder_hash != final_encoder_hash:
        raise RuntimeError("冻结编码器/FPN发生了变化")
    atomic_save({"architecture": model.architecture, "model": model.state_dict(), "config": config,
                 "selection": "fixed_last_step_no_test_or_best_selection", "pilot_settings": settings},
                output / "final.pt", settings.get("minimum_free_gib", 200))
    write_json(output / "completed.json", {"completed": True, "steps": schedule["steps"],
        "exposures": schedule["exposures"], "schedule_sha256": schedule["sha256"],
        "initial_model_sha256": initial_hash, "final_model_sha256": final_hash,
        "initial_encoder_sha256": encoder_hash, "final_encoder_sha256": final_encoder_hash,
        "encoders_frozen": encoders_frozen, "trainable_parameters": sum(p.numel() for _, p in trainable),
        "gradients_finite": True, "maximum_gradient_norm": gradient_max,
        "visibility_weight": visibility_weight, "visibility_counts": visibility_totals,
        "training_seconds": training_seconds, "total_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "test_accessed": False, "checkpoint_selection": "fixed_last"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
