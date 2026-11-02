"""V6 起点的匹配预算短训：原结构/投影域读出/粗层先验 × 数据增强。

只读 TRAIN/VAL；同 seed 共享采样、原始增强与固定末步规则。复用原
数据、监督、评测，不替换生产训练入口。新增模块独立使用较高学习率。
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.losses import masked_mean
from jar.pretraining.runner import atomic_save
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import parameter_hash, prepare, train_mode
from research.paper_ablation_20261102.handoff_evaluate import evaluate
from research.rulermvs_v6_20261101.objective import objective
from research.rulermvs_v6_20261101.pilot import EqualPixels
from research.v7_domain_20261102.augmentation import augment_inputs
from tools.smoke_outdoor_datasets import offline_only


def retention_penalty(prediction, teacher, inputs, targets, domain, margin=0.01):
    """仅 TRAIN 真值上的单边质量约束；允许超过旧模型，不强迫复制其错误。

    max(0, |log z-log GT| - |log z_old-log GT| - margin)。旧模型和
    真值均停止梯度；margin=.01 约为 1% 相对误差容差，不保证 VAL 不退。
    """
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid = valid & inputs["view_mask"][:, :1, None, None]
    valid = valid & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid = valid & (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(valid, truth, 1.).log()
    current = prediction["depth"].float().clamp_min(1e-6).log()
    previous = teacher["depth"].detach().float().clamp_min(1e-6).log()
    regret = ((current - truth).abs() - (previous - truth).abs() - margin).clamp_min(0)
    return masked_mean(regret, valid)


def optimizer_for(model, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("decoder.center.", "decoder.context_prior.")):
            lr = settings.get("new_lr", 3e-4)
        elif name.startswith(("matching.", "bridge.")):
            lr = settings.get("feature_lr", 1e-6)
        else:
            lr = settings.get("lr", 3e-6)
        decay = settings.get("weight_decay", 1e-4) if parameter.ndim > 1 else 0.
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": values, "lr": lr, "weight_decay": decay}
                              for (lr, decay), values in groups.items()])


def run(settings):
    from research.v7_domain_20261102.model import build_model

    offline_only()
    seed_all(settings["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    output = Path(settings["output"])
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("权重和实验结果必须放到数据盘")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    sources = [Path(__file__), Path(__file__).with_name("augmentation.py"),
               Path(sys.modules[build_model.__module__].__file__),
               Path(__file__).with_name("center.py"),
               Path(__file__).with_name("prior.py"),
               ROOT / "research/rulermvs_v6_20261101/objective.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    saved = torch.load(settings["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model, config = build_model(saved, settings["variant"])
    source = json.loads(Path(settings["data_config"]).read_text())
    config = copy.deepcopy(config)
    for key in ("datasets", "validation_datasets", "image_size", "matching_image_size"):
        if key in source:
            config[key] = copy.deepcopy(source[key])
        elif key == "validation_datasets":
            config.pop(key, None)
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {key: value for key, value in domain["manifests"].items()
                               if key in ("train", "val")}
    # 原 gain/15% drop-one 增强在所有组中相同；新增强是独立干预。
    config.update(seed=settings["seed"], augmentation=True)
    if "loss" in settings:
        config["loss"] = copy.deepcopy(settings["loss"])
    model.config = config
    selections, manifests, validation, plan, schedule = prepare(config, settings)
    output.mkdir(parents=True)
    write_json(output / "settings.json", settings)
    write_json(output / "plan.json", {
        "training": selections, "validation": plan, "schedule": schedule,
        "checkpoint_sha256": digest(settings["checkpoint"]), "source_sha256": hashes,
        "test_accessed": False, "selection": "fixed_last_step", "input_mode": "equal_pixels",
    })
    (output / "source_snapshot").mkdir()
    for path in sources:
        (output / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    model.requires_grad_(False).to(device).eval()
    prefixes = tuple(settings.get("trainable_prefixes", ["decoder."]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    install_position_cache(model.foundation.model)
    frozen_names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}

    def frozen_hash():
        state = {name: parameter for name, parameter in model.named_parameters()
                 if name in frozen_names}
        state.update({name: value for name, value in model.named_buffers()
                      if name.startswith(("foundation.", "matching.", "bridge."))})
        return parameter_hash(state)

    frozen_before = frozen_hash()
    teacher = None
    retention_weight = settings.get("retention_weight", 0.)
    if retention_weight:
        teacher, _ = build_model(saved, "control")
        teacher.requires_grad_(False).to(device).eval()
        install_position_cache(teacher.foundation.model)
    del saved
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optimizer_for(model, settings)
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(validation, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=True,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    wrapper = EqualPixels(model)
    write_json(output / "initial.json", evaluate(wrapper, val_data, plan, device))
    seed_all(settings["seed"])
    # 独立 RNG 确保新增增强不改变模型原有的随机流/基线增强。
    generator = torch.Generator().manual_seed(settings["seed"] + 7153)
    train_mode(model)
    largest_gradient, center_gradient, prior_gradient = 0., 0., 0.
    gray_count, seen = 0, 0
    view_counts = {}
    started = time.monotonic()
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            inputs = equal_pixel_inputs(batch["inputs"])
            if settings.get("domain_augmentation", False):
                options = settings.get("augmentation_options", {})
                inputs, record = augment_inputs(inputs, generator=generator, **options)
                gray_count += int(record["grayscale"].sum())
            seen += inputs["rgb"].shape[0]
            for count in inputs["view_mask"].sum(1).tolist():
                view_counts[str(count)] = view_counts.get(str(count), 0) + 1
            prediction = model(inputs, domain, reference=0)
            terms = objective(prediction, inputs, batch["targets"], domain, config.get("loss", {}),
                              grouped=True, correspondence_weight=settings.get("correspondence_weight", 0.))
            if teacher is not None:
                with torch.no_grad():
                    baseline = teacher(inputs, domain, reference=0)
                penalty = retention_penalty(prediction, baseline, inputs, batch["targets"], domain,
                                            settings.get("retention_margin", .01))
                terms["retention"] = penalty
                terms["total"] = terms["total"] + retention_weight * penalty
                del baseline
            if not bool(torch.isfinite(terms["total"])) or not int(terms["supervised_pixels"]):
                raise RuntimeError("训练损失非有限或无有效监督")
            terms["total"].backward()
            for name, parameter in model.named_parameters():
                if name.startswith("decoder.center.") and parameter.grad is not None:
                    center_gradient = max(center_gradient, float(parameter.grad.abs().max()))
                if name.startswith("decoder.context_prior.") and parameter.grad is not None:
                    prior_gradient = max(prior_gradient, float(parameter.grad.abs().max()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            largest_gradient = max(largest_gradient, float(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            row = {"step": step, "dataset": domain["name"], "loss": float(terms["total"].detach()),
                   "gradient_norm": float(norm)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            del prediction, terms, batch
    seconds = time.monotonic() - started
    write_json(output / "final.json", evaluate(wrapper, val_data, plan, device))
    if frozen_before != frozen_hash():
        raise RuntimeError("冻结参数被意外修改")
    if settings["variant"] == "projective" and center_gradient <= 0:
        raise RuntimeError("新中心模块未获得非零梯度")
    if settings["variant"] == "prior" and prior_gradient <= 0:
        raise RuntimeError("新先验模块未获得非零梯度")
    payload = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
               "optimizer": optimizer.state_dict(), "pilot_settings": settings,
               "selection": "fixed_last_step_no_test", "cpu_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state(device), "augmentation_rng": generator.get_state()}
    atomic_save(payload, output / "final.pt", settings.get("minimum_free_gib", 200))
    restored_record = torch.load(output / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    restored, _ = build_model(restored_record, settings["variant"])
    restored.requires_grad_(False).to(device).eval()
    install_position_cache(restored.foundation.model)
    domain = val_data.domains[0]
    batch = move(collate_mixed([val_data[(0, plan[domain["name"]]["indices"][0], -1)]]), device)
    inputs = equal_pixel_inputs(batch["inputs"])
    with torch.no_grad():
        expected = model(inputs, domain)
        actual = restored(inputs, domain)
        batch["targets"]["depth"].mul_(7)
        changed = restored(inputs, domain)
    checks = {key: bool(torch.equal(expected[key], actual[key])) for key in ("depth", "valid")}
    checks["targets_isolated"] = bool(torch.equal(actual["depth"], changed["depth"]))
    if not all(checks.values()):
        raise RuntimeError("落盘重载或标签隔离失败")
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in sources}:
        raise RuntimeError("运行中源码改变；保留结果但不标为同版本完成")
    write_json(output / "completed.json", {
        "completed": True, "test_accessed": False, "steps": schedule["steps"],
        "exposures": schedule["exposures"], "schedule_sha256": schedule["sha256"],
        "training_seconds": seconds, "maximum_gradient_norm": largest_gradient,
        "center_gradient_max": center_gradient, "prior_gradient_max": prior_gradient,
        "frozen_parameters_unchanged": True,
        "checkpoint_checks": checks, "grayscale_exposures": gray_count,
        "actual_view_counts": view_counts, "actual_exposures": seen,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
