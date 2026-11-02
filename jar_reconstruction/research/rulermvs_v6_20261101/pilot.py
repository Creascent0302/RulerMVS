"""V6与同预算V5续训；固定TRAIN/VAL、末步权重，不用TEST选方案。"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.runner import atomic_save
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import evaluate, parameter_hash, prepare, train_mode
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from research.rulermvs_v6_20261101.objective import objective
from tools.smoke_outdoor_datasets import offline_only


class EqualPixels(nn.Module):
    """只统一输入信息，不更改K/T或标签；包装器不写入checkpoint。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, domain, reference=0):
        return self.model(equal_pixel_inputs(inputs), domain, reference)


def run(settings):
    offline_only()
    seed_all(settings["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    saved = torch.load(settings["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if "handoff" in settings:
        if set(settings["handoff"]) != {"upsample_feature"}:
            raise ValueError("此次独立适配只允许设置handoff.upsample_feature")
        saved["config"] = copy.deepcopy(saved["config"])
        saved["config"].setdefault("handoff", {}).update(settings["handoff"])
    model, config = build_from_checkpoint(saved, settings["variant"])
    source = json.loads(Path(settings["data_config"]).read_text())
    for key in ("datasets", "image_size", "matching_image_size"):
        config[key] = copy.deepcopy(source[key])
    config.pop("validation_datasets", None)
    if "validation_datasets" in source:
        config["validation_datasets"] = copy.deepcopy(source["validation_datasets"])
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {k: v for k, v in domain["manifests"].items() if k in ("train", "val")}
    config.update(seed=settings["seed"], augmentation=True)
    if "loss" in settings:
        config["loss"] = copy.deepcopy(settings["loss"])
    model.config = config
    selections, manifests, validation, plan, schedule = prepare(config, settings)
    output.mkdir(parents=True)
    write_json(output / "settings.json", settings)
    objective_source = Path(__file__).with_name("objective.py")
    (output / "objective_used.py").write_text(objective_source.read_text())
    write_json(output / "plan.json", {"training": selections, "validation": plan,
        "schedule": schedule, "checkpoint_sha256": digest(settings["checkpoint"]),
        "test_accessed": False, "input_mode": "equal_pixels", "selection": "fixed_last_step",
        "objective_sha256": digest(objective_source)})
    model.requires_grad_(False).to(device).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(tuple(settings["trainable_prefixes"])))
    install_position_cache(model.foundation.model)
    frozen_before = parameter_hash(model.state_dict(), ("foundation.", "matching.", "bridge."))
    verify_coarse = settings.get("verify_coarse_frozen", False)
    fine_prefixes = ("decoder.fine_inherit.", "decoder.upsample_mask.", "decoder.handoff.")

    def coarse_parameter_hash():
        # depth_lattice是随域搜索范围变化的非学习缓冲量，不当作训练参数。
        state = {name: parameter for name, parameter in model.named_parameters()
                 if name.startswith("decoder.") and not name.startswith(fine_prefixes)}
        return parameter_hash(state)

    coarse_before = coarse_parameter_hash() if verify_coarse else None
    verify_preupsample = settings.get("verify_preupsample_frozen", False)

    def preupsample_parameter_hash():
        state = {name: parameter for name, parameter in model.named_parameters()
                 if not name.startswith("decoder.upsample_mask.")}
        return parameter_hash(state)

    preupsample_before = preupsample_parameter_hash() if verify_preupsample else None
    del saved
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = {}
    for name, parameter in parameters:
        lr = settings["new_lr"] if name.startswith("decoder.handoff.") else settings["lr"]
        decay = settings["weight_decay"] if parameter.ndim > 1 else 0.
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": decay}
                                   for (lr, decay), p in groups.items()])
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(validation, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    wrapper = EqualPixels(model)
    write_json(output / "initial.json", evaluate(wrapper, val_data, plan, device))
    coarse_probe = None
    fine_probe = None
    if verify_coarse or verify_preupsample:
        probe_domain = val_data.domains[0]
        probe_index = plan[probe_domain["name"]]["indices"][0]
        probe = move(collate_mixed([val_data[(0, probe_index, -1)]]), device)
        with torch.no_grad():
            prediction = model(equal_pixel_inputs(probe["inputs"]), probe_domain)
            coarse_probe = {key: prediction["coarse"][key].detach().cpu()
                            for key in ("depth", "scores", "candidate_depth")}
            if verify_preupsample:
                fine_probe = {key: prediction["fine"][key].detach().cpu()
                              for key in ("depth", "scores", "candidate_depth")}
        del prediction, probe
    seed_all(settings["seed"])
    train_mode(model)
    largest_gradient = 0.
    gate_gradient = 0.
    started = time.monotonic()
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            inputs = equal_pixel_inputs(batch["inputs"])
            prediction = model(inputs, domain, reference=0)
            terms = objective(prediction, inputs, batch["targets"], domain, config.get("loss", {}),
                              grouped=settings.get("grouped_loss", True))
            if not bool(torch.isfinite(terms["total"])) or not int(terms["supervised_pixels"]):
                raise RuntimeError("训练损失非有限或无监督像素")
            terms["total"].backward()
            for name, p in parameters:
                if name.startswith("decoder.handoff.") and p.grad is not None:
                    gate_gradient = max(gate_gradient, float(p.grad.abs().max()))
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            largest_gradient = max(largest_gradient, float(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            row = {"step": step, "domain": domain["name"], "loss": float(terms["total"].detach()),
                   "gradient_norm": float(norm)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            del prediction, batch, terms
    seconds = time.monotonic() - started
    write_json(output / "final.json", evaluate(wrapper, val_data, plan, device))
    assert frozen_before == parameter_hash(model.state_dict(), ("foundation.", "matching.", "bridge."))
    payload = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
               "pilot_settings": settings, "selection": "fixed_last_step_no_test"}
    atomic_save(payload, output / "final.pt", settings["minimum_free_gib"])
    # 真正从落盘checkpoint恢复；推理函数只接收inputs，扰动标签不影响输出。
    restored_record = torch.load(output / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    restored, _ = build_from_checkpoint(restored_record, settings["variant"])
    restored.requires_grad_(False).to(device).eval()
    install_position_cache(restored.foundation.model)
    domain = val_data.domains[0]
    index = plan[domain["name"]]["indices"][0]
    batch = move(collate_mixed([val_data[(0, index, -1)]]), device)
    inputs = equal_pixel_inputs(batch["inputs"])
    with torch.no_grad():
        expected = model(inputs, domain)
        actual = restored(inputs, domain)
        batch["targets"]["depth"].mul_(7)
        changed = restored(inputs, domain)
    checks = {key: bool(torch.equal(expected[key], actual[key])) for key in ("depth", "valid")}
    checks["targets_isolated"] = bool(torch.equal(actual["depth"], changed["depth"]))
    if verify_coarse:
        checks["coarse_parameters_unchanged"] = coarse_before == coarse_parameter_hash()
        checks["coarse_outputs_unchanged"] = all(
            torch.equal(value, actual["coarse"][key].detach().cpu()) for key, value in coarse_probe.items())
    if verify_preupsample:
        checks["preupsample_parameters_unchanged"] = (
            preupsample_before == preupsample_parameter_hash())
        checks["fine_outputs_unchanged"] = all(
            torch.equal(value, actual["fine"][key].detach().cpu()) for key, value in fine_probe.items())
    if not all(checks.values()):
        raise RuntimeError("落盘恢复或标签隔离失败")
    write_json(output / "completed.json", {"completed": True, "test_accessed": False,
        "steps": schedule["steps"], "exposures": schedule["exposures"],
        "schedule_sha256": schedule["sha256"], "training_seconds": seconds,
        "trainable_parameters": sum(p.numel() for _, p in parameters),
        "maximum_gradient_norm": largest_gradient, "gate_gradient_max": gate_gradient,
        "frozen_encoder_bridge_unchanged": True, "checkpoint_checks": checks,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
