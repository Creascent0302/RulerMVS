"""V5可拆卸连接的同预算短试；复用既有TRAIN/VAL与分阶段指标。"""

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
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import PretrainingModel
from jar.pretraining.runner import atomic_save
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import evaluate, parameter_hash, path, prepare
from research.v2_training_20261028.objectives import fine_actual_regret
from tools.smoke_outdoor_datasets import offline_only


def load_model(filename, device="cpu"):
    record = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
    if record["architecture"] == "rulermvs_v5":
        from research.rulermvs_v5_20261031.model import V5Model

        model = V5Model(record["config"], initialize=False)
    elif record["architecture"] == "ruler_direct_v2":
        model = PretrainingModel(record["config"], initialize=False)
    else:
        raise ValueError("此试验只支持原V2和独立V5")
    model.load_state_dict(record["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    return model, record


def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    source = json.loads(path(settings["data_config"]).read_text())
    checkpoint = path(settings["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(saved["config"])
    for key in ("datasets", "validation_datasets", "image_size", "matching_image_size"):
        if key in source:
            config[key] = copy.deepcopy(source[key])
        elif key == "validation_datasets":
            config.pop(key, None)
    config.update(seed=settings["seed"], augmentation=settings.get("augmentation", True))
    config["bridge"] = copy.deepcopy(settings["bridge"])
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {key: value for key, value in domain["manifests"].items()
                               if key in ("train", "val")}
    selections, manifests, val_config, val_plan, schedule = prepare(config, settings)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "settings.json", settings)
    write_json(output / "plan.json", {
        "training": selections, "validation": val_plan, "schedule": schedule,
        "checkpoint_sha256": digest(checkpoint), "test_accessed": False,
        "scope": "固定末步；历史VAL分开发/确认；不选择TEST或best",
    })
    for name, content in manifests.items():
        folder = output / "selected_train"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".jsonl")).write_text(content)
    seed_all(settings["seed"])
    torch.set_num_threads(settings.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings.get("device", "cuda:0"))
    torch.cuda.set_device(device)
    if settings["variant"] == "control":
        model = PretrainingModel(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
    else:
        from research.rulermvs_v5_20261031.model import V5Model

        config["architecture"] = "rulermvs_v5"
        model = V5Model(config, initialize=False)
        model.load_v2_state(saved["model"])
    encoder_hash = parameter_hash(saved["model"], ("foundation.", "matching."))
    del saved
    model.requires_grad_(False).eval().to(device)
    prefixes = tuple(settings.get("trainable_prefixes", ["decoder.", "bridge."]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    install_position_cache(model.foundation.model)
    initial_bridge = {name: p.detach().clone() for name, p in model.named_parameters()
                      if name.startswith("bridge.")}
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lr = settings["new_lr"] if name.startswith("bridge.") else settings["lr"]
            decay = settings["weight_decay"] if parameter.ndim > 1 else 0.
            groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": decay}
                                   for (lr, decay), p in groups.items()])
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(val_config, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=True,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    write_json(output / "initial.json", evaluate(model, val_data, val_plan, device))
    seed_all(settings["seed"])
    model.eval()
    model.training = True
    model.decoder.train()
    gradients = {}
    started = time.monotonic()
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            prediction = model(batch["inputs"], domain)
            terms = depth_objective(prediction, batch["inputs"], batch["targets"], domain,
                                    config.get("loss", {}), 0.)
            risk = fine_actual_regret(prediction, batch["inputs"], batch["targets"], domain)
            terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
            if not bool(torch.isfinite(terms["total"])):
                raise RuntimeError(f"非有限loss：{step}")
            terms["total"].backward()
            for name, parameter in model.named_parameters():
                if name in initial_bridge and parameter.grad is not None:
                    gradients[name] = max(gradients.get(name, 0.), float(parameter.grad.abs().max()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, "dataset": domain["name"], "loss": float(terms["total"].detach()),
                      "grad_norm": float(norm)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(record), flush=True)
            del prediction, terms, batch
    seconds = time.monotonic() - started
    write_json(output / "final.json", evaluate(model, val_data, val_plan, device))
    if parameter_hash(model.state_dict(), ("foundation.", "matching.")) != encoder_hash:
        raise RuntimeError("本轮冻结编码器参数或统计发生改变")
    updates = {name: float((p.detach() - initial_bridge[name]).abs().max())
               for name, p in model.named_parameters() if name in initial_bridge}
    if initial_bridge and (max(gradients.values(), default=0.) == 0 or max(updates.values()) == 0):
        raise RuntimeError("连接层没有非零梯度或参数更新")
    record = {"architecture": model.architecture, "model": model.state_dict(), "config": config}
    atomic_save(record, output / "final.pt", settings["minimum_free_gib"])
    restored, _ = load_model(output / "final.pt", device)
    domain = val_data.domains[0]
    index = val_plan[domain["name"]]["indices"][0]
    batch = move(collate_mixed([val_data[(0, index, -1)]]), device)
    with torch.no_grad():
        expected = model(batch["inputs"], domain)
        actual = restored(batch["inputs"], domain)
        batch["targets"]["depth"].mul_(7)
        repeated = restored(batch["inputs"], domain)
    checks = {key: bool(torch.equal(expected[key], actual[key])) for key in ("depth", "valid")}
    checks["changed_targets"] = bool(torch.equal(actual["depth"], repeated["depth"]))
    if not all(checks.values()):
        raise RuntimeError("权重恢复或标签隔离失败")
    write_json(output / "completed.json", {
        "completed": True, "steps": schedule["steps"], "exposures": schedule["exposures"],
        "training_seconds": seconds, "new_parameters": sum(p.numel() for p in initial_bridge.values()),
        "new_gradient_maxima": gradients, "new_update_maxima": updates,
        "encoders_unchanged": True, "checkpoint_checks": checks, "test_accessed": False,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
