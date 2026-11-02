"""十二候选缓存Flow的固定短试：原结构续训、直接残差与MeanFlow同预算。"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import zipfile

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
    if record["architecture"] == "ruler_flow_v2":
        from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

        model = EvidenceFlowModel(record["config"], initialize=False)
    elif record["architecture"] == "ruler_direct_v2":
        model = PretrainingModel(record["config"], initialize=False)
    else:
        raise ValueError("不支持的模型架构")
    model.load_state_dict(record["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    return model, record


def sources():
    files = list(Path(__file__).parent.glob("*.py"))
    files += list((ROOT / "jar/pretraining").glob("*.py"))
    files += [ROOT / name for name in (
        "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py",
        "research/near_evidence_20261030/pilot.py", "research/v2_training_20261028/objectives.py")]
    return sorted(set(files))


def objective(model, prediction, inputs, targets, domain, config, settings):
    terms = depth_objective(prediction, inputs, targets, domain, config.get("loss", {}), 0.)
    risk = fine_actual_regret(prediction, inputs, targets, domain)
    terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
    if settings["variant"] != "control":
        from research.ruler_flow_v2_20261031.losses import flow_objective

        auxiliary = flow_objective(model, prediction, inputs, targets, domain, config["flow"])
        terms["native_total"] = terms["total"]
        terms["total"] = terms["total"] + settings.get("aux_weight", 1.) * auxiliary["total"]
        terms.update({"flow_" + name: value for name, value in auxiliary.items()})
    return terms


@torch.no_grad()
def verify_reload(model, filename, dataset, val_plan, device):
    restored, _ = load_model(filename, device)
    domain = dataset.domains[0]
    index = val_plan[domain["name"]]["indices"][0]
    batch = move(collate_mixed([dataset[(0, index, -1)]]), device)
    expected = model(batch["inputs"], domain)
    actual = restored(batch["inputs"], domain)
    equal = {key: bool(torch.equal(expected[key], actual[key])) for key in ("depth", "valid")}
    equal.update({step: bool(torch.equal(expected[step]["depth"], actual[step]["depth"]))
                  for step in ("coarse", "fine")})
    batch["targets"]["depth"].mul_(7)
    repeated = restored(batch["inputs"], domain)
    equal["changed_targets"] = bool(torch.equal(actual["depth"], repeated["depth"]))
    if not all(equal.values()):
        raise RuntimeError("恢复或标签隔离失败")
    return equal


def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有实验：{output}")
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
    config["flow"] = copy.deepcopy(settings["flow"])
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {key: value for key, value in domain["manifests"].items()
                               if key in ("train", "val")}
    selections, manifests, val_config, val_plan, schedule = prepare(config, settings)
    output.mkdir(parents=True)
    write_json(output / "settings.json", settings)
    write_json(output / "plan.json", {
        "training": selections, "validation": val_plan, "schedule": schedule,
        "checkpoint_sha256": digest(checkpoint), "test_accessed": False,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources()},
        "scope": "固定末步，历史16VAL分8开发/8确认；不读取TEST，不按VAL选权重",
    })
    for name, text in manifests.items():
        folder = output / "selected_train"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".jsonl")).write_text(text)
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for filename in sources():
            archive.write(filename, filename.relative_to(ROOT))
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
        from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

        config["architecture"] = "ruler_flow_v2"
        model = EvidenceFlowModel(config, initialize=False)
        model.load_v2_state(saved["model"])
    original_encoder = parameter_hash(saved["model"], ("foundation.", "matching."))
    del saved
    model.requires_grad_(False).eval().to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("decoder."))
    install_position_cache(model.foundation.model)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = {}
    for name, parameter in trainable:
        lr = settings["new_lr"] if name.startswith("decoder.flow_field.") else settings["lr"]
        decay = settings["weight_decay"] if parameter.ndim > 1 and not name.endswith("bias") else 0.
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": values, "lr": lr, "weight_decay": decay}
                                   for (lr, decay), values in groups.items()])
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(val_config, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=True,
                        generator=torch.Generator().manual_seed(settings["seed"]))
    write_json(output / "initial.json", evaluate(model, val_data, val_plan, device))
    initial_field = {name: p.detach().clone() for name, p in model.named_parameters()
                     if name.startswith("decoder.flow_field.")}
    gradients = {}
    started = time.monotonic()
    model.eval()
    model.training = True
    model.decoder.train()
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            prediction = model(batch["inputs"], domain)
            terms = objective(model, prediction, batch["inputs"], batch["targets"], domain, config, settings)
            if not bool(torch.isfinite(terms["total"])):
                raise RuntimeError(f"非有限loss：{step}/{domain['name']}")
            terms["total"].backward()
            for name, parameter in trainable:
                if name.startswith("decoder.flow_field.") and parameter.grad is not None:
                    gradients[name] = max(gradients.get(name, 0.), float(parameter.grad.abs().max()))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, "dataset": domain["name"], "grad_norm": float(norm),
                      "loss": {name: float(value.detach()) for name, value in terms.items()}}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(record), flush=True)
            del prediction, terms, batch
    seconds = time.monotonic() - started
    model.eval()
    write_json(output / "final.json", evaluate(model, val_data, val_plan, device))
    if parameter_hash(model.state_dict(), ("foundation.", "matching.")) != original_encoder:
        raise RuntimeError("短试不应更新编码器参数或统计")
    updates = {name: float((p.detach() - initial_field[name]).abs().max())
               for name, p in model.named_parameters() if name in initial_field}
    if initial_field and (max(gradients.values(), default=0.) == 0 or max(updates.values(), default=0.) == 0):
        raise RuntimeError("新增流场未获得有效梯度/参数更新")
    record = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
              "pilot_settings": settings, "experiment": "ruler_flow_v2_20261031"}
    atomic_save(record, output / "final.pt", settings["minimum_free_gib"])
    restored = verify_reload(model, output / "final.pt", val_data, val_plan, device)
    write_json(output / "completed.json", {
        "completed": True, "steps": schedule["steps"], "exposures": schedule["exposures"],
        "training_seconds": seconds, "trainable_parameters": sum(p.numel() for _, p in trainable),
        "new_parameters": sum(p.numel() for name, p in trainable if name in initial_field),
        "new_gradient_maxima": gradients, "new_update_maxima": updates,
        "encoders_unchanged": True, "checkpoint_checks": restored, "test_accessed": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
