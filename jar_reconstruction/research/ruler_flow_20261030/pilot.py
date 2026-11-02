"""独立的四分支残差/Flow短训练；不改写生产模型或历史训练入口。"""

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


EXPERIMENT = "ruler_flow_20261030"


def install(model, options):
    if options:
        from research.ruler_flow_20261030.variants import install as install_variant

        install_variant(model, options)
    return model


def set_inference_steps(model, steps):
    if hasattr(model.decoder, "flow_options"):
        model.decoder.flow_options["inference_steps"] = int(steps)
    elif int(steps) != 1:
        raise ValueError("原V2模型不能设置Flow积分步数")


def load_model(filename, device="cpu", inference_steps=None):
    saved = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
    marker = saved.get("experiment")
    if marker not in (None, EXPERIMENT):
        raise ValueError(f"本入口不能静默恢复其他研究结构：{marker}")
    model = PretrainingModel(saved["config"], initialize=False)
    if marker == EXPERIMENT:
        install(model, saved["pilot_settings"].get("variant", {}))
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    if inference_steps is not None:
        set_inference_steps(model, inference_steps)
    return model, saved


def set_trainable(model, new_names, control=False):
    """新结构只训练新增参数；control沿用原decoder续训对照。"""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in new_names or (control and name.startswith("decoder.")))


def train_mode(model, new_names, control=False):
    model.eval()
    model.training = True  # 原forward据此返回训练需要的pyramid；旧子模块仍eval。
    if control:
        model.decoder.train()
    else:
        modules = dict(model.named_modules())
        for name in new_names:
            modules[name.rsplit(".", 1)[0]].training = True


def objective(model, prediction, inputs, targets, domain, loss_config, settings, auxiliary_options):
    terms = depth_objective(prediction, inputs, targets, domain, loss_config,
                            settings.get("correspondence_weight", 0.))
    if settings.get("actual_fine_risk", True):
        risk = fine_actual_regret(prediction, inputs, targets, domain)
        terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
    if settings.get("variant"):
        from research.ruler_flow_20261030.variants import training_loss

        auxiliary = training_loss(model, prediction, inputs, targets, domain, auxiliary_options)
        if not torch.is_tensor(auxiliary["total"]):
            raise TypeError("training_loss.total必须是可反传张量")
        terms["native_total"] = terms["total"]
        terms["total"] = terms["total"] + settings.get("aux_weight", 1.) * auxiliary["total"]
        terms.update({"aux_" + key: value for key, value in auxiliary.items()})
    return terms


def sources():
    names = [
        "jar/pretraining/model.py", "jar/pretraining/encoders.py",
        "jar/pretraining/data.py", "jar/pretraining/losses.py",
        "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py",
        "research/near_evidence_20261030/pilot.py",
        "research/role_fusion_20261030/variants.py",
        "research/v2_training_20261028/objectives.py",
    ]
    names.extend(str(file.relative_to(ROOT)) for file in
                 (ROOT / "research" / EXPERIMENT).glob("*.py"))
    return sorted(set(names))


def old_state(model, names):
    return {name: value for name, value in model.state_dict().items()
            if name in names and name != "decoder.depth_lattice"}


def evaluate_steps(model, dataset, plan, device, steps, output):
    results = {}
    for count in steps:
        set_inference_steps(model, count)
        results[str(count)] = evaluate(model, dataset, plan, device)
        write_json(output / f"final_steps{count}.json", results[str(count)])
    set_inference_steps(model, 1)
    write_json(output / "final.json", results["1"])
    return results


@torch.inference_mode()
def verify_checkpoint(model, filename, dataset, plan, device):
    """真实RGB检查严格恢复；固定inputs改变targets不能改变推理输出。"""
    with torch.inference_mode(False):
        restored, _ = load_model(filename, device)
    domain = dataset.domains[0]
    index = plan[domain["name"]]["indices"][0]
    batch = move(collate_mixed([dataset[(0, index, -1)]]), device)
    expected = model(batch["inputs"], domain, reference=0)
    actual = restored(batch["inputs"], domain, reference=0)
    fields = {"depth": (expected["depth"], actual["depth"]),
              "coarse_depth": (expected["coarse"]["depth"], actual["coarse"]["depth"]),
              "fine_depth": (expected["fine"]["depth"], actual["fine"]["depth"]),
              "fine_scores": (expected["fine"]["scores"], actual["fine"]["scores"])}
    exact = {name: bool(torch.equal(a, b)) for name, (a, b) in fields.items()}
    batch["targets"]["depth"].mul_(7)
    repeated = restored(batch["inputs"], domain, reference=0)
    unchanged = bool(torch.equal(actual["depth"], repeated["depth"]))
    if not all(exact.values()) or not unchanged:
        raise RuntimeError("checkpoint恢复或推理标签隔离检查失败")
    return {"passed": True, "index": index, "dataset": domain["name"],
            "exact_fields": exact, "fixed_inputs_changed_targets_identical": unchanged}


def run(settings):
    offline_only()
    settings = dict(settings)
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有实验：{output}")
    checkpoint = path(settings["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("experiment") is not None:
        raise ValueError("新头训练必须从原V2独立best开始，不能叠加其他研究分支")
    config = copy.deepcopy(saved["config"])
    source = json.loads(path(settings["data_config"]).read_text())
    for key in ("datasets", "validation_datasets", "image_size", "matching_image_size"):
        if key in source:
            config[key] = copy.deepcopy(source[key])
    if "validation_datasets" not in source:
        config.pop("validation_datasets", None)
    config.update(seed=settings["seed"], augmentation=settings.get("augmentation", True))
    for domain in config["datasets"] + config.get("validation_datasets", []):
        domain["manifests"] = {key: value for key, value in domain["manifests"].items()
                               if key in ("train", "val")}
    selections, manifests, val_config, val_plan, schedule = prepare(config, settings)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "settings.json", settings)
    write_json(output / "plan.json", {
        "training": selections, "validation": val_plan, "schedule": schedule,
        "checkpoint_sha256": digest(checkpoint), "test_accessed": False,
        "source_sha256": {name: digest(ROOT / name) for name in sources()},
        "scope": "历史已查看16VAL，固定末步，不读取TEST，不按VAL挑权重。",
    })
    selected = output / "selected_train"
    selected.mkdir()
    for name, content in manifests.items():
        (selected / f"{name}.jsonl").write_text(content)
    if settings.get("prepare_only", False):
        return
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources():
            archive.write(ROOT / name, name)
    seed_all(settings["seed"])
    torch.set_num_threads(settings.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = PretrainingModel(config, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    old_names = set(dict(model.named_parameters()))
    old_keys = set(model.state_dict())
    original_hash = parameter_hash(old_state(model, old_keys))
    encoder_keys = {name for name in old_keys if name.startswith(("foundation.", "matching."))}
    encoder_hash = parameter_hash(old_state(model, encoder_keys))
    install(model, settings.get("variant", {}))
    if parameter_hash(old_state(model, old_keys)) != original_hash:
        raise RuntimeError("安装新头改变了原参数或持久统计")
    new_names = set(dict(model.named_parameters())) - old_names
    control = not bool(settings.get("variant"))
    if not control and not new_names:
        raise RuntimeError("新结构没有新增可训练参数")
    initial_new = {name: value.detach().clone() for name, value in model.named_parameters() if name in new_names}
    del saved
    model.requires_grad_(False).eval().to(device)
    set_trainable(model, new_names, control)
    install_position_cache(model.foundation.model)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    decay = settings.get("weight_decay", 1e-4)
    groups = []
    for regularized in (False, True):
        values = [value for name, value in trainable
                  if (value.ndim > 1 and not name.endswith("bias")) == regularized]
        if values:
            groups.append({"params": values, "weight_decay": decay if regularized else 0.})
    optimizer = torch.optim.AdamW(groups, lr=settings["lr"] if control else settings.get("new_lr", .001))
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(val_config, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(settings["seed"]))
    auxiliary_options = dict(settings.get("variant", {}))
    auxiliary_options["_generator"] = torch.Generator(device=device).manual_seed(settings["seed"] + 1701)
    gradient_maxima = {}
    start = time.monotonic()
    if settings.get("evaluate_reduced4", False):
        set_inference_steps(model, 0)
        write_json(output / "reduced4.json", evaluate(model, val_data, val_plan, device))
    set_inference_steps(model, 1)
    write_json(output / "initial.json", evaluate(model, val_data, val_plan, device))
    training_start = time.monotonic()
    train_mode(model, new_names, control)
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            prediction = model(batch["inputs"], domain, reference=0)
            terms = objective(model, prediction, batch["inputs"], batch["targets"], domain,
                              config["loss"], settings, auxiliary_options)
            if not bool(torch.isfinite(terms["total"])) or not int(terms["supervised_pixels"]):
                raise RuntimeError(f"step {step}没有有效监督或loss非有限")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            for name, value in trainable:
                if name in new_names and value.grad is not None:
                    current = value.grad.detach().abs().amax()
                    previous = gradient_maxima.get(name)
                    gradient_maxima[name] = current if previous is None else torch.maximum(previous, current)
            norm = torch.nn.utils.clip_grad_norm_([value for _, value in trainable],
                                                 settings.get("grad_clip", 1.), error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "dataset": domain["name"], "gradient_norm": float(norm),
                   **{key: float(value.detach()) if torch.is_tensor(value) else value
                      for key, value in terms.items()}}
            log.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0 or step == schedule["steps"]:
                log.flush()
                print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - training_start
    evaluate_steps(model, val_data, val_plan, device, settings.get("evaluation_steps", [1]), output)
    final_old_hash = parameter_hash(old_state(model, old_keys))
    final_encoder_hash = parameter_hash(old_state(model, encoder_keys))
    if encoder_hash != final_encoder_hash or (not control and original_hash != final_old_hash):
        raise RuntimeError("冻结的原参数或统计发生了改变")
    changes = {name: float((value.detach().cpu() - initial_new[name]).abs().max())
               for name, value in model.named_parameters() if name in new_names}
    gradients = {name: float(gradient_maxima.get(name, 0.)) for name in new_names}
    if new_names and not (any(value > 0 for value in gradients.values()) and any(value > 0 for value in changes.values())):
        raise RuntimeError("新网络没有获得梯度或参数更新")
    heads = {}
    for head in ("velocity", "quality"):
        names = [name for name in new_names if f".flow_head.{head}." in name]
        if names:
            heads[head] = {"maximum_absolute_gradient": max(gradients[name] for name in names),
                           "maximum_parameter_change": max(changes[name] for name in names)}
            if min(heads[head].values()) <= 0:
                raise RuntimeError(f"新增{head}输出头未获得非零梯度和参数更新")
    write_json(output / "new_module_checks.json", {"passed": True, "maximum_absolute_gradients": gradients,
                                                   "maximum_parameter_changes": changes, "heads": heads})
    if not control:
        write_json(output / "old_model_frozen.json", {
            "passed": True, "initial_sha256": original_hash, "final_sha256": final_old_hash,
            "parameter_tensors": len(old_names), "state_tensors_checked": len(old_state(model, old_keys)),
            "excluded_runtime_buffers": ["decoder.depth_lattice"],
        })
    atomic_save({"architecture": model.architecture, "experiment": EXPERIMENT,
                 "model": model.state_dict(), "config": config, "pilot_settings": settings,
                 "selection": "fixed_last_no_test_selection"}, output / "final.pt",
                settings.get("minimum_free_gib", 200))
    if settings.get("smoke", False):
        write_json(output / "checkpoint_check.json", verify_checkpoint(
            model, output / "final.pt", val_data, val_plan, device))
    write_json(output / "completed.json", {
        "completed": True, "steps": schedule["steps"], "exposures": schedule["exposures"],
        "schedule_sha256": schedule["sha256"], "original_encoders_frozen": True,
        "all_original_parameters_frozen": not control, "gradients_finite": True,
        "initial_encoder_sha256": encoder_hash, "final_encoder_sha256": final_encoder_hash,
        "trainable_parameters": sum(value.numel() for _, value in trainable),
        "new_parameter_count": sum(value.numel() for name, value in trainable if name in new_names),
        "new_parameter_names": sorted(new_names), "new_parameter_changes": changes,
        "training_seconds": training_seconds, "total_seconds": time.monotonic() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "test_accessed": False, "checkpoint_selection": "fixed_last", "training_inference_steps": 1,
        "evaluation_steps": settings.get("evaluation_steps", [1]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
