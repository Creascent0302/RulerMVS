"""早期相机条件化与后融合的小预算对照；冻结原编码参数，保留适配器反向。"""

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


def install(model, settings):
    from research.early_geometry_20261030.encoding import install_encoding
    from research.early_geometry_20261030.fusion import install_fusion

    install_encoding(model, settings.get("encoding", {}))
    install_fusion(model, settings.get("late_fusion", False))
    return model


def load_model(filename, device="cpu"):
    """本轮权重恢复：先创建附加参数，再严格加载，禁止静默遗漏研究结构。"""
    saved = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
    model = PretrainingModel(saved["config"], initialize=False)
    if saved.get("experiment") == "early_geometry_20261030":
        install(model, saved["pilot_settings"])
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    return model, saved


def base_encoder_hash(model, original_keys):
    state = {name: value for name, value in model.state_dict().items()
             if name in original_keys and name.startswith(("foundation.", "matching."))}
    return parameter_hash(state)


@torch.inference_mode()
def verify_initial_outputs(model, checkpoint, config, dataset, val_plan, device):
    """真实完整编码器初值对照，比较深度和候选张量，不以聚合指标代替。"""
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode(False):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        reference = PretrainingModel(config, initialize=False)
        reference.load_state_dict(saved["model"], strict=True)
        reference.requires_grad_(False).eval().to(device)
        install_position_cache(reference.foundation.model)
    model.eval()

    def compare(left, right, prefix=""):
        count = 0
        if isinstance(right, torch.Tensor):
            if not torch.equal(left, right):
                raise RuntimeError(f"零初始化输出不一致：{prefix}")
            return 1
        if isinstance(right, dict):
            for key, value in right.items():
                count += compare(left[key], value, prefix + "." + key)
        elif isinstance(right, (list, tuple)):
            for index, value in enumerate(right):
                count += compare(left[index], value, prefix + f"[{index}]")
        return count

    records = {}
    for domain_index, domain in enumerate(dataset.domains):
        index = val_plan[domain["name"]]["indices"][0]
        batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
        before = reference(batch["inputs"], domain, reference=0)
        after = model(batch["inputs"], domain, reference=0)
        records[domain["name"]] = {"index": index, "tensors_bitwise_equal": compare(after, before),
                                   "real_foundation_blocks": len(model.foundation.model.blocks)}
    del reference
    return records


def train_mode(model):
    # 原 ViT/ResNet/FPN 的统计和随机层保持 eval；这不等于 no_grad。
    model.eval()
    model.training = True
    model.decoder.train()
    for encoder in (model.foundation, model.matching):
        if hasattr(encoder, "camera_adapter"):
            encoder.camera_adapter.train()


def set_trainable_parameters(model, new_names, prefixes=("decoder.",)):
    """默认保留旧decoder续训；空前缀列表时只训练显式新增参数。"""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(tuple(prefixes)) or name in new_names)


def sources():
    names = ["jar/pretraining/model.py", "jar/pretraining/encoders.py",
             "jar/pretraining/data.py", "jar/pretraining/losses.py",
             "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py",
             "research/near_evidence_20261030/pilot.py",
             "research/v2_training_20261028/objectives.py"]
    names.extend(str(file.relative_to(ROOT)) for file in
                 (ROOT / "research/early_geometry_20261030").glob("*.py"))
    return sorted(set(names))


def run(settings):
    offline_only()
    settings = dict(seed=20261030, **settings) if "seed" not in settings else settings
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有实验：{output}")
    checkpoint = path(settings["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = json.loads(path(settings["data_config"]).read_text())
    config = copy.deepcopy(saved["config"])
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
        "smoke": settings.get("smoke", False),
        "source_sha256": {name: digest(ROOT / name) for name in sources()},
        "scope": "历史已查看的VAL小样本，非盲确认；固定末步，无TEST选模。",
    })
    for name, content in manifests.items():
        folder = output / "selected_train"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".jsonl")).write_text(content)
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
    original_keys = set(model.state_dict())
    original_parameter_names = set(dict(model.named_parameters()))
    original_hash = parameter_hash(model.state_dict())
    install(model, settings)
    new_names = set(dict(model.named_parameters())) - original_parameter_names
    # 安装不能篡改旧权重，原编码器哈希仅统计原有 keys。
    installed_original = {name: value for name, value in model.state_dict().items() if name in original_keys}
    if parameter_hash(installed_original) != original_hash:
        raise RuntimeError("安装研究模块改变了原始权重")
    encoder_hash = base_encoder_hash(model, original_keys)
    initial_new = {name: value.detach().clone() for name, value in model.named_parameters() if name in new_names}
    del saved
    model.requires_grad_(False).eval().to(device)
    set_trainable_parameters(model, new_names, settings.get("trainable_prefixes", ["decoder."]))
    install_position_cache(model.foundation.model)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    decay = settings.get("weight_decay", 1e-4)
    groups = []
    for new in (False, True):
        for regularized in (False, True):
            params = [p for name, p in trainable if (name in new_names) == new
                      and (p.ndim > 1 and not name.endswith("bias")) == regularized]
            if params:
                groups.append({"params": params,
                               "lr": settings.get("new_lr", 1e-3) if new else settings.get("lr", 1e-5),
                               "weight_decay": decay if regularized else 0.})
    optimizer = torch.optim.AdamW(groups)
    train_data, val_data = MixedDataset(config, "train"), MixedDataset(val_config, "val")
    loader = DataLoader(train_data, batch_sampler=schedule["batches"], collate_fn=collate_mixed,
                        num_workers=settings.get("workers", 2), pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(settings["seed"]))
    start = time.monotonic()
    if settings.get("verify_initial", settings.get("smoke", False)):
        write_json(output / "initial_tensor_check.json", verify_initial_outputs(
            model, checkpoint, config, val_data, val_plan, device))
    write_json(output / "initial.json", evaluate(model, val_data, val_plan, device))
    training_start = time.monotonic()
    train_mode(model)
    gradient_max = 0.
    branch_gradient_max = {}
    with (output / "train.jsonl").open("w") as log:
        for step, batch in enumerate(loader, 1):
            batch = move(batch, device)
            domain = train_data.domains[batch["domain_index"]]
            prediction = model(batch["inputs"], domain, reference=0)
            terms = depth_objective(prediction, batch["inputs"], batch["targets"], domain,
                                    config["loss"], settings.get("correspondence_weight", 0.))
            if settings.get("actual_fine_risk", True):
                risk = fine_actual_regret(prediction, batch["inputs"], batch["targets"], domain)
                terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
            if not bool(torch.isfinite(terms["total"])) or not int(terms["supervised_pixels"]):
                raise RuntimeError(f"step {step}: loss非有限或无监督")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            branch_gradients = {}
            for branch in ("foundation", "matching", "decoder"):
                values = [p.grad.detach().norm() for name, p in trainable
                          if name in new_names and name.startswith(branch + ".") and p.grad is not None]
                if values:
                    value = float(torch.stack(values).norm())
                    branch_gradients[branch] = value
                    branch_gradient_max[branch] = max(branch_gradient_max.get(branch, 0.), value)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable],
                                                 settings.get("grad_clip", 1.), error_if_nonfinite=True)
            gradient_max = max(gradient_max, float(norm))
            optimizer.step()
            row = {"step": step, "dataset": domain["name"], "loss": float(terms["total"].detach()),
                   "gradient_norm": float(norm), "new_branch_gradients": branch_gradients}
            log.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0 or step == schedule["steps"]:
                log.flush()
                print(json.dumps(row), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - training_start
    write_json(output / "final.json", evaluate(model, val_data, val_plan, device))
    final_encoder_hash = base_encoder_hash(model, original_keys)
    if encoder_hash != final_encoder_hash:
        raise RuntimeError("原编码器/FPN权重或统计发生变化")
    changed_new = {name: float((parameter.detach().cpu() - initial_new[name]).abs().max())
                   for name, parameter in model.named_parameters() if name in new_names}
    for branch in ("foundation", "matching", "decoder"):
        names = [name for name in new_names if name.startswith(branch + ".")]
        if names and (branch_gradient_max.get(branch, 0.) <= 0 or not any(changed_new[name] > 0 for name in names)):
            raise RuntimeError(f"新增{branch}分支未获得非零梯度或参数未更新")
    atomic_save({"architecture": model.architecture, "experiment": "early_geometry_20261030",
                 "model": model.state_dict(), "config": config, "pilot_settings": settings,
                 "selection": "fixed_last_no_test_selection"}, output / "final.pt",
                settings.get("minimum_free_gib", 200))
    write_json(output / "completed.json", {
        "completed": True, "steps": schedule["steps"], "exposures": schedule["exposures"],
        "schedule_sha256": schedule["sha256"], "initial_original_sha256": original_hash,
        "initial_encoder_sha256": encoder_hash, "final_encoder_sha256": final_encoder_hash,
        "original_encoders_frozen": True, "gradients_finite": True,
        "maximum_gradient_norm": gradient_max, "new_branch_gradient_max": branch_gradient_max,
        "new_parameter_changes": changed_new, "new_parameter_names": sorted(new_names),
        "new_parameter_count": sum(p.numel() for name, p in trainable if name in new_names),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "training_seconds": training_seconds, "total_seconds": time.monotonic() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "test_accessed": False, "checkpoint_selection": "fixed_last",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
