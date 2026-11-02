"""Single-process and dynamic-graph DDP training with exact boundary resumes."""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from . import build_model
from .config import load_config
from .data import build_dataset, collate_frames
from .losses import ReconstructionLoss
from .runtime import (
    setup_runtime,
    cleanup_runtime,
    precision_context,
    make_scaler,
    move,
    seed_all,
    all_ranks_true,
    capture_rng,
    restore_rng,
)


class ModelAndLoss(nn.Module):
    """DDP sees only the scalar used in backward, not unused auxiliary outputs.

    Active query branches vary with data and across ranks. Returning the large scene
    dictionaries would tell DDP that dormant heads are live, causing missing-hook
    errors on later iterations. Detached logging values are stored separately.
    """

    def __init__(self, model, criterion, cfg):
        super().__init__()
        self.model, self.criterion, self.cfg = model, criterion, cfg
        self.last_metrics = {}
        self.last_supervision = None
        self.last_supervised_samples = 0

    def forward(self, inputs, targets):
        m = self.cfg["model"]
        from .multifocal.data import mask_camera_inputs

        inputs = mask_camera_inputs(inputs, self.cfg.get("camera", {}))
        label_mask = inputs["view_mask"][..., None, None]
        if "image_mask" in inputs:
            label_mask = label_mask & inputs["image_mask"]
        sample_has = torch.zeros(
            inputs["rgb"].shape[0], device=inputs["rgb"].device, dtype=torch.bool
        )
        if "depth" in targets:
            depth = targets["depth"]
            depth_valid = targets.get("depth_valid", depth > 0)
            if depth.ndim == 5:
                depth = depth.squeeze(2)
                depth_valid = depth_valid.squeeze(2)
            valid = (
                depth_valid
                & torch.isfinite(depth)
                & (depth > 0)
                & label_mask
            )
            sample_has |= valid.flatten(1).any(-1)
        if m.get("semantic_classes", 0) > 0 and "semantic" in targets:
            sample_has |= ((targets["semantic"] >= 0) & label_mask).flatten(1).any(-1)
        self.last_supervised_samples = int(sample_has.sum())
        prediction = self.model(inputs)
        if hasattr(self.model, "training_loss"):
            losses = self.model.training_loss(prediction, inputs, targets)
        else:
            losses = self.criterion(prediction, inputs, targets)
        if "supervised_samples" in losses:
            # 只计实际输出且有标签的相机；其他相机有GT不代表本次有监督。
            self.last_supervised_samples = int(losses["supervised_samples"])
        self.last_supervision = prediction.get("multifocal_supervision")
        self.last_metrics = {
            k: v.detach() for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1
        }
        return losses["total"]


def wrap_ddp(module, runtime):
    if not runtime.distributed:
        return module
    return DDP(
        module,
        device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
        output_device=runtime.local_rank if runtime.device.type == "cuda" else None,
        find_unused_parameters=True,
        static_graph=False,
        broadcast_buffers=False,
    )


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def select_trainable_parameters(model, settings):
    """只进一步冻结未选模块；不解冻原本冻结的教师或关闭的预测头。"""
    prefixes = settings.get("trainable_prefixes", [])
    if not isinstance(prefixes, list) or any(not isinstance(p, str) or not p for p in prefixes):
        raise ValueError("train.trainable_prefixes须为非空模块前缀的列表")
    parameters = list(model.named_parameters())
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name, _ in parameters):
            raise ValueError("训练前缀没有对应参数: " + prefix)
    if prefixes:
        for name, parameter in parameters:
            if not any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad_(False)
    count = sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    if not count:
        raise ValueError("训练配置没有留下可学习参数")
    return count


def _data_signature(cfg, dataset, synthetic):
    if synthetic:
        return {"synthetic": True, "parameters": cfg["synthetic"], "length": len(dataset)}
    data = cfg["data"]
    manifest = Path(data["train_manifest"])
    if not manifest.is_absolute():
        manifest = Path(data["root"]) / manifest
    return {
        "synthetic": False,
        "parameters": data,
        "length": len(dataset),
        "train_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def training_contract(cfg, dataset, runtime, synthetic, precision, sampler_kind="distributed"):
    t = cfg["train"]
    contract = {
        "world_size": runtime.world_size,
        "batch_size_per_rank": t["batch_size"],
        "accumulation_steps": t.get("accumulation_steps", 1),
        "num_workers": t["num_workers"],
        "precision": precision,
        "seed": t["seed"],
        "sampler": sampler_kind,
        "optimizer": {k: t[k] for k in ("learning_rate", "weight_decay", "grad_clip")},
        "data": _data_signature(cfg, dataset, synthetic),
    }
    if "camera" in cfg:
        contract["camera"] = cfg["camera"]
    if t.get("trainable_prefixes"):
        contract["trainable_prefixes"] = t["trainable_prefixes"]
    return contract


def _main_action(runtime, action):
    error = None
    if runtime.is_main:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if runtime.distributed:
        message = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
    if error:
        raise RuntimeError(f"Rank 0 write failed: {error}")


def save_checkpoint(
    path, model, optimizer, scaler, cfg, contract, runtime, step, epoch, batch_offset
):
    local_rng = capture_rng(runtime)
    if runtime.distributed:
        rngs = [None] * runtime.world_size
        dist.all_gather_object(rngs, local_rng)
    else:
        rngs = [local_rng]

    def write():
        record = {
            "format_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "contract": contract,
            "step": step,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "rng_by_rank": rngs,
            "accumulation_boundary": True,
            "torch_rng": rngs[0]["torch"],
            "python_rng": rngs[0]["python"],
        }
        if hasattr(model, "architecture"):
            record["architecture"] = model.architecture
        if "camera" in cfg:
            from .multifocal.options import effective_mode_metadata

            record["model_mode"] = effective_mode_metadata(cfg)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(record, temporary)
        os.replace(temporary, path)

    _main_action(runtime, write)


def _restore_checkpoint(record, model, optimizer, scaler, cfg, contract, runtime):
    from copy import deepcopy

    saved_model, current_model = deepcopy(record["config"]["model"]), deepcopy(cfg["model"])
    matching = getattr(model, "pretrained_matching", None)
    if matching is not None and matching.initialization_source == "checkpoint":
        # 完整状态已严格载入，初始化文件位置不再是运行模型的一部分。
        # 混合权重等算法设置仍参与完全一致检查。
        for settings in (saved_model, current_model):
            settings.get("pretrained_matching", {}).pop("weights", None)
            settings.get("pretrained_matching", {}).pop("prefix", None)
    foundation = getattr(getattr(model, "encoder", None), "foundation", None)
    if foundation is not None and foundation.initialization_source == "checkpoint":
        for settings in (saved_model, current_model):
            initialization = settings.get("deep_encoder", {}).get("foundation", {})
            initialization.pop("weights", None)
            initialization.pop("prefix", None)
    if saved_model != current_model:
        raise ValueError("Resume requires an identical model configuration")
    if record["config"].get("direct") != cfg.get("direct"):
        raise ValueError("Resume requires identical direct architecture and reference scheduling")
    if record["config"].get("loss", {}) != cfg.get("loss", {}):
        raise ValueError("Resume requires an identical loss configuration")
    if record.get("format_version", 1) >= 2:
        if record.get("contract") != contract:
            keys = sorted(
                k
                for k in set(record.get("contract", {})) | set(contract)
                if record.get("contract", {}).get(k) != contract.get(k)
            )
            raise ValueError("Resume topology/training/data contract changed: " + ", ".join(keys))
        if record.get("accumulation_boundary") is not True:
            raise ValueError("Only checkpoints at a completed accumulation window are supported")
        rng = record["rng_by_rank"][runtime.rank]
    else:
        if runtime.world_size != 1 or contract["accumulation_steps"] != 1:
            raise ValueError(
                "Legacy single-device checkpoints require world_size=1 and accumulation_steps=1"
            )
        old = record["config"]
        if (
            old["train"]["batch_size"] != cfg["train"]["batch_size"]
            or old["train"]["num_workers"] != cfg["train"]["num_workers"]
        ):
            raise ValueError("Legacy resume requires the same batch size and workers")
        if old.get("data", {}) != cfg.get("data", {}) or old.get("synthetic", {}) != cfg.get(
            "synthetic", {}
        ):
            raise ValueError("Legacy resume requires the same data configuration")
        cuda = record.get("cuda_rng", [])
        rng = {
            "torch": record["torch_rng"],
            "python": record["python_rng"],
            "cuda": (
                cuda[runtime.device.index or 0] if cuda and runtime.device.type == "cuda" else None
            ),
        }
        if runtime.is_main:
            print(
                "Legacy checkpoint: restoring original single-device sampler and saved RNG; legacy files did not record NumPy RNG or dataset content hashes.",
                flush=True,
            )
    model.load_state_dict(record["model"])
    optimizer.load_state_dict(record["optimizer"])
    scaler.load_state_dict(record.get("scaler", {}))
    return int(record["step"]), int(record["epoch"]), int(record["batch_offset"]), rng


def _loss_log(metrics, runtime):
    local = {k: float(v) for k, v in metrics.items()}
    if not runtime.distributed:
        return local
    gathered = [None] * runtime.world_size
    dist.all_gather_object(gathered, local)
    keys = set().union(*(x.keys() for x in gathered))
    return {k: sum(x.get(k, 0.0) for x in gathered) / runtime.world_size for k in keys}


def train(args):
    cfg = load_config(args.config)
    precision = cfg["train"]["precision"]
    runtime = setup_runtime(args.device, cfg["runtime"]["dist_timeout"])
    log = None
    try:
        torch.set_num_threads(cfg["runtime"]["threads"])
        seed_all(cfg["train"]["seed"])
        dataset = build_dataset(cfg, "train", synthetic=args.synthetic)
        if not len(dataset):
            raise ValueError("Empty training dataset")
        initial_path = getattr(args, "init_checkpoint", "")
        resume_path = getattr(args, "resume", "")
        if initial_path and resume_path:
            raise ValueError("--init-checkpoint 与 --resume 不能同时使用")
        record_path = resume_path or initial_path
        record = (
            torch.load(record_path, map_location="cpu", weights_only=True) if record_path else None
        )
        checkpoint = record if resume_path else None
        from .multifocal.checkpoint import checkpoint_model_state

        model = build_model(
            cfg, checkpoint_state=checkpoint_model_state(record) if record else None
        ).to(runtime.device)
        init_report = None
        if initial_path:
            from .multifocal.checkpoint import load_model_weights

            init_report = load_model_weights(model, record, allow_base_transfer=True)
        trainable_count = select_trainable_parameters(model, cfg["train"])
        wrapped = ModelAndLoss(model, ReconstructionLoss(cfg), cfg).to(runtime.device)
        forward = wrap_ddp(wrapped, runtime)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        scaler = make_scaler(runtime.device, precision)
        sampler_kind = (
            checkpoint.get("contract", {}).get("sampler", "legacy_random")
            if checkpoint
            else "distributed"
        )
        contract = training_contract(cfg, dataset, runtime, args.synthetic, precision, sampler_kind)
        seed_all(cfg["train"]["seed"] + runtime.rank)
        step = epoch = offset = 0
        pending_rng = None
        if checkpoint:
            step, epoch, offset, pending_rng = _restore_checkpoint(
                checkpoint, model, optimizer, scaler, cfg, contract, runtime
            )
        output = Path(args.output)
        samples_per_rank = math.ceil(len(dataset) / runtime.world_size)
        padding = samples_per_rank * runtime.world_size - len(dataset)
        t = cfg["train"]
        info = {
            "device": str(runtime.device),
            "world_size": runtime.world_size,
            "precision": precision,
            "trainable_parameters": trainable_count,
            "batch_size_per_rank": t["batch_size"],
            "accumulation_steps": t["accumulation_steps"],
            "nominal_effective_batch_size": t["batch_size"]
            * runtime.world_size
            * t["accumulation_steps"],
            "sampler": sampler_kind,
            "sampler_padding_per_epoch": padding,
            "dataset_frames": len(dataset),
            "drop_last": False,
            "sync_every_microbatch": True,
            "find_unused_parameters": True,
        }
        if "camera" in cfg:
            from .multifocal.options import effective_mode_metadata

            info["model_mode"] = effective_mode_metadata(cfg)
        if init_report is not None:
            info["weight_initialization"] = init_report

        def prepare_output():
            output.mkdir(parents=True, exist_ok=True)
            (output / "config.json").write_text(json.dumps(cfg, indent=2))
            (output / "runtime.json").write_text(json.dumps(info, indent=2))

        _main_action(runtime, prepare_output)
        if runtime.is_main:
            log = (output / "train.jsonl").open("a" if checkpoint else "w")
            print(json.dumps({"event": "runtime", **info}), flush=True)
            if "camera" in cfg:
                print(
                    json.dumps({"event": "effective_config", "config": cfg}, ensure_ascii=False),
                    flush=True,
                )
            if padding:
                print(
                    f"DistributedSampler pads {padding} repeated frame(s) per epoch so every rank has equal work; no frame is dropped.",
                    flush=True,
                )
        forward.train()
        start = time.perf_counter()
        consecutive_overflow = 0
        while step < t["max_steps"]:
            generator = torch.Generator().manual_seed(
                t["seed"] + epoch * runtime.world_size + runtime.rank
            )
            if sampler_kind == "legacy_random":
                generator.manual_seed(t["seed"] + epoch)
                sampler = RandomSampler(dataset, generator=generator)
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=runtime.world_size,
                    rank=runtime.rank,
                    shuffle=True,
                    seed=t["seed"],
                    drop_last=False,
                )
                sampler.set_epoch(epoch)
            loader = DataLoader(
                dataset,
                batch_size=t["batch_size"],
                sampler=sampler,
                drop_last=False,
                generator=generator,
                num_workers=t["num_workers"],
                collate_fn=collate_frames,
                worker_init_fn=seed_worker,
                persistent_workers=False,
                multiprocessing_context="spawn" if t["num_workers"] > 0 else None,
                pin_memory=t.get("pin_memory", True) and runtime.device.type == "cuda",
            )
            if not 0 <= offset <= len(loader):
                raise ValueError("Checkpoint batch offset is outside this epoch")
            iterator = iter(loader)
            for _ in range(offset):
                next(iterator)
            # Replaying prior loader batches must not advance the saved model/
            # main-process augmentation RNG before the first resumed batch.
            if pending_rng is not None:
                restore_rng(pending_rng, runtime)
                pending_rng = None
            epoch_start_offset = offset
            epoch_supervised = False
            while offset < len(loader) and step < t["max_steps"]:
                window = min(t["accumulation_steps"], len(loader) - offset)
                window_samples = min(
                    window * t["batch_size"], len(sampler) - offset * t["batch_size"]
                )
                optimizer.zero_grad(set_to_none=True)
                metrics = {}
                supervised_samples = 0
                for _ in range(window):
                    batch = move(
                        next(iterator), runtime.device, non_blocking=runtime.device.type == "cuda"
                    )
                    actual_batch = batch["inputs"]["rgb"].shape[0]
                    weight = actual_batch / window_samples
                    with precision_context(runtime.device, precision):
                        loss = forward(batch["inputs"], batch["targets"])
                    supervised_samples += wrapped.last_supervised_samples
                    if not all_ranks_true(torch.isfinite(loss.detach()).all().item(), runtime):
                        raise FloatingPointError(
                            "A rank produced a nonfinite loss; all ranks aborted before backward"
                        )
                    # No no_sync(): every backward synchronizes, including
                    # dynamic heads that differ between views/ranks/microbatches.
                    scaler.scale(loss * weight).backward()
                    for key, value in wrapped.last_metrics.items():
                        metrics[key] = metrics.get(key, 0.0) + value * weight
                    offset += 1
                if all_ranks_true(supervised_samples == 0, runtime):
                    # AdamW即使零梯度也会衰减权重；无监督窗口不能计作优化步。
                    optimizer.zero_grad(set_to_none=True)
                    if runtime.is_main:
                        message = {
                            "event": "no_supervision",
                            "step": step,
                            "epoch": epoch,
                            "batch_offset": offset,
                            "optimizer_step_skipped": True,
                        }
                        print(json.dumps(message), flush=True)
                        log.write(json.dumps(message) + "\n")
                        log.flush()
                    continue
                epoch_supervised = True
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), t["grad_clip"], error_if_nonfinite=False
                )
                finite = all_ranks_true(torch.isfinite(norm).item(), runtime)
                if not finite:
                    if precision != "fp16":
                        raise FloatingPointError(
                            "A rank produced nonfinite gradients; all ranks aborted before optimizer update"
                        )
                    consecutive_overflow += 1
                    scaler.update(new_scale=scaler.get_scale() * 0.5)
                    if runtime.is_main:
                        message = {
                            "event": "fp16_overflow",
                            "step": step,
                            "epoch": epoch,
                            "batch_offset": offset,
                            "scale": scaler.get_scale(),
                            "optimizer_step_skipped": True,
                        }
                        print(json.dumps(message), flush=True)
                        log.write(json.dumps(message) + "\n")
                        log.flush()
                    if consecutive_overflow > t.get("max_consecutive_overflow", 16):
                        raise FloatingPointError("Too many consecutive fp16 overflows")
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_overflow = 0
                step += 1
                record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_offset": offset,
                    "grad_norm": float(norm),
                    "seconds": time.perf_counter() - start,
                    "effective_batch_size": window_samples * runtime.world_size,
                    "microbatches": window,
                    "supervised_samples_rank0": supervised_samples,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "precision": precision,
                    **_loss_log(metrics, runtime),
                }
                if wrapped.last_supervision is not None:
                    record["multifocal_supervision_rank0"] = wrapped.last_supervision
                if runtime.is_main:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if step % t["log_every"] == 0:
                        print(json.dumps(record), flush=True)
                if step % t["checkpoint_every"] == 0 or step >= t["max_steps"]:
                    save_checkpoint(
                        output / "last.pt",
                        model,
                        optimizer,
                        scaler,
                        cfg,
                        contract,
                        runtime,
                        step,
                        epoch,
                        offset,
                    )
            if step < t["max_steps"]:
                if epoch_start_offset == 0 and not epoch_supervised:
                    raise ValueError(
                        "整个epoch没有可用监督，未更新参数；请使用已核实深度的supervised清单，或单独实现自监督损失"
                    )
                epoch += 1
                offset = 0
        return model
    finally:
        if log is not None:
            log.close()
        cleanup_runtime(runtime)
