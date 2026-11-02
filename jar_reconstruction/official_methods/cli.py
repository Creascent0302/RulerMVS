"""独立进程统一入口：python -m official_methods.cli train|infer|evaluate。"""

from __future__ import annotations
import argparse
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from jar.runtime import setup_runtime, cleanup_runtime, seed_all, move, capture_rng, restore_rng
from .data import FourCameraDataset, collate, audit_split_sequences
from .registry import build_adapter, METHODS
from .postprocess import postprocess
from .metrics import Evaluator
from .prediction_io import export_prediction


def load_config(path):
    config = json.loads(Path(path).read_text())
    model = config.get("model", {})
    config.setdefault("image_size", [model.get("image_height", 256), model.get("image_width", 384)])
    config.setdefault("seed", config.get("training", {}).get("seed", 1701))
    config.setdefault("training", {})
    config.setdefault("evaluation", {})
    if config["training"].get("precision", "fp32") != "fp32":
        raise ValueError("当前官方集成实际验证的是 fp32；其他精度不能仅改配置后静默当作支持")
    evaluation = config["evaluation"]
    for old_key in ("surface_thresholds", "fscore_thresholds_m"):
        if old_key in evaluation:
            evaluation.setdefault("surface_thresholds_m", evaluation.pop(old_key))
    return config


def optimizer_pair(adapter):
    result = adapter.configure_optimizer()
    return result if isinstance(result, tuple) else (result, None)


def save_checkpoint(path, model, optimizer, scheduler, config, runtime, step, epoch, position,
                    training_state=None):
    rank_state = {
        "rng": capture_rng(runtime),
        "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
    }
    states = [None] * runtime.world_size
    if runtime.distributed:
        dist.all_gather_object(states, rank_state)
    else:
        states[0] = rank_state
    if runtime.is_main:
        checkpoint = dict(
            format_version=1,
            method=config["method"],
            config=config,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,
            step=step,
            epoch=epoch,
            batch_in_epoch=position,
            state_by_rank=states,
            world_size=runtime.world_size,
            training_state=training_state or {},
        )
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def load_checkpoint(path, model, runtime, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != model.config["method"]:
        raise ValueError("checkpoint 与方法不匹配；原生权重应通过 model.weights 明确加载")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        if checkpoint["world_size"] != runtime.world_size:
            raise ValueError("精确续训要求相同 world_size；换卡数请仅加载模型权重开启新实验")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        rank_state = checkpoint["state_by_rank"][runtime.rank]
        restore_rng(rank_state["rng"], runtime)
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(rank_state["buffers"][name].to(buffer.device))
    return checkpoint


def make_loader(config, split, runtime, synthetic, training, targets=True):
    dataset = FourCameraDataset(config, split, synthetic, load_targets=targets)
    sampler = DistributedSampler(
        dataset,
        num_replicas=runtime.world_size if training else 1,
        rank=runtime.rank if training else 0,
        shuffle=training,
        seed=config["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 1),
        sampler=sampler,
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=partial(collate, expected_views=dataset.num_views),
        pin_memory=runtime.device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"] + runtime.rank),
    )
    return loader, sampler


def train(args, config, runtime):
    from .training_validation import append_record, check_storage, validate_depth

    loader, sampler = make_loader(config, "train", runtime, args.synthetic, True)
    adapter = build_adapter(config).to(runtime.device)
    config["implementation_limitations"] = list(getattr(adapter, "implementation_limitations", []))
    optimizer, scheduler = optimizer_pair(adapter)
    settings = config["training"]
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    planned_steps = settings.get("max_steps") or settings.get("epochs", 1) * (
        (len(loader) + accumulation - 1) // accumulation
    )
    if scheduler is None and hasattr(adapter, "configure_scheduler"):
        scheduler = adapter.configure_scheduler(optimizer, planned_steps)
    step, epoch_start, position = 0, 0, 0
    training_state = {"best_score": None, "last_validation_step": None, "sample_exposures": 0}
    model = (
        DDP(
            adapter,
            device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            find_unused_parameters=getattr(adapter, "ddp_find_unused_parameters", True),
            broadcast_buffers=False,
        )
        if runtime.distributed
        else adapter
    )
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, adapter, runtime, optimizer, scheduler)
        step, epoch_start, position = (
            checkpoint["step"],
            checkpoint["epoch"],
            checkpoint["batch_in_epoch"],
        )
        training_state.update(checkpoint.get("training_state", {}))
    max_steps = settings.get("max_steps")
    epochs = int(
        settings.get(
            "epochs", ((max_steps or 1) * accumulation + len(loader) - 1) // len(loader) + 1
        )
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    check_storage(output, settings.get("minimum_free_gib", 0))
    if runtime.is_main:
        (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    model.train()
    validation_every = int(settings.get("validation_every_epochs", 0))

    def validate_and_save(epoch, position):
        if training_state["last_validation_step"] == step:
            return
        result = validate_depth(adapter, config, runtime, args.synthetic)
        training_state["last_validation_step"] = step
        score = result["selection_score"]
        best = training_state["best_score"]
        improved = score is not None and (best is None or score < best)
        if improved:
            training_state["best_score"] = score
            save_checkpoint(output / "best.pt", adapter, optimizer, scheduler, config,
                            runtime, step, epoch, position, training_state)
        if runtime.is_main:
            append_record(output, dict(result, step=step, epoch=epoch, selected=improved))

    stop = max_steps is not None and step >= max_steps
    final_epoch, final_position = epoch_start, position
    for epoch in range(epoch_start, epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        if hasattr(adapter, "set_epoch"):
            adapter.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(position):
            next(iterator)
        optimizer.zero_grad(set_to_none=True)
        while position < len(loader):
            count = min(accumulation, len(loader) - position)
            total_value = 0.0
            local_samples = 0
            for micro in range(count):
                batch = move(next(iterator), runtime.device, non_blocking=True)
                inputs, targets = batch["inputs"], batch["targets"]
                local_samples += len(inputs["rgb"])
                if hasattr(adapter, "set_training_step"):
                    adapter.set_training_step(step)
                if hasattr(adapter, "prepare_training_inputs"):
                    inputs = adapter.prepare_training_inputs(inputs, targets)
                sync = (
                    model.no_sync() if runtime.distributed and micro + 1 < count else nullcontext()
                )
                with sync:
                    if getattr(adapter, "requires_training_targets", False):
                        prediction = model(inputs, targets=targets)
                    else:
                        prediction = model(inputs)
                    losses = adapter.loss(prediction, inputs, targets)
                    loss = losses["total"] / count
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"非有限训练损失，epoch={epoch}, step={step}")
                    loss.backward()
                    total_value += float(loss.detach())
                position += 1
            gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("反向传播没有梯度或出现 NaN/Inf")
            if settings.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            interval = settings.get(
                "scheduler_interval", "epoch" if config["method"] == "fast_mvsnet" else "step"
            )
            if scheduler is not None and interval == "step":
                scheduler.step()
            step += 1
            samples = torch.tensor(local_samples, device=runtime.device, dtype=torch.long)
            if runtime.distributed:
                dist.all_reduce(samples)
            training_state["sample_exposures"] += int(samples)
            if runtime.is_main:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_in_epoch": position,
                    "loss": total_value,
                    "lr": [group["lr"] for group in optimizer.param_groups],
                    "synthetic": args.synthetic,
                    "sample_exposures": training_state["sample_exposures"],
                }
                with (output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if position == len(loader):
                if scheduler is not None and interval == "epoch":
                    scheduler.step()
                if validation_every and (epoch + 1) % validation_every == 0:
                    validate_and_save(epoch + 1, 0)
                check_storage(output, settings.get("minimum_free_gib", 0))
                save_checkpoint(
                    output / "last.pt",
                    adapter,
                    optimizer,
                    scheduler,
                    config,
                    runtime,
                    step,
                    epoch + 1,
                    0,
                    training_state,
                )
            elif step % settings.get("checkpoint_every", 1000) == 0 or (
                max_steps is not None and step >= max_steps
            ):
                save_checkpoint(
                    output / "last.pt",
                    adapter,
                    optimizer,
                    scheduler,
                    config,
                    runtime,
                    step,
                    epoch,
                    position,
                    training_state,
                )
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        final_epoch, final_position = (epoch + 1, 0) if position == len(loader) else (epoch, position)
        position = 0
    if step == 0:
        raise ValueError("训练没有执行任何更新")
    if validation_every:
        validate_and_save(final_epoch, final_position)
    check_storage(output, settings.get("minimum_free_gib", 0))
    for filename in ("last.pt", "final.pt"):
        save_checkpoint(output / filename, adapter, optimizer, scheduler, config, runtime,
                        step, final_epoch, final_position, training_state)
    return {
        "optimizer_steps": step,
        "checkpoint": str(output / "last.pt"),
        "synthetic": args.synthetic,
        "world_size": runtime.world_size,
        "method": config["method"],
        "validation_best": str(output / "best.pt") if training_state["best_score"] is not None else None,
        "final_checkpoint": str(output / "final.pt"),
        **training_state,
        "implementation_limitations": config["implementation_limitations"],
    }


class NetworkTimer:
    """挂钩最外层原生网络；适配、搬运、后处理时间单独进入 pipeline。"""

    def __init__(self, adapter, device):
        self.device, self.records, self.starts = device, [], []
        self.memory_records = []
        self.memory_starts = []
        self.previous_peaks = []
        native = getattr(adapter, "native", getattr(adapter, "native_model", None))
        self.available = isinstance(native, torch.nn.Module)
        self.handles = (
            [native.register_forward_pre_hook(self.begin), native.register_forward_hook(self.end)]
            if self.available
            else []
        )

    def begin(self, module, inputs):
        if self.device.type == "cuda":
            self.previous_peaks.append(torch.cuda.max_memory_allocated(self.device))
            self.memory_starts.append(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
        start = (
            torch.cuda.Event(enable_timing=True)
            if self.device.type == "cuda"
            else time.perf_counter()
        )
        if self.device.type == "cuda":
            start.record()
        self.starts.append(start)

    def end(self, module, inputs, output):
        start = self.starts.pop()
        end = (
            torch.cuda.Event(enable_timing=True)
            if self.device.type == "cuda"
            else time.perf_counter()
        )
        if self.device.type == "cuda":
            end.record()
        self.records.append((start, end))
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device)
            self.memory_records.append((peak, peak - self.memory_starts.pop()))

    def memory(self):
        if self.device.type != "cuda":
            return {
                "native_incremental_peak_bytes": None,
                "native_live_peak_bytes": None,
                "pipeline_peak_bytes": None,
            }
        result = {
            "native_incremental_peak_bytes": max((p[1] for p in self.memory_records), default=None),
            "native_live_peak_bytes": max((p[0] for p in self.memory_records), default=None),
            "pipeline_peak_bytes": max(
                self.previous_peaks
                + [p[0] for p in self.memory_records]
                + [torch.cuda.max_memory_allocated(self.device)]
            ),
        }
        self.memory_records.clear()
        self.previous_peaks.clear()
        return result

    def total_ms(self):
        if not self.available:
            return None
        value = sum(
            a.elapsed_time(b) if self.device.type == "cuda" else (b - a) * 1000
            for a, b in self.records
        )
        self.records.clear()
        return value

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def inference(args, config, runtime):
    if runtime.world_size != 1:
        raise ValueError(
            "推理/测评分片请启动独立进程各自处理清单；此入口只用单卡，避免重复计入场景"
        )
    evaluate = args.command == "evaluate"
    split = args.split or ("val" if evaluate else "test")
    loader, _ = make_loader(config, split, runtime, args.synthetic, False, targets=evaluate)
    adapter = build_adapter(config).to(runtime.device).eval()
    config["implementation_limitations"] = list(getattr(adapter, "implementation_limitations", []))
    if args.checkpoint:
        load_checkpoint(args.checkpoint, adapter, runtime)
    elif not args.synthetic and not (
        config.get("model", {}).get("weights") or config.get("model", {}).get("checkpoint")
    ):
        raise ValueError(
            "真实推理/评测需要 --checkpoint 或 model.weights；合成测试须显式 --synthetic"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(config) if evaluate else None
    timer = NetworkTimer(adapter, runtime.device)
    records = []
    warmup = config["evaluation"].get("warmup", 2)
    for index, cpu_batch in enumerate(loader):
        if args.max_batches is not None and index >= args.max_batches:
            break
        if index == 0:
            warm = move(cpu_batch["inputs"], runtime.device)
            for _ in range(warmup):
                adapter(warm)
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            timer.records.clear()
            timer.memory_records.clear()
            timer.previous_peaks.clear()
            del warm
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            torch.cuda.reset_peak_memory_stats(runtime.device)
        start = time.perf_counter()
        inputs = move(cpu_batch["inputs"], runtime.device)
        prediction = adapter(inputs)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            native_peak = torch.cuda.max_memory_allocated(runtime.device)
        else:
            native_peak = None
        adapter_ms = (time.perf_counter() - start) * 1000
        network_ms = timer.total_ms()
        converted = postprocess(prediction, inputs, config)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        pipeline_ms = (time.perf_counter() - start) * 1000
        records.append(
            {
                "batch": index,
                "batch_size": len(inputs["rgb"]),
                "native_network_ms": network_ms,
                "input_transfer_and_adapter_ms": adapter_ms,
                "pipeline_ms": pipeline_ms,
                "network_stage_peak_bytes": native_peak,
                **timer.memory(),
                "reference_indices": prediction.get("reference_indices"),
                "includes_all_references": True,
            }
        )
        if evaluate:
            targets = move(cpu_batch["targets"], runtime.device)
            evaluator.update(prediction, converted, inputs, targets)
            del targets  # 标签不能占用下一帧网络计时区间的显存。
        # 原生输出与独立后处理分开保存；I/O 不计入网络/计算管线时间。
        if args.save_predictions or not evaluate:
            export_prediction(
                output / f"prediction_{index:06d}",
                prediction,
                converted,
                inputs,
                cpu_batch["metadata"],
                config,
            )
        del prediction, converted, inputs
    timer.close()
    return {
        "method": config["method"],
        "synthetic": args.synthetic,
        "labels_loaded": evaluate,
        "implementation_limitations": config["implementation_limitations"],
        "capabilities": getattr(adapter, "capabilities", {}),
        "metrics": evaluator.result() if evaluator is not None else {},
        "geometry_units": config.get("data", {}).get("geometry_units", "meters"),
        "timing": records,
        "warmup_batches": warmup,
        "timing_scope": "网络本体钩子汇总所有参考视角；完整计算管线包括搬运、适配、所有网络调用和后处理，不含磁盘I/O与GT指标计算",
        "precision_claim": (
            "合成模式只验证运行，不代表重建性能或论文指标"
            if args.synthetic
            else "需要结合完整数据/预训练/协议记录解读"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "infer", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="official_methods/artifacts/run")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["_loading_shared_checkpoint"] = bool(args.checkpoint)
    if config["method"] == "gaussianformer2":
        if args.synthetic:
            config.setdefault("model", {})["synthetic_random_init"] = True
        elif config.get("model", {}).get("synthetic_random_init"):
            raise ValueError(
                "synthetic_random_init 只允许显式 --synthetic；真实 GF2 实验必须明确初始化器或完整权重"
            )
    if (
        config.get("profile") == "official"
        or config.get("reference_protocol", {}).get("reference_only")
        or config.get("protocol") == "official_reference"
    ):
        raise ValueError(
            "此配置保存官方原实验协议；共享入口固定同步四目，请使用 *_carla.json。原数据集流程入口见各方法中文文档。"
        )
    if args.command != "train" and "reference_indices" in config["evaluation"]:
        config.setdefault("model", {})["reference_indices"] = config["evaluation"][
            "reference_indices"
        ]
    if not args.synthetic:
        data = config["data"]
        split = (
            "train"
            if args.command == "train"
            else args.split or ("val" if args.command == "evaluate" else "test")
        )
        if not data.get("root") or not data.get(f"{split}_manifest"):
            raise ValueError(
                f"真实 {args.command} 需要非空 data.root 和 {split}_manifest；不自动使用合成数据"
            )
        audit_split_sequences(config)
    torch.set_num_threads(config.get("cpu_threads", 4))
    runtime = setup_runtime(args.device)
    try:
        seed_all(config["seed"] + runtime.rank)
        result = (
            train(args, config, runtime)
            if args.command == "train"
            else inference(args, config, runtime)
        )
        if runtime.is_main:
            path = Path(args.output) / f"{args.command}_report.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            print(f"结果已写入 {path}", flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
