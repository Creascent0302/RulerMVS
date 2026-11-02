"""八卡多域分阶段训练；整轮覆盖、独立验证、自动阶段衔接和恢复。"""

import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from jar.runtime import (capture_rng, cleanup_runtime, move, precision_context,
                         restore_rng, seed_all, setup_runtime)
from .data import MixedDataset, FullPassSampler, collate_mixed
from .losses import depth_objective
from .model import PretrainingModel, load_pretraining_state


def emit(path, record, runtime):
    if runtime.is_main:
        record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def synchronize_action(runtime, action):
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
        raise RuntimeError(error)


def atomic_save(record, destination, minimum_free_gib):
    path = Path(destination)
    # 保留最终200GiB，并额外为临时checkpoint预留4GiB。
    if shutil.disk_usage(path.parent).free < (minimum_free_gib + 4) * 2 ** 30:
        raise RuntimeError("数据盘剩余空间低于checkpoint安全阈值，停止写入和训练")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


def save_training(path, model, optimizer, state, config, runtime):
    rng = capture_rng(runtime)
    rngs = [None] * runtime.world_size
    if runtime.distributed:
        dist.all_gather_object(rngs, rng)
    else:
        rngs[0] = rng
    record = {"architecture": model.architecture, "model": model.state_dict(),
              "optimizer": optimizer.state_dict() if optimizer else None,
              "state": dict(state), "config": config,
              "world_size": runtime.world_size, "rng_by_rank": rngs}
    synchronize_action(runtime, lambda: atomic_save(record, path, config["train"]["minimum_free_gib"]))


def load_incumbent(path, model, config):
    """守门权重与训练恢复点分开；恢复时不允许换配置或换架构。"""
    record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if record["config"] != config or record["architecture"] != model.architecture:
        raise ValueError("incumbent须使用相同配置和架构")
    return record


def export_new_best(source, destination, minimum_free_gib):
    """只导出已保存的本轮最佳参数，不改变仍持有最后参数的训练模型。"""
    record = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    record = dict(record, selected="best_full_validation_this_run")
    atomic_save(record, destination, minimum_free_gib)


def optimizer_for(model, stage, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = "foundation" if name.startswith("foundation.") else "matching" if name.startswith("matching.backbone.") else "new"
        decay = 0. if parameter.ndim <= 1 or name.endswith("bias") else config["train"]["weight_decay"]
        key = (kind, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        {"params": parameters, "lr": stage[kind + "_lr"], "initial_lr": stage[kind + "_lr"],
         "weight_decay": decay, "group_name": kind}
        for (kind, decay), parameters in groups.items()
    ])


@torch.no_grad()
def evaluate(model, dataset, config, runtime, limit=None, split="val"):
    model.eval()
    results = []
    for domain_index, domain in enumerate(dataset.domains):
        length = len(dataset.datasets[domain_index])
        indices = list(range(length)) if limit is None else torch.linspace(0, length - 1, min(length, limit)).long().tolist()
        keys = [(domain_index, index, -1) for index in indices[runtime.rank::runtime.world_size]]
        loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                            num_workers=0, pin_memory=True)
        sums = torch.zeros(8, dtype=torch.float64, device=runtime.device)
        for batch in loader:
            batch = move(batch, runtime.device)
            inputs, targets = batch["inputs"], batch["targets"]
            with precision_context(runtime.device, config["train"]["precision"]):
                result = model(inputs, domain)
            estimate = result["depth"].float() * domain["depth_scale"]
            truth = targets["depth"][:, :1].float()
            valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1] & torch.isfinite(truth) & (truth > 0)
            if domain.get("eval_range"):
                valid &= (truth >= domain["eval_range"][0]) & (truth <= domain["eval_range"][1])
            if not bool(torch.isfinite(estimate[valid]).all()):
                raise RuntimeError(f"{domain['name']}验证预测非有限")
            error = (estimate - truth).abs()[valid].double()
            ratio = torch.maximum(estimate[valid] / truth[valid], truth[valid] / estimate[valid].clamp_min(1e-8))
            sums[0] += len(error)
            sums[1] += error.sum()
            sums[2] += error.square().sum()
            sums[3] += (error / truth[valid]).sum()
            sums[4] += (ratio < 1.25).sum()
            sums[5] += result["valid"][valid].sum()
            sums[6] += ((truth[valid] < domain["near"]) | (truth[valid] > domain["far"])).sum()
            sums[7] += 1
        if runtime.distributed:
            dist.all_reduce(sums)
        count = float(sums[0])
        if count <= 0:
            raise RuntimeError(f"{domain['name']}/{split}验证没有有效GT")
        results.append({"dataset": domain["name"], "units": domain["geometry_units"], "frames": int(sums[7]),
                        "pixels": int(count), "mae": float(sums[1] / count),
                        "rmse": math.sqrt(float(sums[2] / count)), "abs_rel": float(sums[3] / count),
                        "delta1": float(sums[4] / count), "coverage": float(sums[5] / count),
                        "outside_candidate_range": float(sums[6] / count)})
    model.train()
    return {"split": split, "full": limit is None, "domains": results,
            "macro_abs_rel": sum(r["abs_rel"] for r in results) / len(results)}


def run(config, resume=None, hooks=None):
    torch.set_num_threads(config["train"].get("threads", 2))
    runtime = setup_runtime("cuda", timeout_seconds=1800)
    try:
        _run(config, resume, runtime, hooks)
    finally:
        cleanup_runtime(runtime)


def _run(config, resume, runtime, hooks=None):
    # 独立研究入口可替换这五个策略；DDP、恢复、日志和存储仍共用原流程。
    model_factory = hooks.model_factory if hooks else PretrainingModel
    objective = hooks.objective if hooks else depth_objective
    sampler_factory = hooks.sampler_factory if hooks else FullPassSampler
    evaluation = hooks.evaluate if hooks else evaluate
    selection_score = hooks.selection_score if hooks else lambda result: result["macro_abs_rel"]
    make_optimizer = getattr(hooks, "optimizer_factory", optimizer_for)
    prepare_batch = getattr(hooks, "prepare_batch", None)
    retain_global_best = config["train"].get("stage_policy") == "retain_global_best"
    continuous_new = config["train"].get("stage_policy") == "continuous_new"
    output = Path(config["output"])
    synchronize_action(runtime, lambda: output.mkdir(parents=True, exist_ok=True))
    if continuous_new and not resume and any((output / name).exists() for name in
                                            ("last.pt", "best.pt", "final.pt", "pretrained.pt")):
        raise FileExistsError("独立重训目录已有权重，请使用新目录或显式resume，禁止覆盖")
    log = output / "events.jsonl"
    seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = config["train"].get("allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = config["train"].get("allow_tf32", True)
    record = torch.load(resume, map_location="cpu", weights_only=False, mmap=True) if resume else None
    if record and (record["config"] != config or record["world_size"] != runtime.world_size):
        raise ValueError("恢复须使用相同配置、划分和world_size")
    for domain in config["datasets"]:
        actual = hashlib.sha256(Path(domain["manifests"]["train"]).read_bytes()).hexdigest()
        if actual != domain["train_manifest_sha256"]:
            raise ValueError(f"训练清单改变: {domain['name']}")
    warm_start = config.get("initial_model_checkpoint")
    if retain_global_best and not warm_start:
        raise ValueError("retain_global_best需要initial_model_checkpoint作为原始基准")
    model = model_factory(config, initialize=record is None and not warm_start).to(runtime.device)
    if record:
        model.load_state_dict(record["model"], strict=True)
    elif warm_start:
        initial = torch.load(warm_start, map_location="cpu", weights_only=False, mmap=True)
        if initial.get("architecture") != model.architecture:
            raise ValueError("阶段迁移需要同一自研架构的完整checkpoint")
        load_pretraining_state(model, initial["model"])
        del initial
    state = dict(stage_index=0, epoch_in_stage=0, update_in_epoch=0, global_epoch=0, global_step=0,
                 best_subset=float("inf"), best_full=float("inf"), stale_full=0, exposures=0,
                 full_evaluated_in_stage=False, sampler_plan_logged=False)
    if retain_global_best:
        state.update(global_best_score=0., stage_improved=False, previous_stage_improved=False)
    if continuous_new:
        state.update(global_best_score=float("inf"), best_global_step=None)
    if record:
        state.update(record["state"])
    new_best_path = output / "best.pt"
    if continuous_new and record:
        if new_best_path.exists():
            best = load_incumbent(new_best_path, model, config)
            state.update(global_best_score=best["selection_score"],
                         best_global_step=best["global_step"])
            del best
        elif math.isfinite(state["global_best_score"]):
            raise FileNotFoundError("恢复点记录了本轮最佳，但best.pt缺失")
    incumbent_path = output / "incumbent.pt"
    if retain_global_best:
        if record:
            incumbent = load_incumbent(incumbent_path, model, config)
            # 若保存incumbent后、保存last前中断，以独立的守门文件为准。
            state["global_best_score"] = incumbent["selection_score"]
            if incumbent.get("stage_index") == state["stage_index"]:
                state["stage_improved"] = True
            del incumbent
        else:
            incumbent = {"architecture": model.architecture, "model": model.state_dict(),
                         "config": config, "selection_score": 0., "qualified": False,
                         "stage_index": -1, "global_step": 0}
            synchronize_action(runtime, lambda: atomic_save(
                incumbent, incumbent_path, config["train"]["minimum_free_gib"]))
            del incumbent
    synchronize_action(runtime, lambda: (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n"))
    train = MixedDataset(config, "train")
    validation_config = dict(config, datasets=config.get("validation_datasets", config["datasets"]))
    validation = MixedDataset(validation_config, "val")
    if record:
        # 阶段交界的checkpoint没有optimizer，也必须恢复随机状态。
        restore_rng(record["rng_by_rank"][runtime.rank], runtime)
    emit(log, {"event": "start", "world_size": runtime.world_size, "train_groups": len(train),
               "resume": str(resume) if resume else None, "config": str(output / "config.json")}, runtime)
    if config.get("zero_shot_before_training") and not resume:
        initial_test = MixedDataset(config, "test")
        result = evaluation(model, initial_test, config, runtime, split="test")
        emit(log, {"event": "zero_shot_test", **result}, runtime)
        del initial_test
    for stage_index, stage in enumerate(config["stages"]):
        if stage_index < state["stage_index"]:
            continue
        continuing = (record is not None and record["optimizer"] is not None
                      and stage_index == state["stage_index"])
        if not continuing:
            state.update(stage_index=stage_index, epoch_in_stage=0, update_in_epoch=0,
                         best_subset=float("inf"), best_full=float("inf"), stale_full=0,
                         full_evaluated_in_stage=False, sampler_plan_logged=False)
            if retain_global_best:
                state["stage_improved"] = False
                if stage.get("require_previous_improvement") and not state["previous_stage_improved"]:
                    emit(log, {"event": "stage_skipped", "stage": stage["number"],
                               "reason": "previous_stage_did_not_improve_global_incumbent"}, runtime)
                    state.update(stage_index=stage_index + 1, previous_stage_improved=False)
                    save_training(output / "last.pt", model, None, state, config, runtime)
                    record = None
                    continue
        model.set_stage(stage["number"])
        optimizer = make_optimizer(model, stage, config)
        parallel = DistributedDataParallel(model, device_ids=[runtime.local_rank], broadcast_buffers=False,
                                            find_unused_parameters=True) if runtime.distributed else model
        if continuing:
            optimizer.load_state_dict(record["optimizer"])
            restore_rng(record["rng_by_rank"][runtime.rank], runtime)
            record = None
        emit(log, {"event": "stage_start", "stage": stage["number"], "name": stage["name"],
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "max_epochs": stage["max_epochs"]}, runtime)
        stage_best = output / f"stage{stage['number']}_best.pt"
        for epoch in range(state["epoch_in_stage"], stage["max_epochs"]):
            sampler = sampler_factory([len(d) for d in train.datasets], state["global_epoch"], runtime.rank,
                                      runtime.world_size, config["train"]["batch_size"], config["train"]["accumulation"],
                                      config["seed"], state["update_in_epoch"])
            if config.get("smoke"):
                # 每阶段明确覆盖全部域，不能靠随机前7步碰巧覆盖。
                sampler.updates = [next(item for item in sampler.updates if item[0] == i)
                                   for i in range(len(train.domains))]
            total_updates = len(sampler.updates)
            offset = state["update_in_epoch"]
            if not state["sampler_plan_logged"]:
                domain_counts = {domain["name"]: 0 for domain in train.domains}
                for domain_index, indices in sampler.updates:
                    domain_counts[train.domains[domain_index]["name"]] += len(indices)
                emit(log, {"event": "sampling_plan", "stage": stage["number"], "epoch": epoch + 1,
                           "domain_counts": domain_counts, "updates_in_epoch": total_updates,
                           "global_exposures": sum(domain_counts.values()),
                           "resume_update": offset}, runtime)
                state["sampler_plan_logged"] = True
            loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_mixed,
                                num_workers=config["train"]["workers"], pin_memory=True,
                                generator=torch.Generator().manual_seed(config["seed"] + state["global_epoch"]))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            begin = time.monotonic()
            accumulator = {}
            for micro, batch in enumerate(loader):
                batch = move(batch, runtime.device, non_blocking=True)
                domain = train.domains[batch["domain_index"]]
                if prepare_batch is not None:
                    batch, domain = prepare_batch(batch, domain)
                accumulation = config["train"]["accumulation"]
                boundary = (micro + 1) % accumulation == 0
                context = parallel.no_sync() if runtime.distributed and not boundary else contextlib.nullcontext()
                with context:
                    with precision_context(runtime.device, config["train"]["precision"]):
                        prediction = parallel(batch["inputs"], domain)
                    with torch.autocast(device_type="cuda", enabled=False):
                        losses = objective(prediction, batch["inputs"], batch["targets"], domain,
                                                 config.get("loss", {}), stage["correspondence_weight"])
                    finite = torch.tensor(int(torch.isfinite(losses["total"])), device=runtime.device)
                    if runtime.distributed:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not bool(finite):
                        raise RuntimeError(f"非有限loss: {domain['name']}")
                    (losses["total"] / accumulation).backward()
                for key, value in losses.items():
                    accumulator[key] = accumulator.get(key, 0.) + float(value.detach()) / accumulation
                del prediction, losses, batch
                if not boundary:
                    continue
                update = offset + (micro + 1) // accumulation
                progress = epoch * total_updates + update
                maximum = stage["max_epochs"] * total_updates
                warmup = min(500, max(1, round(.05 * maximum)))
                factor = progress / warmup if progress <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (progress - warmup) / max(1, maximum - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.update(update_in_epoch=update, global_step=state["global_step"] + 1)
                state["exposures"] += runtime.world_size * config["train"]["batch_size"] * accumulation
                if update == 1 or state["global_step"] % config["train"]["log_every"] == 0 or config.get("smoke"):
                    values = torch.tensor(list(accumulator.values()), device=runtime.device)
                    if runtime.distributed:
                        dist.all_reduce(values)
                        values /= runtime.world_size
                    averaged = dict(zip(accumulator, values.tolist()))
                    emit(log, {"event": "update", "stage": stage["number"], "epoch": epoch + 1,
                               "update": update, "updates_in_epoch": total_updates, "global_step": state["global_step"],
                               "dataset": domain["name"], "loss": averaged, "grad_norm": float(norm),
                               "seconds_per_update": (time.monotonic() - begin) / ((micro + 1) / accumulation),
                               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                               "exposures": state["exposures"]}, runtime)
                accumulator = {}
                if state["global_step"] == 1 or state["global_step"] % config["train"]["checkpoint_every"] == 0:
                    save_training(output / "last.pt", model, optimizer, state, config, runtime)
                if config.get("stop_after_updates") and state["global_step"] >= config["stop_after_updates"]:
                    save_training(output / "last.pt", model, optimizer, state, config, runtime)
                    emit(log, {"event": "requested_short_run_complete"}, runtime)
                    return
            state.update(epoch_in_stage=epoch + 1, update_in_epoch=0, global_epoch=state["global_epoch"] + 1)
            limit = config["train"]["validation_samples_per_domain"]
            result = evaluation(model, validation, config, runtime, limit=limit)
            emit(log, {"event": "validation", "stage": stage["number"], "epoch": epoch + 1, **result}, runtime)
            score = selection_score(result)
            if score < state["best_subset"]:
                state["best_subset"] = score
                best = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
                        "validation": result, "stage": stage["number"], "epoch": epoch + 1}
                synchronize_action(runtime, lambda: atomic_save(best, stage_best, config["train"]["minimum_free_gib"]))
            full = not config.get("smoke") and ((epoch + 1) % config["train"]["full_validation_every"] == 0 or epoch + 1 == stage["max_epochs"])
            if full:
                result = evaluation(model, validation, config, runtime)
                state["full_evaluated_in_stage"] = True
                emit(log, {"event": "validation", "stage": stage["number"], "epoch": epoch + 1, **result}, runtime)
                score = selection_score(result)
                if score < state["best_full"]:
                    state["best_full"], state["stale_full"] = score, 0
                    best = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
                            "validation": result, "stage": stage["number"], "epoch": epoch + 1}
                    synchronize_action(runtime, lambda: atomic_save(best, output / f"stage{stage['number']}_full_best.pt", config["train"]["minimum_free_gib"]))
                else:
                    state["stale_full"] += 1
                if retain_global_best and math.isfinite(score) and score < state["global_best_score"]:
                    state["global_best_score"] = score
                    state["stage_improved"] = True
                    incumbent = {"architecture": model.architecture, "model": model.state_dict(),
                                 "config": config, "validation": result, "selection_score": score,
                                 "qualified": score < 0., "stage_index": stage_index,
                                 "stage": stage["number"], "epoch": epoch + 1,
                                 "global_step": state["global_step"]}
                    synchronize_action(runtime, lambda: atomic_save(
                        incumbent, incumbent_path, config["train"]["minimum_free_gib"]))
                    emit(log, {"event": "incumbent_updated", "stage": stage["number"],
                               "epoch": epoch + 1, "selection_score": score}, runtime)
                    del incumbent
                if continuous_new and math.isfinite(score) and score < state["global_best_score"]:
                    state.update(global_best_score=score, best_global_step=state["global_step"])
                    best = {"architecture": model.architecture, "model": model.state_dict(),
                            "config": config, "validation": result, "selection_score": score,
                            "selected": "best_full_validation_this_run", "stage_index": stage_index,
                            "stage": stage["number"], "epoch": epoch + 1,
                            "global_step": state["global_step"]}
                    synchronize_action(runtime, lambda: atomic_save(
                        best, new_best_path, config["train"]["minimum_free_gib"]))
                    emit(log, {"event": "best_updated", "stage": stage["number"],
                               "epoch": epoch + 1, "selection_score": score,
                               "global_step": state["global_step"]}, runtime)
                    del best
            save_training(output / "last.pt", model, optimizer, state, config, runtime)
            if epoch + 1 >= stage["min_epochs"] and state["stale_full"] >= stage.get("patience_full_validations", 1000):
                emit(log, {"event": "early_stop", "stage": stage["number"]}, runtime)
                break
        del parallel, optimizer
        if continuous_new:
            selected = output / f"stage{stage['number']}_last.pt"
            stage_last = {"architecture": model.architecture, "model": model.state_dict(),
                          "config": config, "stage": stage["number"],
                          "epoch": state["epoch_in_stage"], "global_step": state["global_step"],
                          "selected": "last_parameters_no_rollback"}
            synchronize_action(runtime, lambda: atomic_save(
                stage_last, selected, config["train"]["minimum_free_gib"]))
            del stage_last
        elif retain_global_best:
            selected = incumbent_path
            best = load_incumbent(selected, model, config)
            model.load_state_dict(best["model"], strict=True)
            state["global_best_score"] = best["selection_score"]
            state["previous_stage_improved"] = state["stage_improved"]
            del best
        else:
            preferred = output / f"stage{stage['number']}_full_best.pt"
            selected = preferred if preferred.exists() else stage_best
            if hooks and state["full_evaluated_in_stage"] and not preferred.exists():
                selected = None
            if selected is not None and selected.exists():
                best = torch.load(selected, map_location="cpu", weights_only=False, mmap=True)
                model.load_state_dict(best["model"], strict=True)
                del best
            elif hooks:
                # 新结构可能需要完整联合训练；没有守门通过者仍可继续，但不称合格模型。
                selected = None
            else:
                raise FileNotFoundError(selected)
        emit(log, {"event": "stage_complete", "stage": stage["number"],
                   "selected": str(selected) if selected else "last_unqualified",
                   "qualified": (None if continuous_new else state["global_best_score"] < 0.
                                 if retain_global_best else selected is not None),
                   **({"selection": "last_parameters_no_rollback",
                       "global_best_score": state["global_best_score"],
                       "best_global_step": state["best_global_step"]} if continuous_new else {}),
                   **({"global_best_score": state["global_best_score"],
                       "stage_improved": state["stage_improved"]} if retain_global_best else {})}, runtime)
        state.update(stage_index=stage_index + 1, epoch_in_stage=0, update_in_epoch=0)
        save_training(output / "last.pt", model, None, state, config, runtime)
        record = None
    if retain_global_best:
        best = load_incumbent(incumbent_path, model, config)
        model.load_state_dict(best["model"], strict=True)
        state["global_best_score"] = best["selection_score"]
        del best
    final = {"architecture": model.architecture, "model": model.state_dict(), "config": config}
    if continuous_new:
        final.update(global_step=state["global_step"], selected="last_parameters_no_rollback",
                     best_selection_score=state["global_best_score"],
                     best_global_step=state["best_global_step"])
        synchronize_action(runtime, lambda: atomic_save(
            final, output / "final.pt", config["train"]["minimum_free_gib"]))
        if new_best_path.exists():
            synchronize_action(runtime, lambda: export_new_best(
                new_best_path, output / "pretrained.pt", config["train"]["minimum_free_gib"]))
        elif config.get("smoke"):
            emit(log, {"event": "no_full_validation_best", "reason": "smoke_only_final_saved"}, runtime)
        else:
            raise RuntimeError("本轮未获得有限完整VAL成绩，已保留final.pt但不导出best/pretrained")
    elif retain_global_best:
        final.update(promotion_eligible=state["global_best_score"] < 0.,
                     selection_score=state["global_best_score"])
    elif hooks:
        final["promotion_eligible"] = math.isfinite(state["best_full"])
    if not continuous_new:
        synchronize_action(runtime, lambda: atomic_save(final, output / "pretrained.pt", config["train"]["minimum_free_gib"]))
    if config["evaluation"].get("test_after_training") and not config.get("smoke"):
        test = MixedDataset(config, "test")
        result = evaluation(model, test, config, runtime, split="test")
        emit(log, {"event": "final_test", **result,
                   **({"checkpoint": "final.pt", "global_step": state["global_step"]}
                      if continuous_new else {})}, runtime)
    emit(log, {"event": "complete", "free_gib": shutil.disk_usage(output).free / 2 ** 30}, runtime)
    if config.get("transfer") and not config.get("smoke"):
        # 公共阶段与CARLA微调分开保存，公共预训练权重不会被覆盖。
        transfer = config["transfer"]
        following = copy.deepcopy(config)
        following.pop("transfer")
        following["datasets"] = transfer["datasets"]
        following["stages"] = transfer["stages"]
        following["output"] = str(output / "carla_finetune")
        following["initial_model_checkpoint"] = str(output / "pretrained.pt")
        following["zero_shot_before_training"] = True
        following["train"]["full_validation_every"] = 1
        following["train"]["validation_samples_per_domain"] = transfer.get("validation_samples_per_domain", 25)
        del model, final, train, validation
        torch.cuda.empty_cache()
        transfer_resume = Path(following["output"]) / "last.pt"
        _run(following, str(transfer_resume) if transfer_resume.exists() else None, runtime, hooks)
