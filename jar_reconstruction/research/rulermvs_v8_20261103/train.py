"""V6/V8独立训练入口；严格暖启动、完整初值验证及可恢复DDP。

模型与增强由配置决定，旧V6训练入口不变。仅完整VAL选best，guard只另存
通过预登记容差的best_guarded，从不回滚正在训练的参数。
"""

import argparse
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import collate_mixed
from jar.runtime import (
    all_ranks_true, cleanup_runtime, move, precision_context, restore_rng,
    seed_all, setup_runtime,
)
from research.overnight_v6_v7_mvs_20261103.train import (
    emit, sampler_for, save, stage_for, validate,
)
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective as v6_objective
from research.rulermvs_v6_20261101.training import optimizer_for
from tools.smoke_outdoor_datasets import offline_only


def build_model(config, initialize=False):
    if config["method"] == "v6":
        return V6Model(config, initialize)
    if config["method"] == "v8":
        from research.rulermvs_v8_20261103.model import V8Model
        return V8Model(config, initialize)
    raise ValueError("method须为v6或v8")


def progress_state(state):
    """日志保留进度标量；完整初始VAL/清单哈希只随checkpoint保存。"""
    return {key: value for key, value in state.items()
            if key not in {"initial_validation", "manifest_sha256", "guard_reference"}}


def training_objective(method):
    if method == "v8":
        from research.rulermvs_v8_20261103.objective import objective
        return objective
    return v6_objective


def write_distributed_json(path, value, runtime):
    error = [None]
    if runtime.is_main:
        try:
            write_json(path, value)
        except Exception as exception:
            error[0] = repr(exception)
    if runtime.distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def group_gradient_norms(optimizer):
    """返回裁剪前各语义参数组的L2 norm；合并相同组的decay子组。"""
    squared = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("group_name", f"group_{index}")
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared[name] = squared.get(name, 0.) + value
    return {name: value.sqrt() for name, value in squared.items()}


def baseline_guard(result, initial, settings):
    """只记录初值保持，阈值为非负相对容差，例如full.mae=0.02。

    schema: {enabled:true, reference_checkpoint:"V6/best.pt", relative_tolerance:.03,
    relative_tolerances:{full:{mae:.03,rmse:.03},near:{...}}, absolute_tolerance:0.}
    各域使用同一预登记阈值；空分区跳过，分母/域变化直接拒绝。
    """
    if not settings or not settings.get("enabled", True):
        return {"enabled": False}
    if not initial.get("full") or not result.get("full"):
        return {"enabled": True, "eligible": False, "passed": False,
                "reason": "初值与候选都必须使用完整VAL"}
    relative = float(settings.get("relative_tolerance", .03))
    tolerances = settings.get("relative_tolerances", {
        region: {metric: relative for metric in ("mae", "rmse")}
        for region in ("full", "near", "mid", "far")})
    if not isinstance(tolerances, dict) or not tolerances:
        raise ValueError("baseline_guard须显式设置relative_tolerances")
    absolute = float(settings.get("absolute_tolerance", 0.))
    if not math.isfinite(absolute) or absolute < 0:
        raise ValueError("guard absolute_tolerance必须有限非负")
    before = {row["dataset"]: row for row in initial["domains"]}
    after = {row["dataset"]: row for row in result["domains"]}
    if set(before) != set(after):
        raise ValueError("guard比较的VAL域不一致")
    aggregate_tolerance = float(settings.get("aggregate_relative_tolerance", relative))
    if not math.isfinite(aggregate_tolerance) or aggregate_tolerance < 0:
        raise ValueError("guard aggregate_relative_tolerance必须有限非负")
    aggregate_value = float(result["selection_risk"])
    aggregate_limit = float(initial["selection_risk"]) * (1 + aggregate_tolerance)
    checks = [dict(dataset="__aggregate__", region="all", metric="selection_risk",
                   initial=float(initial["selection_risk"]), value=aggregate_value,
                   limit=aggregate_limit,
                   passed=math.isfinite(aggregate_value) and aggregate_value <= aggregate_limit)]
    for name, reference in before.items():
        current = after[name]
        if reference["frames"] != current["frames"] or reference["units"] != current["units"]:
            raise ValueError("guard比较的VAL帧数或单位不一致")
        for region, metrics in tolerances.items():
            if region not in {"full", "near", "mid", "far"}:
                raise ValueError(f"guard未知分区: {region}")
            old, new = reference[region], current[region]
            if old["pixels"] != new["pixels"]:
                raise ValueError("guard比较的GT像素分母不一致")
            for metric, tolerance in metrics.items():
                if metric not in {"mae", "rmse", "abs_rel"}:
                    raise ValueError(f"guard未知误差指标: {metric}")
                tolerance = float(tolerance)
                if not math.isfinite(tolerance) or tolerance < 0:
                    raise ValueError("guard相对容差必须有限非负")
                if not old["pixels"]:
                    continue
                limit = float(old[metric]) * (1 + tolerance) + absolute
                value = float(new[metric])
                checks.append(dict(dataset=name, region=region, metric=metric,
                                   initial=float(old[metric]), value=value, limit=limit,
                                   passed=math.isfinite(value) and value <= limit))
    if not checks:
        raise ValueError("guard没有有效比较项")
    return dict(enabled=True, eligible=True, passed=all(row["passed"] for row in checks),
                checks=checks, failures=sum(not row["passed"] for row in checks))


def check_config(config):
    if config.get("method") not in {"v6", "v8"}:
        raise ValueError("只接受v6/v8")
    if config["train"].get("stage_policy") != "continuous_new":
        raise ValueError("必须使用continuous_new，不回滚训练参数")
    if config.get("teacher_checkpoint"):
        raise ValueError("本入口不使用教师")
    if config.get("training_recipe", "generalization") not in {"legacy", "generalization"}:
        raise ValueError("training_recipe须为legacy/generalization")
    for key in ("batch_size", "accumulation", "global_batch_size", "log_every",
                "checkpoint_every", "full_validation_every"):
        if int(config["train"][key]) <= 0:
            raise ValueError(f"train.{key}必须为正整数")
    for key in ("initial_validation_limit", "validation_limit"):
        limit = config["train"].get(key)
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError(f"{key}须为正整数或null")
    if config["train"]["precision"] not in {"fp32", "bf16"}:
        raise ValueError("本入口支持fp32/bf16；未实现fp16 GradScaler")
    if float(config["train"].get("grad_clip", 1.)) <= 0:
        raise ValueError("grad_clip必须为正")


def run(config, resume=None):
    check_config(config)
    objective = training_objective(config["method"])
    recipe = config.get("training_recipe", "generalization")
    if recipe == "legacy":
        from research.overnight_v6_v7_mvs_20261103.common_data import (
            SharedDataset, own_training_domain as compose_training_domain,
        )
    else:
        from research.rulermvs_v8_20261103.data import SharedDataset, compose_training_domain
    offline_only()
    runtime = setup_runtime("cuda", timeout_seconds=config["train"].get("timeout_seconds", 1800))
    output = Path(config["output"])
    try:
        torch.set_num_threads(config["train"].get("threads", 2))
        torch.backends.cuda.matmul.allow_tf32 = bool(config["train"].get("allow_tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(config["train"].get("allow_tf32", False))
        torch.backends.cudnn.benchmark = False
        seed_all(config["seed"])
        startup = [None, None]
        if runtime.is_main:
            try:
                output.mkdir(parents=True, exist_ok=True)
                if not resume:
                    occupied = [name for name in ("last.pt", "best.pt", "best_guarded.pt", "final.pt",
                                                  "initial_validation.json", "events.jsonl")
                                if (output / name).exists()]
                    if occupied:
                        raise FileExistsError(f"输出已有实验产物，须新目录或显式resume: {occupied}")
                    initial_path = Path(config["initial_model_checkpoint"])
                    if not initial_path.is_file():
                        raise FileNotFoundError("须提供已存在的initial_model_checkpoint严格暖启动")
                    if initial_path.resolve().parent == output.resolve():
                        raise ValueError("暖启动源与输出目录必须隔离，禁止覆盖旧V6实验")
                audit = {}
                for domain in config["datasets"]:
                    checksums = {split: digest(path) for split, path in domain["manifests"].items()}
                    if checksums["train"] != domain["train_manifest_sha256"]:
                        raise ValueError(f"{domain['name']} TRAIN清单已改变")
                    audit[domain["name"]] = checksums
                startup[1] = audit
            except Exception as exception:
                startup[0] = repr(exception)
        if runtime.distributed:
            dist.broadcast_object_list(startup, src=0)
        if startup[0]:
            raise RuntimeError(startup[0])
        audit = startup[1]
        batch_size, accumulation = config["train"]["batch_size"], config["train"]["accumulation"]
        global_batch = runtime.world_size * batch_size * accumulation
        if global_batch != config["train"]["global_batch_size"]:
            raise ValueError("显卡数、batch与累积不符合配置global_batch_size")
        training, validation = SharedDataset(config, "train"), SharedDataset(config, "val")
        first_sampler = sampler_for(training, config, runtime, 0)
        epoch_steps = len(first_sampler.updates)
        if epoch_steps <= 0:
            raise ValueError("训练采样器没有更新")
        epochs = sum(stage["epochs"] for stage in config["stages"])
        saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True) if resume else None
        if saved:
            prior, current = copy.deepcopy(saved["config"]), copy.deepcopy(config)
            for settings in (prior, current):
                settings.pop("output", None)
                settings["train"].pop("stop_after_updates", None)
            if prior != current:
                raise ValueError("严格恢复不允许改变数据、阶段、初始化或计算协议")
            if saved["state"].get("manifest_sha256") != audit:
                raise ValueError("恢复TRAIN/VAL清单SHA与原实验不同")
        model = build_model(config, initialize=False)
        state = dict(epoch=0, update=0, global_step=0, exposures=0, best_risk=1e30,
                     best_guarded_risk=1e30, manifest_sha256=audit)
        if saved:
            if saved["architecture"] != model.architecture:
                raise ValueError("恢复checkpoint架构不匹配")
            model.load_state_dict(saved["model"], strict=True)
            state.update(saved["state"])
            if len(saved["rng_by_rank"]) != runtime.world_size:
                raise ValueError("严格恢复须保留相同world size")
            if not state.get("initial_validation"):
                raise ValueError("恢复checkpoint缺少初始验证记录")
            for name, risk_key in (("best.pt", "best_risk"), ("best_guarded.pt", "best_guarded_risk")):
                path = output / name
                if path.exists():
                    best = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                    if best["architecture"] != model.architecture or best["state"].get("manifest_sha256") != audit:
                        raise ValueError(f"已有{name}来自其他模型/数据")
                    state[risk_key] = min(state[risk_key], best["state"][risk_key])
                    del best
                elif state[risk_key] < 1e30:
                    raise FileNotFoundError(f"恢复目录缺少已选{name}，须保留完整运行目录")
        else:
            previous = torch.load(config["initial_model_checkpoint"], map_location="cpu",
                                  weights_only=False, mmap=True)
            allowed = {model.architecture, V6Model.architecture}
            if previous.get("architecture") not in allowed:
                raise ValueError("暖启动只接受同架构或V6 checkpoint")
            model.load_state_dict(previous["model"], strict=True)
            del previous
        guard_settings = config["train"].get("baseline_guard")
        if guard_settings and guard_settings.get("enabled", True):
            reference_path = guard_settings.get("reference_checkpoint")
            if not reference_path:
                raise ValueError("baseline_guard须指定原V6 reference_checkpoint")
            reference = torch.load(reference_path, map_location="cpu", weights_only=False, mmap=True)
            if reference.get("architecture") != V6Model.architecture:
                raise ValueError("baseline_guard参考必须是原V6 checkpoint")
            guard_reference = reference.get("validation")
            if not guard_reference or not guard_reference.get("full"):
                raise ValueError("baseline_guard参考checkpoint必须含完整VAL")
            if reference["config"]["datasets"] != config["datasets"]:
                raise ValueError("baseline_guard参考与当前数据配置不一致")
            if saved and state.get("guard_reference") != guard_reference:
                raise ValueError("baseline_guard参考验证记录在恢复时已改变")
            state["guard_reference"] = guard_reference
            del reference
        model.to(runtime.device)
        seed_all(config["seed"] + runtime.rank)
        write_distributed_json(output / "manifest_sha256.json", audit, runtime)
        write_distributed_json(output / "config.json", config, runtime)
        emit(output, dict(event="initialized", world_size=runtime.world_size,
                         architecture=model.architecture, global_batch_size=global_batch,
                         epochs=epochs, updates_per_epoch=epoch_steps, total_updates=epoch_steps * epochs,
                         parameters=sum(parameter.numel() for parameter in model.parameters()),
                         train_counts=first_sampler.domain_counts,
                         task_warm_start=config.get("initial_model_checkpoint", ""), resume=str(resume or "")), runtime)
        initial_limit = config["train"].get("initial_validation_limit")
        starting_result = validate(model, validation, config, runtime, initial_limit)
        starting_result = dict(starting_result, baseline_guard=baseline_guard(
            starting_result, state.get("guard_reference", {}), guard_settings))
        result = (saved.get("validation") or starting_result) if saved else starting_result
        if saved:
            write_distributed_json(output / "resume_validation.json", starting_result, runtime)
            emit(output, dict(event="resume_validation", global_step=state["global_step"], **starting_result), runtime)
            if state["epoch"] >= epochs:
                emit(output, dict(event="already_completed", **progress_state(state)), runtime)
                return
        else:
            state["initial_validation"] = starting_result
            write_distributed_json(output / "initial_validation.json", starting_result, runtime)
            emit(output, dict(event="initial_validation", global_step=0, **starting_result), runtime)
            if starting_result["full"]:
                state["best_risk"] = starting_result["selection_risk"]
                save(output / "best.pt", model, None, None, state, config, runtime, validation=starting_result)
                guard = baseline_guard(starting_result, state.get("guard_reference", {}), guard_settings)
                if guard.get("passed"):
                    state["best_guarded_risk"] = starting_result["selection_risk"]
                    save(output / "best_guarded.pt", model, None, None, state, config, runtime,
                         validation=dict(starting_result, baseline_guard=guard))
        parallel = optimizer = scheduler = None
        active_stage = None
        for epoch in range(state["epoch"], epochs):
            stage, first_epoch, last_epoch = stage_for(config, epoch)
            if stage["number"] != active_stage:
                del parallel, optimizer, scheduler
                model.set_stage(stage["number"])
                optimizer, scheduler = optimizer_for(model, stage, config), None
                parallel = DistributedDataParallel(model, device_ids=[runtime.local_rank],
                                                   broadcast_buffers=False, find_unused_parameters=True) \
                    if runtime.distributed else model
                active_stage = stage["number"]
                if saved is not None:
                    if state.get("optimizer_stage") == active_stage:
                        optimizer.load_state_dict(saved["optimizer"])
                    restore_rng(saved["rng_by_rank"][runtime.rank], runtime)
                    saved = None
                state["optimizer_stage"] = active_stage
                emit(output, dict(event="stage", stage=active_stage,
                                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)), runtime)
            model.train()
            sampler = sampler_for(training, config, runtime, epoch, state["update"])
            loader = DataLoader(training, batch_sampler=sampler, collate_fn=collate_mixed,
                                num_workers=config["train"]["workers"], pin_memory=True,
                                generator=torch.Generator().manual_seed(config["seed"] + epoch))
            optimizer.zero_grad(set_to_none=True)
            begin, count_updates, start_update = time.monotonic(), 0, state["update"]
            accumulated_terms = {}
            for micro, batch in enumerate(loader):
                batch = move(batch, runtime.device, non_blocking=True)
                domain = training.domains[batch["domain_index"]]
                work = equal_pixel_inputs(batch["inputs"])
                micro_id = state["global_step"] * accumulation + micro % accumulation
                augmentation_seed = config["seed"] + (runtime.rank * 1009 if recipe == "legacy" else 0)
                domain = compose_training_domain(batch, domain, config, micro_id, augmentation_seed)
                boundary = (micro + 1) % accumulation == 0
                sync = nullcontext() if boundary or not runtime.distributed else parallel.no_sync()
                with sync:
                    with precision_context(runtime.device, config["train"]["precision"]):
                        prediction = parallel(work, domain)
                    with torch.autocast(device_type="cuda", enabled=False):
                        terms = objective(prediction, work, batch["targets"], domain, config.get("loss", {}),
                                          correspondence_weight=stage.get("correspondence_weight", 0.))
                        loss = terms["total"]
                    if not all_ranks_true(torch.isfinite(loss).item(), runtime):
                        raise RuntimeError("损失非有限，保留上个checkpoint并停止")
                    (loss / accumulation).backward()
                for name, term in terms.items():
                    if torch.is_tensor(term) and term.numel() == 1:
                        accumulated_terms[name] = accumulated_terms.get(name, 0.) + term.detach().float() / accumulation
                del prediction, loss, terms
                if not boundary:
                    continue
                next_step = state["global_step"] + 1
                log_update = next_step == 1 or next_step % config["train"]["log_every"] == 0
                norms = group_gradient_norms(optimizer) if log_update else {}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"].get("grad_clip", 1.),
                                                     error_if_nonfinite=True)
                update = start_update + (micro + 1) // accumulation
                elapsed, duration = (epoch - first_epoch) * epoch_steps + update, (last_epoch - first_epoch) * epoch_steps
                factor = .1 + .9 * .5 * (1 + math.cos(math.pi * min(elapsed / duration, 1.)))
                warmup = int(config["train"].get("warmup_updates", 100))
                if epoch == 0 and warmup > 0:
                    factor *= min(1., elapsed / warmup)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * factor
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.update(epoch=epoch, update=update, global_step=next_step,
                             exposures=state["exposures"] + global_batch)
                count_updates += 1
                if log_update:
                    keys = sorted(accumulated_terms)
                    values = torch.stack([accumulated_terms[key] for key in keys])
                    if runtime.distributed:
                        dist.all_reduce(values)
                        values /= runtime.world_size
                    logged_terms = dict(zip(keys, values.cpu().tolist()))
                    emit(output, dict(event="update", **progress_state(state), domain=domain["name"],
                                     loss=logged_terms["total"], loss_terms=logged_terms,
                                     grad_norm=float(norm), group_grad_norm={name: float(value) for name, value in norms.items()},
                                     group_learning_rates={group["group_name"]: group["lr"] for group in optimizer.param_groups},
                                     seconds_per_update=(time.monotonic() - begin) / count_updates,
                                     peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30), runtime)
                accumulated_terms = {}
                if next_step == 1 or next_step % config["train"]["checkpoint_every"] == 0:
                    save(output / "last.pt", model, optimizer, scheduler, state, config, runtime)
                stop = config["train"].get("stop_after_updates", 0)
                if stop and next_step >= stop:
                    save(output / "last.pt", model, optimizer, scheduler, state, config, runtime)
                    emit(output, dict(event="smoke_stopped", **progress_state(state)), runtime)
                    return
            state.update(epoch=epoch + 1, update=0)
            scheduled_full = ((epoch + 1) % config["train"]["full_validation_every"] == 0
                              or epoch + 1 == epochs)
            override_limit = config["train"].get("validation_limit")
            limit = override_limit if override_limit is not None else (
                None if scheduled_full else config["train"]["validation_samples_per_domain"])
            result = validate(model, validation, config, runtime, limit)
            full = result["full"]
            guard = baseline_guard(result, state.get("guard_reference", {}), guard_settings)
            result = dict(result, baseline_guard=guard)
            write_distributed_json(output / f"validation_epoch_{epoch + 1:03d}.json", result, runtime)
            emit(output, dict(event="validation", epoch=epoch + 1, global_step=state["global_step"], **result), runtime)
            if full and result["selection_risk"] < state["best_risk"]:
                state["best_risk"] = result["selection_risk"]
                save(output / "best.pt", model, None, None, state, config, runtime, validation=result)
            if guard.get("passed") and result["selection_risk"] < state["best_guarded_risk"]:
                state["best_guarded_risk"] = result["selection_risk"]
                save(output / "best_guarded.pt", model, None, None, state, config, runtime, validation=result)
            save(output / "last.pt", model, optimizer, scheduler, state, config, runtime, validation=result)
        save(output / "final.pt", model, None, None, state, config, runtime, validation=result)
        write_distributed_json(output / "completed.json", dict(completed=True, **progress_state(state),
            test_accessed=False, validation_full=bool(result["full"]),
            best=str(output / "best.pt") if state["best_risk"] < 1e30 else None, final=str(output / "final.pt"),
            best_guarded=str(output / "best_guarded.pt") if state["best_guarded_risk"] < 1e30 else None), runtime)
        emit(output, dict(event="training_complete", **progress_state(state)), runtime)
    finally:
        cleanup_runtime(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.resume or None)


if __name__ == "__main__":
    main()
