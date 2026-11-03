"""三模型共同数据训练入口；保留各自网络、损失和优化器。

仅配置文件决定实验。旧权重不覆盖；last支持恢复，best仅按完整VAL选取。
"""

import argparse
import copy
from contextlib import nullcontext
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import collate_mixed
from jar.pretraining.runner import atomic_save
from jar.runtime import (
    all_ranks_true, capture_rng, cleanup_runtime, move, precision_context,
    restore_rng, seed_all, setup_runtime,
)
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from research.rulermvs_v6_20261101.training import (
    optimizer_for, regional_selection_risk, single_model_validation,
)
from research.v2_restart_20261029.training import RestartHooks
from tools.smoke_outdoor_datasets import offline_only


def build_model(config, initialize=True):
    if config["method"] == "v6":
        return V6Model(config, initialize)
    if config["method"] == "v7":
        from research.overnight_v6_v7_mvs_20261103.v7.model import V7Model
        return V7Model(config, initialize)
    if config["method"] == "mvsanywhere":
        from research.overnight_v6_v7_mvs_20261103.mvs_baseline.model import MVSAnywhereTrainingModel
        return MVSAnywhereTrainingModel(config, initialize)
    raise ValueError("method须为v6/v7/mvsanywhere")


def sampler_for(dataset, config, runtime, epoch, start_update=0):
    from research.v2_training_20261028.sampling import RebalancedFullPassSampler
    sampler = RebalancedFullPassSampler(
        [len(d) for d in dataset.datasets], epoch, runtime.rank, runtime.world_size,
        config["train"]["batch_size"], config["train"]["accumulation"], config["seed"],
        start_update=start_update,
        domain_multipliers=[d.get("train_multiplier", 1.) for d in dataset.domains],
        domain_schedule="balanced_prefix",
    )
    if config["train"].get("smoke_updates_per_epoch"):
        sampler.updates = sampler.updates[:config["train"]["smoke_updates_per_epoch"]]
    return sampler


def stage_for(config, epoch):
    beginning = 0
    for stage in config["stages"]:
        ending = beginning + stage["epochs"]
        if epoch < ending:
            return stage, beginning, ending
        beginning = ending
    raise ValueError("epoch超出配置阶段")


class EvaluationModel(nn.Module):
    """对接既有完整VAL统计；仅处理主图信息与统一深度单位。"""

    def __init__(self, model, method):
        super().__init__()
        self.model, self.method = model, method

    def forward(self, inputs, domain, reference=0):
        if self.method != "mvsanywhere":
            inputs = equal_pixel_inputs(inputs)
        result = self.model(inputs, domain, reference=reference)
        # 旧验证器会乘domain尺度；先把原生米制输出转换到该契约。
        return dict(result, depth=result["depth"].float() * result["unit_scale"] / domain["depth_scale"])


def validate(model, dataset, config, runtime, limit):
    hooks = RestartHooks(config)
    with single_model_validation(model, runtime):
        result = hooks.evaluate(EvaluationModel(model, config["method"]), dataset,
                                config, runtime, limit=limit)
    result["selection_risk"] = regional_selection_risk(
        result["domains"], dataset.domains, config["evaluation"]["selection_region_weights"])
    result["selection_rule"] = "各域等权、full/near/mid/far按事前配置加权；仅完整VAL选择best"
    return result


def emit(output, record, runtime):
    if runtime.is_main:
        record = dict(time=datetime.now().astimezone().isoformat(), **record)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def save(path, model, optimizer, scheduler, state, config, runtime, validation=None):
    rng = capture_rng(runtime)
    gathered = [None] * runtime.world_size
    if runtime.distributed:
        dist.all_gather_object(gathered, rng)
    else:
        gathered = [rng]
    error = [None]
    if runtime.is_main:
        try:
            record = dict(architecture=model.architecture, model=model.state_dict(), config=config,
                          state=dict(state), validation=validation, rng_by_rank=gathered)
            if optimizer is not None:
                record["optimizer"] = optimizer.state_dict()
                record["scheduler"] = scheduler.state_dict() if scheduler else None
            atomic_save(record, path, config["train"]["minimum_free_gib"])
        except Exception as exception:
            error[0] = repr(exception)
    if runtime.distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def run(config, resume=None):
    from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset, own_training_domain
    offline_only()
    runtime = setup_runtime("cuda", timeout_seconds=900)
    output = Path(config["output"])
    try:
        torch.set_num_threads(config["train"].get("threads", 2))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(config["seed"])
        startup_error = [None]
        if runtime.is_main:
            try:
                output.mkdir(parents=True, exist_ok=True)
                if not resume and (output / "last.pt").exists():
                    raise FileExistsError("输出已有last.pt；请显式resume，不能覆盖旧实验")
                audit = {}
                for domain in config["datasets"]:
                    checksums = {split: digest(path) for split, path in domain["manifests"].items()}
                    if checksums["train"] != domain["train_manifest_sha256"]:
                        raise ValueError(f"{domain['name']} TRAIN清单已改变")
                    audit[domain["name"]] = checksums
                write_json(output / "manifest_sha256.json", audit)
            except Exception as exception:
                startup_error[0] = repr(exception)
        if runtime.distributed:
            dist.broadcast_object_list(startup_error, src=0)
        if startup_error[0]:
            raise RuntimeError(startup_error[0])
        batch_size = config["train"]["batch_size"]
        accumulation = config["train"]["accumulation"]
        global_batch = runtime.world_size * batch_size * accumulation
        if global_batch != config["train"]["global_batch_size"]:
            raise ValueError("显卡数、batch与累积不符合登记的共同全局batch")
        training = SharedDataset(config, "train")
        validation = SharedDataset(config, "val")
        first_sampler = sampler_for(training, config, runtime, 0)
        epoch_steps = len(first_sampler.updates)
        epochs = sum(s["epochs"] for s in config["stages"])
        saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True) if resume else None
        if saved:
            prior, current = copy.deepcopy(saved["config"]), copy.deepcopy(config)
            for settings in (prior, current):
                settings.pop("output", None)
                settings["train"].pop("stop_after_updates", None)
            if prior != current:
                raise ValueError("严格恢复不允许改变数据、阶段、初始化或计算协议")
        if runtime.is_main:
            write_json(output / "config.json", config)
        model = build_model(config, initialize=saved is None)
        state = dict(epoch=0, update=0, global_step=0, exposures=0, best_risk=1e30)
        if saved:
            if saved["architecture"] != model.architecture:
                raise ValueError("恢复checkpoint架构不匹配")
            model.load_state_dict(saved["model"], strict=True)
            state.update(saved["state"])
            if len(saved["rng_by_rank"]) != runtime.world_size:
                raise ValueError("严格恢复须保留相同world size")
            best_path = output / "best.pt"
            if best_path.exists():
                best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
                if best["architecture"] != model.architecture or best["config"]["datasets"] != config["datasets"]:
                    raise ValueError("已有best来自不同架构或数据实验")
                state["best_risk"] = min(state["best_risk"], best["state"]["best_risk"])
                del best
            elif state["best_risk"] < 1e30:
                raise FileNotFoundError("恢复目录缺少已选best；须保留整个独立运行目录")
        elif config.get("initial_model_checkpoint"):
            previous = torch.load(config["initial_model_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
            if config["method"] == "v7":
                model.load_v6_state(previous["model"])
            elif config["method"] == "v6":
                model.load_state_dict(previous["model"], strict=True)
            else:
                raise ValueError("本轮MVS禁止初始化任务checkpoint")
        model.to(runtime.device)
        seed_all(config["seed"] + runtime.rank)
        emit(output, dict(event="initialized", world_size=runtime.world_size,
                         architecture=model.architecture, global_batch_size=global_batch,
                         epochs=epochs, updates_per_epoch=epoch_steps,
                         total_updates=epoch_steps * epochs,
                         parameters=sum(p.numel() for p in model.parameters()),
                         train_counts=first_sampler.domain_counts,
                         initialization=getattr(model, "initialization_report", config["initialization"]),
                         task_warm_start=config.get("initial_model_checkpoint", "")), runtime)
        parallel = optimizer = scheduler = None
        active_stage = None
        result = saved.get("validation") if saved else None
        for epoch in range(state["epoch"], epochs):
            stage, first_epoch, last_epoch = stage_for(config, epoch)
            if stage["number"] != active_stage:
                del parallel, optimizer, scheduler
                if config["method"] == "mvsanywhere":
                    optimizer, scheduler = model.configure_optimizer(epoch_steps * epochs)
                else:
                    model.set_stage(stage["number"])
                    optimizer = optimizer_for(model, stage, config)
                    scheduler = None
                parallel = DistributedDataParallel(
                    model, device_ids=[runtime.local_rank], broadcast_buffers=False,
                    find_unused_parameters=True,
                ) if runtime.distributed else model
                active_stage = stage["number"]
                if saved is not None:
                    if state.get("optimizer_stage") == active_stage:
                        optimizer.load_state_dict(saved["optimizer"])
                        if scheduler and saved.get("scheduler"):
                            scheduler.load_state_dict(saved["scheduler"])
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
            begin = time.monotonic()
            count_updates = 0
            accumulated_loss = 0.
            start_update = state["update"]
            for micro, batch in enumerate(loader):
                batch = move(batch, runtime.device, non_blocking=True)
                domain = training.domains[batch["domain_index"]]
                work = batch["inputs"]
                if config["method"] != "mvsanywhere":
                    work = equal_pixel_inputs(work)
                    micro_id = state["global_step"] * accumulation + micro % accumulation
                    domain = own_training_domain(batch, domain, config, micro_id,
                                                 config["seed"] + runtime.rank * 1009)
                boundary = (micro + 1) % accumulation == 0
                sync = nullcontext() if boundary or not runtime.distributed else parallel.no_sync()
                with sync:
                    with precision_context(runtime.device, config["train"]["precision"]):
                        if config["method"] == "mvsanywhere":
                            prediction = parallel(work, domain, targets=batch["targets"])
                            loss = prediction["loss"]
                        else:
                            prediction = parallel(work, domain)
                    if config["method"] != "mvsanywhere":
                        with torch.autocast(device_type="cuda", enabled=False):
                            terms = objective(prediction, work, batch["targets"], domain, config.get("loss", {}),
                                              correspondence_weight=stage.get("correspondence_weight", 0.))
                            loss = terms["total"]
                    if not all_ranks_true(torch.isfinite(loss).item(), runtime):
                        raise RuntimeError("损失非有限，停止并保留上个checkpoint")
                    (loss / accumulation).backward()
                accumulated_loss += float(loss.detach()) / accumulation
                del prediction, loss
                if not boundary:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                update = start_update + (micro + 1) // accumulation
                if config["method"] != "mvsanywhere":
                    elapsed = (epoch - first_epoch) * epoch_steps + update
                    duration = (last_epoch - first_epoch) * epoch_steps
                    factor = .1 + .9 * .5 * (1 + math.cos(math.pi * min(elapsed / duration, 1.)))
                    if epoch == 0:
                        factor *= min(1., elapsed / config["train"].get("warmup_updates", 100))
                    for group in optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * factor
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.update(epoch=epoch, update=update, global_step=state["global_step"] + 1,
                             exposures=state["exposures"] + global_batch)
                count_updates += 1
                if state["global_step"] == 1 or state["global_step"] % config["train"]["log_every"] == 0:
                    value = torch.tensor(accumulated_loss, device=runtime.device)
                    if runtime.distributed:
                        dist.all_reduce(value)
                        value /= runtime.world_size
                    emit(output, dict(event="update", **state, domain=domain["name"], loss=float(value),
                                     grad_norm=float(norm), seconds_per_update=(time.monotonic() - begin) / count_updates,
                                     peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30), runtime)
                accumulated_loss = 0.
                if state["global_step"] == 1 or state["global_step"] % config["train"]["checkpoint_every"] == 0:
                    save(output / "last.pt", model, optimizer, scheduler, state, config, runtime)
                stop = config["train"].get("stop_after_updates", 0)
                if stop and state["global_step"] >= stop:
                    save(output / "last.pt", model, optimizer, scheduler, state, config, runtime)
                    emit(output, dict(event="smoke_stopped", **state), runtime)
                    return
            state.update(epoch=epoch + 1, update=0)
            full = (epoch + 1) % config["train"]["full_validation_every"] == 0 or epoch + 1 == epochs
            limit = None if full else config["train"]["validation_samples_per_domain"]
            result = validate(model, validation, config, runtime, limit)
            emit(output, dict(event="validation", epoch=epoch + 1, global_step=state["global_step"], **result), runtime)
            if full and result["selection_risk"] < state["best_risk"]:
                state["best_risk"] = result["selection_risk"]
                save(output / "best.pt", model, None, None, state, config, runtime, validation=result)
            save(output / "last.pt", model, optimizer, scheduler, state, config, runtime, validation=result)
        if result is None:
            result = validate(model, validation, config, runtime, None)
        save(output / "final.pt", model, None, None, state, config, runtime, validation=result)
        if runtime.is_main:
            write_json(output / "completed.json", dict(completed=True, **state, test_accessed=False,
                                                       best=str(output / "best.pt"), final=str(output / "final.pt")))
        emit(output, dict(event="training_complete", **state), runtime)
    finally:
        cleanup_runtime(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config, args.resume or None)


if __name__ == "__main__":
    main()
