"""torchrun执行的真实DDP训练/恢复检查；tiny ViT不代替正式模型冒烟。

用法：torchrun --standalone --nproc_per_node=2 -m
research.rulermvs_v6_20261101.test_training_ddp --device cuda --output 数据盘测试目录
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from jar.pretraining.runner import save_training
from jar.runtime import cleanup_runtime, move, restore_rng, seed_all, setup_runtime
from research.early_geometry_20261030.test_encoding import inputs
from research.rulermvs_v6_20261101.test_training import tiny_config, tiny_model
from research.rulermvs_v6_20261101.training import optimizer_for, single_model_validation


def step(parallel, optimizer, hooks, batch, domain):
    prepared, domain = hooks.prepare_batch(batch, domain)
    prediction = parallel(prepared["inputs"], domain)
    loss = hooks.objective(prediction, prepared["inputs"], prepared["targets"], domain, {}, 0.)["total"]
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("DDP loss非有限")
    loss.backward()
    model = parallel.module if isinstance(parallel, DistributedDataParallel) else parallel
    gate_gradient = sum(float(p.grad.abs().sum()) for p in model.decoder.handoff.parameters()
                        if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), gate_gradient


def parallelize(model, runtime):
    return DistributedDataParallel(
        model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True,
    ) if runtime.distributed else model


def check(runtime, output):
    torch.set_num_threads(2)
    seed_all(20261103)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = tiny_config()
    config["train"]["minimum_free_gib"] = 0
    model, hooks = tiny_model(config)
    model.to(runtime.device).set_stage(1)
    model.train()
    optimizer = optimizer_for(model, config["stages"][0], config)
    parallel = parallelize(model, runtime)
    seed_all(900 + runtime.rank)
    data = inputs()
    data["view_mask"][:, -1] = False
    batch = move({"inputs": data, "targets": {
        "depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
        "depth_valid": data["image_mask"].clone(),
    }}, runtime.device)
    domain = {"near": 1., "far": 10., "depth_scale": 1., "correspondence": False}
    first_loss, gate_gradient = step(parallel, optimizer, hooks, batch, domain)
    if gate_gradient <= 0:
        raise RuntimeError("门控未得到真实梯度")
    if runtime.is_main:
        output.mkdir(parents=True, exist_ok=True)
    if runtime.distributed:
        dist.barrier()
    save_training(output / "last.pt", model, optimizer, {"global_step": 1}, config, runtime)
    second_loss, _ = step(parallel, optimizer, hooks, batch, domain)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    del parallel, optimizer, model

    saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    restored, restored_hooks = tiny_model(config)
    restored.load_state_dict(saved["model"], strict=True)
    restored.to(runtime.device).set_stage(1)
    restored.train()
    optimizer = optimizer_for(restored, config["stages"][0], config)
    optimizer.load_state_dict(saved["optimizer"])
    parallel = parallelize(restored, runtime)
    restore_rng(saved["rng_by_rank"][runtime.rank], runtime)
    resumed_loss, _ = step(parallel, optimizer, restored_hooks, batch, domain)
    maximum_difference = 0.
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)
        maximum_difference = max(maximum_difference, float((value - expected[name]).abs().max()))
    if runtime.distributed:
        for parameter in restored.parameters():
            rank_zero = parameter.detach().clone()
            dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(parameter, rank_zero, rtol=0., atol=0.)

    # 人为构造不同rank的BN状态，验证共同模型及恢复，而非仅检查导入。
    bn = nn.BatchNorm1d(3).to(runtime.device).train()
    bn.running_mean.fill_(runtime.rank + 1.)
    original_mean = bn.running_mean.clone()
    rng = torch.get_rng_state().clone()
    with single_model_validation(bn, runtime):
        torch.testing.assert_close(bn.running_mean, torch.ones_like(bn.running_mean))
        torch.rand(13)
        bn.running_mean.fill_(98.)
    torch.testing.assert_close(bn.running_mean, original_mean, rtol=0., atol=0.)
    assert bn.training and torch.equal(torch.get_rng_state(), rng)
    del parallel, optimizer
    restored.set_stage(2)
    new_optimizer = optimizer_for(restored, config["stages"][1], config)
    assert not new_optimizer.state
    result = {"completed": True, "rank": runtime.rank, "world_size": runtime.world_size,
              "device": str(runtime.device), "tiny_vit": True, "missing_view": True,
              "first_loss": first_loss, "second_loss": second_loss, "resumed_loss": resumed_loss,
              "gate_gradient_sum": gate_gradient, "resume_max_difference": maximum_difference,
              "rank_parameters_equal": True, "single_model_bn_and_rng_restored": True,
              "new_stage_optimizer_reset": True, "test_accessed": False}
    (output / f"rank{runtime.rank}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = setup_runtime(args.device)
    try:
        check(runtime, args.output)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
