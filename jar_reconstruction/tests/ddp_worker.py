"""Run with torchrun --standalone --nproc_per_node 2 tests/ddp_worker.py.

Two real ranks, MF-ET-JAR, changing local unused heads, two
microbatches per step, synchronized gradients and exact optimizer/RNG resume.
"""

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from jar import build_model
from jar.config import load_config
from jar.losses import ReconstructionLoss
from jar.synthetic import SyntheticRigDataset, make_synthetic_batch
from jar.runtime import (
    setup_runtime,
    cleanup_runtime,
    seed_all,
    make_scaler,
    restore_rng,
    all_ranks_true,
)
from jar.training import (
    ModelAndLoss,
    wrap_ddp,
    training_contract,
    save_checkpoint,
    _restore_checkpoint,
)


def assert_equal_across_ranks(model):
    flattened = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters()
        ]
    )
    expected = flattened.clone()
    dist.broadcast(expected, src=0)
    if not torch.allclose(flattened, expected, atol=1e-7, rtol=1e-6):
        raise AssertionError(f"DDP gradients disagree: {(flattened-expected).abs().max().item()}")


def do_step(forward, model, optimizer, runtime, step):
    optimizer.zero_grad(set_to_none=True)
    values = []
    topologies = []

    def observe_topology(module, args, output):
        scene = output["scenes"][0]
        topologies.append(
            {"stages": len(scene.get("stages", [scene])), "elements": len(scene["points"])}
        )

    hook = model.register_forward_hook(observe_topology)
    for micro in range(2):
        batch = make_synthetic_batch(N=3, H=32, W=48, seed=step * 2 + micro)
        # Camera masks and available supervision differ across ranks, then
        # switch. This exercises dynamically unused parameter sets, not merely
        # duplicated images through a static MLP.
        if runtime.rank == 1:
            batch["inputs"]["view_mask"][0, -1] = False
            batch["inputs"]["image_mask"][0, -1] = False
        if (runtime.rank + step + micro) % 2:
            batch["targets"] = {}
        loss = forward(batch["inputs"], batch["targets"])
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite DDP loss")
        (loss / 2).backward()
        values.append(float(loss.detach()))
    assert_equal_across_ranks(model)
    hook.remove()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    all_topologies = [None] * runtime.world_size
    dist.all_gather_object(all_topologies, topologies)
    return {"local_losses": values, "grad_norm": float(norm), "rank_topologies": all_topologies}


def main(output):
    torch.set_num_threads(2)
    runtime = setup_runtime("cpu", timeout_seconds=90)
    results = []
    try:
        if all_ranks_true(runtime.rank == 0, runtime):
            raise AssertionError("Cross-rank finite/acceptance flag ignored a failing rank")
        if not all_ranks_true(True, runtime):
            raise AssertionError("Cross-rank finite/acceptance flag rejected all-valid ranks")
        for name in ("mf_et_jar",):
            cfg = load_config("configs/unit_tiny.json")
            cfg["model"]["name"] = name
            cfg["train"]["accumulation_steps"] = 2
            seed_all(31)
            model = build_model(cfg)
            wrapper = ModelAndLoss(model, ReconstructionLoss(cfg), cfg)
            forward = wrap_ddp(wrapper, runtime)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            scaler = make_scaler(runtime.device, "fp32")
            dataset = SyntheticRigDataset(length=3, image_size=(32, 48), num_cameras=3)
            contract = training_contract(cfg, dataset, runtime, True, "fp32")
            first = do_step(forward, model, optimizer, runtime, 0)
            path = Path(output) / (name + ".pt")
            if runtime.is_main:
                path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(path, model, optimizer, scaler, cfg, contract, runtime, 1, 0, 2)
            expected_random = (torch.rand(3), random.random(), np.random.rand())
            second = do_step(forward, model, optimizer, runtime, 1)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            resumed_model = build_model(cfg)
            resumed_forward = wrap_ddp(
                ModelAndLoss(resumed_model, ReconstructionLoss(cfg), cfg), runtime
            )
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.0003)
            record = torch.load(path, map_location="cpu", weights_only=True)
            step, epoch, offset, rng = _restore_checkpoint(
                record, resumed_model, resumed_optimizer, scaler, cfg, contract, runtime
            )
            if (step, epoch, offset) != (1, 0, 2):
                raise AssertionError("Checkpoint cursor did not round-trip")
            restore_rng(rng, runtime)
            actual_random = (torch.rand(3), random.random(), np.random.rand())
            torch.testing.assert_close(actual_random[0], expected_random[0], rtol=0, atol=0)
            if actual_random[1:] != expected_random[1:]:
                raise AssertionError("Python/NumPy RNG resume changed")
            resumed = do_step(resumed_forward, resumed_model, resumed_optimizer, runtime, 1)
            for key, value in resumed_model.state_dict().items():
                torch.testing.assert_close(
                    value, expected[key], rtol=0, atol=0, msg=lambda m: key + ": " + m
                )
            changed_contract = dict(contract, world_size=3)
            try:
                _restore_checkpoint(
                    record, resumed_model, resumed_optimizer, scaler, cfg, changed_contract, runtime
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Changed topology was silently accepted")
            results.append(
                {
                    "model": name,
                    "two_steps": [first, second],
                    "resume_step": resumed,
                    "exact_resume": True,
                    "gradients_equal": True,
                }
            )
        if runtime.is_main:
            Path(output, "result.json").write_text(
                json.dumps({"world_size": 2, "backend": "gloo", "models": results}, indent=2)
            )
            print("DDP_TEST_OK " + json.dumps(results), flush=True)
        dist.barrier()
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/ddp_cpu_test")
    main(parser.parse_args().output)
