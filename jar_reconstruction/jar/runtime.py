"""Explicit CPU/CUDA device, precision and torchrun process management."""

from dataclasses import dataclass
from datetime import timedelta
import os
import random
import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    is_main: bool


def setup_runtime(device_string="auto", timeout_seconds=300):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid torchrun WORLD_SIZE/RANK")
    requested = str(device_string)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This runtime supports cpu, cuda, cuda:N or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access CUDA; check driver, GPU visibility and the installed CUDA-enabled PyTorch build"
            )
        index = local if world > 1 or device.index is None else device.index
        if world > 1 and device.index is not None and device.index != local:
            raise ValueError(
                "Under torchrun use --device cuda or auto; each process owns cuda:LOCAL_RANK"
            )
        if index >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    elif device.index is not None:
        device = torch.device("cpu")
    distributed = world > 1
    if distributed:
        if not dist.is_available():
            raise RuntimeError("This PyTorch build lacks torch.distributed")
        backend = "nccl" if device.type == "cuda" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world,
                timeout=timedelta(seconds=timeout_seconds),
            )
        elif dist.get_rank() != rank or dist.get_world_size() != world:
            raise RuntimeError("An incompatible process group is already initialized")
    return Runtime(device, rank, world, local, distributed, rank == 0)


def cleanup_runtime(runtime=None):
    # No unconditional barrier: a failed rank must not leave its peers waiting.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def validate_precision(device, precision):
    device = torch.device(device)
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training/inference requires CUDA; choose fp32 or bf16 on CPU")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device/build does not support bf16")


def precision_context(device, precision="fp32"):
    device = torch.device(device)
    validate_precision(device, precision)
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


def make_scaler(device, precision="fp32"):
    validate_precision(device, precision)
    return torch.amp.GradScaler("cuda", enabled=precision == "fp16")


def move(value, device, non_blocking=False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {k: move(v, device, non_blocking) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device, non_blocking) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device, non_blocking) for v in value)
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_ranks_true(value, runtime):
    flag = torch.as_tensor(bool(value), dtype=torch.int32, device=runtime.device)
    if runtime.distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def capture_rng(runtime):
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), *numpy[2:]),
        "cuda": torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None,
    }


def restore_rng(state, runtime):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if "numpy" in state:
        numpy = state["numpy"]
        np.random.set_state((numpy[0], numpy[1].cpu().numpy().astype(np.uint32), *numpy[2:]))
    if runtime.device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), runtime.device)
