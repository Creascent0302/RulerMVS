#!/usr/bin/env python3
"""Read-only CUDA preflight with actual forward/backward kernels on every GPU.

No install, download, training or environment mutation. A separate Python worker
per device isolates a failed CUDA context. Metadata alone never certifies CUDA.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "gpu_capability": "https://developer.nvidia.com/cuda/gpus",
    "first_blackwell_release": "https://pytorch.org/blog/pytorch-2-7/",
    "cuda_wheel_update": "https://pytorch.org/blog/pytorch-2-12-release-blog/",
    "install_selector": "https://pytorch.org/get-started/locally/",
    "ptx_compatibility": "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html",
}


def version_tuple(value):
    values = re.findall(r"\d+", str(value).split("+")[0])
    return tuple(int(x) for x in values[:2]) if values else (0, 0)


def compatibility_notes(name, capability, arch_list, torch_version, cuda_version):
    """Advisories only. PTX can work without an exact native sm entry."""
    arch = f"sm_{capability[0]}{capability[1]}"
    notes = []
    if arch not in arch_list:
        notes.append(f"编译架构列表没有 {arch}；仍可能使用兼容 PTX，必须以实际内核测试为准。")
    if tuple(capability) >= (12, 0) and (
        version_tuple(torch_version) < (2, 7) or version_tuple(cuda_version) < (12, 8)
    ):
        notes.append(
            "此 Blackwell GPU 搭配旧 PyTorch/CUDA wheel；建议更换官方支持 Blackwell 的稳定 wheel。PyTorch 2.7 + CUDA 12.8 首次提供支持；2.12 公告转向 CUDA 13.0+。"
        )
    if "5090" in name and tuple(capability) != (12, 0):
        notes.append(
            "名称含 RTX 5090，但驱动报告的计算能力并非 NVIDIA 表列 12.0；请核查设备和驱动。"
        )
    return notes


def nvidia_smi_report():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {
            "available": result.returncode == 0,
            "returncode": result.returncode,
            "output": result.stdout.strip(),
            "error": result.stderr.strip()[:4000],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}


def assert_finite(tensor, description):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{description} 含 NaN/Inf")


def kernel_smoke(device_index=0, precision="fp32"):
    """Actually launch GEMM/cuDNN/sampling/cuSOLVER and sparse indexing kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不能执行或认证 GPU 内核。")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("该设备/运行时未报告 BF16 支持。")
    torch.manual_seed(918)
    records = []

    def run(name, operation):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            output, leaves = operation()
            assert_finite(output, name + " forward")
            objective = output.float().square().mean()
        objective.backward()
        for leaf in leaves:
            if leaf.grad is None:
                raise AssertionError(f"{name} 缺少输入梯度")
            assert_finite(leaf.grad, name + " backward")
        torch.cuda.synchronize(device)
        records.append(
            {
                "operation": name,
                "status": "passed",
                "forward_backward_ms": (time.perf_counter() - start) * 1000,
            }
        )

    def gemm():
        x = torch.randn(96, 64, device=device, requires_grad=True)
        w = torch.randn(64, 32, device=device, requires_grad=True)
        return x @ w, (x, w)

    def convolution():
        x = torch.randn(2, 3, 16, 20, device=device, requires_grad=True)
        w = torch.randn(8, 3, 3, 3, device=device, requires_grad=True)
        return F.conv2d(x, w, padding=1), (x, w)

    def sampling():
        x = torch.randn(2, 4, 12, 14, device=device, requires_grad=True)
        grid = (torch.rand(2, 8, 10, 2, device=device) * 1.6 - 0.8).requires_grad_()
        return F.grid_sample(x, grid, align_corners=False), (x, grid)

    def linear_solve():
        # Project geometry and solves intentionally stay FP32 under AMP, matching
        # the model's robust linear-algebra path.
        with torch.autocast("cuda", enabled=False):
            raw = torch.randn(12, 3, 3, device=device, requires_grad=True)
            rhs = torch.randn(12, 3, 1, device=device, requires_grad=True)
            matrix = raw @ raw.transpose(-1, -2) + torch.eye(3, device=device) * 0.5
            solution = torch.linalg.solve(matrix, rhs)
            eig = torch.linalg.eigvalsh(matrix)
            return torch.cat((solution[..., 0], eig), -1), (raw, rhs)

    def sparse_indexing():
        x = torch.randn(24, 4, device=device, requires_grad=True)
        ids = torch.arange(24, device=device) % 7
        unique = torch.unique(ids, sorted=True)
        recovered = torch.searchsorted(unique, ids)
        summed = torch.zeros(7, 4, device=device).index_add(0, recovered, x)
        maximum = torch.zeros(7, 4, device=device).scatter_reduce(
            0, ids[:, None].expand_as(x), x, reduce="amax", include_self=True
        )
        return summed + maximum, (x,)

    for name, operation in (
        ("matmul", gemm),
        ("conv2d", convolution),
        ("grid_sample", sampling),
        ("solve_and_eigvalsh", linear_solve),
        ("sparse_indexing", sparse_indexing),
    ):
        run(name, operation)
    return {
        "status": "passed",
        "device_index": device_index,
        "precision": precision,
        "kernels": records,
    }


def run_kernel_worker(device_index, timeout=180):
    with tempfile.TemporaryDirectory(prefix="jar-gpu-check-") as temporary:
        path = Path(temporary) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--kernel-worker",
            str(device_index),
            "--output",
            str(path),
        ]
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
            )
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = {"status": "failed", "error": "GPU 内核工作进程未产生报告。"}
            record["returncode"] = result.returncode
            if result.returncode or result.stderr:
                record["stderr"] = result.stderr[-6000:]
            if result.returncode:
                record["status"] = "failed"
            return record
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": f"GPU {device_index} 内核预检超过 {timeout} 秒。",
                "timeout": True,
            }


def collect_preflight(
    require_gpus=1,
    require_bf16=False,
    timeout=180,
    torch_module=torch,
    worker=run_kernel_worker,
    smi=nvidia_smi_report,
):
    """Injectable metadata/worker providers make no-GPU and failure handling testable."""
    if require_gpus < 1:
        raise ValueError("require_gpus must be >=1")
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    count = int(cuda.device_count())
    report = {
        "status": "not_verified",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "逐卡实际 CUDA 内核前后向兼容性；不是三维重建精度、完整模型或八卡通信认证。",
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "torch_cuda_runtime": torch_module.version.cuda,
        "cuda_available": available,
        "visible_gpu_count": count,
        "required_gpu_count": require_gpus,
        "compiled_arch_list": list(cuda.get_arch_list()),
        "devices": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_force_ptx_jit": os.environ.get("CUDA_FORCE_PTX_JIT"),
        "nvidia_smi": smi(),
        "sources": SOURCES,
        "messages_zh": [],
    }
    if not available or count == 0:
        report["messages_zh"].append(
            "当前 PyTorch 无法使用 CUDA，未执行任何 GPU 内核；不能认证 RTX 5090 或八卡训练。"
        )
        report["messages_zh"].append(
            "请检查驱动、容器 GPU 映射和 CUDA_VISIBLE_DEVICES，并安装与设备匹配的官方 PyTorch wheel。不会自动回退为 CPU 通过。"
        )
        return report
    if count < require_gpus:
        report["messages_zh"].append(
            f"只看到 {count} 张 GPU，少于要求的 {require_gpus} 张；仍检查可见设备，但整体不通过。"
        )
    passed = count >= require_gpus
    for index in range(count):
        try:
            props = cuda.get_device_properties(index)
            capability = list(cuda.get_device_capability(index))
            record = {
                "index": index,
                "name": props.name,
                "compute_capability": capability,
                "total_memory_bytes": int(props.total_memory),
                "notes_zh": compatibility_notes(
                    props.name,
                    capability,
                    report["compiled_arch_list"],
                    report["torch"],
                    report["torch_cuda_runtime"],
                ),
            }
            record["kernel_test"] = worker(index, timeout=timeout)
            valid = record["kernel_test"].get("status") == "passed"
            if require_bf16 and record["kernel_test"].get("bf16", {}).get("status") != "passed":
                valid = False
                record["notes_zh"].append("本次要求 BF16；BF16 内核未通过，不能认证。")
            passed = passed and valid
        except Exception as error:
            record = {
                "index": index,
                "kernel_test": {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            }
            passed = False
        report["devices"].append(record)
    report["status"] = "passed" if passed else "failed"
    report["messages_zh"].append(
        "要求数量的可见 GPU 已通过实际内核前后向检查；完整模型和 DDP 请继续运行 gpu_acceptance.py。"
        if passed
        else "GPU 数量或实际内核测试未通过；请查看逐卡错误，不能据此启动已认证的训练。"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="逐卡 CUDA 内核预检；无 GPU 返回非零，不安装任何软件"
    )
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="")
    parser.add_argument("--kernel-worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.require_gpus < 1:
        parser.error("timeout 和 require-gpus 必须大于零")
    if args.kernel_worker is not None:
        report = {"status": "failed"}
        try:
            report["fp32"] = kernel_smoke(args.kernel_worker, "fp32")
            if torch.cuda.is_bf16_supported():
                report["bf16"] = kernel_smoke(args.kernel_worker, "bf16")
            else:
                report["bf16"] = {"status": "unsupported"}
            report["status"] = "passed"
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
    else:
        report = collect_preflight(args.require_gpus, args.require_bf16, args.timeout)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    print(serialized, end="", flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
