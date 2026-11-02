#!/usr/bin/env python3
"""Bounded CUDA model acceptance, with optional torchrun + checkpoint resume.

Every model case uses a physically rendered tiny synthetic frame and exactly one
optimizer step. Timings describe this fixture only, not deployment throughput.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from jar.config import load_config
from jar.model import ReconstructionModel
from jar.losses import ReconstructionLoss
from jar.synthetic import make_synthetic_batch
from tools.check_gpu import collect_preflight, assert_finite


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def finite_tree(value, path="output"):
    if torch.is_tensor(value) and value.is_floating_point():
        assert_finite(value, path)
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def model_case(name, cameras, precision, device_index=0, warmup=1, repeats=3):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能用 CPU 测试代替 GPU 验收。")
    if name not in ("mf_et_jar",) or cameras not in (2, 3, 4):
        raise ValueError("Acceptance cases require a supported model and N=2/3/4")
    if precision not in ("fp32", "bf16") or warmup < 0 or repeats < 1:
        raise ValueError("Invalid precision/warmup/repeats")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU/运行时未报告 BF16 支持；不能静默回退为 FP32。")
    torch.manual_seed(919)
    cfg = load_config(ROOT / "configs/unit_tiny.json")
    cfg["model"]["name"] = name
    batch = move(make_synthetic_batch(N=cameras, H=32, W=48, seed=5), device)
    model = ReconstructionModel(cfg).to(device)
    criterion = ReconstructionLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        prediction = model(batch["inputs"])
        losses = criterion(prediction, batch["inputs"], batch["targets"])
    finite_tree(prediction)
    finite_tree(losses, "loss")
    losses["total"].backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or not any(bool(g.abs().sum() > 0) for g in gradients):
        raise AssertionError("训练没有产生任何非零梯度")
    for gradient in gradients:
        assert_finite(gradient, "gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    for parameter in parameters:
        assert_finite(parameter, "updated parameter")
    if not any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters)):
        raise AssertionError("优化步骤没有修改任何模型参数")
    torch.cuda.synchronize(device)
    training_ms = (time.perf_counter() - started) * 1000
    training_memory = torch.cuda.max_memory_allocated(device)
    training_reserved = torch.cuda.max_memory_reserved(device)
    loss_values = {key: float(value.detach()) for key, value in losses.items()}
    gradient_count = len(gradients)
    grad_norm = float(norm)
    # Release training-only allocations before measuring inference. The model is
    # called with inputs only; no labels, teacher or optimizer enter inference.
    inputs = batch["inputs"]
    del batch, prediction, losses, gradients, before, optimizer
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    latencies = []
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                prediction = model(inputs)
            finite_tree(prediction)
            del prediction
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(repeats):
            if iteration:
                del prediction
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                prediction = model(inputs)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000)
            finite_tree(prediction)
        scene = prediction["scenes"][0]
        state_counts = [int((scene["state"] == state).sum()) for state in range(3)]
        points = len(scene["points"])
        inference_memory = torch.cuda.max_memory_allocated(device)
        inference_reserved = torch.cuda.max_memory_reserved(device)
    result = {
        "status": "passed",
        "model": name,
        "device_index": device_index,
        "camera_count": cameras,
        "precision": precision,
        "fixture": "analytic textured planes, boxes and first-hit occlusion; 32x48 RGB",
        "optimizer_steps": 1,
        "grad_parameter_tensors": gradient_count,
        "grad_norm_before_clip": grad_norm,
        "training_losses_not_accuracy": loss_values,
        "training_step_including_cold_start_ms": training_ms,
        "training_peak_allocated_bytes": training_memory,
        "training_peak_reserved_bytes": training_reserved,
        "inference_warmup": warmup,
        "inference_repeats": repeats,
        "inference_ms": latencies,
        "inference_median_ms": statistics.median(latencies),
        "inference_peak_allocated_bytes": inference_memory,
        "inference_peak_reserved_bytes": inference_reserved,
        "emitted_queries_or_leaves": points,
        "free_surface_unknown_counts": state_counts,
        "tested_full_resolution": False,
    }
    del prediction, scene, model, criterion, inputs, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def bounded_command(command, logfile, timeout):
    """Timeout kills the entire torchrun process group, not only its launcher."""
    started = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, _ = process.communicate()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.write_text(output)
    return {
        "command": command,
        "returncode": process.returncode,
        "timeout": timed_out,
        "seconds": time.perf_counter() - started,
        "log": str(logfile),
        "status": "passed" if process.returncode == 0 and not timed_out else "failed",
    }


def ddp_cases(output, processes, precision="bf16", timeout=300):
    if processes < 2:
        raise ValueError("DDP acceptance requires at least two visible CUDA GPUs")
    records = []
    for name in ("mf_et_jar",):
        run_dir = output / "ddp" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = load_config(ROOT / "configs/mf_et_jar_wide4_tiny.json")
        cfg["synthetic"]["num_cameras"] = 3
        cfg["train"].update(precision=precision, max_steps=2)
        config_path = run_dir / "launch_config.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(processes),
            str(ROOT / "train.py"),
            "--config",
            str(config_path),
            "--device",
            "cuda",
            "--synthetic",
            "--output",
            str(run_dir),
        ]
        initial = bounded_command(command, run_dir / "launch.log", timeout)
        record = {
            "model": name,
            "world_size": processes,
            "precision": precision,
            "initial": initial,
            "status": "failed",
        }
        if initial["status"] == "passed":
            checkpoint = run_dir / "last.pt"
            cfg["train"]["max_steps"] = 3
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            resumed = bounded_command(
                command + ["--resume", str(checkpoint)],
                run_dir / "resume.log",
                timeout,
            )
            record["resume"] = resumed
            if resumed["status"] == "passed":
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                finite_tree(saved["model"], "checkpoint.model")
                record["saved_step"] = int(saved["step"])
                record["status"] = "passed" if record["saved_step"] == 3 else "failed"
        records.append(record)
    return records


def unlabeled_fixture(directory):
    """Write actual RGB files and calibration, deliberately omit every label.

    This exercises the real-file interface but remains synthetic scene content.
    """
    import numpy as np
    from PIL import Image

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sample = make_synthetic_batch(N=3, H=32, W=48, seed=23)
    inputs = sample["inputs"]
    cameras = []
    for index in range(3):
        filename = f"camera{index}.png"
        rgb = inputs["rgb"][0, index].permute(1, 2, 0).numpy()
        Image.fromarray(np.rint(rgb * 255).clip(0, 255).astype(np.uint8)).save(directory / filename)
        cameras.append(
            {
                "camera_id": str(index),
                "frame_id": "0",
                "timestamp": 0.0,
                "rgb": filename,
                "K": inputs["K"][0, index].tolist(),
                "T_cam_from_rig": inputs["T_cam_from_rig"][0, index].tolist(),
            }
        )
    frame = {
        "sequence": "analytic_unlabeled_fixture",
        "frame_id": "0",
        "timestamp": 0.0,
        "cameras": cameras,
    }
    (directory / "manifest.jsonl").write_text(json.dumps(frame) + "\n")
    return directory


def infer_cli_cases(output, precision="bf16", device_index=0, timeout=300):
    """Use completed DDP checkpoints through infer.py and validate NPZ exports."""
    import numpy as np

    fixture = unlabeled_fixture(output / "unlabeled_fixture")
    records = []
    for name in ("mf_et_jar",):
        checkpoint = output / "ddp" / name / "last.pt"
        record = {
            "model": name,
            "status": "failed",
            "fixture_is_real_capture": False,
            "interface": "RGB PNG + calibration manifest without targets",
            "checkpoint": str(checkpoint),
        }
        if not checkpoint.exists():
            record["error"] = "DDP checkpoint 不存在，不能执行带权重的真实接口推理。"
            records.append(record)
            continue
        cfg = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]
        cfg["data"] = {
            "root": str(fixture),
            "train_manifest": "",
            "val_manifest": "",
            "test_manifest": "manifest.jsonl",
        }
        cfg["eval"].update(precision=precision, warmup=1)
        config_path = output / "unlabeled_fixture" / f"{name}.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        destination = output / "infer_cli" / name
        command = [
            sys.executable,
            str(ROOT / "infer.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            f"cuda:{device_index}",
            "--limit",
            "1",
            "--output",
            str(destination),
        ]
        record["command"] = bounded_command(command, destination / "launch.log", timeout)
        if record["command"]["status"] == "passed":
            try:
                summary = json.loads((destination / "inference.json").read_text())
                if (
                    summary["labels_loaded"]
                    or summary["synthetic"]
                    or summary["untrained"]
                    or len(summary["samples"]) != 1
                ):
                    raise AssertionError("推理没有使用预期的无标签文件接口或checkpoint。")
                archive = destination / summary["samples"][0]["prediction_file"]
                with np.load(archive, allow_pickle=False) as data:
                    for key in (
                        "points",
                        "normals",
                        "scales",
                        "valid",
                        "support",
                        "confidence",
                        "state",
                        "centers",
                        "sizes",
                    ):
                        if key not in data:
                            raise AssertionError(f"预测导出缺少 {key}")
                    for key in data.files:
                        if (
                            np.issubdtype(data[key].dtype, np.floating)
                            and not np.isfinite(data[key]).all()
                        ):
                            raise FloatingPointError(f"导出数组 {key} 含 NaN/Inf")
                    if len(data["state"]) != len(data["points"]):
                        raise AssertionError("状态和点数量不一致")
                    record["emitted_queries_or_leaves"] = len(data["state"])
                    record["free_surface_unknown_counts"] = [
                        int((data["state"] == state).sum()) for state in range(3)
                    ]
                info = json.loads(archive.with_suffix(".json").read_text())
                if info["contains_ground_truth"]:
                    raise AssertionError("推理导出不应含GT标签")
                record["export"] = str(archive)
                record["status"] = "passed"
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CUDA tiny 验收；无 GPU 非零退出，绝不作为 CPU 通过"
    )
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "gpu_acceptance"))
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--precisions", nargs="+", choices=("fp32", "bf16"), default=["fp32", "bf16"]
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--ddp", action="store_true", help="额外运行最终模型 2 步 DDP + 恢复到第 3 步"
    )
    parser.add_argument("--ddp-processes", type=int, default=None)
    parser.add_argument(
        "--infer-cli",
        action="store_true",
        help="需同时 --ddp：用恢复后的checkpoint测试无标签文件推理和NPZ导出",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    if (
        args.require_gpus < 1
        or args.device_index < 0
        or args.warmup < 0
        or args.repeats < 1
        or args.timeout < 1
    ):
        parser.error("GPU 数量、设备编号、预热/重复次数或超时参数无效")
    if args.infer_cli and not args.ddp:
        parser.error("--infer-cli 需要 --ddp 提供实际训练及恢复后的checkpoint")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "not_verified",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "CUDA tiny 内核和合成训练管线验收；未验证真实精度、全分辨率或吞吐量。",
        "model_cases": [],
        "ddp_cases": [],
        "infer_cli_cases": [],
        "model_tested_device": args.device_index,
        "not_verified": [
            "CARLA/真实数据",
            "全分辨率和最大预算",
            "长训练收敛与重建精度",
            "部署吞吐量",
        ],
    }
    destination = output / "report.json"

    def save():
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    try:
        report["preflight"] = collect_preflight(
            args.require_gpus, "bf16" in args.precisions, args.timeout
        )
        save()
        if report["preflight"]["status"] != "passed":
            report["message_zh"] = "GPU 预检未通过；未执行模型 CUDA 或 DDP 验收，不能认证可用。"
            save()
            print(report["message_zh"], flush=True)
            return 2
        if args.device_index >= torch.cuda.device_count():
            raise ValueError("device-index 超出可见 CUDA 设备范围")
        for cameras in (2, 3, 4):
            for precision in args.precisions:
                try:
                    result = model_case(
                        "mf_et_jar",
                        cameras,
                        precision,
                        args.device_index,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "model": "mf_et_jar",
                        "camera_count": cameras,
                        "precision": precision,
                        "error": f"{type(error).__name__}: {error}",
                    }
                report["model_cases"].append(result)
                save()
                print(f"{result['status'].upper()} MF-ET-JAR N={cameras} {precision}", flush=True)
        if args.ddp:
            processes = args.ddp_processes if args.ddp_processes is not None else args.require_gpus
            if not 2 <= processes <= torch.cuda.device_count():
                raise ValueError(
                    "DDP 进程数必须在 2 与可见 GPU 数量之间；可用 --ddp-processes 指定"
                )
            if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
                raise RuntimeError("当前 PyTorch 缺少 NCCL，不能认证 CUDA DDP。")
            # Free the local model allocations before launching one rank per GPU.
            gc.collect()
            torch.cuda.empty_cache()
            report["ddp_cases"] = ddp_cases(
                output, processes, "bf16" if "bf16" in args.precisions else "fp32", args.timeout
            )
        else:
            report["not_verified"].append("DDP/NCCL（未指定 --ddp）")
        if args.infer_cli:
            report["infer_cli_cases"] = infer_cli_cases(
                output,
                "bf16" if "bf16" in args.precisions else "fp32",
                args.device_index,
                args.timeout,
            )
        else:
            report["not_verified"].append("infer.py 无标签文件读取及NPZ导出（未指定 --infer-cli）")
        passed = all(
            case["status"] == "passed"
            for case in report["model_cases"] + report["ddp_cases"] + report["infer_cli_cases"]
        )
        report["status"] = "passed" if passed else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    save()
    print(f"{report['status']}: {destination}", flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
