"""直接执行官方与集成版损失代码，检查数值和梯度；无需实例化 ViT。

CPU 时只拦截官方硬编码的 Tensor.cuda()，不改损失公式。
全空监督例外：官方总损失为 NaN，集成版应是图连通的有限零。
"""

import argparse
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from official_methods.methods.niantic_common import UPSTREAM


def load_implementation(source, name):
    package = source / "src/mvsanywhere"
    filename = package / "experiment_modules/sr_depth_model.py"
    tree = ast.parse(filename.read_text())
    model = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DepthModel"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_losses"
    )
    namespace = {"torch": torch, "np": np, "nn": nn, "F": F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(filename), "exec"), namespace)
    spec = importlib.util.spec_from_file_location(name, package / "losses.py")
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)
    return namespace["compute_losses"], losses


def loss_fixture(case, device):
    from mvsanywhere.utils.geometry_utils import NormalGenerator

    height, width = 32, 48
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    depth = (5 + cols * 0.02 + rows * 0.01)[None, None].float()
    valid = torch.ones_like(depth, dtype=torch.bool)
    if case == "sparse_patch":
        valid.zero_()
        valid[:, :, 8:21, 10:23] = True
    elif case == "empty":
        valid.zero_()
    depth = depth.masked_fill(~valid, torch.nan)
    prediction = (6 + cols * 0.024 + rows * 0.013 + torch.sin(cols.float()) * 0.01)[None, None]
    prediction.requires_grad_()
    normals = NormalGenerator(height, width).to(device)
    inv_k = torch.eye(4, device=device)[None]
    sky = torch.zeros_like(depth) if case == "dense_sky" else torch.full_like(depth, torch.nan)
    current = {
        "depth_b1hw": depth,
        "normals_b3hw": normals(depth, inv_k),
        "mask_b_b1hw": valid,
        "mask_b1hw": valid.float(),
        "skymask_b1hw": sky,
    }
    output = {
        "depth_pred_s0_b1hw": prediction,
        "log_depth_pred_s0_b1hw": prediction.log(),
        "normals_pred_b3hw": normals(prediction, inv_k),
    }
    for scale in range(4):
        shape = (height // 2**scale, width // 2**scale)
        output[f"log_depth_pred_s{scale}_b1hw"] = F.interpolate(
            prediction.log(), shape, mode="nearest"
        )
        # 官方 forward 的 sky 字段已经经过 sigmoid；按原行为继续喂入其原损失。
        output[f"sky_pred_s{scale}_b1hw"] = F.interpolate(
            prediction.sigmoid(), shape, mode="nearest"
        )
    return current, output, prediction


def evaluate(function, modules, case, device, source_depth=1.0):
    current, outputs, prediction = loss_fixture(case, device)
    context = SimpleNamespace(
        ms_loss_fn=nn.L1Loss(),
        abs_loss=nn.L1Loss(),
        si_loss=modules.ScaleInvariantLoss(),
        grad_loss=modules.MSGradientLoss(),
        normals_loss=modules.NormalsLoss(),
    )
    cpu_cuda_compat = patch.object(torch.Tensor, "cuda", lambda value, *args, **kwargs: value)
    with cpu_cuda_compat if device.type == "cpu" else nullcontext():
        losses = function(
            context, current, {"depth_b1hw": torch.tensor(source_depth, device=device)}, outputs
        )
    losses["loss"].backward()
    gradient = prediction.grad.detach()
    assert torch.isfinite(gradient).all()
    numbers = {
        key: float(value.detach()) if torch.isfinite(value) else None
        for key, value in losses.items()
    }
    return numbers, gradient


def run(official_source, device):
    sys.path.insert(0, str(UPSTREAM / "mvsanywhere/src"))
    implementations = [
        load_implementation(official_source, "pristine_mvsanywhere_losses"),
        load_implementation(UPSTREAM / "mvsanywhere", "integrated_mvsanywhere_losses"),
    ]
    cases = {}
    for case in ("dense", "dense_sky", "sparse_patch", "empty"):
        original, integrated = [
            evaluate(*implementation, case, device) for implementation in implementations
        ]
        if case == "empty":
            assert original[0]["loss"] is None and integrated[0]["loss"] == 0
        else:
            assert original[0] == integrated[0], (case, original[0], integrated[0])
        torch.testing.assert_close(original[1], integrated[1], rtol=0, atol=0)
        altered = evaluate(*implementations[1], case, device, source_depth=1234.0)
        assert integrated[0] == altered[0]
        torch.testing.assert_close(integrated[1], altered[1], rtol=0, atol=0)
        cases[case] = {
            "official": original[0],
            "integrated": integrated[0],
            "gradient_max_difference": float((original[1] - integrated[1]).abs().max()),
            "source_depth_independent": True,
        }
    return {
        "device": str(device),
        "torch": torch.__version__,
        "cases": cases,
        "cpu_device_compatibility": "仅在CPU测试期间将官方硬编码Tensor.cuda替换为恒等设备操作",
        "scope": "原compute_losses及各自官方损失类；不实例化完整Hero。null表示官方空监督NaN诊断项。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run(args.official_source, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
