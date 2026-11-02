"""几何一致场景上的官方网络验收；可作为脚本记录 CPU/CUDA 实测。"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch

from jar.synthetic import make_synthetic_batch
from official_methods.methods.fast_mvsnet import Adapter, UPSTREAM, COMMIT


def run(device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    config = {
        "model": {
            "depth_range": [1.0, 16.0],
            "train_depth_planes": 16,
            "test_depth_planes": 16,
            "reference_indices": [0],
        },
        "training": {"pretrain_epochs": 4},
    }
    batch = make_synthetic_batch(B=1, N=4, H=64, W=128, seed=1701)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["inputs"].items()}
    targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["targets"].items()}
    model = Adapter(config).to(device)
    report = {
        "device": device,
        "torch": torch.__version__,
        "commit": COMMIT,
        "image_size": [64, 128],
        "views": 4,
        "depth_planes": 16,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "pretrained": False,
        "custom_cuda_ops": "none",
    }
    if device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name()
    for epoch in [0, 4]:
        model.train()
        model.set_epoch(epoch)
        model.zero_grad(set_to_none=True)
        predictions = model(inputs)
        losses = model.loss(predictions, inputs, targets)
        assert all(torch.isfinite(v).all() for v in losses.values())
        losses["total"].backward()
        gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
        assert any(
            name.startswith("native.coarse_vol_conv") and g.abs().sum() > 0
            for name, g in gradients.items()
        )
        assert any(
            name.startswith("native.propagation_net") and g.abs().sum() > 0
            for name, g in gradients.items()
        )
        if epoch == 4:
            assert any(
                name.startswith("native.flow_img_conv") and g.abs().sum() > 0
                for name, g in gradients.items()
            )
            assert "flow2" in predictions["native"][0]
        else:
            assert "flow2" not in predictions["native"][0]
        opt, _ = model.configure_optimizer()
        opt.step()
        report[f"epoch_{epoch}"] = {
            "finite_backward": True,
            "gradient_parameters": len(gradients),
            "loss": float(losses["total"].detach()),
            "depth_shape": list(predictions["depth"].shape),
        }
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        # 相同预处理后的输入直接调用官方模型；不是与重写网络比较。
        native = model.native(
            model.prepare_native(inputs, 0), (0.25, 0.5), (0.75, 0.15), True, isTest=True
        )
        max_error = (predictions["depth"] - native["flow2"]).abs().max().item()
        assert max_error == 0.0
        report["native_adapter_max_error"] = max_error
        # 修改标签后 eval 推理不变；推理函数根本不接收 targets。
        changed = {"depth": targets["depth"] * 10.0}
        model.loss(predictions, inputs, changed)
        again = model(inputs)
        assert torch.equal(predictions["depth"], again["depth"])
        report["no_label_leakage"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "config": config}, path)
            restored = Adapter(config).to(device).eval()
            restored.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)["model"]
            )
            assert torch.equal(predictions["depth"], restored(inputs)["depth"])
        report["checkpoint_exact"] = True
        model.config["evaluation"] = {"reference_indices": [0, 1, 2, 3]}
        four = model(inputs)
        assert four["depth"].shape[:2] == (1, 4) and torch.isfinite(four["depth"]).all()
        report["four_reference_shape"] = list(four["depth"].shape)
    # 数学验证：深度网格中的一个像素，对应输入(2*u+.5, 2*v+.5)。
    pixel = torch.tensor([7.0, 9.0, 1.0], device=device)
    ray_small = torch.linalg.solve(four["depth_K"][0, 0], pixel)
    ray_input = torch.linalg.solve(
        inputs["K"][0, 0], torch.tensor([14.5, 18.5, 1.0], device=device)
    )
    assert torch.allclose(ray_small, ray_input, atol=1e-6)
    report["pixel_center_ray_consistency"] = True
    # 官方 feature sampler 的恒等投影及其对特征的梯度。
    from fastmvsnet.utils.feature_fetcher import FeatureFetcher

    features = torch.arange(12.0, device=device).view(1, 1, 1, 3, 4).requires_grad_()
    intrinsic = torch.eye(3, device=device).view(1, 1, 3, 3)
    extrinsic = torch.eye(4, device=device)[:3].view(1, 1, 3, 4)
    point = torch.tensor([2.5, 1.5, 1.0], device=device).view(1, 3, 1)
    sampled = FeatureFetcher()(features, point, intrinsic, extrinsic)
    assert torch.allclose(sampled, torch.tensor(6.0, device=device))
    sampled.sum().backward()
    assert torch.allclose(features.grad.sum(), torch.tensor(1.0, device=device))
    report["native_sampler_identity_backward"] = True
    # 包含在官方仓库的权重可严格加载。此步骤不声称CARLA效果。
    pretrained = Adapter(config)
    pretrained.load_native_weights(UPSTREAM / "outputs" / "pretrained.pth")
    report["official_checkpoint_strict_load"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    start = time.perf_counter()
    report = run(args.device)
    report["elapsed_s"] = time.perf_counter() - start
    report["status"] = "passed"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
