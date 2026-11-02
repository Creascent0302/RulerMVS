"""真实几何合成输入，完整官方架构冒烟测试；不会输出精度声明。"""

import argparse
import importlib
import json
from pathlib import Path
import tempfile
import time
import torch
from jar.synthetic import make_synthetic_batch
from official_methods.methods.niantic_common import native_intrinsics, depth_intrinsics


def run(method, device, checkpoint_root=None):
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    adapter = importlib.import_module(f"official_methods.methods.{method}").Adapter
    cfg = {
        "method": method,
        "model": {
            "num_views": 4,
            "image_height": 64,
            "image_width": 96,
            "depth_bins": 64,
            "pretrained": False,
            "min_depth": 2.0,
            "max_depth": 80.0,
            "range_refinement_steps": 1,
            "reference_indices": [0],
        },
        "training": {"geometric_flip": True},
    }
    batch = make_synthetic_batch(B=1, N=4, H=64, W=96, seed=1701)
    inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    begin = time.perf_counter()
    model = adapter(cfg).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
        cur, src, _ = model.prepare_native(inputs, 0)
        native = model.call_native(cur, src, "test")
        equality = (
            (pred["native"][0]["depth_pred_s0_b1hw"] - native["depth_pred_s0_b1hw"])
            .abs()
            .max()
            .item()
        )
        assert equality == 0, equality
        changed = {k: v.clone() for k, v in targets.items()}
        changed["depth"] *= 3
        # 推理函数根本不接收 targets；额外放入训练内部字段也必须忽略。
        leaked = dict(
            inputs,
            _training_depth_range=torch.full((1, 4, 2), 99.0, device=device),
            _geometric_flip=True,
        )
        result2 = model(leaked)
        assert torch.equal(pred["depth"], result2["depth"])
        assert torch.isfinite(pred["depth"]).all()
        assert pred["depth"].shape == (
            1,
            1,
            int(64 * model.prediction_scale),
            int(96 * model.prediction_scale),
        )
        # 原生光栅使用半像素中心；其同一射线与共享整数中心光栅完全相等。
        kn = native_intrinsics(inputs["K"][:, 0], 0.25)[:, :3, :3]
        kc = depth_intrinsics(inputs["K"][:, 0], 0.25)
        pix = torch.tensor([3.0, 5.0, 1.0], device=device)
        pn = pix + pix.new_tensor([0.5, 0.5, 0.0])
        assert torch.allclose(torch.linalg.solve(kn, pn), torch.linalg.solve(kc, pix), atol=1e-7)
    model.train()
    train_inputs = model.prepare_training_inputs(inputs, targets)
    predicted = model(train_inputs)
    losses = model.loss(predicted, train_inputs, targets)
    assert torch.isfinite(losses["total"]), losses
    losses["total"].backward()
    groups = {}
    for name in ("encoder", "matching_model", "cost_volume", "cost_volume_net", "depth_decoder"):
        params = list(getattr(model.native_model, name).parameters())
        gradients = [p.grad for p in params if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        magnitude = sum(g.abs().sum().item() for g in gradients)
        assert magnitude > 0, (name, magnitude)
        groups[name] = {"gradient_tensors": len(gradients), "absolute_sum": magnitude}
    optimizer, scheduler = model.configure_optimizer()
    first = next(p for p in model.parameters() if p.grad is not None)
    before = first.detach().clone()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, first.detach())
    del predicted, losses, train_inputs, optimizer, scheduler
    model.eval()
    with torch.no_grad():
        expected = model(inputs)["depth"].clone()
    if checkpoint_root is not None:
        Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{method}_checkpoint_", dir=checkpoint_root or "/tmp"
    ) as temp:
        path = Path(temp) / "model.pt"
        torch.save({"model": model.state_dict(), "config": cfg}, path)
        # 实际破坏一个权重，再严格重新加载，以验证加载确实恢复模型。
        with torch.no_grad():
            first.add_(1.0)
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            restored = model(inputs)["depth"]
        assert torch.equal(expected, restored)
        bytes_saved = path.stat().st_size
        del state
    report = {
        "method": method,
        "device": str(device),
        "status": "passed",
        "torch": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "shape": list(pred["depth"].shape),
        "depth_bins": 64,
        "pretrained": False,
        "native_adapter_max_difference": equality,
        "finite_nonzero_gradients": groups,
        "checkpoint_bytes": bytes_saved,
        "checkpoint_reloaded": True,
        "label_leakage_test": "passed",
        "half_pixel_geometry_test": "passed",
        "optimizer_step_changed_weights": True,
        "elapsed_seconds": time.perf_counter() - begin,
        "interpretation": "仅验证真实网络和训练管线可运行，未验证重建性能。",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["simplerecon", "mvsanywhere"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.method, args.device, checkpoint_root=destination.parent)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
