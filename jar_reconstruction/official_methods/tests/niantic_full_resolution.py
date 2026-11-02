"""交付 CARLA 默认分辨率的一次单卡完整训练步验证，不生成性能指标。"""

import argparse
import importlib
import json
from pathlib import Path
import time
import torch
from jar.synthetic import make_synthetic_batch


def run(method):
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    config = json.loads(Path(f"official_methods/configs/{method}_carla.json").read_text())
    config["model"]["pretrained"] = False
    cls = importlib.import_module(f"official_methods.methods.{method}").Adapter
    model = cls(config).cuda().train()
    H, W = config["model"]["image_height"], config["model"]["image_width"]
    batch = make_synthetic_batch(B=1, N=4, H=H, W=W, seed=1701)
    inputs = {key: val.cuda() for key, val in batch["inputs"].items()}
    targets = {key: val.cuda() for key, val in batch["targets"].items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    inputs = model.prepare_training_inputs(inputs, targets)
    output = model(inputs)
    losses = model.loss(output, inputs, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for name in ("encoder", "matching_model", "cost_volume", "cost_volume_net", "depth_decoder"):
        assert any(
            p.grad is not None and (p.grad != 0).any()
            for p in getattr(model.native_model, name).parameters()
        ), name
    torch.cuda.synchronize()
    return dict(
        method=method,
        status="passed",
        device=torch.cuda.get_device_name(),
        image_size=[H, W],
        batch_size=1,
        num_views=4,
        depth_bins=config["model"]["depth_bins"],
        parameter_count=sum(p.numel() for p in model.parameters()),
        precision="fp32",
        finite_nonzero_core_gradients=True,
        loss=float(losses["total"].detach()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        one_forward_loss_backward_seconds=time.perf_counter() - start,
        optimizer_step=False,
        pretrained=False,
        scope="一次真实几何合成训练步；显存包含模型、输入、激活、梯度，不含AdamW状态，不证明正式精度/稳定吞吐。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["simplerecon", "mvsanywhere"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.method)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
