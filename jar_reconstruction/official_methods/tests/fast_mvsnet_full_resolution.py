"""默认 CARLA 尺寸、完整官方网络及 GN 的一次前向/损失/反向验收。"""

import argparse
import json
from pathlib import Path
import time

import torch

from jar.synthetic import make_synthetic_batch
from official_methods.methods.fast_mvsnet import Adapter


def run():
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    config = json.loads(Path("official_methods/configs/fast_mvsnet_carla.json").read_text())
    model = Adapter(config).cuda().train()
    model.set_epoch(config["training"]["pretrain_epochs"])
    h, w = config["image_size"]
    batch = make_synthetic_batch(B=1, N=4, H=h, W=w, seed=1701)
    inputs = {key: value.cuda() for key, value in batch["inputs"].items()}
    targets = {key: value.cuda() for key, value in batch["targets"].items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model(inputs)
    losses = model.loss(output, inputs, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients.values())
    core = {
        branch: any(
            name.startswith("native." + branch) and bool(grad.abs().sum() > 0)
            for name, grad in gradients.items()
        )
        for branch in ("coarse_vol_conv", "propagation_net", "flow_img_conv")
    }
    assert all(core.values()), core
    assert "flow2" in output["native"][0]
    torch.cuda.synchronize()
    return dict(
        method="fast_mvsnet",
        status="passed",
        gpu=torch.cuda.get_device_name(),
        image_size=[h, w],
        batch_size=1,
        views=4,
        depth_planes=model._depth_planes(),
        gauss_newton=model.use_gn,
        parameter_count=sum(p.numel() for p in model.parameters()),
        precision="fp32",
        depth_shape=list(output["depth"].shape),
        loss=float(losses["total"].detach()),
        finite_gradient_parameters=len(gradients),
        core_nonzero_gradients=core,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.perf_counter() - start,
        optimizer_step=False,
        pretrained=False,
        scope="CARLA默认训练尺寸/48深度平面；官方全部模块及Gauss-Newton；只验证一次合成前向损失反向，不含优化器状态，不证明真实精度。",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="official_methods/artifacts/fast_mvsnet_carla_shape.json"
    )
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
