"""TransUNet启发的输入融合成本探针；仅测算子，不接入生产模型或宣称精度。"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.nn import functional as F


class RGBResidual(nn.Module):
    """保持三通道的局部残差，原RGB路径不变。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 3, 1, bias=False),
        )
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, image):
        return image + 0.1 * self.layers(image)


class TokenBridge(nn.Module):
    """复用已算出的参考CNN特征，对原ViT patch token增加局部补充。"""

    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
        self.norm = nn.GroupNorm(8, 64)
        self.projection = nn.Conv2d(64, 768, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        # 匹配格点原图坐标为4j；8格平均后中心14+32j。
        # 主图ViT token中心7.5+16j对应匹配图15.5+32j。
        # 先面积聚合，再对齐中心；这不是无损的高频重建。
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(28), indexing="ij")
        grid = torch.stack((2 * (xx + 0.5 + 1.5 / 32) / 28 - 1,
                            2 * (yy + 0.5 + 1.5 / 32) / 16 - 1), dim=-1)
        self.register_buffer("grid", grid[None])

    def forward(self, feature, token):
        value = F.avg_pool2d(F.gelu(self.local(feature)), 8)
        value = F.grid_sample(value, self.grid.expand(len(value), -1, -1, -1),
                              align_corners=False, padding_mode="border")
        return token + 0.1 * self.projection(self.norm(value))


def main():
    torch.manual_seed(20261031)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    rgb = RGBResidual().to(device).eval()
    bridge = TokenBridge().to(device).eval()
    reference = torch.randn(1, 3, 256, 448, device=device)
    images = torch.randn(4, 3, 512, 896, device=device)
    features = torch.randn(4, 64, 128, 224, device=device)
    tokens = torch.randn(4, 768, 16, 28, device=device)
    calls = {
        "rgb_reference": lambda: rgb(reference),
        "rgb_both_branches": lambda: (rgb(reference), rgb(images)),
        "bridge_reference": lambda: bridge(features[:1], tokens[:1]),
        "bridge_four_batched": lambda: bridge(features, tokens),
    }
    checks = {}
    for name, module, call, expected in (
        ("rgb", rgb, calls["rgb_reference"], reference),
        ("bridge", bridge, calls["bridge_reference"], tokens[:1]),
    ):
        output = call()
        assert torch.equal(output, expected)
        output.square().mean().backward()
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        checks[name] = {
            "zero_output_projection_preserves_input": True,
            "finite_gradients": all(bool(g.isfinite().all()) for g in gradients),
            "has_nonzero_gradient": any(bool(g.abs().sum() > 0) for g in gradients),
        }
        assert checks[name]["finite_gradients"] and checks[name]["has_nonzero_gradient"]
        module.zero_grad(set_to_none=True)

    macs = {}
    hooks = []
    counter = [0]

    def count_conv(layer, inputs, output):
        counter[0] += output.numel() * layer.weight[0].numel()

    for module in (rgb, bridge):
        hooks.extend(layer.register_forward_hook(count_conv)
                     for layer in module.modules() if isinstance(layer, nn.Conv2d))
    with torch.inference_mode():
        for name, call in calls.items():
            counter[0] = 0
            call()
            macs[name] = counter[0]
    for hook in hooks:
        hook.remove()

    measurements = {name: [] for name in calls}
    with torch.inference_mode():
        for call in calls.values():
            for _ in range(20):
                call()
        torch.cuda.synchronize()
        for iteration in range(100):
            names = list(calls)
            names = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in names:
                torch.cuda.synchronize()
                start = time.perf_counter()
                calls[name]()
                torch.cuda.synchronize()
                measurements[name].append((time.perf_counter() - start) * 1000)

    result = {
        "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "protocol": "FP32, TF32 off, eager, GPU resident synthetic tensors; 20 warmup/100 alternating synchronized calls",
        "scope": "operator-only: excludes original encoders, candidate decoder, masking and training; not end-to-end latency or accuracy",
        "parameters": {"rgb": sum(p.numel() for p in rgb.parameters()),
                       "bridge": sum(p.numel() for p in bridge.parameters())},
        "checks": checks,
        "cases": {name: {"median_ms": statistics.median(values),
                         "p10_ms": sorted(values)[9], "p90_ms": sorted(values)[89],
                         "conv_gmac": macs[name] / 1e9}
                  for name, values in measurements.items()},
    }
    output = Path(__file__).resolve().parents[2] / "runs/active/cnn_token_bridge_20261031"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "cost_probe.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
