"""将已有ResNet局部特征对齐到ViT token；不读取深度或改变几何。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


LAYER_SPECS = {"layer1": (64, 4), "layer2": (128, 8)}


def feature_validity(image_mask, shape, stride, dtype):
    """在ResNet真实格点中心读取有效域，供投影前和面积聚合共同使用。"""
    height, width = shape
    match_h, match_w = image_mask.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=image_mask.device, dtype=dtype),
        torch.arange(width, device=image_mask.device, dtype=dtype), indexing="ij")
    grid = torch.stack((2 * (xx * stride + .5) / match_w - 1,
                        2 * (yy * stride + .5) / match_h - 1), -1)
    return F.grid_sample(image_mask[:, None].to(dtype),
                         grid[None].expand(len(image_mask), -1, -1, -1),
                         mode="nearest", align_corners=False)


def aligned_subcells(feature, image_mask, main_size, patch_shape, stride, subcells=2, validity=None):
    """面积聚合后精确对齐，输出[B,L,子格数*C]及[B,L]有效覆盖。

    ResNet格点中心为stride*j；ViT使用scale_factor=14/16缩放原图。
    每个patch内按行优先保留subcells×subcells个区域，不先平均成一个值。
    分子与掩码执行相同聚合/插值，之后才相除，避免缺图区污染有效值。
    """
    batch, channels, height, width = feature.shape
    main_h, main_w = main_size
    match_h, match_w = image_mask.shape[-2:]
    patch_h, patch_w = patch_shape
    dtype, device = feature.dtype, feature.device
    coord_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    mask = feature_validity(image_mask, (height, width), stride, coord_dtype) if validity is None else validity
    mask = mask.to(dtype)
    clean = torch.where(mask > 0, feature, 0)
    footprint_y = 16 * match_h / main_h / subcells / stride
    footprint_x = 16 * match_w / main_w / subcells / stride
    kernel = (min(height, max(1, int(math.floor(footprint_y)))),
              min(width, max(1, int(math.floor(footprint_x)))))
    numerator = F.avg_pool2d(clean * mask, kernel, stride=kernel)
    denominator = F.avg_pool2d(mask, kernel, stride=kernel)

    yy, xx = torch.meshgrid(
        torch.arange(patch_h * subcells, device=device, dtype=coord_dtype),
        torch.arange(patch_w * subcells, device=device, dtype=coord_dtype), indexing="ij")
    # ViT子格中心先映回主图，再用半像素约定映到匹配图。
    match_x = (xx + .5) * (16 / subcells) * match_w / main_w - .5
    match_y = (yy + .5) * (16 / subcells) * match_h / main_h - .5
    pooled_x = (match_x / stride - (kernel[1] - 1) / 2) / kernel[1]
    pooled_y = (match_y / stride - (kernel[0] - 1) / 2) / kernel[0]
    grid = torch.stack((2 * (pooled_x + .5) / numerator.shape[-1] - 1,
                        2 * (pooled_y + .5) / numerator.shape[-2] - 1), -1)
    grid = grid[None].expand(batch, -1, -1, -1)
    numerator = F.grid_sample(numerator.to(coord_dtype), grid, align_corners=False)
    denominator = F.grid_sample(denominator.to(coord_dtype), grid, align_corners=False)
    value = torch.where(denominator > 1e-6, numerator / denominator.clamp_min(1e-6), 0).to(dtype)
    value = value.reshape(batch, channels, patch_h, subcells, patch_w, subcells)
    value = value.permute(0, 2, 4, 3, 5, 1).reshape(batch, patch_h * patch_w, -1)
    coverage = denominator.reshape(batch, 1, patch_h, subcells, patch_w, subcells)
    coverage = coverage.mean((1, 3, 5)).flatten(1).clamp(0, 1)
    return value, coverage


class CNNTokenBridge(nn.Module):
    """只增量修正patch token；不改CLS、位置编码或原token归一化。"""

    def __init__(self, options=None):
        super().__init__()
        self.options = {
            "enabled": True, "layers": ["layer1", "layer2"], "bottleneck": 32,
            "hidden": 128, "subcells": 2, "detach_backbone": True,
            "residual_scale": .1, **(options or {}),
        }
        self.enabled = bool(self.options["enabled"])
        self.layers = tuple(self.options["layers"])
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(
                layer not in LAYER_SPECS for layer in self.layers):
            raise ValueError("bridge.layers须为不重复的layer1/layer2列表")
        self.subcells = int(self.options["subcells"])
        channels, hidden = int(self.options["bottleneck"]), int(self.options["hidden"])
        if self.subcells not in (1, 2) or channels < 1 or hidden < 1:
            raise ValueError("bridge使用1或2子格，通道和隐藏维度必须为正")
        self.scale = float(self.options["residual_scale"])
        if not math.isfinite(self.scale) or not 0 < self.scale <= 1:
            raise ValueError("bridge.residual_scale须在(0,1]，不能与末投影同时置零")
        self.detach_backbone = bool(self.options["detach_backbone"])
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(LAYER_SPECS[name][0], channels, 1, bias=False)
            for name in self.layers
        })
        width = len(self.layers) * channels * self.subcells ** 2
        self.norm = nn.LayerNorm(width)
        self.fusion = nn.Sequential(nn.Linear(width, hidden), nn.GELU())
        self.output = nn.Linear(hidden, 768, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, tokens, features, matching_mask, image_mask, patch_shape):
        if not self.enabled:
            return tokens
        parts, support = [], []
        for name in self.layers:
            value = features[name].detach() if self.detach_backbone else features[name]
            coord_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            validity = feature_validity(matching_mask, value.shape[-2:], LAYER_SPECS[name][1], coord_dtype)
            # 卷积前清除无效值，避免反向时零梯度乘NaN污染投影参数。
            value = torch.where(validity > 0, value, 0)
            value = F.gelu(self.projections[name](value))
            part, coverage = aligned_subcells(
                value, matching_mask, image_mask.shape[-2:], patch_shape,
                LAYER_SPECS[name][1], self.subcells, validity)
            parts.append(part)
            support.append(coverage)
        local = torch.cat(parts, -1)
        delta = self.output(self.fusion(self.norm(local)))
        mask = F.interpolate(image_mask[:, None].float(), scale_factor=14 / 16,
                             mode="bilinear", align_corners=False)
        mask = F.avg_pool2d(mask, 14, stride=14).flatten(1)
        coverage = torch.stack(support).amax(0) * mask
        delta = torch.where(coverage[..., None] > 0, delta * coverage[..., None], 0)
        return tokens + self.scale * delta.to(tokens.dtype)
