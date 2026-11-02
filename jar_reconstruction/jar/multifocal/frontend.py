"""同一编码器按真实图像尺寸编码；独立长焦分段逆深度初始化。"""

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from jar.coarse import PlaneSweep


def segmented_depths(near, far, bins, device=None):
    """近中远三段分别均匀逆深度，不用单段48候选覆盖150米。"""
    if not 0 < near < far or bins < 6:
        raise ValueError("分段深度要求0<near<far且bins>=6")
    edges = torch.logspace(math.log10(near), math.log10(far), 4, device=device)
    counts = [bins // 3 + (i < bins % 3) for i in range(3)]
    parts = [
        torch.linspace(1 / float(edges[i]), 1 / float(edges[i + 1]), counts[i] + 1, device=device)[
            :-1
        ].reciprocal()
        for i in range(3)
    ]
    result = torch.cat(parts)
    result[-1] = far
    return result


def camera_subset(inputs, ids):
    """只切相机轴；标签不会传入此函数。"""
    out = dict(inputs)
    batch, views = inputs["rgb"].shape[:2]
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[:2] == (batch, views):
            out[k] = v[:, ids]
        elif k == "camera_ids":
            out[k] = [[row[i] for i in ids] for row in v]
    return out


def encode_native_sizes(encoder, inputs):
    """防止GroupNorm被其他相机的padding区域改变；仅输出特征做零padding。"""
    if "image_sizes" not in inputs:
        return encoder(inputs)
    bsize, views, _, height, width = inputs["rgb"].shape
    groups = {}
    for b in range(bsize):
        for v in range(views):
            if bool(inputs["view_mask"][b, v]):
                hw = tuple(map(int, inputs["image_sizes"][b, v].tolist()))
                groups.setdefault(hw, []).append((b, v))
    maps = {s: {} for s in (4, 8, 16, 32)}
    for (h, w), pairs in groups.items():
        batch = torch.stack([inputs["rgb"][b, v, :, :h, :w] for b, v in pairs])[:, None]
        mask = torch.stack([inputs["image_mask"][b, v, :h, :w] for b, v in pairs])[:, None]
        encoded = encoder({"rgb": batch, "image_mask": mask})
        for s, fmap in encoded.items():
            for row, pair in enumerate(pairs):
                maps[s][pair] = F.pad(
                    fmap[row, 0],
                    (
                        0,
                        math.ceil(width / s) - fmap.shape[-1],
                        0,
                        math.ceil(height / s) - fmap.shape[-2],
                    ),
                )
    channels = encoder.lateral["4"].out_channels
    result = {}
    for s, values in maps.items():
        zero = inputs["rgb"].new_zeros(channels, math.ceil(height / s), math.ceil(width / s))
        result[s] = torch.stack(
            [values.get((b, v), zero) for b in range(bsize) for v in range(views)]
        ).reshape(bsize, views, *zero.shape)
    return result


class MultiFocalFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.settings = cfg["model"].get("multifocal", {})
        self.enabled = self.settings.get("tele_initialization", True)
        self.near, self.far = self.settings.get("tele_near", 10.0), self.settings.get(
            "tele_far", 150.0
        )
        if self.enabled:
            tele = deepcopy(cfg)
            tele["model"].update(
                near=self.near,
                far=self.far,
                depth_bins=self.settings.get("tele_depth_bins", 72),
                min_support=2,
            )
            self.tele_initializer = PlaneSweep(tele)
            depths = segmented_depths(self.near, self.far, tele["model"]["depth_bins"])
            self.tele_initializer.inverse_depths.copy_(depths.reciprocal())

    def forward(self, inputs, pyramid, wide_coarse):
        # 广角初始化完全不读取长焦。输出仍为公共N视角张量，长焦槽明确不接受广角粗监督。
        wide = camera_subset(inputs, list(range(4)))
        coarse = wide_coarse(wide, {s: v[:, :4] for s, v in pyramid.items()})
        b, n = inputs["rgb"].shape[:2]
        full = {}
        for key, value in coarse.items():
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == (b, 4):
                pad = value.new_zeros((b, n - 4, *value.shape[2:]))
                full[key] = torch.cat((value, pad), 1)
            else:
                full[key] = value
        full["supervised_views"] = inputs["view_mask"].clone()
        full["supervised_views"][:, 4:] = False
        full["tele"] = None
        if self.enabled:
            tele = camera_subset(inputs, [4, 5])
            full["tele"] = self.tele_initializer(tele, {s: v[:, 4:6] for s, v in pyramid.items()})
            for key in ("depth", "peaks", "confidence", "support", "texture"):
                full[key] = torch.cat((coarse[key], full["tele"][key]), 1)
        else:
            full["depth"][:, 4:] = (self.near * self.far) ** 0.5
            full["peaks"][:, 4:] = (self.near * self.far) ** 0.5
        return full
