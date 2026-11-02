"""四个局部表面分支的共享速度网络与训练路径；不依赖真实标签推理。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, features):
        return features + self.body(features)


class LocalVelocity(nn.Module):
    """各分支共享参数，邻域空间计算不混合四个竞争深度的数值。

    分支索引在相邻像素不保证对应同一物体；输入包含绝对候选位置及局部
    状态供网络辨别。普通回归和FM用完全相同的层/输入/初始化。
    """

    def __init__(self, channels, direction_channels, hidden=32, learn_quality=True):
        super().__init__()
        if hidden % 4:
            raise ValueError("hidden必须可被GroupNorm的4组整除")
        self.input = nn.Sequential(
            nn.Conv2d(channels + direction_channels + 10, hidden, 1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
        )
        self.spatial = nn.Sequential(SpatialBlock(hidden, 1), SpatialBlock(hidden, 2))
        self.velocity = nn.Conv2d(hidden, 1, 1)
        self.quality = nn.Conv2d(hidden, 1, 1) if learn_quality else None
        for head in (self.velocity, self.quality):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, features, direction, state, time, position, radius,
                prior, similarity, null_mass, source_fraction, valid):
        batch, channels, branches, height, width = features.shape
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device).expand_as(state)
        scalar = torch.stack((state, time, (math.pi * time).sin(), (math.pi * time).cos(),
                              position, radius.log1p(), prior, similarity,
                              null_mass, source_fraction), 1)
        inputs = torch.cat((features, direction, scalar), 1)
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(batch * branches, -1, height, width)
        hidden = self.spatial(self.input(inputs))
        velocity = 2 * self.velocity(hidden).tanh().reshape(batch, branches, height, width)
        quality = (2 * self.quality(hidden).tanh().reshape(batch, branches, height, width)
                   if self.quality is not None else torch.zeros_like(velocity))
        return velocity * valid, quality * valid


def local_coordinates(anchor, lower, upper):
    """以输入决定的支持区间定义无量纲log深度坐标，不根据GT调整窗口。"""
    anchor_log = anchor.clamp_min(1e-6).log()
    low_log = lower.clamp_min(1e-6).log()
    high_log = upper.clamp_min(1e-6).log()
    radius = torch.maximum(anchor_log - low_log, high_log - anchor_log).clamp_min(1e-3)
    return anchor_log, radius, (low_log - anchor_log) / radius, (high_log - anchor_log) / radius


def clamp_state(state, lower, upper):
    return torch.maximum(lower, torch.minimum(upper, state))


def bridge_targets(anchor_log, radius, lower_state, upper_state, truth, valid,
                   objective, sigma, generator=None, anchor_fraction=0.):
    """只在训练损失中调用。所有推理分支仍保留，不以GT筛掉候选。

    GT落入输入预先确定的分支区间时才给定位监督。多个区间重叠时只选择
    log深度最近的可达锚点，避免把所有分支无条件拉向同一表面。
    没有可达分支的像素不产生速度标签，但仍由完整输出损失与评测计入。
    """
    if objective not in {"direct", "flow"}:
        raise ValueError("objective必须为direct或flow")
    if not 0 <= float(anchor_fraction) < 1:
        raise ValueError("anchor_fraction须在[0,1)内")
    with torch.no_grad():
        safe_truth = torch.where(valid, truth, torch.ones_like(truth)).clamp_min(1e-6)
        target = (safe_truth.log() - anchor_log) / radius
        covered = valid & (target >= lower_state) & (target <= upper_state)
        distances = (safe_truth.log() - anchor_log).abs().masked_fill(~covered, torch.inf)
        nearest = distances.argmin(1, keepdim=True)
        assigned = torch.zeros_like(covered).scatter_(1, nearest, True) & covered
        noise = torch.randn(anchor_log.shape, device=anchor_log.device,
                            dtype=anchor_log.dtype, generator=generator)
        initial = clamp_state(float(sigma) * noise, lower_state, upper_state)
        # direct也消耗同一time随机数，保证后续随机流一致；只把其数值置零。
        time = torch.rand((anchor_log.shape[0], 1, 1, 1), device=anchor_log.device,
                          dtype=anchor_log.dtype, generator=generator)
        if objective == "direct":
            time = torch.zeros_like(time)
        elif anchor_fraction:
            # 用同一随机数重新分配训练时间，不增加取证或改变后续RNG序列。
            time = torch.where(time < anchor_fraction, 0., (time - anchor_fraction) / (1 - anchor_fraction))
        endpoint = torch.where(assigned, target, initial)
        state = (1 - time) * initial + time * endpoint
        velocity = endpoint - initial
    return {"state": state, "time": time, "velocity": velocity, "assigned": assigned,
            "covered": covered, "initial": initial, "endpoint": endpoint}


def velocity_loss(prediction, target, assigned):
    # FM的标准平方速度损失；普通残差对照使用同一个目标与归约。
    squared = (prediction.float() - target.float()).square()
    return torch.where(assigned, squared, 0).sum() / assigned.sum().clamp_min(1)
