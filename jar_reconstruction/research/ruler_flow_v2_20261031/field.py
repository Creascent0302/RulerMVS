"""一次取证、反复读出的局部平均速度场。

时间约定为 t=1 的原预测向 t=0 的目标深度移动。空间上下文只在
真实十二候选上计算一次；中间状态不是重新观测到的匹配证据。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def grouped(value):
    """[B,3G,H,W] -> [BG,3,H,W]，三个采样的位置次序保持不变。"""
    batch, count, height, width = value.shape
    return value.reshape(batch * (count // 3), 3, height, width)


def time_map(value, state):
    """接受标量、[B,1,1,1] 或与 state 同形的时间场。"""
    value = torch.as_tensor(value, device=state.device, dtype=state.dtype)
    return value + torch.zeros_like(state)


class EvidenceVelocity(nn.Module):
    """每组三点共享一个上下文，分别预测三个有界平均速度。

    默认上下文输入为 3*64+26=218 通道，投影到48。几何门控邻域
    聚合仅联系具有重叠搜索区间的相邻同序号组，减少rank错配传播。
    区间重叠只代表计算区域相容，不能认证两个位置属于同一表面。
    """

    def __init__(self, channels, hidden=48):
        super().__init__()
        groups = next(number for number in (8, 4, 2, 1) if hidden % number == 0)
        self.context = nn.Sequential(
            nn.Conv2d(3 * channels + 26, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.neighbor = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.context_norm = nn.GroupNorm(groups, hidden)
        self.velocity_net = nn.Sequential(
            nn.Conv2d(hidden + 10, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def neighborhood(value, lower, upper, valid):
        """四邻域差分，明确屏蔽环绕边界和不相交区间。"""
        total = torch.zeros_like(value)
        mass = torch.ones_like(lower)
        width = (upper - lower).clamp_min(1e-6)
        for axis, shift in ((-2, -1), (-2, 1), (-1, -1), (-1, 1)):
            neighbor_lower = torch.roll(lower, shift, axis)
            neighbor_upper = torch.roll(upper, shift, axis)
            neighbor_valid = torch.roll(valid, shift, axis)
            overlap = (torch.minimum(upper, neighbor_upper)
                       - torch.maximum(lower, neighbor_lower)).clamp_min(0)
            union = torch.maximum(width, neighbor_upper - neighbor_lower).clamp_min(1e-6)
            gate = overlap / union
            gate = gate * (valid & neighbor_valid)
            boundary = torch.ones_like(gate)
            index = [slice(None)] * 4
            index[axis] = 0 if shift == 1 else -1
            boundary[tuple(index)] = 0
            gate = gate * boundary
            total = total + gate * (torch.roll(value, shift, axis) - value)
            mass = mass + gate
        return total / mass

    def compute_condition(self, features, candidates, lower, upper,
                          baseline_prediction, observation, near, far):
        batch, channels, count, height, image_width = features.shape
        if count % 3:
            raise ValueError("细候选必须保留原有每组三点")
        groups = count // 3
        lower = lower.expand_as(candidates).float().clamp(near, far)
        upper = upper.expand_as(candidates).float().clamp(near, far)
        width = upper - lower
        valid = baseline_prediction["valid"] & (width > 1e-7)
        old_depth = baseline_prediction["candidate_depth"]
        anchor = (2 * (old_depth - lower) / width.clamp_min(1e-7) - 1).clamp(-1, 1)
        anchor = torch.where(valid, anchor, 0)
        sample = grouped(candidates.float())
        center = sample[:, 1:2].clamp_min(1e-7)
        group_lower = grouped(lower).amin(1, keepdim=True)
        group_upper = grouped(upper).amax(1, keepdim=True)
        group_valid = grouped(valid).any(1, keepdim=True)
        span = (group_upper - group_lower).clamp_min(1e-7)
        score = grouped(baseline_prediction["scores"].float())
        score = (score - score.amax(1, keepdim=True)).clamp(-16, 0) / 16
        support = grouped(baseline_prediction["support"].float()) / max(observation["view_count"] - 1, 1)
        null = grouped(observation["flow_null_mass"].float())
        derivative = torch.linalg.vector_norm(observation["ray_derivative"].float(), dim=2)
        derivative = torch.where(observation["source_valid"], derivative, 0).amax(1)
        footprint = grouped((derivative * width * .5).log1p().clamp_max(8)) / 8
        similarity = grouped(baseline_prediction["flow_similarity"].float())
        relative_depth = ((sample - center) / span).clamp(-1, 1)
        relative_width = (grouped(width) / center).log1p().clamp_max(8) / 8
        position = ((center.log() - math.log(near)) / math.log(far / near)).clamp(0, 1)
        scale = (span / center).log1p().clamp_max(8) / 8
        geometry = torch.cat((score, similarity, support, null, relative_depth,
                              relative_width, footprint, grouped(valid).float(), position, scale), 1)
        bank = features.reshape(batch, channels, groups, 3, height, image_width)
        bank = bank.permute(0, 2, 1, 3, 4, 5).reshape(batch * groups, 3 * channels, height, image_width)
        condition = self.context(torch.cat((bank, geometry.to(bank.dtype)), 1))
        message = self.neighborhood(condition, group_lower, group_upper, group_valid)
        condition = F.silu(self.context_norm(condition + self.neighbor(message)))
        condition = condition * group_valid
        return {
            "condition": condition,
            "anchor": anchor,
            "lower": lower,
            "upper": upper,
            "width": width,
            "valid": valid,
            "old_depth": old_depth,
            "sampling_depth": candidates,
        }

    def velocity(self, context, state, r, t):
        """平均速度u(state,r,t)；仅此小头参与MeanFlow的JVP。"""
        r = grouped(time_map(r, state))[:, :1]
        t = grouped(time_map(t, state))[:, :1]
        time = torch.cat((r, t, t - r, (math.pi * t).sin()), 1)
        value = torch.cat((context["condition"], grouped(state),
                           grouped(context["anchor"]), time), 1)
        result = 2 * self.output(self.velocity_net(value)).tanh()
        return result.reshape_as(state) * context["valid"]
