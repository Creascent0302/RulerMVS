"""把公制标尺用于源视图取舍和候选间的内容注意力。

标定只描述可辨别性，不能证明匹配正确。源分支保留真实描述子，并允许
所有源落到 null；关系分支逐个读取两个平面的四向邻居，不先平均。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import safe_masked_softmax
from .initial_surface_memory import CandidateChannelNorm
from .metric_ruler_network import sample_depth_neighbor


def ray_information(ray_derivative, cell_half_width, pixel_sigma=0.5):
    """候选单元的投影足迹及一维深度辨别能力，不是匹配置信度。

    g 的单位是匹配特征像素/参考光轴米；h 是光轴深度半宽（米）。
    S = sigma² I + h² g gᵀ，gᵀ S⁻¹ g = |g|²/(sigma²+h²|g|²)。
    h² 是保守单元尺度，不将它称为已知真值的统计方差。
    """
    if pixel_sigma <= 0:
        raise ValueError("pixel_sigma 必须为正")
    g = ray_derivative.float()
    if g.ndim != 6 or g.shape[2] != 2:
        raise ValueError("ray_derivative 必须为 [B,V,2,D,H,W]")
    width = torch.as_tensor(cell_half_width, device=g.device, dtype=torch.float32)
    if width.ndim == 1:
        width = width[None, None, :, None, None]
    elif width.ndim == 4:
        width = width[:, None]
    else:
        raise ValueError("cell_half_width 必须为 [D] 或 [B,D,H,W]")
    sensitivity = g.square().sum(2)
    footprint_squared = width.square() * sensitivity
    information = sensitivity / (pixel_sigma**2 + footprint_squared)
    # vector_norm在零向量处取有限零梯度；sqrt(sum(g²))的拆写会产生
    # 0×inf，缺图/零基线且候选可微时可污染整个反向传播。
    footprint = torch.linalg.vector_norm(g * width[:, :, None], dim=2) / pixel_sigma
    return information, footprint, sensitivity


class RulerSourceAttention(nn.Module):
    """保留逐源完整描述子，以内容和公制辨别能力联合分配权重。

    几何偏置和温度调节零初始化，可以独立关闭。几何项使用有界的相对
    log-information；不存在靠大基线无限提高权重的通路。null 是学习的
    无观测选项，其质量不被重归一化到剩余源中。
    """

    def __init__(self, feature_channels=16, channels=16, pixel_sigma=0.5):
        super().__init__()
        self.pixel_sigma = float(pixel_sigma)
        self.query = nn.Conv2d(feature_channels, channels, 1, bias=False)
        self.key = nn.Conv3d(feature_channels, channels, 1, bias=False)
        self.value = nn.Sequential(
            nn.Conv3d(4 * feature_channels, channels, 1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1),
        )
        self.content_reliability = nn.Conv3d(channels, 1, 1)
        self.null = nn.Conv2d(feature_channels, 1, 1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.25)))
        self.geometry_strength = nn.Parameter(torch.zeros(()))
        self.footprint_strength = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.content_reliability.weight)
        nn.init.zeros_(self.content_reliability.bias)
        nn.init.zeros_(self.null.weight)
        nn.init.zeros_(self.null.bias)
        if feature_channels == channels:
            with torch.no_grad():
                self.query.weight.copy_(torch.eye(channels)[:, :, None, None])
                self.key.weight.copy_(torch.eye(channels)[:, :, None, None, None])

    def forward(self, reference, source, source_valid, ray_derivative,
                cell_half_width, use_geometry=True):
        batch, views, channels, depth, height, width = source.shape
        valid = source_valid.bool()
        # 在投影和归一化前清除缺失槽；NaN * 0 并不能屏蔽坏输入。
        finite = torch.isfinite(source).all(2) & torch.isfinite(ray_derivative).all(2)
        valid = valid & finite & torch.isfinite(reference).all(1)[:, None, None]
        source = torch.where(valid[:, :, None], source, 0).float()
        reference = torch.where(torch.isfinite(reference), reference, 0).float()
        derivatives = torch.where(valid[:, :, None], ray_derivative.float(), 0)
        information, footprint, sensitivity = ray_information(
            derivatives, cell_half_width, self.pixel_sigma
        )

        query = F.normalize(self.query(reference).float(), dim=1, eps=1e-6)
        key = self.key(source.flatten(0, 1)).unflatten(0, (batch, views)).float()
        key = F.normalize(key, dim=2, eps=1e-6)
        cosine = (key * query[:, None, :, None]).sum(2)
        anchor = reference[:, None, :, None].expand_as(source)
        pair = torch.cat((anchor, source, anchor * source, (anchor - source).abs()), 2)
        values = self.value(pair.flatten(0, 1)).unflatten(0, (batch, views))
        reliability = self.content_reliability(values.flatten(0, 1))
        reliability = reliability.unflatten(0, (batch, views))[:, :, 0].float()

        # 消融乘零而不删除参数，保留相同预算与完整的可微计算图。
        geometry_enabled = float(use_geometry)
        log_temperature = self.log_temperature.float()
        # 分辨率同时缩放 g 和 sigma 时，足迹比与 information 均不变。
        log_temperature = log_temperature + geometry_enabled * self.footprint_strength.tanh() * footprint.log1p().clamp_max(4)
        log_information = information.clamp_min(1e-8).log()
        mean = (log_information * valid).sum(1, keepdim=True)
        mean = mean / valid.sum(1, keepdim=True).clamp_min(1)
        relative_information = ((log_information - mean) / 4).tanh()
        geometry_bias = geometry_enabled * 2 * self.geometry_strength.tanh() * relative_information
        temperature = log_temperature.exp().clamp(0.05, 2.0)
        logits = cosine / temperature + reliability + geometry_bias
        null_logits = self.null(reference).float()[:, :, None].expand(batch, 1, depth, height, width)
        all_logits = torch.cat((logits, null_logits), 1)
        all_valid = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
        attention = safe_masked_softmax(all_logits, all_valid, dim=1)
        weights, null_mass = attention[:, :views], attention[:, views:]
        pooled = (values.float() * weights[:, :, None]).sum(1)
        return {
            "pooled": pooled,
            "weights": weights,
            "logits": logits.masked_fill(~valid, -1e4),
            "null_mass": null_mass,
            "information": (weights * information).sum(1, keepdim=True),
            "sensitivity_squared": (weights * sensitivity).sum(1, keepdim=True),
            "source_valid": valid,
        }


def plane_neighbors(features, valid, plans):
    """依次返回 direction-major 的八个平面邻居；不跨空洞插值。"""
    batch, channels = features.shape[:2]
    for dy, dx, left, right, fraction, known in plans:
        dimension, shift = (-2, -dy) if dy else (-1, -dx)
        neighbor = torch.roll(features, shifts=shift, dims=dimension)
        neighbor_valid = torch.roll(valid, shifts=shift, dims=dimension)
        for plane in range(2):
            low_index = left[plane][None].expand(batch, -1, -1, -1)
            high_index = right[plane][None].expand_as(low_index)
            low = low_index[:, None].expand(-1, channels, -1, -1, -1)
            high = high_index[:, None].expand_as(low)
            alpha = fraction[plane][None, None]
            sampled = torch.lerp(neighbor.gather(2, low), neighbor.gather(2, high), alpha)
            low_valid = neighbor_valid.gather(1, low_index)
            high_valid = neighbor_valid.gather(1, high_index)
            mask = valid & known[plane][None]
            mask = mask & (low_valid | (alpha[:, 0] >= 1 - 1e-6))
            mask = mask & (high_valid | (alpha[:, 0] <= 1e-6))
            yield torch.where(mask[:, None], sampled, 0), mask


class RulerRelationBlock(nn.Module):
    """中心、六个深度邻居和八个平面邻居的实际 q·k 注意力。

    每个邻居单独竞争，计算量 O(15 C D H W)。depth_bias/plane_bias
    必须由已知标定或网络预测产生，不能使用真值；其值是关系先验而非
    可见性认证。返回注意力仅用于诊断，默认仅返回更新后的候选特征。
    """

    def __init__(self, channels=24, heads=2):
        super().__init__()
        if channels % heads:
            raise ValueError("channels 必须能被 heads 整除")
        self.channels = channels
        self.heads = heads
        self.query = nn.Conv3d(channels, channels, 1, bias=False)
        self.key_value = nn.Conv3d(channels, 2 * channels, 1, bias=False)
        self.output = nn.Conv3d(channels, channels, 1, bias=False)
        self.log_scale = nn.Parameter(torch.full((heads,), math.log(4.0)))
        self.slot_bias = nn.Parameter(torch.zeros(heads, 15))
        self.norm1 = CandidateChannelNorm(channels)
        self.norm2 = CandidateChannelNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, 2 * channels, 1), nn.GELU(),
            nn.Conv3d(2 * channels, channels, 1),
        )

    def forward(self, features, valid, depth_plans, plane_plans,
                depth_bias=None, plane_bias=None, return_attention=False):
        if len(depth_plans) != 6 or len(plane_plans) != 4:
            raise ValueError("需要六个深度邻居和四方向两平面计划")
        batch, channels, depth, height, width = features.shape
        features = torch.where(valid[:, None], features, 0)
        query = self.query(features).float().unflatten(1, (self.heads, channels // self.heads))
        query = F.normalize(query, dim=2, eps=1e-6)
        # 几何插值计划是 FP32；BF16 autocast 下同样在 FP32 插值。
        key_value = self.key_value(features).float()
        observations = [(key_value, valid)]
        observations.extend(sample_depth_neighbor(key_value, valid, plan) for plan in depth_plans)
        observations.extend(plane_neighbors(key_value, valid, plane_plans))
        logits, values, masks = [], [], []
        scale = self.log_scale.exp().clamp(1, 20)[None, :, None, None, None]
        for index, (neighbor, mask) in enumerate(observations):
            key, value = neighbor.chunk(2, dim=1)
            key = key.float().unflatten(1, (self.heads, channels // self.heads))
            key = F.normalize(key, dim=2, eps=1e-6)
            logit = (query * key).sum(2) * scale
            logit = logit + self.slot_bias[:, index][None, :, None, None, None]
            if 1 <= index <= 6 and depth_bias is not None:
                logit = logit + depth_bias[:, index - 1:index].float()
            elif index > 6 and plane_bias is not None:
                logit = logit + plane_bias[:, index - 7:index - 6].float()
            logits.append(logit)
            values.append(value.float().unflatten(1, (self.heads, channels // self.heads)))
            masks.append(mask)
        accepted = torch.stack(masks, 1)
        weights = safe_masked_softmax(torch.stack(logits, 2), accepted[:, None], dim=2)
        message = sum(value * weights[:, :, index:index + 1] for index, value in enumerate(values))
        message = message.reshape(batch, channels, depth, height, width)
        updated = self.norm1(features + self.output(message)) * valid[:, None]
        updated = self.norm2(updated + self.ffn(updated)) * valid[:, None]
        if return_attention:
            return updated, weights, accepted
        return updated
