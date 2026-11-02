"""公制候选证据网络：在选面之前比较观测、刻度和局部空间关系。

不生成虚拟 RGB，不把投影合法当成表面。所有候选始终保留独立深度轴；
标尺改变逐源取舍与深度消息的实际物理跨度，原有平面邻域运输继续复用。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import pixel_grid, safe_masked_softmax, sample_maps
from .initial_surface_memory import CandidateChannelNorm, transport_neighbors


def depth_cell_edges(depths):
    """严格递增的深度锚点 → 不等距单元边界，单位米。"""
    middle = (depths[:-1] + depths[1:]) * 0.5
    return torch.cat((depths[:1], middle, depths[-1:]))


def metric_depth_plan(inverse_depths, steps=(0.25, 1.0, 4.0), metric=True):
    """保留原逆深度顺序；只为有限邻接预计算索引，不删掉任何候选。"""
    rho = inverse_depths.float()
    if len(rho) < 3 or not bool((rho[1:] > rho[:-1]).all()):
        raise ValueError("逆深度格点必须严格递增且至少三个")
    depth = rho.reciprocal()
    plans = []
    for index, step in enumerate(steps):
        for direction in (-1, 1):
            if metric:
                query_depth = depth + direction * float(step)
                known = (query_depth >= depth[-1]) & (query_depth <= depth[0])
                query = query_depth.clamp_min(1e-6).reciprocal()
                high = torch.searchsorted(rho, query.contiguous()).clamp(1, len(rho) - 1)
                low = high - 1
                alpha = ((query - rho[low]) / (rho[high] - rho[low])).clamp(0, 1)
            else:
                # 同参数预算对照：仍用三种跨度，但以格点编号定义，不使用米制距离。
                query_index = torch.arange(len(rho), device=rho.device) + direction * 2**index
                known = (query_index >= 0) & (query_index < len(rho))
                low = high = query_index.clamp(0, len(rho) - 1)
                alpha = torch.zeros_like(depth)
                query_depth = depth[low]
            plans.append((low, high, alpha, known, query_depth - depth))
    return plans


def sample_depth_neighbor(features, valid, plan):
    low, high, alpha, known, _ = plan
    batch, channels, count, height, width = features.shape
    indices = low[None, None, :, None, None].expand(batch, channels, -1, height, width)
    upper = high[None, None, :, None, None].expand_as(indices)
    mix = alpha[None, None, :, None, None]
    value = torch.lerp(features.gather(2, indices), features.gather(2, upper), mix)
    lower_valid = valid.gather(1, indices[:, 0])
    upper_valid = valid.gather(1, upper[:, 0])
    mask = valid & known[None, :, None, None]
    mask = mask & (lower_valid | (mix[:, 0] >= 1 - 1e-6))
    mask = mask & (upper_valid | (mix[:, 0] <= 1e-6))
    return torch.where(mask[:, None], value, 0), mask


class MetricCandidateBlock(nn.Module):
    """有限公制邻接＋图像内容选择；不执行全候选两两注意力。"""

    def __init__(self, channels, neighbor_count=6):
        super().__init__()
        self.depth_weight = nn.Conv3d(channels, neighbor_count + 1, 1)
        self.relative = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
        self.depth_value = nn.Conv3d(channels, channels, 1, bias=False)
        self.plane_weight = nn.Conv3d(channels, 3, 1)
        self.plane_value = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm1 = CandidateChannelNorm(channels)
        self.norm2 = CandidateChannelNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, features, valid, depth_plans, plane_plans, use_metric):
        logits = self.depth_weight(features).float()
        values, masks = [features], [valid]
        for index, plan in enumerate(depth_plans):
            neighbor, mask = sample_depth_neighbor(features, valid, plan)
            delta = plan[-1]
            # 有界相对量只影响关系权重，不把公制坐标加成虚假的视觉相似度。
            relation = torch.stack((delta / 4, delta.sign(), torch.log1p(delta.abs())), -1)
            if not use_metric:
                relation = torch.zeros_like(relation)
            bias = self.relative(relation).squeeze(-1)[None, :, None, None]
            logits[:, index + 1] = logits[:, index + 1] + bias
            values.append(neighbor)
            masks.append(mask)
        available = torch.stack(masks, 1)
        weight = safe_masked_softmax(logits, available, dim=1)
        message = sum(value * weight[:, index:index + 1] for index, value in enumerate(values))
        features = F.gelu(self.norm1(features + self.depth_value(message))) * valid[:, None]
        planes, known = transport_neighbors(features, valid, plane_plans)
        legal = torch.cat((known[:, 0], valid[:, None]), 1)
        plane_weight = safe_masked_softmax(self.plane_weight(features).float(), legal, dim=1)
        message = (planes * plane_weight[:, None, :2]).sum(2)
        features = F.gelu(self.norm2(features + self.plane_value(message))) * valid[:, None]
        return (features + self.ffn(features)) * valid[:, None]


class MetricRulerCandidateNetwork(nn.Module):
    """跨源取证 → 分层视觉读取 → 候选图推理 → 多峰与连续定位。

    mode='appearance' 关闭新增标尺条件；'channels' 仅拼接常规几何；
    'metric' 使用观测导数和公制邻接。所有模式均保留真实标定投影。
    这是完整新架构中的候选核心，不能单凭此模块的缓存指标声称整网改善。
    """

    def __init__(self, settings):
        super().__init__()
        self.mode = settings.get("mode", "metric")
        if self.mode not in ("appearance", "channels", "metric"):
            raise ValueError("mode 必须为 appearance、channels 或 metric")
        self.use_jet = bool(settings.get("use_jet", True))
        self.channels = int(settings.get("channels", 24))
        channels = self.channels
        self.view_encoder = nn.Sequential(
            nn.Conv3d(6, 16, 1), nn.SiLU(), nn.Conv3d(16, 16, 1), nn.SiLU(),
        )
        self.view_reliability = nn.Conv3d(16, 1, 1)
        self.entry = nn.Conv3d(12 + 2 + 16 + 16 + 8, channels, 1)
        self.visual_keys = nn.ModuleList([nn.Conv2d(768, channels, 1) for _ in range(4)])
        self.visual_values = nn.ModuleList([nn.Conv2d(768, channels, 1) for _ in range(4)])
        self.visual_query = nn.Conv3d(channels, channels, 1)
        self.norm = CandidateChannelNorm(channels)
        self.blocks = nn.ModuleList([
            MetricCandidateBlock(channels) for _ in range(int(settings.get("blocks", 2)))
        ])
        self.score = nn.Conv3d(channels, 1, 1)
        self.offset = nn.Conv3d(channels, 1, 1)
        self.state = nn.Conv3d(channels, 3, 1)
        nn.init.normal_(self.score.weight, std=1e-3)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    @staticmethod
    def aligned_raw(raw, shape, stride):
        uv = pixel_grid(*shape, device=raw.device, stride=stride).reshape(1, -1, 2) - 7.5
        values = sample_maps(raw, uv.expand(len(raw), -1, -1), stride=16, padding_mode="border")
        return values.transpose(1, 2).reshape(len(raw), raw.shape[1], *shape)

    def forward(self, inputs, ruler, depth_plans, plane_plans, base_scores=None):
        cost, hero = inputs["cost"].float(), inputs["hero"].float()
        cosine, source_valid = inputs["source_cosine"].float(), inputs["source_valid"].bool()
        valid = inputs["candidate_valid"].bool()
        batch, views, count, height, width = cosine.shape
        geometric = ruler["features"].float()
        energy = ruler["stats"]["source_jet_energy"].float()
        parallax = ruler["stats"]["ruler_parallax"].float()
        defect = geometric[:, :1].expand(-1, views, -1, -1, -1)
        if self.mode != "metric" or not self.use_jet:
            geometric = geometric.clone()
            geometric[:, :5] = 0
            energy = torch.zeros_like(energy)
            defect = torch.zeros_like(defect)
        if self.mode == "appearance":
            geometric = torch.zeros_like(geometric)
            parallax = torch.zeros_like(parallax)
        descriptor = torch.stack((
            cosine, cosine - hero[:, :1], torch.log1p(energy).clamp_max(10) / 10,
            torch.log1p(parallax).clamp_max(10) / 10, defect, source_valid.float(),
        ), 2).flatten(0, 1)
        view_features = self.view_encoder(descriptor)
        view_logits = self.view_reliability(view_features).unflatten(0, (batch, views))[:, :, 0]
        # null observation 的固定权重为1；允许所有可采样源都被拒绝。
        weights = view_logits.sigmoid() * source_valid
        encoded = view_features.unflatten(0, (batch, views))
        pooled = (encoded * weights[:, :, None]).sum(1) / (1 + weights.sum(1, keepdim=True))
        # max 通路也必须服从同一个观测权重，不能绕过坏视图的拒绝。
        maximum = (encoded * weights[:, :, None]).amax(1).clamp_min(0)
        features = self.entry(torch.cat((cost, hero, pooled, maximum, geometric), 1))
        query = self.visual_query(features)
        visual_keys, visual_values = [], []
        for raw, key, value in zip(inputs["raw_maps"], self.visual_keys, self.visual_values):
            aligned = self.aligned_raw(raw.float(), (height, width), int(inputs["stride"]))
            visual_keys.append(key(aligned))
            visual_values.append(value(aligned))
        keys = torch.stack(visual_keys, 2)
        values = torch.stack(visual_values, 2)
        # 每个深度候选独立读取四层视觉信息，而非先把候选轴混成通道。
        layer_score = torch.einsum("bcdhw,bclhw->bldhw", query, keys) / math.sqrt(self.channels)
        visual = torch.einsum("bldhw,bclhw->bcdhw", layer_score.softmax(1), values)
        features = F.gelu(self.norm(features + visual)) * valid[:, None]
        for block in self.blocks:
            features = block(features, valid, depth_plans, plane_plans, self.mode == "metric")
        scores = self.score(features)[:, 0]
        scores = scores + (4 * hero[:, 0] if base_scores is None else base_scores)
        scores = scores.masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(scores.float(), valid, dim=1)
        depths = inputs["inverse_depths"].float().reciprocal()
        # 各候选局部区间互不混合；最终取峰，再在峰单元内连续定位。
        edges = depth_cell_edges(depths.flip(0)).flip(0)
        lower, upper = edges[1:], edges[:-1]
        anchor = depths[None, :, None, None]
        width = (upper - lower)[None, :, None, None]
        candidate_depth = anchor + width * self.offset(features)[:, 0].tanh()
        candidate_depth = candidate_depth.maximum(lower[None, :, None, None])
        candidate_depth = candidate_depth.minimum(upper[None, :, None, None])
        index = scores.argmax(1, keepdim=True)
        depth = candidate_depth.gather(1, index)
        support = source_valid.sum(1).gather(1, index)
        return {
            "scores": scores, "probability": probability,
            "candidate_depth": candidate_depth, "depth": depth,
            "state_logits": self.state(features), "view_logits": view_logits,
            "valid": valid.any(1, keepdim=True), "support": support,
            "confidence": probability.gather(1, index),
            "surface_verified": torch.zeros_like(depth, dtype=torch.bool),
        }
