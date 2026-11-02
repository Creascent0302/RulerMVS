"""以真实源描述子和标尺关系替换中心门控的候选网络。

保留原候选轴、四层视觉读取和输出合同；替换逐源标量池化与邻域门控，
不叠加第二个候选核。family='attention'显式选择，历史默认不改变。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import safe_masked_softmax
from .initial_surface_memory import CandidateChannelNorm
from .metric_ruler_geometry_evidence import RulerProjectionCache
from .metric_ruler_network import MetricRulerCandidateNetwork, depth_cell_edges
from .ruler_attention_evidence import attention_evidence, projected_candidate_distance


def build_ruler_network(settings):
    if settings.get("family", "metric") == "attention":
        return RulerAttentionNetwork(settings)
    if settings.get("family") == "stencil":
        return StencilRulerNetwork(settings)
    return MetricRulerCandidateNetwork(settings)


class StencilRulerNetwork(MetricRulerCandidateNetwork):
    """同参数替换中心统计为真实五刻度面片匹配；保留中心证据作对照底座。"""

    def __init__(self, settings):
        super().__init__(settings)
        self.stencil_mode = settings.get("stencil_mode", "physical")

    def forward(self, inputs, ruler, depth_plans, plane_plans, base_scores=None):
        stencil = inputs.get("_ruler_stencil")
        if stencil is None:
            from .ruler_stencil import attention_stencil_evidence

            stencil = attention_stencil_evidence(inputs, mode=self.stencil_mode)
        summary = torch.where(stencil["valid"], stencil["scores"], inputs["hero"][:, :1])
        work = {**inputs, "hero": summary}
        return super().forward(work, ruler, depth_plans, plane_plans, base_scores)


class RulerAttentionNetwork(nn.Module):
    def __init__(self, settings):
        super().__init__()
        from .ruler_attention import RulerRelationBlock, RulerSourceAttention

        self.settings = dict(settings)
        self.mode = "metric"  # 两组均用相同物理采样计划，只比较关系权重。
        self.use_geometry = bool(settings.get("use_geometry", True))
        self.relation = settings.get("relation", "local")
        self.channels = int(settings.get("channels", 24))
        channels = self.channels
        self.source_attention = RulerSourceAttention(channels=16)
        self.entry = nn.Conv3d(12 + 2 + 16 + 1, channels, 1)
        self.visual_keys = nn.ModuleList([nn.Conv2d(768, channels, 1) for _ in range(4)])
        self.visual_values = nn.ModuleList([nn.Conv2d(768, channels, 1) for _ in range(4)])
        self.visual_query = nn.Conv3d(channels, channels, 1)
        self.norm = CandidateChannelNorm(channels)
        if self.relation == "ray":
            from .ruler_ray_attention import RulerRayAttention

            block = RulerRayAttention
        elif self.relation == "local":
            block = RulerRelationBlock
            self.depth_strength = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError("relation只允许local或ray")
        self.blocks = nn.ModuleList([block(channels) for _ in range(int(settings.get("blocks", 1)))])
        self.score = nn.Conv3d(channels, 1, 1)
        self.offset = nn.Conv3d(channels, 1, 1)
        self.state = nn.Conv3d(channels, 3, 1)
        nn.init.normal_(self.score.weight, std=1e-3)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.projection_cache = RulerProjectionCache(capacity=4)

    def _apply(self, function, recurse=True):
        self.projection_cache.clear()
        return super()._apply(function, recurse=recurse)

    def forward(self, inputs, ruler, depth_plans, plane_plans, base_scores=None):
        evidence = inputs.get("_ruler_attention")
        if evidence is None:
            evidence = attention_evidence(inputs, self.projection_cache)
        source = self.source_attention(**evidence, use_geometry=self.use_geometry)
        cost, hero = inputs["cost"].float(), inputs["hero"].float()
        valid = inputs["candidate_valid"].bool()
        features = self.entry(torch.cat((cost, hero, source["pooled"], source["null_mass"]), 1))
        shape = cost.shape[-2:]
        keys, values = [], []
        for raw, key, value in zip(inputs["raw_maps"], self.visual_keys, self.visual_values):
            aligned = MetricRulerCandidateNetwork.aligned_raw(raw.float(), shape, int(inputs["stride"]))
            keys.append(key(aligned))
            values.append(value(aligned))
        keys, values = torch.stack(keys, 2), torch.stack(values, 2)
        query = self.visual_query(features)
        logits = torch.einsum("bcdhw,bclhw->bldhw", query, keys) / math.sqrt(self.channels)
        visual = torch.einsum("bldhw,bclhw->bcdhw", logits.softmax(1), values)
        features = F.gelu(self.norm(features + visual)) * valid[:, None]
        if self.relation == "ray":
            pair_distance = inputs.get("_ruler_pair_distance")
            if pair_distance is None:
                pair_distance = projected_candidate_distance(inputs, self.projection_cache)
            for block in self.blocks:
                features = block(features, valid, pair_bias=-pair_distance, use_geometry=self.use_geometry)
        else:
            information = source["information"][:, 0]
            # 信息量单位1/m²，乘(ΔZ)²得到无量纲关系代价。
            distance = torch.stack([plan[-1].square() for plan in depth_plans])
            penalty = (information[:, None] * distance[None, :, :, None, None]).clamp_max(8)
            depth_bias = -0.5 * self.depth_strength.sigmoid() * penalty * float(self.use_geometry)
            for block in self.blocks:
                features = block(features, valid, depth_plans, plane_plans, depth_bias=depth_bias)
        scores = self.score(features)[:, 0]
        scores = scores + (4 * hero[:, 0] if base_scores is None else base_scores)
        scores = scores.masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(scores, valid, dim=1)
        depths = inputs["inverse_depths"].float().reciprocal()
        edges = depth_cell_edges(depths.flip(0)).flip(0)
        lower, upper = edges[1:][None, :, None, None], edges[:-1][None, :, None, None]
        candidate_depth = depths[None, :, None, None] + (upper - lower) * self.offset(features)[:, 0].tanh()
        candidate_depth = candidate_depth.maximum(lower).minimum(upper)
        selected = scores.argmax(1, keepdim=True)
        depth = candidate_depth.gather(1, selected)
        return {
            "scores": scores, "probability": probability,
            "candidate_depth": candidate_depth, "depth": depth,
            "state_logits": self.state(features), "view_logits": source["logits"],
            "valid": valid.any(1, keepdim=True),
            "support": evidence["source_valid"].sum(1).gather(1, selected),
            "confidence": probability.gather(1, selected),
            "surface_verified": torch.zeros_like(depth, dtype=torch.bool),
        }
