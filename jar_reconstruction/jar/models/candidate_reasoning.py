"""对同一射线的候选表面做多层推理，不构造全场景注意力矩阵。

原解码器的 query 在读取图像后得到更新，但深度分类原先不读取这一更新。
本模块让候选分类消费更新后的 query 和 value 特征，从而让深度监督训练
完整的注意力与前馈分支。候选位置使用实际逆深度编码；不同射线的第 k 个
局部候选并不被假定为同一空间平面。这里只修改分数，不创建几何证据。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CandidateReasoningLayer(nn.Module):
    """query 汇总候选，再将上下文反馈给候选，最后沿射线比较候选。"""

    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(hidden)
        self.candidate_norm = nn.LayerNorm(hidden)
        self.read_query = nn.Linear(hidden, hidden)
        self.read_kv = nn.Linear(hidden, 2 * hidden)
        self.read_output = nn.Linear(hidden, hidden)
        self.query_to_candidate = nn.Linear(hidden, hidden)
        self.query_gate = nn.Linear(hidden, hidden)
        self.depth_norm = nn.LayerNorm(hidden)
        self.depth_qkv = nn.Linear(hidden, 3 * hidden)
        self.depth_output = nn.Linear(hidden, hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 3 * hidden), nn.GELU(), nn.Linear(3 * hidden, hidden)
        )

    def _heads(self, value):
        batch, count, hidden = value.shape
        return value.reshape(batch, count, self.heads, hidden // self.heads).transpose(1, 2)

    def _attention(self, query, key, value, valid):
        # Boolean True means an allowed key in scaled_dot_product_attention.
        # Give an empty row one harmless key, then zero the result explicitly.
        # This also works with older attention backends that produce NaN for
        # an all-masked row. Invalid candidates never enter the returned scores.
        has_evidence = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~has_evidence
        result = F.scaled_dot_product_attention(
            self._heads(query),
            self._heads(key),
            self._heads(value),
            attn_mask=safe_valid[:, None, None],
            dropout_p=0.0,
        )
        result = result.transpose(1, 2).flatten(2)
        return result * has_evidence[:, None, None]

    def forward(self, query, candidates, valid):
        normalized = self.candidate_norm(candidates)
        key, value = self.read_kv(normalized).chunk(2, -1)
        summary = self._attention(
            self.read_query(self.query_norm(query))[:, None], key, value, valid
        )[:, 0]
        query = query + self.read_output(summary) * valid.any(-1, keepdim=True)

        # A single query is a conditioning vector, so a gated broadcast is
        # clearer than a one-key attention whose softmax would always be 1.
        context = self.query_to_candidate(self.query_norm(query))[:, None]
        candidates = candidates + self.query_gate(normalized).sigmoid() * context
        query_depth, key_depth, value_depth = self.depth_qkv(self.depth_norm(candidates)).chunk(
            3, -1
        )
        compared = self._attention(query_depth, key_depth, value_depth, valid)
        candidates = candidates + self.depth_output(compared)
        candidates = candidates + self.ffn(self.ffn_norm(candidates))
        return query, candidates * valid[..., None]


class CandidateReasoning(nn.Module):
    """可靠性加权多视图 value，加上米制位置，再预测有界分数残差。"""

    def __init__(self, query_dim, settings, default_heads=4):
        super().__init__()
        hidden = int(settings.get("hidden_dim", query_dim))
        heads = int(settings.get("heads", default_heads))
        layers = int(settings.get("layers", 3))
        self.near = float(settings.get("near", 2.0))
        self.far = float(settings.get("far", 80.0))
        self.max_residual = float(settings.get("max_residual", 4.0))
        if hidden < 1 or heads < 1 or hidden % heads or layers < 1:
            raise ValueError("candidate_reasoning 需要正层数及可被 heads 整除的 hidden_dim")
        if not 0 < self.near < self.far or not math.isfinite(self.far):
            raise ValueError("candidate_reasoning 的 near/far 必须有限、为正且递增")
        if not 0 < self.max_residual < float("inf"):
            raise ValueError("candidate_reasoning.max_residual 必须有限且为正")
        self.query_projection = nn.Linear(query_dim, hidden)
        self.value_projection = nn.Linear(query_dim, hidden)
        self.metadata_projection = nn.Linear(8, hidden)
        self.depth_encoding = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList([CandidateReasoningLayer(hidden, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        # A small nonzero output keeps initial corrections small while allowing
        # the very first classification step to train every preceding layer.
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def coordinates(self, depths):
        depths = depths.float().clamp_min(1e-6)
        inverse = (depths.reciprocal() - 1 / self.far) / (1 / self.near - 1 / self.far)
        log_depth = (depths / self.near).log() / math.log(self.far / self.near)
        phase = math.pi * inverse
        return torch.stack(
            (inverse, log_depth, phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()),
            dim=-1,
        )

    def forward(self, query, values, metadata, weights, valid, depths):
        if depths is None or depths.shape != valid.shape[:2]:
            raise ValueError("candidate_reasoning 需要实际候选深度 [query, depth]")
        # The mask is supplied by the existing exact projection / visibility
        # checks. Neither attention nor confidence can turn it into evidence.
        weights = weights.float() * valid
        denominator = weights.sum(-1, keepdim=True).clamp_min(1e-8)
        normalized = weights / denominator
        observed = valid.any(-1) & (weights.sum(-1) > 0)
        pooled_values = (values.float() * normalized[..., None]).sum(-2)
        pooled_metadata = (metadata.float() * normalized[..., None]).sum(-2)
        query = self.query_projection(query)
        candidates = self.value_projection(pooled_values.to(query.dtype))
        candidates = candidates + self.metadata_projection(pooled_metadata.to(query.dtype))
        candidates = candidates + self.depth_encoding(self.coordinates(depths).to(query.dtype))
        candidates = (candidates + query[:, None]) * observed[..., None]
        for layer in self.layers:
            query, candidates = layer(query, candidates, observed)
        residual = self.max_residual * self.output(candidates)[..., 0].float().tanh()
        return residual * observed
