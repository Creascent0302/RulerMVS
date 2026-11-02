"""保留深度假设的稀疏上下文；测量有效性由原匹配器独立决定。

只在共同的全局深度格上交替处理深度轴与同相机图像邻域，局部候选通过
物理逆深度坐标读取残差。没有全场景查询两两 attention，也不预测标定。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import safe_masked_softmax
from .curve_transport import image_neighbors, interpolate_curves


class HypothesisBlock(nn.Module):
    """沿同一射线交流深度假设，再沿同一深度格交流空间邻居。"""

    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.channels = channels
        self.depth_norm = nn.LayerNorm(channels)
        self.depth_qkv = nn.Linear(channels, channels * 3)
        self.depth_output = nn.Linear(channels, channels)
        self.spatial_norm = nn.LayerNorm(channels)
        self.spatial_qkv = nn.Linear(channels, channels * 3)
        self.spatial_output = nn.Linear(channels, channels)
        self.relative_position = nn.Sequential(
            nn.Linear(2, channels), nn.GELU(), nn.Linear(channels, heads)
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, tokens, valid, ids, neighbor_valid, relative_uv):
        queries, depths, channels = tokens.shape
        head_size = channels // self.heads
        qkv = self.depth_qkv(self.depth_norm(tokens))
        q, k, v = qkv.reshape(queries, depths, 3, self.heads, head_size).unbind(2)
        # PyTorch SDPA 对全 False 的 key mask 返回零；输出仍再次屏蔽无效查询。
        depth_context = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=valid[:, None, None, :], dropout_p=0.0,
        )
        depth_context = depth_context.transpose(1, 2).reshape(queries, depths, channels)
        tokens = tokens + self.depth_output(depth_context) * valid[..., None]

        qkv = self.spatial_qkv(self.spatial_norm(tokens))
        q, k, v = qkv.reshape(queries, depths, 3, self.heads, head_size).unbind(2)
        neighbor_keys = k[ids].permute(0, 2, 1, 3, 4)
        neighbor_values = v[ids].permute(0, 2, 1, 3, 4)
        scores = (q[:, :, None].float() * neighbor_keys.float()).sum(-1)
        scores = scores / math.sqrt(head_size)
        scores = scores + self.relative_position(relative_uv).float()[:, None]
        supported = valid[ids].transpose(1, 2) & neighbor_valid[:, None]
        weights = safe_masked_softmax(scores, supported[..., None], dim=2)
        spatial_context = (weights[..., None] * neighbor_values.float()).sum(2)
        spatial_context = spatial_context.reshape(queries, depths, channels)
        has_neighbor = supported.any(-1) & valid
        tokens = tokens + self.spatial_output(spatial_context.to(tokens.dtype)) * has_neighbor[..., None]
        return (tokens + self.ffn(tokens)) * valid[..., None]


class HypothesisContext(nn.Module):
    """候选级跨视角融合→深度/空间交替→有界 logit 残差。

    不修改 candidate_valid，不用输出上下文构造新的纹理或视角支持证据。
    最后投影为零初始化，新分支迁移起点与旧模型一致；第一步只有该投影
    获得非零梯度，后续步骤才会训练更早的注意力层。
    """

    def __init__(self, cfg):
        super().__init__()
        model = cfg["model"]
        settings = model["hypothesis_context"]
        channels = int(settings.get("hidden_dim", 32))
        heads = int(settings.get("heads", 4))
        layers = int(settings.get("layers", 1))
        self.radius = float(settings.get("radius", 32.0))
        self.neighbors = int(settings.get("neighbors", 8))
        self.max_residual = float(settings.get("max_residual", 4.0))
        self.spatial = bool(settings.get("spatial", True))
        self.near = float(model.get("near", 2.0))
        self.far = float(model.get("far", 80.0))
        if channels < 1 or heads < 1 or channels % heads or layers < 1:
            raise ValueError("hypothesis_context 的通道/头数/层数不合法")
        if not math.isfinite(self.radius) or self.radius <= 0 or self.neighbors < 1:
            raise ValueError("hypothesis_context 邻域半径和数量必须为正")
        if not math.isfinite(self.max_residual) or self.max_residual <= 0:
            raise ValueError("hypothesis_context.max_residual 必须有限且为正")
        self.view_encoder = nn.Sequential(
            nn.Linear(int(model.get("feature_dim", 64)) + 8, channels),
            nn.LayerNorm(channels), nn.GELU(), nn.Linear(channels, channels),
        )
        self.view_score = nn.Linear(channels, 1)
        self.geometry = nn.Sequential(nn.Linear(7, channels), nn.GELU(), nn.Linear(channels, channels))
        self.score_embedding = nn.Linear(1, channels)
        self.stage_embedding = nn.Embedding(4, channels)
        self.blocks = nn.ModuleList([HypothesisBlock(channels, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def encode_views(self, sampled, metadata, valid):
        """[Q,D,N,C] → [Q,D,H]；只跨源视图聚合，不提前消除深度维度。"""
        values = self.view_encoder(torch.cat((sampled, metadata), -1))
        logits = self.view_score(values)[..., 0].float()
        weights = safe_masked_softmax(logits, valid, dim=-1)
        result = (weights[..., None] * values.float()).sum(-2)
        return result.to(values.dtype) * valid.any(-1)[..., None]

    def forward(self, features, logits, candidates, valid, uv, refs, centers, rays, stage):
        start = 9  # 前9个是每条射线自己的局部候选，不能按编号跨查询通信。
        global_depth = candidates[:, start:]
        if global_depth.shape[1] < 2 or not torch.allclose(global_depth, global_depth[:1]):
            raise ValueError("候选上下文需要至少两个共享的全局米制深度候选")
        global_valid = valid[:, start:]
        points = centers[:, None] + rays[:, None] * global_depth[..., None]
        direction = F.normalize(rays.float(), dim=-1)[:, None].expand_as(points)
        inverse = (global_depth.reciprocal() - 1 / self.far) / (1 / self.near - 1 / self.far)
        geometry = torch.cat((points / self.far, direction, inverse[..., None]), -1)
        tokens = features[:, start:] + self.geometry(geometry)
        raw_score = logits[:, start:].float().clamp(-8, 8) / 8
        tokens = tokens + self.score_embedding(raw_score[..., None])
        tokens = tokens + self.stage_embedding.weight[stage].to(tokens.dtype)
        tokens = tokens * global_valid[..., None]
        ids, neighbor_valid = image_neighbors(uv, refs, self.radius, self.neighbors)
        if not self.spatial:
            neighbor_valid = torch.zeros_like(neighbor_valid)
        relative_uv = (uv[ids] - uv[:, None]) / self.radius
        for block in self.blocks:
            tokens = block(tokens, global_valid, ids, neighbor_valid, relative_uv)
        residual = self.max_residual * torch.tanh(self.output(tokens)[..., 0].float())
        residual = residual * global_valid
        sampled, known = interpolate_curves(
            global_depth[0].reciprocal(), residual, global_valid, candidates.reciprocal()
        )
        return sampled * known * valid
