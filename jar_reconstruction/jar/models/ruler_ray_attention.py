"""以全射线候选竞争替换有限深度邻接；不再叠加另一套候选块。

同一参考射线上的所有深度假设可以直接交换信息。公制标尺通过已知
标定精确投影得到的 candidate-pair bias 进入注意力；本模块不猜测投影。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .initial_surface_memory import CandidateChannelNorm


class RulerRayAttention(nn.Module):
    """逐射线 D×D 内容注意力；空间像素之间没有全场景两两计算。

    pair_bias:[B,H*W,D,D]，应由精确投影差构造，且对相机顺序不变。
    bias 是视觉可分辨性的关系先验，不能将其当作已观测表面概率。
    attention 使用 SDPA，具体 CUDA kernel 由输入精度和设备选择。
    """

    def __init__(self, channels=24, heads=1):
        super().__init__()
        if channels % heads:
            raise ValueError("channels 必须能被 heads 整除")
        self.channels = channels
        self.heads = heads
        self.query_key_value = nn.Conv3d(channels, 3 * channels, 1, bias=False)
        self.output = nn.Conv3d(channels, channels, 1, bias=False)
        self.log_scale = nn.Parameter(torch.full((heads,), math.log(4.0)))
        self.geometry_logit = nn.Parameter(torch.zeros(()))
        self.norm1 = CandidateChannelNorm(channels)
        self.norm2 = CandidateChannelNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, features, valid, depth_plans=None, plane_plans=None,
                pair_bias=None, use_geometry=True):
        batch, channels, depth, height, width = features.shape
        clean = torch.where(valid[:, None], features, 0)
        projected = self.query_key_value(clean)
        tokens = projected.permute(0, 3, 4, 2, 1).reshape(batch * height * width, depth, 3 * channels)
        query, key, value = tokens.chunk(3, -1)

        def split_heads(tensor):
            return tensor.unflatten(-1, (self.heads, channels // self.heads)).transpose(1, 2)

        query, key, value = (split_heads(tensor) for tensor in (query, key, value))
        # Q/K规范化控制内容logit，scale与SDPA默认1/sqrt(d)互相补偿。
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
        scale = self.log_scale.exp().clamp(1, 20)[None, :, None, None]
        query = query * (scale * math.sqrt(channels // self.heads)).to(query)
        key_valid = valid.permute(0, 2, 3, 1).reshape(batch * height * width, depth)
        key_valid = key_valid[:, None, None]
        # 缺几何时保留该参数的零梯度，便于同参数预算消融。
        strength = 2 * self.geometry_logit.sigmoid() * float(use_geometry)
        if pair_bias is None:
            attention_mask = torch.zeros_like(key_valid, dtype=query.dtype) + strength * 0
        else:
            if pair_bias.shape != (batch, height * width, depth, depth):
                raise ValueError("pair_bias 必须为 [B,H*W,D,D]")
            attention_mask = pair_bias.reshape(batch * height * width, 1, depth, depth).to(query)
            attention_mask = attention_mask * strength
        attention_mask = attention_mask.masked_fill(~key_valid, -torch.inf)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, height, width, depth, channels)
        attended = attended.permute(0, 4, 3, 1, 2).contiguous()
        updated = self.norm1(clean + self.output(attended)) * valid[:, None]
        return self.norm2(updated + self.ffn(updated)) * valid[:, None]
