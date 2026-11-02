"""由几何候选读取原始单图特征，再预测有界的选面分数修正。

原始特征在候选条件非线性之后才压缩。这里不新增图像编码器，
也不将单图结构解释为多目观测；无效候选保持原来的分数。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import pixel_grid
from .deep_encoder import sample_foundation_maps


class RawFeatureSurfaceReader(nn.Module):
    """共享候选读取器，输入为单个样本、单个参考相机。

    features: [K,19,H4,W4]；raw_maps: [L,C,H16,W16]。
    valid、base_logits 和返回分数均为 [K,H4,W4]。
    memory_rank>0 先在原生格点上做线性低秩压缩与展开，用于消融。
    """

    def __init__(self, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 64))
        self.channels = int(settings.get("memory_channels", 768))
        self.layers = int(settings.get("memory_layers", 4))
        self.rank = int(settings.get("memory_rank", 0))
        self.conditioned = bool(settings.get("conditioned", True))
        self.chunk_size = int(settings.get("chunk_size", 1024))
        self.max_residual = float(settings.get("max_residual", 4.0))
        self.residual_scale = float(settings.get("residual_scale", 1.0))
        if min(hidden, self.channels, self.layers, self.chunk_size) < 1:
            raise ValueError("原始特征读取器的维度、层数和chunk_size须为正数")
        if not 0 <= self.rank <= self.channels:
            raise ValueError("memory_rank须在0与memory_channels之间；0表示不压缩")
        if not math.isfinite(self.max_residual) or self.max_residual <= 0:
            raise ValueError("max_residual须为有限正数")
        if not math.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= 1:
            raise ValueError("residual_scale须在[0,1]内；0表示直接旁路")

        self.query_encoder = nn.Sequential(
            nn.Linear(19, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.competition = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU())
        if self.conditioned:
            self.layer_router = nn.Linear(hidden, self.layers)
            self.condition = nn.Linear(hidden, self.channels)
        else:
            self.layer_logits = nn.Parameter(torch.zeros(self.layers))
        if self.rank:
            self.memory_reduce = nn.Linear(self.channels, self.rank, bias=False)
            self.memory_expand = nn.Linear(self.rank, self.channels, bias=False)
        self.memory_norm = nn.LayerNorm(self.channels)
        self.readout = nn.Sequential(
            nn.Linear(self.channels, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _prepare_memory(self, raw_maps):
        # 缓存与几何采样保留FP32；低秩对照只作用于原生patch格点。
        with torch.autocast(device_type=raw_maps.device.type, enabled=False):
            memory = raw_maps.float()
            if self.rank:
                tokens = memory.permute(0, 2, 3, 1)
                tokens = F.linear(tokens, self.memory_reduce.weight.float())
                tokens = F.linear(tokens, self.memory_expand.weight.float())
                memory = tokens.permute(0, 3, 1, 2)
        return memory

    def _query(self, features, valid):
        features = torch.where(valid[..., None], features, 0.0)
        query = self.query_encoder(features)
        query = torch.where(valid[..., None], query, 0.0)
        count = valid.sum(0).clamp_min(1)[..., None]
        mean = query.sum(0) / count
        maximum = query.masked_fill(~valid[..., None], torch.finfo(query.dtype).min).amax(0)
        maximum = torch.where(valid.any(0)[..., None], maximum, 0.0)
        competitors = torch.cat(
            (query, mean[None].expand_as(query), maximum[None].expand_as(query)), -1
        )
        return self.competition(competitors)

    def _read(self, query, memory, uv):
        with torch.autocast(device_type=memory.device.type, enabled=False):
            sampled = sample_foundation_maps(memory, uv[None].expand(self.layers, -1, -1))
        if not self.conditioned:
            # 静态读取与候选无关：同一像素只计算一次高维读取，再共享给所有候选。
            with torch.autocast(device_type=memory.device.type, enabled=False):
                weights = self.layer_logits.float().softmax(-1)
                mixed = torch.einsum("l,lpc->pc", weights, sampled.float())
            read = self.readout(F.gelu(self.memory_norm(mixed)))
            return read[None].expand(query.shape[0], -1, -1)
        weights = self.layer_router(query).float().softmax(-1)
        shift = self.condition(query)
        # 先在完整C维空间中按候选混合，再执行条件非线性；不能交换成固定压缩。
        with torch.autocast(device_type=memory.device.type, enabled=False):
            mixed = torch.einsum("kpl,lpc->kpc", weights.float(), sampled.float())
        conditioned = F.gelu(self.memory_norm(mixed) + shift)
        return self.readout(conditioned)

    def forward(self, features, valid, raw_maps, base_logits):
        if self.residual_scale == 0:
            return base_logits
        if features.ndim != 4 or features.shape[1] != 19:
            raise ValueError("features须为[K,19,H4,W4]")
        count, _, height, width = features.shape
        if valid.shape != (count, height, width) or base_logits.shape != valid.shape:
            raise ValueError("valid与base_logits须为[K,H4,W4]")
        if raw_maps.ndim != 4 or raw_maps.shape[:2] != (self.layers, self.channels):
            raise ValueError("raw_maps须为[memory_layers,memory_channels,H16,W16]")
        memory = self._prepare_memory(raw_maps)
        pixels = pixel_grid(height, width, device=features.device, stride=4).reshape(-1, 2)
        tokens = features.permute(0, 2, 3, 1).reshape(count, -1, 19)
        valid = valid.bool().reshape(count, -1)
        residuals = []
        for start in range(0, height * width, self.chunk_size):
            stop = start + self.chunk_size
            legal = valid[:, start:stop]
            query = self._query(tokens[:, start:stop], legal)
            read = self._read(query, memory, pixels[start:stop])
            score = self.score(torch.cat((query, read), -1)).tanh()
            residual = self.max_residual * self.residual_scale * score
            residuals.append(torch.where(legal, residual[..., 0], 0.0))
        residual = torch.cat(residuals, 1).reshape(count, height, width)
        return base_logits + residual.to(base_logits.dtype)
