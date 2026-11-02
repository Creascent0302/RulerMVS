"""跨ViT层选择与局部图像融合，为实际匹配特征提供可选的语义残差。

每个CNN格点都读取四层先验，而不是把一个ViT层永久绑定到一个空间尺度。
语义交互保持较宽的隐藏表示，最后才投回公共FPN通道数；没有跨相机注意力。
"""

import math

import torch
from torch import nn

from jar.encoder import norm
from jar.geometry import pixel_grid, sample_maps


class SpatialResidualBlock(nn.Module):
    """在当前图像分辨率联合处理空间位置与通道，不再次下采样。"""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, features):
        return self.activation(features + self.body(features))


class FeatureEnrichment(nn.Module):
    """保留CNN匹配路径，以局部特征选择四层冻结先验，再添加门控残差。

    ``forward(pyramid, foundation_maps, masks)`` 返回相同形状的FPN字典。
    foundation_maps 是本次编码已经取得的四层 [B*N,768,Hp,Wp]，本模块
    不运行ViT。masks各项为 [B,N,1,Hs,Ws]，沿用原编码器的有效格点。
    """

    strides = (4, 8, 16, 32)

    def __init__(self, feature_channels, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 128))
        blocks = int(settings.get("blocks", 2))
        gate = float(settings.get("initial_gate", 0.1))
        if hidden < 1 or blocks < 1 or not 0 < gate < 1:
            raise ValueError("feature_enrichment需正hidden_dim/blocks及(0,1)内的initial_gate")
        self.hidden = hidden
        self.prior_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(768, hidden, 1, bias=False),
                    norm(hidden),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        # 四层共享读取空间；独立先验适配器仍能保留不同层的含义。
        self.key = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.value = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.queries = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.gate_logits = nn.ParameterDict()
        for stride in self.strides:
            name = str(stride)
            self.queries[name] = nn.Sequential(
                nn.Conv2d(feature_channels, hidden, 1, bias=False),
                norm(hidden),
                nn.GELU(),
            )
            self.fusions[name] = nn.Sequential(
                nn.Conv2d(hidden * 2, hidden, 1, bias=False),
                norm(hidden),
                nn.GELU(),
                *[SpatialResidualBlock(hidden) for _ in range(blocks)],
            )
            self.outputs[name] = nn.Conv2d(hidden, feature_channels, 1, bias=False)
            self.gate_logits[name] = nn.Parameter(torch.tensor(math.log(gate / (1 - gate))))

    @staticmethod
    def _sample_layers(features, uv, shape):
        # 与FrozenFoundation相同：原图patch中心为7.5+16*i，CNN为stride*i。
        token_uv = (uv - 7.5) / 16.0
        samples = [sample_maps(layer, token_uv, padding_mode="border") for layer in features]
        sampled = torch.stack(samples, dim=1).transpose(-1, -2)
        return sampled.reshape(len(uv), len(features), -1, *shape)

    def forward(self, pyramid, foundation_maps, masks):
        if len(foundation_maps) != 4:
            raise ValueError("feature_enrichment需要四个原始ViT中间层")
        prior = [adapter(feature) for adapter, feature in zip(self.prior_adapters, foundation_maps)]
        keys = [self.key(feature) for feature in prior]
        values = [self.value(feature) for feature in prior]
        enriched = {}
        for stride in self.strides:
            features, valid = pyramid[stride], masks[stride]
            batch, views, channels, height, width = features.shape
            local = torch.where(valid, features, 0).flatten(0, 1)
            query = self.queries[str(stride)](local)
            uv = pixel_grid(height, width, device=features.device, stride=stride)
            uv = uv.reshape(1, -1, 2).expand(batch * views, -1, -1)
            sampled_keys = self._sample_layers(keys, uv, (height, width))
            sampled_values = self._sample_layers(values, uv, (height, width))
            logits = (sampled_keys * query[:, None].float()).sum(2) / math.sqrt(self.hidden)
            layer_weights = logits.softmax(dim=1)
            semantic = (sampled_values * layer_weights[:, :, None]).sum(1)
            fused = self.fusions[str(stride)](torch.cat((query, semantic.to(query.dtype)), dim=1))
            residual = self.outputs[str(stride)](fused).to(features.dtype)
            # 门控位于所有分支归一化之后。后续处理的是CNN+残差，不再单独归一化残差。
            gate = self.gate_logits[str(stride)].sigmoid().to(residual.dtype)
            output = local + gate * residual
            output = output.reshape(batch, views, channels, height, width)
            enriched[stride] = torch.where(valid, output, 0)
        return enriched
