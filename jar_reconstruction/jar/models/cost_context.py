"""粗尺度代价体的残差上下文，保留原始匹配项与几何有效性。

这里只处理统一深度候选上的图像代价体，不建立完整场景稠密体素。
把深度方向和图像方向卷积分开，以较小开销累积邻域证据。
"""

import torch
from torch import nn
from torch.nn import functional as F

from jar.encoder import norm


class CostResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depth = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.space = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.depth_norm = norm(channels)
        self.space_norm = norm(channels)

    def forward(self, features):
        update = F.gelu(self.depth_norm(self.depth(features)))
        update = self.space_norm(self.space(update))
        return F.gelu(features + update)


class CostContext(nn.Module):
    def __init__(self, input_channels, settings):
        super().__init__()
        channels = int(settings.get("channels", 24))
        layers = int(settings.get("layers", 3))
        self.maximum = float(settings.get("max_residual", 4.0))
        if channels < 4 or layers < 1 or self.maximum <= 0:
            raise ValueError("cost_context 通道、层数和残差范围必须为正")
        # 逆深度坐标是固定物理候选位置，不从标签估计范围。
        self.input = nn.Conv3d(input_channels + 1, channels, 1)
        self.blocks = nn.Sequential(*[CostResidualBlock(channels) for _ in range(layers)])
        self.output = nn.Conv3d(channels, 1, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, cost, inverse_depths):
        minimum, maximum = inverse_depths.amin(), inverse_depths.amax()
        coordinate = 2 * (inverse_depths - minimum) / (maximum - minimum).clamp_min(1e-8) - 1
        coordinate = coordinate[None, None, :, None, None].expand(
            cost.shape[0], 1, -1, cost.shape[-2], cost.shape[-1]
        )
        features = self.input(torch.cat((cost, coordinate.to(cost.dtype)), dim=1))
        residual = self.output(self.blocks(features)).squeeze(1)
        return self.maximum * torch.tanh(residual / self.maximum)
