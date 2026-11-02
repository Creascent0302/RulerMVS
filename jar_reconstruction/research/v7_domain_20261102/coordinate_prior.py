"""用实际候选深度查询上下文，避免固定第k通道跨域代表不同深度。

候选深度已经由原模型除以该域 depth_scale；这里不读取GT或重新归一
每帧范围。log深度的连续项和Fourier项共同保留尺度与多频率变化。
"""

import math

import torch
from torch import nn

from .prior import EvidencePrior


class CoordinateEvidencePrior(EvidencePrior):
    """共享图像系数与候选坐标码点积，粗层任意候选数复用同一参数。

    四层raw→64维空间上下文沿用第一轮；仅把固定96通道读出替换为
    32维系数，与共享深度MLP的32维编码做点积。图像输出零初始化，
    第一轮更新首先训练输出层，之后梯度自然进入上下文和深度编码。
    """

    def __init__(self, raw_channels=128, levels=4, candidates=96, hidden=64,
                 parallax_pixels=0.5, code_channels=32):
        super().__init__(raw_channels, levels, candidates, hidden, parallax_pixels)
        self.code_channels = int(code_channels)
        self.output = nn.Conv2d(hidden, self.code_channels, 1)
        self.register_buffer("frequencies", torch.tensor([.25, .5, 1., 2., 4., 8.]))
        self.depth_code = nn.Sequential(
            nn.Linear(13, self.code_channels), nn.SiLU(),
            nn.Linear(self.code_channels, self.code_channels),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_depth(self, candidates):
        """[B,D,H,W]→[B,D,H,W,32]；同一深度不受候选索引影响。"""
        log_depth = candidates.float().clamp_min(1e-6).log()
        phase = log_depth[..., None] * self.frequencies
        position = torch.cat((log_depth[..., None] / 8., phase.sin(), phase.cos()), dim=-1)
        return self.depth_code(position.to(self.depth_code[0].weight.dtype))

    def forward(self, raw, information, null_mass, candidates):
        if len(raw) != self.levels:
            raise ValueError("上下文特征层数与CoordinateEvidencePrior配置不符")
        features = self.project(torch.cat(raw, dim=1))
        spatial = (self.local(features) + self.context(features)) * .5
        coefficients = self.output(self.activation(features + spatial)).float()
        code = self.encode_depth(candidates).float()
        logits = torch.einsum("bchw,bdhwc->bdhw", coefficients, code) / math.sqrt(self.code_channels)
        return logits * self.quality_gate(information, null_mass, candidates)
