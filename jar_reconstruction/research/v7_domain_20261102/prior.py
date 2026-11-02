"""粗候选的可观测性条件上下文修正；不新增图像编码或投影。

先验只回答缺少可靠匹配时哪些深度更合理，不能认证表面可见性。
输入为已经对齐的四层图像特征、源质量与候选深度，不接收标签。
"""

import torch
from torch import nn


class EvidencePrior(nn.Module):
    """在弱几何证据区域补充粗层 logits，初始化严格输出零。

    information 是已有源注意力的 sum(w * |d pi / dZ|²)，单位为
    特征像素²/规范化深度单位²。Z * sqrt(information) 无量纲；
    其中 w 包含源拒绝质量，因此它不是纹理充分性或 Fisher 信息证明。

    默认 raw 为四个 [B, 128, H, W] 张量；输出 [B, 96, H, W]。
    source 质量停止梯度，避免模型通过主动降低证据质量放大此旁路。
    """

    def __init__(self, raw_channels=128, levels=4, candidates=96, hidden=64,
                 parallax_pixels=0.5):
        super().__init__()
        if hidden % 8 or parallax_pixels <= 0:
            raise ValueError("hidden须为8的倍数，parallax_pixels须大于0")
        self.levels = int(levels)
        self.candidates = int(candidates)
        self.parallax_pixels = float(parallax_pixels)
        self.project = nn.Sequential(
            nn.Conv2d(raw_channels * levels, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=3, dilation=3,
                                 groups=hidden)
        self.activation = nn.SiLU()
        self.output = nn.Conv2d(hidden, candidates, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def quality_gate(self, information, null_mass, candidates):
        """返回[B,D,H,W]软门；不是可见性掩码或可删除评测像素的置信度。"""
        with torch.no_grad():
            information = information.detach().float()
            null_mass = null_mass.detach().float()
            if information.ndim == 5:
                information = information[:, 0]
            if null_mass.ndim == 5:
                null_mass = null_mass[:, 0]
            information = information.nan_to_num(0., posinf=0., neginf=0.).clamp_min(0.)
            null_mass = null_mass.nan_to_num(1., posinf=1., neginf=1.).clamp(0., 1.)
            depth = candidates.detach().float().clamp_min(0.)
            parallax_squared = information * depth.square()
            threshold_squared = self.parallax_pixels ** 2
            weak = threshold_squared / (threshold_squared + parallax_squared)
            return null_mass + (1. - null_mass) * weak

    def forward(self, raw, information, null_mass, candidates):
        if len(raw) != self.levels or candidates.shape[1] != self.candidates:
            raise ValueError("粗层上下文层数或深度候选数与EvidencePrior配置不符")
        features = self.project(torch.cat(raw, dim=1))
        spatial = (self.local(features) + self.context(features)) * 0.5
        logits = self.output(self.activation(features + spatial)).float()
        return logits * self.quality_gate(information, null_mass, candidates)
