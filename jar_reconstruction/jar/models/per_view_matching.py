"""先保留各源视图的匹配证据，再学习可靠性；不创造几何支持。"""

import torch
from torch import nn

from jar.geometry import safe_masked_softmax


class PerViewMatchingReliability(nn.Module):
    """共享的小型评分器，对相机排列等变，并支持任意有效源视图数量。

    输入均为 [source, depth, pixel]。不读取视图编号或监督标签，最后一层
    零初始化使初始聚合等同有效源视图的算术平均。首步更新末层，之后
    梯度进入隐藏层；这是为了从现有最佳 checkpoint 连续地开始实验。
    """

    def __init__(self, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 16))
        self.max_logit = float(settings.get("max_logit", 4.0))
        if hidden < 1 or self.max_logit <= 0:
            raise ValueError("逐视图匹配要求hidden_dim>0且max_logit>0")
        self.diagnostics = bool(settings.get("diagnostics", False))
        self.reliability = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            # 共享输出偏置会在跨视图softmax中抵消，因此不引入该冗余参数。
            nn.Linear(hidden, 1, bias=False),
        )
        nn.init.zeros_(self.reliability[-1].weight)

    def forward(
        self, cosine, valid, local_correlation, parallax, source_texture, reference_texture
    ):
        count = valid.sum(0).clamp_min(1)
        original_mean = torch.where(valid, cosine, 0).sum(0) / count
        source_fraction = valid.sum(0).float() / max(cosine.shape[0] - 1, 1)
        reference_texture = reference_texture.expand_as(cosine)
        description = torch.stack(
            (
                cosine,
                cosine - original_mean[None],
                local_correlation,
                torch.log1p(parallax.clamp_min(0)).clamp_max(8) / 8,
                torch.log1p(100 * source_texture.clamp_min(0)).clamp_max(6) / 6,
                torch.log1p(100 * reference_texture.clamp_min(0)).clamp_max(6) / 6,
                original_mean[None].expand_as(cosine),
                source_fraction[None].expand_as(cosine),
            ),
            dim=-1,
        )
        # 不让缺失视图中的任意填充值影响有效视图；网络也没有跨视图归一化。
        description = torch.where(valid[..., None], description.float(), 0)
        raw = self.reliability(description).float().squeeze(-1)
        logits = self.max_logit * torch.tanh(raw / self.max_logit)
        weights = safe_masked_softmax(logits, valid, dim=0)
        masked_cosine = torch.where(valid, cosine.float(), 0)
        mean = (weights * masked_cosine).sum(0)
        squared_error = torch.where(valid, (cosine.float() - mean[None]).square(), 0)
        variance = (weights * squared_error).sum(0)
        return torch.stack((mean, variance), dim=0), weights, logits
