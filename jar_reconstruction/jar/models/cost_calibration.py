"""以观测分数的尺度限制语义残差，保留原深度分支的总概率质量。"""

import torch
from torch import nn


class ObservationScaledCost(nn.Module):
    """沿候选轴归一化残差形状，而不是让其任意放大匹配分数。

    scores/evidence/valid: [B,D,H,W]。evidence是冻结描述子的实际相似度，
    不代表匹配正确。合法候选、no-match与遮挡定义均由原模型负责。
    """

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = float(epsilon)
        if not 0 < self.epsilon < 1:
            raise ValueError("候选校准epsilon须为(0,1)内的数")

    def forward(self, scores, evidence, valid):
        valid = valid.bool()
        original = torch.where(valid, scores.float(), 0.0)
        observed = torch.where(valid, evidence.float(), 0.0)
        count = valid.sum(1, keepdim=True).clamp_min(1)

        def center(values):
            mean = values.sum(1, keepdim=True) / count
            return torch.where(valid, values - mean, 0.0)

        observed = center(observed)
        residual = center(original - torch.where(valid, evidence.float(), 0.0))
        observed_variance = observed.square().sum(1, keepdim=True) / count
        observed_scale = (observed_variance + self.epsilon**2).sqrt()
        observed_scale = torch.where(observed_variance > self.epsilon**2, observed_scale, 0.0)
        residual_scale = (residual.square().sum(1, keepdim=True) / count + self.epsilon**2).sqrt()
        correction = observed_scale * (residual / residual_scale).tanh()
        proposed = observed + center(torch.where(valid, correction, 0.0))

        # 增加同一常数不改变条件深度分布。匹配原logsumexp，精确保留
        # 深度分支相对no-match的总质量，而非仅保留分数均值。
        # 无合法候选时临时开放一个零哨兵，避免全-inf的反向出现NaN。
        empty = ~valid.any(1, keepdim=True)
        first = torch.arange(scores.shape[1], device=scores.device)[None, :, None, None] == 0
        safe_valid = valid | (empty & first)
        before = original.masked_fill(~safe_valid, -torch.inf).logsumexp(1, keepdim=True)
        after = proposed.masked_fill(~safe_valid, -torch.inf).logsumexp(1, keepdim=True)
        return torch.where(valid, proposed + before - after, 0.0)
