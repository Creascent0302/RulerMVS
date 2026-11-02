"""跨表面判断的粗细质量池化；细层保留峰内定位职责。

该池化不把同一RGB的粗细判断视为独立似然。门控在两个组分布之间
选择保留程度；它不能恢复已被topK丢弃的表面，也不认证候选组互斥。
"""

import math

import torch
from torch import nn


def group_log_probabilities(scores, parent_probability):
    """返回粗组log p、细组log q；内部始终使用FP32且粗质量停止梯度。"""
    batch, depth, height, width = scores.shape
    if depth % 3:
        raise ValueError("细候选须为K组，每组3点")
    groups = depth // 3
    if parent_probability.shape != (batch, groups, height, width):
        raise ValueError("parent_probability须对应每个细候选组")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        parent = parent_probability.detach().float().clamp_min(0)
        total = parent.sum(1, keepdim=True)
        parent = torch.where(total > 0, parent / total.clamp_min(1e-12),
                             torch.full_like(parent, 1 / groups))
        parent = parent.clamp_min(1e-8)
        parent = parent / parent.sum(1, keepdim=True)
        grouped = scores.float().reshape(batch, groups, 3, height, width)
        group_score = grouped.logsumexp(2)
        return parent.log(), group_score.log_softmax(1)


def pool_group_scores(scores, parent_probability, gate):
    """eta=0保留旧logits；eta=1保留粗组质量，组内细层分布不变。"""
    log_parent, log_fine = group_log_probabilities(scores, parent_probability)
    with torch.autocast(device_type=scores.device.type, enabled=False):
        gate = torch.as_tensor(gate, dtype=torch.float32, device=scores.device)
        correction = gate * (log_parent - log_fine)
        return scores.float() + correction.repeat_interleave(3, dim=1)


class HandoffGate(nn.Module):
    """根据已有候选特征和6个判断质量量，调节粗细组分布的保留比例。

    新输出层weight=0、bias=-2，初始eta约0.1192，明确不是恒等初始化。
    gate的输入统计停止梯度；实际分布池化对细scores仍保持正确梯度。
    """

    def __init__(self, channels, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 6, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.)

    def forward(self, features, scores, parent_probability, null_mass, source_valid):
        log_parent, log_fine = group_log_probabilities(scores, parent_probability)
        with torch.no_grad(), torch.autocast(device_type=scores.device.type, enabled=False):
            parent = log_parent.exp()
            fine = log_fine.detach().exp()
            full_probability = scores.detach().float().softmax(1)
            pooled = (features.detach().float() * full_probability[:, None]).sum(2)
            # K=1时两熵自然为0，使用log(2)避免0/0。
            denominator = math.log(max(parent.shape[1], 2))
            parent_entropy = -(parent * log_parent).sum(1, keepdim=True) / denominator
            fine_entropy = -(fine * log_fine.detach()).sum(1, keepdim=True) / denominator
            agreement = (parent * fine).sum(1, keepdim=True)
            divergence = (parent * (log_parent - log_fine.detach())).sum(1, keepdim=True)
            divergence = divergence.clamp(0, 8) / 8
            mean_null = null_mass.detach().float().mean(2)
            if source_valid.shape[1]:
                mean_source = source_valid.detach().float().mean((1, 2))[:, None]
            else:
                mean_source = torch.zeros_like(mean_null)
            quality = torch.cat((parent_entropy, fine_entropy, agreement, divergence,
                                 mean_null, mean_source), 1)
            gate_input = torch.cat((pooled, quality), 1)
        gate = self.net(gate_input).float().sigmoid()
        with torch.autocast(device_type=scores.device.type, enabled=False):
            correction = gate * (log_parent - log_fine)
            corrected = scores.float() + correction.repeat_interleave(3, dim=1)
        return corrected, gate
