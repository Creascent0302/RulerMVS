"""先选择表面组，再在同一组内连续定位；禁止跨组平均深度。

每三个连续候选属于同一父组。组内深度期望消除离散点赢家跳变，
组间依据总质量选择，避免低质量尖峰击败高质量宽峰。
候选组仍可能重叠，不能把组标签理解为已经认证的物理表面。
"""

import math

import torch
from torch import nn


def masked_probability(logits, valid, dim):
    """全无效时返回全零；softmax与概率始终采用FP32。"""
    safe = logits.float().masked_fill(~valid, -1e9)
    probability = safe.softmax(dim) * valid
    return probability / probability.sum(dim, keepdim=True).clamp_min(1e-12)


class QualityGate(nn.Module):
    """只读六个标量质量统计，129参数；不池化高维候选特征。

    输入依次为父/细组熵、分布内积、Jensen–Shannon分歧、null质量、
    有效源比例。统计停止梯度，门控本身仍由实际评分损失学习。
    """

    def __init__(self, hidden=16, bias=-2.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias)

    def forward(self, parent, fine, source_null, source_valid):
        with torch.no_grad():
            p, q = parent.detach(), fine.detach()
            lp, lq = p.clamp_min(1e-12).log(), q.clamp_min(1e-12).log()
            normalizer = math.log(max(p.shape[1], 2))
            parent_entropy = -(p * lp).sum(1, keepdim=True) / normalizer
            fine_entropy = -(q * lq).sum(1, keepdim=True) / normalizer
            agreement = (p * q).sum(1, keepdim=True)
            middle = ((p + q) * .5).clamp_min(1e-12).log()
            divergence = .5 * (p * (lp - middle) + q * (lq - middle)).sum(1, keepdim=True)
            divergence = (divergence / math.log(2)).clamp(0, 1)
            quality = torch.cat((parent_entropy, fine_entropy, agreement, divergence,
                                 source_null.detach(), source_valid.detach()), dim=1)
        return self.net(quality.to(self.net[0].weight.dtype)).float().sigmoid()


class HierarchicalHandoff(nn.Module):
    """独立的候选读出，可选粗/细组质量池化。

    marginal：仅由细候选总质量选组；无新增参数。
    parent：门控执行log-opinion pool，不把同一RGB的两次判断视为
    独立Bayes证据。父质量停止梯度，缺父质量时退回当前细组分布。

    输入scores/valid/candidate_depth为[B,3K,H,W]，parent_probability为
    [B,K,H,W]。source_null/source_valid是外部聚合的[B,1,H,W]质量量，
    也接受[B,H,W]；它们只影响门控，不替代候选有效性判断。
    """

    def __init__(self, mode='marginal', hidden=16, gate_bias=-2.):
        super().__init__()
        if mode not in ('marginal', 'parent'):
            raise ValueError("mode须为marginal或parent")
        self.mode = mode
        self.gate = QualityGate(hidden, gate_bias) if mode == 'parent' else None

    @staticmethod
    def _quality_map(value, default, example):
        if value is None:
            return torch.full_like(example, default)
        if value.ndim == 3:
            value = value[:, None]
        if value.shape != example.shape:
            raise ValueError("源质量须为[B,1,H,W]或[B,H,W]")
        return torch.nan_to_num(value.float(), nan=default).clamp(0, 1)

    def forward(self, scores, valid, candidate_depth, parent_probability=None,
                source_null=None, source_valid=None):
        if scores.ndim != 4 or scores.shape[1] % 3:
            raise ValueError("候选须为[B,3K,H,W]，每组连续三个点")
        if valid.shape != scores.shape or candidate_depth.shape != scores.shape:
            raise ValueError("scores、valid、candidate_depth须同形状")
        batch, count, height, width = scores.shape
        groups = count // 3
        if groups == 0:
            raise ValueError("至少需要一个候选组")

        with torch.autocast(device_type=scores.device.type, enabled=False):
            valid = valid.bool() & torch.isfinite(scores) & torch.isfinite(candidate_depth)
            valid = valid & (candidate_depth > 0)
            safe_scores = torch.where(valid, scores.float(), -1e9)
            grouped = safe_scores.reshape(batch, groups, 3, height, width)
            grouped_valid = valid.reshape_as(grouped)
            group_valid = grouped_valid.any(2)
            pixel_valid = group_valid.any(1, keepdim=True)
            group_evidence = grouped.logsumexp(2).masked_fill(~group_valid, -1e9)
            log_fine = group_evidence.log_softmax(1)
            fine = masked_probability(group_evidence, group_valid, 1)
            conditional = masked_probability(grouped, grouped_valid, 2)
            gate = torch.zeros_like(pixel_valid, dtype=torch.float32)
            corrected = safe_scores
            probability = fine

            if self.mode == 'parent':
                if parent_probability is None or parent_probability.shape != fine.shape:
                    raise ValueError("parent模式需要与每组对应的[B,K,H,W]父质量")
                parent = torch.nan_to_num(parent_probability.detach().float(),
                                          nan=0., posinf=0., neginf=0.).clamp_min(0)
                parent = parent * group_valid
                mass = parent.sum(1, keepdim=True)
                parent = torch.where(mass > 0, parent / mass.clamp_min(1e-12), fine.detach())
                # 零父质量不是物理遮挡证明，不永久否决细层的新证据。
                parent = parent.clamp_min(1e-8) * group_valid
                parent = parent / parent.sum(1, keepdim=True).clamp_min(1e-12)
                null = self._quality_map(source_null, 0., gate)
                support = self._quality_map(source_valid, 1., gate)
                gate = self.gate(parent, fine, null, support) * pixel_valid
                correction = gate * (parent.clamp_min(1e-12).log() - log_fine)
                probability = masked_probability(log_fine + correction, group_valid, 1)
                corrected = (safe_scores + correction.repeat_interleave(3, dim=1)).masked_fill(~valid, -1e9)

            joint = (probability[:, :, None] * conditional).flatten(1, 2)
            safe_depth = torch.where(valid, candidate_depth.float(), 0.).reshape_as(grouped)
            group_depth = (conditional * safe_depth).sum(2)
            selected_group = probability.argmax(1, keepdim=True)
            within_group = conditional.argmax(2).gather(1, selected_group)
            selected_index = selected_group * 3 + within_group
            depth = group_depth.gather(1, selected_group) * pixel_valid
            confidence = probability.gather(1, selected_group) * pixel_valid
            return {
                'scores': corrected,
                'joint_probability': joint,
                'conditional_probability': conditional,
                'group_probability': probability,
                'group_depth': group_depth,
                'selected_group': selected_group,
                'selected_index': selected_index,
                'depth': depth,
                'confidence': confidence,
                'valid': pixel_valid,
                'gate': gate,
            }
