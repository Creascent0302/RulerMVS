"""把冻结教师的深度候选排序蒸馏到本地匹配特征；没有推理参数。

这只是知识蒸馏研究基线。教师与学生必须使用同一参考像素、深度候选
和逐源几何掩码；不能在不同depth chunk内分别归一化。
"""

import math

import torch
from torch import nn


class MatchingDistillation(nn.Module):
    """对[source, depth, pixel]曲线返回加权KL总和及有效权重总和。"""

    def __init__(self, settings):
        super().__init__()
        self.teacher_temperature = float(settings.get("temperature_teacher", 12.0))
        self.student_temperature = float(settings.get("temperature_student", 8.0))
        self.information_weighting = bool(settings.get("information_weighting", True))
        self.minimum_information = float(settings.get("minimum_information", 0.001))
        values = (self.teacher_temperature, self.student_temperature)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("蒸馏温度须为有限正数；这里温度表示相似度乘数")
        if not 0 <= self.minimum_information < 1:
            raise ValueError("minimum_information须位于[0,1)")

    def forward(self, student_cosine, teacher_cosine, geometric_mask):
        if student_cosine.shape != teacher_cosine.shape or student_cosine.shape != geometric_mask.shape:
            raise ValueError("学生、教师相似度和逐源掩码须同为[source,depth,pixel]")
        if student_cosine.ndim != 3:
            raise ValueError("匹配蒸馏需要完整的[source,depth,pixel]候选曲线")
        # 教师的概率、有效性和信息量均停止梯度，不能通过软化教师逃避监督。
        teacher = teacher_cosine.detach().float()
        student = student_cosine.float()
        valid = geometric_mask.bool() & torch.isfinite(teacher) & torch.isfinite(student)
        teacher = torch.where(valid, teacher, 0)
        student = torch.where(valid, student, 0)
        teacher_logits = (self.teacher_temperature * teacher).masked_fill(~valid, -1e4)
        student_logits = (self.student_temperature * student).masked_fill(~valid, -1e4)
        teacher_log_probability = teacher_logits.log_softmax(dim=1)
        student_log_probability = student_logits.log_softmax(dim=1)
        probability = torch.where(valid, teacher_log_probability.exp(), 0)
        divergence = (probability * (teacher_log_probability - student_log_probability)).sum(1)
        count = valid.sum(1)
        weight = (count >= 2).float()
        if self.information_weighting:
            log_count = count.clamp_min(2).float().log()
            entropy = -(probability * teacher_log_probability).sum(1)
            information = (1 - entropy / log_count).clamp(0, 1)
            information = torch.where(information >= self.minimum_information, information, 0)
            weight = weight * information
        return (divergence * weight).sum(), weight.sum()
