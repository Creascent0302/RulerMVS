"""第二轮保持约束：与旧hinge取值相同，在无退化边界选择零子梯度。"""

import torch
from torch.nn import functional as F

from jar.pretraining.losses import masked_mean


def retention_penalty(prediction, teacher, inputs, targets, domain, margin=0.0):
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid = valid & inputs["view_mask"][:, :1, None, None]
    valid = valid & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid = valid & (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(valid, truth, 1.).log()
    current = prediction["depth"].float().clamp_min(1e-6).log()
    previous = teacher["depth"].detach().float().clamp_min(1e-6).log()
    regret = F.relu((current - truth).abs() - (previous - truth).abs() - margin)
    return masked_mean(regret, valid)
