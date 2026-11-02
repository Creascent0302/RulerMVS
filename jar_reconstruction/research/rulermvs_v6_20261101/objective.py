"""以实际族内读出深度监督选族；替换旧细层分类/风险，不重复叠加。"""

import torch
from torch.nn import functional as F

from jar.pretraining.losses import depth_objective, masked_mean
from research.v2_training_20261028.objectives import fine_actual_regret


@torch.no_grad()
def group_targets(depth, truth, valid):
    """分类保留真实误差次序；仅期望风险截断，避免极端错面任意并列。"""
    valid = valid & torch.isfinite(depth) & (depth > 0)
    safe = torch.where(valid, depth.float(), 1.)
    error = (safe.clamp_min(1e-6).log() - truth.log()).abs()
    error = error.masked_fill(~valid, torch.inf)
    return error.argmin(1), error.clamp_max(4.), valid.any(1, keepdim=True)


def group_classification(scores, valid, target):
    """直接从logits计算组分类，极小目标概率也保留纠错梯度。"""
    grouped = scores.float().masked_fill(~valid, -1e9).unflatten(1, (-1, 3))
    return F.cross_entropy(grouped.logsumexp(2), target, reduction="none")


def fine_readout_loss(prediction, inputs, targets, domain):
    """直接监督stride4实际深度，避免最终上采样用邻居补偿本格点的错深度。

    误差形式与原全图目标相同。组选择仍由组分类负责，此项只对已经
    选择的深度读出求导；不增加推理操作、不跨候选族混合。
    """
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid = valid & inputs["view_mask"][:, :1, None, None]
    valid = valid & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid = valid & (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(valid, truth, 1.)[..., ::4, ::4]
    valid = valid[..., ::4, ::4] & prediction["fine"]["ref_valid"]
    estimate = torch.where(valid, prediction["fine"]["depth"].float(), 1.).clamp_min(1e-6)
    error = (estimate.log() - truth.log()).abs()
    error = error + .2 * ((estimate - truth).abs() / truth.clamp_min(.1)).clamp_max(100.)
    return masked_mean(error, valid)


def objective(prediction, inputs, targets, domain, settings, grouped=True,
              correspondence_weight=0.):
    terms = depth_objective(prediction, inputs, targets, domain, settings, correspondence_weight)
    readout_weight = float(settings.get("fine_readout_weight", 0.))
    if readout_weight < 0:
        raise ValueError("fine_readout_weight必须非负")
    if readout_weight:
        readout = fine_readout_loss(prediction, inputs, targets, domain)
        terms["total"] = terms["total"] + readout_weight * readout
        terms["fine_readout"] = readout
    original = fine_actual_regret(prediction, inputs, targets, domain)
    if not grouped:
        terms["total"] = terms["total"] + .1 * (original["new_risk"] - original["old_risk"])
        return terms
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid &= inputs["view_mask"][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid &= (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(valid, truth, 1.)[..., ::4, ::4]
    fine = prediction["fine"]
    valid = valid[..., ::4, ::4] & fine["ref_valid"]
    group_valid = fine["valid"].unflatten(1, (-1, 3)).any(2)
    target, error, eligible = group_targets(fine["group_depth"], truth, group_valid)
    valid &= eligible
    probability = fine["group_probability"].float()
    group_ce = group_classification(fine["scores"], fine["valid"], target)
    group_ce = masked_mean(group_ce, valid[:, 0])
    group_risk = masked_mean((probability * error).sum(1), valid[:, 0])
    # 位置头仍受原最近采样点的连续范围监督；最终深度损失训练组内权重。
    terms["total"] = terms["total"] + .25 * (group_ce - terms["fine_ce"])
    terms["total"] = terms["total"] + .1 * (group_risk - original["old_risk"])
    terms.update(group_ce=group_ce, group_risk=group_risk)
    return terms
