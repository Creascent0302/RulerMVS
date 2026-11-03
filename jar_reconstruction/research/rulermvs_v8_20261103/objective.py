"""V8 复用 V6 监督；仅使单位扰动不改变近处相对误差的分母下限。"""

import torch

from jar.pretraining.losses import masked_mean
from research.rulermvs_v6_20261101.objective import objective as v6_objective


def objective(prediction, inputs, targets, domain, settings, **kwargs):
    terms = v6_objective(prediction, inputs, targets, domain, settings, **kwargs)
    if not settings.get("unit_invariant_relative_floor", False):
        return terms
    scale = float(prediction["unit_scale"])
    nominal = float(domain.get("nominal_depth_scale", domain["depth_scale"]))
    truth = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid &= inputs["view_mask"][:, :1, None, None]
    valid &= torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid &= (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(valid, truth, 1.)
    estimate = prediction["depth"].float().clamp_min(1e-6)
    relative = ((estimate - truth).abs() / truth.clamp_min(.1 * nominal / scale)).clamp_max(100.)
    updated = masked_mean(relative, valid)
    terms["total"] = terms["total"] + .2 * (updated - terms["abs_rel"])
    terms["abs_rel"] = updated
    if settings.get("fine_readout_weight", 0):
        raise ValueError("V8当前仅支持fine_readout_weight=0，避免未修正该独立分母")
    return terms
