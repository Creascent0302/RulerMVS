"""连续候选的集合监督；只在训练时读取真值，不增加推理模块。"""

import torch
from torch.nn import functional as F


def candidate_supervision(stage, truth, mask, temperature=0.05, depth_range=None):
    """先判断哪些候选能到达真值，再校准候选分数和连续位置。

    分类目标在合法候选中按 exp(-|log(z_candidate / z_gt)| / temperature)
    分配，并停止目标梯度；评价连续候选的实际质量，而非采样中心标签。
    所有可达候选均接受位置监督，避免只有旧赢家学会定位。

    若没有区间覆盖真值，只给距离真值最近的区间分类监督；不给无法
    到达真值的 offset 强加精确位置监督。重叠或重复的等质量候选平等
    分配目标质量，不用 argmin 随意指定其中一个为唯一正确答案。
    重复峰总质量可能随副本数增加；当前粗层无重复，不额外做复杂去重。

    stage 的张量均为 [B,K,H,W]，truth/mask 为 [B,1,H,W]。
    depth_range 是模型实际输出截断范围，来自配置，不可由真值设置。
    返回分类、风险、定位项，由调用方保留原损失权重，便于等预算比较。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    scores = stage["scores"].float()
    candidates = stage["candidate_depth"].float()
    lower, upper = stage["lower"].float(), stage["upper"].float()
    legal = stage["valid"].bool()
    legal = legal & torch.isfinite(scores) & torch.isfinite(candidates)
    legal = legal & torch.isfinite(lower) & torch.isfinite(upper) & (candidates > 0)
    candidates = torch.where(legal, candidates, 1.0)
    lower = torch.where(legal, lower, 1.0).clamp_min(1e-6)
    upper = torch.where(legal, upper, 1.0)
    if depth_range is not None:
        near, far = depth_range
        if not 0 < near < far:
            raise ValueError("depth_range 必须满足 0 < near < far")
        lower = lower.clamp_min(near)
        upper = upper.clamp_max(far)
    legal = legal & (upper >= lower) & (upper > 0)
    upper = upper.clamp_min(1e-6)

    labelled = mask.bool() & torch.isfinite(truth) & (truth > 0)
    target = torch.where(labelled, truth.float(), 1.0)
    log_target = target.log()
    quality_error = (candidates.clamp_min(1e-6).log() - log_target).abs()
    interval_error = torch.maximum(lower.log() - log_target, log_target - upper.log()).clamp_min(0)

    with torch.no_grad():
        reachable = legal & (interval_error <= 1e-6)
        covered = reachable.any(1, keepdim=True)
        interval_distance = interval_error.masked_fill(~legal, torch.inf)
        closest = interval_distance.amin(1, keepdim=True)
        nearest_interval = legal & (interval_distance <= closest + 1e-7)
        eligible = torch.where(covered, legal, nearest_interval)
        quality_logits = (-quality_error / temperature).masked_fill(~eligible, -1e4)
        target_probability = quality_logits.softmax(1) * eligible
        target_probability = target_probability / target_probability.sum(1, keepdim=True).clamp_min(1e-12)

    clean_scores = torch.where(legal, scores, -1e4)
    log_probability = F.log_softmax(clean_scores, dim=1)
    probability = log_probability.exp() * legal
    supervised = labelled & legal.any(1, keepdim=True)
    count = supervised.sum().clamp_min(1)

    classification_map = -(target_probability * log_probability).sum(1, keepdim=True)
    risk_map = (probability * quality_error.detach()).sum(1, keepdim=True)
    location_map = (quality_error * reachable).sum(1, keepdim=True) / reachable.sum(1, keepdim=True).clamp_min(1)
    classification = torch.where(supervised, classification_map, 0).sum() / count
    risk = torch.where(supervised, risk_map, 0).sum() / count
    location_mask = supervised & covered
    location = torch.where(location_mask, location_map, 0).sum() / location_mask.sum().clamp_min(1)
    return {
        "classification": classification,
        "risk": risk,
        "location": location,
        "covered": (supervised & covered).sum() / count,
        "supervised_pixels": supervised.sum(),
        "target_probability": target_probability,
    }
