"""从多视角候选场选择初始表面，保留旧深度作为保守回退。

输入仅含模型推理的图像匹配证据、米制候选和旧初值，不接收训练标签。
默认19维接口中第9/10通道为零，兼容既有无额外 coarse 的 checkpoint。
include_current 仅用于额外评分曲线的研究消融。
"""

import torch
from torch import nn
from torch.nn import functional as F

class SpatialRiskHead(nn.Module):
    """共享候选评分；深度卷积只在同一候选的 3×3 邻域传播证据。"""

    def __init__(self, channels=19, width=24):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(channels, width, 1), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width),
                nn.SiLU(),
                nn.Conv2d(width, width, 1),
            ) for _ in range(2)
        ])
        self.exit = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)

    def forward(self, features, valid):
        # features: [K,C,H,W]，旧深度始终是候选0。
        x = self.entry(features)
        for block in self.blocks:
            x = F.silu(x + block(x))
        logits = self.exit(x)[:, 0]
        logits[0] = logits[0] + 1.0
        return logits.masked_fill(~valid, -1e4)


def interpolation_indices(rho, lattice):
    """同一组候选在所有证据曲线上共用逆深度插值位置。"""
    depth_count = lattice.numel()
    right = torch.searchsorted(lattice.contiguous(), rho.contiguous()).clamp(1, depth_count - 1)
    left = right - 1
    blend = ((rho - lattice[left]) / (lattice[right] - lattice[left])).clamp(0, 1)
    return right, left, blend


def sample(curves, rho, lattice, indices=None):
    """从同一真实逆深度格上插值，curves 为 [C,D,H,W]。"""
    count = curves.shape[0]
    right, left, blend = interpolation_indices(rho, lattice) if indices is None else indices
    low = curves.gather(1, left[None].expand(count, -1, -1, -1))
    high = curves.gather(1, right[None].expand(count, -1, -1, -1))
    return torch.lerp(low, high, blend[None]).permute(1, 0, 2, 3)


def standardize(curve, valid):
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (curve * valid).sum(1, keepdim=True) / count
    variance = ((curve - mean).square() * valid).sum(1, keepdim=True) / count
    return ((curve - mean) / variance.sqrt().clamp_min(.05)).clamp(-6, 6)


def _context_standardize(curve, valid):
    """先验上下文只对已知项统计；缺失项不充当负匹配分数。"""
    count = valid.sum(1, keepdim=True).clamp_min(1)
    clean = torch.where(valid, curve, 0.0)
    mean = clean.sum(1, keepdim=True) / count
    variance = ((clean - mean).square() * valid).sum(1, keepdim=True) / count
    value = (clean - mean) / variance.clamp_min(0.05**2).sqrt()
    return torch.where(valid, value.clamp(-6, 6), 0.0)


def _sample_context(curve, valid, rho, lattice, indices):
    """不跨缺失端点插入哨兵；返回数值和独立的已知标记。"""
    clean = torch.where(valid, curve, 0.0)
    coverage = sample(valid.float(), rho, lattice, indices)[:, 0]
    value = sample(clean, rho, lattice, indices)[:, 0] / coverage.clamp_min(1e-8)
    known = coverage >= 1 - 1e-5
    return torch.where(known, value, 0.0), known


def leave_one_view_out_vote(source_z, source_known, candidate_valid):
    """统计候选在删除任一有效源视图后仍为最优的比例。

    所有源视图共用同一候选深度；缺失/图外采样由 source_known 屏蔽。
    这衡量匹配结论对单视图扰动的稳定性，并非表面真实性的证明。
    """
    usable = source_known & candidate_valid[:, None]
    observed = usable.any(0)
    value = source_z * usable
    remaining_count = usable.sum(1, keepdim=True) - usable.long()
    remaining_sum = value.sum(1, keepdim=True) - value
    remaining_mean = remaining_sum / remaining_count.clamp_min(1)
    remaining_mean = remaining_mean.masked_fill(
        (remaining_count == 0) | ~candidate_valid[:, None], -1e4
    )
    best = remaining_mean.argmax(0)
    votes = F.one_hot(best, candidate_valid.shape[0]).permute(3, 0, 1, 2)
    vote_count = (votes * observed[None]).sum(1)
    return vote_count / observed.sum(0).clamp_min(1)


def build_inference_features(
    entry, memory_scores, old, include_current=False, proposal_mode="legacy",
    proposal_count=3, view_consensus=False, near=2.0, far=80.0,
    peak_separation_m=1.0,
):
    """纯推理构造：输入原生候选证据与二维旧初值，返回同 device 张量。

    entry 不需 targets、current_scores（默认关闭），可直接用于生产前向接线。
    """
    old = old.float().clamp(near, far)
    lattice = entry["inverse_depths"].float()
    valid_curve = entry["candidate_valid"][0]
    scores = memory_scores[0]
    if proposal_mode == "coarse_prior":
        prior = entry.get("proposal_prior")
        if prior is None:
            raise ValueError("coarse_prior 需要独立粗网格提议先验")
        scores, valid_curve = prior["scores"][0], prior["eligible"][0]
    depths = lattice.reciprocal()[:, None, None]
    if proposal_mode == "legacy":
        first = scores.argmax(0)
        first_depth = depths.expand_as(scores).gather(0, first[None])[0]
        # 保留旧检查点的逐像素行为，作为完全可复现的对照。
        separation = torch.maximum(first_depth * .08, first_depth.new_tensor(1.0))
        separated = (depths - first_depth).abs() > separation
        second_scores = scores.masked_fill(~separated, -1e4)
        second = second_scores.argmax(0)
        second_depth = depths.expand_as(scores).gather(0, second[None])[0]
        candidates = torch.stack((old, first_depth, second_depth))
        valid = torch.stack((torch.ones_like(first, dtype=torch.bool),
                             valid_curve.gather(0, first[None])[0] & (first_depth - old).abs().gt(.5),
                             valid_curve.gather(0, second[None])[0] & (second_depth - old).abs().gt(.5)
                             & (second_scores.amax(0) > -9999)))
    elif proposal_mode in ("metric_supported", "coarse_prior"):
        if proposal_count < 2:
            raise ValueError("proposal_count至少为2，含旧初值和一个新候选")
        # 几何合法不等于匹配正确；两源门槛只防止无证据的候选占用预算。
        remaining = valid_curve
        if proposal_mode != "coarse_prior":
            remaining = remaining & (entry["source_valid"][0].sum(0) >= 2)
        peak_depths, peak_valid = [], []
        for _ in range(proposal_count - 1):
            available = remaining.any(0)
            best = scores.masked_fill(~remaining, -1e4).argmax(0)
            peak = depths.expand_as(scores).gather(0, best[None])[0]
            peak_depths.append(peak)
            peak_valid.append(available)
            remaining = remaining & ((depths - peak).abs() > peak_separation_m)
        candidates = torch.cat((old[None], torch.stack(peak_depths)))
        valid = torch.cat((torch.ones_like(old[None], dtype=torch.bool),
                           torch.stack(peak_valid)))
        first_depth = peak_depths[0]
    else:
        raise ValueError(f"不支持的候选模式: {proposal_mode}")
    rho = candidates.reciprocal().clamp(lattice[0], lattice[-1])
    indices = interpolation_indices(rho, lattice)
    hero = 4 * entry["hero"][0, 0].float()[None]
    current = entry["current_scores"][0].float()[None] if include_current else None
    source = entry["source_cosine"][0].float()
    source_valid = entry["source_valid"][0].bool()
    context_known = None
    if proposal_mode == "coarse_prior":
        # memory 是粗节点提出的先验，hero 仍是原field的上下文；二者的
        # 可用范围不同，不能把新增先验的范围广播为hero实测支持。
        memory_valid = valid_curve[None]
        hero_valid = entry["candidate_valid"][0][None]
        memory_score, memory_known = _sample_context(scores[None], memory_valid, rho, lattice, indices)
        memory_z, _ = _sample_context(_context_standardize(scores[None], memory_valid), memory_valid, rho, lattice, indices)
        hero_score, hero_known = _sample_context(hero, hero_valid, rho, lattice, indices)
        hero_z, _ = _sample_context(_context_standardize(hero, hero_valid), hero_valid, rho, lattice, indices)
        current_score = torch.zeros_like(memory_score)
        current_z = torch.zeros_like(memory_z)
        if include_current:
            current_score, _ = _sample_context(current, hero_valid, rho, lattice, indices)
            current_z, _ = _sample_context(_context_standardize(current, hero_valid), hero_valid, rho, lattice, indices)
        relative_memory = torch.where(memory_known & memory_known[:1], memory_score - memory_score[:1], 0.0)
        context_known = {"memory": memory_known, "hero": hero_known}
    else:
        hero_score = sample(hero, rho, lattice, indices)[:, 0]
        memory_score = sample(scores[None], rho, lattice, indices)[:, 0]
        current_score = (
            sample(current, rho, lattice, indices)[:, 0]
            if include_current else torch.zeros_like(memory_score)
        )
        hero_z = sample(standardize(hero, valid_curve[None]), rho, lattice, indices)[:, 0]
        memory_z = sample(standardize(scores[None], valid_curve[None]), rho, lattice, indices)[:, 0]
        current_z = (sample(standardize(current, valid_curve[None]), rho, lattice, indices)[:, 0]
                     if include_current else torch.zeros_like(memory_z))
        relative_memory = memory_score - memory_score[:1]
    source_value = sample(source, rho, lattice, indices)
    source_z = sample(standardize(source, source_valid), rho, lattice, indices)
    source_known = sample(source_valid.float(), rho, lattice, indices) > .999
    # 至少两台源相机可合法采样才允许离开旧初值；掩码本身不证明无遮挡。
    if proposal_mode != "coarse_prior":
        valid[1:] &= source_known[1:].sum(1) >= 2
    support = source_known.float()
    known_count = support.sum(1).clamp_min(1)
    source_mean = (source_value * support).sum(1) / known_count
    source_z_mean = (source_z * support).sum(1) / known_count
    source_z_low = source_z.masked_fill(~source_known, 6).amin(1).clamp(-6, 6)
    source_z_high = source_z.masked_fill(~source_known, -6).amax(1).clamp(-6, 6)
    source_spread = (
        ((source_z - source_z_mean[:, None]).square() * support)
        .sum(1).div(known_count).sqrt()
    )
    source_count = source_known.float().mean(1)
    old_near = F.avg_pool2d(old[None, None], 3, 1, 1)[0, 0]
    first_near = F.avg_pool2d(first_depth[None, None], 3, 1, 1)[0, 0]
    old_gap = (candidates - old).abs()
    features = torch.stack((
        candidates.log().div(4.4), (candidates - old).div(20).clamp(-4, 4),
        (candidates - old_near).div(20).clamp(-4, 4),
        (candidates - first_near).div(20).clamp(-4, 4),
        old_gap.div(torch.maximum(old, old.new_tensor(2))).clamp(0, 3),
        memory_score.clamp(-10, 10).div(10), memory_z.div(6),
        hero_score.clamp(-10, 10).div(10), hero_z.div(6),
        current_score.clamp(-10, 10).div(10), current_z.div(6),
        relative_memory.clamp(-10, 10).div(10),
        source_mean, source_z_mean.div(6), source_z_low.div(6),
        source_z_high.div(6), source_spread.div(6), source_count,
        (candidates - old).sign(),
    ), dim=1)
    features = torch.nan_to_num(features).clamp(-6, 6)
    result = {"features": features, "candidate_depth": candidates,
              "candidate_valid": valid, "old": old}
    if proposal_mode == "coarse_prior":
        result["proposal_eligible"] = valid.clone()
        result["observation_required"] = True
        result["context_known"] = context_known
    if view_consensus:
        result["view_consensus"] = leave_one_view_out_vote(source_z, source_known, valid)
    return result


