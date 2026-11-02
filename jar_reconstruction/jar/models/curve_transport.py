"""邻近图像查询间传递匹配曲线；不平均深度、不增加观测视角计数。

这是可单独消融的研究模块。邻接使用分格索引，图存储量 O(QK)，曲线消息
存储量 O(QKD)，避免全场景 query 两两 attention。沿物理逆深度插值，
支持非均匀深度候选；候选 logit 不是连续空间概率密度。
"""

import torch
from torch.nn import functional as F


def image_neighbors(uv, refs, radius=32.0, neighbors=8, cell_capacity=16):
    """同相机半径邻域；每格最多检查 cell_capacity 个坐标排序后的候选。"""
    count = len(uv)
    if count == 0:
        empty = refs.new_empty((0, neighbors))
        return empty, empty.bool()
    cells = torch.floor(uv.detach() / radius).long()
    lower = cells.amin(0) - 1
    cells = cells - lower
    shape = cells.amax(0) + 2
    keys = (refs * shape[1] + cells[:, 1]) * shape[0] + cells[:, 0]
    # 相同格子内按图像坐标排序，不依赖模型将 query 分成多少个 chunk。
    order = torch.argsort(uv[:, 0], stable=True)
    order = order[torch.argsort(uv[order, 1], stable=True)]
    order = order[torch.argsort(keys[order], stable=True)]
    ordered = keys[order]
    dy, dx = torch.meshgrid(
        torch.arange(-1, 2, device=uv.device),
        torch.arange(-1, 2, device=uv.device),
        indexing="ij",
    )
    search = keys[:, None] + dy.flatten()[None] * shape[0] + dx.flatten()[None]
    start = torch.searchsorted(ordered, search.contiguous())
    offset = torch.arange(cell_capacity, device=uv.device)
    positions = start[..., None] + offset
    bounded = positions.clamp_max(count - 1)
    ids = order[bounded].flatten(1)
    matches = (ordered[bounded] == search[..., None]) & (positions < count)
    distance = (uv[:, None] - uv[ids]).square().sum(-1)
    matches = matches.flatten(1) & (refs[:, None] == refs[ids])
    matches &= distance > 1e-8
    matches &= distance <= radius**2
    distances, selected = distance.masked_fill(~matches, torch.inf).topk(
        min(neighbors, distance.shape[1]), dim=1, largest=False
    )
    return ids.gather(1, selected), torch.isfinite(distances)


def interpolate_curves(inverse_lattice, values, valid, inverse_query):
    """values[...,D] 与 inverse_query[...,M]，不跨无效深度间隙补证据。"""
    order = inverse_lattice.argsort()
    lattice = inverse_lattice[order].contiguous()
    values, valid = values[..., order], valid[..., order]
    right = torch.searchsorted(lattice, inverse_query.contiguous()).clamp(
        1, len(lattice) - 1
    )
    left = right - 1
    fraction = (inverse_query - lattice[left]) / (
        lattice[right] - lattice[left]
    ).clamp_min(1e-8)
    lo, hi = values.gather(-1, left), values.gather(-1, right)
    good = (valid.gather(-1, left) | (fraction >= 1 - 1e-6))
    good &= valid.gather(-1, right) | (fraction <= 1e-6)
    good &= (inverse_query >= lattice[0]) & (inverse_query <= lattice[-1])
    result = torch.lerp(lo, hi, fraction.clamp(0, 1))
    return result, good


def curve_messages(
    uv,
    refs,
    candidates,
    logits,
    valid,
    reference_features,
    reference_rgb,
    rays=None,
    normals=None,
    radius=32.0,
    neighbors=8,
    global_start=9,
):
    """固定一次消息，保留多峰；可选平面法线均在同一 rig 坐标下。

    每个节点自己的候选不变。邻居的相对 log-likelihood 在预测平面与邻居射线
    的交点处读取，禁止使用 GT 法线/深度。弱颜色或特征一致性会降低消息权重。
    不输出新的 validity，图像邻居不构成独立相机证据。
    """
    ids, neighbor_valid = image_neighbors(uv, refs, radius, neighbors)
    if len(uv) == 0:
        return logits.clone(), logits.new_empty((0, neighbors))
    feature = F.normalize(reference_features.float(), dim=-1, eps=1e-6)
    feature_difference = (1 - (feature[:, None] * feature[ids]).sum(-1)).clamp_min(0)
    color_difference = (reference_rgb[:, None] - reference_rgb[ids]).abs().mean(-1)
    distance = (uv[:, None] - uv[ids]).square().sum(-1) / radius**2
    affinity = torch.exp(-feature_difference / 0.2 - color_difference / 0.1 - distance)
    affinity = affinity * neighbor_valid
    global_logits, global_valid = logits[:, global_start:], valid[:, global_start:]
    lattice = candidates[0, global_start:].reciprocal()
    if not torch.allclose(candidates[:, global_start:], candidates[:1, global_start:]):
        raise ValueError("当前消息模块要求共享全局深度格；局部候选可不同。")
    relative = global_logits - global_logits.masked_fill(~global_valid, -1e4).amax(
        -1, keepdim=True
    )
    relative = relative.clamp(-8, 0)
    inverse = candidates.reciprocal()[:, None].expand(-1, ids.shape[1], -1)
    if normals is not None:
        if rays is None:
            raise ValueError("平面曲线传递需要光轴深度参数化的 rig 射线。")
        numerator = (normals[:, None] * rays[ids]).sum(-1)
        denominator = (normals * rays).sum(-1)
        ratio = numerator / torch.where(
            denominator.abs() > 1e-5, denominator, torch.ones_like(denominator)
        )[:, None]
        affinity = affinity * (denominator.abs() > 1e-5)[:, None] * (ratio > 0)
        inverse = inverse * ratio[..., None]
    sampled, known = interpolate_curves(lattice, relative[ids], global_valid[ids], inverse)
    # 同一邻居必须能评价本点所有有效竞争候选。否则负代价会使没有邻居证据
    # 的候选反而占优；禁止让每个候选自行选择对它有利的邻居集合。
    common = (known | ~valid[:, None]).all(-1)
    affinity = affinity * common
    weights = affinity[..., None] * known
    message = (sampled * weights).sum(1) / weights.sum(1).clamp_min(1)
    return message, affinity
