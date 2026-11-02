"""从父表面附近产生子查询；保留独立边缘候选以发现遗漏物体。"""

import torch

from .geometry import reference_rays, sample_maps
from .query_selection import ray_roi_mask


@torch.no_grad()
def child_queries(inputs, batch, previous, budget, stride, edges=None, roi=None, near=2, far=80):
    """在父锚点周围取候选并去重，返回像素、相机及明确的父索引。

    父子关系由生成过程确定，避免细层重新查找稀疏图像桶而丢失父信息。
    图像边缘只增加采样优先级，候选还必须经过后续多视图匹配。
    """
    rgb = inputs["rgb"][batch]
    empty = dict(uv=rgb.new_empty(0, 2), refs=torch.empty(0, device=rgb.device, dtype=torch.long))
    empty["parent_id"] = empty["refs"].clone()
    if previous is None or budget <= 0:
        return empty
    cameras, _, height, width = rgb.shape
    parent_uv, parent_refs = previous["uv"], previous["refs"]
    active = torch.isfinite(parent_uv).all(-1)
    active &= (parent_refs >= 0) & (parent_refs < cameras)
    active &= (parent_uv[:, 0] >= 0) & (parent_uv[:, 0] <= width - 1)
    active &= (parent_uv[:, 1] >= 0) & (parent_uv[:, 1] <= height - 1)
    parent_ids = active.nonzero().flatten()
    if "view_mask" in inputs:
        parent_ids = parent_ids[inputs["view_mask"][batch, parent_refs[parent_ids]]]
    if "image_mask" in inputs:
        old_xy = parent_uv[parent_ids].round().long()
        visible = inputs["image_mask"][batch, parent_refs[parent_ids], old_xy[:, 1], old_xy[:, 0]]
        parent_ids = parent_ids[visible]
    if not len(parent_ids):
        return empty
    parent_uv = parent_uv[parent_ids]
    parent_refs = parent_refs[parent_ids]
    count = len(parent_ids)
    radius = max(stride // 2, 1)
    offsets = rgb.new_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * radius
    uv = parent_uv[:, None] + offsets[None]
    if edges is not None:
        sampled = sample_maps(edges["direction"], parent_uv[None].expand(cameras, -1, -1))
        direction = sampled[parent_refs, torch.arange(count, device=rgb.device)]
        side = direction * max(1, stride // 4)
        uv = torch.cat((uv, parent_uv[:, None] + torch.stack((-side, side), 1)), 1)
    children = uv.shape[1]
    uv = uv.round().reshape(-1, 2)
    parents = parent_ids.repeat_interleave(children)
    refs = previous["refs"][parents]
    usable = (uv[:, 0] >= 2) & (uv[:, 0] <= width - 3)
    usable &= (uv[:, 1] >= 2) & (uv[:, 1] <= height - 3)
    if "view_mask" in inputs:
        usable &= inputs["view_mask"][batch, refs]
    x = uv[:, 0].long().clamp(0, width - 1)
    y = uv[:, 1].long().clamp(0, height - 1)
    if "image_mask" in inputs:
        usable &= inputs["image_mask"][batch, refs, y, x]
    if roi is not None:
        usable &= ray_roi_mask(inputs, batch, refs, uv, roi, near, far)

    # 用整数像素键去掉父锚点和重复子候选；总空间为 N*H*W，而非 Q*Q。
    key = (refs * height + y) * width + x
    occupied = torch.zeros(cameras * height * width, dtype=torch.bool, device=rgb.device)
    old_xy = parent_uv.round().long()
    old_key = (parent_refs * height + old_xy[:, 1]) * width + old_xy[:, 0]
    occupied[old_key] = True
    usable &= ~occupied[key]
    first = torch.full_like(occupied, len(key), dtype=torch.long)
    positions = torch.arange(len(key), device=rgb.device)
    first.scatter_reduce_(0, key[usable], positions[usable], reduce="amin", include_self=True)
    usable &= first[key] == positions

    score = torch.ones(len(key), device=rgb.device)
    if edges is not None:
        score += 2 * edges["strength"][refs, 0, y, x]
    # 同一父点的次选子点略降权，避免预算全被少数强边缘占据。
    score += 0.25 * previous["valid"][parents].float()
    order_in_parent = score.reshape(count, children).argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order_in_parent)
    rank.scatter_(1, order_in_parent, torch.arange(children, device=rgb.device).expand(count, -1))
    score -= rank.flatten().float() * 0.2
    order = score.masked_fill(~usable, -torch.inf).argsort(descending=True, stable=True)
    order = order[usable[order]]
    active_views = refs[order].unique()
    selected = []
    for number, camera in enumerate(active_views):
        quota = budget // len(active_views) + int(number < budget % len(active_views))
        selected.append(order[refs[order] == camera][:quota])
    selected = torch.cat(selected) if selected else order[:0]
    # 某台相机的父区域不足时，把剩余预算交给其他真实候选，不复制点补数。
    chosen = torch.zeros(len(key), dtype=torch.bool, device=rgb.device)
    chosen[selected] = True
    selected = torch.cat((selected, order[~chosen[order]][: budget - len(selected)]))
    return dict(uv=uv[selected], refs=refs[selected], parent_id=parents[selected])


def inherit_plane_depth(inputs, batch, uv, refs, parent_id, previous, near, far):
    """把父局部平面与子射线求交；掠射、过远或无效平面退回父光轴深度。

    计算使用同一 rig 坐标系；子射线有自己的方向，不能直接复制父三维点。
    只做初始化，新的深度突变仍可通过独立全局候选改正。
    """
    centers, rays, _ = reference_rays(
        torch.zeros(len(uv), 3, device=uv.device),
        refs,
        uv,
        inputs["K"][batch],
        inputs["T_cam_from_rig"][batch],
    )
    normal = previous["normals"][parent_id]
    point = previous["points"][parent_id]
    finite_plane = torch.isfinite(normal).all(-1) & torch.isfinite(point).all(-1)
    normal = torch.where(torch.isfinite(normal), normal, 0.0)
    point = torch.where(torch.isfinite(point), point, 0.0)
    normal_length = normal.norm(dim=-1, keepdim=True)
    normal = normal / normal_length.clamp_min(1e-8)
    denominator = (normal * rays).sum(-1)
    safe = torch.where(denominator.abs() > 1e-3, denominator, torch.ones_like(denominator))
    intersection = ((point - centers) * normal).sum(-1) / safe
    fallback = previous["depth"][parent_id].nan_to_num(nan=near, posinf=far, neginf=near)
    reasonable = previous["valid"][parent_id] & finite_plane & (normal_length[:, 0] > 1e-8)
    reasonable &= denominator.abs() > 0.1
    reasonable &= torch.isfinite(intersection) & (intersection >= near) & (intersection <= far)
    reasonable &= (intersection - fallback).abs() <= (0.2 * fallback + 0.1)
    return torch.where(reasonable, intersection, fallback).clamp(near, far)
