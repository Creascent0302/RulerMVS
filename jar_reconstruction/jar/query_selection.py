"""输入图像驱动的候选位置策略；候选不是三维表面证据。

本模块只分配新增 query 的参考相机和整数像素，不修改既有 query。
phase_grid 用相机间错开的图像格点减轻规则采样相位遗漏；
coverage_edge_pairs 将均匀覆盖与局部边缘两侧的候选组合。它们均不读取
深度真值或粗深度预测，也不宣称已通过跨视图几何验证。
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .geometry import pixel_grid, spatial_gradients, unproject, sample_maps


COVERAGE, EDGE_SIDE, FALLBACK, PHASE_GRID = 0, 1, 2, 3


@torch.no_grad()
def ray_roi_mask(inputs, b, ref_index, ref_uv, roi, near=0.0, far=float("inf")):
    """参考像素射线在光轴深度区间内能否与 rig ROI 相交。

    这是候选位置筛选，不是表面/空闲判断。只读标定、输入有效掩码与 ROI，
    不读取 RGB、预测深度或标签。不修改图像的 source-view image_mask。
    slab 交集保留边界相切射线，避免浮点边界误差造成激进剔除。
    ref_uv 可以是连续像素；部分有效图像要求其双线性采样邻域有效。
    """
    if ref_index.ndim != 1 or ref_uv.shape != (len(ref_index), 2):
        raise ValueError("ref_index and ref_uv must be [Q] and [Q,2]")
    if near is None:
        near = 0.0
    if far is None:
        far = float("inf")
    if not math.isfinite(float(near)) or near < 0 or math.isnan(float(far)) or far <= near:
        raise ValueError("require finite near >= 0 and far > near")
    device = ref_uv.device
    bounds = torch.as_tensor(roi, dtype=torch.float32, device=device)
    if (
        bounds.shape != (3, 2)
        or not torch.isfinite(bounds).all()
        or not (bounds[:, 1] > bounds[:, 0]).all()
    ):
        raise ValueError("roi must be finite ordered [3,2] bounds")
    n, _, h, w = inputs["rgb"][b].shape
    result = torch.zeros(len(ref_index), dtype=torch.bool, device=device)
    # Per-view batching bounds temporary ray arrays even for full-resolution masks.
    for camera in range(n):
        if "view_mask" in inputs and not bool(inputs["view_mask"][b, camera]):
            continue
        ids = (ref_index == camera).nonzero().flatten()
        for chunk in ids.split(65536):
            if not len(chunk):
                continue
            uv = ref_uv[chunk].float()
            valid = torch.isfinite(uv).all(-1)
            valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
            if "image_mask" in inputs:
                mask = inputs["image_mask"][b, camera][None, None].float()
                valid &= sample_maps(mask, uv[None])[0, :, 0] > 0.999
            with torch.autocast(device_type=device.type, enabled=False):
                center, direction = unproject(
                    uv,
                    torch.zeros(len(uv), device=device),
                    inputs["K"][b, camera],
                    inputs["T_cam_from_rig"][b, camera],
                )
                finite = torch.isfinite(center).all(-1) & torch.isfinite(direction).all(-1)
                parallel = direction.abs() < 1e-12
                outside_parallel = parallel & ((center < bounds[:, 0]) | (center > bounds[:, 1]))
                safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
                a = (bounds[:, 0] - center) / safe_direction
                z = (bounds[:, 1] - center) / safe_direction
                slab_near = torch.minimum(a, z).masked_fill(parallel, -float("inf"))
                slab_far = torch.maximum(a, z).masked_fill(parallel, float("inf"))
                entry = slab_near.max(-1).values.clamp_min(float(near))
                exit_ = slab_far.min(-1).values.clamp_max(float(far))
                hit = finite & ~outside_parallel.any(-1) & (exit_ >= entry) & (exit_ > 0)
                result[chunk] = valid & hit
    return result


def _ordered_unique(values):
    """保留第一次出现的整数键；无全局随机数、无逐点 Python 循环。"""
    if not len(values):
        return values
    unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
    first = torch.full((len(unique),), len(values), device=values.device, dtype=torch.long)
    first.scatter_reduce_(
        0,
        inverse,
        torch.arange(len(values), device=values.device),
        reduce="amin",
        include_self=True,
    )
    return values[first.sort().values]


def _halton(index, base):
    # Integer digit expansion is deterministic and consumes no RNG state.
    value = torch.zeros_like(index, dtype=torch.float64)
    remaining = index.clone()
    factor = 1.0
    for _ in range(max(1, int(math.ceil(math.log(max(int(index.max()) + 1, 2), base))))):
        factor /= base
        value += (remaining % base) * factor
        remaining = torch.div(remaining, base, rounding_mode="floor")
    return value


def _coverage_order(h, w, camera, quota, available, margin):
    # Bound the Halton pool by image area and eight times the requested count.
    pool = min(h * w, max(64, quota * 8))
    index = torch.arange(1, pool + 1, device=available.device) + camera * max(quota, 1)
    x = (margin + _halton(index, 2) * max(w - 2 * margin, 1)).floor().long().clamp(0, w - 1)
    y = (margin + _halton(index, 3) * max(h - 2 * margin, 1)).floor().long().clamp(0, h - 1)
    keys = _ordered_unique(y * w + x)
    return keys[available.flatten()[keys]]


def _edge_pairs(gx, gy, available, cell_size, margin):
    h, w = available.shape
    score = (gx.square() + gy.square()).sqrt().masked_fill(~available, -float("inf"))
    ch, cw = math.ceil(h / cell_size), math.ceil(w / cell_size)
    padded = F.pad(score, (0, cw * cell_size - w, 0, ch * cell_size - h), value=-float("inf"))
    windows = (
        padded.unfold(0, cell_size, cell_size)
        .unfold(1, cell_size, cell_size)
        .reshape(-1, cell_size**2)
    )
    best, local = windows.max(-1)
    cell = pixel_grid(ch, cw, device=available.device, stride=cell_size).reshape(-1, 2).long()
    x = cell[:, 0] + local % cell_size
    y = cell[:, 1] + torch.div(local, cell_size, rounding_mode="floor")
    usable = torch.isfinite(best) & (best > 1e-8) & (x < w) & (y < h)
    x, y, best = x[usable], y[usable], best[usable]
    order = torch.argsort(best, descending=True, stable=True)
    x, y = x[order], y[order]
    vertical = gy[y, x].abs() > gx[y, x].abs()
    dx, dy = (~vertical).long(), vertical.long()
    xs = torch.stack((x - dx, x + dx), -1).clamp(margin, max(margin, w - margin - 1))
    ys = torch.stack((y - dy, y + dy), -1).clamp(margin, max(margin, h - margin - 1))
    keys = _ordered_unique((ys * w + xs).flatten())
    return keys[available.flatten()[keys]]


@torch.no_grad()
def select_query_candidates(
    inputs,
    b,
    budget,
    old_refs=None,
    old_uv=None,
    policy="coverage_edge_pairs",
    stride=4,
    coverage_fraction=0.5,
    edge_cell_size=8,
    roi=None,
    near=None,
    far=None,
    edge_level=None,
):
    """返回新增候选 ``ref_index/ref_uv/proposal_kind/diagnostics``。

    budget 仅是新增预算，旧位置只用于精确去重，不放入返回结果。每个可用
    相机分配相等名额（余数按相机顺序）；不足的相机不以重复/无效点填补。
    输出像素是原图整数中心，以 float32 存储。相机/像素掩码必须有效。
    梯度方案要求完整 3×3 梯度邻域，且避免原图边缘的两像素区域。
    选择没有可训练参数，固定输入与参数产生固定输出。给定 roi 时，先排除
    在 [near,far] 光轴深度范围不可能穿过 ROI 的候选；不裁剪预测深度。
    """
    if policy not in ("phase_grid", "coverage_edge_pairs"):
        raise ValueError("policy must be phase_grid or coverage_edge_pairs")
    if not isinstance(budget, int) or budget < 0 or stride < 1 or edge_cell_size < 2:
        raise ValueError("budget must be a nonnegative integer; stride/cell size must be positive")
    if not 0 <= coverage_fraction <= 1:
        raise ValueError("coverage_fraction must be in [0,1]")
    if (old_refs is None) != (old_uv is None):
        raise ValueError("old_refs and old_uv must be provided together")
    rgb = inputs["rgb"][b]
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must be [B,N,3,H,W]")
    n, _, h, w = rgb.shape
    device = rgb.device
    view_mask = inputs.get("view_mask")
    view_mask = (
        torch.ones(n, dtype=torch.bool, device=device) if view_mask is None else view_mask[b].bool()
    )
    image_mask = inputs.get("image_mask")
    image_mask = (
        torch.ones(n, h, w, dtype=torch.bool, device=device)
        if image_mask is None
        else image_mask[b].bool()
    )
    if old_refs is not None and (old_refs.ndim != 1 or old_uv.shape != (len(old_refs), 2)):
        raise ValueError("old refs and coordinates must have matching shapes")
    views = view_mask.nonzero().flatten().tolist()
    margin = min(2, max(0, (min(h, w) - 1) // 2))
    rgb_safe = torch.where(image_mask[:, None] & view_mask[:, None, None, None], rgb.float(), 0.0)
    gx, gy = spatial_gradients(rgb_safe.mean(1, keepdim=True))
    gx, gy = gx[:, 0], gy[:, 0]
    if edge_level is not None:
        edge_gradient = edge_level["direction"] * edge_level["strength"]
        gx, gy = edge_gradient[:, 0], edge_gradient[:, 1]
    missing = (~image_mask).float()[:, None]
    gradient_valid = (
        F.max_pool2d(F.pad(missing, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[:, 0] == 0
    )
    if edge_level is not None:
        gradient_valid &= edge_level["valid"][:, 0]
    collected_refs, collected_uv, kinds = [], [], []
    available_counts = [0] * n
    roi_counts = [0] * n
    for number, camera in enumerate(views):
        quota = budget // len(views) + int(number < budget % len(views))
        available = image_mask[camera].clone()
        if roi is not None:
            all_uv = pixel_grid(h, w, device=device).reshape(-1, 2)
            all_refs = torch.full((len(all_uv),), camera, dtype=torch.long, device=device)
            roi_valid = ray_roi_mask(inputs, b, all_refs, all_uv, roi, near, far).reshape(h, w)
            available &= roi_valid
            roi_counts[camera] = int(roi_valid.sum())
        if policy == "coverage_edge_pairs":
            available &= gradient_valid[camera]
            if margin:
                available[:margin] = False
                available[-margin:] = False
                available[:, :margin] = False
                available[:, -margin:] = False
        if old_refs is not None and len(old_refs):
            old = old_uv[old_refs == camera]
            finite = torch.isfinite(old).all(-1)
            old = old[finite]
            integer = old.long()
            exact = (old == integer).all(-1)
            exact &= (
                (integer[:, 0] >= 0)
                & (integer[:, 0] < w)
                & (integer[:, 1] >= 0)
                & (integer[:, 1] < h)
            )
            integer = integer[exact]
            available[integer[:, 1], integer[:, 0]] = False
        if policy == "phase_grid":
            grid = (
                pixel_grid(
                    math.ceil(h / stride), math.ceil(w / stride), device=device, stride=stride
                )
                .reshape(-1, 2)
                .long()
            )
            offset = grid.new_tensor(
                [(camera % 2) * (stride // 2), ((camera // 2) % 2) * (stride // 2)]
            )
            grid = grid + offset
            inside = (grid[:, 0] < w) & (grid[:, 1] < h)
            grid = grid[inside]
            grid = grid[available[grid[:, 1], grid[:, 0]]]
            available_counts[camera] = len(grid)
            count = min(quota, len(grid))
            positions = (
                grid[torch.linspace(0, len(grid) - 1, count, device=device).long()]
                if count
                else grid[:0]
            )
            labels = torch.full((count,), PHASE_GRID, device=device, dtype=torch.long)
        else:
            available_counts[camera] = int(available.sum())
            quota = min(quota, available_counts[camera])
            if not quota:
                continue
            coverage = _coverage_order(h, w, camera, max(quota, 1), available, margin)
            cover_count = min(len(coverage), int(math.ceil(quota * coverage_fraction)))
            initial = coverage[:cover_count]
            chosen = torch.zeros(h * w, dtype=torch.bool, device=device)
            chosen[initial] = True
            pairs = _edge_pairs(gx[camera], gy[camera], available, edge_cell_size, margin)
            edge = pairs[~chosen[pairs]][: quota - len(initial)]
            chosen[edge] = True
            remaining = quota - int(chosen.sum())
            extra = coverage[~chosen[coverage]][:remaining]
            chosen[extra] = True
            remaining = quota - int(chosen.sum())
            fallback = available.flatten().nonzero().flatten()
            fallback = fallback[~chosen[fallback]]
            if remaining:
                fallback = fallback[
                    torch.linspace(
                        0, len(fallback) - 1, min(remaining, len(fallback)), device=device
                    ).long()
                ]
            else:
                fallback = fallback[:0]
            selected = torch.cat((initial, edge, extra, fallback))
            positions = torch.stack(
                (selected % w, torch.div(selected, w, rounding_mode="floor")), -1
            )
            labels = torch.cat(
                (
                    torch.full_like(initial, COVERAGE),
                    torch.full_like(edge, EDGE_SIDE),
                    torch.full_like(extra, COVERAGE),
                    torch.full_like(fallback, FALLBACK),
                )
            )
        collected_refs.append(
            torch.full((len(positions),), camera, device=device, dtype=torch.long)
        )
        collected_uv.append(positions.float())
        kinds.append(labels)
    refs = (
        torch.cat(collected_refs)
        if collected_refs
        else torch.empty(0, dtype=torch.long, device=device)
    )
    uv = torch.cat(collected_uv) if collected_uv else torch.empty(0, 2, device=device)
    kind = torch.cat(kinds) if kinds else torch.empty(0, dtype=torch.long, device=device)
    return dict(
        ref_index=refs,
        ref_uv=uv,
        proposal_kind=kind,
        diagnostics=dict(
            policy=policy,
            requested_new_budget=budget,
            actual_new_count=len(refs),
            per_camera_selected=torch.bincount(refs, minlength=n).tolist(),
            per_camera_available=available_counts,
            uses_ground_truth=False,
            roi_ray_filter=roi is not None,
            per_camera_roi_ray_count=roi_counts,
            uses_predicted_depth=False,
            contains_3d_surface_confirmation=False,
        ),
    )
