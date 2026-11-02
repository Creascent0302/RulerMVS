"""精确有向平面窗口匹配原型；局部平面不是占用标签或可见性证明。

每个参考窗口像素分别与过候选中心的平面求交，再精确投影到源相机。
可传入网络预测法线；本函数不读取标签、不更新标定，也不选择最终深度。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import pixel_grid, unproject, project, sample_maps


def _patch_cost(
    inputs,
    b,
    camera,
    uv,
    depth,
    normals,
    *,
    method,
    radius,
    dilation,
    temperature,
    texture_threshold,
    grazing_cosine,
    min_sources,
    agreement_cost,
    max_patch_depth,
):
    rgb = inputs["rgb"][b].float()
    K, T = inputs["K"][b].float(), inputs["T_cam_from_rig"][b].float()
    n, _, h, w = rgb.shape
    view_mask = inputs.get("view_mask")
    view_mask = (
        torch.ones(n, dtype=torch.bool, device=rgb.device)
        if view_mask is None
        else view_mask[b].bool()
    )
    image_mask = inputs.get("image_mask")
    image_mask = None if image_mask is None else image_mask[b].bool()
    finite_image = torch.isfinite(rgb).all(1)
    image_mask = finite_image if image_mask is None else image_mask & finite_image
    # Masked source RGB cannot leak through gradients at missing-image borders.
    rgb_mask = view_mask[:, None, None, None]
    if image_mask is not None:
        rgb_mask = rgb_mask & image_mask[:, None]
    rgb = torch.where(rgb_mask & torch.isfinite(rgb), rgb, 0.0)
    q, d = depth.shape
    p = normals.shape[1]
    offsets = (pixel_grid(2 * radius + 1, 2 * radius + 1, rgb.device) - radius).reshape(-1, 2)
    offsets = offsets * dilation
    window = uv[:, None] + offsets[None]
    k = len(offsets)
    center_index = k // 2
    finite_uv = torch.isfinite(uv).all(-1)
    safe_window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    center, patch_ray = unproject(
        safe_window, torch.zeros(q, k, device=rgb.device), K[camera], T[camera]
    )
    ref_ray = patch_ray[:, center_index]
    finite_normal = torch.isfinite(normals).all(-1)
    safe_normal = torch.nan_to_num(normals.float(), nan=0.0, posinf=0.0, neginf=0.0)
    normal_valid = finite_normal & (safe_normal.norm(dim=-1) > 1e-8)
    normal = F.normalize(safe_normal, dim=-1, eps=1e-8)
    denominator = torch.einsum("qpi,qki->qpk", normal, patch_ray)
    numerator = torch.einsum("qpi,qi->qp", normal, ref_ray)
    grazing = denominator.abs() <= grazing_cosine * patch_ray.norm(dim=-1)[:, None]
    plane_valid = normal_valid[..., None] & ~grazing
    safe_denominator = torch.where(plane_valid, denominator, torch.ones_like(denominator))
    ratio = numerator[..., None] / safe_denominator
    depth_valid = torch.isfinite(depth) & (depth > 0) & (depth < max_patch_depth)
    safe_depth = torch.where(depth_valid, depth.float(), torch.ones_like(depth.float()))
    patch_depth = safe_depth[:, :, None, None] * ratio[:, None]
    intersection_valid = (
        plane_valid[:, None]
        & torch.isfinite(patch_depth)
        & (patch_depth > 0)
        & (patch_depth < max_patch_depth)
    )
    safe_patch_depth = torch.where(intersection_valid, patch_depth, torch.ones_like(patch_depth))
    points = center[:, None, None] + safe_patch_depth[..., None] * patch_ray[:, None, None]
    projected, _, projection_valid, jacobian = project(
        points.reshape(-1, 3), K, T, (h, w), view_mask, image_mask
    )
    values = sample_maps(rgb, projected).reshape(n, q, d, p, k, 3)
    geometric = projection_valid.reshape(n, q, d, p, k).all(-1)
    geometric &= intersection_valid.all(-1)[None] & depth_valid[None, :, :, None]
    geometric &= finite_uv[None, :, None, None]
    geometric &= geometric[camera : camera + 1].clone()
    # Depth information comes from translation; a perfect pure-rotation warp
    # does not establish metric depth. Use the exact central projection J.
    center_j = jacobian.reshape(n, q, d, p, k, 2, 3)[..., center_index, :, :]
    derivative = torch.einsum("nqdpcj,qj->nqdpc", center_j, ref_ray).norm(dim=-1)
    source = torch.arange(n, device=rgb.device) != camera
    geometric &= source[:, None, None, None] & (derivative > 1e-5)
    reference = sample_maps(rgb[camera : camera + 1], safe_window.reshape(1, -1, 2))[0].reshape(
        q, k, 3
    )
    centered_ref = reference - reference.mean(-2, keepdim=True)
    centered_src = values - values.mean(-2, keepdim=True)
    ref_variance = centered_ref.square().mean((-2, -1))
    src_variance = centered_src.square().mean((-2, -1))
    informative = geometric & (ref_variance[None, :, None, None] > texture_threshold**2)
    informative &= src_variance > texture_threshold**2
    if method == "zncc":
        dot = (centered_src * centered_ref[None, :, None, None]).sum((-2, -1))
        norm = (
            (
                centered_src.square().sum((-2, -1))
                * centered_ref.square().sum((-2, -1))[None, :, None, None]
            )
            .clamp_min(1e-12)
            .sqrt()
        )
        per_view = (1 - dot / norm).clamp(0, 2)
    else:
        luma = rgb.new_tensor([0.299, 0.587, 0.114])
        gray_src = (values * luma).sum(-1)
        gray_ref = (reference * luma).sum(-1)
        src = torch.tanh((gray_src - gray_src[..., center_index : center_index + 1]) / temperature)
        ref = torch.tanh((gray_ref - gray_ref[..., center_index : center_index + 1]) / temperature)
        per_view = (src - ref[None, :, None, None]).abs().sum(-1) / (k - 1)
    per_view = torch.nan_to_num(per_view, nan=2.0, posinf=2.0, neginf=2.0).clamp(0, 2)
    count = informative.sum(0)
    source_agreement = (informative & (per_view <= agreement_cost)).sum(0)
    # Never use a single best camera from a larger valid set: every usable
    # source contributes, and multiple sources must agree on the same Q,D,P.
    raw_cost = torch.where(informative, per_view, 0.0).sum(0) / count.clamp_min(1)
    valid = (count >= min_sources) & (source_agreement >= min_sources)
    cost = torch.where(valid, raw_cost, raw_cost * 0 + 2.0)
    return dict(
        cost=cost,
        raw_cost=torch.where(count > 0, raw_cost, raw_cost * 0 + 2.0),
        valid=valid,
        source_count=count,
        support=torch.where(valid, source_agreement + 1, 0),
        per_view_cost=per_view.permute(1, 2, 3, 0),
        per_view_valid=informative.permute(1, 2, 3, 0),
        geometric_valid=geometric.permute(1, 2, 3, 0),
        texture=ref_variance.clamp_min(1e-12).sqrt()[:, None, None].expand(q, d, p),
    )


def oriented_patch_cost(
    inputs,
    b,
    refs,
    uv,
    center_depth,
    normals,
    *,
    method="census",
    patch_radius=2,
    patch_dilation=1,
    include_fronto=True,
    query_chunk=64,
    depth_chunk=8,
    census_temperature=0.02,
    texture_threshold=0.006,
    grazing_cosine=0.025,
    min_sources=None,
    agreement_cost=0.6,
    max_patch_depth=1e4,
):
    """可微的候选代价；``cost/valid/support`` 为 [Q,D,P]。

    normals 为 rig 坐标 [Q,3] 或 [Q,P,3]，其正负号不影响交点。
    include_fronto=True 在最后追加参考相机光轴平面（独立 fallback 假设）。
    per_view_cost/valid 为 [Q,D,P,N]。只在相同深度和法线假设上聚合源视图，
    不逐相机挑不同假设凑支持。默认至少两台 source 共识；总共只有两台有效
    相机时允许一台 source。valid 不是遮挡真值或表面存在证明。
    patch_dilation 只放大像素采样间距；样本数仍为 (2*patch_radius+1)²。
    支持 CPU/CUDA 与 RGB/深度/法线的一阶梯度；几何以 FP32 计算。
    """
    if method not in ("census", "zncc"):
        raise ValueError("method must be census or zncc")
    if min(patch_radius, patch_dilation, query_chunk, depth_chunk) < 1 or census_temperature <= 0:
        raise ValueError("positive radius, dilation, chunk sizes and census temperature required")
    if (
        texture_threshold < 0
        or not 0 < grazing_cosine < 1
        or not 0 < agreement_cost <= 2
        or max_patch_depth <= 0
    ):
        raise ValueError("invalid texture, grazing, agreement or depth options")
    if (
        refs.ndim != 1
        or uv.shape != (len(refs), 2)
        or center_depth.ndim != 2
        or center_depth.shape[0] != len(refs)
    ):
        raise ValueError("refs/uv/center_depth must be [Q]/[Q,2]/[Q,D]")
    if normals.ndim == 2:
        normals = normals[:, None]
    if (
        normals.ndim != 3
        or normals.shape[0] != len(refs)
        or normals.shape[-1] != 3
        or normals.shape[1] < 1
    ):
        raise ValueError("normals must be [Q,3] or [Q,P,3]")
    rgb = inputs["rgb"][b]
    n = rgb.shape[0]
    if refs.dtype != torch.long or ((refs < 0) | (refs >= n)).any():
        raise ValueError("refs must contain valid integer camera indices")
    if min_sources is None:
        active = n if "view_mask" not in inputs else int(inputs["view_mask"][b].sum())
        min_sources = min(2, max(active - 1, 1))
    if not isinstance(min_sources, int) or min_sources < 1:
        raise ValueError("min_sources must be a positive integer")
    if include_fronto:
        fronto = inputs["T_cam_from_rig"][b, refs, 2, :3].float()
        normals = torch.cat((normals.float(), fronto[:, None]), 1)
    q, d, p = len(refs), center_depth.shape[1], normals.shape[1]
    if d < 1:
        raise ValueError("at least one center depth candidate is required")
    if not q:
        result = {
            key: rgb.new_empty((0, d, p), dtype=torch.float32)
            for key in ("cost", "raw_cost", "texture")
        }
        result.update(
            valid=torch.empty(0, d, p, dtype=torch.bool, device=rgb.device),
            support=torch.empty(0, d, p, dtype=torch.long, device=rgb.device),
            source_count=torch.empty(0, d, p, dtype=torch.long, device=rgb.device),
        )
        result.update(
            {
                key: torch.empty(
                    0,
                    d,
                    p,
                    n,
                    device=rgb.device,
                    dtype=torch.float32 if key == "per_view_cost" else torch.bool,
                )
                for key in ("per_view_cost", "per_view_valid", "geometric_valid")
            }
        )
    else:
        pieces, order = [], []
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            for camera in range(n):
                ids = (refs == camera).nonzero().flatten()
                for chunk in ids.split(query_chunk):
                    if not len(chunk):
                        continue
                    depth_pieces = [
                        _patch_cost(
                            inputs,
                            b,
                            camera,
                            uv[chunk].float(),
                            depth.float(),
                            normals[chunk].float(),
                            method=method,
                            radius=patch_radius,
                            dilation=patch_dilation,
                            temperature=census_temperature,
                            texture_threshold=texture_threshold,
                            grazing_cosine=grazing_cosine,
                            min_sources=min_sources,
                            agreement_cost=agreement_cost,
                            max_patch_depth=max_patch_depth,
                        )
                        for depth in center_depth[chunk].split(depth_chunk, 1)
                    ]
                    pieces.append(
                        {
                            key: torch.cat([part[key] for part in depth_pieces], 1)
                            for key in depth_pieces[0]
                        }
                    )
                    order.append(chunk)
        inverse = torch.cat(order).argsort()
        result = {key: torch.cat([part[key] for part in pieces], 0)[inverse] for key in pieces[0]}
    result["includes_fronto"] = include_fronto
    result["min_sources"] = min_sources
    result["normal_hypothesis_count"] = p
    return result
