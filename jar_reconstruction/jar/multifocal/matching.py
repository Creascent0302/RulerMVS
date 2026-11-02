"""多焦距采样：参考层固定，源层按真实投影尺度选择，RGB窗口共享三维平面。

所有像素坐标采用 geometry.py 的整数中心约定。本模块不使用标签，不判定
全局深度唯一性，也不把低纹理、同光心或窗口越界视图算成测量证据。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from jar.geometry import camera_centers, pixel_grid, project, sample_maps


def _options(cfg):
    return cfg.get("model", cfg).get("multifocal", {})


def tangent_basis(normal):
    """[... ,3]单位法线 -> [...,3,2]正交切向基；调用前已处理无效法线。"""
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    axis = F.one_hot(normal.abs().argmin(-1), 3).to(normal)
    first = F.normalize(torch.linalg.cross(normal, axis), dim=-1, eps=1e-8)
    return torch.stack((first, torch.linalg.cross(normal, first)), -1)


def _normal_bank(inputs, b, refs, normals, include_fronto=False):
    fronto = inputs["T_cam_from_rig"][b, refs, 2, :3].float()
    if normals is None:
        return fronto[:, None]
    if normals.ndim == 2:
        normals = normals[:, None]
    good = torch.isfinite(normals).all(-1) & (torch.nan_to_num(normals).norm(dim=-1) > 1e-8)
    normal = torch.where(good[..., None], torch.nan_to_num(normals.float()), fronto[:, None])
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    return torch.cat((normal, fronto[:, None]), 1) if include_fronto else normal


def _projection_scales(jacobian, refs, normals):
    """J[N,Q,D,2,3], normals[Q,P,3] -> ref-to-source metric[N,Q,D,P,2,2]."""
    with torch.autocast(device_type=jacobian.device.type, enabled=False):
        basis = tangent_basis(normals)
        g = torch.einsum("nqdij,qpjk->nqdpik", jacobian, basis)
        qi = torch.arange(len(refs), device=refs.device)
        ref_g = g[refs, qi]
        det = ref_g[..., 0, 0] * ref_g[..., 1, 1] - ref_g[..., 0, 1] * ref_g[..., 1, 0]
        safe = torch.where(det.abs() > 1e-8, det, torch.ones_like(det))
        inv = (
            torch.stack(
                (ref_g[..., 1, 1], -ref_g[..., 0, 1], -ref_g[..., 1, 0], ref_g[..., 0, 0]), -1
            ).reshape_as(ref_g)
            / safe[..., None, None]
        )
        transform = g @ inv[None]
        gram = transform.transpose(-1, -2) @ transform
        trace = gram[..., 0, 0] + gram[..., 1, 1]
        discriminant = (
            ((gram[..., 0, 0] - gram[..., 1, 1]).square() + 4 * gram[..., 0, 1].square())
            .clamp_min(1e-12)
            .sqrt()
        )
        minimum = ((trace - discriminant) * 0.5).clamp_min(1e-12).sqrt()
        rms = (trace * 0.5).clamp_min(1e-12).sqrt()
        return rms, minimum, det.abs() > 1e-8


def _paired_projection(points, K, T, image_hw, view_mask, image_mask):
    """每台源相机只投影自己的[N,...,3]点；不生成N²份交叉积。"""
    n = len(K)
    shape = points.shape[1:-1]
    flat = points.reshape(n, -1, 3)
    camera = torch.einsum("nij,nqj->nqi", T[:, :3, :3], flat) + T[:, None, :3, 3]
    z = camera[..., 2]
    hom = torch.einsum("nij,nqj->nqi", K, camera)
    uv = hom[..., :2] / z.clamp_min(1e-5)[..., None]
    h, w = image_hw
    valid = torch.isfinite(camera).all(-1) & torch.isfinite(uv).all(-1) & (z > 1e-5)
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    valid &= view_mask[:, None]
    uv = torch.nan_to_num(uv, nan=0.0, posinf=1e6, neginf=-1e6)
    valid &= sample_maps(image_mask[:, None].float(), uv)[..., 0] > 0.999
    return uv.reshape(n, *shape, 2), valid.reshape(n, *shape)


def _patch_group(
    inputs,
    b,
    camera,
    uv,
    depth,
    normals,
    *,
    radius,
    antialias,
    aa_samples,
    texture_threshold,
    temperature,
    grazing_cosine,
    method,
    min_resolution_px,
):
    rgb = inputs["rgb"][b].float()
    K, T = inputs["K"][b].float(), inputs["T_cam_from_rig"][b].float()
    n, _, h, w = rgb.shape
    device = rgb.device
    q, d = depth.shape
    p = normals.shape[1]
    roles = inputs.get(
        "camera_roles", torch.zeros(inputs["rgb"].shape[:2], device=device, dtype=torch.long)
    )[b]
    view_mask = inputs.get(
        "view_mask", torch.ones(inputs["rgb"].shape[:2], device=device, dtype=torch.bool)
    )[b].bool()
    image_mask = torch.isfinite(rgb).all(1)
    if "image_mask" in inputs:
        image_mask = image_mask & inputs["image_mask"][b].bool()
    image_mask = image_mask & view_mask[:, None, None]
    rgb = torch.where(image_mask[:, None], torch.nan_to_num(rgb), 0.0)
    refs = torch.full((q,), camera, device=device, dtype=torch.long)
    finite_uv = torch.isfinite(uv).all(-1)
    safe_uv = torch.nan_to_num(uv.float(), nan=0.0, posinf=0.0, neginf=0.0)
    hom = torch.cat((safe_uv, torch.ones_like(safe_uv[:, :1])), -1)
    ref_ray = (hom @ torch.linalg.inv(K[camera]).T) @ T[camera, :3, :3]
    center = camera_centers(T)[camera]
    depth_valid = torch.isfinite(depth) & (depth > 0) & (depth < 1e4)
    safe_depth = torch.where(depth_valid, depth.float(), torch.ones_like(depth.float()))
    points = center + safe_depth[..., None] * ref_ray[:, None]
    _, _, center_valid, J = project(points.reshape(-1, 3), K, T, (h, w), view_mask, image_mask)
    J = J.reshape(n, q, d, 2, 3)
    rms, minimum, tangent_valid = _projection_scales(J, refs, normals)
    derivative = torch.einsum("nqdij,qj->nqdi", J, ref_ray).norm(dim=-1)
    mixed = roles != roles[camera]
    # Match the coarser physical bandwidth. The exact 3-D intersection below,
    # not this first-order width estimate, determines all actual samples.
    width = minimum.detach().clamp_min(0.125).reciprocal().clamp(1.0, 8.0)
    width = torch.where(mixed[:, None, None, None] & antialias, width, torch.ones_like(width))
    offsets = (pixel_grid(2 * radius + 1, 2 * radius + 1, device) - radius).reshape(-1, 2)
    k = len(offsets)
    if antialias and aa_samples > 1:
        sub = (pixel_grid(aa_samples, aa_samples, device) + 0.5) / aa_samples - 0.5
        sub = sub.reshape(-1, 2)
    else:
        sub = rgb.new_zeros(1, 2)
    # Mixed-focal observations integrate equal physical cells in both images.
    # Same-role tele pairs retain their native one-pixel support and detail.
    sub = sub[None, None, None, None, None] * mixed[:, None, None, None, None, None, None]
    reference_uv = safe_uv[None, :, None, None, None, None] + width[..., None, None, None] * (
        offsets[None, None, None, None, :, None] + sub
    )
    hom_window = torch.cat((reference_uv, torch.ones_like(reference_uv[..., :1])), -1)
    rays = (hom_window @ torch.linalg.inv(K[camera]).T) @ T[camera, :3, :3]
    denominator = (rays * normals[None, :, None, :, None, None]).sum(-1)
    numerator = (ref_ray[:, None] * normals).sum(-1)
    plane_ok = denominator.abs() > grazing_cosine * rays.norm(dim=-1)
    ratio = numerator[None, :, None, :, None, None] / torch.where(
        plane_ok, denominator, torch.ones_like(denominator)
    )
    patch_depth = safe_depth[None, :, :, None, None, None] * ratio
    plane_ok = plane_ok & (patch_depth > 0) & (patch_depth < 1e4) & torch.isfinite(patch_depth)
    patch_depth = torch.where(plane_ok, patch_depth, torch.ones_like(patch_depth))
    world = center + patch_depth[..., None] * rays
    projected, geometric = _paired_projection(world, K, T, (h, w), view_mask, image_mask)
    values = sample_maps(rgb, projected.reshape(n, -1, 2)).reshape(n, q, d, p, k, -1, 3).mean(-2)
    ref_sample = (
        sample_maps(rgb[camera : camera + 1].expand(n, -1, -1, -1), reference_uv.reshape(n, -1, 2))
        .reshape(n, q, d, p, k, -1, 3)
        .mean(-2)
    )
    ref_mask = (
        sample_maps(
            image_mask[camera][None, None].float().expand(n, -1, -1, -1),
            reference_uv.reshape(n, -1, 2),
        )[..., 0].reshape(n, q, d, p, k, -1)
        > 0.999
    )
    ref_inside = (
        (reference_uv[..., 0] >= 0)
        & (reference_uv[..., 0] <= w - 1)
        & (reference_uv[..., 1] >= 0)
        & (reference_uv[..., 1] <= h - 1)
    )
    geometric = (geometric & ref_mask & ref_inside & plane_ok).all((-1, -2))
    geometric &= center_valid.reshape(n, q, d)[..., None] & tangent_valid[None]
    geometric &= depth_valid[None, :, :, None] & finite_uv[None, :, None, None]
    geometric &= (torch.arange(n, device=device) != camera)[:, None, None, None] & (
        derivative > 1e-5
    )[..., None]
    ref_variance = (ref_sample - ref_sample.mean(-2, keepdim=True)).square().mean((-2, -1))
    src_variance = (values - values.mean(-2, keepdim=True)).square().mean((-2, -1))
    texture = torch.minimum(ref_variance, src_variance).clamp_min(1e-12).sqrt()
    informative = geometric & (texture >= texture_threshold)
    # An unresolved wide observation is not a bad match. It contributes no
    # depth measurement, while its raw cost/geometry remain visible in debug.
    informative &= minimum * (2 * radius + 1) >= min_resolution_px
    luma = rgb.new_tensor([0.299, 0.587, 0.114])
    src_gray = (values * luma).sum(-1)
    ref_gray = (ref_sample * luma).sum(-1)
    descriptor_src = torch.tanh((src_gray - src_gray[..., k // 2 : k // 2 + 1]) / temperature)
    descriptor_ref = torch.tanh((ref_gray - ref_gray[..., k // 2 : k // 2 + 1]) / temperature)
    if method == "census":
        cost = (descriptor_src - descriptor_ref).abs().sum(-1) / max(k - 1, 1)
    else:
        left = values - values.mean(-2, keepdim=True)
        right = ref_sample - ref_sample.mean(-2, keepdim=True)
        denom = (left.square().sum((-2, -1)) * right.square().sum((-2, -1))).clamp_min(1e-12).sqrt()
        cost = 1 - (left * right).sum((-2, -1)) / denom
    cost = torch.nan_to_num(cost, nan=2.0, posinf=2.0, neginf=2.0).clamp(0, 2)
    return dict(
        per_view_cost=cost.permute(1, 2, 3, 0),
        per_view_valid=informative.permute(1, 2, 3, 0),
        geometric_valid=geometric.permute(1, 2, 3, 0),
        per_view_texture=texture.permute(1, 2, 3, 0),
        resolution_px=(minimum * (2 * radius + 1)).permute(1, 2, 3, 0),
        relative_scale=rms.permute(1, 2, 3, 0),
    )


def multifocal_patch_cost(
    inputs,
    b,
    refs,
    uv,
    depth,
    normals=None,
    *,
    include_fronto=False,
    patch_radius=1,
    query_chunk=32,
    depth_chunk=8,
    antialias=True,
    aa_samples=2,
    texture_threshold=0.006,
    census_temperature=0.02,
    grazing_cosine=0.025,
    agreement_cost=0.6,
    min_sources=1,
    method="census",
    min_resolution_px=1.5,
):
    """与oriented_patch_cost相同的[Q,D,P]输出，另暴露逐视图分辨率。

    允许独立长焦双目提供局部测量，不在这里作六目多数票；最终需独立的
    双向、全局模式、分辨率与矛盾证据验收。min_sources不是最终表面认证。
    """
    if method not in ("census", "zncc"):
        raise ValueError("method must be census or zncc")
    if patch_radius < 1 or min(query_chunk, depth_chunk, aa_samples, min_sources) < 1:
        raise ValueError("window radius, chunk sizes, AA samples and min_sources must be positive")
    if census_temperature <= 0 or texture_threshold < 0:
        raise ValueError("invalid photometric thresholds")
    normals = _normal_bank(inputs, b, refs, normals, include_fronto)
    n = inputs["rgb"].shape[1]
    q, d, p = len(refs), depth.shape[1], normals.shape[1]
    if (
        refs.dtype != torch.long
        or ((refs < 0) | (refs >= n)).any()
        or uv.shape != (q, 2)
        or depth.shape[0] != q
    ):
        raise ValueError("expected valid refs[Q], uv[Q,2], optical depth[Q,D]")
    pieces, order = [], []
    with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
        for camera in range(n):
            ids = (refs == camera).nonzero().flatten()
            for chunk in ids.split(query_chunk):
                if not len(chunk):
                    continue
                parts = [
                    _patch_group(
                        inputs,
                        b,
                        camera,
                        uv[chunk],
                        z,
                        normals[chunk],
                        radius=patch_radius,
                        antialias=antialias,
                        aa_samples=aa_samples,
                        texture_threshold=texture_threshold,
                        temperature=census_temperature,
                        grazing_cosine=grazing_cosine,
                        method=method,
                        min_resolution_px=min_resolution_px,
                    )
                    for z in depth[chunk].split(depth_chunk, 1)
                ]
                pieces.append(
                    {key: torch.cat([part[key] for part in parts], 1) for key in parts[0]}
                )
                order.append(chunk)
    if pieces:
        inverse = torch.cat(order).argsort()
        result = {key: torch.cat([part[key] for part in pieces])[inverse] for key in pieces[0]}
    else:
        result = {
            key: inputs["rgb"].new_empty(
                (q, d, p, n),
                dtype=torch.bool if key in ("per_view_valid", "geometric_valid") else torch.float32,
            )
            for key in (
                "per_view_cost",
                "per_view_valid",
                "geometric_valid",
                "per_view_texture",
                "resolution_px",
                "relative_scale",
            )
        }
    valid, cost = result["per_view_valid"], result["per_view_cost"]
    count = valid.sum(-1)
    agreement = (valid & (cost <= agreement_cost)).sum(-1)
    raw = torch.where(valid, cost, 0.0).sum(-1) / count.clamp_min(1)
    accepted = (count >= min_sources) & (agreement >= min_sources)
    result.update(
        cost=torch.where(accepted, raw, raw * 0 + 2),
        raw_cost=torch.where(count > 0, raw, raw * 0 + 2),
        valid=accepted,
        source_count=count,
        support=torch.where(accepted, agreement + 1, 0),
        texture=result["per_view_texture"].amax(-1),
        includes_fronto=include_fronto,
        normal_hypothesis_count=p,
        min_sources=min_sources,
    )
    return result


class MultiFocalSampler(nn.Module):
    """固定参考编码层，按J切向投影尺度对相邻两个源层进行线性融合。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.get("model", cfg)
        options = _options(cfg)
        self.routing = bool(options.get("scale_routing", True))
        self.antialias = bool(options.get("common_scale_antialias", True))
        self.aa_samples = int(options.get("antialias_samples", 2))
        self.radius = int(options.get("matching_patch_radius", 1))
        self.min_resolution_px = float(options.get("min_resolution_px", 1.5))
        self.texture_threshold = float(
            options.get("matching_texture_threshold", m.get("texture_threshold", 0.002))
        )
        self.temperature = float(m.get("et_census_temperature", 0.02))
        channels = int(m.get("feature_dim", 64))
        self.adapters = nn.ModuleDict(
            {
                str(s): nn.Linear(channels, channels, bias=False)
                for s in options.get("feature_strides", (4, 8, 16, 32))
            }
        )
        for layer in self.adapters.values():
            nn.init.eye_(layer.weight)

    def score(self, inputs, b, refs, uv, depths, normals=None, stride=4):
        result = multifocal_patch_cost(
            inputs,
            b,
            refs,
            uv,
            depths,
            normals,
            patch_radius=self.radius,
            antialias=self.antialias,
            aa_samples=self.aa_samples,
            texture_threshold=self.texture_threshold,
            census_temperature=self.temperature,
            min_resolution_px=self.min_resolution_px,
        )
        return dict(
            photo_cost=result["per_view_cost"][:, :, 0],
            valid=result["per_view_valid"][:, :, 0],
            texture=result["per_view_texture"][:, :, 0],
            resolution_px=result["resolution_px"][:, :, 0],
            geometric_valid=result["geometric_valid"][:, :, 0],
            ref_resolution_px=2 * self.radius + 1,
        )

    def sample(self, inputs, pyramid, b, stage_stride, points, refs, ref_uv, normals=None):
        n = inputs["rgb"].shape[1]
        q, d = points.shape[:2]
        if not q:
            c = pyramid[stage_stride].shape[2]
            empty = points.new_empty(q, d, n)
            return dict(
                sampled=points.new_empty(q, d, n, c),
                reference=points.new_empty(q, c),
                photo_cost=empty,
                valid=empty.bool(),
                texture=empty,
                scale_debug=dict(
                    desired_stride=empty,
                    layer_weights=points.new_empty(q, d, n, len(pyramid)),
                    strides=sorted(pyramid),
                    resolution_px=empty,
                    geometric_valid=empty.bool(),
                ),
            )
        K, T = inputs["K"][b].float(), inputs["T_cam_from_rig"][b].float()
        vm = inputs.get("view_mask")
        im = inputs.get("image_mask")
        uv, _, _, J = project(
            points.reshape(-1, 3),
            K,
            T,
            inputs["rgb"].shape[-2:],
            None if vm is None else vm[b],
            None if im is None else im[b],
        )
        normal = _normal_bank(inputs, b, refs, normals)
        rms, _, _ = _projection_scales(J.reshape(n, q, d, 2, 3), refs, normal)
        desired = (rms[..., 0].permute(1, 2, 0) * stage_stride).clamp_min(1e-6)
        strides = sorted(pyramid)
        if stage_stride not in strides:
            raise ValueError("reference stage_stride must exist in feature pyramid")
        log_stride = points.new_tensor(strides).float().log2()
        target = desired.log2().clamp(log_stride[0], log_stride[-1])
        if not self.routing:
            target = torch.full_like(target, float(torch.log2(torch.tensor(float(stage_stride)))))
        # Adjacent layers only: identity initialization gives a neutral learned
        # channel projection while all used adapters receive task gradients.
        weights = torch.zeros(*target.shape, len(strides), device=points.device, dtype=target.dtype)
        if len(strides) == 1:
            weights[..., 0] = 1
        else:
            hi = torch.searchsorted(log_stride, target.contiguous(), right=True).clamp(
                1, len(strides) - 1
            )
            lo = hi - 1
            alpha = (target - log_stride[lo]) / (log_stride[hi] - log_stride[lo])
            weights.scatter_(-1, lo[..., None], (1 - alpha)[..., None])
            weights.scatter_add_(-1, hi[..., None], alpha[..., None])
        samples = []
        for level in strides:
            maps = pyramid[level][b]
            feature = sample_maps(maps, uv, level).reshape(n, q, d, -1).permute(1, 2, 0, 3)
            samples.append(self.adapters[str(level)](feature) if self.routing else feature)
        sampled = (torch.stack(samples, -2) * weights[..., None]).sum(-2)
        reference_all = sample_maps(
            pyramid[stage_stride][b], ref_uv[None].expand(n, -1, -1), stage_stride
        )
        reference = reference_all[refs, torch.arange(q, device=refs.device)]
        reference = self.adapters[str(stage_stride)](reference) if self.routing else reference
        with torch.autocast(device_type=points.device.type, enabled=False):
            ref_camera = (
                torch.einsum("qij,qdj->qdi", T[refs, :3, :3], points.float()) + T[refs, None, :3, 3]
            )
        scored = self.score(inputs, b, refs, ref_uv, ref_camera[..., 2], normals, stage_stride)
        return dict(
            sampled=sampled,
            reference=reference,
            photo_cost=scored["photo_cost"],
            valid=scored["valid"],
            texture=scored["texture"],
            scale_debug=dict(
                desired_stride=desired,
                layer_weights=weights,
                strides=strides,
                resolution_px=scored["resolution_px"],
                geometric_valid=scored["geometric_valid"],
            ),
        )


def photometric_observations(inputs, b, refs, uv, depths, normals=None, stride=4, *, cfg=None):
    """仅RGB的验收接口，无编码器/GT；stride保留统一调用契约。"""
    options = _options(cfg or {})
    m = (cfg or {}).get("model", cfg or {})
    radius = int(options.get("matching_patch_radius", 1))
    result = multifocal_patch_cost(
        inputs,
        b,
        refs,
        uv,
        depths,
        normals,
        patch_radius=radius,
        antialias=bool(options.get("common_scale_antialias", True)),
        aa_samples=int(options.get("antialias_samples", 2)),
        texture_threshold=float(
            options.get("matching_texture_threshold", m.get("texture_threshold", 0.002))
        ),
        census_temperature=float(m.get("et_census_temperature", 0.02)),
        min_resolution_px=float(options.get("min_resolution_px", 1.5)),
    )
    return dict(
        photo_cost=result["per_view_cost"][:, :, 0],
        valid=result["per_view_valid"][:, :, 0],
        texture=result["per_view_texture"][:, :, 0],
        resolution_px=result["resolution_px"][:, :, 0],
        geometric_valid=result["geometric_valid"][:, :, 0],
        ref_resolution_px=2 * radius + 1,
    )
