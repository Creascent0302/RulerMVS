"""用真实图像和固定标定筛选稠密预测，不把网络深度直接当作空间观测。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .geometry import camera_centers, pixel_grid, project, sample_maps, unproject
from .models.token_hierarchy import _census_patch_maps, _soft_census


def _complete_patch_mask(mask):
    """5×5补丁必须全部落在真实有效图像内，包括边缘的双线性采样足迹。"""
    missing = F.pad(1 - mask[:, None].float(), (2, 2, 2, 2), value=1)
    return 1 - F.max_pool2d(missing, kernel_size=5, stride=1)


def _depth_footprint(depth, uv):
    """返回双线性足迹的深度上下界，避免前景/背景插值制造假的一致深度。"""
    n, h, w = depth.shape
    x = uv[..., 0].clamp(0, w - 1)
    y = uv[..., 1].clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = x.ceil().long(), y.ceil().long()
    view = torch.arange(n, device=depth.device)[:, None]
    neighbors = torch.stack(
        [depth[view, y0, x0], depth[view, y0, x1], depth[view, y1, x0], depth[view, y1, x1]],
        dim=-1,
    )
    valid = torch.isfinite(neighbors).all(-1) & (neighbors > 0).all(-1)
    return neighbors.amin(-1), neighbors.amax(-1), valid


@torch.no_grad()
def verify_dense_depth(inputs, depth, cfg):
    """筛选 ``depth[B,N,H,W]`` 的表面候选，返回同形状 support/observed/state。

    输入遵循项目的 RGB、K、T_cam_from_rig、view_mask、image_mask 契约，
    深度为米制相机光轴深度；只读取这些输入与多视图预测深度，不读取标签或置信度。
    参数位于完整配置的 ``model.dense_evidence``，距离阈值均为米：

    - chunk_size=4096，min_support=2（包括参考相机）。
    - census_threshold=0.12，texture_threshold=0.002（灰度补丁标准差）。
    - min_parallax_pixels_per_meter=0.001，min_baseline_m=0.0001。
    - depth_absolute_tolerance=0.10，depth_relative_tolerance=0.01。

    每个源视图必须同时通过真实5×5 Census、完整补丁掩码、纹理、视差和
    源深度一致性。源预测更近时拒绝遮挡后方候选；深度插值的四个邻点均须一致。
    support 只统计通过筛选的视图；没有任何源支持时为0，不以参考自身证明表面。
    observed 表示“经过一致性筛选”，不是物理保证：重复纹理、共同错误的深度
    和遮挡边缘仍可能误通过。无证据保留 unknown=2，通过为 surface=1，绝不产生 free=0。
    这里没有训练梯度，也不定义整个三维空间的自由/占据状态。
    """
    rgb = inputs["rgb"]
    batch, cameras, _, height, width = rgb.shape
    if depth.shape != (batch, cameras, height, width):
        raise ValueError("dense depth 必须为 [B,N,H,W]，且与 RGB 的原图尺寸一致")
    model = cfg.get("model", {})
    settings = model.get("dense_evidence", {})
    chunk_size = int(settings.get("chunk_size", 4096))
    min_support = int(settings.get("min_support", 2))
    if chunk_size < 1 or min_support < 2:
        raise ValueError("chunk_size 必须为正，min_support 必须至少为2")
    photo_threshold = float(settings.get("census_threshold", 0.12))
    texture_threshold = float(settings.get("texture_threshold", 0.002))
    parallax_threshold = float(settings.get("min_parallax_pixels_per_meter", 0.001))
    baseline_threshold = max(float(settings.get("min_baseline_m", 0.0001)), 1e-6)
    absolute_tolerance = float(settings.get("depth_absolute_tolerance", 0.10))
    relative_tolerance = float(settings.get("depth_relative_tolerance", 0.01))
    near, far = float(model.get("near", 1e-4)), float(model.get("far", float("inf")))
    if min(texture_threshold, parallax_threshold, absolute_tolerance, relative_tolerance) < 0:
        raise ValueError("证据筛选阈值不能为负")

    device = rgb.device
    views = inputs.get("view_mask", torch.ones(batch, cameras, device=device, dtype=torch.bool))
    masks = inputs.get(
        "image_mask", torch.ones(batch, cameras, height, width, device=device, dtype=torch.bool)
    )
    support = torch.zeros_like(depth, dtype=torch.long)
    pixels = pixel_grid(height, width, device=device).reshape(-1, 2)

    # 几何、Census及阈值判断固定用FP32，外层AMP不会改变证据判定的数值精度。
    with torch.autocast(device_type=device.type, enabled=False):
        for b in range(batch):
            image_valid = masks[b].bool() & views[b, :, None, None].bool()
            image_valid &= torch.isfinite(rgb[b]).all(1)
            images = torch.where(image_valid[:, None], rgb[b].float(), 0.0)
            patches = _census_patch_maps(images)
            patch_valid = _complete_patch_mask(image_valid)
            K, T = inputs["K"][b].float(), inputs["T_cam_from_rig"][b].float()
            centers = camera_centers(T)
            predicted = depth[b].float()
            depth_valid = torch.isfinite(predicted) & (predicted >= near) & (predicted <= far)
            depth_valid &= predicted > 0

            for ref in range(cameras):
                reference_valid = depth_valid[ref] & (patch_valid[ref, 0] > 0.999)
                selected = reference_valid.flatten().nonzero().flatten()
                sources = (centers - centers[ref]).norm(dim=-1) > baseline_threshold
                sources[ref] = False
                sources &= views[b].bool()
                # 未解码或整张无有效深度的源图不可能通过后面的深度足迹检验。
                # 提前拒绝只省去无效计算，不将缺失预测当作可见性证据。
                sources &= depth_valid.flatten(1).any(1)
                if selected.numel() == 0 or not sources.any():
                    continue
                for indices in selected.split(chunk_size):
                    uv_ref = pixels[indices]
                    z_ref = predicted[ref].flatten()[indices]
                    points, rays = unproject(uv_ref, z_ref, K[ref], T[ref])
                    uv, z, valid, jacobian = project(points, K, T, (height, width), views[b])
                    parallax = (jacobian * rays[None, :, None]).sum(-1).norm(dim=-1)
                    valid &= sources[:, None] & (parallax > parallax_threshold)
                    valid &= sample_maps(patch_valid, uv)[..., 0] > 0.999

                    raw_ref = patches[ref].flatten(1)[:, indices].T
                    raw_source = sample_maps(patches, uv)
                    ref_texture = raw_ref.std(dim=-1, correction=0)
                    source_texture = raw_source.std(dim=-1, correction=0)
                    valid &= ref_texture[None] > texture_threshold
                    valid &= source_texture > texture_threshold
                    photo_cost = (_soft_census(raw_source) - _soft_census(raw_ref)[None]).abs().sum(
                        -1
                    ) / 24
                    valid &= photo_cost < photo_threshold

                    low, high, footprint_valid = _depth_footprint(predicted, uv)
                    tolerance = absolute_tolerance + relative_tolerance * z.clamp_min(0)
                    # 更近的源深度表示潜在遮挡；更远则几何不一致。两种均不能支持本表面。
                    valid &= footprint_valid & (low >= near) & (high <= far)
                    valid &= (low >= z - tolerance) & (high <= z + tolerance)
                    source_support = valid.sum(0)
                    count = torch.where(source_support > 0, source_support + 1, 0)
                    support[b, ref].view(-1)[indices] = count

    observed = support >= min_support
    state = torch.where(observed, 1, 2).long()
    return {"support": support, "observed": observed, "state": state}
