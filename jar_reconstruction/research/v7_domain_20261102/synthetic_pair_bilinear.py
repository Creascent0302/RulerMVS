"""TRAIN专用：有遮挡筛选的双线性短基线再生成，保留nearest历史。

限定相同K/R和相机x平移，精确映射u'=u+fx*tx/Z。只合并z-buffer
近表面相容点，洞不补纹理。颜色与逆深度使用同一投放权重；保留完整
对应映射供审计。非平面局部插值仍是近似，不能当作真实新采集数据。
"""

import math

import torch
from torch.nn import functional as F

from research.v7_domain_20261102.synthetic_pair import correspondence_targets


@torch.no_grad()
def forward_splat(rgb, depth, valid, intrinsic, translation_x, *,
                  surface_relative_tolerance=0.01, minimum_weight=0.5,
                  centroid_tolerance=0.25):
    batch, _, height, width = rgb.shape
    if depth.shape != (batch, height, width) or valid.shape != depth.shape:
        raise ValueError("depth/valid须为与RGB对应的[B,H,W]")
    if not 0 <= surface_relative_tolerance < 1 or minimum_weight <= 0 or centroid_tolerance < 0:
        raise ValueError("投放阈值非法")
    if not torch.allclose(intrinsic[:, 1, 0], torch.zeros_like(intrinsic[:, 1, 0])):
        raise ValueError("本增强限定标准CV内参K[1,0]=0的水平整流几何")
    if not torch.allclose(intrinsic[:, 2], intrinsic.new_tensor((0., 0., 1.)).expand(batch, -1)):
        raise ValueError("本增强限定标准CV内参最后一行为[0,0,1]")
    device, count = rgb.device, height * width
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    x, y = x.flatten().float(), y.flatten().float()
    z = depth.float().flatten(1)
    valid = valid.bool().flatten(1) & torch.isfinite(z) & (z > 0)
    safe_z = torch.where(valid, z, 1.)
    u = x[None] + intrinsic[:, 0, 0, None].float() * translation_x[:, None].float() / safe_z
    lower = u.floor()
    fraction = u - lower
    # 相同K/R、x平移时v严格不变，避免FP32近整数v产生虚假跨行遮挡。
    columns = torch.stack((lower, lower + 1), 1)
    raw_weight = torch.stack((1. - fraction, fraction), 1)
    legal = (valid[:, None] & (columns >= 0) & (columns < width)
             & (raw_weight > 1e-6) & torch.isfinite(u[:, None]))
    destination = (y[None, None] * width + columns.clamp(0, width - 1)).long().flatten(1)
    raw_weight, legal = raw_weight.flatten(1), legal.flatten(1)
    incoming_depth = safe_z[:, None].expand(-1, 2, -1).flatten(1)
    nearest = torch.full((batch, count), torch.inf, device=device)
    nearest.scatter_reduce_(1, destination, torch.where(legal, incoming_depth, torch.inf),
                             reduce="amin", include_self=True)
    front = nearest.gather(1, destination)
    compatible = legal & (incoming_depth <= front * (1. + surface_relative_tolerance))
    contribution = torch.where(compatible, raw_weight, 0.)
    mass = torch.zeros(batch, count, device=device).scatter_add_(1, destination, contribution)
    weights = contribution / mass.gather(1, destination).clamp_min(1e-12)
    projected_u = u[:, None].expand(-1, 2, -1).flatten(1)
    centroid = torch.zeros_like(mass).scatter_add_(1, destination, weights * projected_u)
    displacement = (centroid - x[None]).abs()
    covered = (mass >= minimum_weight) & (displacement <= centroid_tolerance)
    # 不完整的单点外伸不能冒充已知连续表面；其贡献也从对应标签中清除。
    weights = weights * covered.gather(1, destination)
    source_index = torch.arange(count, device=device)[None, None].expand(batch, 2, -1).flatten(1)
    colour = rgb.flatten(2).gather(2, source_index[:, None].expand(-1, 3, -1))
    colour_out = torch.zeros(batch, 3, count, device=device, dtype=torch.float32)
    colour_out.scatter_add_(2, destination[:, None].expand(-1, 3, -1), colour.float() * weights[:, None])
    inverse_depth = torch.zeros_like(mass).scatter_add_(1, destination, weights / incoming_depth)
    depth_out = torch.where(covered, inverse_depth.clamp_min(1e-12).reciprocal(), 0.)
    strongest = torch.zeros_like(mass).scatter_reduce_(1, destination, weights, reduce="amax", include_self=True)
    winner = compatible & (weights > 0) & (weights == strongest.gather(1, destination))
    chosen = torch.full((batch, count), count, dtype=torch.long, device=device)
    chosen.scatter_reduce_(1, destination, torch.where(winner, source_index, count),
                            reduce="amin", include_self=True)
    centroid = torch.where(covered, centroid, 0.)
    source_uv = torch.stack((centroid, y[None].expand(batch, -1) * covered), 1)
    return {
        "rgb": colour_out.reshape(batch, 3, height, width).to(rgb.dtype),
        "depth": depth_out.reshape(batch, height, width),
        "valid": covered.reshape(batch, height, width),
        "reference_pixel_index": torch.where(covered, chosen, -1).reshape(batch, height, width),
        "projected_uv": source_uv.reshape(batch, 2, height, width),
        "reference_projected_uv": torch.stack((u, y[None].expand(batch, -1)), 1).reshape(batch, 2, height, width),
        "splat_mass": mass.reshape(batch, height, width),
        "centroid_error": displacement.reshape(batch, height, width),
        "occluded_contributions": (legal & ~compatible).sum(1),
        "mapping": {"reference_index": source_index, "source_index": destination,
                    "weight": weights, "source_depth": incoming_depth,
                    "projected_u": projected_u},
    }


@torch.no_grad()
def synthesize_pair(inputs, targets, domain, *, generator, probability=1.,
                    baseline_norm=(0.03, 0.3), reference=0, source_slot=1,
                    keep_other_sources=False, surface_relative_tolerance=0.01,
                    minimum_weight=0.5, centroid_tolerance=0.25):
    if not 0 <= probability <= 1 or not 0 < baseline_norm[0] <= baseline_norm[1]:
        raise ValueError("概率或基线范围非法")
    scale = float(domain["depth_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale须为有限正数")
    rgb = inputs["rgb"]
    batch, views, _, height, width = rgb.shape
    if not 0 <= reference < views or not 0 <= source_slot < views or reference == source_slot:
        raise ValueError("源槽必须存在且不同于参考槽")
    valid = targets["depth_valid"][:, reference] & inputs["image_mask"][:, reference]
    valid = valid & inputs["view_mask"][:, reference, None, None]
    random = torch.rand(batch, 3, generator=generator, device=generator.device).to(rgb.device)
    magnitude = (math.log(baseline_norm[0])
                 + random[:, 1] * math.log(baseline_norm[1] / baseline_norm[0])).exp() * scale
    requested_shift = magnitude * torch.where(random[:, 2] < .5, -1., 1.)
    transform = inputs["T_cam_from_rig"][:, reference].clone()
    transform[:, 0, 3] += requested_shift
    # 渲染采用实际可存储的FP32标定差值，避免大世界坐标下加法舍入后不一致。
    shift = transform[:, 0, 3] - inputs["T_cam_from_rig"][:, reference, 0, 3]
    rendered = forward_splat(rgb[:, reference], targets["depth"][:, reference], valid,
                              inputs["K"][:, reference], shift,
                              surface_relative_tolerance=surface_relative_tolerance,
                              minimum_weight=minimum_weight, centroid_tolerance=centroid_tolerance)
    active = (random[:, 0] < probability) & rendered["valid"].flatten(1).any(1)
    result = dict(inputs)
    for key in ("rgb", "K", "T_cam_from_rig", "view_mask", "image_mask",
                "camera_valid", "image_sizes"):
        if key in inputs:
            result[key] = inputs[key].clone()
    result["rgb"][active, source_slot] = rendered["rgb"][active]
    result["image_mask"][active, source_slot] = rendered["valid"][active]
    result["view_mask"][active, source_slot] = True
    result["K"][active, source_slot] = inputs["K"][active, reference]
    result["T_cam_from_rig"][active, source_slot] = transform[active]
    if "camera_valid" in result:
        result["camera_valid"][active, source_slot] = True
    if "image_sizes" in result:
        result["image_sizes"][active, source_slot] = result["image_sizes"].new_tensor((height, width))
    if not keep_other_sources:
        other = torch.ones(views, device=rgb.device, dtype=torch.bool)
        other[reference] = other[source_slot] = False
        removed = active[:, None] & other[None]
        result["view_mask"] &= ~removed
        result["image_mask"] &= ~removed[:, :, None, None]
        result["rgb"] *= (~removed)[:, :, None, None, None]
        if "camera_valid" in result:
            result["camera_valid"] &= ~removed
        if "image_sizes" in result:
            result["image_sizes"] *= (~removed)[:, :, None]
    if "matching_rgb" in inputs:
        size = inputs["matching_rgb"].shape[-2:]
        rebuilt = F.interpolate(result["rgb"].flatten(0, 1), size, mode="bilinear",
                                align_corners=False).unflatten(0, (batch, views))
        mask = F.interpolate(result["image_mask"].flatten(0, 1)[:, None].float(), size,
                             mode="bilinear", align_corners=False)[:, 0] > 1. - 1e-6
        mask = mask.unflatten(0, (batch, views)) & result["view_mask"][:, :, None, None]
        result["matching_rgb"] = torch.where(active[:, None, None, None, None], rebuilt,
                                               inputs["matching_rgb"])
        result["matching_image_mask"] = torch.where(active[:, None, None, None], mask,
                                                      inputs["matching_image_mask"])
    mapping = dict(rendered["mapping"])
    mapping["weight"] = mapping["weight"] * active[:, None]
    supervision = {"active": active, "reference_index": reference, "source_slot": source_slot,
                   "source_depth": rendered["depth"],
                   "source_depth_valid": rendered["valid"] & active[:, None, None],
                   "reference_pixel_index": rendered["reference_pixel_index"],
                   "projected_uv": rendered["projected_uv"], "translation_x": shift,
                   "mapping": mapping}
    record = {"active": active, "baseline_normalized": shift.abs() / scale,
              "baseline_data_units": shift.abs(), "requested_baseline_data_units": magnitude,
              "source_coverage": rendered["valid"].float().mean((1, 2)),
              "occluded_contributions": rendered["occluded_contributions"],
              "retained_views": result["view_mask"].sum(1),
              "centroid_error": rendered["centroid_error"]}
    return result, supervision, record


def augment_training_batch(inputs, targets, domain, *, generator, probability=0.25,
                           baseline_norm=(0.03, 0.3), **render_options):
    """训练入口薄包装：默认25%再生成，其余样本保留；灰度由外部统一控制。

    显式替换虚拟源监督，返回合成/未合成数量用于记录真实曝光。
    本函数不用于评测，也不按dataset名选择算法。
    """
    augmented, supervision, record = synthesize_pair(
        inputs, targets, domain, generator=generator, probability=probability,
        baseline_norm=baseline_norm, **render_options)
    updated_targets = correspondence_targets(targets, supervision)
    if "depth_resize_interpolated" in updated_targets:
        active = supervision["active"]
        slot = supervision["source_slot"]
        updated_targets["depth_resize_interpolated"][active, slot] = supervision["source_depth_valid"][active]
    record["synthetic_count"] = int(supervision["active"].sum())
    record["original_count"] = inputs["rgb"].shape[0] - record["synthetic_count"]
    return augmented, updated_targets, record
