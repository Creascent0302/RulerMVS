"""仅 TRAIN 使用的真深度前向再投影：构造新的短基线源视图。

不凭空生成遮挡后纹理。像素按最近栅格点投放，z-buffer 保留近表面，
未覆盖像素显式无效。该增强会改变数据，不能用于测试或模型推理。
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def forward_splat(rgb, depth, valid, intrinsic, translation_x):
    """同 K / R 的虚拟相机前向投放，所有张量支持 B 维和 CPU/CUDA。

    translation_x 是 T_new_from_reference 的 x 平移，单位和深度一致。
    每个 reference 像素视作一个带颜色的点；先精确投影，再最近栅格化。
    返回的 index 与 projected_uv 明确说明每个源像素来自哪个参考点。
    源深度是这个点的光轴深度，不等于隐藏区域或连续网格的真实渲染。
    """
    batch, _, height, width = rgb.shape
    if depth.shape != (batch, height, width) or valid.shape != depth.shape:
        raise ValueError("depth和valid须为[B,H,W]，与RGB网格对应")
    device = rgb.device
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing="ij")
    pixels = torch.stack((x, y, torch.ones_like(x))).flatten(1).float()
    rays = torch.linalg.solve(intrinsic.float(), pixels[None].expand(batch, -1, -1))
    z = depth.float().flatten(1)
    legal = valid.bool().flatten(1) & torch.isfinite(z) & (z > 0)
    safe_z = torch.where(legal, z, 1.)
    points = rays * safe_z[:, None]
    points[:, 0] += translation_x.float()[:, None]
    homogeneous = intrinsic.float() @ points
    denominator = homogeneous[:, 2].clamp_min(1e-12)
    uv = homogeneous[:, :2] / denominator[:, None]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    legal = legal & inside & (points[:, 2] > 0) & torch.isfinite(uv).all(1)
    rounded = uv.nan_to_num(0.).round().long()
    destination = rounded[:, 1].clamp(0, height - 1) * width + rounded[:, 0].clamp(0, width - 1)
    count = height * width
    nearest = torch.full((batch, count), torch.inf, device=device)
    incoming = torch.where(legal, points[:, 2], torch.inf)
    nearest.scatter_reduce_(1, destination, incoming, reduce="amin", include_self=True)
    wins_depth = legal & (incoming == nearest.gather(1, destination))
    source_index = torch.arange(count, device=device)[None].expand(batch, -1)
    chosen = torch.full_like(source_index, count)
    chosen.scatter_reduce_(1, destination, torch.where(wins_depth, source_index, count),
                            reduce="amin", include_self=True)
    covered = chosen < count
    safe_index = chosen.clamp_max(count - 1)
    colour = rgb.flatten(2).gather(2, safe_index[:, None].expand(-1, 3, -1))
    colour = torch.where(covered[:, None], colour, 0.)
    projected = uv.gather(2, safe_index[:, None].expand(-1, 2, -1))
    projected = torch.where(covered[:, None], projected, 0.)
    return {
        "rgb": colour.reshape(batch, 3, height, width),
        "depth": torch.where(covered, nearest, 0.).reshape(batch, height, width),
        "valid": covered.reshape(batch, height, width),
        "reference_pixel_index": torch.where(covered, chosen, -1).reshape(batch, height, width),
        "projected_uv": projected.reshape(batch, 2, height, width),
        "reference_projected_uv": uv.reshape(batch, 2, height, width),
        "reference_in_view": legal.reshape(batch, height, width),
        "collision_count": legal.sum(1) - covered.sum(1),
    }


@torch.no_grad()
def synthesize_pair(inputs, targets, domain, *, generator, probability=1.,
                    baseline_norm=(0.03, 0.3), reference=0, source_slot=1,
                    keep_other_sources=False):
    """返回新 inputs、显式源监督 mapping、增强记录；不修改原 inputs/targets。

    baseline = LogUniform(baseline_norm) * domain.depth_scale，不读取GT来
    选基线。正负号独立等概率。domain.depth_scale是已有规范化单位约定，
    非米制数据的baseline不是物理米。未选中样本原输入逐值保留。

    新源GT单独返回，不能继续使用旧source槽的原targets；需要对应监督
    时显式调用 correspondence_targets()。参考GT始终保持原值。
    """
    if not 0 <= probability <= 1 or not 0 < baseline_norm[0] <= baseline_norm[1]:
        raise ValueError("概率或预设基线范围非法")
    scale = float(domain["depth_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale须为有限正数")
    rgb = inputs["rgb"]
    batch, views, _, height, width = rgb.shape
    if not 0 <= reference < views or not 0 <= source_slot < views or source_slot == reference:
        raise ValueError("源槽必须存在且不同于参考槽")
    depth = targets["depth"][:, reference]
    valid = targets["depth_valid"][:, reference] & inputs["image_mask"][:, reference]
    valid = valid & inputs["view_mask"][:, reference, None, None]
    random = torch.rand(batch, 3, generator=generator, device=generator.device).to(rgb.device)
    magnitude = (math.log(baseline_norm[0])
                 + random[:, 1] * math.log(baseline_norm[1] / baseline_norm[0])).exp() * scale
    shift = magnitude * torch.where(random[:, 2] < .5, -1., 1.)
    active = (random[:, 0] < probability) & valid.flatten(1).any(1)
    rendered = forward_splat(rgb[:, reference], depth, valid,
                              inputs["K"][:, reference], shift)
    active = active & rendered["valid"].flatten(1).any(1)
    result = dict(inputs)
    for key in ("rgb", "K", "T_cam_from_rig", "view_mask", "image_mask",
                "camera_valid", "image_sizes"):
        if key in inputs:
            result[key] = inputs[key].clone()
    result["rgb"][active, source_slot] = rendered["rgb"][active]
    result["image_mask"][active, source_slot] = rendered["valid"][active]
    result["view_mask"][active, source_slot] = True
    result["K"][active, source_slot] = inputs["K"][active, reference]
    transform = inputs["T_cam_from_rig"][:, reference].clone()
    transform[:, 0, 3] += shift
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
        matching_size = inputs["matching_rgb"].shape[-2:]
        rebuilt = F.interpolate(result["rgb"].flatten(0, 1), matching_size,
                                mode="bilinear", align_corners=False).unflatten(0, (batch, views))
        mask = F.interpolate(result["image_mask"].flatten(0, 1)[:, None].float(), matching_size,
                             mode="bilinear", align_corners=False)[:, 0] > 1. - 1e-6
        mask = mask.unflatten(0, (batch, views)) & result["view_mask"][:, :, None, None]
        result["matching_rgb"] = torch.where(active[:, None, None, None, None], rebuilt,
                                               inputs["matching_rgb"])
        result["matching_image_mask"] = torch.where(active[:, None, None, None], mask,
                                                      inputs["matching_image_mask"])
    supervision = {
        "active": active,
        "reference_index": reference,
        "source_slot": source_slot,
        "source_depth": rendered["depth"],
        "source_depth_valid": rendered["valid"] & active[:, None, None],
        "reference_pixel_index": rendered["reference_pixel_index"],
        "projected_uv": rendered["projected_uv"],
        "translation_x": shift,
    }
    record = {
        "active": active,
        "baseline_normalized": magnitude / scale,
        "baseline_data_units": magnitude,
        "source_coverage": rendered["valid"].float().mean((1, 2)),
        "collisions": rendered["collision_count"],
        "retained_views": result["view_mask"].sum(1),
    }
    return result, supervision, record


def correspondence_targets(targets, supervision):
    """显式替换虚拟源的深度标签，废弃其原语义/法线等标签。

    参考标签逐值保留；只用于训练监督，返回值不进入模型forward。
    """
    active = supervision["active"]
    slot = supervision["source_slot"]
    result = dict(targets)
    for key, value in targets.items():
        if not torch.is_tensor(value) or value.ndim < 3:
            continue
        result[key] = value.clone()
        if key == "depth":
            result[key][active, slot] = supervision["source_depth"][active]
        elif key == "depth_valid":
            result[key][active, slot] = supervision["source_depth_valid"][active]
        elif key in ("semantic", "instance"):
            result[key][active, slot] = -1
        elif key in ("normals", "depth_resize_interpolated", "depth_resize_boundary"):
            result[key][active, slot] = 0
    return result
