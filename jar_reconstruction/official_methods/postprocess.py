"""共同评价所用的显式后处理；不是五种网络的组成部分。"""

import torch
import torch.nn.functional as F


def depth_on_image_grid(prediction, inputs):
    """按内参映射重采样原生深度；不以张量形状猜测像素中心。"""
    if "depth" not in prediction:
        return None
    depth = prediction["depth"]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    b, r, hd, wd = depth.shape
    refs = prediction["reference_indices"]
    h, w = inputs["rgb"].shape[-2:]
    k_image = inputs["K"][:, refs]
    k_depth = prediction.get("depth_K")
    if k_depth is None:
        if (h, w) != (hd, wd):
            raise ValueError("低分辨率深度必须提供 depth_K，不能猜测其像素中心/步幅")
        k_depth = k_image
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij"
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), -1).float()
    uvh = torch.einsum(
        "brij,hwj->brhwi", k_depth.float() @ torch.linalg.inv(k_image.float()), pixels
    )
    uv = uvh[..., :2] / uvh[..., 2:].clamp_min(1e-8)
    grid = (uv + 0.5) * uv.new_tensor([2 / wd, 2 / hd]) - 1
    valid = torch.isfinite(depth) & (depth > 0)
    declared = prediction.get("depth_valid", prediction.get("valid"))
    if declared is not None:
        if declared.ndim == 5:
            declared = declared[:, :, 0]
        valid &= declared.bool()
    clean = torch.where(valid, depth, 0)
    sample = lambda value: F.grid_sample(
        value.flatten(0, 1)[:, None].float(),
        grid.flatten(0, 1),
        padding_mode="border",
        align_corners=False,
    ).reshape(b, r, h, w)
    weights = sample(valid)
    out = sample(clean) / weights.clamp_min(1e-8)
    inside = (
        (uv[..., 0] >= -0.5)
        & (uv[..., 0] <= wd - 0.5)
        & (uv[..., 1] >= -0.5)
        & (uv[..., 1] <= hd - 0.5)
    )
    mask = inside & (weights > 0.999) & inputs["image_mask"][:, refs]
    mask &= inputs["view_mask"][:, refs, None, None]
    return {
        "depth": out,
        "depth_valid": mask,
        "reference_indices": refs,
        "conversion": "按 depth_K 精确映射，双线性深度重采样；无几何融合或置信度筛选",
    }


def depth_points(depth, valid, k, t_cam_from_rig, roi=None, max_points=8192):
    """米制反投影；固定像素顺序等距取样，不读取真实几何挑选预测点。"""
    n, h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij"
    )
    pix = torch.stack((x, y, torch.ones_like(x)), -1).float()
    rays = torch.einsum("nij,hwj->nhwi", torch.linalg.inv(k.float()), pix)
    camera = rays * depth[..., None]
    rig_from_cam = torch.linalg.inv(t_cam_from_rig.float())
    points = (
        torch.einsum("nij,nhwj->nhwi", rig_from_cam[:, :3, :3], camera)
        + rig_from_cam[:, None, None, :3, 3]
    )
    mask = valid & torch.isfinite(points).all(-1)
    if roi is not None:
        bounds = points.new_tensor(roi)
        mask &= ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all(-1)
    points = points[mask]
    if len(points) > max_points:
        indices = torch.linspace(0, len(points) - 1, max_points, device=depth.device).round().long()
        points = points[indices]
    return points


def postprocess(prediction, inputs, config):
    result = depth_on_image_grid(prediction, inputs)
    if result is None:
        return {}
    refs = result["reference_indices"]
    settings = config.get("evaluation", {})
    result["points"] = [
        depth_points(
            result["depth"][b],
            result["depth_valid"][b],
            inputs["K"][b, refs],
            inputs["T_cam_from_rig"][b, refs],
            settings.get("roi"),
            settings.get("max_surface_points", 8192),
        )
        for b in range(len(result["depth"]))
    ]
    return result
