"""显式缩放样本；RGB 抗混叠，K 同步半像素变换，深度可选保边插值。"""

import torch
import torch.nn.functional as F
from .geometry import resize_crop_intrinsics


def _aligned_inverse_depth(depth, valid, size, absolute_jump, relative_jump):
    """平滑邻域按目标像素中心插值逆深度，跳变处保留最近邻标签。

    平面上的逆深度随像素坐标线性变化，直接最近邻采样则偏离中心射线。
    阈值是保守的跳变判据：很陡的连续表面也可能被拒绝，并非语义边界检测。
    """
    old_h, old_w = depth.shape[-2:]
    height, width = size
    rows = (torch.arange(height, device=depth.device) + 0.5) * old_h / height - 0.5
    columns = (torch.arange(width, device=depth.device) + 0.5) * old_w / width - 0.5
    y0 = rows.floor().long().clamp(0, old_h - 1)
    x0 = columns.floor().long().clamp(0, old_w - 1)
    y1 = rows.ceil().long().clamp(0, old_h - 1)
    x1 = columns.ceil().long().clamp(0, old_w - 1)
    corners = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
    samples = torch.stack([depth[:, y[:, None], x[None, :]] for y, x in corners], dim=1)
    supported = torch.stack([valid[:, y[:, None], x[None, :]] for y, x in corners], dim=1).all(1)
    minimum = samples.amin(1)
    jump = samples.amax(1) - minimum
    boundary = supported & (jump > absolute_jump + relative_jump * minimum)
    smooth = supported & ~boundary
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    inverse_depth = F.interpolate(
        safe_depth[:, None].reciprocal(), size=size, mode="bilinear", align_corners=False
    )[:, 0]
    return inverse_depth.clamp_min(1e-8).reciprocal(), smooth, boundary


def resize_frame(
    item,
    image_size,
    depth_resize_mode="nearest-exact",
    depth_jump_absolute_m=0.05,
    depth_jump_relative=0.01,
):
    if not image_size:
        return item
    height, width = map(int, image_size)
    if min(height, width) < 1:
        raise ValueError("data.image_size必须为正的[H,W]")
    if depth_resize_mode not in ("nearest-exact", "edge_aware_inverse"):
        raise ValueError("depth_resize_mode must be nearest-exact or edge_aware_inverse")
    if depth_jump_absolute_m < 0 or depth_jump_relative < 0:
        raise ValueError("Depth jump thresholds must be nonnegative")
    inputs = item["inputs"]
    old_h, old_w = inputs["rgb"].shape[-2:]
    if (height, width) == (old_h, old_w):
        return item
    a = dict(inputs)
    targets = {}
    a["rgb"] = F.interpolate(
        inputs["rgb"], size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )
    a["K"] = torch.stack(
        [resize_crop_intrinsics(k, (width / old_w, height / old_h)) for k in inputs["K"]]
    )
    mode = "area" if height <= old_h and width <= old_w else "nearest-exact"
    a["image_mask"] = (
        F.interpolate(inputs["image_mask"][:, None].float(), size=(height, width), mode=mode)[:, 0]
        > 0.999
    )
    for key, value in item["targets"].items():
        shape = value.shape
        result = F.interpolate(
            value.reshape(-1, 1, old_h, old_w).float(), size=(height, width), mode="nearest-exact"
        )
        targets[key] = result.reshape(*shape[:-2], height, width).to(value.dtype)
    if depth_resize_mode == "edge_aware_inverse" and "depth" in targets:
        depth = item["targets"]["depth"].float()
        valid = item["targets"].get("depth_valid", depth > 0)
        valid = valid & torch.isfinite(depth) & (depth > 0) & inputs["image_mask"]
        if "view_mask" in inputs:
            valid &= inputs["view_mask"][:, None, None]
        aligned, smooth, boundary = _aligned_inverse_depth(
            depth, valid, (height, width), depth_jump_absolute_m, depth_jump_relative
        )
        targets["depth"] = torch.where(smooth, aligned, targets["depth"])
        # 诊断标签只进入 targets，不作为推理输入，也不更改原 depth_valid。
        targets["depth_resize_interpolated"] = smooth
        targets["depth_resize_boundary"] = boundary
    if "image_sizes" in inputs:
        a["image_sizes"] = torch.where(
            inputs["view_mask"][:, None], inputs["image_sizes"].new_tensor([height, width]), 0
        )
    meta = dict(item.get("metadata", {}))
    meta["resize"] = {
        "from_hw": [old_h, old_w],
        "to_hw": [height, width],
        "scale_xy": [width / old_w, height / old_h],
        "pixel_mapping": "u_new=(u_old+0.5)*sx-0.5; v同理",
        "depth_resize_mode": depth_resize_mode,
    }
    if depth_resize_mode == "edge_aware_inverse":
        meta["resize"].update(
            depth_jump_absolute_m=depth_jump_absolute_m,
            depth_jump_relative=depth_jump_relative,
        )
    return {"inputs": a, "targets": targets, "metadata": meta}
