"""只由已确认参考射线建立稀疏深度证据，不读取标签、不补全未观测像素。

空闲判断条件于“预测的表面命中正确”。它不是绝对安全证书；错误表面、
遮挡误匹配和最近像素读取的有限角分辨率仍可能产生错误空闲判断。
"""

from __future__ import annotations

import torch

from .geometry import camera_centers, project


def _masks(inputs, b):
    n, _, h, w = inputs["rgb"][b].shape
    device = inputs["rgb"].device
    view = inputs.get("view_mask")
    view = torch.ones(n, device=device, dtype=torch.bool) if view is None else view[b].bool()
    image = inputs.get("image_mask")
    image = (
        torch.ones(n, h, w, device=device, dtype=torch.bool) if image is None else image[b].bool()
    )
    return view, image


@torch.no_grad()
def build_ray_evidence(stage, inputs, b, roi):
    """把融合前 stage 的已确认参考锚点写成 [N,H,W] 第一命中图。

    必须使用与 ``aux.ref_index/ref_uv`` 一一对应的融合前点。锚点需在原图
    整数像素中心的 1e-3 像素内，点的精确投影需在同一锚点的 1e-3 像素内。
    不把这些点额外投到其他相机作为可见命中。其他视图只用于检查宣称的
    至少双视图支持是否在几何上可能，尤其拒绝纯旋转和退化共线观察。
    """
    rgb = inputs["rgb"][b]
    n, _, h, w = rgb.shape
    device = rgb.device
    bounds = torch.as_tensor(roi, device=device, dtype=torch.float32)
    if (
        bounds.shape != (3, 2)
        or not bool(torch.isfinite(bounds).all())
        or not bool((bounds[:, 1] > bounds[:, 0]).all())
    ):
        raise ValueError("roi must contain three finite increasing intervals")
    points = stage["points"]
    q = len(points)
    aux = stage.get("aux", {})
    if "ref_index" not in aux or "ref_uv" not in aux:
        raise ValueError("ray evidence requires unfused reference-index and pixel auxiliaries")
    refs, anchors = aux["ref_index"], aux["ref_uv"]
    if points.shape != (q, 3) or refs.shape != (q,) or anchors.shape != (q, 2):
        raise ValueError("points and reference auxiliaries must have the same unfused query count")
    for name in ("valid", "state", "confidence", "support"):
        if stage[name].shape != (q,):
            raise ValueError(f"stage.{name} must have shape [Q]")
    if refs.dtype not in (torch.int32, torch.int64):
        raise ValueError("reference indices must be integer tensors")
    shape = (n, h, w)
    depth_map = torch.full((n * h * w,), float("inf"), device=device, dtype=torch.float32)
    confidence_map = torch.zeros_like(depth_map)
    support_map = torch.zeros(n * h * w, device=device, dtype=torch.long)
    view_mask, image_mask = _masks(inputs, b)
    K, T = inputs["K"][b], inputs["T_cam_from_rig"][b]
    anchor_tolerance = 1e-3
    candidates = stage["valid"].bool() & (stage["state"] == 1) & (stage["support"] >= 2)
    candidates &= torch.isfinite(stage["confidence"]) & (stage["confidence"] > 0)
    candidates &= torch.isfinite(points).all(-1) & torch.isfinite(anchors).all(-1)
    candidates &= (refs >= 0) & (refs < n)
    candidates &= ((points >= bounds[:, 0]) & (points < bounds[:, 1])).all(-1)
    selected = candidates.nonzero().flatten()
    all_keys, all_depths, all_confidences, all_supports = [], [], [], []
    with torch.autocast(device_type=device.type, enabled=False):
        centers = camera_centers(T).float()
        for indices in selected.split(4096):
            X = points[indices].float()
            reference = refs[indices].long()
            anchor = anchors[indices].float()
            uv, depth, valid, J = project(X, K, T, (h, w), view_mask, image_mask)
            rows = torch.arange(len(indices), device=device)
            ref_uv, ref_depth = uv[reference, rows], depth[reference, rows]
            pixel = anchor.round().long()
            ok = valid[reference, rows]
            ok &= (ref_uv - anchor).abs().amax(-1) <= anchor_tolerance
            ok &= (anchor - pixel).abs().amax(-1) <= anchor_tolerance
            ok &= (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
            # dX/dZ along the already-confirmed reference ray. Only geometric
            # non-degeneracy is checked here; J never proves correct matching.
            ray = (X - centers[reference]) / ref_depth.clamp_min(1e-6)[:, None]
            g = (J.float() * ray[None, :, None]).sum(-1).norm(dim=-1)
            independent = (torch.arange(n, device=device)[:, None] != reference[None]) & (g > 1e-5)
            possible_support = (valid & independent).sum(0) + 1
            ok &= possible_support >= 2
            pixel = pixel[ok]
            key = reference[ok] * h * w + pixel[:, 1] * w + pixel[:, 0]
            all_keys.append(key)
            all_depths.append(ref_depth[ok].float())
            all_confidences.append(stage["confidence"][indices][ok].float().clamp(0, 1))
            all_supports.append(
                torch.minimum(stage["support"][indices][ok].long(), possible_support[ok])
            )
        if all_keys:
            key, z, confidence, support = map(
                torch.cat, (all_keys, all_depths, all_confidences, all_supports)
            )
            depth_map.scatter_reduce_(0, key, z, reduce="amin", include_self=True)
            nearest = z == depth_map[key]
            # Max never adds repeated-query support counts. Attributes belong
            # only to the nearest hit, not a confident occluded rear surface.
            confidence_map.scatter_reduce_(
                0, key[nearest], confidence[nearest], reduce="amax", include_self=True
            )
            support_map.scatter_reduce_(
                0, key[nearest], support[nearest], reduce="amax", include_self=True
            )
    valid_map = torch.isfinite(depth_map) & (support_map >= 2) & (confidence_map > 0)
    depth_map = torch.where(valid_map, depth_map, torch.zeros_like(depth_map))
    return dict(
        depth=depth_map.reshape(shape),
        valid=valid_map.reshape(shape),
        confidence=confidence_map.reshape(shape),
        support=support_map.reshape(shape),
        roi=bounds,
        image_hw=(h, w),
        anchor_tolerance_px=anchor_tolerance,
        query_pixel_tolerance_px=0.5,
        source="verified_reference_queries_only",
        confirmed_input_queries=int(selected.numel()),
        represented_pixels=int(valid_map.sum()),
    )


@torch.no_grad()
def query_ray_evidence(points, evidence, inputs, b, band=0.1):
    """在精确投影的最近原图像素读取预测命中。

    band 是相机光轴深度的米制命中带。读取接受离像素中心最多 0.5 像素
    的点；这是有限像素足迹近似，不代表整个连续视锥内均得到传感证明。
    无命中像素、命中后方、被屏蔽相机/像素及 ROI 外均保留 UNKNOWN。
    任一参考命中带与空闲判断冲突时，优先返回 surface，不返回 free。
    surface 仅是深度证据，最终表面仍应由调用者的实际几何判断。
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [Q,3]")
    rgb = inputs["rgb"][b]
    n, _, h, w = rgb.shape
    for key in ("depth", "valid", "confidence", "support"):
        if evidence[key].shape != (n, h, w):
            raise ValueError("evidence maps must match the original image dimensions")
    band_value = torch.as_tensor(band, device=points.device, dtype=torch.float32)
    if band_value.numel() != 1 or not bool(torch.isfinite(band_value)) or float(band_value) < 0:
        raise ValueError("band must be a nonnegative finite scalar in optical-depth metres")
    view_mask, image_mask = _masks(inputs, b)
    K, T = inputs["K"][b], inputs["T_cam_from_rig"][b]
    bounds = evidence["roi"].to(device=points.device)
    outputs = []
    with torch.autocast(device_type=points.device.type, enabled=False):
        for piece in points.split(4096):
            X = piece.float()
            uv, z, geometrical, _ = project(X, K, T, (h, w), view_mask, image_mask)
            xy = uv.round().long()
            # Exact project valid handles boundaries. Clamp only protects the
            # gather for invalid points and never creates valid observations.
            x = xy[..., 0].clamp(0, w - 1)
            y = xy[..., 1].clamp(0, h - 1)
            camera = torch.arange(n, device=points.device)[:, None]
            depth = evidence["depth"][camera, y, x].float()
            confidence = evidence["confidence"][camera, y, x].float()
            support = evidence["support"][camera, y, x].long()
            reliable = geometrical & evidence["valid"][camera, y, x].bool()
            reliable &= image_mask[camera, y, x] & view_mask[:, None]
            reliable &= (uv - xy).abs().amax(-1) <= evidence.get("query_pixel_tolerance_px", 0.5)
            reliable &= torch.isfinite(depth) & (depth > 0) & (support >= 2)
            reliable &= torch.isfinite(confidence) & (confidence > 0)
            inside = torch.isfinite(X).all(-1) & ((X >= bounds[:, 0]) & (X < bounds[:, 1])).all(-1)
            reliable &= inside[None]
            near = reliable & ((z - depth).abs() <= band_value)
            before = reliable & (z < depth - band_value)
            surface = near.any(0)
            free = before.any(0) & ~surface
            contributing = near | (before & free[None])
            score = torch.where(contributing, confidence.clamp(0, 1), 0).amax(0)
            count = torch.where(contributing, support, 0).amax(0)
            outputs.append((free, surface, free | surface, count, score))
    if not outputs:
        return dict(
            free=torch.zeros(0, device=points.device, dtype=torch.bool),
            surface=torch.zeros(0, device=points.device, dtype=torch.bool),
            known=torch.zeros(0, device=points.device, dtype=torch.bool),
            support=torch.zeros(0, device=points.device, dtype=torch.long),
            confidence=torch.zeros(0, device=points.device, dtype=torch.float32),
        )
    return {
        key: torch.cat([part[i] for part in outputs])
        for i, key in enumerate(("free", "surface", "known", "support", "confidence"))
    }
