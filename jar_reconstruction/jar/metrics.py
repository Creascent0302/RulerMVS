"""Metrics explicitly report missing/unknown predictions and depth coverage."""

from __future__ import annotations
import torch
from .geometry import project, unproject, pixel_grid
from .losses import target_cloud, label_query_points, FREE, SURFACE, UNKNOWN


SURFACE_METRIC_VERSION = "query_sample_full_reference_v2"


@torch.no_grad()
def nearest_distances(a, b, chunk=512):
    if not len(a):
        return a.new_empty(0)
    if not len(b):
        return a.new_full((len(a),), float("inf"))
    result = []
    for x in a.split(chunk):
        best = x.new_full((len(x),), float("inf"))
        for y in b.split(chunk):
            best = torch.minimum(best, torch.cdist(x, y).amin(-1))
        result.append(best)
    return torch.cat(result)


@torch.no_grad()
def surface_fscore(predicted, target, threshold=0.1, max_points=4096):
    """v2：仅限制距离查询点，最近邻参考保留对方完整点集。

    仍是按输入顺序等距抽样的查询近似，不能当作全量精确F-score；返回查询
    与完整参考的数量。缺少GT时分数未定义；有GT而没有预测时得分为0。
    """

    def cap(p):
        if len(p) > max_points:
            return p[torch.linspace(0, len(p) - 1, max_points, device=p.device).long()]
        return p

    p, t = cap(predicted), cap(target)
    precision = float((nearest_distances(p, target) < threshold).float().mean()) if len(p) else 0.0
    recall = float((nearest_distances(t, predicted) < threshold).float().mean()) if len(t) else 0.0
    return {
        "surface_precision": precision if len(target) else None,
        "surface_recall": recall if len(target) else None,
        "surface_fscore": (
            2 * precision * recall / max(precision + recall, 1e-12) if len(target) else None
        ),
        "surface_pred_samples": len(p),
        "surface_gt_samples": len(t),
        "surface_pred_reference_points": len(predicted),
        "surface_gt_reference_points": len(target),
        "surface_distance_threshold_m": threshold,
    }


@torch.no_grad()
def render_surface_depth(scene, inputs, batch_index=0):
    """Nearest-pixel point splat z-buffer; coverage is essential for this sparse output."""
    n = inputs["rgb"].shape[1]
    h, w = inputs["rgb"].shape[-2:]
    depth = inputs["rgb"].new_full((n, h * w), float("inf"))
    points = scene["points"][scene["valid"] & (scene["state"] == SURFACE)]
    if len(points):
        uv, z, valid, _ = project(
            points,
            inputs["K"][batch_index],
            inputs["T_cam_from_rig"][batch_index],
            (h, w),
            inputs["view_mask"][batch_index],
            inputs["image_mask"][batch_index] if "image_mask" in inputs else None,
        )
        u = uv[..., 0].round().long().clamp(0, w - 1)
        v = uv[..., 1].round().long().clamp(0, h - 1)
        for camera in range(n):
            mask = valid[camera]
            depth[camera].scatter_reduce_(
                0,
                (v[camera, mask] * w + u[camera, mask]),
                z[camera, mask],
                reduce="amin",
                include_self=True,
            )
    depth = depth.reshape(n, h, w)
    valid = torch.isfinite(depth)
    return torch.where(valid, depth, torch.zeros_like(depth)), valid


@torch.no_grad()
def evaluate_dense_depth(outputs, inputs, targets, roi=None, observed_only=False, depth_range=None):
    """原生稠密深度独立计分；输入有效性与已观测空间证据严格分开。"""
    if "dense" not in outputs:
        return {}
    dense = outputs["dense"]
    prediction = dense["depth"]
    input_valid = inputs["view_mask"][..., None, None].expand_as(prediction)
    if "image_mask" in inputs:
        input_valid = input_valid & inputs["image_mask"]
    predicted = dense.get("valid", input_valid) & input_valid
    predicted &= torch.isfinite(prediction) & (prediction > 0)
    result = {
        "dense_depth_mae_m": None,
        "dense_depth_rmse_m": None,
        "dense_depth_absrel": None,
        "dense_depth_delta1": None,
        "dense_depth_coverage": None,
        "dense_depth_missing_fraction": None,
        "dense_labeled_depth_pixels": 0,
        "dense_predicted_depth_pixels": 0,
    }
    observed = outputs.get("dense_observed", dense.get("observed"))
    if observed is not None:
        observed_count = int((observed.bool() & input_valid).sum())
        result["dense_observed_fraction"] = observed_count / max(int(input_valid.sum()), 1)
    if "depth" not in targets:
        return result
    target = targets["depth"].float()
    if prediction.shape != target.shape:
        raise ValueError("dense评测要求stride=1且深度图与原输入/标签同形状")
    valid = targets.get("depth_valid", target > 0) & input_valid
    valid &= torch.isfinite(target) & (target > 0)
    if depth_range is not None:
        valid &= (target >= depth_range[0]) & (target <= depth_range[1])
    if observed_only and roi is not None:
        grid = pixel_grid(*target.shape[-2:], device=target.device)
        bounds = target.new_tensor(roi)
        for b in range(len(target)):
            for n in range(target.shape[1]):
                pixels = valid[b, n].clone()
                if pixels.any():
                    xyz, _ = unproject(
                        grid[pixels],
                        target[b, n][pixels],
                        inputs["K"][b, n],
                        inputs["T_cam_from_rig"][b, n],
                    )
                    inside = ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(-1)
                    valid[b, n][pixels] = inside
    selected = predicted & valid
    labeled_count, predicted_count = int(valid.sum()), int(selected.sum())
    result["dense_labeled_depth_pixels"] = labeled_count
    result["dense_predicted_depth_pixels"] = predicted_count
    if labeled_count:
        result["dense_depth_coverage"] = predicted_count / labeled_count
        result["dense_depth_missing_fraction"] = 1 - predicted_count / labeled_count
    if predicted_count:
        actual, expected = prediction[selected].double(), target[selected].double()
        error = actual - expected
        relative_ratio = torch.maximum(actual / expected, expected / actual)
        result.update(
            dense_depth_mae_m=float(error.abs().mean()),
            dense_depth_rmse_m=float(error.square().mean().sqrt()),
            dense_depth_absrel=float((error.abs() / expected).mean()),
            dense_depth_delta1=float((relative_ratio < 1.25).double().mean()),
        )
    return result


@torch.no_grad()
def evaluate_batch(
    outputs,
    inputs,
    targets,
    threshold=0.1,
    max_points=4096,
    roi=None,
    observed_only=False,
    dense_depth_range=None,
):
    result = evaluate_dense_depth(outputs, inputs, targets, roi, observed_only, dense_depth_range)
    scenes = outputs.get("scenes", [])
    if not scenes:
        result.update(
            frames=inputs["rgb"].shape[0],
            evaluated_nodes=0,
            has_depth_labels=int(result.get("dense_labeled_depth_pixels", 0) > 0),
            surface_precision=None,
            surface_recall=None,
            surface_fscore=None,
            unknown_fraction=None,
        )
        return result
    if "depth" not in targets:
        states = [scene["state"] for scene in scenes]
        count = sum(len(x) for x in states)
        result.update(
            {
                "frames": len(states),
                "has_depth_labels": 0,
                "evaluated_nodes": count,
                "unknown_fraction": sum(int((x == UNKNOWN).sum()) for x in states) / max(count, 1),
            }
        )
        return result
    sums = {
        "absolute": 0.0,
        "squared": 0.0,
        "depth_n": 0,
        "gt_n": 0,
        "unknown": 0,
        "nodes": 0,
        "surface_tp": 0,
        "surface_union": 0,
        "known": 0,
        "correct": 0,
        "wrong_free": 0,
        "gt_surface": 0,
        "free_unknown": 0,
        "gt_unknown": 0,
        "pred_in_labeled_views": 0,
        "pred_without_depth_label": 0,
        "pred_outside_gt_roi": 0,
    }
    fmetrics = []
    for b, scene in enumerate(scenes):
        predicted, mask = render_surface_depth(scene, inputs, b)
        gt = targets["depth"][b]
        gt_valid = (
            targets.get("depth_valid", targets["depth"] > 0)[b]
            & inputs["view_mask"][b, :, None, None]
        )
        if "image_mask" in inputs:
            gt_valid &= inputs["image_mask"][b]
        # 报告深度误差看不到的预测，不把缺标签误称为错误或天空。
        # 仅统计至少含一个有效深度像素的相机；完全缺标签的相机不混入。
        depth_labeled = gt_valid.clone()
        labeled_view = depth_labeled.flatten(1).any(-1)[:, None, None]
        sums["pred_in_labeled_views"] += int((mask & labeled_view).sum())
        sums["pred_without_depth_label"] += int((mask & labeled_view & ~depth_labeled).sum())
        if observed_only and roi is not None:
            grid = pixel_grid(*gt.shape[-2:], device=gt.device, dtype=gt.dtype)
            bounds = gt.new_tensor(roi)
            gt_valid = gt_valid.clone()
            for n in range(len(gt)):
                pixels = gt_valid[n].clone()
                if not pixels.any():
                    continue
                xyz, _ = unproject(
                    grid[pixels], gt[n][pixels], inputs["K"][b, n], inputs["T_cam_from_rig"][b, n]
                )
                gt_valid[n][pixels] = ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(-1)
        sums["pred_outside_gt_roi"] += int((mask & depth_labeled & ~gt_valid).sum())
        ok = mask & gt_valid
        err = predicted[ok] - gt[ok]
        sums["absolute"] += float(err.abs().sum())
        sums["squared"] += float(err.square().sum())
        sums["depth_n"] += int(ok.sum())
        sums["gt_n"] += int(gt_valid.sum())
        p, _ = target_cloud(inputs, targets, b)
        prediction = scene["points"][scene["valid"] & (scene["state"] == SURFACE)]
        if roi is not None:
            bounds = p.new_tensor(roi)
            p = p[((p >= bounds[:, 0]) & (p <= bounds[:, 1])).all(-1)]
            prediction = prediction[
                ((prediction >= bounds[:, 0]) & (prediction <= bounds[:, 1])).all(-1)
            ]
        if observed_only:
            # 只有depth相机覆盖的已观测free/命中带可评精度；不可把未标注视野算错。
            observed = label_query_points(prediction, inputs, targets, b, threshold)["observed"]
            prediction = prediction[observed]
        if len(p):
            fmetrics.append(surface_fscore(prediction, p, threshold, max_points))
        label = label_query_points(scene["points"], inputs, targets, b)["state"]
        state = scene["state"]
        known = label != UNKNOWN
        sums["unknown"] += int((state == UNKNOWN).sum())
        sums["nodes"] += len(state)
        sums["known"] += int(known.sum())
        sums["correct"] += int(((state == label) & known).sum())
        sums["surface_tp"] += int(((label == SURFACE) & (state == SURFACE)).sum())
        union = (label == SURFACE) | (state == SURFACE)
        if observed_only:
            union &= known
        sums["surface_union"] += int(union.sum())
        sums["wrong_free"] += int(((label == SURFACE) & (state == FREE)).sum())
        sums["gt_surface"] += int((label == SURFACE).sum())
        sums["free_unknown"] += int(((label == UNKNOWN) & (state == FREE)).sum())
        sums["gt_unknown"] += int((label == UNKNOWN).sum())
    result.update(
        depth_mae_m=sums["absolute"] / sums["depth_n"] if sums["depth_n"] else None,
        depth_rmse_m=(sums["squared"] / sums["depth_n"]) ** 0.5 if sums["depth_n"] else None,
        depth_coverage=sums["depth_n"] / sums["gt_n"] if sums["gt_n"] else None,
        depth_missing_fraction=1 - sums["depth_n"] / sums["gt_n"] if sums["gt_n"] else None,
        unknown_fraction=sums["unknown"] / max(sums["nodes"], 1),
        output_element_state_accuracy=sums["correct"] / max(sums["known"], 1),
        output_element_surface_iou=sums["surface_tp"] / max(sums["surface_union"], 1),
        output_element_wrong_free_on_surface=sums["wrong_free"] / max(sums["gt_surface"], 1),
        output_element_free_on_unknown=sums["free_unknown"] / max(sums["gt_unknown"], 1),
        evaluated_nodes=sums["nodes"],
        labeled_depth_pixels=sums["gt_n"],
        predicted_depth_pixels=sums["depth_n"],
        predicted_pixels_in_depth_labeled_views=sums["pred_in_labeled_views"],
        predicted_pixels_without_depth_label=sums["pred_without_depth_label"],
        predicted_pixels_outside_gt_roi=sums["pred_outside_gt_roi"],
        frames=len(scenes),
    )
    if observed_only:
        result["has_depth_labels"] = int(sums["gt_n"] > 0)
        result["surface_observed_only"] = True
    if not fmetrics:
        result.update(
            surface_precision=None,
            surface_recall=None,
            surface_fscore=None,
            surface_gt_samples=0,
        )
    for key in fmetrics[0] if fmetrics else []:
        result[key] = sum(v[key] for v in fmetrics) / len(fmetrics)
    return result
