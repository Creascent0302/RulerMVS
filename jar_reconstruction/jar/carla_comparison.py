"""CARLA 的固定评价合同；网络只收到 inputs，标签仅用于离线评价。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .geometry import pixel_grid, unproject
from .metrics import nearest_distances, render_surface_depth


def read_json(path):
    return json.loads(Path(path).read_text())


def comparison_dataset(protocol, split, load_targets=True):
    """所有方法使用相同的稳定相机槽位、RGB 缩放和最近邻深度标签。"""
    from .multifocal.data import MultifocalFrameDataset

    data = protocol["data"]
    cfg = {
        "data": {
            "root": data["root"],
            "test_manifest": data["manifests"][split],
            "image_size": data["image_size"],
            "depth_resize_mode": "nearest-exact",
        },
        "camera": {
            "mode": "wide4",
            "wide_ids": data["camera_ids"],
            "tele_ids": ["tele_left", "tele_right"],
            "allow_missing": True,
            "disabled_ids": [],
        },
    }
    if "matching_image_size" in data:
        cfg["data"]["matching_image_size"] = data["matching_image_size"]
    return MultifocalFrameDataset(cfg, "test", load_targets=load_targets)


def gt_reference(inputs, targets, settings):
    """GT 决定固定评价分母；预测不能改变范围、ROI 或有效像素。"""
    ref = settings["reference_index"]
    h, w = inputs["rgb"].shape[-2:]
    if "depth" not in targets:
        empty = inputs["rgb"].new_zeros(h, w)
        return empty, torch.zeros_like(empty, dtype=torch.bool)
    depth = targets["depth"][0, ref].float()
    valid = targets.get("depth_valid", targets["depth"] > 0)[0, ref].clone()
    valid &= inputs["view_mask"][0, ref] & inputs["image_mask"][0, ref]
    near, far = settings["depth_range"]
    valid &= torch.isfinite(depth) & (depth >= near) & (depth <= far)
    pixels = pixel_grid(h, w, device=depth.device)
    xyz, _ = unproject(pixels, depth, inputs["K"][0, ref], inputs["T_cam_from_rig"][0, ref])
    bounds = depth.new_tensor(settings["roi"])
    valid &= ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(-1)
    return depth, valid


def depth_edges(depth, valid):
    """深度跳变两侧；只是几何边界，不等价于小物体实例。"""
    edge = torch.zeros_like(valid)
    for axis in (-1, -2):
        left = [slice(None)] * depth.ndim
        right = [slice(None)] * depth.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        jump = (depth[left] - depth[right]).abs()
        threshold = 0.1 + 0.02 * torch.minimum(depth[left], depth[right])
        pair = valid[left] & valid[right] & (jump > threshold)
        edge[left] |= pair
        edge[right] |= pair
    return edge


def depth_counts(prediction, accepted, target, valid, thresholds):
    """交集误差与全 GT 分母成功率同时保留，拒绝全部预测不能获得高分。"""
    finite = torch.isfinite(prediction) & (prediction > 0)
    selected = valid & accepted & finite
    error = (prediction[selected] - target[selected]).abs().double()
    ratio = torch.maximum(
        prediction[selected] / target[selected], target[selected] / prediction[selected]
    )
    counts = {
        "gt_pixels": int(valid.sum()),
        "accepted_pixels": int(selected.sum()),
        "absolute_sum": float(error.sum()),
        "square_sum": float(error.square().sum()),
        "relative_sum": float((error / target[selected]).sum()),
        "declared_nonfinite_pixels": int((accepted & valid & ~finite).sum()),
    }
    for threshold in thresholds:
        counts[f"absolute_hits_{threshold:g}m"] = int((error < threshold).sum())
    for power in (1, 2, 3):
        counts[f"delta{power}_hits"] = int((ratio < 1.25**power).sum())
    return counts


def finish_counts(counts, thresholds):
    n, total = counts["accepted_pixels"], counts["gt_pixels"]
    result = {
        "counts": counts,
        "MAE_m": counts["absolute_sum"] / n if n else None,
        "RMSE_m": math.sqrt(counts["square_sum"] / n) if n else None,
        "AbsRel": counts["relative_sum"] / n if n else None,
        "coverage": n / total if total else None,
        "missing_fraction": 1 - n / total if total else None,
    }
    for threshold in thresholds:
        hits = counts[f"absolute_hits_{threshold:g}m"]
        result[f"success_{threshold:g}m_all_gt"] = hits / total if total else None
        result[f"wrong_accept_{threshold:g}m_of_accepted"] = (n - hits) / n if n else None
        result[f"wrong_accept_{threshold:g}m_all_gt"] = (n - hits) / total if total else None
    for power in (1, 2, 3):
        result[f"delta{power}_all_gt"] = counts[f"delta{power}_hits"] / total if total else None
    return result


def evaluate_depth(prediction, accepted, target, valid, settings):
    thresholds = settings["absolute_thresholds_m"]
    masks = {"all": valid, "depth_boundary": valid & depth_edges(target, valid)}
    for index, (near, far) in enumerate(settings["distance_bands_m"]):
        last = index == len(settings["distance_bands_m"]) - 1
        masks[f"{near:g}_{far:g}m"] = (
            valid & (target >= near) & ((target <= far) if last else (target < far))
        )
    return {
        name: finish_counts(
            depth_counts(prediction, accepted, target, mask, thresholds), thresholds
        )
        for name, mask in masks.items()
    }


def _image_depth(depth, valid, depth_k, refs, inputs):
    from official_methods.postprocess import depth_on_image_grid

    return depth_on_image_grid(
        {"depth": depth, "depth_valid": valid, "depth_K": depth_k, "reference_indices": refs},
        inputs,
    )


def reference_query_depth(scene, inputs, reference):
    """末级未融合射线锚点深度；其他相机的可见背景不投到该参考图中。"""
    if not scene.get("stages"):
        return None
    stage = scene["stages"][-1]
    aux = stage.get("aux", {})
    if not all(key in aux for key in ("ref_index", "ref_uv", "reference_depth")):
        return None
    uv, depth = aux["ref_uv"], aux["reference_depth"]
    height, width = inputs["rgb"].shape[-2:]
    selected = stage["valid"] & (stage["state"] == 1) & (aux["ref_index"] == reference)
    selected &= torch.isfinite(uv).all(-1) & torch.isfinite(depth) & (depth > 0)
    selected &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
    selected &= (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    selected &= inputs["view_mask"][0, reference]
    xy = uv.nan_to_num().round().long()
    x, y = xy[:, 0].clamp(0, width - 1), xy[:, 1].clamp(0, height - 1)
    selected &= inputs["image_mask"][0, reference, y, x]
    raster = depth.new_full((height * width,), torch.inf)
    raster.scatter_reduce_(
        0, (y * width + x)[selected], depth[selected], reduce="amin", include_self=True
    )
    raster = raster.reshape(height, width)
    valid = torch.isfinite(raster)
    return {
        "depth": torch.where(valid, raster, 0),
        "valid": valid,
        "definition": (
            "最后解码阶段、融合前、仅本参考相机的有效表面射线锚点；"
            "aux.ref_uv/reference_depth四舍五入单像素并保留最近Z，"
            "无GT过滤、无跨相机投影、无补洞；适合相同参考UV对照，"
            "仍是稀疏诊断而非全图深度，不替代最终重建。"
        ),
    }


def prediction_streams(outputs, inputs, method, model_cfg, reference=2):
    """每种表示单列。自研 coarse 仅作诊断，不代替最终稀疏表面。"""
    streams = {}
    if "depth" in outputs and "reference_indices" in outputs:
        from official_methods.postprocess import depth_on_image_grid

        mapped = depth_on_image_grid(outputs, inputs)
        refs = list(mapped["reference_indices"])
        if reference in refs:
            position = refs.index(reference)
            streams["native_depth"] = {
                "depth": mapped["depth"][0, position],
                "valid": mapped["depth_valid"][0, position],
                "definition": "原生深度按 depth_K 映射到共同图像；双线性重采样，不做几何筛选。",
            }
    elif "dense_depth" in outputs or "coarse" in outputs:
        dense = "dense_depth" in outputs
        depth = outputs["dense_depth"] if dense else outputs["coarse"]["depth"]
        stride = int(
            outputs.get("depth_stride", 1 if dense else outputs["coarse"].get("stride", 16))
        )
        depth_k = inputs["K"].clone()
        depth_k[..., :2, :] /= stride
        refs = list(range(depth.shape[1]))
        finite = torch.isfinite(depth) & (depth > 0)
        # Cas 的 dense_valid 是适配层几何过滤，原论文网络的深度另行保留。
        accepted = outputs.get("dense_valid", finite) if dense else finite
        native_valid = finite if method == "cas_mvsnet" else accepted
        mapped = _image_depth(depth, native_valid, depth_k, refs, inputs)
        name = "native_depth" if dense else "coarse_depth_diagnostic"
        streams[name] = {
            "depth": mapped["depth"][0, reference],
            "valid": mapped["depth_valid"][0, reference],
            "definition": "原生对齐网格深度按标定双线性重采样；coarse 仅为中间层诊断。",
        }
        if dense and method == "mf_et_jar":
            streams[name]["definition"] = (
                "独立稠密解码头的全图深度假设，包含学习先验；valid只表示参考图像有效，"
                "没有按置信度或多视图验证结果缩小评测总体，不等同于已观测表面。"
            )
            if "dense_observed" in outputs:
                verified = _image_depth(
                    depth, outputs["dense_observed"] & finite, depth_k, refs, inputs
                )
                streams["verified_dense_diagnostic"] = {
                    "depth": verified["depth"][0, reference],
                    "valid": verified["depth_valid"][0, reference],
                    "definition": "稠密预测经真实外观、视差与跨视图深度一致性筛选；另列覆盖率，不能代替全图结果。",
                }
        if method == "cas_mvsnet":
            filtered = _image_depth(depth, accepted, depth_k, refs, inputs)
            streams["geometric_filter_diagnostic"] = {
                "depth": filtered["depth"][0, reference],
                "valid": filtered["depth_valid"][0, reference],
                "definition": "项目适配器的跨视图过滤，独立于 CasMVSNet 原生网络。",
            }
    if outputs.get("scenes"):
        depth, valid = render_surface_depth(outputs["scenes"][0], inputs)
        streams["final_point_splat"] = {
            "depth": depth[reference],
            "valid": valid[reference],
            "definition": (
                "最终有效表面点投影到指定参考图，四舍五入单像素、最近Z缓冲；"
                "不插值补洞，不按GT删点。这是稀疏点首命中渲染诊断；"
                "缺少遮挡前景时，真实背景点也可能被记为深度错误，"
                "不能直接解释为纯三维定位误差或与稠密全图结果直接排名。"
            ),
        }
        if method == "mf_et_jar":
            queries = reference_query_depth(outputs["scenes"][0], inputs, reference)
            if queries is not None:
                streams["unfused_reference_query_depth"] = queries
    return streams


def aggregate_depth(records, settings):
    """像素加权聚合；无标签帧不伪造深度值，完整与缺相机由外层分表。"""
    if not records:
        return {}
    expected = set(records[0].get("streams", {}))
    if not expected:
        raise ValueError("深度评测stream集合为空；不能把缺失输出当作无GT帧。")
    expected_groups = {name: set(records[0]["streams"][name]) for name in expected}
    for index, row in enumerate(records):
        if set(row.get("streams", {})) != expected:
            raise ValueError(f"第{index}帧的深度stream集合不同；缺失预测不能删除GT分母。")
        for name in expected:
            if not expected_groups[name] or set(row["streams"][name]) != expected_groups[name]:
                raise ValueError(f"第{index}帧的{name}分组集合为空或不同；不能跳过评价分母。")
    result = {}
    for name in sorted(expected):
        result[name] = {}
        for group in sorted(expected_groups[name]):
            values = [row["streams"][name][group]["counts"] for row in records]
            sums = {key: sum(value[key] for value in values) for key in values[0]}
            result[name][group] = finish_counts(sums, settings["absolute_thresholds_m"])
    return result


def surface_diagnostics(points, inputs, target, valid, settings):
    """参考相机首命中范围内的点集诊断，查询端抽样、目标集合不抽稀。"""
    from .geometry import project

    ref = settings["reference_index"]
    h, w = target.shape
    pixels = pixel_grid(h, w, device=target.device)
    gt, _ = unproject(
        pixels[valid], target[valid], inputs["K"][0, ref], inputs["T_cam_from_rig"][0, ref]
    )
    bounds = points.new_tensor(settings["roi"])
    points = points[
        torch.isfinite(points).all(-1)
        & ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all(-1)
    ]
    uv, _, projected, _ = project(
        points,
        inputs["K"][0, ref : ref + 1],
        inputs["T_cam_from_rig"][0, ref : ref + 1],
        (h, w),
    )
    xy = uv[0].round().long()
    x, y = xy[:, 0].clamp(0, w - 1), xy[:, 1].clamp(0, h - 1)
    eligible = projected[0] & valid[y, x]
    ignored = int((~eligible).sum())
    points = points[eligible]
    cap = settings.get("max_surface_queries", 4096)

    def sample(values):
        if len(values) <= cap:
            return values
        return values[torch.linspace(0, len(values) - 1, cap, device=values.device).long()]

    to_gt = nearest_distances(sample(points), gt)
    to_pred = nearest_distances(sample(gt), points)
    result = {"pred_points": len(points), "gt_points": len(gt), "outside_labeled_rays": ignored}
    for threshold in settings["surface_thresholds_m"]:
        precision = float((to_gt < threshold).float().mean()) if len(to_gt) else 0.0
        recall = float((to_pred < threshold).float().mean()) if len(to_pred) else 0.0
        result[f"Fscore_{threshold:g}m"] = (
            2 * precision * recall / max(precision + recall, 1e-12) if len(gt) else None
        )
    return result


def raw_depth_diagnostic(frame, root, camera_id, image_size):
    """原始标签边界库存；不把下采样边界或二维边缘称为小物体检测成绩。"""
    import numpy as np
    from .data import to_optical_depth

    camera = next((c for c in frame["cameras"] if c["camera_id"] == camera_id), None)
    label = camera.get("targets", {}) if camera else {}
    if not label.get("depth"):
        return {"available": False}
    if label.get("depth_encoding", "metric_npy") != "metric_npy":
        return {
            "available": False,
            "reason": "原始边界库存当前只支持 metric_npy；主指标不受影响。",
        }
    depth = torch.from_numpy(np.load(Path(root) / label["depth"], allow_pickle=False)).float()
    depth *= float(label.get("depth_scale", 1))
    valid = torch.isfinite(depth) & (depth > 0) & (depth < label.get("max_valid_depth", 1000))
    depth = to_optical_depth(depth, torch.tensor(camera["K"]), label["depth_type"])
    if label.get("depth_valid"):
        valid &= torch.from_numpy(
            np.load(Path(root) / label["depth_valid"], allow_pickle=False)
        ).bool()
    edge = depth_edges(depth, valid)
    picked = F.interpolate(edge[None, None].float(), image_size, mode="nearest-exact")[0, 0].bool()
    any_edge = F.adaptive_max_pool2d(edge[None, None].float(), image_size)[0, 0].bool()
    return {
        "available": True,
        "raw_image_size": list(depth.shape),
        "raw_valid_pixels": int(valid.sum()),
        "raw_depth_boundary_pixels": int(edge.sum()),
        "resized_cells_with_raw_boundary": int(any_edge.sum()),
        "nearest_samples_on_raw_boundary": int(picked.sum()),
        "note": "原始深度所有有效距离的边界库存，未按预测筛选；不是模型精度或小物体召回。",
    }
