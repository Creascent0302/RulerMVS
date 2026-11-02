"""统一定义的可比较指标；不把缺失预测或未知标签静默删掉。"""

import math
import torch
from jar.metrics import nearest_distances
from .postprocess import depth_points


@torch.no_grad()
def depth_statistics(pred, gt, gt_valid, pred_valid, near, far):
    mask = gt_valid & torch.isfinite(gt) & (gt >= near) & (gt <= far)
    usable = mask & pred_valid & torch.isfinite(pred) & (pred > 0)
    delta = pred[usable] - gt[usable]
    ratio = torch.maximum(pred[usable] / gt[usable], gt[usable] / pred[usable])
    return {
        "gt_pixels": int(mask.sum()),
        "predicted_pixels": int(usable.sum()),
        "abs_sum": float(delta.abs().sum()),
        "sq_sum": float(delta.square().sum()),
        "abs_rel_sum": float((delta.abs() / gt[usable]).sum()),
        **{f"delta_{i}_hits": int((ratio < 1.25**i).sum()) for i in (1, 2, 3)},
    }


def finish_depth(s, units="meters"):
    n, total = s["predicted_pixels"], s["gt_pixels"]
    suffix = "m" if units == "meters" else "scene_unit"
    return {
        "AbsRel": s["abs_rel_sum"] / n if n else None,
        f"RMSE_{suffix}": math.sqrt(s["sq_sum"] / n) if n else None,
        f"MAE_{suffix}": s["abs_sum"] / n if n else None,
        "prediction_coverage": n / total if total else None,
        "unknown_or_missing_fraction": 1 - n / total if total else None,
        **{f"delta_{i}": s[f"delta_{i}_hits"] / total if total else None for i in (1, 2, 3)},
        "gt_pixels": total,
        "predicted_pixels": n,
        "error_denominator": "有效预测像素；同时报告覆盖率。delta 分母为全部有效 GT，缺失预测记失败",
    }


@torch.no_grad()
def surface_metrics(pred, target, thresholds, units="meters"):
    suffix = "m" if units == "meters" else "scene_unit"
    a, c = nearest_distances(pred, target), nearest_distances(target, pred)
    accuracy = float(a.mean()) if len(a) and len(target) else None
    completeness = float(c.mean()) if len(c) and len(pred) else None
    out = {
        f"Accuracy_{suffix}": accuracy,
        f"Completeness_{suffix}": completeness,
        f"Chamfer_L1_{suffix}": (
            (accuracy + completeness) / 2
            if accuracy is not None and completeness is not None
            else None
        ),
        "pred_samples": len(pred),
        "gt_samples": len(target),
    }
    for threshold in thresholds:
        # 完全没有可评价 GT 时不能虚构零分；有 GT 而未预测才是零分。
        precision = (
            (float((a < threshold).float().mean()) if len(a) else 0.0) if len(target) else None
        )
        recall = float((c < threshold).float().mean()) if len(c) else None
        out[f"Fscore_{threshold:g}{suffix}"] = (
            2 * precision * recall / max(precision + recall, 1e-12) if len(target) else None
        )
        out[f"precision_{threshold:g}{suffix}"] = precision
        out[f"recall_{threshold:g}{suffix}"] = recall
    return out


@torch.no_grad()
def occupancy_statistics(pred, target, classes=None):
    """同一网格、同一语义映射才允许调用；unknown 预测不会从 GT 分母删除。"""
    for key in ("grid_origin", "voxel_size"):
        if not torch.allclose(
            torch.as_tensor(pred[key]).float().cpu(), torch.as_tensor(target[key]).float().cpu()
        ):
            raise ValueError(f"占据 {key} 不一致，必须先声明并执行独立重采样")
    p, t = pred["state"], target["state"]
    if p.shape != t.shape:
        raise ValueError("占据网格形状不同")
    known = t != 2
    hit, guess = t == 1, p == 1
    intersection = int((hit & guess & known).sum())
    union = int(((hit | guess) & known).sum())
    out = {
        "intersection": intersection,
        "union": union,
        "known_voxels": int(known.sum()),
        "unknown_predictions": int((p == 2).sum()),
        "total_voxels": t.numel(),
        "wrong_free_on_occupied": int(((p == 0) & hit).sum()),
        "gt_occupied": int(hit.sum()),
        "free_on_unknown": int(((p == 0) & (t == 2)).sum()),
        "gt_unknown": int((t == 2).sum()),
    }
    if classes is not None and "semantics" in pred and "semantics" in target:
        out["semantic_intersection"] = []
        out["semantic_union"] = []
        for category in classes:
            gt = hit & (target["semantics"] == category)
            pr = guess & (pred["semantics"] == category)
            eligible = known & (~hit | (target["semantics"] >= 0))
            out["semantic_intersection"].append(int((gt & pr & eligible).sum()))
            out["semantic_union"].append(int(((gt | pr) & eligible).sum()))
    return out


class Evaluator:
    def __init__(self, config):
        self.settings = config.get("evaluation", {})
        self.units = config.get("data", {}).get("geometry_units", "meters")
        if self.units not in ("meters", "scene_unit"):
            raise ValueError("评价单位必须明确为meters或scene_unit")
        if self.units == "scene_unit" and "surface_thresholds_scene_unit" not in self.settings:
            raise ValueError("scene_unit评价需要显式surface_thresholds_scene_unit，不能沿用米制阈值")
        self.thresholds = self.settings.get(
            "surface_thresholds_scene_unit" if self.units == "scene_unit" else "surface_thresholds_m",
            [0.05, 0.1, 0.2],
        )
        self.depth_sums, self.surfaces, self.occupancies = {}, [], []

    @torch.no_grad()
    def update(self, prediction, converted, inputs, targets):
        settings = self.settings
        if "depth" in converted and "depth" in targets:
            refs = converted["reference_indices"]
            gt = targets["depth"][:, refs]
            valid = targets.get("depth_valid", targets["depth"] > 0)[:, refs].clone()
            valid &= inputs["image_mask"][:, refs] & inputs["view_mask"][:, refs, None, None]
            near, far = settings.get("depth_range", [2.0, 80.0])
            stats = depth_statistics(
                converted["depth"], gt, valid, converted["depth_valid"], near, far
            )
            for key, value in stats.items():
                self.depth_sums[key] = self.depth_sums.get(key, 0) + value
            for b in range(len(gt)):
                observed = valid[b] & (gt[b] >= near) & (gt[b] <= far)
                points = depth_points(
                    converted["depth"][b],
                    converted["depth_valid"][b] & observed,
                    inputs["K"][b, refs],
                    inputs["T_cam_from_rig"][b, refs],
                    settings.get("roi"),
                    settings.get("max_surface_points", 8192),
                )
                gt_points = depth_points(
                    gt[b],
                    observed,
                    inputs["K"][b, refs],
                    inputs["T_cam_from_rig"][b, refs],
                    settings.get("roi"),
                    settings.get("max_surface_points", 8192),
                )
                self.surfaces.append(
                    surface_metrics(
                        points, gt_points, self.thresholds, self.units
                    )
                )
        if "occupancy" in prediction and "occupancy" in targets:
            for pred, target in zip(prediction["occupancy"], targets["occupancy"]):
                if target is not None:
                    self.occupancies.append(
                        occupancy_statistics(pred, target, settings.get("semantic_classes"))
                    )

    def result(self):
        result = {}
        if self.depth_sums:
            result["depth"] = finish_depth(self.depth_sums, self.units)
        if self.surfaces:
            result["surface_per_frame"] = self.surfaces
            result["surface"] = {
                key: (
                    sum(s[key] for s in self.surfaces) / len(self.surfaces)
                    if all(s[key] is not None for s in self.surfaces)
                    else None
                )
                for key in self.surfaces[0]
            }
        if self.occupancies:
            sums = {
                k: sum(s[k] for s in self.occupancies)
                for k in self.occupancies[0]
                if not k.startswith("semantic_")
            }
            result["occupancy"] = {
                "IoU": sums["intersection"] / sums["union"] if sums["union"] else None,
                "unknown_prediction_fraction": sums["unknown_predictions"] / sums["total_voxels"],
                "counts": sums,
            }
            if "semantic_union" in self.occupancies[0]:
                ints = torch.tensor([s["semantic_intersection"] for s in self.occupancies]).sum(0)
                unions = torch.tensor([s["semantic_union"] for s in self.occupancies]).sum(0)
                ious = [float(i / u) if u else None for i, u in zip(ints, unions)]
                present = [i for i in ious if i is not None]
                result["occupancy"].update(
                    per_class_IoU=ious, mIoU=sum(present) / len(present) if present else None
                )
        return result
