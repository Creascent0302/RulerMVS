"""Supervision exclusively consumes targets here, never in inference modules."""

from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import project, unproject, sample_maps

FREE, SURFACE, UNKNOWN = 0, 1, 2


@torch.no_grad()
def normals_from_depth(inputs, targets, absolute_jump=0.05, relative_jump=0.02):
    """Rig-space camera-facing GT normals; discontinuities and holes get zero."""
    depth = targets["depth"]
    valid = targets.get("depth_valid", depth > 0) & (depth > 0) & torch.isfinite(depth)
    valid &= inputs["view_mask"][..., None, None]
    if "image_mask" in inputs:
        valid &= inputs["image_mask"]
    bsize, cameras, h, w = depth.shape
    normals = depth.new_zeros((bsize, cameras, 3, h, w))
    if h < 3 or w < 3:
        return normals
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device),
        torch.arange(w, device=depth.device),
        indexing="ij",
    )
    uv = torch.stack((xx, yy), -1).to(depth.dtype)
    for b in range(bsize):
        for n in range(cameras):
            if not inputs["view_mask"][b, n]:
                continue
            X, _ = unproject(uv, depth[b, n], inputs["K"][b, n], inputs["T_cam_from_rig"][b, n])
            dx = X[1:-1, 2:] - X[1:-1, :-2]
            dy = X[2:, 1:-1] - X[:-2, 1:-1]
            normal = torch.linalg.cross(dx, dy, dim=-1)
            center = (
                -inputs["T_cam_from_rig"][b, n, :3, :3].T @ inputs["T_cam_from_rig"][b, n, :3, 3]
            )
            normal = torch.where(
                ((X[1:-1, 1:-1] - center) * normal).sum(-1, keepdim=True) > 0,
                -normal,
                normal,
            )
            ok = valid[b, n, 1:-1, 1:-1].clone()
            d = depth[b, n, 1:-1, 1:-1]
            for ys, xs in (
                (slice(1, -1), slice(2, None)),
                (slice(1, -1), slice(None, -2)),
                (slice(2, None), slice(1, -1)),
                (slice(None, -2), slice(1, -1)),
            ):
                ok &= valid[b, n, ys, xs] & (
                    (depth[b, n, ys, xs] - d).abs() < absolute_jump + relative_jump * d
                )
            ok &= normal.norm(dim=-1) > 1e-8
            normal = F.normalize(normal, dim=-1, eps=1e-8) * ok[..., None]
            normals[b, n, :, 1:-1, 1:-1] = normal.permute(2, 0, 1)
    return normals


def target_cloud(inputs, targets, batch_index=0, max_points=None):
    """All observed first hits in rig coordinates; no object-box occupancy labels."""
    if "depth" not in targets:
        return inputs["rgb"].new_empty((0, 3)), inputs["rgb"].new_empty((0, 3))
    depths = targets["depth"][batch_index]
    points, normals = [], []
    h, w = depths.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=depths.device),
        torch.arange(w, device=depths.device),
        indexing="ij",
    )
    uv = torch.stack([x, y], -1).to(depths.dtype)
    for n in range(depths.shape[0]):
        if not bool(inputs["view_mask"][batch_index, n]):
            continue
        valid = targets.get("depth_valid", targets["depth"] > 0)[batch_index, n]
        valid = valid & torch.isfinite(depths[n]) & (depths[n] > 0)
        if "image_mask" in inputs:
            valid &= inputs["image_mask"][batch_index, n]
        X, _ = unproject(
            uv[valid],
            depths[n][valid],
            inputs["K"][batch_index, n],
            inputs["T_cam_from_rig"][batch_index, n],
        )
        points.append(X)
        normals.append(
            targets["normals"][batch_index, n].permute(1, 2, 0)[valid]
            if "normals" in targets
            else torch.zeros_like(X)
        )
    if not points:
        return depths.new_empty((0, 3)), depths.new_empty((0, 3))
    p, n = torch.cat(points), torch.cat(normals)
    if max_points and len(p) > max_points:
        index = torch.linspace(0, len(p) - 1, max_points, device=p.device).long()
        p, n = p[index], n[index]
    return p, n


@torch.no_grad()
def label_query_points(points, inputs, targets, batch_index=0, band=0.08):
    """Per-point ray supervision: before first hit free, hit band surface, behind unknown."""
    q = len(points)
    state = torch.full((q,), UNKNOWN, dtype=torch.long, device=points.device)
    if "depth" not in targets or q == 0:
        return {
            "state": state,
            "observed": state != UNKNOWN,
            "surface_point": points.clone(),
            "normal": torch.zeros_like(points),
        }
    h, w = inputs["rgb"].shape[-2:]
    uv, z, valid, _ = project(
        points,
        inputs["K"][batch_index],
        inputs["T_cam_from_rig"][batch_index],
        (h, w),
        inputs["view_mask"][batch_index],
        inputs.get("image_mask", None)[batch_index] if "image_mask" in inputs else None,
    )
    # Nearest first-hit labels prevent interpolation through depth discontinuities.
    ix = uv[..., 0].round().long().clamp(0, w - 1)
    iy = uv[..., 1].round().long().clamp(0, h - 1)
    ni = torch.arange(uv.shape[0], device=points.device)[:, None]
    d = targets["depth"][batch_index][ni, iy, ix]
    dv = targets.get("depth_valid", targets["depth"] > 0)[batch_index][ni, iy, ix]
    valid &= dv & torch.isfinite(d) & (d > 0)
    b = torch.as_tensor(band, device=points.device, dtype=points.dtype)
    free = (valid & (z < d - b)).any(0)
    hit = (valid & ((z - d).abs() <= b)).any(0)
    # Hit takes precedence over other views' free evidence near silhouettes.
    state[free] = FREE
    state[hit] = SURFACE
    distance = (d - z).abs().masked_fill(~valid, float("inf"))
    best = distance.argmin(0)
    qi = torch.arange(q, device=points.device)
    surface = points.clone()
    normal = torch.zeros_like(points)
    for n in range(uv.shape[0]):
        mask = (best == n) & valid[n]
        if mask.any():
            surface[mask], _ = unproject(
                uv[n, mask],
                d[n, mask],
                inputs["K"][batch_index, n],
                inputs["T_cam_from_rig"][batch_index, n],
            )
            if "normals" in targets:
                normal[mask] = targets["normals"][batch_index, n, :, iy[n, mask], ix[n, mask]].T
    return {
        "state": state,
        "observed": state != UNKNOWN,
        "surface_point": surface,
        "normal": normal,
        "valid_ray": valid.any(0),
    }


def _masked_mean(value, mask):
    return (torch.where(mask, value, torch.zeros_like(value))).sum() / mask.sum().clamp_min(1)


def weighted_stage_mean(values, weights):
    """按实际出现的损失标量归一化；默认全1保持原来的计算路径。"""
    stacked = torch.stack(values)
    if all(weight == 1 for weight in weights):
        return stacked.mean()
    total_weight = sum(weights)
    if total_weight == 0:
        return stacked.sum() * 0
    return (stacked * stacked.new_tensor(weights)).sum() / total_weight


def candidate_depth_loss(
    logits, candidates, candidate_valid, target_depth, target_valid, metric_risk=False
):
    """仅监督覆盖真值的有效深度候选；离散选峰也有明确的训练信号。

    最近候选在逆深度域选取，和标定相机的视差采样一致。无匹配、
    无标签或候选范围之外的目标不被强行分到边界类别。
    """
    valid = candidate_valid.bool() & torch.isfinite(candidates) & (candidates > 0)
    finite_candidate = torch.isfinite(candidates) & (candidates > 0)
    safe = torch.nan_to_num(candidates.detach().float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(
        1e-6
    )
    target = target_depth.detach().float()
    lower = safe.masked_fill(~valid, float("inf")).amin(-1)
    upper = safe.masked_fill(~valid, -float("inf")).amax(-1)
    supervised = target_valid & torch.isfinite(target) & (target >= lower) & (target <= upper)
    supervised &= valid.sum(-1) >= 2
    distances = (safe.reciprocal() - target.clamp_min(1e-6).reciprocal()[:, None]).abs()
    # 最近的真实候选若因视野/纹理被屏蔽，不能改监督另一个远处错误深度。
    # 允许同深度重复候选中任意一个有效（局部候选与全局候选可能重合）。
    nearest_distance = distances.masked_fill(~finite_candidate, float("inf")).amin(-1)
    nearest_valid = valid & (distances <= nearest_distance[:, None] + 1e-7)
    supervised &= nearest_valid.any(-1)
    safe_logits = logits.float().masked_fill(~valid, -1e4)
    if metric_risk:
        # 离散选峰前就惩罚大距离错误，梯度可到达全部候选，而非仅胜出峰附近。
        # 对数缩放限制离群点影响；归一化尺度与近细远粗的几何损失一致。
        safe_target = torch.nan_to_num(target, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
        scale = 0.05 + 0.01 * safe_target
        cost = torch.log1p((safe - safe_target[:, None]).abs() / scale[:, None])
        loss = (safe_logits.softmax(-1) * cost).sum(-1)
        return _masked_mean(loss, supervised)
    # 同一物理深度可能同时出现在局部、全局或边界截断候选中。
    # 将所有最近有效候选视为同一正组，避免等价预测互相惩罚。
    positive_logits = safe_logits.masked_fill(~nearest_valid, -1e4)
    loss = torch.logsumexp(safe_logits, -1) - torch.logsumexp(positive_logits, -1)
    return _masked_mean(loss, supervised)


@torch.no_grad()
def visible_semantic_labels(points, inputs, targets, batch_index, classes):
    """Vote only labels of depth-verified views of the same physical surface."""
    h, w = inputs["rgb"].shape[-2:]
    uv, z, valid, _ = project(
        points,
        inputs["K"][batch_index],
        inputs["T_cam_from_rig"][batch_index],
        (h, w),
        inputs["view_mask"][batch_index],
        inputs["image_mask"][batch_index] if "image_mask" in inputs else None,
    )
    x = uv[..., 0].round().long().clamp(0, w - 1)
    y = uv[..., 1].round().long().clamp(0, h - 1)
    ni = torch.arange(uv.shape[0], device=points.device)[:, None]
    depth = targets["depth"][batch_index][ni, y, x]
    depth_valid = targets.get("depth_valid", targets["depth"] > 0)[batch_index][ni, y, x]
    semantic = targets["semantic"][batch_index][ni, y, x]
    valid &= depth_valid & ((z - depth).abs() < 0.08 + 0.01 * depth) & (semantic >= 0)
    if ((semantic >= classes) & valid).any():
        raise ValueError(
            "semantic_classes does not cover the saved semantic IDs; remap labels or increase classes"
        )
    votes = points.new_zeros((len(points), classes))
    for camera in range(uv.shape[0]):
        votes.scatter_add_(
            1,
            semantic[camera].clamp(0, classes - 1)[:, None],
            valid[camera, :, None].to(points.dtype),
        )
    labels = votes.argmax(-1)
    return torch.where(votes.sum(-1) > 0, labels, -1)


def reference_semantic_labels(inputs, targets, b, refs, uv, classes):
    label = targets["semantic"][b]
    x = uv[:, 0].round().long().clamp(0, label.shape[-1] - 1)
    y = uv[:, 1].round().long().clamp(0, label.shape[-2] - 1)
    refs = refs.long()
    valid = inputs["view_mask"][b, refs].clone()
    if "image_mask" in inputs:
        valid &= inputs["image_mask"][b, refs, y, x]
    cls = torch.where(valid, label[refs, y, x], -1)
    if (cls >= classes).any():
        raise ValueError("semantic_classes does not cover saved semantic IDs")
    return cls


class ReconstructionLoss(nn.Module):
    """Balanced multi-stage research loss; missing label families are skipped."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = (cfg or {}).get("loss", {})
        self.camera_config = (cfg or {}).get("camera", {})
        self.texture_threshold = (cfg or {}).get("model", {}).get("texture_threshold", 0.002)
        m = (cfg or {}).get("model", {})
        self.model_config = m
        self.reference_bounds = (
            (m.get("near", 2.0), m.get("far", 80.0))
            if self.cfg.get("enforce_reference_depth_range", False)
            else None
        )
        self.geometry_beta_m = self.cfg.get("geometry_beta_m", 1.0)
        self.absolute_scale_m = self.cfg.get("geometry_absolute_scale_m", 0.05)
        self.relative_scale = self.cfg.get("geometry_relative_scale", 0.01)
        self.stage_weights = self.cfg.get("stage_weights")
        if self.stage_weights is not None:
            if (
                not isinstance(self.stage_weights, (list, tuple))
                or not self.stage_weights
                or any(
                    not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0
                    for weight in self.stage_weights
                )
                or sum(self.stage_weights) <= 0
            ):
                raise ValueError("loss.stage_weights须为有限非负权重列表，且至少一项为正")
        if self.geometry_beta_m < 0 or self.absolute_scale_m <= 0 or self.relative_scale < 0:
            raise ValueError(
                "Geometry beta and relative scale must be nonnegative; "
                "absolute scale must be positive"
            )

    def _regression(self, prediction, target, reference_depth=None):
        """米制 Huber 误差；可按绝对尺度 + 相对深度尺度提高近处的权重。

        beta 始终以米为单位。可选的深度加权发生在 Huber 之后，不改变
        其二次区间，也不把目标精度当成已经达到的性能保证。
        """
        error = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.geometry_beta_m)
        if self.cfg.get("relative_geometry", False) and reference_depth is not None:
            scale = (
                self.absolute_scale_m + self.relative_scale * reference_depth.detach().clamp_min(0)
            )
            while scale.ndim < error.ndim:
                scale = scale.unsqueeze(-1)
            error = error / scale
        return error

    def forward(self, outputs, inputs, targets):
        terms = self._sparse_forward(outputs, inputs, targets)
        if self.cfg.get("matching_distillation", 0.0) > 0:
            if "matching_distillation_loss" not in outputs["coarse"]:
                raise ValueError("matching_distillation监督需要启用同名模型模块")
            terms["matching_distillation"] = outputs["coarse"]["matching_distillation_loss"]
            terms["total"] = terms["total"] + self.cfg["matching_distillation"] * terms[
                "matching_distillation"
            ]
        if self.cfg.get("observability_matching", 0.0) > 0:
            from .matching_losses import observability_matching_loss
            from .multifocal.data import mask_camera_inputs

            matching = observability_matching_loss(
                outputs["coarse"],
                mask_camera_inputs(inputs, self.camera_config),
                targets,
                self.cfg,
            )
            terms["total"] = terms["total"] + self.cfg["observability_matching"] * matching[
                "observability_matching"
            ]
            terms.update(matching)
        if "dense" not in outputs:
            if self.cfg.get("posterior_matching", 0.0) > 0:
                raise ValueError("posterior_matching监督需要启用稠密posterior_readout解码头")
            return terms
        from .dense_losses import dense_depth_losses
        from .multifocal.data import mask_camera_inputs

        if self.cfg.get("posterior_matching", 0.0) > 0:
            from .matching_losses import posterior_matching_loss

            if "posterior" not in outputs["dense"]:
                raise ValueError("posterior_matching监督需要启用dense_decoder.posterior_readout")
            posterior = posterior_matching_loss(
                outputs["dense"]["posterior"],
                mask_camera_inputs(inputs, self.camera_config),
                targets,
                self.cfg,
            )
            terms["total"] = terms["total"] + self.cfg["posterior_matching"] * posterior[
                "posterior_matching"
            ]
            terms.update(posterior)
        dense = dense_depth_losses(
            outputs["dense"],
            mask_camera_inputs(inputs, self.camera_config),
            targets,
            self.model_config,
            self.cfg,
        )
        weights = {
            "dense_depth": 1.0,
            "dense_gradient": 0.1,
            "dense_confidence": 0.05,
            "dense_initial_depth": 0.0,
            "dense_surface_recovery": 0.0,
        }
        terms["total"] = terms["total"] + sum(
            value * self.cfg.get(key, weights[key]) for key, value in dense.items()
        )
        terms.update(dense)
        return terms

    def _sparse_forward(self, outputs, inputs, targets):
        from .multifocal.data import mask_camera_inputs

        inputs = mask_camera_inputs(inputs, self.camera_config)
        coarse = outputs["coarse"]
        zero = coarse["logits"].sum() * 0.0
        terms = {}
        if "depth" not in targets:
            if "tele" in coarse or "mf_birth_logits" in outputs:
                outputs["multifocal_supervision"] = {
                    "tele_initializer": "skipped_missing_depth",
                    "birth_score": "skipped_missing_depth",
                }
            semantic_terms = []
            if "semantic" in targets:
                for b, scene in enumerate(outputs.get("scenes", [])):
                    stage = scene.get("query_stage", scene)
                    aux = stage.get("aux", {})
                    refs = aux.get("ref_index", aux.get("query_ref_index"))
                    uv = aux.get("ref_uv", aux.get("query_uv"))
                    if "semantic_logits" in stage and refs is not None and uv is not None:
                        cls = reference_semantic_labels(
                            inputs,
                            targets,
                            b,
                            refs,
                            uv,
                            stage["semantic_logits"].shape[-1],
                        )
                        if (cls >= 0).any():
                            semantic_terms.append(
                                F.cross_entropy(stage["semantic_logits"], cls, ignore_index=-1)
                            )
            if semantic_terms:
                semantic = torch.stack(semantic_terms).mean()
                return {
                    "semantic": semantic,
                    "total": zero + semantic * self.cfg.get("semantic", 0.1),
                }
            return {"total": zero}
        if "normals" not in targets and outputs.get("scenes"):
            targets = dict(targets)
            targets["normals"] = normals_from_depth(
                inputs,
                targets,
                self.cfg.get("normal_absolute_jump", 0.05),
                self.cfg.get("normal_relative_jump", 0.02),
            )
        depth = targets["depth"]
        depth_valid = targets.get("depth_valid", depth > 0) & inputs["view_mask"][..., None, None]
        if "image_mask" in inputs:
            depth_valid &= inputs["image_mask"]
        h, w = coarse["depth"].shape[-2:]
        # CNN centers are stride*x, not an align_corners=False resize grid.
        y = (torch.arange(h, device=depth.device) * 16).clamp(max=depth.shape[-2] - 1)
        x = (torch.arange(w, device=depth.device) * 16).clamp(max=depth.shape[-1] - 1)
        d = depth[..., y[:, None], x[None, :]]
        valid = depth_valid[..., y[:, None], x[None, :]] & (d > 0)
        if "supervised_views" in coarse:
            valid = valid & coarse["supervised_views"][..., None, None]
        inv = coarse["inverse_depths"]
        rho = d.clamp_min(1e-5).reciprocal()
        in_range = (rho >= inv.min()) & (rho <= inv.max())
        # View evidence target is geometric support, not GT validity alone.
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pixel = torch.stack([xx, yy], -1).to(depth.dtype)
        match = torch.zeros_like(valid)
        match_known = torch.zeros_like(valid)
        for b in range(depth.shape[0]):
            for n in range(depth.shape[1]):
                if not bool(inputs["view_mask"][b, n]):
                    continue
                X, _ = unproject(
                    pixel.reshape(-1, 2),
                    d[b, n].reshape(-1),
                    inputs["K"][b, n],
                    inputs["T_cam_from_rig"][b, n],
                )
                uv, zz, vv, _ = project(
                    X,
                    inputs["K"][b],
                    inputs["T_cam_from_rig"][b],
                    depth.shape[-2:],
                    inputs["view_mask"][b],
                    inputs["image_mask"][b] if "image_mask" in inputs else None,
                )
                ix = uv[..., 0].round().long().clamp(0, depth.shape[-1] - 1)
                iy = uv[..., 1].round().long().clamp(0, depth.shape[-2] - 1)
                ni = torch.arange(depth.shape[1], device=depth.device)[:, None]
                dd = depth[b][ni, iy, ix]
                visible = vv & depth_valid[b][ni, iy, ix] & ((dd - zz).abs() < (0.08 + 0.01 * dd))
                visible[n] = False
                # Pure rotation has no independent metric depth evidence.
                ctr = -torch.einsum(
                    "nij,nj->ni",
                    inputs["T_cam_from_rig"][b, :, :3, :3].transpose(-1, -2),
                    inputs["T_cam_from_rig"][b, :, :3, 3],
                )
                independent = ((ctr - ctr[n]).norm(dim=-1) > 1e-4)[:, None]
                visible &= independent
                match[b, n] = visible.any(0).reshape(h, w)
                # 缺少源深度是可见性未知，不能当成已证实的no-match。
                unresolved = (vv & independent & ~depth_valid[b][ni, iy, ix]).any(0)
                match_known[b, n] = (visible.any(0) | ~unresolved).reshape(h, w)
        if "texture" in coarse:
            match &= coarse["texture"].detach() >= self.texture_threshold
        nearest = (rho[..., None] - inv).abs().argmin(-1)
        # A depth class can be structurally masked by source FOV, texture or
        # ray derivative. Never train against its constant -1e4 sentinel.
        candidate_logit = (
            coarse["logits"][:, :, :-1].detach().gather(2, nearest[:, :, None]).squeeze(2)
        )
        match &= candidate_logit > -1e3
        class_id = torch.where(valid & in_range & match, nearest, len(inv))
        logits = coarse["logits"].flatten(0, 1)
        # Missing depth labels are not evidence of no-match.
        class_error = F.cross_entropy(logits, class_id.flatten(0, 1), reduction="none")
        class_supervised = (valid & (~in_range | match_known)).flatten(0, 1)
        terms["coarse_class"] = _masked_mean(
            class_error,
            class_supervised,
        )
        if self.cfg.get("observability_matching", 0.0) > 0:
            # 软目标替换条件深度分类时，保留原本已知的no-match监督。
            # 缺源深度的未决位置仍由原class_supervised屏蔽，不变成负例。
            terms["coarse_no_match"] = _masked_mean(
                class_error,
                class_supervised & (class_id == len(inv)).flatten(0, 1),
            )
        terms["coarse_depth"] = _masked_mean(
            F.smooth_l1_loss(
                coarse["depth"] / d.clamp_min(1), torch.ones_like(d), reduction="none"
            ),
            valid & in_range & match,
        )
        if self.cfg.get("reference_depth_regression", False):
            # 参考相机的第一命中深度已经是有效标量标签，无需源相机也有深度。
            # 这里只训练深度值，不修改跨视图可见性或 no-match 的监督。
            terms["coarse_reference_depth"] = _masked_mean(
                self._regression(coarse["depth"], d, d),
                valid & in_range & (candidate_logit > -1e3),
            )
        # 参考图有米制深度、源图没有深度时，仍可监督条件深度分布；
        # 去掉no-match通道，避免将未知的跨视图可见性伪造为支持或反对。
        partial = valid & in_range & ~match_known & (candidate_logit > -1e3)
        if partial.any():
            depth_logits = coarse["logits"][:, :, :-1].flatten(0, 1)
            terms["coarse_depth_class_partial"] = _masked_mean(
                F.cross_entropy(depth_logits, nearest.flatten(0, 1), reduction="none"),
                partial.flatten(0, 1),
            )
        stage_terms = {
            k: []
            for k in (
                "state",
                "split",
                "geometry",
                "normal",
                "correspondence",
                "update",
                "confidence",
                "semantic",
                "candidate_class",
                "candidate_metric_risk",
                "reference_depth",
            )
        }
        stage_term_weights = {key: [] for key in stage_terms}
        for b, scene in enumerate(outputs.get("scenes", [])):
            if not depth_valid[b].any():
                continue
            # An unlabeled camera is not a GT assertion of unknown space.
            all_views_labeled = bool(
                (depth_valid[b].flatten(1).any(-1) | ~inputs["view_mask"][b]).all()
            )
            # Refiner auxiliaries describe pre-fusion queries, not fused points.
            stages = (
                [scene["query_stage"]]
                if "query_stage" in scene
                else list(scene.get("stages", [])) or [scene]
            )
            if self.stage_weights is not None and len(self.stage_weights) != len(stages):
                raise ValueError("loss.stage_weights长度必须与实际解码阶段数一致")
            for stage_index, stage in enumerate(stages):
                if not isinstance(stage, dict) or "points" not in stage or not len(stage["points"]):
                    continue
                points = stage["points"]
                reference_label = None
                reference_observed = torch.zeros(
                    len(points), device=points.device, dtype=torch.bool
                )
                aux = dict(stage.get("aux", {}))
                for old, new in (
                    ("ref_index", "query_ref_index"),
                    ("ref_uv", "query_uv"),
                    ("points_before", "initial_points"),
                ):
                    if old in aux:
                        aux[new] = aux[old]
                if "query_ref_index" in aux and "query_uv" in aux:
                    refs, uv = aux["query_ref_index"].long(), aux["query_uv"]
                    ix = uv[:, 0].round().long().clamp(0, depth.shape[-1] - 1)
                    iy = uv[:, 1].round().long().clamp(0, depth.shape[-2] - 1)
                    ok = depth_valid[b, refs, iy, ix]
                    reference_observed = ok.clone()
                    reference_label = depth[b, refs, iy, ix]
                    if "depth_bounds_m" in aux or self.reference_bounds is not None:
                        bounds = aux.get(
                            "depth_bounds_m",
                            (
                                points.new_tensor(self.reference_bounds).expand(len(points), 2)
                                if self.reference_bounds
                                else None
                            ),
                        )
                        ok = (
                            ok
                            & (reference_label >= bounds[:, 0])
                            & (reference_label <= bounds[:, 1])
                        )
                    target = torch.zeros_like(points)
                    normal = torch.zeros_like(points)
                    for n in refs.unique():
                        sel = refs == n
                        target[sel], _ = unproject(
                            uv[sel],
                            depth[b, n, iy[sel], ix[sel]],
                            inputs["K"][b, n],
                            inputs["T_cam_from_rig"][b, n],
                        )
                        if "normals" in targets:
                            normal[sel] = targets["normals"][b, n, :, iy[sel], ix[sel]].T
                    state = torch.where(ok, SURFACE, UNKNOWN)
                    labels = {
                        "state": state,
                        "split": ok,
                        "surface_point": target,
                        "normal": normal,
                    }
                    if self.cfg.get("reference_depth_regression", False):
                        # 从预测三维点计算参考光轴深度，避免用射线长度代替 Z-depth。
                        transform = inputs["T_cam_from_rig"][b, refs]
                        predicted_depth = (points * transform[:, 2, :3]).sum(-1) + transform[
                            :, 2, 3
                        ]
                        stage_terms["reference_depth"].append(
                            _masked_mean(
                                self._regression(predicted_depth, reference_label, reference_label),
                                ok,
                            )
                        )
                    if all(
                        key in aux
                        for key in (
                            "candidate_logits",
                            "candidate_depths",
                            "candidate_valid",
                        )
                    ):
                        stage_terms["candidate_class"].append(
                            candidate_depth_loss(
                                aux["candidate_logits"],
                                aux["candidate_depths"],
                                aux["candidate_valid"],
                                depth[b, refs, iy, ix],
                                ok,
                            )
                        )
                        if self.cfg.get("candidate_metric_risk", 0) > 0:
                            stage_terms["candidate_metric_risk"].append(
                                candidate_depth_loss(
                                    aux["candidate_logits"],
                                    aux["candidate_depths"],
                                    aux["candidate_valid"],
                                    depth[b, refs, iy, ix],
                                    ok,
                                    metric_risk=True,
                                )
                            )
                else:
                    labels = label_query_points(
                        points.detach(),
                        inputs,
                        targets,
                        b,
                        self.cfg.get("surface_band", 0.08),
                    )
                hit = labels["state"] == SURFACE
                state_supervised = (labels["state"] != UNKNOWN) | all_views_labeled
                if self.cfg.get("supervise_out_of_range", False):
                    # 有效参考深度超出搜索范围，也是明确的拒绝有限表面监督。
                    # 缺失标签/无效深度不因此变成天空、free 或已证实 unknown。
                    state_supervised = state_supervised | reference_observed
                if "state_logits" in stage:
                    # Unknown remains a supervised class; it is never silently dropped.
                    stage_terms["state"].append(
                        _masked_mean(
                            F.cross_entropy(
                                stage["state_logits"], labels["state"], reduction="none"
                            ),
                            state_supervised,
                        )
                    )
                target = labels["surface_point"]
                geometry = self._regression(points, target, reference_label).mean(-1)
                stage_terms["geometry"].append(_masked_mean(geometry, hit))
                if "confidence_logits" in stage or "quality" in stage:
                    # Calibrate quality to metric surface accuracy; never let
                    # predicted quality gate its own geometry supervision.
                    accuracy = (points.detach() - target).norm(dim=-1)
                    accurate = hit & (accuracy < self.cfg.get("confidence_tolerance", 0.2))
                    if "confidence_logits" in stage:
                        quality_loss = F.binary_cross_entropy_with_logits(
                            stage["confidence_logits"],
                            accurate.float(),
                            reduction="none",
                        )
                    else:
                        quality_loss = F.binary_cross_entropy(
                            stage["quality"].clamp(1e-6, 1 - 1e-6),
                            accurate.float(),
                            reduction="none",
                        )
                    stage_terms["confidence"].append(_masked_mean(quality_loss, state_supervised))
                if "normals" in stage and "normals" in targets:
                    n = labels["normal"]
                    normal_loss = 1 - (F.normalize(stage["normals"], dim=-1) * n).sum(-1).abs()
                    stage_terms["normal"].append(
                        _masked_mean(normal_loss, hit & (n.norm(dim=-1) > 0.5))
                    )
                if "initial_points" in aux and "delta_xyz" in aux:
                    gt_delta = target - aux["initial_points"].detach()
                    stage_terms["update"].append(
                        _masked_mean(
                            self._regression(aux["delta_xyz"], gt_delta, reference_label).mean(-1),
                            hit,
                        )
                    )
                for step in aux.get("steps", []):
                    # Every refinement stage is supervised, including rejected
                    # zero updates; its source image correction branch has a
                    # direct target and cannot hide behind update gating.
                    correction = step["points_after"] - step["points_before"]
                    desired_correction = target - step["points_before"].detach()
                    stage_terms["update"].append(
                        _masked_mean(
                            self._regression(correction, desired_correction, reference_label).mean(
                                -1
                            ),
                            hit,
                        )
                    )
                    if "proposed_view_offsets" in step:
                        goal_uv, gz, gv, _ = project(
                            target,
                            inputs["K"][b],
                            inputs["T_cam_from_rig"][b],
                            depth.shape[-2:],
                            inputs["view_mask"][b],
                        )
                        ix = goal_uv[..., 0].round().long().clamp(0, depth.shape[-1] - 1)
                        iy = goal_uv[..., 1].round().long().clamp(0, depth.shape[-2] - 1)
                        ni = torch.arange(depth.shape[1], device=depth.device)[:, None]
                        gd = depth[b][ni, iy, ix]
                        visible = (
                            gv
                            & depth_valid[b][ni, iy, ix]
                            & ((gd - gz).abs() < 0.1 + 0.01 * gd)
                            & hit[None]
                        )
                        desired = goal_uv.transpose(0, 1) - step["source_uv"].detach()
                        flow_error = F.smooth_l1_loss(
                            step["proposed_view_offsets"], desired, reduction="none"
                        ).mean(-1)
                        stage_terms["correspondence"].append(_masked_mean(flow_error, visible.T))
                # Per-view correction supervision anchors motion to real visible first hits.
                if "view_offsets" in aux and "initial_points" in aux:
                    base_uv, _, base_valid, _ = project(
                        aux["initial_points"].detach(),
                        inputs["K"][b],
                        inputs["T_cam_from_rig"][b],
                        depth.shape[-2:],
                        inputs["view_mask"][b],
                    )
                    goal_uv, gz, gv, _ = project(
                        target,
                        inputs["K"][b],
                        inputs["T_cam_from_rig"][b],
                        depth.shape[-2:],
                        inputs["view_mask"][b],
                    )
                    ix = goal_uv[..., 0].round().long().clamp(0, depth.shape[-1] - 1)
                    iy = goal_uv[..., 1].round().long().clamp(0, depth.shape[-2] - 1)
                    ni = torch.arange(depth.shape[1], device=depth.device)[:, None]
                    gd = depth[b][ni, iy, ix]
                    visible = (
                        gv
                        & base_valid
                        & depth_valid[b][ni, iy, ix]
                        & ((gd - gz).abs() < 0.1 + 0.01 * gd)
                        & hit[None]
                    )
                    pred = aux["view_offsets"]
                    desired = (goal_uv - base_uv).transpose(0, 1)
                    if pred.shape == desired.shape:
                        stage_terms["correspondence"].append(
                            _masked_mean(
                                F.smooth_l1_loss(pred, desired, reduction="none").mean(-1),
                                visible.T,
                            )
                        )
                if (
                    "semantic_logits" in stage
                    and "semantic" in targets
                    and "query_ref_index" in aux
                ):
                    refs, uv = aux["query_ref_index"].long(), aux["query_uv"]
                    cls = reference_semantic_labels(
                        inputs, targets, b, refs, uv, stage["semantic_logits"].shape[-1]
                    )
                    stage_terms["semantic"].append(
                        F.cross_entropy(stage["semantic_logits"], cls, ignore_index=-1)
                        if (cls >= 0).any()
                        else stage["semantic_logits"].sum() * 0
                    )
                elif "semantic_logits" in stage and "semantic" in targets:
                    cls = visible_semantic_labels(
                        labels.get("representative_point", target),
                        inputs,
                        targets,
                        b,
                        stage["semantic_logits"].shape[-1],
                    )
                    cls = torch.where(hit, cls, -1)
                    stage_terms["semantic"].append(
                        F.cross_entropy(stage["semantic_logits"], cls, ignore_index=-1)
                        if (cls >= 0).any()
                        else stage["semantic_logits"].sum() * 0
                    )
                # 同一层的某项可能添加多次（例如对应关系），缺标签时也可能不添加。
                # 对实际新增标量逐个记权重，不能把四个stage权重直接zip到所有损失。
                stage_weight = 1 if self.stage_weights is None else self.stage_weights[stage_index]
                for key, values in stage_terms.items():
                    added = len(values) - len(stage_term_weights[key])
                    stage_term_weights[key].extend([stage_weight] * added)
        for key, values in stage_terms.items():
            if values:
                terms[key] = weighted_stage_mean(values, stage_term_weights[key])
        defaults = {
            "coarse_class": 1.0,
            "coarse_depth": 0.2,
            "coarse_reference_depth": 0.2,
            "reference_depth": 0.5,
            "state": 1.0,
            "split": 0.1,
            "geometry": 1.0,
            "normal": 0.1,
            "correspondence": 0.1,
            "update": 0.25,
            "confidence": 0.1,
            "semantic": 0.1,
            "candidate_class": 0.5,
            "candidate_metric_risk": 0.0,
        }
        if "tele" in coarse or "mf_birth_logits" in outputs:
            from .multifocal.losses import auxiliary_losses

            extra, status = auxiliary_losses(outputs, inputs, targets)
            terms.update(extra)
            outputs["multifocal_supervision"] = status
            defaults.update(mf_tele_class=0.5, mf_birth_score=0.2)
        terms["total"] = sum(
            (value * self.cfg.get(key, defaults.get(key, 1.0)) for key, value in terms.items()),
            zero,
        )
        return terms
