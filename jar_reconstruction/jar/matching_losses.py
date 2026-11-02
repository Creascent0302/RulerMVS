"""用固定像素容差监督条件深度分布，不将投影可见域当成真实可见性。"""

import math

import torch
from torch.nn import functional as F

from .dense_losses import sample_stage_labels
from .geometry import camera_centers, pixel_grid, project, unproject


@torch.no_grad()
def pixel_tolerance_targets(
    uv,
    target_depth,
    target_valid,
    candidate_depths,
    candidate_valid,
    intrinsic,
    transform,
    reference,
    image_hw,
    view_mask=None,
    image_mask=None,
    pixel_sigma=0.5,
    chunk_size=32,
):
    """生成 [Q,D] 软目标，所有几何和标签均脱离反向传播。

    对同一条参考射线上的候选与GT做精确重投影，以源图中最大的
    像素位移定义高斯形状的软标签。pixel_sigma 使用输入RGB像素单位，
    不是特征像素；更多重复相机不会人为缩窄容差。

    该分布描述标定下的潜在分辨能力，不是已校准的深度后验。源图的
    遮挡与纹理并未被GT确认，不生成逐源对应标签，也不监督no-match。
    无独立视差、GT超范围或最近候选被结构屏蔽时跳过整条射线。
    """
    if not math.isfinite(pixel_sigma) or pixel_sigma <= 0 or chunk_size < 1:
        raise ValueError("pixel_sigma须为正且有限，chunk_size须为正整数")
    target = target_depth.detach().float()
    depths = candidate_depths.detach().float()
    if depths.ndim == 1:
        depths = depths[None].expand(len(target), -1)
    finite_depth = torch.isfinite(depths) & (depths > 0)
    legal = candidate_valid.detach().bool() & finite_depth
    safe_depth = torch.where(finite_depth, depths, 1.0)
    finite_target = torch.isfinite(target) & (target > 0)
    safe_target = torch.where(finite_target, target, 1.0)
    lower = safe_depth.masked_fill(~legal, torch.inf).amin(-1)
    upper = safe_depth.masked_fill(~legal, -torch.inf).amax(-1)
    supervised = target_valid.bool() & finite_target & (target >= lower) & (target <= upper)
    supervised &= legal.sum(-1) >= 2
    inverse_distance = (safe_depth.reciprocal() - safe_target[:, None].reciprocal()).abs()
    nearest = inverse_distance.masked_fill(~finite_depth, torch.inf).amin(-1)
    supervised &= (legal & (inverse_distance <= nearest[:, None] + 1e-7)).any(-1)

    points, rays = unproject(uv, safe_target, intrinsic[reference], transform[reference])
    gt_uv, _, gt_valid, jacobian = project(
        points, intrinsic, transform, image_hw, view_mask, image_mask
    )
    derivative = (jacobian * rays[None, :, None]).sum(-1).norm(dim=-1)
    centers = camera_centers(transform)
    independent = (centers - centers[reference]).norm(dim=-1) > 1e-4
    potential = gt_valid & independent[:, None] & (derivative > 1e-5)
    potential[reference] = False
    supervised &= gt_valid[reference] & potential.any(0)

    distances = []
    for start in range(0, depths.shape[1], chunk_size):
        selected = safe_depth[:, start : start + chunk_size]
        count = selected.shape[1]
        candidate_points, _ = unproject(
            uv[:, None].expand(-1, count, -1),
            selected,
            intrinsic[reference],
            transform[reference],
        )
        candidate_uv, _, visible_domain, _ = project(
            candidate_points.reshape(-1, 3), intrinsic, transform, image_hw, view_mask, image_mask
        )
        candidate_uv = candidate_uv.reshape(len(intrinsic), len(target), count, 2)
        visible_domain = visible_domain.reshape(len(intrinsic), len(target), count)
        displacement = (candidate_uv - gt_uv[:, :, None]).norm(dim=-1)
        # GT位于某源视图时，出该视图的候选不能被当作零位移匹配。
        displacement = displacement.masked_fill(
            ~visible_domain & potential[:, :, None], torch.inf
        )
        displacement = torch.where(potential[:, :, None], displacement, 0.0)
        distances.append(displacement.amax(0))
    pixel_distance = torch.cat(distances, -1)
    allowed = legal & torch.isfinite(pixel_distance)
    supervised &= allowed.any(-1)
    log_weight = -0.5 * (pixel_distance / pixel_sigma).square()
    log_weight = log_weight.masked_fill(~allowed, -torch.inf)
    maximum = log_weight.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weight = (log_weight - maximum).exp()
    distribution = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    distribution = torch.where(supervised[:, None], distribution, 0.0)
    return {
        "distribution": distribution,
        "supervised": supervised,
        "potential_sources": potential.sum(0),
        "pixel_distance": pixel_distance,
    }


def conditional_matching_loss(prediction, inputs, targets, settings, prefix):
    """通用条件候选监督；prediction.logits不含no-match，valid只表示候选合法性。"""
    logits = prediction["logits"].float()
    zero = logits.reshape(-1)[:0].sum()
    empty = {
        f"{prefix}_matching": zero,
        f"{prefix}_target_entropy": zero.detach(),
        f"{prefix}_supervised_fraction": zero.detach(),
    }
    if "depth" not in targets:
        return empty
    if "depths" in prediction:
        candidates = prediction["depths"].detach().float()
    else:
        candidates = prediction["inverse_depths"].detach().float().reciprocal()
    if candidates.ndim != 1 and candidates.shape != logits.shape:
        raise ValueError("候选depths须为[D]或与logits同形状[B,N,D,Hs,Ws]")
    finite = torch.isfinite(candidates) & (candidates > 0)
    if not finite.any():
        return empty
    stride = prediction.get("stride", 16)
    depth = targets["depth"]
    truth, valid = sample_stage_labels(
        depth,
        targets.get("depth_valid", depth > 0),
        inputs,
        stride,
        logits.shape[-2:],
        float(candidates[finite].min()),
        float(candidates[finite].max()),
    )
    if "supervised_views" in prediction:
        valid &= prediction["supervised_views"][..., None, None]
    uv = pixel_grid(*logits.shape[-2:], device=logits.device, stride=stride).reshape(-1, 2)
    total, entropy_sum = zero, zero.detach()
    supervised_count = zero.detach()
    pixel_sigma = settings.get(
        f"{prefix}_pixel_sigma", settings.get("observability_pixel_sigma", 0.5)
    )
    chunk = settings.get(f"{prefix}_chunk", settings.get("observability_chunk", 32))
    for batch in range(logits.shape[0]):
        for reference in range(logits.shape[1]):
            selected = valid[batch, reference].flatten()
            if not selected.any():
                continue
            scores = logits[batch, reference].flatten(1).T[selected]
            legal = torch.isfinite(scores.detach()) & (scores.detach() > -1e3)
            if "valid" in prediction:
                legal &= prediction["valid"][batch, reference].flatten(1).T[selected].bool()
            depths = candidates
            if candidates.ndim != 1:
                depths = candidates[batch, reference].flatten(1).T[selected]
            reference_truth = truth[batch, reference].flatten()[selected]
            target = pixel_tolerance_targets(
                uv[selected],
                reference_truth,
                torch.ones_like(reference_truth, dtype=torch.bool),
                depths,
                legal,
                inputs["K"][batch],
                inputs["T_cam_from_rig"][batch],
                reference,
                inputs["rgb"].shape[-2:],
                inputs["view_mask"][batch],
                inputs["image_mask"][batch] if "image_mask" in inputs else None,
                pixel_sigma=pixel_sigma,
                chunk_size=chunk,
            )
            desired = target["distribution"]
            supervised = target["supervised"]
            safe_scores = torch.where(legal, scores, -1e4)
            error = -(desired * F.log_softmax(safe_scores, dim=-1)).sum(-1)
            total = total + torch.where(supervised, error, 0.0).sum()
            entropy = -(desired * desired.clamp_min(1e-12).log()).sum(-1)
            entropy_sum = entropy_sum + entropy.sum()
            supervised_count = supervised_count + supervised.sum()
    denominator = supervised_count.clamp_min(1)
    return {
        f"{prefix}_matching": total / denominator,
        f"{prefix}_target_entropy": entropy_sum / denominator,
        f"{prefix}_supervised_fraction": supervised_count / valid.sum().clamp_min(1),
    }


def observability_matching_loss(coarse, inputs, targets, settings):
    """仅去掉粗深度no-match通道，再复用通用条件候选监督。"""
    prediction = dict(coarse, logits=coarse["logits"][:, :, :-1])
    return conditional_matching_loss(prediction, inputs, targets, settings, "observability")


def posterior_matching_loss(posterior, inputs, targets, settings):
    """稠密后验候选没有no-match通道，不修改空间状态或置信度定义。"""
    return conditional_matching_loss(posterior, inputs, targets, settings, "posterior")
