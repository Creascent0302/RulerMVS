"""跨单位深度与可见对应监督；标签只在此训练模块使用。"""

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, project, sample_maps, unproject


def masked_mean(value, valid):
    return torch.where(valid, value, 0).sum() / valid.sum().clamp_min(1)


def visible_correspondence(pyramid, inputs, targets, max_points=384):
    """只用双方深度认证的可见对应；沿标尺生成亚像素到大错峰负例。"""
    terms, pairs = [], 0
    features = pyramid[4].float()
    batch, views, channels = features.shape[:3]
    h, w = inputs["rgb"].shape[-2:]
    ratio = features.new_tensor((inputs["matching_rgb"].shape[-1] / w, inputs["matching_rgb"].shape[-2] / h))
    for sample in range(batch):
        depth = targets["depth"][sample, 0, ::8, ::8].flatten().float()
        mask = targets["depth_valid"][sample, 0, ::8, ::8].flatten()
        mask = mask & inputs["image_mask"][sample, 0, ::8, ::8].flatten() & torch.isfinite(depth) & (depth > 0)
        indices = mask.nonzero().flatten()
        if not len(indices):
            continue
        indices = indices[torch.linspace(0, len(indices) - 1, min(max_points, len(indices)), device=depth.device).long()]
        grid = pixel_grid(h // 8, w // 8, device=depth.device, stride=8).reshape(-1, 2)[indices]
        z = depth[indices]
        K, T = inputs["K"][sample].float(), inputs["T_cam_from_rig"][sample].float()
        points, rays = unproject(grid, z, K[0], T[0])
        uv, source_z, legal, J = project(points, K, T, (h, w), inputs["view_mask"][sample], inputs["image_mask"][sample])
        observed = sample_maps(targets["depth"][sample, :, None].float(), uv, mode="nearest")[..., 0]
        known = sample_maps(targets["depth_valid"][sample, :, None].float(), uv, mode="nearest")[..., 0] > .5
        tolerance = .02 * source_z.abs().clamp_min(1e-3)
        visible = legal & known & ((observed - source_z).abs() < tolerance)
        anchor_uv = ((grid[None] + .5) * ratio - .5) / 4
        anchor = F.normalize(sample_maps(features[sample, :1], anchor_uv)[0], dim=-1)
        for view in range(1, views):
            with torch.no_grad():
                derivative = (J[view] * rays[:, None]).sum(-1) * ratio / 4
                step = .75 / torch.linalg.vector_norm(derivative, dim=-1).clamp_min(1e-5)
                step = torch.minimum(step, z * .25)
                offsets = z.new_tensor((-8., -4., -2., -1., 0., 1., 2., 4., 8.))
                choices = (z[:, None] + step[:, None] * offsets).clamp_min(1e-4)
                candidate, _ = unproject(grid[:, None].expand(-1, 9, -1), choices, K[0], T[0])
                pixel, _, ok, _ = project(candidate.reshape(-1, 3), K[view:view + 1], T[view:view + 1], (h, w))
                pixel = pixel.reshape(-1, 9, 2)
                ok = ok.reshape(-1, 9)
                displacement = torch.linalg.vector_norm((pixel - uv[view, :, None]) * ratio / 4, dim=-1)
                ok &= displacement >= .5
                ok[:, 4] = True
                supported = visible[view] & (ok.sum(1) > 1) & (torch.linalg.vector_norm(derivative, dim=-1) > 1e-5)
            locations = ((pixel.reshape(1, -1, 2) + .5) * ratio - .5) / 4
            sampled = sample_maps(features[sample, view:view + 1], locations)[0].reshape(-1, 9, channels)
            logits = (F.normalize(sampled, dim=-1) * anchor[:, None]).sum(-1) / .07
            logits = logits.masked_fill(~ok, -1e4)
            ce = F.cross_entropy(logits, torch.full((len(logits),), 4, device=logits.device), reduction="none")
            if bool(supported.any()):
                terms.append(ce[supported].mean())
                pairs += int(supported.sum())
    zero = features.sum() * 0
    return (torch.stack(terms).mean() if terms else zero), pairs


def depth_objective(prediction, inputs, targets, domain, settings, correspondence_weight=0.):
    scale = prediction["unit_scale"]
    gt = targets["depth"][:, :1].float() / scale
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1] & inputs["view_mask"][:, :1, None, None]
    valid = valid & torch.isfinite(gt) & (gt > 0)
    if domain.get("eval_range"):
        valid &= (gt * scale >= domain["eval_range"][0]) & (gt * scale <= domain["eval_range"][1])
    safe_gt = torch.where(valid, gt, 1.)
    estimate = prediction["depth"].float().clamp_min(1e-6)
    log_error = (estimate.log() - safe_gt.log()).abs()
    relative = ((estimate - safe_gt).abs() / safe_gt.clamp_min(.1)).clamp_max(100)
    loss = masked_mean(log_error + .2 * relative, valid)
    metrics = {"depth_log": masked_mean(log_error, valid), "abs_rel": masked_mean(relative, valid)}
    gradients = []
    for axis in (-1, -2):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        error = estimate.log()[a] - estimate.log()[b] - safe_gt.log()[a] + safe_gt.log()[b]
        gradients.append(masked_mean(error.abs(), valid[a] & valid[b]))
    loss = loss + .05 * torch.stack(gradients).mean()
    for name, stride in (("coarse", 8), ("fine", 4)):
        stage = prediction[name]
        truth = safe_gt[..., ::stride, ::stride]
        mask = valid[..., ::stride, ::stride] & stage["ref_valid"]
        if name in settings.get("quality_stages", []):
            from .candidate_supervision import candidate_supervision

            terms = candidate_supervision(
                stage, truth, mask, temperature=settings.get("quality_temperature", .05),
                depth_range=(domain["near"] / scale, domain["far"] / scale),
            )
            loss = loss + .25 * terms["classification"] + .2 * terms["location"] + .1 * terms["risk"]
            metrics[name + "_ce"] = terms["classification"]
            metrics[name + "_covered"] = terms["covered"]
            continue
        lattice = stage["sampling_depth"].detach().clamp_min(1e-6)
        distance = (lattice.log() - truth.log()).abs()
        index = distance.argmin(1)
        ce = F.cross_entropy(stage["scores"].float(), index, reduction="none")
        classification = masked_mean(ce, mask[:, 0])
        closest = stage["candidate_depth"].gather(1, index[:, None]).clamp_min(1e-6)
        lower = stage["lower"].gather(1, index[:, None])
        upper = stage["upper"].gather(1, index[:, None])
        reachable = mask & (truth >= lower) & (truth <= upper)
        location = masked_mean((closest.log() - truth.log()).abs(), reachable)
        risk = masked_mean((stage["probability"] * distance.clamp_max(4)).sum(1), mask[:, 0])
        loss = loss + .25 * classification + .2 * location + .1 * risk
        metrics[name + "_ce"] = classification
        metrics[name + "_covered"] = masked_mean(reachable.float(), mask)
    correspondence = estimate.sum() * 0
    pairs = 0
    if correspondence_weight > 0 and domain.get("correspondence", True):
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            correspondence, pairs = visible_correspondence(prediction["pyramid"], inputs, targets)
        loss = loss + correspondence_weight * correspondence
    metrics.update(total=loss, correspondence=correspondence, supervised_pixels=valid.sum(),
                   correspondence_pairs=loss.new_tensor(pairs))
    return metrics
