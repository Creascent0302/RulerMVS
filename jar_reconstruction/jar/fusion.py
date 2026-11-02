"""可见表面的保守空间融合与 ROI 限制。"""

import torch
from torch.nn import functional as F


def _segment_mean(values, inverse, count, weights=None):
    output_dtype = values.dtype
    # AMP features may be BF16 while calibrated quality weights are FP32.
    # Accumulate in FP32 for precision and an index_add source/destination match.
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    weights = values.new_ones(values.shape[0]) if weights is None else weights
    weights = weights.to(values.dtype)
    result = values.new_zeros((count,) + values.shape[1:])
    expand = (weights.shape[0],) + (1,) * (values.ndim - 1)
    result.index_add_(0, inverse, values * weights.reshape(expand))
    total = weights.new_zeros(count).index_add_(0, inverse, weights).clamp_min(1e-8)
    return (result / total.reshape((count,) + (1,) * (values.ndim - 1))).to(output_dtype)


def spatial_hash_fuse(scene, cell_size=0.05, normal_cosine=0.97):
    """Conservative local fusion; ambiguous/boundary groups remain separate.

    Only confirmed surface samples may merge. A candidate hash group must fit
    inside a quarter cell in Euclidean distance, and all normals must agree.
    Unknown samples remain in the result. This is not volumetric occupancy.
    """
    points = scene["points"]
    count = points.shape[0]
    if count == 0 or cell_size <= 0:
        return scene
    normals = scene["normals"]
    cells = torch.floor(points.detach() / cell_size).to(torch.int64)
    normal_bins = torch.round(normals.detach() * 4).to(torch.int64)
    # Quantizing the normal offset more finely separates nearby parallel layers.
    plane_bin = torch.round(
        (normals.detach() * points.detach()).sum(-1, keepdim=True) / (cell_size * 0.125)
    ).long()
    keys = torch.cat((cells, normal_bins, plane_bin), -1)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    groups = int(inverse.max().item()) + 1
    lower = points.new_full((groups, 3), float("inf"))
    upper = points.new_full((groups, 3), -float("inf"))
    lower.scatter_reduce_(
        0, inverse[:, None].expand(-1, 3), points.detach(), reduce="amin", include_self=True
    )
    upper.scatter_reduce_(
        0, inverse[:, None].expand(-1, 3), points.detach(), reduce="amax", include_self=True
    )
    mean_normal = F.normalize(_segment_mean(normals.detach(), inverse, groups), dim=-1, eps=1e-6)
    cos = (normals.detach() * mean_normal[inverse]).sum(-1)
    min_cos = cos.new_ones(groups).scatter_reduce_(
        0, inverse, cos, reduce="amin", include_self=True
    )
    surface = scene["valid"] & (scene["state"] == 1)
    all_surface = torch.ones(groups, device=points.device, dtype=torch.long)
    all_surface.scatter_reduce_(0, inverse, surface.long(), reduce="amin", include_self=True)
    safe = (
        ((upper - lower).norm(dim=-1) < cell_size * 0.25)
        & (min_cos > normal_cosine)
        & all_surface.bool()
    )
    # Rejected groups receive unique per-sample IDs, so no boundary is collapsed.
    labels = torch.where(safe[inverse], inverse, groups + torch.arange(count, device=points.device))
    _, final = torch.unique(labels, sorted=True, return_inverse=True)
    fused_count = int(final.max().item()) + 1
    if fused_count == count:
        scene["fusion_assignment"] = torch.arange(count, device=points.device)
        return scene
    weights = scene["confidence"].detach().clamp_min(0.05)
    result = dict(scene)
    for key in (
        "points",
        "normals",
        "scales",
        "features",
        "state_logits",
        "centers",
        "sizes",
        "split_logits",
        "confidence",
        "confidence_logits",
        "semantic_logits",
    ):
        if key in scene:
            result[key] = _segment_mean(scene[key], final, fused_count, weights)
    result["normals"] = F.normalize(result["normals"], dim=-1, eps=1e-6)
    # max, rather than sum: duplicated queries do not create independent views.
    result["support"] = (
        scene["support"]
        .new_zeros(fused_count)
        .scatter_reduce_(0, final, scene["support"], reduce="amax", include_self=True)
    )
    result["valid"] = (
        scene["valid"]
        .new_zeros(fused_count, dtype=torch.long)
        .scatter_reduce_(0, final, scene["valid"].long(), reduce="amax", include_self=True)
        .bool()
    )
    result["state"] = scene["state"].new_full((fused_count,), 2)
    result["state"].scatter_(0, final, scene["state"])
    result["fusion_assignment"] = final
    result["unfused_count"] = count
    return result


def enforce_roi(scene, roi):
    """Expose only in-ROI predictions while retaining finite unknown queries.

    This is an inference-domain restriction, not a change to the predicted
    camera rays. Pre-fusion query_stage remains available for supervised loss.
    """
    bounds = torch.as_tensor(roi, dtype=scene["points"].dtype, device=scene["points"].device)
    inside = ((scene["points"] >= bounds[:, 0]) & (scene["points"] <= bounds[:, 1])).all(-1)
    result = dict(scene)
    result["in_roi"] = inside
    result["valid"] = scene["valid"] & inside
    result["state"] = torch.where(inside, scene["state"], torch.full_like(scene["state"], 2))
    result["confidence"] = scene["confidence"] * inside
    return result
