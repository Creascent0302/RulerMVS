"""分辨率感知表面保留和严格有限射线证据，不将像素锥视作已证空闲。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from jar.geometry import project
from jar.ray_evidence import build_ray_evidence


def resolution_aware_fuse(scene, cell_size=0.05, normal_cosine=0.97):
    """Conservative hash deduplication; keep finest real sample as representative.

    A group is merged only if EVERY member is an accepted surface, oriented
    normals agree, and its extent fits both the smallest footprint AND the
    independently estimated position uncertainty. Different depth modes never
    merge merely because a coarse background sample has a large footprint.
    Choosing a real sample preserves its ray anchor, attributes and provenance.
    """
    points = scene["points"]
    q = len(points)
    result = dict(scene)
    if q == 0 or cell_size <= 0:
        result["fusion_assignment"] = torch.arange(q, device=points.device)
        return result
    normals = F.normalize(scene["normals"].detach(), dim=-1, eps=1e-6)
    footprint = scene["scales"].detach().abs().amin(-1).clamp_min(1e-6)
    aux = scene.get("aux", {})
    sigma = aux.get("depth_sigma", points.new_full((q,), 0.001)).detach()
    if sigma.ndim > 1:
        sigma = sigma.reshape(q, -1).amin(-1)
    # Footprints and sigma are separate quantities, never substituted for one
    # another. Hash bins merely produce candidate groups, not merge decisions.
    keys = torch.cat(
        (torch.floor(points.detach() / cell_size).long(), torch.round(normals * 4).long()), -1
    )
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    groups = int(inverse.max()) + 1
    lower = points.new_full((groups, 3), float("inf"))
    upper = points.new_full((groups, 3), -float("inf"))
    idx = inverse[:, None].expand(-1, 3)
    lower.scatter_reduce_(0, idx, points.detach(), reduce="amin", include_self=True)
    upper.scatter_reduce_(0, idx, points.detach(), reduce="amax", include_self=True)
    nsum = normals.new_zeros(groups, 3).index_add_(0, inverse, normals)
    nmean = F.normalize(nsum, dim=-1, eps=1e-6)
    cosine = (normals * nmean[inverse]).sum(-1)
    minimum_cosine = points.new_ones(groups).scatter_reduce_(
        0, inverse, cosine, reduce="amin", include_self=True
    )
    tolerance = torch.minimum(footprint * 0.25, sigma.clamp_min(1e-5) * 2)
    tolerance = tolerance.clamp_max(cell_size * 0.25)
    minimum_tolerance = points.new_full((groups,), float("inf")).scatter_reduce_(
        0, inverse, tolerance, reduce="amin", include_self=True
    )
    accepted = scene["valid"].bool() & (scene["state"] == 1) & torch.isfinite(sigma)
    all_accepted = (
        torch.ones(groups, device=points.device, dtype=torch.long)
        .scatter_reduce_(0, inverse, accepted.long(), reduce="amin", include_self=True)
        .bool()
    )
    safe = all_accepted & (minimum_cosine >= normal_cosine)
    safe &= (upper - lower).norm(dim=-1) <= minimum_tolerance
    labels = torch.where(safe[inverse], inverse, groups + torch.arange(q, device=points.device))
    _, assignment = torch.unique(labels, sorted=True, return_inverse=True)
    count = int(assignment.max()) + 1
    min_footprint = points.new_full((count,), float("inf")).scatter_reduce_(
        0, assignment, footprint, reduce="amin", include_self=True
    )
    finest = footprint == min_footprint[assignment]
    confidence = scene["confidence"].detach().float()
    quality = torch.where(finest, confidence, -float("inf"))
    best_quality = points.new_full((count,), -float("inf")).scatter_reduce_(
        0, assignment, quality, reduce="amax", include_self=True
    )
    eligible = finest & (quality == best_quality[assignment])
    candidates = torch.where(eligible, torch.arange(q, device=points.device), q)
    representatives = torch.full(
        (count,), q, device=points.device, dtype=torch.long
    ).scatter_reduce_(0, assignment, candidates, reduce="amin", include_self=True)
    for key, value in scene.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == q:
            result[key] = value[representatives]
    result["aux"] = {
        key: (
            value[representatives]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == q
            else value
        )
        for key, value in aux.items()
    }
    result["aux"]["fusion_representative_indices"] = representatives
    result["aux"]["fusion_original_ref_index"] = aux.get("ref_index")
    result["aux"]["fusion_original_ref_uv"] = aux.get("ref_uv")
    result["fusion_assignment"] = assignment
    result["unfused_count"] = q
    # Selecting a sample preserves its actual independent camera count; no
    # addition of repeated-query counts or averaging of foreground/background.
    return result


@torch.no_grad()
def build_multifocal_ray_evidence(stage, inputs, b, roi, conflict_band_m=0.05):
    """Verified hits only, exact-ray free tests, fine-foreground veto of background.

    Original pixel-centre rays remain sparse. Off-axis points within a pixel's
    half-pixel cone are UNKNOWN, because a low-resolution background pixel is
    not evidence that the entire cone is empty. A verified surface in front of
    a background hit invalidates that background pixel's free evidence even
    when the foreground is subpixel in that view. It does not fabricate a new
    first hit there; the foreground retains its own true reference anchor.
    """
    evidence = build_ray_evidence(stage, inputs, b, roi)
    evidence["source"] = "mf_verified_reference_queries_only"
    evidence["query_pixel_tolerance_px"] = 1e-3
    h, w = inputs["rgb"].shape[-2:]
    K, T = inputs["K"][b], inputs["T_cam_from_rig"][b]
    n = len(K)
    view = inputs.get("view_mask")
    view = None if view is None else view[b]
    image = inputs.get("image_mask")
    image = None if image is None else image[b]
    valid = stage["valid"] & (stage["state"] == 1) & (stage["support"] >= 2)
    valid &= torch.isfinite(stage["confidence"]) & (stage["confidence"] > 0)
    points = stage["points"][valid]
    conflicts = torch.zeros(n * h * w, dtype=torch.bool, device=points.device)
    camera = torch.arange(n, device=points.device)[:, None]
    for piece in points.split(2048):
        uv, z, visible, _ = project(piece.float(), K, T, (h, w), view, image)
        xy = uv.round().long()
        x, y = xy[..., 0].clamp(0, w - 1), xy[..., 1].clamp(0, h - 1)
        first = evidence["depth"][camera, y, x]
        has_hit = evidence["valid"][camera, y, x]
        foreground = visible & has_hit & (z < first - conflict_band_m)
        foreground &= (uv - xy).abs().amax(-1) <= 0.5
        keys = (camera * h + y) * w + x
        conflicts[keys[foreground]] = True
    conflicts = conflicts.reshape(n, h, w)
    evidence["valid"] = evidence["valid"] & ~conflicts
    evidence["confidence"] = evidence["confidence"] * evidence["valid"]
    evidence["support"] = evidence["support"] * evidence["valid"]
    evidence["foreground_conflict_pixels"] = conflicts
    evidence["foreground_conflict_count"] = int(conflicts.sum())
    evidence["represented_pixels"] = int(evidence["valid"].sum())
    evidence["free_policy"] = "verified_exact_rays_with_subpixel_foreground_veto"
    return evidence
