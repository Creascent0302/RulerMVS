"""Prediction-only spatial queries and fixed-ROI evaluation probes.

An affine zero plane is a surface patch, never a filled voxel. Unrepresented
space stays UNKNOWN. These queries are not a high-resolution dense decoder.
"""

import torch
from .common import FREE, SURFACE, UNKNOWN, coarse_evidence


def roi_probes(roi, spacing=2.0, device=None, dtype=torch.float32):
    roi = torch.as_tensor(roi, device=device, dtype=dtype)
    if spacing <= 0:
        raise ValueError("probe spacing must be positive")
    counts = torch.ceil((roi[:, 1] - roi[:, 0]) / spacing).long().clamp_min(1)
    size = (roi[:, 1] - roi[:, 0]) / counts
    axes = [
        roi[k, 0] + (torch.arange(int(counts[k]), device=device, dtype=dtype) + 0.5) * size[k]
        for k in range(3)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), size


@torch.no_grad()
def query_space(points, outputs, inputs, batch_index=0, roi=None, surface_band=0.1, chunk=512):
    """Return [Q] FREE/SURFACE/UNKNOWN, support and confidence without targets.

    Surface queries use small oriented surfel disks. Free space in all models
    additionally requires calibrated, confidence-gated first-hit evidence.
    Sampling finite input images cannot certify a continuum of empty space.
    """
    scene = outputs["scenes"][batch_index]
    if roi is None:
        roi = scene.get("aux", {}).get("roi")
    bounds = None if roi is None else torch.as_tensor(roi, device=points.device, dtype=points.dtype)
    inside = torch.ones(len(points), device=points.device, dtype=torch.bool)
    if bounds is not None:
        inside = ((points >= bounds[:, 0]) & (points < bounds[:, 1])).all(-1)
    pieces = []
    for p in points.split(chunk):
        if "ray_evidence" in scene:
            from .ray_evidence import query_ray_evidence

            evidence = query_ray_evidence(
                p, scene["ray_evidence"], inputs, batch_index, band=surface_band
            )
        else:
            evidence = coarse_evidence(
                p, inputs, outputs["coarse"], batch_index, tolerance=surface_band
            )
        state = torch.full((len(p),), UNKNOWN, device=p.device, dtype=torch.long)
        support = torch.zeros_like(state)
        confidence = p.new_zeros(len(p))
        free = evidence["free"]
        hit = torch.zeros_like(free)
        mask = scene["valid"] & (scene["state"] == SURFACE)
        centers = scene["points"][mask]
        normals = scene["normals"][mask]
        scales = scene["scales"][mask]
        confs = scene["confidence"][mask]
        sup = scene["support"][mask]
        # Evaluation uses bounded blocks; no Q-by-Q attention or persistent
        # dense volume is introduced into any trainable architecture.
        for start in range(0, len(centers), chunk):
            x = centers[start : start + chunk]
            n = normals[start : start + chunk]
            delta = p[:, None] - x[None]
            normal_distance = (delta * n[None]).sum(-1)
            tangent_squared = (delta.square().sum(-1) - normal_distance.square()).clamp_min(0)
            radius = scales[start : start + chunk].amax(-1)
            match = (normal_distance.abs() <= surface_band) & (
                tangent_squared <= radius.square()[None]
            )
            quality = match * confs[start : start + chunk][None]
            best, index = quality.max(-1)
            improve = best > confidence
            support = torch.where(improve, sup[start : start + chunk][index], support)
            confidence = torch.maximum(confidence, best)
            hit = hit | match.any(-1)
        state[free] = FREE
        state[hit] = SURFACE
        support = torch.where(free & ~hit, evidence["support"], support)
        confidence = torch.where(free & ~hit, evidence["confidence"], confidence)
        pieces.append((state, support, confidence))
    if not pieces:
        return {
            "state": torch.empty(0, device=points.device, dtype=torch.long),
            "support": torch.empty(0, device=points.device, dtype=torch.long),
            "confidence": points.new_empty(0),
        }
    state, support, confidence = [torch.cat([piece[k] for piece in pieces]) for k in range(3)]
    return {
        "state": torch.where(inside, state, UNKNOWN),
        "support": support * inside,
        "confidence": confidence * inside,
    }


@torch.no_grad()
def evaluate_space(
    outputs, inputs, targets, roi, spacing=2.0, surface_band=0.1, surface_probes=1024
):
    from .losses import label_query_points, target_cloud

    probes, size = roi_probes(roi, spacing, device=inputs["rgb"].device)
    confusion = torch.zeros(3, 3, device=probes.device, dtype=torch.long)
    hit_counts = torch.zeros(3, device=probes.device, dtype=torch.long)
    for b in range(len(outputs["scenes"])):
        pred = query_space(probes, outputs, inputs, b, roi, surface_band)["state"]
        label = label_query_points(probes, inputs, targets, b, surface_band)["state"]
        confusion += torch.bincount(label * 3 + pred, minlength=9).reshape(3, 3)
        # A uniform grid can miss every thin object. A separate, identically
        # sampled set of observed GT first hits measures recall/false free.
        # These evaluation locations never enter the reconstruction network.
        cloud, _ = target_cloud(inputs, targets, b)
        bounds = torch.as_tensor(roi, device=cloud.device, dtype=cloud.dtype)
        cloud = cloud[((cloud >= bounds[:, 0]) & (cloud < bounds[:, 1])).all(-1)]
        if len(cloud) > surface_probes:
            cloud = cloud[
                torch.linspace(0, len(cloud) - 1, surface_probes, device=cloud.device).long()
            ]
        if len(cloud):
            predicted_hit = query_space(cloud, outputs, inputs, b, roi, surface_band)["state"]
            hit_counts += torch.bincount(predicted_hit, minlength=3)
    c = confusion.float()
    total = float(c.sum())
    known = float(c[:2].sum())
    hit_total = int(hit_counts.sum())
    return {
        "spatial_probe_count": int(c.sum()),
        "spatial_probe_cell_volume_m3": float(size.prod()),
        "spatial_unknown_fraction": float(c[:, UNKNOWN].sum()) / max(total, 1),
        "spatial_observed_state_accuracy": float(c[FREE, FREE] + c[SURFACE, SURFACE])
        / max(known, 1),
        "spatial_observed_surface_iou": float(c[SURFACE, SURFACE])
        / max(float(c[:2, SURFACE].sum() + c[SURFACE, :].sum() - c[SURFACE, SURFACE]), 1),
        "spatial_wrong_free_on_surface": (
            float(c[SURFACE, FREE]) / float(c[SURFACE].sum()) if c[SURFACE].sum() > 0 else None
        ),
        "spatial_free_on_unknown": (
            float(c[UNKNOWN, FREE]) / float(c[UNKNOWN].sum()) if c[UNKNOWN].sum() > 0 else None
        ),
        "spatial_gt_free_count": int(c[FREE].sum()),
        "spatial_gt_surface_count": int(c[SURFACE].sum()),
        "spatial_gt_unknown_count": int(c[UNKNOWN].sum()),
        "spatial_known_prediction_count": int(c[:, :2].sum()),
        "spatial_gt_hit_probe_count": hit_total,
        "spatial_hit_surface_recall": int(hit_counts[SURFACE]) / hit_total if hit_total else None,
        "spatial_hit_false_free_rate": int(hit_counts[FREE]) / hit_total if hit_total else None,
        "spatial_hit_unknown_fraction": int(hit_counts[UNKNOWN]) / hit_total if hit_total else None,
    }
