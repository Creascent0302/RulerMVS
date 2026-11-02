"""空间证据与公共表面输出。"""

import torch
from .geometry import camera_centers, project, sample_maps

FREE, SURFACE, UNKNOWN = 0, 1, 2


def coarse_evidence(points, inputs, coarse, batch_index, tolerance=0.1):
    b = batch_index
    rgb = inputs["rgb"]
    N = rgb.shape[1]
    vm = inputs.get("view_mask")
    vm = None if vm is None else vm[b]
    im = inputs.get("image_mask")
    im = None if im is None else im[b]
    uv, z, valid, J = project(
        points, inputs["K"][b], inputs["T_cam_from_rig"][b], rgb.shape[-2:], vm, im
    )

    def read(key):
        return sample_maps(
            coarse[key][b, :, None].float(), uv, stride=coarse.get("stride", 16), mode="nearest"
        )[..., 0]

    depth = read("depth")
    conf = read("confidence")
    support = read("support")
    texture = read("texture")
    minimum_support = coarse.get("min_support", 2)
    reliable = (
        valid
        & (support >= minimum_support)
        & (conf >= coarse.get("evidence_threshold", 0.15))
        & (texture >= coarse.get("texture_threshold", 0.002))
        & (depth > 0)
    )
    tol = torch.as_tensor(tolerance, device=points.device, dtype=points.dtype)
    near = (z - depth).abs() <= tol
    matches = near & reliable
    c = camera_centers(inputs["T_cam_from_rig"][b]).float()
    mass = matches.sum(0).clamp_min(1)
    mean = (matches[..., None] * c[:, None]).sum(0) / mass[:, None]
    variance = (((c[:, None] - mean[None]) ** 2).sum(-1) * matches).sum(0) / mass
    surface = (matches.sum(0) >= minimum_support) & (variance > 1e-8)
    infront = reliable & (z < depth - tol)
    # A contradiction from a near-hit view is not certified empty space.
    free = infront.any(0) & ~matches.any(0) & ~surface
    known = surface | free
    supported = torch.where(
        surface, matches.sum(0), torch.where(free, infront.sum(0), torch.zeros_like(mass))
    )
    confidence = (conf * reliable).amax(0) * known
    return dict(
        free=free,
        surface=surface,
        known=known,
        support=supported,
        confidence=confidence,
        texture=(texture * valid).sum(0) / valid.sum(0).clamp_min(1),
    )


def make_scene(
    points,
    normals,
    scales,
    features,
    state_logits,
    evidence,
    split_logits=None,
    centers=None,
    sizes=None,
):
    length = normals.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(normals)
    fallback[..., 2] = 1
    normals = torch.where(length > 1e-6, normals / length.clamp_min(1e-6), fallback)
    pred = state_logits.argmax(-1)
    state = torch.full_like(pred, UNKNOWN)
    state = torch.where((pred == FREE) & evidence["free"], FREE, state)
    state = torch.where((pred == SURFACE) & evidence["surface"], SURFACE, state)
    probability = state_logits.float().softmax(-1)
    confidence = evidence["confidence"].clamp(0, 1) * probability[:, SURFACE]
    return dict(
        points=points,
        normals=normals,
        scales=scales.clamp_min(1e-5),
        valid=state == SURFACE,
        support=evidence["support"].long(),
        confidence=confidence,
        state_logits=state_logits,
        state=state,
        centers=points if centers is None else centers,
        sizes=scales * 2 if sizes is None else sizes,
        features=features,
        split_logits=features.new_zeros(len(points)) if split_logits is None else split_logits,
    )
