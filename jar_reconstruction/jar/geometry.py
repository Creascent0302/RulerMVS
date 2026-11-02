"""Calibrated geometry. Integer pixel centers; all grid_sample uses align_corners=False.

A stride-s CNN with odd padded kernels has original pixel center s*j at feature j.
Image resizing is a different operation and explicitly uses the half-pixel K update.
"""

import torch
import torch.nn.functional as F


def _dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def pixel_grid(height, width, device=None, dtype=torch.float32, stride=1):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), -1) * stride


def normalize_grid(uv, height, width, stride=1):
    scale = uv.new_tensor([width, height])
    return 2 * (uv / stride + 0.5) / scale - 1


def sample_maps(maps, uv, stride=1, mode="bilinear", padding_mode="zeros"):
    """[N,C,H,W], [N,Q,2] -> [N,Q,C]; nonfinite coordinates sample zeros."""
    grid = normalize_grid(uv, maps.shape[-2], maps.shape[-1], stride)
    grid = torch.nan_to_num(grid, nan=1e6, posinf=1e6, neginf=-1e6)
    # CPU grid_sample does not support half. Geometry remains at least float32.
    dt = _dtype(maps)
    sampled = F.grid_sample(
        maps.to(dt),
        grid.to(dt).unsqueeze(2),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    return sampled.squeeze(-1).transpose(1, 2)


def project(points, K, T, image_hw, view_mask=None, image_mask=None):
    """Points [Q,3], cameras [N,...] -> uv,z,valid,J in [N,Q,...].

    J is the derivative in ORIGINAL image pixels / rig meters. Invalid
    projections are finite but must always be used together with valid.
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        dt = _dtype(points)
        X = points.to(dt)
        K = K.to(dt)
        T = T.to(dt)
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        xc = torch.einsum("nij,qj->nqi", R, X) + t[:, None]
        z = xc[..., 2]
        zs = z.clamp_min(1e-4)
        p = torch.einsum("nij,nqj->nqi", K, xc)
        uv = p[..., :2] / zs[..., None]
        h, w = image_hw
        valid = (z > 1e-4) & torch.isfinite(xc).all(-1) & torch.isfinite(uv).all(-1)
        valid = (
            valid
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= w - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= h - 1)
        )
        if view_mask is not None:
            valid = valid & view_mask[:, None].bool()
        Jc = K[:, :2, :][:, None].expand(-1, len(X), -1, -1) / zs[..., None, None]
        e3 = xc.new_tensor([0.0, 0.0, 1.0])
        Jc = Jc - p[..., :2, None] * e3 / (zs[..., None, None] ** 2)
        J = torch.einsum("nqij,njk->nqik", Jc, R)
        uv = torch.nan_to_num(uv, nan=0.0, posinf=1e6, neginf=-1e6)
        J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        if image_mask is not None:
            valid = valid & (
                sample_maps(image_mask[:, None].float(), uv, mode="bilinear")[..., 0] > 0.999
            )
        return uv, torch.nan_to_num(z), valid, J


def unproject(uv, z, K, T):
    with torch.autocast(device_type=uv.device.type, enabled=False):
        dt = _dtype(uv)
        uv = uv.to(dt)
        z = z.to(dt)
        K = K.to(dt)
        T = T.to(dt)
        homogeneous = torch.cat([uv, torch.ones_like(uv[..., :1])], -1)
        camera_ray = homogeneous @ torch.linalg.inv(K).T
        # Standard K has last row (0,0,1); normalization makes z semantics explicit.
        camera_ray = camera_ray / camera_ray[..., 2:].clamp_min(1e-8)
        direction = camera_ray @ T[:3, :3]
        center = -T[:3, :3].T @ T[:3, 3]
        return center + z[..., None] * direction, direction


def camera_centers(T):
    with torch.autocast(device_type=T.device.type, enabled=False):
        T = T.to(_dtype(T))
        return -torch.einsum("...ji,...j->...i", T[..., :3, :3], T[..., :3, 3])


def resize_crop_intrinsics(K, scale_xy=(1.0, 1.0), crop_xy=(0.0, 0.0)):
    """Resize with align_corners=False, THEN crop (left,top) in resized pixels."""
    sx, sy = scale_xy
    left, top = crop_xy
    A = K.new_tensor(
        [[sx, 0.0, (sx - 1) / 2 - left], [0.0, sy, (sy - 1) / 2 - top], [0.0, 0.0, 1.0]]
    )
    return A @ K


def spatial_gradients(maps):
    """Central derivatives per feature pixel; replicate padding, finite for 1x1."""
    p = F.pad(maps, (1, 1, 1, 1), mode="replicate")
    gx = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) * 0.5
    gy = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) * 0.5
    return gx, gy


def image_texture(rgb):
    # RGB gradients, not learned features: constant images cannot create evidence.
    gx, gy = spatial_gradients(rgb.float())
    magnitude = (gx.square() + gy.square()).mean(1, keepdim=True).sqrt()
    return F.avg_pool2d(magnitude, 5, stride=1, padding=2)


def safe_masked_softmax(logits, mask, dim=-1):
    masked = logits.masked_fill(~mask, -1e4)
    p = masked.softmax(dim) * mask.to(logits.dtype)
    return p / p.sum(dim, keepdim=True).clamp_min(1e-8)


def validate_inputs(inputs):
    rgb, K, T = inputs["rgb"], inputs["K"], inputs["T_cam_from_rig"]
    if rgb.ndim != 5 or rgb.shape[2] != 3:
        raise ValueError("rgb must be [B,N,3,H,W]")
    B, N = rgb.shape[:2]
    if K.shape != (B, N, 3, 3) or T.shape != (B, N, 4, 4):
        raise ValueError("camera tensor shape mismatch")
    if N < 1:
        raise ValueError("at least one camera slot is required")
    if inputs.get("view_mask", torch.ones(B, N)).shape != (B, N):
        raise ValueError("view_mask shape mismatch")
    for key in ["rgb", "K", "T_cam_from_rig"]:
        if not torch.isfinite(inputs[key]).all():
            raise ValueError(f"nonfinite {key}")
    if not ((K[..., 0, 0] > 0) & (K[..., 1, 1] > 0)).all():
        raise ValueError("focal lengths must be positive")
    if not torch.allclose(K[..., 2, :], K.new_tensor([0.0, 0.0, 1.0]).expand(B, N, 3), atol=1e-5):
        raise ValueError("K must use standard pinhole last row")
    if not torch.allclose(
        T[..., 3, :], T.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(B, N, 4), atol=1e-5
    ):
        raise ValueError("T must be homogeneous")
    with torch.autocast(device_type=T.device.type, enabled=False):
        R = T[..., :3, :3].to(_dtype(T))
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        if not torch.allclose(R.transpose(-1, -2) @ R, eye, atol=2e-4) or not torch.allclose(
            torch.linalg.det(R), R.new_ones(B, N), atol=2e-4
        ):
            raise ValueError(
                "T rotation must be right-handed and orthonormal; convert CARLA frames explicitly"
            )


def reference_rays(points, ref_index, ref_uv, K, T):
    """Return rig ray origin, dX/dZ, and optical depth for mixed references."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
        K, T, points, ref_uv = K.to(dtype), T.to(dtype), points.to(dtype), ref_uv.to(dtype)
        rotation, translation = T[ref_index, :3, :3], T[ref_index, :3, 3]
        homogeneous = torch.cat((ref_uv, torch.ones_like(ref_uv[:, :1])), -1)
        ray_cam = torch.linalg.solve(K[ref_index], homogeneous.unsqueeze(-1)).squeeze(-1)
        ray_cam = ray_cam / ray_cam[:, 2:].clamp_min(1e-8)
        direction = (rotation.transpose(-1, -2) @ ray_cam.unsqueeze(-1)).squeeze(-1)
        center = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
        depth = (rotation @ points.unsqueeze(-1)).squeeze(-1)[:, 2] + translation[:, 2]
    return center, direction, depth
