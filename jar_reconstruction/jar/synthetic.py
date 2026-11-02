"""Analytic first-hit renderer of textured planes and axis-aligned boxes."""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from .data import collate_frames


def synthetic_rig(num_cameras=4, image_size=(64, 96), pure_rotation=False):
    h, w = image_size
    f = 0.5 * w
    K = torch.tensor([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]]).repeat(
        num_cameras, 1, 1
    )
    R = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T = torch.eye(4).repeat(num_cameras, 1, 1)
    T[:, :3, :3] = R
    centers = torch.zeros(num_cameras, 3)
    if num_cameras > 1 and not pure_rotation:
        centers[:, 1] = torch.linspace(-0.6, 0.6, num_cameras)
        centers[1::2, 2] = 0.20
    for i in range(num_cameras):
        if pure_rotation:
            a = torch.tensor((i - (num_cameras - 1) / 2) * 0.04)
            yaw = torch.tensor([[a.cos(), -a.sin(), 0], [a.sin(), a.cos(), 0], [0, 0, 1]])
            T[i, :3, :3] = R @ yaw
        T[i, :3, 3] = -T[i, :3, :3] @ centers[i]
    return K, T


def _texture(points, normals, seed):
    # World-space analytic texture is shared across all cameras and surfaces.
    # Band-limit the procedural texture for the tiny camera grids. Otherwise
    # point-sampled distant walls alias differently across cameras despite
    # identical continuous textures, obscuring geometric pipeline tests.
    p = points * 0.22
    phase = float(seed) * 0.371
    channels = []
    for k in range(3):
        val = 0.48 + 0.16 * torch.sin(p[..., 0] * (2.7 + k) + p[..., 1] * (4.3 + k) + phase)
        val += 0.15 * torch.sin(p[..., 1] * (11.1 + k) + p[..., 2] * (8.5 + k * 2) - phase)
        val += 0.10 * torch.cos(p[..., 0] * 6.3 + p[..., 2] * (16.4 + k))
        channels.append(val)
    shade = 0.8 + 0.2 * normals[..., 2].abs()
    return (torch.stack(channels, -1) * shade[..., None]).clamp(0, 1)


def render_scene(K, T, image_size=(64, 96), seed=0, texture=True):
    h, w = image_size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pixel = torch.stack([x, y, torch.ones_like(x)], -1).float()
    rgb_all, depth_all, normal_all, semantic_all = [], [], [], []
    for k, t in zip(K, T):
        center = -t[:3, :3].T @ t[:3, 3]
        direction = (pixel @ torch.linalg.inv(k).T) @ t[:3, :3]
        depth = torch.full((h, w), float("inf"))
        normals = torch.zeros(h, w, 3)
        semantic = torch.full((h, w), -1, dtype=torch.long)
        # Two infinite planes plus true solid boxes, ray-tested jointly for occlusion.
        planes = [(torch.tensor([1.0, -0.08, 0.0]), 16.0), (torch.tensor([0.0, 0.0, 1.0]), -1.5)]
        for label, (normal, offset) in enumerate(planes):
            den = (direction * normal).sum(-1)
            dist = (offset - (center * normal).sum()) / torch.where(
                den.abs() > 1e-8, den, torch.ones_like(den)
            )
            update = (den.abs() > 1e-8) & (dist > 0.1) & (dist < depth)
            n = normal / normal.norm()
            n = torch.where((direction * n).sum(-1)[..., None] > 0, -n, n)
            depth = torch.where(update, dist, depth)
            normals = torch.where(update[..., None], n, normals)
            semantic = torch.where(update, label, semantic)
        boxes = [
            (torch.tensor([6.0, -0.8, -0.30]), torch.tensor([1.6, 1.4, 2.4])),
            (torch.tensor([9.0, 2.1, -0.7]), torch.tensor([2.0, 1.2, 1.6])),
            (torch.tensor([4.2, 0.65, -0.05]), torch.tensor([0.18, 0.18, 2.7])),
        ]
        for label, (c, size) in enumerate(boxes, 2):
            lo, hi = c - size / 2, c + size / 2
            parallel = direction.abs() < 1e-8
            safe = torch.where(parallel, torch.ones_like(direction), direction)
            a, b = (lo - center) / safe, (hi - center) / safe
            lower, upper = torch.minimum(a, b), torch.maximum(a, b)
            lower = torch.where(parallel, torch.full_like(lower, -float("inf")), lower)
            upper = torch.where(parallel, torch.full_like(upper, float("inf")), upper)
            outside = (parallel & ((center < lo) | (center > hi))).any(-1)
            enter, face = lower.max(-1)
            leave = upper.min(-1).values
            update = ~outside & (leave >= enter) & (enter > 0.1) & (enter < depth)
            n = torch.zeros_like(direction)
            n.scatter_(-1, face[..., None], -direction.gather(-1, face[..., None]).sign())
            depth = torch.where(update, enter, depth)
            normals = torch.where(update[..., None], n, normals)
            semantic = torch.where(update, label, semantic)
        valid = torch.isfinite(depth) & (depth < 80)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        points = center + direction * depth[..., None]
        rgb = _texture(points, normals, seed) if texture else torch.full((h, w, 3), 0.5)
        rgb = torch.where(valid[..., None], rgb, torch.zeros_like(rgb))
        rgb_all.append(rgb.permute(2, 0, 1))
        depth_all.append(depth)
        normal_all.append(normals.permute(2, 0, 1))
        semantic_all.append(semantic)
    return (
        torch.stack(rgb_all),
        torch.stack(depth_all),
        torch.stack(normal_all),
        torch.stack(semantic_all),
    )


class SyntheticRigDataset(Dataset):
    def __init__(
        self,
        length=8,
        image_size=(64, 96),
        num_cameras=4,
        seed=0,
        texture=True,
        pure_rotation=False,
    ):
        self.length, self.image_size, self.num_cameras, self.seed = (
            length,
            image_size,
            num_cameras,
            seed,
        )
        self.texture, self.pure_rotation = texture, pure_rotation

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        K, T = synthetic_rig(self.num_cameras, self.image_size, self.pure_rotation)
        rgb, depth, normals, semantic = render_scene(
            K, T, self.image_size, self.seed + index, self.texture
        )
        n, _, h, w = rgb.shape
        return {
            "inputs": {
                "rgb": rgb,
                "K": K,
                "T_cam_from_rig": T,
                "view_mask": torch.ones(n, dtype=torch.bool),
                "image_mask": torch.ones(n, h, w, dtype=torch.bool),
            },
            "targets": {
                "depth": depth,
                "depth_valid": depth > 0,
                "normals": normals,
                "semantic": semantic,
            },
            "metadata": {
                "sequence": "synthetic",
                "frame_id": index,
                "timestamp": float(index),
                "geometry_note": "Fixed geometric scene; seed varies world-space texture only",
            },
        }


def make_synthetic_batch(B=1, N=4, H=64, W=96, seed=0, texture=True, pure_rotation=False):
    ds = SyntheticRigDataset(B, (H, W), N, seed, texture, pure_rotation)
    return collate_frames([ds[i] for i in range(B)])
