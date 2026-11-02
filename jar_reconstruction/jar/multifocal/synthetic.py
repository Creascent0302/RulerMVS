"""多焦距解析首命中渲染器：共享世界纹理、远墙/细杆/遮挡箱体。"""

from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset
from ..synthetic import _texture
from .data import pack_cameras
from .options import camera_ids


def camera_calibration(size, center, focal_multiplier=1.0):
    h, w = size
    focal = 0.5 * w * focal_multiplier
    K = torch.tensor([[focal, 0.0, (w - 1) / 2], [0.0, focal, (h - 1) / 2], [0.0, 0.0, 1.0]])
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T[:3, 3] = -T[:3, :3] @ torch.tensor(center)
    return K, T


def render_multifocal_camera(K, T, size, scene, seed=0, texture=True):
    """direction 的相机 z 分量为1，射线参数直接是米制 optical Z。"""
    h, w = size
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pixel = torch.stack([x, y, torch.ones_like(x)], -1).float()
    center = -T[:3, :3].T @ T[:3, 3]
    direction = pixel @ torch.linalg.inv(K).T @ T[:3, :3]
    depth = torch.full((h, w), float("inf"))
    normal = torch.zeros(h, w, 3)
    semantic = torch.full((h, w), -1, dtype=torch.long)
    planes = [
        (torch.tensor([1.0, 0.0, 0.0]), float(scene.get("wall_depth", 120.0))),
        (torch.tensor([0.0, 0.0, 1.0]), float(scene.get("ground_height", -2.0))),
    ]
    for label, (n, offset) in enumerate(planes):
        den = (direction * n).sum(-1)
        z = (offset - center @ n) / torch.where(den.abs() > 1e-8, den, torch.ones_like(den))
        hit = (den.abs() > 1e-8) & (z > 0.1) & (z < depth)
        facing = n.expand_as(direction) * torch.where(den > 0, -1.0, 1.0)[..., None]
        depth = torch.where(hit, z, depth)
        normal = torch.where(hit[..., None], facing, normal)
        semantic = torch.where(hit, label, semantic)
    default_boxes = [
        {"center": [45.0, 0.0, -0.2], "size": [0.4, 0.35, 2.4]},
        {"center": [24.0, 0.65, -0.35], "size": [1.5, 0.6, 2.6]},
        {"center": [35.0, -5.0, -0.5], "size": [3.0, 3.0, 3.0]},
    ]
    for label, box in enumerate(scene.get("boxes", default_boxes), 2):
        c, size3 = torch.tensor(box["center"]), torch.tensor(box["size"])
        if size3.shape != (3,) or bool((size3 <= 0).any()):
            raise ValueError("合成箱体 size 必须是三个正数")
        lo, hi = c - size3 / 2, c + size3 / 2
        parallel = direction.abs() < 1e-8
        safe = torch.where(parallel, torch.ones_like(direction), direction)
        a, b = (lo - center) / safe, (hi - center) / safe
        low = torch.where(parallel, -torch.inf, torch.minimum(a, b))
        high = torch.where(parallel, torch.inf, torch.maximum(a, b))
        outside = (parallel & ((center < lo) | (center > hi))).any(-1)
        enter, face = low.max(-1)
        leave = high.min(-1).values
        hit = ~outside & (leave >= enter) & (enter > 0.1) & (enter < depth)
        n = torch.zeros_like(direction)
        n.scatter_(-1, face[..., None], -direction.gather(-1, face[..., None]).sign())
        depth = torch.where(hit, enter, depth)
        normal = torch.where(hit[..., None], n, normal)
        semantic = torch.where(hit, label, semantic)
    valid = torch.isfinite(depth) & (depth < float(scene.get("max_depth", 200.0)))
    depth = torch.where(valid, depth, 0.0)
    point = center + direction * depth[..., None]
    rgb = _texture(point * 0.2, normal, seed) if texture else torch.full((h, w, 3), 0.5)
    # 细杆的世界材质与背景不同，但所有视角共享同一材质，不注入对应关系。
    rod_material = rgb * torch.tensor([1.0, 0.55, 0.35]) + torch.tensor([0.0, 0.05, 0.05])
    rgb = torch.where((semantic == 2)[..., None], rod_material, rgb)
    rgb = torch.where(valid[..., None], rgb, 0.0)
    return rgb.permute(2, 0, 1), depth, normal.permute(2, 0, 1), semantic


class MultifocalSyntheticDataset(Dataset):
    def __init__(self, cfg, split="train", load_targets=True):
        self.cfg = cfg
        self.p = cfg["synthetic"]
        self.ids = camera_ids(cfg)
        self.load_targets = load_targets
        self.seed_offset = 0 if split == "train" else 10000
        self.roles = [0] * 4 + ([1] * 2 if len(self.ids) == 6 else [])
        self.disabled = set(cfg["camera"].get("disabled_ids", []))
        self.missing = set(self.p.get("missing_camera_ids", []))
        if self.missing - set(self.ids):
            raise ValueError("synthetic.missing_camera_ids 含当前模式外的 ID")
        if (self.missing - self.disabled) and not cfg["camera"]["allow_missing"]:
            raise ValueError("缺失合成相机需要显式 allow_missing=true")
        self.missing |= self.disabled

    def __len__(self):
        return self.p.get("length", 8)

    def __getitem__(self, index):
        p = self.p
        h, w = p.get("height", 128), p.get("width", 192)
        tele_size = (p.get("tele_height", h), p.get("tele_width", w))
        centers = [
            [0.0, -0.6, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, -0.3],
            [0.0, -0.6, 0.0],
            [0.0, 0.6, 0.0],
        ]
        baseline = float(p.get("baseline_scale", 1.0))
        records = []
        scene = p.get("scene", {})
        for i, (cid, role) in enumerate(zip(self.ids, self.roles)):
            if cid in self.missing:
                records.append(None)
                continue
            size = tele_size if role else (h, w)
            K, T = camera_calibration(
                size,
                [v * baseline for v in centers[i]],
                p.get("tele_focal_multiplier", 4.0) if role else 1.0,
            )
            rgb, depth, normals, semantic = render_multifocal_camera(
                K,
                T,
                size,
                scene,
                p.get("seed", 0) + self.seed_offset + index,
                p.get("texture", True),
            )
            a = {
                "rgb": rgb[None],
                "K": K[None],
                "T_cam_from_rig": T[None],
                "view_mask": torch.ones(1, dtype=torch.bool),
                "image_mask": torch.ones(1, *size, dtype=torch.bool),
            }
            target = (
                {
                    "depth": depth[None],
                    "depth_valid": (depth > 0)[None],
                    "normals": normals[None],
                    "semantic": semantic[None],
                }
                if self.load_targets
                else {}
            )
            records.append({"inputs": a, "targets": target})
        metadata = {
            "sequence": "multifocal_synthetic_"
            + ("train" if self.seed_offset == 0 else "heldout_texture"),
            "frame_id": index,
            "timestamp": float(index),
            "camera_ids": self.ids,
            "role_names": [("wide", "tele")[r] for r in self.roles],
            "missing_camera_ids": sorted(self.missing),
            "disabled_camera_ids": sorted(self.disabled),
            "geometry_note": "共享世界纹理的首命中远墙、细杆和遮挡箱体；分割只进入 targets；非真实精度验证",
            "pixel_transform": "identity; right/bottom padding only",
        }
        return pack_cameras(records, self.ids, self.roles, metadata)
