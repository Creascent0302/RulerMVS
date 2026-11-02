"""Explicit, label-separated frame data contract; no CARLA runtime dependency."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def carla_transform_to_cv(T_rig_from_camera_carla):
    """CARLA local camera->rig (x forward,y right,z up) to RH rig->CV."""
    T = torch.as_tensor(T_rig_from_camera_carla, dtype=torch.float32)
    if T.shape != (4, 4):
        raise ValueError("CARLA transform must be a 4x4 camera-to-rig matrix")
    rig_to_carla = torch.diag(T.new_tensor([1.0, -1.0, 1.0, 1.0]))
    local_to_cv = T.new_tensor([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    return local_to_cv @ torch.linalg.inv(T) @ rig_to_carla


def decode_carla_depth(raw, channel_order="RGB", far=1000.0):
    """Decode ORIGINAL 24-bit sensor bytes, never a color-converted preview."""
    a = torch.as_tensor(np.asarray(raw).copy() if not torch.is_tensor(raw) else raw)
    if a.dtype != torch.uint8 or a.shape[-1] not in (3, 4):
        raise ValueError("Raw CARLA depth requires uint8 RGB/BGRA encoded bytes")
    a = a.float()
    if channel_order in ("RGB", "RGBA"):
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
    elif channel_order in ("BGR", "BGRA"):
        b, g, r = a[..., 0], a[..., 1], a[..., 2]
    else:
        raise ValueError("Specify RGB, RGBA, BGR or BGRA channel_order")
    return (r + 256 * g + 65536 * b) * (float(far) / 16777215.0)


def to_optical_depth(depth, K, depth_type):
    if depth_type == "optical_z":
        return depth
    if depth_type != "ray_range":
        raise ValueError("depth_type must explicitly be optical_z or ray_range")
    h, w = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij"
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], -1).to(depth.dtype)
    rays = pixels @ torch.linalg.inv(K).T
    return depth / torch.linalg.vector_norm(rays, dim=-1).clamp_min(1e-8)


def _read_image(path):
    from PIL import Image

    return np.asarray(Image.open(path)).copy()


def add_matching_images(item, image_size):
    """从同一原始RGB派生匹配图，不读取标签，也不修改场景K/T。"""
    if image_size is None:
        return item
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("data.matching_image_size须为正的[H,W]")
    size = tuple(map(int, image_size))
    inputs = item["inputs"]
    old_size = inputs["rgb"].shape[-2:]
    if size == tuple(old_size):
        rgb, mask = inputs["rgb"], inputs["image_mask"]
    else:
        rgb = F.interpolate(inputs["rgb"], size=size, mode="bilinear", align_corners=False, antialias=True)
        mode = "area" if all(a <= b for a, b in zip(size, old_size)) else "nearest-exact"
        mask = F.interpolate(inputs["image_mask"][:, None].float(), size=size, mode=mode)[:, 0] > 0.999
    return {**item, "inputs": {**inputs, "matching_rgb": rgb, "matching_image_mask": mask}}


class MatchingImageDataset(Dataset):
    """合成接口复用同一RGB的缩放；不声称插值能创造原图缺失的细节。"""

    def __init__(self, dataset, image_size):
        self.dataset, self.image_size = dataset, image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return add_matching_images(self.dataset[index], self.image_size)


class CalibratedMultiviewDataset(Dataset):
    """公共多视图：保留真实相机数量，从原图分别产生主图和匹配图。"""

    def __init__(self, dataset, settings):
        self.dataset = dataset
        self.settings = settings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from .data_transforms import resize_frame

        sample = self.dataset[index]
        if sample["inputs"]["rgb"].shape[0] != self.settings["num_views"]:
            raise ValueError("公共数据真实相机数与data.num_views不一致")
        sample = add_matching_images(sample, self.settings.get("matching_image_size"))
        return resize_frame(sample, self.settings.get("image_size"),
                            depth_resize_mode=self.settings.get("depth_resize_mode", "nearest-exact"))


class CarlaFrameDataset(Dataset):
    """JSONL frames with synchronized per-camera inputs and separate labels."""

    def __init__(self, root, manifest, split="train", load_targets=True, geometry_units="meters"):
        self.load_targets = bool(load_targets)
        if geometry_units not in ("meters", "scene_unit"):
            raise ValueError("geometry_units必须显式为meters或scene_unit")
        self.geometry_units = geometry_units
        if not root or not manifest:
            raise ValueError(
                f"Real {split} mode requires nonempty data.root and {split}_manifest; use --synthetic explicitly for synthetic data"
            )
        self.root = Path(root)
        path = Path(manifest)
        path = path if path.is_absolute() else self.root / path
        self.frames = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not self.frames:
            raise ValueError("Manifest has no frames")
        seen = set()
        for f in self.frames:
            if f.get("geometry_units", "meters") != geometry_units:
                raise ValueError("清单与配置的几何单位不一致；非米制数据须显式配置data.geometry_units=scene_unit")
            key = (f["sequence"], str(f["frame_id"]))
            if key in seen:
                raise ValueError(f"Duplicate sequence/frame_id: {key}")
            seen.add(key)
            if not f.get("cameras"):
                raise ValueError("A frame needs at least one camera")
            for c in f["cameras"]:
                if str(c["frame_id"]) != str(f["frame_id"]):
                    raise ValueError("Camera frame_id does not match synchronized frame")
                if f.get("sync_basis") == "frame_id" and f.get("timestamp") is None:
                    if c.get("timestamp") is not None:
                        raise ValueError("frame_id-only同步不能混入无帧时间参考的camera timestamp")
                elif abs(float(c["timestamp"]) - float(f["timestamp"])) > 1e-5:
                    raise ValueError("Camera timestamps are not synchronized")
                label = c.get("targets", {}) if self.load_targets else {}
                if "depth" in label and label.get("depth_units", "meters") != geometry_units:
                    raise ValueError("深度与相机平移必须使用相同单位")
                if label and str(label.get("frame_id", "")) != str(f["frame_id"]):
                    raise ValueError("Target frame_id must match its camera frame")
                if "depth" in label and label.get("depth_type") not in ("optical_z", "ray_range"):
                    raise ValueError(
                        "深度定义未核实：每个depth标签须明确 optical_z 或 ray_range；请先核对采集脚本或同帧传感器证据，推理可不读取targets"
                    )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        rgbs, intrinsics, extrinsics, depths, masks, semantics, instances = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        depth_present, semantic_present, instance_present = False, False, False
        hw = None
        for c in frame["cameras"]:
            rgb = (
                torch.from_numpy(_read_image(self.root / c["rgb"])[..., :3])
                .float()
                .permute(2, 0, 1)
                / 255
            )
            if rgb.shape[0] != 3 or (hw is not None and rgb.shape[-2:] != hw):
                raise ValueError(
                    "All cameras in a frame must have equal RGB image size; rectify/resize explicitly"
                )
            hw = rgb.shape[-2:]
            K = torch.tensor(c["K"], dtype=torch.float32)
            T = torch.tensor(c["T_cam_from_rig"], dtype=torch.float32)
            if (
                K.shape != (3, 3)
                or T.shape != (4, 4)
                or not torch.isfinite(K).all()
                or not torch.isfinite(T).all()
            ):
                raise ValueError("Invalid calibrated K/T matrices")
            if not torch.allclose(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
                raise ValueError("T must be a homogeneous world-to-camera transform")
            if (
                not torch.allclose(T[:3, :3] @ T[:3, :3].T, torch.eye(3), atol=1e-4)
                or torch.det(T[:3, :3]) < 0.999
            ):
                raise ValueError(
                    "Internal CV-from-right-handed-rig rotation must be proper orthonormal"
                )
            if not self.load_targets:
                rgbs.append(rgb)
                intrinsics.append(K)
                extrinsics.append(T)
                continue
            lab = c.get("targets", {}) if self.load_targets else {}
            d = torch.zeros(hw)
            valid = torch.zeros(hw, dtype=torch.bool)
            if "depth" in lab:
                p = self.root / lab["depth"]
                encoding = lab.get("depth_encoding", "metric_npy")
                if encoding in ("metric_npy", "scene_unit_npy"):
                    expected = "meters" if encoding == "metric_npy" else "scene_unit"
                    if getattr(self, "geometry_units", "meters") != expected:
                        raise ValueError("深度编码声明的单位与数据配置不一致")
                    d = torch.from_numpy(np.load(p, allow_pickle=False)).float()
                elif encoding == "dynamic_replica_float16_png":
                    from PIL import Image

                    if lab.get("depth_units") != getattr(self, "geometry_units", "meters"):
                        raise ValueError("DynamicReplica深度单位与数据配置不一致")
                    with Image.open(p) as image:
                        bits = np.asarray(image)
                    if bits.ndim != 2 or not np.issubdtype(bits.dtype, np.integer):
                        raise ValueError("DynamicReplica深度必须是单通道整数PNG位模式")
                    if bits.min() < 0 or bits.max() > 65535:
                        raise ValueError("DynamicReplica深度超出uint16位模式")
                    d = torch.from_numpy(bits.astype(np.uint16).view(np.float16).astype(np.float32))
                elif encoding == "carla_rgb24":
                    if lab.get("raw_sensor_bytes") is not True:
                        raise ValueError(
                            "carla_rgb24 requires raw_sensor_bytes=true, never color-converted depth"
                        )
                    d = decode_carla_depth(
                        _read_image(p), lab.get("channel_order", "RGB"), lab.get("far", 1000.0)
                    )
                else:
                    raise ValueError(f"Unsupported depth encoding: {encoding}")
                if d.shape != hw:
                    raise ValueError("Depth must match the raw RGB pixel grid")
                scale = float(lab.get("depth_scale", 1.0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("depth_scale必须是有限正数")
                metric = d * scale
                valid = (
                    torch.isfinite(metric)
                    & (metric > 0)
                    & (metric < float(lab.get("max_valid_depth", 1000.0)))
                )
                d = to_optical_depth(metric, K, lab["depth_type"])
                valid &= torch.isfinite(d) & (d > 0)
                if "depth_valid" in lab:
                    label_mask = torch.from_numpy(
                        np.load(self.root / lab["depth_valid"], allow_pickle=False)
                    ).bool()
                    if label_mask.shape != hw:
                        raise ValueError("depth_valid必须与本相机原始RGB像素格完全一致")
                    valid &= label_mask
                d = torch.where(valid, d, torch.zeros_like(d))
                depth_present = True
            sem = torch.full(hw, -1, dtype=torch.long)
            if "semantic" in lab:
                sem = torch.from_numpy(
                    np.load(self.root / lab["semantic"], allow_pickle=False)
                ).long()
                if sem.shape != hw:
                    raise ValueError("Semantic IDs must match raw RGB grid")
                semantic_present = True
            instance = torch.full(hw, -1, dtype=torch.long)
            if "instance" in lab:
                instance = torch.from_numpy(
                    np.load(self.root / lab["instance"], allow_pickle=False)
                ).long()
                if instance.shape != hw:
                    raise ValueError("Instance IDs must match raw RGB grid")
                instance_present = True
            rgbs.append(rgb)
            intrinsics.append(K)
            extrinsics.append(T)
            depths.append(d)
            masks.append(valid)
            semantics.append(sem)
            instances.append(instance)
        inputs = {
            "rgb": torch.stack(rgbs),
            "K": torch.stack(intrinsics),
            "T_cam_from_rig": torch.stack(extrinsics),
            "view_mask": torch.ones(len(rgbs), dtype=torch.bool),
            "image_mask": torch.ones((len(rgbs), *hw), dtype=torch.bool),
        }
        for index, camera in enumerate(frame["cameras"]):
            for key in ("valid_mask", "image_valid_mask"):
                if camera.get(key):
                    mask = torch.from_numpy(np.load(self.root / camera[key], allow_pickle=False)).bool()
                    if mask.shape != hw:
                        raise ValueError(f"{key}必须与原始RGB像素网格一致")
                    inputs["image_mask"][index] &= mask
        targets = {}
        if depth_present:
            targets.update(depth=torch.stack(depths), depth_valid=torch.stack(masks))
        if semantic_present:
            targets["semantic"] = torch.stack(semantics)
        if instance_present:
            targets["instance"] = torch.stack(instances)
        return {
            "inputs": inputs,
            "targets": targets,
            "metadata": {
                **{k: frame[k] for k in ("sequence", "frame_id", "timestamp")},
                **{k: frame[k] for k in ("geometry_units", "sample_type", "synchronized_capture")
                   if k in frame},
            },
        }


def collate_frames(samples):
    """Pad only missing cameras; their K/T are safe identities and masks false."""
    if not samples:
        raise ValueError("Empty batch")
    if any("camera_ids" in s["inputs"] for s in samples):
        from .multifocal.data import collate_multifocal

        return collate_multifocal(samples)
    n = max(s["inputs"]["rgb"].shape[0] for s in samples)
    h, w = samples[0]["inputs"]["rgb"].shape[-2:]
    inputs = {"rgb": [], "K": [], "T_cam_from_rig": [], "view_mask": [], "image_mask": []}
    if all("matching_rgb" in sample["inputs"] for sample in samples):
        inputs.update(matching_rgb=[], matching_image_mask=[])
    camera_targets = {
        "depth",
        "depth_valid",
        "normals",
        "semantic",
        "instance",
        "depth_resize_interpolated",
        "depth_resize_boundary",
    }
    target_keys = set().union(*(s["targets"].keys() for s in samples))
    targets = {k: [] for k in target_keys}
    for s in samples:
        a = s["inputs"]
        m = a["rgb"].shape[0]
        if a["rgb"].shape[-2:] != (h, w):
            raise ValueError("Batch images must have equal size; preprocess with calibrated resize")
        for key in inputs:
            shape = (n, *a[key].shape[1:]) if key in a else (n, h, w)
            if key == "K":
                out = torch.eye(3, dtype=a["K"].dtype).repeat(n, 1, 1)
            elif key == "T_cam_from_rig":
                out = torch.eye(4, dtype=a[key].dtype).repeat(n, 1, 1)
            else:
                out = torch.zeros(shape, dtype=a[key].dtype if key in a else torch.bool)
            out[:m] = a.get(key, torch.ones((m, h, w), dtype=torch.bool))
            inputs[key].append(out)
        for key in targets:
            exemplar = next(t["targets"][key] for t in samples if key in t["targets"])
            if key in camera_targets:
                out = torch.full(
                    (n, *exemplar.shape[1:]),
                    -1 if key in ("semantic", "instance") else 0,
                    dtype=exemplar.dtype,
                )
                if key in s["targets"]:
                    out[:m] = s["targets"][key]
                targets[key].append(out)
            else:
                targets[key].append(s["targets"].get(key))
    return {
        "inputs": {k: torch.stack(v) for k, v in inputs.items()},
        "targets": {k: (torch.stack(v) if k in camera_targets else v) for k, v in targets.items()},
        "metadata": [s.get("metadata", {}) for s in samples],
    }


class InputsOnlyDataset(Dataset):
    """用于合成接口的无标签视图；真实数据在读取阶段就不打开标签文件。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return {"inputs": item["inputs"], "targets": {}, "metadata": item.get("metadata", {})}


def build_dataset(cfg, split="train", synthetic=False, load_targets=True):
    data = cfg.get("data", {})
    if data.get("protocol") == "calibrated_multiview":
        if "camera" in cfg or synthetic:
            raise ValueError("公共多视图协议使用真实清单，不与固定CARLA相机身份或合成模式混用")
        source = CarlaFrameDataset(
            data.get("root", ""), data.get(f"{split}_manifest", ""), split,
            load_targets=load_targets, geometry_units=data.get("geometry_units", "meters"),
        )
        return CalibratedMultiviewDataset(source, data)
    if "camera" in cfg:
        from .multifocal.data import build_multifocal_dataset

        return build_multifocal_dataset(cfg, split, synthetic, load_targets)
    if cfg.get("data", {}).get("matching_image_size") is not None:
        raise ValueError("matching_image_size当前需要显式wide4相机数据接口")
    if synthetic:
        from .synthetic import SyntheticRigDataset

        p = cfg.get("synthetic", {})
        dataset = SyntheticRigDataset(
            length=p.get("length", 8),
            image_size=(p.get("height", 64), p.get("width", 96)),
            num_cameras=p.get("num_cameras", 4),
            seed=p.get("seed", 0) + (0 if split == "train" else 10000),
        )
        return dataset if load_targets else InputsOnlyDataset(dataset)
    d = cfg.get("data", {})
    return CarlaFrameDataset(
        d.get("root", ""), d.get(f"{split}_manifest", ""), split, load_targets=load_targets,
        geometry_units=d.get("geometry_units", "meters"),
    )
