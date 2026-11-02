"""复用现有逐帧接口；真实数据与显式合成模式互不回退。"""

from pathlib import Path
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from jar.data import CarlaFrameDataset, collate_frames
from jar.synthetic import SyntheticRigDataset


def canonical_inputs(batch, expected_views=4):
    """接受项目格式或公开格式；外参转换只能在此显式求逆一次。"""
    if "rgb" in batch:
        out = dict(batch)
    else:
        out = dict(
            rgb=batch["images"],
            K=batch["intrinsics"],
            T_cam_from_rig=torch.linalg.inv(batch["T_cam_to_rig"]),
        )
        for key in ("view_mask", "image_mask"):
            if key in batch:
                out[key] = batch[key]
    b, n, _, h, w = out["rgb"].shape
    if n != expected_views or n < 2:
        count = "四个" if expected_views == 4 else f"{expected_views}个"
        raise ValueError(f"配置协议要求{count}真实视角，不能复制图片补足")
    out.setdefault("view_mask", torch.ones(b, n, device=out["rgb"].device, dtype=torch.bool))
    out.setdefault("image_mask", torch.ones(b, n, h, w, device=out["rgb"].device, dtype=torch.bool))
    if not out["view_mask"].all():
        raise ValueError("当前共同协议要求四台相机均可用；缺失帧应在采集/清单阶段显式剔除")
    return out


def resize_sample(sample, image_size):
    """align_corners=False 的像素中心变换：u'=(u+0.5)*s-0.5。"""
    inputs = dict(sample["inputs"])
    h, w = inputs["rgb"].shape[-2:]
    nh, nw = map(int, image_size)
    if (h, w) == (nh, nw):
        return sample
    a = inputs["K"].new_tensor(
        [[nw / w, 0, (nw / w - 1) / 2], [0, nh / h, (nh / h - 1) / 2], [0, 0, 1]]
    )
    inputs["rgb"] = F.interpolate(
        inputs["rgb"], (nh, nw), mode="bilinear", align_corners=False, antialias=True
    )
    inputs["K"] = a @ inputs["K"]
    mask_mode = "area" if nh <= h and nw <= w else "nearest-exact"
    inputs["image_mask"] = F.interpolate(
        inputs["image_mask"][:, None].float(), (nh, nw), mode=mask_mode
    )[:, 0] > 0.999
    targets = dict(sample["targets"])
    for key in ("depth", "depth_valid", "semantic", "instance", "normals", "sky"):
        if key in targets:
            value = targets[key]
            expanded = value[:, None] if value.ndim == 3 else value
            scaled = F.interpolate(expanded.float(), (nh, nw), mode="nearest-exact")
            targets[key] = (scaled[:, 0] if value.ndim == 3 else scaled).to(value.dtype)
    return dict(sample, inputs=inputs, targets=targets)


def crop_sample(sample, top, left, height, width):
    inputs, targets = dict(sample["inputs"]), dict(sample["targets"])
    h, w = inputs["rgb"].shape[-2:]
    if min(top, left) < 0 or min(height, width) <= 0 or top + height > h or left + width > w:
        raise ValueError("裁剪区域必须位于图像内")
    for container, keys in (
        (inputs, ("rgb", "image_mask")),
        (targets, ("depth", "depth_valid", "normals", "semantic", "instance", "sky")),
    ):
        for key in keys:
            if key in container:
                container[key] = container[key][..., top : top + height, left : left + width]
    inputs["K"] = inputs["K"].clone()
    inputs["K"][..., 0, 2] -= left
    inputs["K"][..., 1, 2] -= top
    return dict(sample, inputs=inputs, targets=targets)


def horizontal_flip_sample(sample):
    """纯图像变换使用负焦距 K 表达反射，三维 rig 不变；默认流程不启用。

    Niantic 训练调用其官方正焦距/相机反射增强，不与本函数叠加。
    """
    inputs, targets = dict(sample["inputs"]), dict(sample["targets"])
    w = inputs["rgb"].shape[-1]
    a = inputs["K"].new_tensor([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]])
    inputs["K"] = a @ inputs["K"]
    for container, keys in (
        (inputs, ("rgb", "image_mask")),
        (targets, ("depth", "depth_valid", "normals", "semantic", "instance", "sky")),
    ):
        for key in keys:
            if key in container:
                container[key] = container[key].flip(-1)
    return dict(sample, inputs=inputs, targets=targets)


class FourCameraDataset(Dataset):
    def __init__(self, config, split, synthetic=False, load_targets=True):
        self.config, self.load_targets = config, load_targets
        self.num_views = int(config.get("data", {}).get("num_views", 4))
        if self.num_views < 2:
            raise ValueError("多视图数据至少需要两个真实视角")
        self.image_size = tuple(config.get("image_size", [256, 384]))
        self.supervision_depth_range = config.get("data", {}).get("supervision_depth_range")
        if self.supervision_depth_range is not None:
            near, far = self.supervision_depth_range
            if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
                raise ValueError("supervision_depth_range 必须为有限米制范围 0 < near < far")
        if synthetic:
            self.source = SyntheticRigDataset(
                config.get("synthetic_length", 4), self.image_size, self.num_views, seed=config.get("seed", 1701)
            )
        else:
            data = config["data"]
            self.source = CarlaFrameDataset(
                data["root"], data[f"{split}_manifest"], split, load_targets,
                geometry_units=data.get("geometry_units", "meters"),
            )
        self.synthetic = synthetic

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        if sample["inputs"]["rgb"].shape[0] != self.num_views:
            raise ValueError(f"样本必须含{self.num_views}个真实视角")
        if not self.load_targets:
            sample = dict(sample, targets={})
        elif not self.synthetic:
            cameras = self.source.frames[index]["cameras"]
            if any(c.get("targets", {}).get("sky") for c in cameras):
                sky = torch.full((self.num_views, *sample["inputs"]["rgb"].shape[-2:]), float("nan"))
                for camera, info in enumerate(cameras):
                    label_path = info.get("targets", {}).get("sky")
                    if label_path:
                        value = torch.from_numpy(
                            np.load(self.source.root / label_path, allow_pickle=False)
                        ).float()
                        if (
                            value.shape != sky.shape[-2:]
                            or not ((value == 0) | (value == 1) | torch.isnan(value)).all()
                        ):
                            raise ValueError("sky 标签必须与原图等大，0=非天空、1=天空、NaN=未知")
                        sky[camera] = value
                sample["targets"]["sky"] = sky
            label = self.source.frames[index].get("occupancy_targets")
            if label:
                frame = self.source.frames[index]
                if str(label["frame_id"]) != str(frame["frame_id"]):
                    raise ValueError("占据标签与 RGB 不同步")
                with np.load(self.source.root / label["path"], allow_pickle=False) as values:
                    occ = {key: torch.from_numpy(values[key].copy()) for key in values.files}
                validate_occupancy(occ)
                sample["targets"]["occupancy"] = occ
        if self.config.get("image_crop"):
            sample = crop_sample(sample, *self.config["image_crop"])
        sample = resize_sample(sample, self.image_size)
        if self.supervision_depth_range is not None and "depth" in sample["targets"]:
            near, far = self.supervision_depth_range
            targets = dict(sample["targets"])
            depth = targets["depth"]
            valid = targets.get("depth_valid", torch.isfinite(depth) & (depth > 0))
            targets["depth_valid"] = (
                valid & torch.isfinite(depth) & (depth >= near) & (depth <= far)
            )
            sample = dict(sample, targets=targets)
        derive = self.config.get("data", {}).get("derive_observed_occupancy", False)
        if (
            self.load_targets
            and "occupancy" not in sample["targets"]
            and (derive or (self.synthetic and self.config.get("method") == "gaussianformer2"))
        ):
            from .occupancy_labels import observed_occupancy

            batch = collate_frames([sample])
            roi = self.config.get("evaluation", {}).get("roi", [[0, 64], [-24, 24], [-2, 6]])
            voxel_size = self.config.get("model", {}).get("voxel_size", 0.5)
            sample["targets"]["occupancy"] = observed_occupancy(
                batch["inputs"], batch["targets"], roi, voxel_size
            )
        sample["metadata"] = dict(
            sample["metadata"],
            sample_id=f"{sample['metadata']['sequence']}/{sample['metadata']['frame_id']}",
        )
        return sample


def validate_occupancy(occ):
    required = ("state", "grid_origin", "voxel_size")
    if any(k not in occ for k in required):
        raise ValueError(f"占据 npz 必须包含 {required}")
    state = occ["state"]
    if state.ndim != 3 or not ((state == 0) | (state == 1) | (state == 2)).all():
        raise ValueError("state 必须为 [X,Y,Z]，0=free，1=occupied，2=unknown")
    if (
        occ["grid_origin"].numel() != 3
        or occ["voxel_size"].numel() not in (1, 3)
        or not torch.isfinite(occ["grid_origin"]).all()
        or not torch.isfinite(occ["voxel_size"]).all()
        or not (occ["voxel_size"] > 0).all()
    ):
        raise ValueError("grid_origin/voxel_size 必须以米为单位")
    if "semantics" in occ and occ["semantics"].shape != state.shape:
        raise ValueError("semantics 与 state 网格不一致")


def collate(samples, expected_views=4):
    # 原项目将非图像标签保留为逐样本列表；此处同样保留稀疏或不同网格标签。
    out = collate_frames(samples)
    if "sky" in out["targets"]:
        shape = (expected_views, *out["inputs"]["rgb"].shape[-2:])
        out["targets"]["sky"] = torch.stack(
            [s["targets"].get("sky", torch.full(shape, float("nan"))) for s in samples]
        )
    out["inputs"] = canonical_inputs(out["inputs"], expected_views=expected_views)
    return out


def audit_split_sequences(config):
    """默认按序列隔离；显式 temporal_gap 只允许同轨迹的先训练、后验证。"""
    data = config["data"]
    protocol = data.get("split_protocol", "sequence")
    if protocol not in ("sequence", "temporal_gap"):
        raise ValueError("split_protocol 必须为 sequence 或 temporal_gap")
    gap = int(data.get("min_frame_gap", 40))
    if protocol == "temporal_gap" and gap < 1:
        raise ValueError("temporal_gap 必须至少隔离一个 frame ID")
    seen, frames = {}, {}
    for split in ("train", "val", "test"):
        path = data.get(f"{split}_manifest", "")
        if not path:
            continue
        path = Path(path) if Path(path).is_absolute() else Path(data["root"]) / path
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            seq = frame["sequence"]
            if protocol == "sequence" and seq in seen and seen[seq] != split:
                raise ValueError(f"序列 {seq} 同时出现在 {seen[seq]} 与 {split}")
            seen.setdefault(seq, split)
            frames.setdefault(seq, {}).setdefault(split, []).append(frame["frame_id"])
    if protocol == "temporal_gap":
        for seq, groups in frames.items():
            if len(groups) < 2:
                continue
            if set(groups) != {"train", "val"}:
                raise ValueError(f"测试序列 {seq} 不能与训练或验证序列重合")
            train_ids = [int(frame) for frame in groups["train"]]
            val_ids = [int(frame) for frame in groups["val"]]
            if min(val_ids) - max(train_ids) - 1 < gap:
                raise ValueError(f"序列 {seq} 的训练/验证时间区间重叠或间隔小于 {gap} 帧")
    return seen
