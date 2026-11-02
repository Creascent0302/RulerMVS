"""基于稳定相机 ID 的多尺寸数据适配；标签和推理输入严格分开。"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from ..data import CarlaFrameDataset, MatchingImageDataset, add_matching_images
from .options import camera_ids


def pack_cameras(records, ids, roles, metadata):
    """按 ID 顺序打包单相机样本，右/下补零；标定仍对应原图。"""
    present = [r for r in records if r is not None]
    if not present:
        raise ValueError("一帧至少需要一个真实相机，不能从全缺失输入构造图像尺寸")
    height = max(r["inputs"]["rgb"].shape[-2] for r in present)
    width = max(r["inputs"]["rgb"].shape[-1] for r in present)
    n = len(records)
    inputs = {
        "rgb": torch.zeros(n, 3, height, width),
        "K": torch.eye(3).repeat(n, 1, 1),
        "T_cam_from_rig": torch.eye(4).repeat(n, 1, 1),
        "view_mask": torch.zeros(n, dtype=torch.bool),
        "image_mask": torch.zeros(n, height, width, dtype=torch.bool),
        "camera_ids": list(ids),
        "camera_roles": torch.tensor(roles, dtype=torch.long),
        "camera_valid": torch.zeros(n, dtype=torch.bool),
        "image_sizes": torch.zeros(n, 2, dtype=torch.long),
    }
    matching_size = None
    if "matching_rgb" in present[0]["inputs"]:
        matching_size = present[0]["inputs"]["matching_rgb"].shape[-2:]
        inputs["matching_rgb"] = torch.zeros(n, 3, *matching_size)
        inputs["matching_image_mask"] = torch.zeros(n, *matching_size, dtype=torch.bool)
    targets = {}
    keys = set().union(*(r["targets"] for r in present))
    for key in keys:
        example = next(r["targets"][key] for r in present if key in r["targets"])
        targets[key] = torch.full(
            (n, *example.shape[1:-2], height, width),
            -1 if key in ("semantic", "instance") else 0,
            dtype=example.dtype,
        )
    for i, record in enumerate(records):
        if record is None:
            continue
        a = record["inputs"]
        h, w = a["rgb"].shape[-2:]
        for key in ("rgb", "image_mask"):
            inputs[key][i, ..., :h, :w] = a[key][0]
        for key in ("K", "T_cam_from_rig"):
            inputs[key][i] = a[key][0]
        if matching_size is not None:
            if a["matching_rgb"].shape[-2:] != matching_size:
                raise ValueError("匹配支路要求各相机使用相同matching_image_size")
            inputs["matching_rgb"][i] = a["matching_rgb"][0]
            inputs["matching_image_mask"][i] = a["matching_image_mask"][0]
        inputs["view_mask"][i] = inputs["camera_valid"][i] = bool(a["view_mask"][0])
        inputs["image_sizes"][i] = torch.tensor([h, w])
        for key, value in record["targets"].items():
            targets[key][i, ..., :h, :w] = value[0]
    return {"inputs": inputs, "targets": targets, "metadata": metadata}


class MultifocalFrameDataset(CarlaFrameDataset):
    """复用原加载器的标定/深度编码校验，逐相机加载后按身份排序。"""

    def __init__(self, cfg, split="train", load_targets=True):
        d = cfg["data"]
        super().__init__(d.get("root", ""), d.get(split + "_manifest", ""), split, load_targets,
                         geometry_units=d.get("geometry_units", "meters"))
        self.ids = camera_ids(cfg)
        self.roles = [0] * 4 + ([1] * 2 if len(self.ids) == 6 else [])
        self.allow_missing = cfg["camera"]["allow_missing"]
        self.disabled = set(cfg["camera"].get("disabled_ids", []))
        self.image_size = d.get("image_size")
        self.matching_image_size = d.get("matching_image_size")
        if self.matching_image_size is not None and cfg["camera"]["mode"] != "wide4":
            raise ValueError("matching_image_size当前仅支持wide4")
        self.depth_resize_options = {
            key: d[key]
            for key in ("depth_resize_mode", "depth_jump_absolute_m", "depth_jump_relative")
            if key in d
        }
        self.camera_maps = []
        for frame in self.frames:
            lookup = {}
            for camera in frame["cameras"]:
                cid = camera.get("camera_id", camera.get("id"))
                if not isinstance(cid, str) or not cid:
                    raise ValueError("多焦距 manifest 每台相机必须提供稳定 camera_id")
                if cid in lookup:
                    raise ValueError("重复 camera_id: " + cid)
                lookup[cid] = camera
            for cid, role in zip(self.ids, self.roles):
                if cid in self.disabled:
                    continue
                c = lookup.get(cid)
                if (
                    c
                    and c.get("role", c.get("camera_role", ("wide", "tele")[role]))
                    != ("wide", "tele")[role]
                ):
                    raise ValueError("manifest 相机角色与配置 ID 分组不一致: " + cid)
                if (c is None or not c.get("valid", True)) and not self.allow_missing:
                    raise ValueError(
                        "缺少必需相机 " + cid + "；仅 camera.allow_missing=true 可允许掩码降级"
                    )
            self.camera_maps.append(lookup)

    def __getitem__(self, index):
        frame = self.frames[index]
        lookup = self.camera_maps[index]
        records = []
        missing = []
        for cid in self.ids:
            camera = lookup.get(cid)
            if cid in self.disabled or camera is None or not camera.get("valid", True):
                records.append(None)
                missing.append(cid)
                continue
            # 独立轻量单相机视图复用原读取实现，不修改 self.frames（多worker安全）。
            single = object.__new__(CarlaFrameDataset)
            single.root, single.load_targets = self.root, self.load_targets
            single.geometry_units = self.geometry_units
            single.frames = [{**frame, "cameras": [camera]}]
            item = CarlaFrameDataset.__getitem__(single, 0)
            from ..data_transforms import resize_frame

            item = add_matching_images(item, getattr(self, "matching_image_size", None))
            item = resize_frame(item, self.image_size, **self.depth_resize_options)
            records.append(item)
        metadata = {k: frame[k] for k in ("sequence", "frame_id", "timestamp")}
        for key in (
            "geometry_units",
            "sample_type",
            "synchronized_capture",
            "sync_basis",
            "timestamp_source",
            "camera_completeness",
            "sensor_completeness",
            "valid_camera_count",
            "depth_available",
            "radar_available",
            "direct_depth_supervision_available",
            "depth_definition_verified",
        ):
            if key in frame:
                metadata[key] = frame[key]
        resizes = [r["metadata"].get("resize") for r in records if r is not None]
        if any(resizes):
            metadata["resize_per_present_camera"] = resizes
        metadata.update(
            camera_ids=self.ids,
            role_names=[("wide", "tele")[r] for r in self.roles],
            missing_camera_ids=missing,
            disabled_camera_ids=sorted(self.disabled),
            pixel_transform=(
                "calibrated resize + right/bottom padding"
                if self.image_size
                else "identity; right/bottom padding only"
            ),
        )
        return pack_cameras(records, self.ids, self.roles, metadata)


def collate_multifocal(samples):
    """同时按 batch 和相机 padding；所有标签也保留原像素格。"""
    first = samples[0]["inputs"]
    ids = first["camera_ids"]
    n = len(ids)
    # 四广角轮换参考图时，每个样本可以有不同顺序；标定、图像、标签
    # 已在样本内同步重排。保留每个样本的 ID，不能再强行按首样本排序。
    wide_permutations = n == 4 and len(set(ids)) == 4 and all(
        len(sample["inputs"]["camera_ids"]) == 4
        and set(sample["inputs"]["camera_ids"]) == set(ids)
        and bool((sample["inputs"]["camera_roles"] == 0).all())
        for sample in samples
    )
    if not wide_permutations and any(s["inputs"]["camera_ids"] != ids for s in samples):
        raise ValueError("一个 batch 必须使用相同有序相机 ID 协议")
    h = max(s["inputs"]["rgb"].shape[-2] for s in samples)
    w = max(s["inputs"]["rgb"].shape[-1] for s in samples)
    inputs = {}
    targets = {}
    for key in first:
        if key == "camera_ids":
            inputs[key] = [list(s["inputs"][key]) for s in samples]
            continue
        values = []
        for s in samples:
            x = s["inputs"][key]
            if key in ("rgb", "image_mask"):
                out = x.new_zeros(*x.shape[:-2], h, w)
                out[..., : x.shape[-2], : x.shape[-1]] = x
                x = out
            values.append(x)
        inputs[key] = torch.stack(values)
    keys = set().union(*(s["targets"] for s in samples))
    for key in keys:
        sample = next(s["targets"][key] for s in samples if key in s["targets"])
        rows = []
        for s in samples:
            out = sample.new_full(
                (n, *sample.shape[1:-2], h, w), -1 if key in ("semantic", "instance") else 0
            )
            if key in s["targets"]:
                value = s["targets"][key]
                out[..., : value.shape[-2], : value.shape[-1]] = value
            rows.append(out)
        targets[key] = torch.stack(rows)
    return {"inputs": inputs, "targets": targets, "metadata": [s["metadata"] for s in samples]}


def build_multifocal_dataset(cfg, split, synthetic, load_targets):
    if not synthetic:
        return MultifocalFrameDataset(cfg, split, load_targets)
    from .synthetic import MultifocalSyntheticDataset

    dataset = MultifocalSyntheticDataset(cfg, split, load_targets)
    matching_size = cfg["data"].get("matching_image_size")
    return MatchingImageDataset(dataset, matching_size) if matching_size is not None else dataset


def validate_camera_inputs(inputs, cfg):
    """推理入口也验证身份契约，不用数组下标猜测广角/长焦。"""
    required = (
        "camera_ids",
        "camera_roles",
        "camera_valid",
        "image_sizes",
        "view_mask",
        "image_mask",
    )
    missing = [key for key in required if key not in inputs]
    if missing:
        raise ValueError("多焦距输入缺少显式相机元数据: " + ", ".join(missing))
    rgb = inputs["rgb"]
    batch, views, _, height, width = rgb.shape
    expected = camera_ids(cfg)
    if views != len(expected) or len(inputs["camera_ids"]) != batch:
        raise ValueError("相机数量与 camera.mode/有序 ID 配置不一致")
    model = cfg["model"]
    all_view_dense = (
        views == 4
        and model.get("dense_decoder", {}).get("enabled", False)
        and not model.get("sparse_decoder", {}).get("enabled", True)
        and model.get("execution", {}).get("reference_indices") is None
    )
    if all_view_dense:
        # 广角相机使用共享网络，身份由随图像提供的 ID 和标定决定，
        # 不把某个数组槽位当作固定主相机。稀疏预算分配、单参考调度
        # 以及六目分组前端仍保留原有顺序契约。
        if any(len(row) != views or set(row) != set(expected) for row in inputs["camera_ids"]):
            raise ValueError("四目 camera_ids 必须恰好包含配置的四个不同广角 ID")
    elif any(list(row) != expected for row in inputs["camera_ids"]):
        raise ValueError("当前路径的 camera_ids 顺序必须与配置一致（四个稳定广角 ID 在前）")
    roles, valid, view, sizes, mask = (
        inputs[k]
        for k in ("camera_roles", "camera_valid", "view_mask", "image_sizes", "image_mask")
    )
    if roles.shape != (batch, views) or roles.dtype != torch.long:
        raise ValueError("camera_roles 必须是 long[B,N]，0=wide，1=tele")
    expected_roles = torch.tensor([0] * 4 + ([1] * 2 if views == 6 else []), device=roles.device)
    if not torch.equal(roles, expected_roles[None].expand(batch, -1)):
        raise ValueError("camera_roles 与配置的 ID 分组不一致")
    if (
        valid.shape != (batch, views)
        or view.shape != valid.shape
        or valid.dtype != torch.bool
        or view.dtype != torch.bool
    ):
        raise ValueError("camera_valid/view_mask 必须是 bool[B,N]")
    if not torch.equal(valid, view):
        raise ValueError("camera_valid 与 view_mask 必须一致")
    disabled = set(cfg["camera"].get("disabled_ids", []))
    required_views = valid.new_tensor(
        [[cid not in disabled for cid in row] for row in inputs["camera_ids"]]
    )
    if not cfg["camera"]["allow_missing"] and not bool(valid[required_views].all()):
        raise ValueError("缺少必需相机；必须显式 allow_missing=true 才能降级")
    if sizes.shape != (batch, views, 2) or sizes.dtype != torch.long:
        raise ValueError("image_sizes 必须是 long[B,N,2]，顺序为原始 H,W")
    bounds = sizes.new_tensor([height, width])
    if bool(((sizes < 0) | (sizes > bounds)).any()) or bool((sizes[valid] <= 0).any()):
        raise ValueError("有效相机 image_sizes 必须为正且不超过 padding 后 RGB 尺寸")
    if mask.shape != (batch, views, height, width) or mask.dtype != torch.bool:
        raise ValueError("image_mask 必须是 bool[B,N,H,W]")
    rows = torch.arange(height, device=sizes.device)[None, None, :, None]
    cols = torch.arange(width, device=sizes.device)[None, None, None, :]
    outside = (
        (rows >= sizes[..., 0, None, None])
        | (cols >= sizes[..., 1, None, None])
        | ~valid[..., None, None]
    )
    if bool((mask & outside).any()):
        raise ValueError("padding/缺失相机区域的 image_mask 必须为 false")


def mask_camera_inputs(inputs, camera_config):
    """主动关闭相机，不修改调用者的张量；禁用视图不参与匹配、监督和编码。"""
    disabled = set(camera_config.get("disabled_ids", []))
    if not disabled:
        return inputs
    if "camera_ids" not in inputs:
        raise ValueError("按名称禁用相机需要 inputs.camera_ids")
    rgb = inputs["rgb"]
    selected = torch.tensor(
        [[cid not in disabled for cid in row] for row in inputs["camera_ids"]],
        device=rgb.device,
        dtype=torch.bool,
    )
    view_mask = inputs["view_mask"] & selected
    result = dict(inputs)
    result["view_mask"] = view_mask
    result["camera_valid"] = view_mask
    result["image_mask"] = inputs["image_mask"] & view_mask[:, :, None, None]
    result["image_sizes"] = torch.where(view_mask[:, :, None], inputs["image_sizes"], 0)
    result["rgb"] = torch.where(view_mask[:, :, None, None, None], rgb, 0)
    if "matching_rgb" in inputs:
        result["matching_rgb"] = torch.where(view_mask[:, :, None, None, None], inputs["matching_rgb"], 0)
        result["matching_image_mask"] = inputs["matching_image_mask"] & view_mask[:, :, None, None]
    return result
