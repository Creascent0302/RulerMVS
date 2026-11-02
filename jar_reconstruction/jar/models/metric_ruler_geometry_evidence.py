"""只用已知标定的轻量标尺取证；不计算图像导数或纹理 Gram。

图像相关相似度仍来自真实 PlaneSweep。这里只生成每米像素变化与可用
源比例，供公制候选网络解释观测。几何缓存绝不保存 RGB、特征或标签。
"""

from collections import OrderedDict

import torch

from jar.geometry import pixel_grid, project, sample_maps, unproject


class RulerProjectionCache:
    """固定阵列的小型有界缓存，图像掩码与缺失视图每帧重新应用。"""

    def __init__(self, capacity=4):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity 必须为正")
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(inputs):
        def tensor_key(value):
            data = value.detach().contiguous().cpu()
            return str(data.dtype), tuple(data.shape), data.numpy().tobytes()

        return (
            str(inputs["K"].device), tensor_key(inputs["K"]),
            tensor_key(inputs["T_cam_from_rig"]), tensor_key(inputs["inverse_depths"]),
            tuple(inputs["image_size"]), int(inputs["stride"]), int(inputs["reference_index"]),
            tuple(inputs["source_cosine"].shape[-2:]),
        )

    def get(self, inputs):
        key = self.key(inputs)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        with torch.inference_mode(False), torch.no_grad():
            value = projected_ruler(inputs)
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value

    def clear(self):
        self.entries.clear()


def projected_ruler(inputs):
    """三维刻度精确投影及投影导数；与图像内容和标签无关。"""
    _, _, count, height, width = inputs["source_cosine"].shape
    device = inputs["K"].device
    reference = int(inputs["reference_index"])
    uv = pixel_grid(height, width, device=device, stride=inputs["stride"]).reshape(-1, 2)
    depth = inputs["inverse_depths"].float().reciprocal()[:, None].expand(count, len(uv))
    intrinsic = inputs["K"][0].float()
    transform = inputs["T_cam_from_rig"][0].float()
    points, rays = unproject(uv[None].expand(count, -1, -1), depth,
                             intrinsic[reference], transform[reference])
    projected, _, valid, jacobian = project(
        points.reshape(-1, 3), intrinsic, transform, inputs["image_size"]
    )
    derivative = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1)
    return projected, valid, derivative


def extract_geometry_ruler_features(record, ruler_scale=1.0, cache=None):
    """与完整版同八通道合同；前五通道为零，后三通道为几何条件。

    没有图像导数，不用其纹理阈值选择视图。沿用原匹配的 RGB 纹理、
    视差及图像有效性掩码；另保留相同的特征采样足迹边界。
    """
    inputs = record.get("inputs", record)
    _, views, count, height, width = inputs["source_cosine"].shape
    projected, valid, derivative = projected_ruler(inputs) if cache is None else cache.get(inputs)
    valid = valid & inputs["view_mask"][0, :, None].bool()
    image_mask = inputs["image_mask"][0, :, None].float()
    valid = valid & (sample_maps(image_mask, projected)[..., 0] > 0.999)
    scale = inputs["matching_scale"].reshape(-1, 2)[0].to(projected)
    feature_hw = inputs["matching_features"].shape[-2:]
    feature_uv = (projected + 0.5) * scale / 4 - 0.5
    upper = feature_uv.new_tensor([feature_hw[1] - 2, feature_hw[0] - 2])
    footprint = ((feature_uv >= 1) & (feature_uv <= upper)).all(-1)
    reference = int(inputs["reference_index"])
    valid = valid & footprint & (valid[reference] & footprint[reference])[None]
    valid = valid & inputs["source_valid"][0].reshape(views, -1).bool()
    valid[reference] = False
    parallax = derivative.norm(dim=-1) * float(ruler_scale)
    valid = valid & torch.isfinite(parallax) & (parallax > 1e-5)
    support = valid.sum(0)
    mean = (parallax * valid).sum(0) / support.clamp_min(1)
    maximum = parallax.masked_fill(~valid, 0).amax(0)
    zeros = torch.zeros_like(mean)
    values = [zeros] * 5 + [mean.log1p(), maximum.log1p(), support.float() / max(views - 1, 1)]
    features = torch.stack(values).reshape(1, 8, count, height, width)
    stats = {
        "source_valid": valid.reshape(1, views, count, height, width),
        "source_jet_energy": torch.zeros_like(inputs["source_cosine"]),
        "ruler_parallax": parallax.reshape(1, views, count, height, width),
        "source_cosine": inputs["source_cosine"],
    }
    return features, stats
