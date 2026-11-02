"""物理尺度一致的候选残差场：场景判断保留到每一轮精确匹配。

场仅提供候选评分的空间修正，不替换原初值，不声明已确认表面。
新观测仍由细化器精确重投影取得，避免把原始相关性重复相加。
"""

import torch
from torch import nn

from jar.geometry import sample_maps
from .initial_surface_memory import InitialSurfaceMemory, plane_sampling_plan


def sample_candidate_field(field, candidates, uv):
    """在原图像素和实际逆深度坐标读取残差；跨无效邻点时不传播。"""
    residual = field["residual"]
    valid = field["valid"]
    lattice = field["inverse_depths"]
    values = torch.where(valid, residual, 0.0)
    values = sample_maps(values, uv[None], stride=field["stride"])[0]
    support = sample_maps(valid.float(), uv[None], stride=field["stride"])[0]
    legal = support >= 1 - 1e-5
    rho = candidates.float().clamp_min(1e-6).reciprocal()
    right = torch.searchsorted(lattice, rho.contiguous()).clamp(1, len(lattice) - 1)
    left = right - 1
    alpha = ((rho - lattice[left]) / (lattice[right] - lattice[left])).clamp(0, 1)
    available = (rho >= lattice[0]) & (rho <= lattice[-1])
    available &= legal.gather(1, left) | (alpha >= 1 - 1e-6)
    available &= legal.gather(1, right) | (alpha <= 1e-6)
    correction = torch.lerp(values.gather(1, left), values.gather(1, right), alpha)
    return field["weight"] * torch.where(available, correction, 0.0)


class PersistentCandidateField(nn.Module):
    """只新增小型几何记忆头；图像、描述子和精确投影均复用原网络。"""

    def __init__(self, settings, checkpoint_state=None):
        super().__init__()
        self.stride = int(settings.get("stride", 8))
        self.cnn_stride = int(settings.get("cnn_stride", 16))
        self.neighbor_pixels = int(settings.get("neighbor_pixels", 16))
        self.weight = float(settings.get("weight", 3.0))
        self.capture_recovery = bool(settings.get("capture_recovery", False))
        if self.stride not in (4, 8, 16) or self.cnn_stride not in (4, 8, 16, 32):
            raise ValueError("候选场和CNN采样步长不合法")
        if self.neighbor_pixels < self.stride or self.neighbor_pixels % self.stride:
            raise ValueError("邻域像素跨度须为候选场步长的正整数倍")
        if not 0 <= self.weight <= 4:
            raise ValueError("候选场权重须在[0,4]内，初始3对应已有评分温度换算")
        memory_settings = settings.get("memory", {})
        self.memory = InitialSurfaceMemory(**memory_settings)
        if checkpoint_state:
            self.load_state_dict(checkpoint_state, strict=True)
        elif settings.get("weights"):
            checkpoint = torch.load(settings["weights"], map_location="cpu", weights_only=False)
            if checkpoint["plan"]["model"] != memory_settings:
                raise ValueError("候选记忆层配置与其训练checkpoint不一致")
            self.memory.load_state_dict(checkpoint["model"], strict=True)

    def forward(self, inputs, pyramid, sweep):
        # 与训练缓存相同：改变射线密度，保留CNN来源和空间邻域的物理跨度。
        from .dense_decoder import aligned_resample

        source = pyramid[self.cnn_stride]
        batch, views = source.shape[:2]
        size = pyramid[self.stride].shape[-2:]
        resampled = aligned_resample(
            source.flatten(0, 1), self.cnn_stride, self.stride, size
        )
        sampling_pyramid = dict(pyramid)
        sampling_pyramid[self.stride] = resampled.reshape(batch, views, *resampled.shape[1:])
        entries = sweep(inputs, sampling_pyramid, stride=self.stride, return_evidence=True)
        fields = []
        for sample in entries:
            sample_fields = {}
            for reference, entry in sample.items():
                plans = plane_sampling_plan(
                    entry["inverse_depths"], entry["intrinsic"], entry["transform"],
                    size, self.stride, neighbor_step=self.neighbor_pixels // self.stride,
                )
                scores = self.memory(
                    entry["cost"], entry["hero"], entry["candidate_valid"],
                    entry["inverse_depths"], plans,
                    entry["source_cosine"], entry["source_valid"],
                )
                field = {
                    "residual": scores - 4 * entry["hero"][:, 0],
                    "valid": entry["candidate_valid"],
                    "inverse_depths": entry["inverse_depths"],
                    "stride": self.stride,
                    "weight": self.weight,
                }
                if self.capture_recovery:
                    source_valid = entry["source_valid"]
                    support = source_valid.sum(1)
                    field["recovery"] = {
                        "scores": scores,
                        "hero": entry["hero"],
                        "source_cosine": entry["source_cosine"],
                        "source_valid": source_valid,
                        "source_mean": (
                            entry["source_cosine"] * source_valid
                        ).sum(1) / support.clamp_min(1),
                        "source_support": support,
                    }
                sample_fields[reference] = field
            fields.append(sample_fields)
        return fields
