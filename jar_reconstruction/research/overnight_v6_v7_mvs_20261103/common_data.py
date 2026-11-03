"""三模型共同数据与曝光调度；GT 范围增强仅供自研模型训练使用。"""

import math
import random

import torch

from jar.pretraining.data import MixedDataset, collate_mixed
from research.v2_training_20261028.sampling import RebalancedFullPassSampler


class SharedDataset(MixedDataset):
    """保留参考轮换和亮度增强，不制造官方 MVS 无法掩码的缺失源图。

    增强种子由样本键决定，与模型、GPU 数量、加载线程和访问次序无关。
    原始数据中已有的 view_mask 不改动；没有复制图像或补造有效相机。
    """

    def __init__(self, config, split="train"):
        self.brightness_enabled = bool(config.get("augmentation", True))
        super().__init__(dict(config, augmentation=False), split)

    def __getitem__(self, key):
        sample = super().__getitem__(key)
        domain_index, index, visit_epoch = key
        if self.split != "train" or visit_epoch < 0 or not self.brightness_enabled:
            return sample
        inputs = sample["inputs"]
        views = inputs["rgb"].shape[0]
        seed = self.config["seed"] + visit_epoch * 100003 + index * 13 + domain_index
        generator = torch.Generator().manual_seed(seed)
        gain = .85 + .3 * torch.rand(views, 1, 1, 1, generator=generator)
        for name in ("rgb", "matching_rgb"):
            inputs[name] = (inputs[name] * gain).clamp(0, 1)
        return sample


def make_sampler(config, lengths, epoch, rank, world_size, start_update=0):
    """相同全局 batch、seed、epoch 对应相同三模型样本和增强键。"""
    settings = config["train"]
    batch_size = int(settings["batch_size"])
    accumulation = int(settings["accumulation"])
    global_batch = world_size * batch_size * accumulation
    expected = int(settings.get("global_batch_size", 32))
    if global_batch != expected:
        raise ValueError(f"实际全局 batch {global_batch} 与共同协议 {expected} 不一致")
    return RebalancedFullPassSampler(
        lengths, epoch, rank, world_size, batch_size, accumulation, config["seed"],
        start_update=start_update,
        domain_multipliers=[domain.get("train_multiplier", 1.) for domain in config["datasets"]],
        domain_schedule="balanced_prefix",
    )


def _interval(settings, name, default):
    low, high = map(float, settings.get(name, default))
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(f"{name} 需要有限且为正的递增区间")
    return low, high


def own_training_domain(batch, domain, config, update, seed):
    """只改训练输入范围与内部单位，不改图像、标定或物理单位标签。

    一半微批沿用 TRAIN 登记范围，另一半使用本微批参考 GT 范围并向外
    扩展。模型自行按 depth_scale 同时转换位移/候选、在输出时恢复单位。
    验证和推理禁止调用此函数；其范围继续使用事前登记的无标签协议。
    """
    settings = config.get("range_augmentation", {})
    if not settings.get("enabled", False):
        return dict(domain)
    nominal_probability = float(settings.get("nominal_probability", .5))
    unit_probability = float(settings.get("unit_jitter_probability", .5))
    if not 0 <= nominal_probability <= 1 or not 0 <= unit_probability <= 1:
        raise ValueError("范围与单位增强概率必须位于 [0,1]")
    near_interval = _interval(settings, "near_multiplier", (.5, 1.))
    far_interval = _interval(settings, "far_multiplier", (1., 2.))
    if near_interval[1] > 1 or far_interval[0] < 1:
        raise ValueError("GT 范围增强只能向外扩展，不能裁掉已观测训练表面")
    unit_interval = _interval(settings, "unit_scale_bounds", (.5, 2.))
    if not math.isfinite(float(domain["depth_scale"])) or domain["depth_scale"] <= 0:
        raise ValueError("内部 depth_scale 必须为有限正数")
    targets, inputs = batch["targets"], batch["inputs"]
    truth = targets["depth"][:, 0].float()
    valid = targets["depth_valid"][:, 0] & inputs["image_mask"][:, 0]
    valid = valid & inputs["view_mask"][:, 0, None, None]
    valid = valid & torch.isfinite(truth) & (truth > 0)
    observed = truth[valid]
    if observed.numel() == 0:
        raise ValueError("训练微批的参考相机没有有效 GT，不能生成监督范围")
    rng = random.Random(int(seed) + int(update) * 1000003)
    result = dict(domain, nominal_near=domain["near"], nominal_far=domain["far"])
    if rng.random() >= nominal_probability:
        lower = float(observed.min()) * rng.uniform(*near_interval)
        upper = float(observed.max()) * rng.uniform(*far_interval)
        result["near"] = max(1e-4, lower)
        result["far"] = max(result["near"] * (1 + 1e-4), upper)
    if rng.random() < unit_probability:
        factor = math.exp(rng.uniform(math.log(unit_interval[0]), math.log(unit_interval[1])))
        result["depth_scale"] = float(domain["depth_scale"]) * factor
    return result


__all__ = ["SharedDataset", "collate_mixed", "make_sampler", "own_training_domain"]
