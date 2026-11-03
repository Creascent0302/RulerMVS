"""V8 的确定性训练增强；相机几何不变，GT 只用于训练范围覆盖。

图像配置：photometric_augmentation.enabled，clean_probability（默认 .4），
gamma/contrast/saturation/color_balance/exposure 区间，以及 source_*_probability。
范围配置：range_augmentation.enabled，nominal_probability、broad_probability，
gt_occupancy_bounds、max_log_padding、broad_range 或 broad_multipliers。
两个 enabled 默认关闭，运行配置须显式启用。

compose_training_domain 与旧 own_training_domain 签名相同。每个 update 共用
一次范围分支和 padding 抽样，避免逐样本随机范围取 union 后，宽范围概率随
batch 大小放大。共同标量范围覆盖本批全部有效 GT，故 GT 极值本身仍随批次
组成变化；不能声称严格 batch-size 不变。VAL/推理始终返回登记范围。
"""

import hashlib
import math
import random

import torch
from torch.nn import functional as F

from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset as _SharedDataset


def _seed(*parts):
    encoded = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little") % (2 ** 63)


def _probability(settings, name, default):
    value = float(settings.get(name, default))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} 必须位于 [0,1]")
    return value


def _interval(settings, name, default, *, positive=True):
    low, high = map(float, settings.get(name, default))
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError(f"{name} 必须为有限递增区间")
    if low < 0 or (positive and low == 0):
        raise ValueError(f"{name} 下限必须{'大于' if positive else '不小于'}零")
    return low, high


def _draw(generator, limits):
    low, high = limits
    return low + (high - low) * float(torch.rand((), generator=generator))


class V8Dataset(_SharedDataset):
    """先沿用既有参考轮换，再只改变 RGB；不裁剪，不修改 K/T/GT。"""

    def __init__(self, config, split="train"):
        # 禁止旧版逐视图亮度与本增强叠加，参考轮换仍由父类完成。
        super().__init__(dict(config, augmentation=False), split)
        self.v8_config = config
        self.photo = dict(config.get("photometric_augmentation", {}))
        self.photo_enabled = bool(config.get("augmentation", True)) and bool(self.photo.get("enabled", False))
        self.clean_probability = _probability(self.photo, "clean_probability", .4)

    def __getitem__(self, key):
        sample = super().__getitem__(key)
        domain_index, index, epoch = key
        training = self.split == "train" and epoch >= 0
        seed = _seed(self.v8_config["seed"], domain_index, index, epoch, "photo")
        generator = torch.Generator().manual_seed(seed)
        clean = not (training and self.photo_enabled)
        if not clean:
            clean = float(torch.rand((), generator=generator)) < self.clean_probability
        sample["metadata"] = dict(sample.get("metadata", {}), v8_augmentation={
            "training": training, "seed": seed, "clean": clean,
            "key": [domain_index, index, epoch],
        })
        if clean:
            # 不重采样 matching_rgb；clean 路径逐元素保留原输入。
            return sample
        inputs = sample["inputs"] = dict(sample["inputs"])
        rgb = inputs["rgb"].clone()
        settings = self.photo
        valid = inputs["image_mask"] & inputs["view_mask"][:, None, None]

        # 所有视图共用相机风格变化；同一像素颜色经历相同变换。
        gamma = _draw(generator, _interval(settings, "gamma", (.9, 1.1)))
        contrast = _draw(generator, _interval(settings, "contrast", (.9, 1.1)))
        saturation = _draw(generator, _interval(settings, "saturation", (.9, 1.1)))
        balance = _interval(settings, "color_balance", (.95, 1.05))
        color = rgb.new_tensor([_draw(generator, balance) for _ in range(3)])[None, :, None, None]
        rgb = rgb.clamp(0, 1).pow(gamma)
        rgb = (rgb - .5) * contrast + .5
        gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        rgb = (gray + saturation * (rgb - gray)) * color

        for view in range(1, len(rgb)):
            if not bool(inputs["view_mask"][view]):
                continue
            image = rgb[view]
            if float(torch.rand((), generator=generator)) < _probability(settings, "source_exposure_probability", .8):
                image = image * _draw(generator, _interval(settings, "exposure", (.9, 1.1)))
            if float(torch.rand((), generator=generator)) < _probability(settings, "source_noise_probability", .15):
                sigma = _draw(generator, _interval(settings, "noise_std", (0., .015), positive=False))
                image = image + torch.randn(image.shape, generator=generator, dtype=image.dtype) * sigma
            if float(torch.rand((), generator=generator)) < _probability(settings, "source_blur_probability", .15):
                amount = _draw(generator, _interval(settings, "blur_mix", (.1, .4), positive=False))
                if amount > 1:
                    raise ValueError("blur_mix 不能超过1")
                smooth = F.avg_pool2d(F.pad(image[None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0]
                image = image.lerp(smooth, amount)
            if float(torch.rand((), generator=generator)) < _probability(settings, "source_occlusion_probability", .1):
                fraction = _draw(generator, _interval(settings, "occlusion_area", (.002, .015)))
                if fraction > .1:
                    raise ValueError("source 遮挡面积不能超过10%")
                height, width = image.shape[-2:]
                aspect = _draw(generator, (.5, 2.))
                box_h = min(height, max(1, round(math.sqrt(fraction * height * width / aspect))))
                box_w = min(width, max(1, round(math.sqrt(fraction * height * width * aspect))))
                top = int(torch.randint(height - box_h + 1, (), generator=generator))
                left = int(torch.randint(width - box_w + 1, (), generator=generator))
                fill = torch.rand((3, 1, 1), generator=generator, dtype=image.dtype)
                image[:, top:top + box_h, left:left + box_w] = fill
            rgb[view] = image
        inputs["rgb"] = torch.where(valid[:, None], rgb.clamp(0, 1), 0.)

        # 可选整源drop只影响输入有效性，不删除GT，也不能制造单目样本。
        dropout = _probability(settings, "source_dropout_probability", 0.)
        sources = inputs["view_mask"][1:].nonzero().flatten() + 1
        if len(sources) >= 2 and float(torch.rand((), generator=generator)) < dropout:
            dropped = int(sources[int(torch.randint(len(sources), (), generator=generator))])
            for name in ("view_mask", "image_mask", "matching_image_mask", "camera_valid"):
                if name in inputs:
                    inputs[name] = inputs[name].clone()
                    inputs[name][dropped] = False
            inputs["rgb"][dropped] = 0
        # 由同一增强RGB构造匹配图，与equal_pixel_inputs的插值约定完全一致。
        inputs["matching_rgb"] = F.interpolate(
            inputs["rgb"], inputs["matching_rgb"].shape[-2:], mode="bilinear", align_corners=False)
        return sample


SharedDataset = V8Dataset


def compose_training_domain(batch, domain, config, update, seed):
    """TRAIN-only 范围变换；不原地修改 batch/domain，返回旧接口的标量dict。

固定 broad 分支的基础范围不读GT，超界时仅向外补齐。极端有效标签超出
安全界限时明确失败，不能静默裁掉它们后声称全覆盖。未带metadata的旧批次
视为显式训练调用；V8Dataset的VAL标记则在读取任何targets之前返回。
"""
    settings = config.get("range_augmentation", {})
    if not settings.get("enabled", False):
        return dict(domain)
    metadata = batch.get("metadata", [])
    flags = [item["v8_augmentation"]["training"] for item in metadata if "v8_augmentation" in item]
    if batch.get("split") in ("val", "test", "inference") or (flags and not any(flags)):
        return dict(domain)
    if flags and (not all(flags) or len(flags) != len(metadata)):
        raise ValueError("同一范围增强微批不能混合训练与验证样本")
    nominal_probability = _probability(settings, "nominal_probability", .4)
    broad_probability = _probability(settings, "broad_probability", .2)
    if nominal_probability + broad_probability > 1:
        raise ValueError("nominal_probability + broad_probability 不能超过1")
    safe_low, safe_high = _interval(settings, "absolute_depth_bounds", (1e-8, 1e8))
    nominal_near, nominal_far = float(domain["near"]), float(domain["far"])
    scale = float(domain["depth_scale"])
    if not (math.isfinite(scale) and scale > 0 and safe_low <= nominal_near < nominal_far <= safe_high):
        raise ValueError("登记范围/内部尺度非法或超出absolute_depth_bounds")
    truth = batch["targets"]["depth"][:, 0].float()
    inputs = batch["inputs"]
    valid = batch["targets"]["depth_valid"][:, 0] & inputs["image_mask"][:, 0]
    valid = valid & inputs["view_mask"][:, 0, None, None] & torch.isfinite(truth) & (truth > 0)
    observed = truth[valid]
    if not observed.numel():
        raise ValueError("训练微批没有有效参考GT")
    lower, upper = float(observed.min()), float(observed.max())
    if lower < safe_low or upper > safe_high:
        raise ValueError("有效GT超出absolute_depth_bounds；请审计标签或显式调整安全界限，不能裁掉GT")

    # config seed覆盖传入的rank seed；同update/domain共享随机分支分布。
    rng = random.Random(_seed(config.get("seed", seed), domain.get("name", ""), update, "range"))
    draw = rng.random()
    if draw < nominal_probability:
        near, far, mode = nominal_near, nominal_far, "nominal"
    elif draw < nominal_probability + broad_probability:
        limits = settings.get("broad_ranges", {}).get(domain.get("name"), settings.get("broad_range"))
        if limits is None:
            left, right = _interval(settings, "broad_multipliers", (.25, 4.))
            if left > 1 or right < 1:
                raise ValueError("broad_multipliers 必须向外扩展登记范围")
            limits = (nominal_near * left, nominal_far * right)
        near, far = map(float, limits)
        if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
            raise ValueError("broad_range 必须为有限正递增范围")
        mode = "broad"
    else:
        low_occupancy, high_occupancy = _interval(settings, "gt_occupancy_bounds", (.25, .85))
        if high_occupancy > 1:
            raise ValueError("gt_occupancy_bounds 不能超过1")
        cap = float(settings.get("max_log_padding", math.log(64.)))
        if not math.isfinite(cap) or cap < 0:
            raise ValueError("max_log_padding 必须为有限非负值")
        # 未触及cap时GT的log区间占候选轴比例等于occupancy；空余位置随机分配。
        span = max(math.log(upper) - math.log(lower), math.log(1.2))
        total = min(span * (1 / rng.uniform(low_occupancy, high_occupancy) - 1), 2 * cap)
        left = rng.uniform(max(0., total - cap), min(cap, total))
        near, far = lower * math.exp(-left), upper * math.exp(total - left)
        mode = "gt_padding"
    clipped_near, clipped_far = max(safe_low, near), min(safe_high, far)
    repaired = clipped_near > lower or clipped_far < upper
    near, far = min(clipped_near, lower), max(clipped_far, upper)
    if near == far:
        near, far = max(safe_low, near / 1.01), min(safe_high, far * 1.01)
    factor = 1.
    if rng.random() < _probability(settings, "unit_jitter_probability", 0.):
        low, high = _interval(settings, "unit_scale_bounds", (.5, 2.))
        factor = math.exp(rng.uniform(math.log(low), math.log(high)))
    if not (math.isfinite(scale * factor) and scale * factor > 0):
        raise ValueError("增强后的depth_scale非法")
    return dict(domain, near=near, far=far, depth_scale=scale * factor,
                nominal_near=nominal_near, nominal_far=nominal_far,
                nominal_depth_scale=scale, unit_jitter_factor=factor,
                range_augmentation_mode=mode, range_was_repaired=repaired)


__all__ = ["V8Dataset", "SharedDataset", "compose_training_domain"]
