"""可选的官方预训练匹配特征支路；不包含深度头，不替代自研分层解码器。"""

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import sample_maps


def sample_pretrained_matching(features, image_uv, image_scale=None):
    """将原图整数像素坐标转换为 Niantic stride-4 特征索引。

    原生特征中心为原图的 1.5、5.5、9.5……像素，不能直接使用 u / 4。
    features: [N,C,H/4,W/4]；image_uv: [N,Q,2]；返回 [N,Q,C]。
    """
    if image_scale is None:
        feature_uv = (image_uv + 0.5) / 4 - 0.5
    else:
        # u_match=(u_scene+.5)*scale-.5，再映射至1.5+4*j格点。
        scale = torch.as_tensor(image_scale, device=image_uv.device, dtype=image_uv.dtype)
        feature_uv = (image_uv + 0.5) * (scale / 4) - 0.5
    return sample_maps(features, feature_uv, stride=1)


def pretrained_cosine_cost(features, reference_uv, reference_ids, projected_uv, candidates):
    """与原候选一一对应的 [query,depth,view] 代价，不改变其几何有效性。"""
    views, queries = len(features), len(reference_uv)
    reference = sample_pretrained_matching(
        features, reference_uv[None].expand(views, -1, -1)
    )[reference_ids, torch.arange(queries, device=reference_uv.device)]
    sampled = sample_pretrained_matching(features, projected_uv)
    sampled = sampled.reshape(views, queries, candidates, -1).permute(1, 2, 0, 3)
    reference = F.normalize(reference.float(), dim=-1, eps=1e-6)
    sampled = F.normalize(sampled.float(), dim=-1, eps=1e-6)
    return (1 - (sampled * reference[:, None, None]).sum(-1)).clamp(0, 2)


class FrozenMVSMatching(nn.Module):
    """严格加载 MVSAnywhere 的 ResNet 匹配编码器；仅作为已知方法迁移。

    默认期望完整官方 Hero checkpoint 的 matching_model.* 子树；使用本项目
    CARLA checkpoint 时需显式 prefix='native_model.matching_model.'。
    无默认下载，不加载深度/语义标签，不读取官方深度预测头。
    """

    output_channels = 16
    stride = 4

    def __init__(self, weights=None, prefix="matching_model.", *, state=None):
        super().__init__()
        source = Path(__file__).resolve().parents[1] / "official_methods/upstream/mvsanywhere/src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from mvsanywhere.modules.networks import ResnetMatchingEncoder

        # 加载固定分支不应推进主模型/训练采样器使用的随机数状态。
        with torch.random.fork_rng(devices=[]):
            self.encoder = ResnetMatchingEncoder(18, self.output_channels, pretrained=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        if state is not None:
            # 完整自研 checkpoint 自带编码器及归一化状态；不读取旧外部路径。
            # 有部分子树也必须严格匹配，不能从外部文件偷偷补齐损坏状态。
            self.load_state_dict(state, strict=True)
            self.initialization_source = "checkpoint"
        else:
            if not weights:
                raise ValueError("启用预训练匹配须提供完整checkpoint子树或已缓存weights路径")
            saved = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
            source_state = saved.get("state_dict", saved.get("model", saved))
            selected = {
                key[len(prefix) :]: value
                for key, value in source_state.items()
                if key.startswith(prefix)
            }
            self.encoder.load_state_dict(selected, strict=True)
            self.initialization_source = str(weights)
        self.encoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # 冻结参数时同样冻结 BatchNorm 的运行统计。
        return self

    def forward(self, inputs):
        matching = "matching_rgb" in inputs
        rgb = inputs["matching_rgb"] if matching else inputs["rgb"]
        batch, views, channels, height, width = rgb.shape
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, dtype=torch.bool, device=rgb.device)
        )
        image = torch.where(view_mask[:, :, None, None, None], rgb.float(), 0.0)
        mask_key = "matching_image_mask" if matching else "image_mask"
        if matching and mask_key not in inputs:
            raise ValueError("matching_rgb必须同时提供matching_image_mask")
        if mask_key in inputs:
            image = torch.where(inputs[mask_key][:, :, None], image, 0.0)
        image = image.reshape(batch * views, channels, height, width)
        # 参数固定不等于候选位置固定；后续 grid_sample 对采样坐标仍可求梯度。
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            features = self.encoder((image - self.mean) / self.std)
        features = features.reshape(batch, views, *features.shape[1:])
        return features * view_mask[:, :, None, None, None]
