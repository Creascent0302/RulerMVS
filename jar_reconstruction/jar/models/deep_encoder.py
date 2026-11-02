"""保留像素格点的深残差编码，以及独立的冻结预训练场景上下文。

深 CNN 仍可提供局部匹配特征；DINO 上下文默认只增强解码 query，不增加
观测支持数。forward_features 显式返回二者，forward 返回相加后的特征。
"""

from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from jar.encoder import Block, ImageFPN, norm
from jar.geometry import pixel_grid, sample_maps


class ResidualStageBlock(nn.Module):
    """小的非零残差初值接近恒等映射，且首步即可训练全部卷积分支。"""

    def __init__(self, channels, initial_scale):
        super().__init__()
        self.body = Block(channels, channels).body
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, features):
        return features + self.scale * self.body(features)


def sample_foundation_maps(features, image_uv):
    """官方 14/16 缩放后，patch 中心在原图 7.5 + 16*i 像素。

    在图内边缘采用最近 patch 的上下文，不将该外推解释为观测；图外及
    image_mask 由调用方独立屏蔽。原 CNN 格点仍为 stride*i。
    """
    token_uv = (image_uv - 7.5) / 16.0
    return sample_maps(features, token_uv, stride=1, padding_mode="border")


def _reference_mask(inputs, views):
    """可选执行参考列表只裁减语义编码，不能删除用于匹配的源相机。"""
    references = inputs.get("_reference_indices")
    if references is None:
        return None
    if (
        not isinstance(references, (list, tuple))
        or not references
        or any(type(index) is not int or not 0 <= index < views for index in references)
        or len(set(references)) != len(references)
    ):
        raise ValueError("_reference_indices需要非空、不重复且位于相机范围内的整数列表")
    selected = torch.zeros(views, dtype=torch.bool, device=inputs["rgb"].device)
    selected[list(references)] = True
    return selected


def _build_native_foundation():
    """只加载项目固定的官方源码，不联网、不下载，也不更换官方 ViT 层。"""
    root = Path(__file__).resolve().parents[2]
    source = root / "official_methods/upstream/mvsanywhere/src"
    hub = root / "official_methods/upstream/dinov2"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mvsanywhere.modules.vit_modules import DINOv2

    return DINOv2(model_name="dinov2_vitb14", hub_source=str(hub), pretrained=False).model


class FrozenFoundation(nn.Module):
    """从 Hero 的 encoder.model.* 严格提取冻结的完整 12 层 ViT-B。

    已有自研 checkpoint 的 foundation 子树优先，完整恢复不再读取外部权重。
    返回第 3、6、9、12 个 block 的归一化 patch 特征；这是自研融合选择，
    不宣称复现 Hero 的 cost-volume 解码。
    """

    channels = 768

    def __init__(self, settings, state=None):
        super().__init__()
        self.indices = tuple(settings.get("indices", [2, 5, 8, 11]))
        if len(self.indices) != 4 or sorted(set(self.indices)) != list(self.indices):
            raise ValueError("foundation.indices 需要四个递增且不重复的零基层编号")
        if self.indices[0] < 0 or self.indices[-1] > 11:
            raise ValueError("foundation.indices 必须位于 0..11")
        self.model = _build_native_foundation()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        if state is not None:
            self.load_state_dict(state, strict=True)
            self.initialization_source = "checkpoint"
        else:
            weights = settings.get("weights", "")
            if not weights:
                raise ValueError("foundation 需要已有 Hero weights 或完整自研 checkpoint")
            saved = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
            state_dict = saved.get("state_dict", saved.get("model", saved))
            prefix = settings.get("prefix", "encoder.model.")
            selected = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            self.model.load_state_dict(selected, strict=True)
            self.initialization_source = str(weights)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, inputs):
        rgb = inputs["rgb"]
        batch, views, _, height, width = rgb.shape
        valid = inputs["view_mask"]
        selected = _reference_mask(inputs, views)
        if selected is not None:
            valid = valid & selected[None]
        valid = valid.reshape(-1)
        pad_height = math.ceil(height / 16) * 16
        pad_width = math.ceil(width / 16) * 16
        feature_shape = (batch * views, self.channels, pad_height // 16, pad_width // 16)
        if not bool(valid.any()):
            return [rgb.new_zeros(feature_shape, dtype=torch.float32) for _ in self.indices]
        # 仅处理真实相机。补齐右/下边界不会改变原图像素中心。
        image = rgb.reshape(batch * views, 3, height, width)[valid].float()
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, enabled=False):
            image = (image - self.mean) / self.std
            image = F.pad(image, (0, pad_width - width, 0, pad_height - height))
            image = F.interpolate(image, scale_factor=14 / 16, mode="bilinear", align_corners=False)
            features = self.model.get_intermediate_layers(image, self.indices, reshape=True)
        results = []
        for feature in features:
            full = feature.new_zeros(feature_shape)
            full[valid] = feature
            results.append(full)
        return results


class DeepImageFPN(ImageFPN):
    """五级深 CNN + 可选冻结场景编码，旧 ImageFPN 参数键保持不变。"""

    def __init__(self, cfg, checkpoint_state=None):
        super().__init__(cfg)
        settings = cfg["model"].get("deep_encoder", {})
        depths = settings.get("stage_blocks", [2, 3, 4, 3, 2])
        if len(depths) != 5 or any(type(depth) is not int or depth < 1 for depth in depths):
            raise ValueError("deep_encoder.stage_blocks 需要五个正整数，包含原下采样块")
        scale = float(settings.get("initial_scale", 0.01))
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("deep_encoder.initial_scale 必须位于 (0,1]")
        width = cfg["model"].get("backbone_width", 24)
        channels = cfg["model"].get("feature_dim", 64)
        stage_channels = [width, width * 2, width * 3, width * 4, width * 4]
        # 不推进旧编码层之后的主模型/解码器随机数序列。
        with torch.random.fork_rng(devices=[]):
            self.extra_stages = nn.ModuleList(
                [
                    nn.Sequential(*[ResidualStageBlock(ch, scale) for _ in range(depth - 1)])
                    for ch, depth in zip(stage_channels, depths)
                ]
            )
            foundation = settings.get("foundation", {})
            if foundation.get("enabled", False):
                checkpoint_state = checkpoint_state or {}
                prefix = "encoder.foundation."
                if not any(key.startswith(prefix) for key in checkpoint_state):
                    prefix = "foundation."
                state = {
                    key[len(prefix) :]: value
                    for key, value in checkpoint_state.items()
                    if key.startswith(prefix)
                }
                self.foundation = FrozenFoundation(foundation, state=state or None)
                self.foundation_adapters = nn.ModuleDict(
                    {
                        str(stride): nn.Sequential(
                            nn.Conv2d(self.foundation.channels, channels, 1, bias=False),
                            norm(channels),
                            nn.GELU(),
                        )
                        for stride in self.feature_strides
                    }
                )
                gate = float(foundation.get("initial_gate", 0.01))
                if not math.isfinite(gate) or not 0 < gate < 1:
                    raise ValueError("foundation.initial_gate 必须位于 (0,1)")
                self.foundation_gates = nn.ParameterDict(
                    {
                        str(stride): nn.Parameter(torch.tensor(math.atanh(gate)))
                        for stride in self.feature_strides
                    }
                )
            enrichment = cfg["model"].get("feature_enrichment", {})
            if enrichment.get("enabled", False):
                if not hasattr(self, "foundation"):
                    raise ValueError("feature_enrichment需要同时开启deep_encoder.foundation")
                from .feature_enrichment import FeatureEnrichment

                self.feature_enrichment = FeatureEnrichment(channels, enrichment)

    def refine_stage(self, stage, features):
        return self.extra_stages[stage](features)

    @staticmethod
    def _prepare(inputs):
        rgb = inputs["rgb"]
        batch, views, _, height, width = rgb.shape
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, device=rgb.device, dtype=torch.bool)
        ).bool()
        image_mask = inputs.get(
            "image_mask",
            torch.ones(batch, views, height, width, device=rgb.device, dtype=torch.bool),
        ).bool()
        image_mask = image_mask & view_mask[:, :, None, None]
        prepared = dict(inputs)
        prepared["rgb"] = torch.where(image_mask[:, :, None], rgb, 0.0)
        prepared["view_mask"] = view_mask
        prepared["image_mask"] = image_mask
        return prepared

    def forward_features(self, inputs, *, return_raw=False):
        """可选返回同次编码的原始特征，不重新运行 ViT 或保存跨帧状态。"""
        inputs = self._prepare(inputs)
        selected = _reference_mask(inputs, inputs["rgb"].shape[1])
        if selected is not None and hasattr(self, "feature_enrichment"):
            raise ValueError("feature_enrichment读取所有源视图语义特征，不能裁减参考语义编码")
        base = super().forward(inputs)
        context = {}
        valid_masks = {}
        foundation_maps = self.foundation(inputs) if hasattr(self, "foundation") else None
        batch, views, _, height, width = inputs["rgb"].shape
        masks = inputs["image_mask"].reshape(batch * views, 1, height, width).float()
        for level, stride in enumerate(self.feature_strides):
            features = base[stride]
            feature_height, feature_width = features.shape[-2:]
            uv = pixel_grid(feature_height, feature_width, device=features.device, stride=stride)
            uv = uv.reshape(1, -1, 2).expand(batch * views, -1, -1)
            valid = sample_maps(masks, uv, mode="nearest")[..., 0] > 0.5
            valid &= (uv[..., 0] < width) & (uv[..., 1] < height)
            valid = valid.reshape(batch, views, 1, feature_height, feature_width)
            valid_masks[stride] = valid
            base[stride] = features * valid
            if foundation_maps is None:
                continue
            adapted = self.foundation_adapters[str(stride)](foundation_maps[level])
            sampled = sample_foundation_maps(adapted, uv).transpose(1, 2)
            sampled = sampled.reshape(batch, views, *features.shape[2:])
            context_valid = valid
            if selected is not None:
                context_valid = context_valid & selected[None, :, None, None, None]
            context[stride] = sampled * self.foundation_gates[str(stride)].tanh() * context_valid
        if hasattr(self, "feature_enrichment"):
            base = self.feature_enrichment(base, foundation_maps, valid_masks)
        if return_raw:
            if foundation_maps is None:
                raise ValueError("原始特征读取需要启用 foundation")
            raw_maps = [
                feature.reshape(batch, views, *feature.shape[1:])
                for feature in foundation_maps
            ]
            return base, context, raw_maps
        return base, context

    def forward(self, inputs):
        base, context = self.forward_features(inputs)
        return {stride: features + context.get(stride, 0) for stride, features in base.items()}
