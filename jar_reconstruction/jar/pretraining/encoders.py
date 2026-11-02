"""独立于Hero权重的可训练编码器；构造时不联网。"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18

from jar.geometry import pixel_grid, sample_maps
from jar.models.deep_encoder import _build_native_foundation


class TrainableFoundation(nn.Module):
    def __init__(self, weights, initialize=True):
        super().__init__()
        self.model = _build_native_foundation()
        self.indices = (2, 5, 8, 11)
        if initialize:
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            encoder = {key.removeprefix("pretrained."): value for key, value in state.items()
                       if key.startswith("pretrained.")}
            if not encoder:
                raise ValueError("需要原始DA-V2 pretrained.*权重；禁止Hero子树")
            self.model.load_state_dict(encoder, strict=True)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def set_stage(self, stage):
        self.requires_grad_(stage >= 3)
        if stage == 2:
            for block in self.model.blocks[-4:]:
                block.requires_grad_(True)
            self.model.norm.requires_grad_(True)

    def forward(self, image):
        image = (image - self.mean) / self.std
        image = F.interpolate(image, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        tokens = self.model.prepare_tokens_with_masks(image)
        height, width = image.shape[-2] // 14, image.shape[-1] // 14
        outputs = []
        for index, block in enumerate(self.model.blocks):
            if self.training and any(p.requires_grad for p in block.parameters()):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self.indices:
                normalized = self.model.norm(tokens)[:, 1:]
                outputs.append(normalized.transpose(1, 2).reshape(len(image), 768, height, width))
        return outputs


class MatchingPyramid(nn.Module):
    def __init__(self, weights, channels=64, initialize=True):
        super().__init__()
        self.backbone = resnet18(weights=None)
        if initialize:
            self.backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.backbone.fc = nn.Identity()
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in (64, 128, 256, 512))
        self.smooth = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(8, channels), nn.GELU()
        ) for _ in range(3))
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, inputs):
        image = inputs["matching_rgb"]
        batch, views = image.shape[:2]
        valid = inputs["view_mask"][:, :, None, None] & inputs["matching_image_mask"]
        image = torch.where(valid[:, :, None], image, 0).flatten(0, 1)
        image = (image - self.mean) / self.std
        net = self.backbone
        value = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        raw = []
        for layer, projection in zip((net.layer1, net.layer2, net.layer3, net.layer4), self.lateral):
            value = checkpoint(layer, value, use_reentrant=False) if self.training else layer(value)
            raw.append(projection(value))
        maps, top = {}, None
        for index in (3, 2, 1, 0):
            stride = 4 * 2 ** index
            value = raw[index]
            if top is not None:
                grid = pixel_grid(*value.shape[-2:], device=value.device, stride=stride)
                uv = grid.reshape(1, -1, 2).expand(batch * views, -1, -1)
                up = sample_maps(top.float(), uv, stride=stride * 2, padding_mode="border")
                value = value + up.transpose(1, 2).reshape_as(value).to(value.dtype)
            top = value
            if index < 3:
                maps[stride] = self.smooth[index](value).unflatten(0, (batch, views))
        return maps
