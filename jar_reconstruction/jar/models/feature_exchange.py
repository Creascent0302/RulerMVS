"""沿标定候选交流图像特征；交流后的相似性不等同于独立观测证据。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.depth_sampling import configured_inverse_depths
from jar.geometry import (
    camera_centers,
    image_texture,
    pixel_grid,
    project,
    safe_masked_softmax,
    sample_maps,
    unproject,
)
from jar.pretrained_matching import sample_pretrained_matching


class ExchangeLayer(nn.Module):
    """查询/键读取局部特征和上下文，消息值始终读取原CNN特征。"""

    def __init__(self, channels, hidden):
        super().__init__()
        self.query = nn.Conv2d(channels * 2, hidden, 1, bias=False)
        self.key = nn.Conv2d(channels * 2, hidden, 1, bias=False)
        self.no_message = nn.Conv2d(channels * 2, 1, 1)
        nn.init.normal_(self.no_message.weight, std=1e-3)
        nn.init.zeros_(self.no_message.bias)
        self.scale = math.sqrt(hidden)


class CalibratedFeatureExchange(nn.Module):
    """stride16候选交流，再向4/8/16/32格点传播小幅残差。"""

    strides = (4, 8, 16, 32)

    def __init__(self, cfg):
        super().__init__()
        model = cfg["model"]
        settings = model.get("feature_exchange", {})
        channels = int(model.get("feature_dim", 64))
        hidden = int(settings.get("hidden_dim", 64))
        layers = int(settings.get("layers", 1))
        bins = int(settings.get("depth_bins", 32))
        self.chunk_size = int(settings.get("chunk_size", 128))
        self.prior_weight = float(settings.get("prior_weight", 2.0))
        self.texture_threshold = float(model.get("texture_threshold", 0.002))
        gate = float(settings.get("initial_gate", 0.1))
        if min(channels, hidden, layers, self.chunk_size) < 1 or not 0 < gate < 1:
            raise ValueError("feature_exchange维度、层数和chunk须为正，initial_gate须在(0,1)")
        if self.prior_weight < 0:
            raise ValueError("feature_exchange.prior_weight须非负")
        self.register_buffer("inverse_depths", configured_inverse_depths(model, bins))
        self.layers = nn.ModuleList([ExchangeLayer(channels, hidden) for _ in range(layers)])
        self.adapters = nn.ModuleDict()
        self.gate_logits = nn.ParameterDict()
        for stride in self.strides:
            name = str(stride)
            adapter = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.eye_(adapter.weight[:, :, 0, 0])
            self.adapters[name] = adapter
            self.gate_logits[name] = nn.Parameter(torch.tensor(math.log(gate / (1 - gate))))

    def _chunk(
        self,
        uv,
        reference,
        values,
        keys,
        queries,
        null_logits,
        intrinsic,
        transform,
        view_mask,
        image_mask,
        texture,
        prior_features,
        image_size,
        scale,
    ):
        views, channels = values.shape[:2]
        pixels, depths = len(uv), len(self.inverse_depths)
        z = self.inverse_depths.reciprocal()[None].expand(pixels, -1)
        points, rays = unproject(
            uv[:, None].expand(-1, depths, -1), z, intrinsic[reference], transform[reference]
        )
        projected, _, valid, jacobian = project(
            points.reshape(-1, 3), intrinsic, transform, image_size, view_mask, image_mask
        )
        derivative = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1).norm(dim=-1)
        centers = camera_centers(transform)
        independent = (centers - centers[reference]).norm(dim=-1) > 1e-4
        source_texture = sample_maps(texture, projected)[..., 0]
        reference_texture = sample_maps(texture[reference : reference + 1], uv[None])[0, :, 0]
        reference_mask = (
            sample_maps(image_mask[reference : reference + 1, None].float(), uv[None])[0, :, 0]
            > 0.999
        )
        valid = valid & independent[:, None] & (derivative > 1e-5)
        valid = valid & (source_texture > self.texture_threshold)
        valid = valid.reshape(views, pixels, depths)
        valid = valid & view_mask[reference] & reference_mask[None, :, None]
        valid = valid & (reference_texture > self.texture_threshold)[None, :, None]
        valid[reference] = False

        sampled_keys = sample_maps(keys, projected, stride=16).reshape(views, pixels, depths, -1)
        source_keys = F.normalize(sampled_keys.float(), dim=-1, eps=1e-6)
        query = F.normalize(queries.float(), dim=-1, eps=1e-6)
        score = (source_keys * query[None, :, None]).sum(-1) * scale
        if prior_features is not None:
            ref_prior = sample_pretrained_matching(
                prior_features[reference : reference + 1], uv[None]
            )[0]
            source_prior = sample_pretrained_matching(prior_features, projected)
            source_prior = source_prior.reshape(views, pixels, depths, -1)
            cosine = (
                F.normalize(source_prior.float(), dim=-1, eps=1e-6)
                * F.normalize(ref_prior.float(), dim=-1, eps=1e-6)[None, :, None]
            ).sum(-1)
            score = score + self.prior_weight * cosine

        score = score.permute(1, 0, 2).reshape(pixels, -1)
        mask = valid.permute(1, 0, 2).reshape(pixels, -1)
        # 消除候选数对消息/拒绝概率的机械偏置；无效候选仍严格屏蔽。
        score = score - mask.sum(-1, keepdim=True).clamp_min(1).float().log()
        score = torch.cat((score, null_logits.float()), -1)
        mask = torch.cat((mask, torch.ones_like(null_logits, dtype=torch.bool)), -1)
        probability = safe_masked_softmax(score, mask, dim=-1)[:, :-1]
        probability = probability.reshape(pixels, views, depths).permute(1, 0, 2)
        source_values = sample_maps(values, projected, stride=16).reshape(
            views, pixels, depths, channels
        )
        reference_values = sample_maps(values[reference : reference + 1], uv[None], stride=16)[0]
        difference = torch.where(
            valid[..., None], source_values - reference_values[None, :, None], 0
        )
        return (probability[..., None] * difference).sum((0, 2))

    def forward(self, inputs, pyramid, context=None):
        original = pyramid[16]
        batch, views, channels, height, width = original.shape
        image_h, image_w = inputs["rgb"].shape[-2:]
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, device=original.device, dtype=torch.bool)
        ).bool()
        image_mask = inputs.get(
            "image_mask",
            torch.ones(batch, views, image_h, image_w, device=original.device, dtype=torch.bool),
        ).bool()
        image_mask = image_mask & view_mask[:, :, None, None]
        if "image_sizes" in inputs:
            image_uv = pixel_grid(image_h, image_w, device=original.device)
            sizes = inputs["image_sizes"]
            image_mask = (
                image_mask
                & (image_uv[..., 0] < sizes[..., 1, None, None])
                & (image_uv[..., 1] < sizes[..., 0, None, None])
            )
        texture = inputs.get("_texture")
        if texture is None:
            texture = image_texture(inputs["rgb"].flatten(0, 1)).reshape(
                batch, views, 1, image_h, image_w
            )
        prior = inputs.get("_pretrained_matching_features") if self.prior_weight else None
        if self.prior_weight and prior is None:
            raise ValueError(
                "feature_exchange需要冻结的_pretrained_matching_features，或显式prior_weight=0"
            )
        semantic = (context or {}).get(16, torch.zeros_like(original))
        uv = pixel_grid(height, width, device=original.device, stride=16).reshape(-1, 2)
        current = original
        for layer in self.layers:
            combined = torch.cat((current, semantic), 2).flatten(0, 1)
            query = layer.query(combined).reshape(batch, views, -1, height, width)
            keys = layer.key(combined).reshape(batch, views, -1, height, width)
            null = layer.no_message(combined).reshape(batch, views, 1, height, width)
            updates = []
            for sample in range(batch):
                view_updates = []
                intrinsic = inputs["K"][sample].detach().float()
                transform = inputs["T_cam_from_rig"][sample].detach().float()
                for reference in range(views):
                    q = query[sample, reference].flatten(1).T
                    no_message = null[sample, reference].flatten(1).T
                    pieces = []
                    for start in range(0, len(uv), self.chunk_size):
                        stop = start + self.chunk_size
                        pieces.append(
                            self._chunk(
                                uv[start:stop],
                                reference,
                                original[sample],
                                keys[sample],
                                q[start:stop],
                                no_message[start:stop],
                                intrinsic,
                                transform,
                                view_mask[sample],
                                image_mask[sample],
                                texture[sample],
                                None if prior is None else prior[sample],
                                (image_h, image_w),
                                layer.scale,
                            )
                        )
                    view_updates.append(torch.cat(pieces).T.reshape(channels, height, width))
                updates.append(torch.stack(view_updates))
            current = current + torch.stack(updates).to(current.dtype)

        delta = (current - original).flatten(0, 1)
        result = {}
        for stride in self.strides:
            feature = pyramid[stride]
            target_uv = pixel_grid(*feature.shape[-2:], device=feature.device, stride=stride)
            target_uv = target_uv.reshape(1, -1, 2).expand(batch * views, -1, -1)
            sampled = sample_maps(delta, target_uv, stride=16, padding_mode="border")
            sampled = sampled.transpose(1, 2).reshape(batch * views, channels, *feature.shape[-2:])
            valid = (
                sample_maps(image_mask.flatten(0, 1)[:, None].float(), target_uv)[..., 0] > 0.999
            )
            valid = valid.reshape(batch, views, 1, *feature.shape[-2:])
            correction = self.adapters[str(stride)](sampled.to(feature.dtype)).reshape_as(feature)
            gate = self.gate_logits[str(stride)].sigmoid()
            result[stride] = feature + gate * correction * valid
        return result
