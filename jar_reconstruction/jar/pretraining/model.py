"""可训练标尺候选模型，复用已验证的分布传递与连续解码。"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from jar.geometry import pixel_grid, project, sample_maps, safe_masked_softmax, unproject
from jar.models.ruler_direct import RulerDirectDecoder
from .encoders import MatchingPyramid, TrainableFoundation
from .fast_fusion import factorized_visual
from .projective_ruler import compile_projective_ruler


class CandidateViewAttention(nn.Module):
    """分组相关、源外观和标尺共同控制视图权重，保留null质量。"""

    def __init__(self, channels, heads=4, groups=8):
        super().__init__()
        self.channels, self.heads, self.groups = channels, heads, groups
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.pair = nn.Sequential(nn.Linear(channels * 2 + groups + 3, channels),
                                  nn.GELU(), nn.Linear(channels, channels))
        self.reliability = nn.Linear(channels, heads)
        self.null = nn.Linear(channels, heads)
        self.hoist_reference = False
        self.compact_pair = False
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        nn.init.zeros_(self.null.weight)
        nn.init.zeros_(self.null.bias)

    def forward(self, reference, source, source_valid, ray_derivative,
                cell_half_width, use_geometry=True):
        batch, views, channels, depth, height, width = source.shape
        reference_pixels = reference.permute(0, 2, 3, 1)
        anchor = reference_pixels[:, None].expand(-1, depth, -1, -1, -1)
        # query/null 与候选深度无关，只计算每个参考像素的一份投影。
        projection_input = reference_pixels if self.hoist_reference else anchor
        query = self.query(projection_input).unflatten(-1, (self.heads, channels // self.heads))
        null = self.null(projection_input).float()
        if self.hoist_reference:
            query = query[:, None]
            null = null[:, None].expand(-1, depth, -1, -1, -1)
        # 分组求和后再线性映射，可将分组权重折入对应乘积通道。
        # 混合精度会在原线性层入口舍入，不能跨越此舍入合并权重。
        compact_pair = self.compact_pair and not torch.is_autocast_enabled(source.device.type)
        if compact_pair:
            weight = self.pair[0].weight
            grouped_weight = weight[:, 2 * channels:2 * channels + self.groups]
            product_weight = weight[:, :channels] + grouped_weight.repeat_interleave(channels // self.groups, 1)
            weight = torch.cat((product_weight, weight[:, channels:2 * channels], weight[:, -3:]), 1)
            padding = (-weight.shape[1]) % 16
            pair_weight = F.pad(weight, (0, padding))
        values, logits = [], []
        half = torch.as_tensor(cell_half_width, device=source.device).float()
        if half.ndim == 1:
            half = half[None, :, None, None]
        for view in range(views):
            sampled = source[:, view].permute(0, 2, 3, 4, 1)
            derivative = ray_derivative[:, view].float().movedim(1, -1)
            length = torch.linalg.vector_norm(derivative, dim=-1)
            direction = derivative / length[..., None].clamp_min(1e-6)
            geometry = torch.cat((direction, (length * half).log1p().clamp_max(8)[..., None]), -1)
            geometry = geometry * float(use_geometry)
            product = anchor * sampled
            difference = (anchor - sampled).abs()
            if compact_pair:
                zeros = product.new_zeros((*product.shape[:-1], padding))
                pair = torch.cat((product, difference, geometry, zeros), -1)
                hidden = F.linear(pair, pair_weight, self.pair[0].bias)
                value = self.pair[2](self.pair[1](hidden))
                del hidden, zeros
            else:
                grouped = product.unflatten(-1, (self.groups, channels // self.groups)).sum(-1)
                pair = torch.cat((product, difference, grouped, geometry), -1)
                value = self.pair(pair)
                del grouped
            # 推理时及时释放大候选临时张量，避免与下一视图的临时张量重叠。
            del pair, product, difference, geometry
            key = self.key(sampled).unflatten(-1, (self.heads, channels // self.heads))
            score = (query.float() * key.float()).sum(-1) / math.sqrt(channels // self.heads)
            del key
            values.append(value)
            score = score + self.reliability(value).float()
            logits.append(score)
        logits = torch.stack(logits, 1)
        null = null[:, None]
        valid = torch.cat((source_valid, torch.ones_like(source_valid[:, :1])), 1)
        weights = safe_masked_softmax(torch.cat((logits, null), 1), valid[..., None], dim=1)
        values = torch.stack(values, 1).unflatten(-1, (self.heads, channels // self.heads))
        pooled = (values * weights[:, :views, ..., None]).sum(1).flatten(-2).movedim(-1, 1)
        mean_weight = weights[:, :views].mean(-1)
        information = (mean_weight * ray_derivative.square().sum(2)).sum(1, keepdim=True)
        result = {"pooled": pooled, "weights": mean_weight,
                  "null_mass": weights[:, views:].mean(-1), "information": information}
        return result


class CalibratedDecoder(RulerDirectDecoder):
    def __init__(self, settings):
        super().__init__(settings)
        channels = settings["channels"]
        self.source = CandidateViewAttention(channels)
        self.evidence = nn.Sequential(nn.Conv3d(channels + 2, channels, 1), nn.GELU(),
                                      nn.Conv3d(channels, channels, 1))
        # 可选场景分支不进入本次模型，避免保留无用参数。
        self.scene = None
        self.independent_fine_heads = bool(settings.get("independent_fine_heads", False))
        if self.independent_fine_heads:
            self.fine_score = copy.deepcopy(self.score)
            self.fine_offset = copy.deepcopy(self.offset)

    def set_range(self, near, far):
        self.near, self.far = float(near), float(far)
        self.depth_lattice = torch.logspace(math.log10(near), math.log10(far), self.count,
                                            device=self.depth_lattice.device)

    def _prepare(self, inputs):
        raw = [adapter(value) for adapter, value in zip(self.raw_adapters, inputs["raw_maps"])]
        return {"matching": inputs["pyramid"][4], "pyramid": inputs["pyramid"],
                "K": inputs["K"].float(), "T": inputs["T_cam_from_rig"].float(),
                "view_mask": inputs["view_mask"], "image_mask": inputs["image_mask"],
                "scale": inputs["matching_scale"], "reference": inputs["reference_index"],
                "image_size": inputs["image_size"], "raw": raw}

    def _observe(self, prepared, candidates, stride, half_width):
        batch, count, height, width = candidates.shape
        reference = prepared["reference"]
        channels = self.settings["channels"]
        views = prepared["K"].shape[1]
        # 参考槽的源权重恒为零，直接省去其投影、采样与候选注意力。
        # 单图仍保留旧的全无效占位槽，供下游空源归约安全使用。
        compact = self.settings.get("compact_sources", False) and views > 1
        source_indices = [index for index in range(views) if not compact or index != reference]
        grid = pixel_grid(height, width, device=candidates.device, stride=stride).reshape(-1, 2)
        records = []
        for sample in range(batch):
            K, T = prepared["K"][sample], prepared["T"][sample]
            # 几何必须FP32；候选位置的梯度仍保留。
            with torch.autocast(device_type=candidates.device.type, enabled=False):
                scale = prepared["scale"][sample]
                widths = half_width if half_width.ndim == 1 else half_width[sample]
                widths = widths[:, None].expand(-1, height * width).flatten() if widths.ndim == 1 else widths.flatten()
                if self.settings.get("projective_ruler", False):
                    ruler = compile_projective_ruler(grid, K[reference], T[reference],
                                                     K[source_indices], T[source_indices])
                    depths = candidates[sample].float().flatten(1)
                    geometry = ruler.project(depths, prepared["image_size"],
                                             prepared["view_mask"][sample, source_indices],
                                             prepared["image_mask"][sample, source_indices])
                    uv = geometry["uv"].flatten(1, 2)
                    valid = geometry["valid"].flatten(1, 2)
                    derivative = geometry["depth_derivative"].flatten(1, 2)
                    footprint = geometry["pixel_derivative"].flatten(1, 2)
                    spans = torch.stack((
                        torch.linalg.vector_norm(footprint[..., 0] * scale, dim=-1) * stride,
                        torch.linalg.vector_norm(footprint[..., 1] * scale, dim=-1) * stride,
                        torch.linalg.vector_norm(derivative * scale, dim=-1) * widths * 2), -1)
                    needs_points = self.use_scene or self.local_kind in {"plane_flat", "plane_metric"}
                    points = ruler.world_points(depths) if needs_points else None
                else:
                    points, rays = unproject(grid[None].expand(count, -1, -1), candidates[sample].float().flatten(1),
                                             K[reference], T[reference])
                    uv, _, valid, J = project(
                        points.reshape(-1, 3), K[source_indices], T[source_indices], prepared["image_size"],
                        prepared["view_mask"][sample, source_indices], prepared["image_mask"][sample, source_indices])
                    derivative = (J * rays.reshape(1, -1, 1, 3)).sum(-1)
                    dx = torch.linalg.inv(K[reference])[:, 0] @ T[reference, :3, :3]
                    dy = torch.linalg.inv(K[reference])[:, 1] @ T[reference, :3, :3]
                    z = candidates[sample].float().flatten()
                    spans = torch.stack((
                        torch.linalg.vector_norm((J @ dx) * scale, dim=-1) * z * stride,
                        torch.linalg.vector_norm((J @ dy) * scale, dim=-1) * z * stride,
                        torch.linalg.vector_norm(derivative * scale, dim=-1) * widths * 2), -1)
                # 深度与像素横向的投影跨度决定读取尺度；几何分辨率不等于匹配置信度。
                level = (spans.amax(-1).clamp_min(1) / 16).log2().clamp(0, 2)
                if not self.settings.get("scale_routing", True):
                    level = torch.zeros_like(level)
                feature_uv = (uv + .5) * scale - .5
            sampled = 0
            for index, feature_stride in enumerate((4, 8, 16)):
                maps = prepared["pyramid"][feature_stride][sample, source_indices]
                weight = (1 - (level - index).abs()).clamp_min(0)
                value = sample_maps(maps.float(), feature_uv, stride=feature_stride, padding_mode="border")
                sampled = sampled + value * weight[..., None]
            anchor_uv = (grid[None] + .5) * scale - .5
            anchor = sample_maps(prepared["pyramid"][4][sample, reference:reference + 1].float(),
                                 anchor_uv, stride=4, padding_mode="border")[0].T.reshape(channels, height, width)
            ref_valid = sample_maps(prepared["image_mask"][sample, reference:reference + 1, None].float(),
                                    grid[None])[0, :, 0] > .999
            ref_valid = ref_valid & prepared["view_mask"][sample, reference]
            valid = valid & (torch.linalg.vector_norm(derivative, dim=-1) > 1e-5)
            if not compact:
                valid[reference] = False
            valid = valid.reshape(-1, count, height, width) & ref_valid.reshape(1, 1, height, width)
            source = sampled.transpose(1, 2).reshape(-1, channels, count, height, width)
            source = F.normalize(torch.where(valid[:, None], source, 0), dim=1, eps=1e-6)
            records.append({"reference": F.normalize(anchor, dim=0, eps=1e-6), "source": source,
                            "source_valid": valid, "ref_valid": ref_valid.reshape(1, height, width),
                            "ray_derivative": (derivative * scale / 4).transpose(1, 2).reshape(-1, 2, count, height, width),
                            "points": points.reshape(count, height, width, 3).permute(3, 0, 1, 2)
                            if points is not None else None})
        result = {key: torch.stack([record[key] for record in records])
                  for key in records[0] if records[0][key] is not None}
        result["cell_half_width"] = half_width
        result["stride"] = stride
        result["reference_index"] = reference
        result["view_count"] = views
        return result

    def _tokens(self, prepared, observation, candidates, stride):
        # 对最大候选源体重算注意力，避免保存成倍增长的拼接激活。
        if self.training:
            original = self.source
            source = checkpoint(original, observation["reference"], observation["source"],
                                observation["source_valid"], observation["ray_derivative"],
                                observation["cell_half_width"], self.use_geometry, use_reentrant=False)
            return self._tokens_from_source(prepared, observation, candidates, stride, source)
        source = self.source(observation["reference"], observation["source"],
                             observation["source_valid"], observation["ray_derivative"],
                             observation["cell_half_width"], self.use_geometry)
        return self._tokens_from_source(prepared, observation, candidates, stride, source)

    def _prediction_heads(self, observation):
        if self.independent_fine_heads and observation["stride"] == 4:
            return self.fine_score, self.fine_offset
        return self.score, self.offset

    def _prediction(self, features, candidates, valid, similarity, support, lower, upper, observation, use_offset=True):
        result = super()._prediction(features, candidates, valid, similarity, support, lower, upper, observation, use_offset)
        # 对外仍以原相机编号返回掩码；只压缩内部计算，不改变接口含义。
        sources = result["source_valid"]
        if sources.shape[1] != observation["view_count"]:
            reference = observation["reference_index"]
            result["source_valid"] = torch.cat((sources[:, :reference], torch.zeros_like(sources[:, :1]),
                                                sources[:, reference:]), dim=1)
        result.update(lower=lower.expand_as(candidates), upper=upper.expand_as(candidates))
        return result

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        metric = source["information"].clamp_min(0).log1p() * float(self.use_geometry)
        features = self.evidence(torch.cat((source["pooled"], source["null_mass"], metric), 1))
        position = (candidates.log() - math.log(self.near)) / math.log(self.far / self.near)
        position = torch.stack((position, (math.pi * position).sin(), (math.pi * position).cos()), -1)
        features = features + self.position(position).movedim(-1, 1)
        raw = [self._aligned(value, candidates.shape[-2:], stride) for value in prepared["raw"]]
        bank = torch.stack(raw, 2)
        if self.settings.get("factorized_fusion", False):
            visual = factorized_visual(features, bank, self.raw_query, self.raw_output)
        else:
            query = self.raw_query(features)
            logits = torch.einsum("bcdhw,bclhw->bldhw", query, bank).float() / math.sqrt(bank.shape[1])
            visual = torch.einsum("bldhw,bclhw->bcdhw", logits.softmax(1).to(bank.dtype), bank)
            visual = self.raw_output(visual)
        valid = observation["ref_valid"].expand_as(candidates)
        features = self.entry_norm(features + visual) * valid[:, None]
        cosine = (observation["source"] * observation["reference"][:, None, :, None]).sum(2)
        similarity = (cosine * source["weights"]).sum(1)
        return features, raw, valid, similarity, observation["source_valid"].sum(1), source


class PretrainingModel(nn.Module):
    # 旧checkpoint使用此序列化标识；公开版本名由实际解码结构决定。
    architecture = "ruler_direct_v2"

    def __init__(self, config, initialize=True):
        super().__init__()
        self.config = config
        self.foundation = TrainableFoundation(config["initialization"]["foundation"], initialize)
        self.matching = MatchingPyramid(config["initialization"]["matching"], config["model"]["channels"], initialize)
        decoder_kind = config['model'].get('decoder_kind', 'calibrated')
        if decoder_kind == 'surface_transport':
            from .surface_decoder import SurfaceDecoder

            self.decoder = SurfaceDecoder(config['model'])
            self.model_name = "RulerMVS-V3"
        elif decoder_kind == 'calibrated':
            self.decoder = CalibratedDecoder(config['model'])
            self.model_name = "RulerMVS-V2"
        else:
            raise ValueError(f'未知decoder_kind: {decoder_kind}')
        if config["model"].get("spatial_layout", False):
            from .fast_spatial import install_spatial_layout

            install_spatial_layout(self)

    def set_stage(self, number):
        self.foundation.set_stage(number)

    def encode(self, inputs, domain, reference=0):
        """编码RGB与标定，不读取标签；冻结编码器实验可缓存返回值。"""
        batch, _, _, height, width = inputs["rgb"].shape
        scale = float(domain["depth_scale"])
        transformed = inputs["T_cam_from_rig"].clone().float()
        transformed[..., :3, 3] = transformed[..., :3, 3] / scale
        reference_valid = inputs["image_mask"][:, reference] & inputs["view_mask"][:, reference, None, None]
        image = torch.where(reference_valid[:, None], inputs["rgb"][:, reference], 0.)
        raw = self.foundation(image)
        pyramid = self.matching(inputs)
        match_hw = inputs["matching_rgb"].shape[-2:]
        return {"pyramid": pyramid, "raw_maps": raw, "K": inputs["K"], "T_cam_from_rig": transformed,
                "view_mask": inputs["view_mask"], "image_mask": inputs["image_mask"],
                "reference_index": reference, "image_size": (height, width),
                "matching_scale": inputs["K"].new_tensor((match_hw[1] / width, match_hw[0] / height))[None].expand(batch, -1)}

    def forward(self, inputs, domain, reference=0):
        scale = float(domain["depth_scale"])
        self.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
        work = self.encode(inputs, domain, reference)
        prediction = self.decoder(work)
        prediction["pyramid"] = work["pyramid"] if self.training else None
        prediction["unit_scale"] = scale
        return prediction


def load_pretraining_state(model, state):
    """严格迁移：显式处理独立细头，以及可选的新表面readout替换。"""
    standalone = isinstance(model, CalibratedDecoder)
    decoder = model if standalone else model.decoder
    prefix = "" if standalone else "decoder."
    copies = {prefix + "fine_" + head + "." + parameter: prefix + head + "." + parameter
              for head in ("score", "offset") for parameter in ("weight", "bias")}
    if decoder.independent_fine_heads and copies.keys().isdisjoint(state):
        state = dict(state)
        state.update({target: state[source].clone() for target, source in copies.items()})
    surface_prefix = prefix + 'surface_readout.'
    if hasattr(decoder, 'surface_readout') and not any(key.startswith(surface_prefix) for key in state):
        legacy_prefix = prefix + 'upsample_mask.'
        expected = {legacy_prefix + name for name in ('0.weight', '0.bias', '2.weight', '2.bias')}
        found = {key for key in state if key.startswith(legacy_prefix)}
        if found != expected:
            raise ValueError('表面版本迁移需要完整且已知的旧上采样头')
        state = {key: value for key, value in state.items() if key not in expected}
        state.update({key: value for key, value in model.state_dict().items() if key.startswith(surface_prefix)})
    return model.load_state_dict(state, strict=True)
