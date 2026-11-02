"""未通过升级门槛的实验插拔，生产入口不加载本文件。

构造/加载普通 PretrainingModel 后，研究入口显式调用 install_variants(model)。
只切换既有实例的方法类，不重建参数、不改变 state_dict 键、不消费全局 RNG。
注意力 forward 保留实验时的运算顺序，便于精确重现实验；不是生产扩展接口。
"""

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from jar.geometry import safe_masked_softmax
from jar.pretraining.model import CalibratedDecoder, CandidateViewAttention
from .ray_evidence import corrupt_local_source


class ExperimentalAttention(CandidateViewAttention):
    """保留 round1/round2 原型的数学与浮点运算顺序。"""

    _variant_relative = False

    def forward(self, reference, source, source_valid, ray_derivative,
                cell_half_width, use_geometry=True, relative_evidence=None):
        relative_evidence = self._variant_relative if relative_evidence is None else relative_evidence
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
        values, logits, relative = [], [], []
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
            if relative_evidence:
                from .ray_evidence import relative_response

                contrast = relative_response(product.sum(-1), source_valid[:, view])
                relative.append(contrast)
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
            if relative_evidence:
                score = score + 4 * contrast[..., None]
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
        if relative_evidence:
            result["relative_similarity"] = (torch.stack(relative, 1) * mean_weight).sum(1)
        return result


class ExperimentalDecoder(CalibratedDecoder):
    def _observe(self, prepared, candidates, stride, half_width):
        result = super()._observe(prepared, candidates, stride, half_width)
        if self.training and self._variant_intervention > 0:
            result["source"] = corrupt_local_source(
                result["source"], result["source_valid"],
                self._variant_intervention, self._variant_generator,
            )
        return result

    def _tokens(self, prepared, observation, candidates, stride):
        relative = self._variant_relative or (self._variant_coarse and stride == 8)
        arguments = (observation["reference"], observation["source"],
                     observation["source_valid"], observation["ray_derivative"],
                     observation["cell_half_width"], self.use_geometry, relative)
        if self.training:
            source = checkpoint(self.source, *arguments, use_reentrant=False)
        else:
            source = self.source(*arguments)
        result = super()._tokens_from_source(prepared, observation, candidates, stride, source)
        if self._variant_relative:
            # round1 同时替换评分旁路；round2 coarse-only 保留绝对 cosine。
            result = (*result[:3], source["relative_similarity"], *result[4:])
        return result


def install_variants(model, settings=None):
    """显式研究安装；可传普通模型/DDP包装或单独的 CalibratedDecoder。"""
    core = getattr(model, "module", model)
    decoder = getattr(core, "decoder", core)
    settings = decoder.settings if settings is None else settings
    relative = bool(settings.get("source_relative_evidence", False))
    coarse = bool(settings.get("source_relative_coarse_attention", False))
    probability = float(settings.get("source_intervention_probability", 0.))
    seed = int(settings.get("source_intervention_seed", 20261030))
    if not 0 <= probability <= 1:
        raise ValueError("source_intervention_probability 必须位于 [0,1]")
    signature = relative, coarse, probability, seed
    if isinstance(decoder, ExperimentalDecoder):
        if signature != decoder._variant_signature:
            raise ValueError("不在已安装实例上切换实验；请重新构造模型")
        return model
    if not (relative or coarse or probability):
        return model
    if type(decoder) is not CalibratedDecoder or type(decoder.source) is not CandidateViewAttention:
        raise TypeError("本原型仅支持原 V2 CalibratedDecoder/Attention")
    # 仅研究插拔：复用原实例、Parameter 和 hook，禁止重新执行 __init__。
    decoder.__class__ = ExperimentalDecoder
    decoder.source.__class__ = ExperimentalAttention
    decoder._variant_signature = signature
    decoder._variant_relative = relative
    decoder._variant_coarse = coarse
    decoder._variant_intervention = probability
    decoder.source._variant_relative = relative
    decoder._variant_generator = torch.Generator().manual_seed(seed) if probability else None
    return model
