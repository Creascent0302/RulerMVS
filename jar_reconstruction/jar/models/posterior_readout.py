"""由匹配分布和连续深度软先验读出表面，不跨不同峰直接平均。"""

import torch
from torch import nn

from jar.geometry import pixel_grid, safe_masked_softmax, sample_maps


def resample_field(field, source_stride, target_stride, size):
    if source_stride == target_stride and tuple(field.shape[-2:]) == tuple(size):
        return field
    uv = pixel_grid(*size, device=field.device, stride=target_stride)
    uv = uv.reshape(1, -1, 2).expand(len(field), -1, -1)
    sampled = sample_maps(field, uv, stride=source_stride, padding_mode="border")
    return sampled.transpose(1, 2).reshape(len(field), field.shape[1], *size)


class PosteriorDepthReadout(nn.Module):
    """保留全候选的后验，语义仅修正候选分数及获胜单元内的位置。

    argmax选择峰不可微；峰内连续定位可微。训练必须另加候选后验监督，
    才能直接训练尚未获胜的候选。候选合法不代表表面已观测确认。
    """

    def __init__(self, channels, bins, near, far, settings):
        super().__init__()
        self.prior_sigma = float(settings.get("sigma_log_depth", 0.10))
        self.prior_limit = float(settings.get("prior_logit_limit", 8.0))
        self.matching_limit = float(settings.get("matching_logit_limit", 4.0))
        self.residual_limit = float(settings.get("residual_logit_limit", 4.0))
        self.full_offset_fraction = float(settings.get("full_offset_fraction", 0.5))
        self.row_chunk = int(settings.get("row_chunk", 32))
        self.use_matching = bool(settings.get("use_matching", True))
        self.near, self.far = near, far
        if min(self.prior_sigma, self.prior_limit, self.matching_limit, self.residual_limit) <= 0:
            raise ValueError("posterior_readout的sigma和logit范围须为正")
        if not 0 <= self.full_offset_fraction <= 1 or self.row_chunk < 1:
            raise ValueError("full_offset_fraction须在[0,1]且row_chunk>0")
        self.candidate_residual = nn.Conv2d(channels, bins, 1)
        nn.init.zeros_(self.candidate_residual.weight)
        nn.init.zeros_(self.candidate_residual.bias)

    @staticmethod
    def read_peak(probability, log_depths, fallback_log_depth):
        """选峰后连续定位，返回深度及该峰的Voronoi单元边界。"""
        count = len(log_depths)
        winner = probability.argmax(1, keepdim=True)
        offsets = torch.tensor([-1, 0, 1], device=probability.device)[None, :, None, None]
        indices = winner + offsets
        inside = (indices >= 0) & (indices < count)
        indices = indices.clamp(0, count - 1)
        local_probability = probability.gather(1, indices)
        local_valid = inside & (local_probability > 0)
        local_probability = torch.where(local_valid, local_probability, 0)
        samples = log_depths[indices]
        weights = local_probability / local_probability.sum(1, keepdim=True).clamp_min(1e-12)
        average = (weights * samples).sum(1, keepdim=True)

        # 在非均匀log深度格点拟合局部二次峰。纯Gaussian且峰在内部时，
        # 可恢复其连续均值，而不是被离散候选中心量化。
        values = local_probability.clamp_min(1e-30).log()
        left_step = (samples[:, 1:2] - samples[:, :1]).clamp_min(1e-6)
        right_step = (samples[:, 2:] - samples[:, 1:2]).clamp_min(1e-6)
        left_slope = (values[:, 1:2] - values[:, :1]) / left_step
        right_slope = (values[:, 2:] - values[:, 1:2]) / right_step
        curvature = 2 * (right_slope - left_slope) / (left_step + right_step)
        derivative = (right_step * left_slope + left_step * right_slope) / (left_step + right_step)
        reliable_fit = local_valid.all(1, keepdim=True) & (curvature < -1e-3)
        safe_curvature = torch.where(reliable_fit, curvature, -torch.ones_like(curvature))
        vertex = samples[:, 1:2] - derivative / safe_curvature
        center = samples[:, 1:2]
        lower = torch.where(winner > 0, (samples[:, :1] + center) / 2, center)
        upper = torch.where(winner < count - 1, (center + samples[:, 2:]) / 2, center)
        selected = torch.where(reliable_fit, vertex, average)
        selected = selected.clamp(min=lower, max=upper)
        valid = probability.sum(1, keepdim=True) > 0
        selected = torch.where(valid, selected, fallback_log_depth)
        return selected, lower, upper, winner, valid

    def forward(self, features, reference_depth, coarse, stride=4):
        batch, views = coarse["logits"].shape[:2]
        size = features.shape[-2:]
        depths, order = coarse["inverse_depths"].float().reciprocal().sort()
        log_depths = depths.log()
        scores = coarse["logits"][:, :, :-1].float().index_select(2, order).flatten(0, 1)
        valid = torch.isfinite(scores) & (scores > -9999)
        clean = torch.where(valid, scores, 0)
        coarse_stride = int(coarse.get("stride", 16))
        support = resample_field(valid.float(), coarse_stride, stride, size)
        scores = resample_field(clean, coarse_stride, stride, size) / support.clamp_min(1e-8)
        valid = support > 0.999
        maximum = scores.masked_fill(~valid, -1e4).amax(1, keepdim=True)
        matching = (scores - maximum).clamp(-self.matching_limit, 0)
        if not self.use_matching:
            matching = torch.zeros_like(matching)
        reference_log = reference_depth.float().clamp(self.near, self.far).log()
        prior = -0.5 * (
            (log_depths[None, :, None, None] - reference_log) / self.prior_sigma
        ).square()
        # 截断Gaussian尾部，保留远处其他合法候选翻峰的可能。
        prior = prior.clamp_min(-self.prior_limit)
        residual = self.candidate_residual(features).float()
        residual = self.residual_limit * torch.tanh(residual / self.residual_limit)
        logits = (matching + prior + residual).masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(logits, valid, dim=1)
        selected, lower, upper, winner, candidate_valid = self.read_peak(
            probability, log_depths, reference_log
        )
        shape = (batch, views, len(depths), *size)
        return {
            "logits": logits.reshape(shape),
            "valid": valid.reshape(shape),
            "inverse_depths": depths.reciprocal(),
            "stride": stride,
            "depth": selected.exp().reshape(batch, views, *size),
            "lower_depth": lower.exp().reshape(batch, views, *size),
            "upper_depth": upper.exp().reshape(batch, views, *size),
            "mode_index": winner.reshape(batch, views, *size),
            "candidate_valid": candidate_valid.reshape(batch, views, *size),
            "probability": probability.reshape(shape),
            "reference_log_depth": reference_log.reshape(batch, views, *size),
        }

    def full_resolution(self, posterior, original_depth):
        """上采样概率混合再选峰，避免上采样单深度混合前后表面。

        原完整分辨率头只能在获胜单元内修正；不参与跨峰深度覆盖。
        逐行块暂存全候选，完整分辨率只返回单深度及其边界。
        """
        batch, views, height, width = original_depth.shape
        probability = posterior["probability"].flatten(0, 1)
        reference = posterior["reference_log_depth"].flatten(0, 1)[:, None]
        log_depths = posterior["inverse_depths"].float().reciprocal().log()
        original_log = original_depth.flatten(0, 1)[:, None].clamp(self.near, self.far).log()
        rows = {key: [] for key in ("depth", "lower_depth", "upper_depth", "mode_index", "candidate_valid")}
        for start in range(0, height, self.row_chunk):
            stop = min(start + self.row_chunk, height)
            uv = pixel_grid(stop - start, width, device=probability.device)
            uv[..., 1] += start
            uv = uv.reshape(1, -1, 2).expand(batch * views, -1, -1)

            def sample(field):
                result = sample_maps(field, uv, stride=posterior["stride"], padding_mode="border")
                return result.transpose(1, 2).reshape(batch * views, field.shape[1], stop - start, width)

            mixed_probability = sample(probability)
            fallback = original_log[:, :, start:stop]
            selected, lower, upper, winner, valid = self.read_peak(
                mixed_probability, log_depths, fallback
            )
            inherited_log = sample(reference)
            change = fallback - inherited_log
            room = torch.where(change >= 0, upper - selected, selected - lower).clamp_min(0)
            limit = self.full_offset_fraction * room
            offset = limit * torch.tanh(change / limit.clamp_min(1e-6))
            selected = torch.where(valid, selected + offset, fallback)
            values = (selected.exp(), lower.exp(), upper.exp(), winner, valid)
            for key, value in zip(rows, values):
                rows[key].append(value)
        return {
            key: torch.cat(chunks, dim=2).reshape(batch, views, height, width)
            for key, chunks in rows.items()
        }
