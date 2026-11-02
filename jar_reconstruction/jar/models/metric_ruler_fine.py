"""只对保留候选重新取真实图像证据，不增加图像编码器或深度候选。"""

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import image_texture, pixel_grid, project, safe_masked_softmax, sample_maps, unproject
from jar.pretrained_matching import sample_pretrained_matching
from .spatial_surface_recovery import leave_one_view_out_vote


def coarse_proposal_prior(scores, valid, size, source_stride=16, target_stride=4):
    """粗分数只提出细射线深度假设，不生成细网格可见性。

    各粗节点内部先归一化，消除评分常数偏移；最近不同2×2节点按固定
    sigma=source_stride像素混合。只在相同depth channel相加，不平均深度。
    eligible 要求距离不超过 sqrt(2)*stride 的真实有效粗观测，不外借空节点。
    """
    if scores.ndim != 4 or valid.shape != scores.shape:
        raise ValueError("scores/valid 须为同尺寸 [B,D,H,W]")
    batch, _, height, width = scores.shape
    fine_height, fine_width = size
    y = torch.arange(fine_height, device=scores.device).float() * target_stride
    x = torch.arange(fine_width, device=scores.device).float() * target_stride
    dy = (torch.arange(height, device=scores.device).float()[:, None] * source_stride - y).abs()
    dx = (torch.arange(width, device=scores.device).float()[:, None] * source_stride - x).abs()
    yi = dy.argsort(dim=0, stable=True)[:min(2, height)]
    xi = dx.argsort(dim=0, stable=True)[:min(2, width)]
    probability = safe_masked_softmax(scores.float(), valid, dim=1)
    mass = scores.new_zeros(batch, scores.shape[1], fine_height, fine_width, dtype=torch.float32)
    total_weight = scores.new_zeros(batch, fine_height, fine_width, dtype=torch.float32)
    eligible = torch.zeros_like(mass, dtype=torch.bool)
    nodes = valid.any(1)
    for row in yi:
        for column in xi:
            distance_squared = (row.float()[:, None] * source_stride - y[:, None]).square()
            distance_squared = distance_squared + (column.float()[None] * source_stride - x[None]).square()
            close = distance_squared <= 2 * source_stride**2 + 1e-5
            weight = (-distance_squared / (2 * source_stride**2)).exp() * close
            weight = weight[None] * nodes[:, row[:, None], column[None]]
            mass = mass + probability[:, :, row[:, None], column[None]] * weight[:, None]
            total_weight = total_weight + weight
            eligible |= valid[:, :, row[:, None], column[None]] & close[None, None]
    prior = (mass / total_weight[:, None].clamp_min(1e-8)).clamp_min(1e-12).log()
    return {"scores": prior.masked_fill(~eligible, -1e4), "eligible": eligible}


def _bank_standardize(cosine, valid):
    """每台源相机只在当前 K 个候选中标准化，区别于原全192格点统计。"""
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (cosine * valid).sum(1, keepdim=True) / count
    variance = ((cosine - mean).square() * valid).sum(1, keepdim=True) / count
    return ((cosine - mean) / variance.clamp_min(0.05**2).sqrt()).clamp(-6, 6)


class ExactRulerProposalObserver(nn.Module):
    """无参数、无GT：9候选 × stride4射线的精确匹配取证。

    仅覆盖19维选择特征的12:18列；保留粗候选几何与score上下文，不把
    粗曲线插值称为精确观测。图内/纹理/视差合法仍不等于无遮挡。
    候选0保留参考有效性的回退；其它候选还须满足原候选有效和两源支持。
    """

    def __init__(self, settings=None):
        super().__init__()
        settings = settings or {}
        self.texture_threshold = float(settings.get("texture_threshold", 0.002))
        self.min_parallax = float(settings.get("min_parallax", 1e-5))
        self.stride = int(settings.get("stride", 4))

    def forward(self, proposal, entry, source_inputs, sample, reference):
        candidates = proposal["candidate_depth"].float()
        count, height, width = candidates.shape
        rgb = source_inputs["rgb"][sample].float()
        views, _, image_height, image_width = rgb.shape
        intrinsic = source_inputs["K"][sample].float()
        transform = source_inputs["T_cam_from_rig"][sample].float()
        view_mask = source_inputs.get("view_mask")
        view_mask = None if view_mask is None else view_mask[sample]
        image_mask = source_inputs.get("image_mask")
        image_mask = None if image_mask is None else image_mask[sample]
        pixels = pixel_grid(height, width, device=rgb.device, stride=self.stride)
        reference_uv = pixels.reshape(-1, 2)
        points, rays = unproject(pixels[None].expand(count, -1, -1, -1), candidates,
                                  intrinsic[reference], transform[reference])
        uv, _, valid, jacobian = project(points.reshape(-1, 3), intrinsic, transform,
                                         (image_height, image_width), view_mask, image_mask)
        parallax = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1).norm(dim=-1)
        texture = image_texture(rgb)
        local_texture = sample_maps(texture, uv)[..., 0]
        reference_texture = sample_maps(texture[reference:reference + 1], reference_uv[None])[0, :, 0]
        # 参考锚点本来就是已知像素，不经浮点rig往返投影重新决定边界。
        # 否则 u=0 可能因 -1e-5px 舍入被误杀，连保守回退都丢失。
        reference_anchor = (pixels[..., 0] < image_width) & (pixels[..., 1] < image_height)
        if view_mask is not None:
            reference_anchor &= view_mask[reference]
        if image_mask is not None:
            reference_anchor &= sample_maps(image_mask[reference:reference + 1, None].float(), reference_uv[None])[0, :, 0].reshape(height, width) > 0.999
        reference_valid = reference_anchor[None] & torch.isfinite(candidates) & (candidates > 0)
        known = valid & (parallax > self.min_parallax) & (local_texture > self.texture_threshold)
        known &= (reference_texture[None].expand(count, -1).reshape(1, -1) > self.texture_threshold)
        known &= reference_valid.reshape(1, -1)
        known[reference] = False
        features = source_inputs["_pretrained_matching_features"][sample]
        scale = source_inputs.get("_pretrained_matching_scale")
        scale = None if scale is None else scale[sample]
        reference_features = sample_pretrained_matching(features[reference:reference + 1], reference_uv[None], scale)[0]
        observed = sample_pretrained_matching(features, uv, scale).reshape(views, count, height * width, -1)
        normalized_reference = F.normalize(reference_features.float(), dim=-1, eps=1e-6)
        normalized_observed = F.normalize(observed.float(), dim=-1, eps=1e-6)
        cosine = (normalized_observed * normalized_reference[None, None]).sum(-1).reshape(views, count, height, width)
        known = known.reshape_as(cosine) & torch.isfinite(cosine)
        cosine = torch.where(known, cosine, 0.0)
        standardized = _bank_standardize(cosine, known)
        cosine = cosine.permute(1, 0, 2, 3)
        standardized = standardized.permute(1, 0, 2, 3)
        known = known.permute(1, 0, 2, 3)
        support = known.sum(1)
        denominator = support.clamp_min(1)
        mean = cosine.sum(1) / denominator
        zmean = (standardized * known).sum(1) / denominator
        low = standardized.masked_fill(~known, 6).amin(1)
        high = standardized.masked_fill(~known, -6).amax(1)
        low, high = torch.where(support > 0, low, 0.0), torch.where(support > 0, high, 0.0)
        variance = ((standardized - zmean[:, None]).square() * known).sum(1) / denominator
        spread = torch.where(variance > 0, variance.clamp_min(1e-12).sqrt(), 0.0)
        statistics = torch.stack((mean, zmean / 6, low / 6, high / 6, spread / 6, support.float() / views), 1)
        result = dict(proposal)
        result["features"] = torch.cat((proposal["features"][:, :12].float(), statistics,
                                         proposal["features"][:, 18:].float()), 1)
        key = "proposal_eligible" if proposal.get("observation_required", False) else "candidate_valid"
        candidate_valid = proposal[key].clone()
        candidate_valid[1:] &= (support[1:] >= 2) & reference_valid[1:]
        candidate_valid[0] = reference_valid[0]
        result["candidate_valid"] = candidate_valid
        if "view_consensus" in proposal:
            result["view_consensus"] = leave_one_view_out_vote(standardized, known, candidate_valid)
        result["exact_observation"] = {"source_cosine": cosine, "source_valid": known,
                                        "source_support": support, "parallax": parallax.reshape(views, count, height, width).permute(1, 0, 2, 3)}
        return result
