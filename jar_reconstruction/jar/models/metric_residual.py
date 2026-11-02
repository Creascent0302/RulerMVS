"""共享强初值的米制几何残差：不重新编码图像，不重新生成整张深度。

证据采样位于官方匹配网格 1.5 + 4*j。标定导数只控制局部搜索跨度，
九个候选仍精确投影。完整描述子留给小网络判断，原始全图深度始终
保留为恒等旁路；米制限幅避免逆深度残差在远距离被平方放大。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

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


@torch.no_grad()
def prepare_evidence(inputs, depth, matching, reference_index=2, settings=None):
    """从 RGB、标定和冻结初值构造可缓存的张量包，完全不读取 targets。

    depth 为 [B,1,H,W]，matching 为 [B,N,16,Hm,Wm]。锚点始终是
    原图 stride 4 网格；匹配分辨率可提高，按实际尺寸推导采样倍率。
    packet 的 source_features 保留每源、每候选完整的归一化 16 维向量。
    """
    settings = settings or {}
    batch, views = inputs["rgb"].shape[:2]
    height, width = depth.shape[-2:]
    if depth.shape[:2] != (batch, 1) or tuple(inputs["rgb"].shape[-2:]) != (height, width):
        raise ValueError("初值须为与输入图像同尺寸的 [B,1,H,W]")
    if height % 4 or width % 4:
        raise ValueError("几何残差当前需要图像高宽为 4 的倍数")
    if matching.shape[:3] != (batch, views, 16) or not 0 <= reference_index < views:
        raise ValueError("matching 须为 [B,N,16,Hm,Wm]，参考相机索引须有效")
    near = float(settings.get("near", 0.1))
    far = float(settings.get("far", 200.0))
    minimum_step = float(settings.get("min_step_m", 0.02))
    maximum_step = float(settings.get("max_step_m", 2.0))
    minimum_sources = int(settings.get("min_sources", 2))
    if not 0 < near < far or not 0 < minimum_step <= maximum_step or minimum_sources < 1:
        raise ValueError("深度范围、搜索跨度和 min_sources 设置无效")
    feature_h, feature_w = height // 4, width // 4
    uv = pixel_grid(feature_h, feature_w, device=depth.device).reshape(-1, 2) * 4 + 1.5
    count = len(uv)
    offsets = torch.linspace(-1, 1, 9, device=depth.device)
    image_scale = depth.new_tensor(
        [4 * matching.shape[-1] / width, 4 * matching.shape[-2] / height]
    )
    view_mask = inputs.get(
        "view_mask", torch.ones(batch, views, device=depth.device, dtype=torch.bool)
    )
    image_mask = inputs.get(
        "image_mask",
        torch.ones(batch, views, height, width, device=depth.device, dtype=torch.bool),
    )
    texture = inputs.get("_texture")
    if texture is None:
        rgb = torch.nan_to_num(inputs["rgb"].float()).flatten(0, 1)
        texture = image_texture(rgb).reshape(batch, views, 1, height, width)
    texture_threshold = float(settings.get("texture_threshold", 0.002))
    minimum_gain = float(settings.get("min_gain", 0.002))
    peak_tolerance = float(settings.get("peak_tolerance", 1.0))
    packets = []
    for sample in range(batch):
        base = depth[sample].float()
        valid_depth = torch.isfinite(base) & (base > 0)
        anchor = sample_maps(torch.nan_to_num(base)[None], uv[None])[0, :, 0]
        anchor_valid = sample_maps(valid_depth.float()[None], uv[None])[0, :, 0] > 0.999
        reference_mask = image_mask[sample, reference_index][None, None].float()
        anchor_valid &= sample_maps(reference_mask, uv[None])[0, :, 0] > 0.999
        anchor_valid &= view_mask[sample, reference_index] & (anchor >= near) & (anchor <= far)
        safe_anchor = torch.where(anchor_valid, anchor, torch.full_like(anchor, near))
        K = inputs["K"][sample].float()
        T = inputs["T_cam_from_rig"][sample].float()
        points, rays = unproject(uv, safe_anchor, K[reference_index], T[reference_index])
        center_uv, center_z, _, jacobian = project(points, K, T, (height, width))
        derivative = (jacobian * rays[None, :, None]).sum(-1).norm(dim=-1)
        centers = camera_centers(T)
        baseline = (centers - centers[reference_index]).norm(dim=-1)
        geometric = (baseline[:, None] > 1e-4) & (derivative > 1e-4) & (center_z > 0)
        geometric &= view_mask[sample, :, None] & anchor_valid[None]
        geometric[reference_index] = False
        average_derivative = (derivative * geometric).sum(0) / geometric.sum(0).clamp_min(1)
        step = float(settings.get("step_pixels", 1.0)) * 4 / average_derivative.clamp_min(1e-6)
        step = step.clamp(minimum_step, maximum_step)
        displacement_m = step[:, None] * offsets
        candidates = safe_anchor[:, None] + displacement_m
        positive_candidates = candidates.clamp_min(1e-4)
        candidate_points = points[:, None] + rays[:, None] * displacement_m[..., None]
        projected, source_z, valid, _ = project(
            candidate_points.reshape(-1, 3),
            K,
            T,
            (height, width),
            view_mask[sample],
            image_mask[sample],
        )
        valid = valid.reshape(views, count, 9) & geometric[:, :, None]
        valid &= ((candidates >= near) & (candidates <= far))[None]
        reference_slice = slice(reference_index, reference_index + 1)
        ref_texture = sample_maps(texture[sample, reference_slice], uv[None])[0, :, 0]
        source_texture = sample_maps(texture[sample], projected).reshape(views, count, 9)
        valid &= ref_texture[None, :, None] > texture_threshold
        valid &= source_texture > texture_threshold
        reference = sample_pretrained_matching(
            matching[sample, reference_slice].float(), uv[None], image_scale
        )[0]
        source = sample_pretrained_matching(
            matching[sample].float(), projected, image_scale
        ).reshape(views, count, 9, 16)
        reference_ok = torch.isfinite(reference).all(-1) & (reference.norm(dim=-1) > 1e-6)
        source_ok = torch.isfinite(source).all(-1) & (source.norm(dim=-1) > 1e-6)
        valid &= reference_ok[None, :, None] & source_ok
        reference = F.normalize(torch.nan_to_num(reference), dim=-1, eps=1e-6)
        source = F.normalize(torch.nan_to_num(source), dim=-1, eps=1e-6)
        source = torch.where(valid[..., None], source, 0)
        cosine = (source * reference[None, :, None]).sum(-1)
        source_count = valid[:, :, 4].sum(0)
        legal_gate = anchor_valid & (source_count >= minimum_sources)
        peak = cosine.masked_fill(~valid, -1e4).argmax(-1)
        peak_gain = cosine.masked_fill(~valid, -1e4).amax(-1) - cosine[:, :, 4]
        usable = valid[:, :, 4] & (peak_gain >= minimum_gain)
        index_distance = (peak[:, :, None] - torch.arange(9, device=depth.device)).abs()
        compatible = index_distance <= peak_tolerance
        compatible &= usable[:, :, None] & valid
        compatible_count = compatible.sum(0)
        gain = cosine - cosine[:, :, 4:5]
        compatible_gain = (gain * compatible).sum(0) / compatible_count.clamp_min(1)
        accepted = (compatible_count >= minimum_sources) & (compatible_gain >= minimum_gain)
        winning = compatible_gain.masked_fill(~accepted, -1e4).argmax(-1)
        evidence_gain = compatible_gain.gather(1, winning[:, None])[:, 0]
        strict_gate = legal_gate & accepted.any(-1)
        mean_gain = (gain * valid).sum(0) / valid.sum(0).clamp_min(1)
        fallback = mean_gain.masked_fill(valid.sum(0) < minimum_sources, -1e4).argmax(-1)
        winning = torch.where(strict_gate, winning, fallback)
        proposal = displacement_m.gather(1, winning[:, None])[:, 0]
        proposal = torch.where(legal_gate, proposal, 0)
        evidence_gain = torch.where(strict_gate, evidence_gain, 0)
        source_direction = F.normalize(candidate_points[None] - centers[:, None, None], dim=-1)
        pixel_displacement = (projected.reshape(views, count, 9, 2) - center_uv[:, :, None]) / 4
        metadata = torch.stack((
            offsets.expand(views, count, -1),
            torch.log1p(derivative / 4)[:, :, None].expand(-1, -1, 9),
            (baseline[:, None] / safe_anchor[None])[:, :, None].expand(-1, -1, 9),
            (source_z.reshape(views, count, 9).clamp_min(1e-4) / positive_candidates[None]).log(),
            (source_direction * F.normalize(rays, dim=-1)[None, :, None]).sum(-1),
            pixel_displacement[..., 0],
            pixel_displacement[..., 1],
        ), -1)
        metadata = torch.where(valid[..., None], metadata.nan_to_num().clamp(-20, 20), 0)
        def spatial(value):
            return value.reshape(*value.shape[:-1], feature_h, feature_w)

        packets.append({
            "source_features": source.permute(0, 2, 3, 1).reshape(
                views, 9, 16, feature_h, feature_w
            ),
            "reference_features": reference.T.reshape(16, feature_h, feature_w),
            "metadata": metadata.permute(0, 2, 3, 1).reshape(views, 9, 7, feature_h, feature_w),
            "candidate_valid": valid.permute(0, 2, 1).reshape(views, 9, feature_h, feature_w),
            "candidate_offsets": displacement_m.T.reshape(9, feature_h, feature_w),
            "anchor_depth": spatial(anchor[None]),
            "anchor_valid": spatial(anchor_valid[None]),
            "legal_gate": spatial(legal_gate[None]),
            "strict_gate": spatial(strict_gate[None]),
            "geometric_gate": spatial(strict_gate[None]),
            "source_count": spatial(source_count[None]),
            "proposal_m": spatial(proposal[None]),
            "evidence_gain": spatial(evidence_gain[None]),
            "step_m": spatial(step[None]),
        })
    return {key: torch.stack([item[key] for item in packets]) for key in packets[0]}


class _LocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, value):
        return value + self.body(value)


class MetricResidualHead(nn.Module):
    """逐源完整匹配向量→九候选局部推理→保留原图的有界米制修正。"""

    def __init__(self, context_channels=128, settings=None):
        super().__init__()
        settings = settings or {}
        self.settings = dict(settings)
        hidden = int(settings.get("hidden_dim", 32))
        descriptor_hidden = int(settings.get("descriptor_dim", 16))
        self.maximum_delta = float(settings.get("max_delta_m", 1.0))
        self.maximum_relative_delta = float(settings.get("max_relative_delta", 0.1))
        self.geometry_gate = settings.get("geometry_gate", "strict")
        if min(hidden, descriptor_hidden, self.maximum_delta, self.maximum_relative_delta) <= 0:
            raise ValueError("残差通道数与米制限幅须为正")
        if self.maximum_relative_delta >= 1:
            raise ValueError("max_relative_delta 须小于 1，以保持正深度")
        if self.geometry_gate not in ("strict", "none"):
            raise ValueError("geometry_gate 须为 strict 或 none")
        self.source_encoder = nn.Sequential(nn.Conv3d(71, descriptor_hidden, 1), nn.SiLU())
        self.source_score = nn.Conv3d(descriptor_hidden, 2, 1)
        self.context = nn.Sequential(nn.Conv2d(context_channels, hidden, 1), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Conv2d(9 * (descriptor_hidden + 3) + 5 + hidden, hidden, 1),
            nn.SiLU(),
            *[_LocalBlock(hidden) for _ in range(int(settings.get("blocks", 2)))],
        )
        self.output = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _upsample(self, residual, packet, base):
        """按匹配网格插值修正，原深度只作为边界引导，不被低分辨率重建。"""
        if not self.settings.get("edge_aware", True):
            return F.interpolate(residual, base.shape[-2:], mode="bilinear", align_corners=False)
        batch, _, low_h, low_w = residual.shape
        height, width = base.shape[-2:]
        uv = (pixel_grid(height, width, device=base.device).reshape(-1, 2) - 1.5) / 4
        lower = uv.floor().long()
        fraction = uv - lower
        reference = base.detach().float().flatten(2)
        total = torch.zeros_like(reference)
        weight_sum = torch.zeros_like(reference)
        threshold = float(self.settings.get("edge_threshold", 0.05))
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x = (lower[:, 0] + dx).clamp(0, low_w - 1)
            y = (lower[:, 1] + dy).clamp(0, low_h - 1)
            index = (y * low_w + x)[None, None].expand(batch, 1, -1)
            anchor = packet["anchor_depth"].float().flatten(2).gather(2, index)
            valid = packet["anchor_valid"].flatten(2).gather(2, index)
            distance = (reference.clamp_min(1e-4).log() - anchor.clamp_min(1e-4).log()).abs()
            edge_weight = torch.exp(-(distance / max(threshold, 1e-6)).square())
            weight_x = fraction[:, 0] if dx else 1 - fraction[:, 0]
            weight_y = fraction[:, 1] if dy else 1 - fraction[:, 1]
            weight = (weight_x * weight_y)[None, None] * edge_weight * valid
            value = residual.float().flatten(2).gather(2, index)
            total = total + weight * value
            weight_sum = weight_sum + weight
        return (total / weight_sum.clamp_min(1e-8)).reshape(batch, 1, height, width)

    def forward(self, packet, decoder_features, full_depth):
        source = packet["source_features"].float()
        batch, views, candidates, _, height, width = source.shape
        if (
            candidates != 9
            or decoder_features.shape[0] != batch
            or tuple(decoder_features.shape[-2:]) != (height, width)
        ):
            raise ValueError("残差头需要九候选和对齐的 stride 4 decoder_features")
        reference = packet["reference_features"].float()[:, None, None].expand_as(source)
        descriptor = torch.cat(
            (source, reference, source * reference, (source - reference).abs(), packet["metadata"].float()),
            dim=3,
        )
        descriptor = descriptor.permute(0, 1, 3, 2, 4, 5)
        encoded = self.source_encoder(descriptor.reshape(batch * views, 71, 9, height, width))
        prediction = self.source_score(encoded).reshape(batch, views, 2, 9, height, width)
        valid = packet["candidate_valid"].bool()
        reliability = safe_masked_softmax(prediction[:, :, 1].float(), valid, dim=1)
        encoded = encoded.reshape(batch, views, -1, 9, height, width)
        aggregate = (encoded * reliability[:, :, None]).sum(1).flatten(1, 2)
        cosine = (source * reference).sum(3)
        score = ((12 * cosine + prediction[:, :, 0].float().tanh()) * reliability).sum(1)
        support = valid.sum(1).float() / max(views - 1, 1)
        offsets = packet["candidate_offsets"].float() / self.maximum_delta
        statistics = torch.cat((
            packet["anchor_depth"].float().clamp_min(1e-4).log() / math.log(200),
            packet["step_m"].float() / self.maximum_delta,
            packet["proposal_m"].float() / self.maximum_delta,
            packet["evidence_gain"].float(),
            packet["source_count"].float() / max(views - 1, 1),
        ), dim=1)
        context = self.context(decoder_features.float())
        fused = self.fusion(
            torch.cat((aggregate, score, support, offsets, statistics, context), dim=1)
        )
        prediction = self.output(fused).float()
        gate_key = "strict_gate" if self.geometry_gate == "strict" else "legal_gate"
        gate = prediction[:, 1:2].sigmoid() * packet[gate_key]
        relative_limit = packet["anchor_depth"].float() * self.maximum_relative_delta
        limit = torch.minimum(torch.full_like(gate, self.maximum_delta), relative_limit)
        limit = torch.minimum(limit.clamp_min(0), packet["step_m"].float())
        low_delta = limit * prediction[:, :1].tanh() * gate
        delta = self._upsample(low_delta, packet, full_depth)
        full_limit = (full_depth.detach().float() * self.maximum_relative_delta).clamp(
            0, self.maximum_delta
        )
        delta = delta.clamp(-full_limit, full_limit)
        valid = torch.isfinite(full_depth) & (full_depth > 0)
        delta = torch.where(valid, delta, 0)
        # 加零保持原正深度逐值不变；不能用 where(delta == 0, base, ...)，
        # 否则零初始化时会把残差分支的首步梯度也切断。
        depth = full_depth + delta.to(full_depth.dtype)
        return {
            "depth": depth,
            "delta_m": delta,
            "low_delta_m": low_delta,
            "gate": gate,
            "source_support": packet["source_count"],
            "evidence_gain": packet["evidence_gain"],
            "proposal_m": packet["proposal_m"],
        }
