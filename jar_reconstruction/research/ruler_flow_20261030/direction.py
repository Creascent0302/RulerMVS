"""固定 stride4 匹配场的方向证据；不改生产模型，不读取监督标签。

depths 与 prepared['T'] 使用同一域归一单位。局部状态 q 的定义为
log(depth) = log(center) + log_radius * q，因此链式导数必须乘 depth
和 log_radius。此分支不尝试近似原动态 LOD 路由的完整导数。
"""

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, project, sample_maps, unproject


CHANNELS = (
    "cosine", "signed_cosine_slope", "bounded_gn_step", "log_curvature",
    "residual_norm", "textured_support", "geometric_support",
    "direction_agreement", "log_projected_motion", "reliable_weight_mass",
)
DIRECTION_CHANNELS = len(CHANNELS)


def prepare_direction(prepared):
    """每次编码后调用一次；保留描述子/梯度到可训练编码器的计算图。"""
    features = prepared["pyramid"][4].float()
    features = torch.where(prepared["view_mask"][:, :, None, None, None], features, 0.)
    # 前向差与双线性单元配合，给出当前插值函数的精确分段导数。
    dx = F.pad(features[..., 1:] - features[..., :-1], (0, 1, 0, 0))
    dy = F.pad(features[..., 1:, :] - features[..., :-1, :], (0, 0, 0, 1))
    return {"features": features, "dx": dx, "dy": dy}


def _sample_feature_and_gradient(features, dx, dy, feature_uv):
    """feature_uv 已是 stride4 特征格坐标；border 区间外导数为零。"""
    height, width = features.shape[-2:]
    x, y = feature_uv.unbind(-1)
    safe_x, safe_y = x.clamp(0, width - 1), y.clamp(0, height - 1)
    uv = torch.stack((safe_x, safe_y), -1)
    value = sample_maps(features, uv, padding_mode="border")
    # dF/dx 在 x 方向是常量、y 方向线性；dF/dy 则反之。
    gx = sample_maps(dx, torch.stack((safe_x.floor(), safe_y), -1), padding_mode="border")
    gy = sample_maps(dy, torch.stack((safe_x, safe_y.floor()), -1), padding_mode="border")
    gx = gx * ((x >= 0) & (x < width - 1))[..., None]
    gy = gy * ((y >= 0) & (y < height - 1))[..., None]
    return value, gx, gy


def _directional_samples(prepared, cache, depths, log_radius):
    """返回逐源证据 [B,S,K,H,W]，用于聚合及有限差分验证。"""
    batch, count, height, width = depths.shape
    reference = prepared["reference"]
    sources = [i for i in range(prepared["K"].shape[1]) if i != reference]
    if not sources:
        return None
    grid = pixel_grid(height, width, device=depths.device, stride=4).reshape(-1, 2)
    radius = torch.broadcast_to(log_radius.float(), depths.shape)
    records = []
    with torch.autocast(device_type=depths.device.type, enabled=False):
        for sample in range(batch):
            K, T = prepared["K"][sample].float(), prepared["T"][sample].float()
            scale = prepared["scale"][sample].float()
            z = depths[sample].float()
            legal_depth = torch.isfinite(z) & (z > 0) & torch.isfinite(radius[sample]) & (radius[sample] > 0)
            z = torch.where(legal_depth, z, 1.)
            r = torch.where(legal_depth, radius[sample], 0.)
            points, rays = unproject(grid[None].expand(count, -1, -1), z.flatten(1), K[reference], T[reference])
            uv, _, valid, jacobian = project(
                points.reshape(-1, 3), K[sources], T[sources], prepared["image_size"],
                prepared["view_mask"][sample, sources], prepared["image_mask"][sample, sources],
            )
            # J*dX/dZ * dZ/dq，随后由原图像素转到 matching stride4 特征像素。
            motion = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1)
            motion = motion * (z * r).flatten()[None, :, None] * scale / 4
            feature_uv = ((uv + .5) * scale - .5) / 4
            value, gx, gy = _sample_feature_and_gradient(
                cache["features"][sample, sources], cache["dx"][sample, sources],
                cache["dy"][sample, sources], feature_uv,
            )
            anchor_uv = ((grid[None] + .5) * scale - .5) / 4
            anchor = sample_maps(cache["features"][sample, reference:reference + 1], anchor_uv, padding_mode="border")
            anchor_valid = anchor.norm(dim=-1)[0] > 1e-6
            anchor = F.normalize(anchor, dim=-1, eps=1e-6).expand(count, -1, -1).reshape(1, -1, value.shape[-1])
            ref_valid = sample_maps(prepared["image_mask"][sample, reference:reference + 1, None].float(), grid[None])[0, :, 0] > .999
            ref_valid = ref_valid & prepared["view_mask"][sample, reference] & anchor_valid
            motion_length = torch.linalg.vector_norm(motion, dim=-1)
            valid = valid & ref_valid.repeat(count)[None] & legal_depth.flatten()[None]
            valid = valid & (motion_length > 1e-5)
            value = torch.where(valid[..., None], value, 0.)
            raw_gradient = gx * motion[..., :1] + gy * motion[..., 1:]
            norm = value.norm(dim=-1, keepdim=True)
            descriptor = value / norm.clamp_min(1e-6)
            # d(v/||v||)/dq = (dv/dq - n*(n^T dv/dq))/||v||。
            tangent = raw_gradient - descriptor * (descriptor * raw_gradient).sum(-1, keepdim=True)
            tangent = tangent / norm.clamp_min(1e-6)
            tangent = torch.where((valid & (norm[..., 0] > 1e-6))[..., None], tangent, 0.)
            residual = descriptor - anchor
            cosine = (descriptor * anchor).sum(-1)
            slope = (tangent * anchor).sum(-1)
            gradient = (tangent * residual).sum(-1)
            curvature = tangent.square().sum(-1)
            textured = valid & (norm[..., 0] > 1e-6) & (curvature > 1e-8)
            values = {
                "cosine": cosine, "slope": slope, "gradient": gradient,
                "curvature": curvature, "residual_squared": residual.square().sum(-1),
                "motion_length": motion_length, "valid": valid, "textured": textured,
            }
            records.append({key: value.masked_fill(~valid, 0).reshape(len(sources), count, height, width)
                            for key, value in values.items()})
    return {key: torch.stack([record[key] for record in records]) for key in records[0]}


def directional_evidence(prepared, cache, depths, log_radius, source_weights=None):
    """返回 [B,10,K,H,W]；第2通道为局部 q 的有界 GN 建议步长。

    source_weights 可用 compact 源顺序或原 N 相机顺序，非负权重保留梯度。
    几何合法不等于无遮挡，textured 只确认局部方向可测，不认证匹配正确。
    全缺图、共心旋转和常量描述子均不给非零建议更新。
    """
    samples = _directional_samples(prepared, cache, depths, log_radius)
    if samples is None:
        finite = torch.where(torch.isfinite(depths), depths, 0.)
        return finite[:, None].expand(-1, len(CHANNELS), -1, -1, -1) * 0
    source_count = samples["valid"].shape[1]
    if source_weights is None:
        weights = torch.ones_like(samples["cosine"]) / source_count
    else:
        weights = source_weights.float()
        if weights.shape[1] == prepared["K"].shape[1]:
            reference = prepared["reference"]
            weights = torch.cat((weights[:, :reference], weights[:, reference + 1:]), 1)
        if weights.shape != samples["cosine"].shape:
            raise ValueError("source_weights 必须是 [B,S,K,H,W] 或 [B,N,K,H,W]")
        weights = torch.where(torch.isfinite(weights), weights.clamp_min(0), 0.)
    geometric_weights = weights * samples["valid"]
    reliable_weights = weights * samples["textured"]
    geometric_mass = geometric_weights.sum(1)
    reliable_mass = reliable_weights.sum(1)

    def average(key, reliable=True):
        weight, mass = (reliable_weights, reliable_mass) if reliable else (geometric_weights, geometric_mass)
        return (weight * samples[key]).sum(1) / mass.clamp_min(1e-8)

    gradient = average("gradient")
    curvature = average("curvature")
    slope = average("slope")
    absolute_slope = (reliable_weights * samples["slope"].abs()).sum(1) / reliable_mass.clamp_min(1e-8)
    # 描述子已单位化，q 亦归一化；固定小阻尼只稳定微弱方向，不声称最优校准。
    gn_step = (-gradient / (curvature + 1e-4)).clamp(-1, 1)
    channels = (
        average("cosine", reliable=False), slope.tanh(), gn_step,
        curvature.log1p(), (average("residual_squared", reliable=False).clamp_min(0) + 1e-12).sqrt() - 1e-6,
        samples["textured"].float().mean(1), samples["valid"].float().mean(1),
        slope.abs() / absolute_slope.clamp_min(1e-8),
        average("motion_length", reliable=False).log1p(), reliable_mass,
    )
    return torch.stack(channels, 1)
