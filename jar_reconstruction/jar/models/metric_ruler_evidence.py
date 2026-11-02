"""从真实描述子与已知标定提取公制纹理证据，不读取任何标签。"""

import torch

from jar.geometry import pixel_grid, project, unproject
from jar.pretrained_matching import sample_pretrained_matching
from .metric_ruler_geometry import (
    feature_spatial_gradients,
    metric_jet_statistics,
    pullback_feature_jets,
)


RULER_FEATURE_NAMES = (
    "rank_defect", "usable_rank_penalty", "log_jet_energy",
    "log_reference_texture", "log_source_texture", "log_mean_parallax",
    "log_max_parallax", "informative_source_fraction",
)


def extract_ruler_features(record, ruler_scale=1.0):
    """返回 [1,8,D,H,W] 与逐视图诊断量，输入为独立 inputs 或缓存 record。

    当前帧缓存 batch=1、参考相机由 reference_index 明确指定。所有候选
    都精确投影；没有使用 GT 生成视图权重、候选位置或有效性。
    rank 相容只是必要条件，不能以这些特征增加已验证表面支持数。
    """
    inputs = record.get("inputs", record)
    descriptors = inputs["matching_features"].float()
    views, channels, fh, fw = descriptors.shape
    _, _, depths, height, width = inputs["source_cosine"].shape
    stride = inputs["stride"]
    reference = inputs["reference_index"]
    scale = inputs["matching_scale"].reshape(-1, 2)[0].to(descriptors)
    uvref = pixel_grid(height, width, device=descriptors.device, stride=stride).reshape(-1, 2)
    z = inputs["inverse_depths"].float().reciprocal()[:, None].expand(depths, len(uvref))
    intrinsic = inputs["K"][0].float()
    transform = inputs["T_cam_from_rig"][0].float()
    points, rays = unproject(uvref[None].expand(depths, -1, -1), z, intrinsic[reference], transform[reference])
    uv, _, projected_valid, jacobian = project(
        points.reshape(-1, 3), intrinsic, transform, inputs["image_size"],
        inputs["view_mask"][0], inputs["image_mask"][0],
    )
    gradients = feature_spatial_gradients(descriptors, image_scale=scale, feature_stride=4)
    sampled = sample_pretrained_matching(gradients.flatten(1, 2), uv, scale)
    sampled = sampled.reshape(views, depths * height * width, channels, 2)
    jets = pullback_feature_jets(sampled, jacobian)
    parallax = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1)
    feature_uv = (uv + 0.5) * (scale / 4) - 0.5
    gradient_valid = ((feature_uv >= 1) & (feature_uv <= feature_uv.new_tensor([fw - 2, fh - 2]))).all(-1)
    reference_valid = projected_valid[reference] & gradient_valid[reference]
    valid = inputs["source_valid"][0].reshape(views, -1).bool()
    valid = valid & projected_valid & gradient_valid & reference_valid[None]
    valid = valid & torch.isfinite(jets).all((-2, -1)) & torch.isfinite(jets[reference]).all((-2, -1))[None]
    valid[reference] = False
    stats = metric_jet_statistics(jets[reference], jets, valid, parallax, ruler_scale=ruler_scale)
    jets = torch.nan_to_num(jets, nan=0.0, posinf=0.0, neginf=0.0)
    ref_energy = jets[reference].square().mean((-2, -1)) * ruler_scale**2
    source_energy = jets.square().mean((-2, -1)) * ruler_scale**2
    count = valid.sum(0).clamp_min(1)
    mean_energy = (source_energy * valid).sum(0) / count
    mean_parallax = (stats["ruler_parallax"] * valid).sum(0) / count
    max_parallax = stats["ruler_parallax"].masked_fill(~valid, 0).amax(0)
    defect = stats["rank_defect"]
    features = torch.stack(
        (
            defect, defect * stats["usable"], stats["jet_energy"].log1p(),
            ref_energy.log1p(), mean_energy.log1p(), mean_parallax.log1p(),
            max_parallax.log1p(), stats["support"].float() / max(views - 1, 1),
        ),
        0,
    ).reshape(1, len(RULER_FEATURE_NAMES), depths, height, width)
    result = {name: stats[name].reshape(1, depths, height, width)
              for name in ("rank_defect", "jet_energy", "usable", "support")}
    result.update(
        source_valid=valid.reshape(1, views, depths, height, width),
        source_jet_energy=source_energy.reshape(1, views, depths, height, width),
        ruler_parallax=stats["ruler_parallax"].reshape(1, views, depths, height, width),
        source_cosine=inputs["source_cosine"],
    )
    return features, result
