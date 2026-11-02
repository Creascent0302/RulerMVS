"""只从标定和候选构造公制面片取证，不使用真实深度或法线。

面片是待检验的局部表面假设。邻点全部精确投影；J只用于把期望图像
足迹换成米制采样步长，不用一阶近似替代投影，也不认证可见性。
"""

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, project, sample_maps, unproject


def sample_ruler_stencil(inputs, pixels, depth, mode="frontal", radius_pixels=2.0,
                         minimum_step_m=0.02, maximum_step_m=1.0):
    """返回 reference[Q,P,C]、source[V,Q,P,C]、known[V,Q,P]。

    mode=center为单点，image_patch为各视图固定二维十字，frontal/ground
    为参考光轴正视面/rig水平面。后两者都不读取场景真实法线。
    radius_pixels 的单位为匹配特征像素；公制步长越界时显式截断。
    """
    if mode not in ("center", "image_patch", "frontal", "ground", "frontal_matched", "ground_matched"):
        raise ValueError("未支持的标尺面片类型")
    reference = int(inputs["reference_index"])
    intrinsic = inputs["K"][0].float()
    transform = inputs["T_cam_from_rig"][0].float()
    features = inputs["matching_features"].float()
    scale = inputs["matching_scale"].reshape(-1, 2)[0].float()
    view_mask = inputs["view_mask"][0].bool()
    image_mask = inputs["image_mask"][0].bool()
    image_size = tuple(inputs["image_size"])
    points, _ = unproject(pixels, depth, intrinsic[reference], transform[reference])
    offsets = points.new_tensor([[0., 0.], [-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    center_uv, _, center_known, center_jacobian = project(
        points, intrinsic, transform, image_size, view_mask, image_mask
    )
    # 参考像素已知，不由浮点往返误差重新判图像左/上边界。
    center_uv[reference] = pixels
    center_known[reference] = ((pixels >= 0) &
        (pixels <= pixels.new_tensor([image_size[1] - 1, image_size[0] - 1]))).all(-1)
    center_known[reference] &= view_mask[reference]
    center_known[reference] &= sample_maps(image_mask[reference:reference + 1, None].float(), pixels[None])[0, :, 0] > 0.999
    metric_steps = points.new_zeros(len(points), 2)
    if mode in ("center", "image_patch"):
        offset = offsets[:1] if mode == "center" else offsets * float(radius_pixels)
        feature_uv = (center_uv[:, :, None] + .5) * scale / 4 - .5 + offset[None, None]
        known = center_known[:, :, None].expand(-1, -1, len(offset)).clone()
        uv = (feature_uv + .5) * 4 / scale - .5
    else:
        if mode in ("frontal", "frontal_matched"):
            basis = transform[reference, :2, :3].T
        else:
            basis = points.new_tensor([[1., 0.], [0., 1.], [0., 0.]])
        projected_basis = torch.einsum("qij,jk->qik", center_jacobian[reference], basis)
        projected_basis = projected_basis * (scale / 4)[None, :, None]
        sensitivity = projected_basis.norm(dim=1)
        metric_steps = (float(radius_pixels) / sensitivity.clamp_min(1e-6)).clamp(minimum_step_m, maximum_step_m)
        tangent = offsets[None] * metric_steps[:, None]
        stencil_points = points[:, None] + torch.einsum("qpk,jk->qpj", tangent, basis)
        uv, _, known, _ = project(stencil_points.reshape(-1, 3), intrinsic, transform,
                                  image_size, view_mask, image_mask)
        uv = uv.reshape(len(features), len(points), 5, 2)
        known = known.reshape(len(features), len(points), 5)
        uv[:, :, 0] = center_uv
        known[:, :, 0] = center_known
        feature_uv = (uv + .5) * scale / 4 - .5
        # 切向接近视线时，不让几乎重复的采样点伪装成额外图像证据。
        displacement = (feature_uv[reference] - feature_uv[reference, :, :1]).norm(dim=-1)
        noncollapsed = displacement >= 0.25
        noncollapsed[:, 0] = True
        known &= noncollapsed[None]
        if mode.endswith("_matched"):
            # 仅因果诊断：保留实际参考面片足迹，把其二维偏移直接平移到
            # 各源中心，从而删除源图的平面透视形变。主 physical/image 不变。
            reference_known = known[reference].clone()
            reference_offsets = feature_uv[reference] - feature_uv[reference, :, :1]
            feature_uv = ((center_uv[:, :, None] + .5) * scale / 4 - .5) + reference_offsets[None]
            uv = (feature_uv + .5) * 4 / scale - .5
            known = center_known[:, :, None] & reference_known[None]
    upper = feature_uv.new_tensor([features.shape[-1] - 1, features.shape[-2] - 1])
    known &= ((feature_uv >= 0) & (feature_uv <= upper)).all(-1)
    sampled_mask = sample_maps(image_mask[:, None].float(), uv.reshape(len(features), -1, 2))
    known &= sampled_mask[..., 0].reshape_as(known) > .999
    sampled = sample_maps(features, feature_uv.reshape(len(features), -1, 2))
    sampled = sampled.reshape(len(features), len(points), -1, features.shape[1])
    known = known & known[reference].clone()[None]
    known[reference] = False
    return {"reference": F.normalize(sampled[reference], dim=-1),
            "source": F.normalize(sampled, dim=-1), "known": known,
            "feature_coordinates": feature_uv, "metric_steps": metric_steps}


def stencil_matching_score(sample, center_known=None):
    """固定同刻度对应的均值与去中心模式分数；无自由重排匹配。

    source_patch至少含中心与两个其他有效点；图像单点只有中心即可。
    尚无学习式遮挡拒绝，结果仅用于决定是否值得训练面片注意力。
    """
    reference, source = sample["reference"], sample["source"]
    known = sample["known"].clone()
    if center_known is not None:
        known &= center_known[:, :, None]
    cosine = (source * reference[None]).sum(-1)
    source_count = known.sum(-1)
    required = 1 if reference.shape[1] == 1 else 3
    source_valid = (source_count >= required) & known[:, :, 0]
    per_source = (cosine * known).sum(-1) / source_count.clamp_min(1)
    support = source_valid.sum(0)
    score = (per_source * source_valid).sum(0) / support.clamp_min(1)
    contrast_ref = reference - reference[:, :1]
    contrast_src = source - source[:, :, :1]
    contrast_known = known & (contrast_ref.norm(dim=-1)[None] > 1e-3) & (contrast_src.norm(dim=-1) > 1e-3)
    contrast = (F.normalize(contrast_ref, dim=-1)[None] * F.normalize(contrast_src, dim=-1)).sum(-1)
    contrast_count = contrast_known.sum((0, 2))
    contrast_score = (contrast * contrast_known).sum((0, 2)) / contrast_count.clamp_min(1)
    return {"score": score, "known": support >= 2, "support": support,
            "contrast_score": contrast_score, "contrast_known": contrast_count >= 4,
            "patch_coverage": known.float().mean((0, 2))}


def attention_stencil_evidence(inputs, mode="physical", radius_pixels=2.0,
                               maximum_step_m=1.0):
    """对全部粗射线与深度假设取证，输出 [1,2,D,H,W]。

    physical 的两槽为 frontal/ground；image 的两槽都为 image_patch。
    对照保留相同的十个刻度槽和输出尺寸，当前也执行相同次数的采样。
    不按 GT、旧预测或置信度选择待处理点，推理与训练完全同一入口。
    """
    if mode not in ("physical", "image"):
        raise ValueError("mode 必须为 physical 或 image")
    batch, views, count, height, width = inputs["source_cosine"].shape
    if batch != 1:
        raise ValueError("本取证入口按单个场景调用，batch由外层处理")
    depths = inputs["inverse_depths"].float().reciprocal()
    pixels = pixel_grid(height, width, device=depths.device, stride=int(inputs["stride"]))
    pixels = pixels.reshape(-1, 2).repeat(count, 1)
    candidate_depth = depths[:, None].expand(count, height * width).reshape(-1)
    center_known = inputs["source_valid"][0].reshape(views, -1)
    modes = ("frontal", "ground") if mode == "physical" else ("image_patch", "image_patch")
    columns = {name: [] for name in ("scores", "valid", "support", "contrast", "contrast_valid", "coverage")}
    mapping = {"scores": "score", "valid": "known", "support": "support", "contrast": "contrast_score",
               "contrast_valid": "contrast_known", "coverage": "patch_coverage"}
    for surface_mode in modes:
        sample = sample_ruler_stencil(inputs, pixels, candidate_depth, mode=surface_mode,
                                      radius_pixels=radius_pixels, maximum_step_m=maximum_step_m)
        result = stencil_matching_score(sample, center_known)
        for destination, source in mapping.items():
            columns[destination].append(result[source].reshape(count, height, width))
    return {name: torch.stack(values, 0)[None] for name, values in columns.items()}
