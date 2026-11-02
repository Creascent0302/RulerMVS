"""公制纹理导数的独立几何工具；低秩相容不等于正确匹配。

像素特征梯度经投影导数拉回 rig 坐标，得到每米位移的特征变化。
在真实光滑表面的切平面上，不同视图的导数应一致。因此其差的
所有行应平行于同一个表面法向。该必要条件只能否决不相容候选：
线性纹理等错误匹配也可能满足它，必须另有中心匹配与可见性证据。
"""

import torch
import torch.nn.functional as F

from jar.geometry import spatial_gradients


def feature_spatial_gradients(
    features, image_scale=(1.0, 1.0), feature_stride=1.0, normalize=True, eps=1e-6
):
    """[N,C,H,W] → [N,C,2,H,W]，对 scene 原图像素求导。

    原生匹配特征的步长为 4；matching 图相对 scene 图放大 2 倍时，
    特征格点导数乘 2/4 才是对 scene 像素的导数。像素中心偏移不影响
    此导数，但后续读取梯度图仍须使用 sample_pretrained_matching。
    中心差分是有限足迹近似；图像边界使用复制填充，不提供额外证据。
    """
    if features.ndim != 4 or feature_stride <= 0:
        raise ValueError("features 须为 [N,C,H,W]，feature_stride 须为正数")
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    values = features.to(dtype)
    if normalize:
        values = F.normalize(values, dim=1, eps=eps)
    scale = torch.as_tensor(image_scale, device=values.device, dtype=dtype)
    if scale.shape != (2,) or not bool((scale > 0).all()):
        raise ValueError("image_scale 须为正的 (sx, sy)")
    gx, gy = spatial_gradients(values)
    return torch.stack((gx * scale[0], gy * scale[1]), dim=2) / feature_stride


def pullback_feature_jets(feature_gradients, projection_jacobians):
    """[...,C,2] @ [...,2,3] → [...,C,3]，单位为每 rig 米。

    采样位置必须由真实候选精确投影。这里的链式法则只计算局部导数，
    不用于近似大范围重投影，也不把投影合法性当作真实可见性。
    """
    dtype = torch.float64 if feature_gradients.dtype == torch.float64 else torch.float32
    return feature_gradients.to(dtype) @ projection_jacobians.to(dtype)


def metric_jet_statistics(
    reference_jets,
    source_jets,
    source_valid,
    ray_projection_derivatives,
    *,
    ruler_scale=1.0,
    weights=None,
    min_sources=2,
    min_parallax=1e-5,
    min_texture=1e-6,
    eps=1e-12,
    compute_normal=False,
):
    """比较公制导数的低秩相容性，不输出表面概率。

    reference_jets: [Q,C,3]；source_jets: [V,Q,C,3]。
    source_valid: [V,Q]，只表示外部已接受的观测，不能省略遮挡处理。
    ray_projection_derivatives: [V,Q,2]，源投影对参考光轴深度的导数。
    ruler_scale: 标尺长度，标量或 [Q]，单位米；将 Gram 矩阵无量纲化。

    rank_defect=1-trace(A²)/trace(A)²，A 为加权 ΔGᵀΔG。
    默认只算矩阵不变量，不逐候选执行特征分解；rank-1 时为零。
    usable 要求多源、非零纹理及视差；它依然不代表真实无遮挡。
    compute_normal=True 才求停止梯度的候选法向，仅 normal_valid 时可用。
    """
    if reference_jets.ndim != 3 or source_jets.ndim != 4:
        raise ValueError("jets 须为 [Q,C,3] 和 [V,Q,C,3]")
    if source_jets.shape[1:] != reference_jets.shape or reference_jets.shape[-1] != 3:
        raise ValueError("源和参考的候选、通道、坐标尺寸须一致")
    shape = source_jets.shape[:2]
    if source_valid.shape != shape or ray_projection_derivatives.shape != (*shape, 2):
        raise ValueError("source_valid 或 ray_projection_derivatives 尺寸错误")
    dtype = torch.float64 if reference_jets.dtype == torch.float64 else torch.float32
    ref = reference_jets.to(dtype)
    source = source_jets.to(dtype)
    ray = ray_projection_derivatives.to(dtype)
    scale = torch.as_tensor(ruler_scale, device=ref.device, dtype=dtype)
    if scale.ndim > 1 or (scale.ndim == 1 and scale.shape != ref.shape[:1]):
        raise ValueError("ruler_scale 须为标量或 [Q]")
    if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
        raise ValueError("ruler_scale 须为有限正数")
    valid = source_valid.bool()
    valid = valid & torch.isfinite(source).all(dim=(-2, -1))
    valid = valid & torch.isfinite(ref).all(dim=(-2, -1))[None]
    valid = valid & torch.isfinite(ray).all(dim=-1)
    source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    ref = torch.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    ray = torch.nan_to_num(ray, nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.ones_like(valid, dtype=dtype) if weights is None else weights.to(dtype)
    if weight.shape != shape:
        raise ValueError("weights 须为 [V,Q]")
    weight = torch.where(valid & torch.isfinite(weight), weight.clamp_min(0), 0.0)
    accepted = valid & (weight > 0)
    source_texture = source.square().mean(dim=(-2, -1)) * scale.square()
    ref_texture = ref.square().mean(dim=(-2, -1)) * scale.square()
    textured = (source_texture > min_texture**2) & (ref_texture[None] > min_texture**2)
    parallax = ray.square().sum(-1).sqrt() * scale
    informative = accepted & textured & (parallax > min_parallax)
    # 没有可测纹理/视差的视图不得污染其它视图的相容性矩阵。
    weight = weight * informative
    delta = (source - ref[None]) * scale[..., None, None]
    denominator = weight.sum(0).clamp_min(eps) * ref.shape[1]
    gram = torch.einsum("vqcj,vqck,vq->qjk", delta, delta, weight) / denominator[:, None, None]
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    squared_trace = trace.square()
    defect = (1 - gram.square().sum(dim=(-2, -1)) / squared_trace.clamp_min(eps**2)).clamp(0, 1)
    defect = torch.where(trace > eps, defect, 0.0)
    support = informative.sum(0)
    usable = support >= min_sources
    normal = torch.zeros_like(gram[:, 0])
    normal_valid = torch.zeros_like(usable)
    if compute_normal:
        # 退化重根下法向没有唯一梯度，明确停止法向梯度。
        with torch.no_grad():
            values, vectors = torch.linalg.eigh(gram.detach())
            gap = (values[:, -1] - values[:, -2]).clamp_min(0) / trace.detach().clamp_min(eps)
            normal_valid = usable & (trace > eps) & (gap > 1e-3)
            normal = torch.where(normal_valid[:, None], vectors[:, :, -1], 0.0)
    return {
        "rank_defect": defect,
        "gram": gram,
        "jet_energy": trace,
        "ruler_parallax": parallax,
        "support": support,
        "usable": usable,
        "normal": normal,
        "normal_valid": normal_valid,
    }
