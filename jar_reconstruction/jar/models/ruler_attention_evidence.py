"""标尺注意力的真实源特征：精确投影用于匹配，J只描述局部分辨能力。

与旧标尺的区别是保留各源16维描述子，不先压成一个余弦分数。
本模块不读取监督；几何缓存只有K/T与候选相关量，不保存图像内容。
"""

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, sample_maps
from .metric_ruler_geometry_evidence import projected_ruler
from .metric_ruler_network import depth_cell_edges


@torch.no_grad()
def projected_candidate_distance(inputs, cache=None, feature_sigma=1.0):
    """候选对的实际投影距离[B,HW,D,D]，不是J的一阶远距离外推。

    只由标定、相机/图像有效区域决定；不使用图像纹理证明表面。
    返回有界无量纲距离；无共同可采样源时关系代价为0、保持未知。
    """
    _, views, count, height, width = inputs["source_cosine"].shape
    uv, valid, _ = projected_ruler(inputs) if cache is None else cache.get(inputs)
    valid = valid & inputs["view_mask"][0, :, None].bool()
    valid = valid & (sample_maps(inputs["image_mask"][0, :, None].float(), uv)[..., 0] > 0.999)
    valid[int(inputs["reference_index"])] = False
    scale = inputs["matching_scale"].reshape(-1, 2)[0].to(uv) / 4
    positions = (uv * scale).reshape(views, count, height * width, 2).permute(0, 2, 1, 3)
    valid = valid.reshape(views, count, height * width).transpose(1, 2)
    total = uv.new_zeros(height * width, count, count)
    support = torch.zeros_like(total)
    for view in range(views):
        point = positions[view]
        # 差值相减不使用norm²+norm²-2dot，避免近邻大像素坐标的消减误差。
        delta = point[:, :, None] - point[:, None, :]
        distance = delta.square().sum(-1) / (2 * float(feature_sigma)**2)
        known = valid[view, :, :, None] & valid[view, :, None, :]
        total += distance.clamp_max(16) * known
        support += known
    return (total / support.clamp_min(1)).tanh()[None]


def attention_evidence(inputs, cache=None):
    """单场景入口；完整模型在batch/reference外层逐场景调用。

    source:[1,V,16,D,H,W]，reference:[1,16,H,W]，g:[1,V,2,D,H,W]。
    g单位为匹配特征像素/参考光轴米，包含匹配图缩放和stride4。
    """
    batch, views, count, height, width = inputs["source_cosine"].shape
    if batch != 1:
        raise ValueError("attention_evidence按单场景调用，外层处理batch")
    uv, geometric, derivative = projected_ruler(inputs) if cache is None else cache.get(inputs)
    reference = int(inputs["reference_index"])
    scale = inputs["matching_scale"].reshape(-1, 2)[0].to(uv)
    features = inputs["matching_features"].float()
    feature_uv = (uv + 0.5) * scale / 4 - 0.5
    source = sample_maps(features, feature_uv)
    source = source.transpose(1, 2).reshape(1, views, features.shape[1], count, height, width)
    grid = pixel_grid(height, width, device=uv.device, stride=inputs["stride"]).reshape(1, -1, 2)
    # 参考位置直接用输入格点，避免投影往返在u=0边界出现负浮点误差。
    reference_uv = (grid + 0.5) * scale / 4 - 0.5
    anchor = sample_maps(features[reference:reference + 1], reference_uv)
    anchor = anchor.transpose(1, 2).reshape(1, features.shape[1], height, width)
    valid = geometric & inputs["view_mask"][0, :, None].bool()
    reference_valid = sample_maps(inputs["image_mask"][0, reference:reference + 1, None].float(), grid)[0, :, 0] > 0.999
    reference_valid = reference_valid & inputs["view_mask"][0, reference].bool()
    valid = valid & (sample_maps(inputs["image_mask"][0, :, None].float(), uv)[..., 0] > 0.999)
    upper = uv.new_tensor([features.shape[-1] - 1, features.shape[-2] - 1])
    valid = valid & ((feature_uv >= 0) & (feature_uv <= upper)).all(-1)
    valid = valid.reshape(1, views, count, height, width) & inputs["source_valid"].bool()
    valid = valid & reference_valid.reshape(1, 1, 1, height, width)
    valid[:, reference] = False
    source = torch.where(valid[:, :, None], source, 0)
    gradient = derivative * scale / 4
    gradient = gradient.transpose(1, 2).reshape(1, views, 2, count, height, width)
    gradient = torch.where(valid[:, :, None], gradient, 0)
    depths = inputs["inverse_depths"].float().reciprocal()
    edges = depth_cell_edges(depths.flip(0)).flip(0)
    half_width = (edges[:-1] - edges[1:]) * 0.5
    return {
        "reference": F.normalize(anchor, dim=1),
        "source": F.normalize(source, dim=2),
        "source_valid": valid,
        "ray_derivative": gradient,
        "cell_half_width": half_width,
    }
