"""仅由 RGB 构造多尺度边缘证据；图像边缘不等同于三维深度边界。"""

import math

import torch
import torch.nn.functional as F

from .geometry import sample_maps, spatial_gradients


def _smooth(rgb, sigma):
    """可分离高斯平滑，返回图像及实际使用的核半径。"""
    if sigma == 0:
        return rgb, 0
    radius = math.ceil(3 * sigma)
    position = torch.arange(-radius, radius + 1, device=rgb.device, dtype=rgb.dtype)
    kernel = torch.exp(-0.5 * (position / sigma).square())
    kernel = kernel / kernel.sum()
    channels = rgb.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    rgb = F.conv2d(
        F.pad(rgb, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels
    )
    rgb = F.conv2d(F.pad(rgb, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)
    return rgb, radius


@torch.no_grad()
def build_edge_pyramid(rgb, view_mask=None, image_mask=None, strides=(32, 16, 8, 4)):
    """生成原图坐标下的边缘图，不改变像素中心，也不读取监督标签。

    rgb 为 [N,3,H,W]、范围通常为 [0,1]。输出字典按编码器 stride 索引；
    所有输出仍是 H×W，供原图坐标直接采样，不能再传特征 stride。
    strength 为 [N,1,H,W] 的 [0,1] 强度；direction 为 [N,2,H,W] 的
    边缘法向；tensor 为 [N,3,H,W] 的局部结构张量 (xx,xy,yy)，迹为 1
    或 0；valid 为 [N,1,H,W] 的完整有效邻域掩码。

    粗层先高斯平滑以抑制细碎纹理，细层保留窄边缘。使用 RGB 结构张量，
    避免亮度相近但颜色不同的物体边界丢失。单一边缘只约束法向位移，
    不能独立确定沿边缘的匹配；后续仍须进行多视图几何验证。
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must be [N,3,H,W]")
    if any(stride < 1 for stride in strides):
        raise ValueError("strides must be positive")
    n, _, height, width = rgb.shape
    valid = torch.ones(n, 1, height, width, device=rgb.device, dtype=torch.bool)
    if view_mask is not None:
        valid = valid & view_mask[:, None, None, None].bool()
    if image_mask is not None:
        valid = valid & image_mask[:, None].bool()
    valid = valid & torch.isfinite(rgb).all(1, keepdim=True)
    levels = {}
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        safe_rgb = torch.where(valid, rgb.float(), 0.0)
        for stride in strides:
            sigma = 0.0 if stride <= 4 else stride / 16.0
            smoothed, radius = _smooth(safe_rgb, sigma)
            gx, gy = spatial_gradients(smoothed)
            xx = gx.square().mean(1, keepdim=True)
            xy = (gx * gy).mean(1, keepdim=True)
            yy = gy.square().mean(1, keepdim=True)
            tensor = torch.cat((xx, xy, yy), dim=1)
            tensor = F.avg_pool2d(tensor, 3, stride=1, padding=1)
            # 高斯核 + 中心差分 + 张量的 3×3 汇聚均必须处于有效图像内。
            support = radius + 2
            missing = F.pad((~valid).float(), (support, support, 0, 0), value=1.0)
            missing = F.max_pool2d(missing, (1, 2 * support + 1), stride=1)
            missing = F.pad(missing, (0, 0, support, support), value=1.0)
            neighborhood_valid = F.max_pool2d(missing, (2 * support + 1, 1), stride=1) == 0
            trace = tensor[:, :1] + tensor[:, 2:]
            has_edge = neighborhood_valid & (trace > 1e-12)
            strength = (2 * max(1.0, sigma) * trace.sqrt()).clamp(0, 1) * has_edge
            tensor = tensor / trace.clamp_min(1e-12) * has_edge
            angle = 0.5 * torch.atan2(2 * tensor[:, 1:2], tensor[:, :1] - tensor[:, 2:])
            direction = torch.cat((angle.cos(), angle.sin()), dim=1) * has_edge
            levels[stride] = dict(
                strength=strength,
                direction=direction,
                tensor=tensor,
                valid=neighborhood_valid,
            )
    return levels


def edge_motion_observability(level, uv, motion):
    """测量原图中运动方向是否跨越边缘，返回 [N,Q] 的 [0,1] 权重。

    uv 和 motion 都为 [N,Q,2]；motion 可直接使用 J·dX/dZ。权重是
    局部结构张量在该方向的归一化能量平方根，不包含边缘强度。沿单边缘
    运动、零运动、无纹理或无效邻域返回零。使用者应把它用于边缘证据，
    不应据此屏蔽其他有效纹理，更不能据此确认三维表面。
    """
    with torch.autocast(device_type=uv.device.type, enabled=False):
        tensor = sample_maps(level["tensor"], uv)
        valid = sample_maps(level["valid"].float(), uv)[..., 0] > 0.999
        motion = motion.float()
        finite = torch.isfinite(motion).all(-1) & torch.isfinite(uv).all(-1)
        motion = torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
        motion = motion / motion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x, y = motion.unbind(-1)
        energy = (
            tensor[..., 0] * x.square() + 2 * tensor[..., 1] * x * y + tensor[..., 2] * y.square()
        )
        # eps 后再减 eps 的平方根，避免 sqrt(0) 在反向传播时生成无限梯度。
        result = ((energy.clamp(0, 1) + 1e-12).sqrt() - 1e-6).clamp(0, 1)
        return result * valid * finite
