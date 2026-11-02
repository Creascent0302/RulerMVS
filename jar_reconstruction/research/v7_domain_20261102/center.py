"""保留候选预算，在标定的投影坐标中计算族内中心。

这不是新增匹配证据。沿同一参考射线，源相机光轴深度为 a*z+b。
投影均值对应的参考深度为 E[z/(a*z+b)] / E[1/(a*z+b)]。
平行相机时退化为深度调和均值；纯旋转没有深度约束，明确回退。
"""

import math

import torch
from torch import nn

from jar.geometry import pixel_grid
from jar.pretraining.projective_ruler import compile_projective_ruler
from research.rulermvs_v6_20261101.handoff import masked_probability


def projective_group_center(depths, conditional, prepared, stride=4):
    """[B,G,3,H,W] -> 每族投影中心、可用性和无量纲深度敏感度。

    每个候选精确投影检查有效性；不以线性近似产生候选或改动标定。
    用相对于族均值的位移表达式避免小基线下减去两个大像素坐标。
    多源按投影导数平方加权；该权重仅表示几何条件，不认证可见性。
    """
    batch, groups, count, height, width = depths.shape
    reference = prepared["reference"]
    grid = pixel_grid(height, width, device=depths.device, stride=stride).reshape(-1, 2)
    arithmetic = (conditional * depths).sum(2)
    centers, supports, sensitivities = [], [], []
    for sample in range(batch):
        with torch.autocast(device_type=depths.device.type, enabled=False):
            z = depths[sample].float().flatten(-2)
            p = conditional[sample].float().flatten(-2)
            mean = arithmetic[sample].float().flatten(-2)
            ruler = compile_projective_ruler(
                grid, prepared["K"][sample, reference], prepared["T"][sample, reference],
                prepared["K"][sample], prepared["T"][sample])
            geometry = ruler.project(
                z.flatten(0, 1), prepared["image_size"],
                prepared["view_mask"][sample], prepared["image_mask"][sample])
            valid = geometry["valid"].reshape(-1, groups, count, height * width)
            valid[reference] = False
            optical = geometry["optical_depth"].reshape_as(valid).clamp_min(1e-4)
            weight = p[None] * valid
            mass = weight.sum(2)
            weight = weight / mass[:, :, None].clamp_min(1e-12)
            # (z-z_mean)/optical_z 是精确分式投影的标量因子。
            t = (weight * (z[None] - mean[None, :, None]) / optical).sum(2)
            a = ruler.camera_direction[..., 2][:, None]
            b = ruler.translation[:, 2, None, None]
            denominator = 1 - t * a
            legal = (mass > 0) & (denominator.abs() > 1e-6)
            safe_denominator = torch.where(legal, denominator, 1.)
            view_center = (mean[None] + t * b) / safe_denominator
            optical_mean = (a * mean[None] + b).clamp_min(1e-4)
            derivative = ruler.depth_numerator[:, None] / optical_mean[..., None].square()
            information = (derivative.square().sum(-1) * mean[None].square()).clamp_max(1e6)
            legal = legal & torch.isfinite(view_center) & (view_center > 0)
            information = information * legal
            total = information.sum(0)
            supported = total > 1e-8
            center = (torch.where(legal, view_center, mean[None]) * information).sum(0)
            center = center / total.clamp_min(1e-8)
            center = torch.where(supported, center, mean)
            center = center.clamp(z.amin(1), z.amax(1))
            centers.append(center.reshape(groups, height, width))
            supports.append(supported.reshape(groups, height, width))
            sensitivities.append(total.sqrt().reshape(groups, height, width))
    return torch.stack(centers), torch.stack(supports), torch.stack(sensitivities)


class ProjectiveCenter(nn.Module):
    """129参数，零初始化的可拆卸中心残差，不改变选族和图像采样。

    门可根据证据学习正/负修正，最终限制在本族实际候选范围内。
    六项输入：族内熵、相对跨度、两种中心差、投影敏感度、族质量、父质量。
    只训练残差及其上游深度，不通过质量统计制造梯度捷径。
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fine, prepared):
        depth = fine["candidate_depth"].float().unflatten(1, (-1, 3))
        conditional = masked_probability(fine["scores"].unflatten(1, (-1, 3)),
                                         fine["valid"].unflatten(1, (-1, 3)), 2)
        mean = fine["group_depth"]
        projected, supported, sensitivity = projective_group_center(depth, conditional, prepared)
        with torch.no_grad():
            entropy = -(conditional * conditional.clamp_min(1e-12).log()).sum(2) / math.log(3)
            spread = (depth.amax(2) - depth.amin(2)) / mean.clamp_min(1e-6)
            shift = (mean - projected) / mean.clamp_min(1e-6)
            parent = fine["parent_probability"]
            parent = parent / parent.sum(1, keepdim=True).clamp_min(1e-12)
            quality = torch.stack((entropy, spread.log1p(), shift,
                                   sensitivity.log1p(), fine["group_probability"], parent), 2)
            batch, groups, _, height, width = quality.shape
            quality = quality.reshape(batch * groups, 6, height, width).clamp(-8, 8)
        gate = self.net(quality.to(self.net[0].weight.dtype)).float().tanh()
        gate = gate.reshape_as(mean) * supported
        corrected = (mean + gate * (projected - mean)).clamp(depth.amin(2), depth.amax(2))
        return corrected, gate, projected
