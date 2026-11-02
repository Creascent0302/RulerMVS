"""在完整候选曲线上传递局部平面支持，不提前合并成唯一深度。

每个候选分别假设正对相机、rig 水平、学习法线三个局部平面，并用同一个
平面解释四个邻居。学习法线仅是前向法线的小幅残差，不能覆盖任意斜面；
固定水平面补充地面假设。算法不把邻居分别最喜欢的平面拼成一个表面。

传递量只是有界、非负的上下文评分，不是新增的相机观测。几何无效、缺少
邻居、平坦匹配曲线均不产生支持；源视图计数及 unknown 判定仍由调用者保持。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import pixel_grid, unproject


def _neighbor(value, dy, dx):
    """返回同尺寸邻居张量；调用者用边界掩码排除 roll 的环绕部分。"""
    return torch.roll(value, shifts=(-dy, -dx), dims=(-2, -1))


def _sample_curves(curves, valid, inverse_depths, query):
    """[B,D,H,W] 曲线在 [B,L,D,H,W] 物理逆深度处线性插值。

    候选格可不均匀或乱序；不跨越无效候选间隙补证据。若恰好落在端点，
    只要求该端点有效，另一个零权重端点不影响采样。
    """
    order = inverse_depths.argsort()
    lattice = inverse_depths[order].float().contiguous()
    values = curves[:, order].unsqueeze(1).expand(-1, query.shape[1], -1, -1, -1)
    masks = valid[:, order].unsqueeze(1).expand_as(values)
    right = torch.searchsorted(lattice, query.contiguous()).clamp(1, len(lattice) - 1)
    left = right - 1
    fraction = (query - lattice[left]) / (lattice[right] - lattice[left]).clamp_min(1e-8)
    lower = values.gather(2, left)
    upper = values.gather(2, right)
    lower_valid = masks.gather(2, left)
    upper_valid = masks.gather(2, right)
    known = (lower_valid | (fraction >= 1 - 1e-6)) & (upper_valid | (fraction <= 1e-6))
    known = known & (query >= lattice[0]) & (query <= lattice[-1])
    sampled = torch.lerp(lower, upper, fraction.clamp(0, 1))
    return sampled, known


class CandidateSurfaceTransport(nn.Module):
    """候选平面运输→同平面邻域支持→含拒绝项的平面混合。

    同一相机中 X=C+r/rho。经过候选点、法线为 n 的平面与邻居射线的交点
    满足 rho_q=rho_p*(n·r_q)/(n·r_p)，这是精确平面关系，不是雅可比近似。
    same_depth 消融只将该比率设为 1，其余参数和聚合过程保持相同。
    """

    def __init__(self, cost_channels, feature_channels, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 16))
        self.transport = settings.get("transport", "plane")
        self.maximum = float(settings.get("max_residual", 2.0))
        self.temperature = float(settings.get("temperature", 1.0))
        self.affinity_temperature = float(settings.get("affinity_temperature", 0.25))
        initial_gate = float(settings.get("initial_gate", 0.25))
        if self.transport not in ("plane", "same_depth"):
            raise ValueError("candidate_transport.transport 须为 plane 或 same_depth")
        if min(hidden, self.maximum, self.temperature, self.affinity_temperature) <= 0:
            raise ValueError("候选运输的维度、幅度与温度须为正")
        if not 0 < initial_gate < 1:
            raise ValueError("候选运输 initial_gate 须位于 (0,1)")
        self.condition = nn.Sequential(nn.Conv3d(cost_channels, hidden, 1), nn.SiLU())
        self.normal = nn.Conv3d(hidden, 3, 1)
        self.plane_logits = nn.Conv3d(hidden, 4, 1)
        self.update_gate = nn.Conv2d(feature_channels, 1, 1)
        nn.init.normal_(self.normal.weight, std=1e-3)
        nn.init.zeros_(self.normal.bias)
        nn.init.normal_(self.plane_logits.weight, std=1e-3)
        nn.init.zeros_(self.plane_logits.bias)
        # 初始时三个平面共占一半权重，另一半保留给“不传递”。
        with torch.no_grad():
            self.plane_logits.bias[-1] = math.log(3)
        nn.init.normal_(self.update_gate.weight, std=1e-3)
        nn.init.constant_(self.update_gate.bias, math.log(initial_gate / (1 - initial_gate)))

    def forward(
        self,
        cost,
        scores,
        valid,
        inverse_depths,
        K_ref,
        T_ref,
        stride,
        image_features,
    ):
        """cost[B,C,D,H,W]、scores/valid[B,D,H,W] → 同形状评分增量。

        K/T 是该参考相机的固定标定。对 batch 中不同标定应分别调用。
        """
        _, depths, height, width = scores.shape
        if depths < 2 or len(inverse_depths) != depths:
            raise ValueError("候选运输需要至少两个、与 scores 对齐的逆深度候选")
        hidden = self.condition(cost)
        device = scores.device
        uv = pixel_grid(height, width, device=device, stride=stride)
        _, rays = unproject(
            uv,
            torch.ones((height, width), device=device),
            K_ref.detach().float(),
            T_ref.detach().float(),
        )
        rays = rays.permute(2, 0, 1)
        frontal = T_ref.detach().float()[2, :3].reshape(1, 3, 1, 1, 1)
        learned = F.normalize(frontal + 0.5 * self.normal(hidden).float().tanh(), dim=1)
        frontal = frontal.expand_as(learned)
        horizontal = torch.zeros_like(learned)
        horizontal[:, 2] = 1
        normals = torch.stack((frontal, horizontal, learned), dim=1)
        denominator = (normals * rays[None, None, :, None]).sum(2)
        safe_denominator = torch.where(denominator.abs() > 1e-4, denominator, 1)

        # 相对于本条曲线平均响应的支持：常数曲线严格不支持任何候选。
        # 这是评分对比，不把非均匀候选格上的离散 logit 称为连续概率密度。
        masked = scores.float().masked_fill(~valid, -1e4)
        maximum = masked.amax(1, keepdim=True)
        average = ((masked - maximum).exp() * valid).sum(1, keepdim=True)
        average = average / valid.sum(1, keepdim=True).clamp_min(1)
        baseline = maximum + average.clamp_min(1e-12).log()
        relative = torch.where(valid, (scores.float() - baseline).clamp(-8, 8), 0)
        features = F.normalize(image_features.float(), dim=1, eps=1e-6)
        rho = inverse_depths.float()[None, None, :, None, None]
        log_support = torch.zeros_like(denominator)
        weight_sum = torch.zeros_like(denominator)
        neighbor_count = torch.zeros_like(denominator)
        epsilon = 1e-4
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            neighbor_rays = _neighbor(rays, dy, dx)
            numerator = (normals * neighbor_rays[None, None, :, None]).sum(2)
            if self.transport == "plane":
                ratio = numerator / safe_denominator
                geometric = (denominator.abs() > 1e-4) & (ratio > 0)
            else:
                # 保留计算图，DDP 不会因消融而出现未触发的 normal 参数。
                ratio = 1 + numerator * 0
                geometric = torch.ones_like(denominator, dtype=torch.bool)
            query = rho * ratio
            neighbor_relative = _neighbor(relative, dy, dx)
            neighbor_valid = _neighbor(valid, dy, dx)
            sampled, known = _sample_curves(
                neighbor_relative, neighbor_valid, inverse_depths, query
            )
            border = torch.ones((height, width), dtype=torch.bool, device=device)
            if dy < 0:
                border[0] = False
            elif dy > 0:
                border[-1] = False
            if dx < 0:
                border[:, 0] = False
            elif dx > 0:
                border[:, -1] = False
            known = known & geometric & border & valid[:, None]
            difference = (1 - (features * _neighbor(features, dy, dx)).sum(1)).clamp_min(0)
            affinity = torch.exp(-difference / self.affinity_temperature)[:, None, None]
            weight = affinity * known
            evidence = (sampled / self.temperature).tanh().clamp_min(0)
            log_support = log_support + weight * (evidence + epsilon).log()
            weight_sum = weight_sum + weight
            neighbor_count = neighbor_count + known

        # 先在同一个平面下整合邻居，再混合平面；几何平均抑制少量孤立高分。
        # 绝对权重不足时降低幅度，不能把非常弱的相似性归一化成强支持。
        support = (log_support / weight_sum.clamp_min(1e-8)).exp() - epsilon
        support = support.clamp(0, 1) * (weight_sum / 2).clamp(0, 1)
        support = support * (neighbor_count >= 2)
        mixture = self.plane_logits(hidden).float().softmax(1)[:, :3]
        message = (support * mixture).sum(1)
        gate = self.update_gate(image_features).float().sigmoid()
        return self.maximum * gate * message * valid
