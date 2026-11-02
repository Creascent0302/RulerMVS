"""候选体的紧凑空间消息：替换旧逐射线 local，外部负责残差相加。

输入维度为 [batch, channel, candidate, height, width]。这些候选仍是
待判断的深度假设；空间邻接不等于表面存在，也不生成 free 标签。
"""

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import safe_masked_softmax


class CandidateSpatialConv(nn.Module):
    """同预算控制：从深度轴一维卷积换成一次深度可分离三维卷积。

每通道独立观察 3×3×3 候选邻域，再由 1×1×1 卷积交换通道信息。
不循环边界，不让无效位置的数值进入其他位置。零填充位置只是缺少
消息，不会被解释为零米深度或已观测空闲空间。
    """

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv3d(channels, channels, 1)

    @torch.no_grad()
    def initialize_from_depth(self, local):
        """与旧 3×1×1 起点等价；新空间邻接从零开始学习。"""
        self.depthwise.weight.zero_()
        self.depthwise.weight[:, :, :, 1:2, 1:2].copy_(local[0].weight)
        self.depthwise.bias.copy_(local[0].bias)
        self.pointwise.load_state_dict(local[2].state_dict())
        return self

    def forward(self, features, valid, **geometry):
        mask = valid[:, None].bool()
        value = torch.where(mask, features, 0)
        message = self.pointwise(self.activation(self.depthwise(value)))
        return torch.where(mask, message, 0)


class CandidateScaleSpace(CandidateSpatialConv):
    """在较低空间分辨率扩大匹配判断范围，始终保留全部公制候选。

    与将候选聚为少量全局节点不同，96个深度格不会在空间汇聚时合并。
    两级1/2空间尺度仅用深度可分离卷积，输出替代原local消息。
    """

    def __init__(self, channels):
        super().__init__(channels)
        self.scales = nn.ModuleList([CandidateSpatialConv(channels) for _ in range(2)])
        for block in self.scales:
            nn.init.zeros_(block.pointwise.weight)
            nn.init.zeros_(block.pointwise.bias)

    @staticmethod
    def resize(value, shape):
        # 深度轴不插值、不降采样；只改变图像平面。
        batch, channels, depth = value.shape[:3]
        planes = value.transpose(1, 2).reshape(batch * depth, channels, *value.shape[-2:])
        planes = F.interpolate(planes, size=shape, mode="bilinear", align_corners=False)
        return planes.reshape(batch, depth, channels, *shape).transpose(1, 2)

    def forward(self, features, valid, **geometry):
        values, masks = [features], [valid]
        for _ in self.scales:
            mask = masks[-1][:, None].float()
            count = F.avg_pool3d(mask, (1, 2, 2), ceil_mode=True)
            pooled = F.avg_pool3d(values[-1] * mask, (1, 2, 2), ceil_mode=True)
            values.append(pooled / count.clamp_min(1e-6))
            masks.append(count[:, 0] > 0)
        message = torch.zeros_like(values[-1])
        for level in (2, 1):
            value = values[level] + message
            message = self.scales[level - 1](value, masks[level]) + message
            shape = values[level - 1].shape[-2:]
            weight = self.resize(masks[level][:, None].float(), shape)
            message = self.resize(message, shape) / weight.clamp_min(1e-6)
            message = message * masks[level - 1][:, None]
        return super().forward(features, valid) + message


def candidate_plane_grid(depth_candidates, points, K_ref, T_ref, stride, use_metric=True):
    """把前向/水平面与四条邻射线的精确交点映射到既有候选体。

points 为 rig 坐标 [B,3,D,H,W]，T_ref 是 cam_from_rig；depth_candidates
是全图共用、严格递增的 Z-depth 格点 [D] 或 [B,D]，可非均匀。这里只有
粗候选格；不能将每像素不同的细候选序号当成统一深度轴。返回 grid 的
维度为 [B,plane,neighbor,D,H,W,xyz]，坐标遵守 align_corners=False。
    """
    batch, _, count, height, width = points.shape
    if count < 2:
        raise ValueError("公制空间邻接至少需要两个递增深度格点")
    if depth_candidates.ndim not in (1, 2):
        raise ValueError("公制空间邻接需要全图共用的 [D] 或 [B,D] 深度格")
    with torch.autocast(device_type=points.device.type, enabled=False):
        lattice = depth_candidates.float().reshape(-1, count).expand(batch, -1)
        intrinsic = K_ref.float().reshape(-1, 3, 3).expand(batch, -1, -1)
        transform = T_ref.float().reshape(-1, 4, 4).expand(batch, -1, -1)
        rotation = transform[:, :3, :3]
        origin = -(rotation.transpose(-1, -2) @ transform[:, :3, 3, None])[..., 0]
        y, x = torch.meshgrid(
            torch.arange(height, device=points.device),
            torch.arange(width, device=points.device), indexing="ij",
        )
        # 格点位置采用与 decoder pixel_grid 相同的整数像素中心。
        offsets = points.new_tensor(((0, -1), (0, 1), (-1, 0), (1, 0)))
        neighbor_x = x.float()[None] + offsets[:, 0, None, None]
        neighbor_y = y.float()[None] + offsets[:, 1, None, None]
        pixels = torch.stack((neighbor_x * stride, neighbor_y * stride,
                              torch.ones_like(neighbor_x)), 0).flatten(1)
        camera_rays = torch.linalg.solve(intrinsic, pixels[None].expand(batch, -1, -1))
        rays = (rotation.transpose(-1, -2) @ camera_rays).reshape(batch, 3, 4, height, width)
        front = rotation[:, 2]
        up = front.new_tensor((0, 0, 1)).expand(batch, -1)
        normals = torch.stack((front, up), 1)
        displacement = points.float() - origin[:, :, None, None, None]
        numerator = torch.einsum("bpc,bcdhw->bpdhw", normals, displacement)
        denominator = torch.einsum("bpc,bcnhw->bpnhw", normals, rays)
        horizon = denominator.abs() > 1e-6
        safe_denominator = torch.where(horizon, denominator, 1)
        target_depth = numerator[:, :, None] / safe_denominator[:, :, :, None]
        # 前向平面解析上恒定 Z，直接保留格点值，避免 RᵀR 的浮点误差
        # 将 near/far 端点错误判为越界；倾斜相机同样成立。
        front_depth = lattice[:, None, None, :, None, None].expand(batch, 1, 4, count, height, width)
        target_depth = torch.cat((front_depth, target_depth[:, 1:]), 1)
        if not use_metric:
            # 相同参数/邻居数的空间控制，只移除由标定平面产生的深度剪切。
            target_depth = lattice[:, None, None, :, None, None].expand_as(target_depth)
            horizon = torch.ones_like(horizon)
        inside_image = ((neighbor_x >= 0) & (neighbor_x < width)
                        & (neighbor_y >= 0) & (neighbor_y < height))
        legal = horizon[:, :, :, None] & inside_image[None, None, :, None]
        legal = legal & torch.isfinite(target_depth) & (target_depth > 0)
        legal = legal & (target_depth >= lattice[:, :1, None, None, None, None])
        legal = legal & (target_depth <= lattice[:, -1:, None, None, None, None])
        # searchsorted 给出真实非均匀格的包围区间；不能将 Z 或 1/Z
        # 直接线性归一化后交给 grid_sample，否则 hybrid 格会错位。
        finite_depth = torch.where(legal, target_depth, lattice[:, :1, None, None, None, None])
        flat_depth = finite_depth.reshape(batch, -1).contiguous()
        upper = torch.searchsorted(lattice.contiguous(), flat_depth, right=True).clamp(1, count - 1)
        lower = upper - 1
        low_depth = lattice.gather(1, lower)
        high_depth = lattice.gather(1, upper)
        fraction = (flat_depth - low_depth) / (high_depth - low_depth).clamp_min(1e-8)
        depth_index = (lower + fraction).reshape_as(target_depth)
        gx = (2 * (neighbor_x + 0.5) / width - 1)[None, None, :, None].expand_as(target_depth)
        gy = (2 * (neighbor_y + 0.5) / height - 1)[None, None, :, None].expand_as(target_depth)
        gz = 2 * (depth_index + 0.5) / count - 1
        grid = torch.stack((gx, gy, gz), -1)
        return grid, legal, target_depth


class CandidatePlaneMessage(nn.Module):
    """一次公制面片邻接，替换逐射线 local，不重复读取四目特征。

两种平面只是邻接假设，不是强制表面形状。候选本身保留；相邻候选经
内容相容性、几何合法性及 null 竞争后提供消息。输出是 delta，由调用者
相加一次。use_metric=False 保留同样的参数/八个邻居，作为不剪切控制。
    """

    def __init__(self, channels, use_metric=True):
        super().__init__()
        self.use_metric = bool(use_metric)
        self.depthwise = nn.Conv3d(channels, channels, (3, 1, 1),
                                   padding=(1, 0, 0), groups=channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv3d(channels, channels, 1)
        self.query = nn.Conv3d(channels, 3, 1)
        self.neighbor_weight = nn.Parameter(torch.zeros(1, channels, 2, 4, 1, 1, 1))
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    @torch.no_grad()
    def initialize_from_depth(self, local):
        self.depthwise.load_state_dict(local[0].state_dict())
        self.pointwise.load_state_dict(local[2].state_dict())
        self.neighbor_weight.zero_()
        return self

    def forward(self, features, valid, depth_candidates, points, K_ref, T_ref, stride, **geometry):
        batch, channels, count, height, width = features.shape
        mask = valid[:, None].bool()
        value = torch.where(mask, features, 0)
        grid, legal, _ = candidate_plane_grid(depth_candidates, points, K_ref, T_ref,
                                             stride, self.use_metric)
        # 八组查询沿输出 depth 维拼接，只调用一次 grid_sample，不复制候选体。
        source = torch.cat((value, valid[:, None].to(value.dtype)), 1)
        sampled = F.grid_sample(source, grid.reshape(batch, 8 * count, height, width, 3),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(batch, channels + 1, 2, 4, count, height, width)
        # 要求插值支撑点均有效，缺失格不能以零特征参与匹配。
        available = legal & (sampled[:, -1] > 0.999) & valid[:, None, None]
        neighbors = torch.where(available[:, None], sampled[:, :-1], 0)
        normalized = F.normalize(value.float(), dim=1, eps=1e-6)
        neighbor_norm = torch.linalg.vector_norm(neighbors.float(), dim=1).clamp_min(1e-6)
        similarity = (neighbors.float() * normalized[:, :, None, None]).sum(1) / neighbor_norm
        query = self.query(value).float()
        scores = similarity * F.softplus(self.similarity_scale) + query[:, :2, None]
        scores = scores.flatten(1, 2)
        available = available.flatten(1, 2)
        null_score = query[:, 2:3] + available.sum(1, keepdim=True).clamp_min(1).float().log()
        scores = torch.cat((scores, null_score), 1)
        availability = torch.cat((available, torch.ones_like(valid[:, None])), 1)
        weights = safe_masked_softmax(scores, availability, dim=1)[:, :8]
        weights = weights.reshape(batch, 2, 4, count, height, width).to(neighbors.dtype)
        # 固定八个方向分块累积，避免 einsum 为重排整个候选体再复制一次。
        # 这不是逐候选循环，且前面的三维采样仍只有一次调用。
        message = torch.zeros_like(value)
        for plane in range(2):
            for neighbor in range(4):
                weight = weights[:, plane, neighbor, None] * self.neighbor_weight[:, :, plane, neighbor]
                message = torch.addcmul(message, neighbors[:, :, plane, neighbor], weight)
        result = self.pointwise(self.activation(self.depthwise(value) + message))
        return torch.where(mask, result, 0)
