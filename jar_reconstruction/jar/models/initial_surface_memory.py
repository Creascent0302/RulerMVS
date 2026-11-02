"""保留深度轴的候选记忆与精确平面邻域运输。

由可选PersistentCandidateField调用，也支持独立候选排序训练。
输入来自普通图像前向，不读取深度标签。
同深度消融保留全部参数，仅关闭空间邻居读取时的平面几何变换。
"""

import torch
from torch import nn
from torch.nn import functional as F

from jar.encoder import norm
from jar.geometry import pixel_grid, unproject


class CandidateGroupNorm(nn.GroupNorm):
    """保持分组统计，避免巨大候选场落入低并行度的归约路径。"""

    def forward(self, features):
        elements_per_group = features.numel() // (len(features) * self.num_groups)
        if (
            not features.is_cuda
            or features.dtype != torch.float32
            or elements_per_group < 1_000_000
        ):
            return super().forward(features)

        # 与GroupNorm相同的总体方差、eps和affine；浮点归约顺序不同。
        grouped = features.reshape(len(features), self.num_groups, -1)
        variance, mean = torch.var_mean(grouped, dim=-1, correction=0, keepdim=True)
        normalized = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape_as(features)
        affine_shape = (1, self.num_channels) + (1,) * (features.ndim - 2)
        if self.affine:
            normalized = normalized * self.weight.reshape(affine_shape)
            normalized = normalized + self.bias.reshape(affine_shape)
        return normalized


class CandidateChannelNorm(nn.LayerNorm):
    """只归一化同一候选的通道，统计不跨深度、像素或采样相位。"""

    def forward(self, features):
        # [B,C,D,H,W] → [B,D,H,W,C]。继承LayerNorm保持weight/bias参数键。
        normalized = super().forward(features.movedim(1, -1))
        return normalized.movedim(-1, 1)


def candidate_norm(channels, normalization):
    if normalization == "group":
        original = norm(channels)
        return CandidateGroupNorm(original.num_groups, channels, eps=original.eps)
    if normalization == "candidate":
        return CandidateChannelNorm(channels)
    raise ValueError("normalization须为group或candidate")


def plane_sampling_plan(inverse_depths, intrinsic, transform, shape, stride, mode="plane", neighbor_step=1):
    """为两种平面、四个邻居生成逆深度插值索引。

参考射线采用光轴深度参数 X=C+r/rho。经过当前候选的平面满足
rho_neighbor=rho*(n·r_neighbor)/(n·r)。这里只使用精确平面交点关系。
"""
    height, width = shape
    uv = pixel_grid(height, width, device=inverse_depths.device, stride=stride)
    _, rays = unproject(uv, torch.ones_like(uv[..., 0]), intrinsic.float(), transform.float())
    normals = torch.stack((transform.float()[2, :3], transform.new_tensor([0., 0., 1.])))
    denominator = torch.einsum("pc,hwc->phw", normals, rays)
    lattice = inverse_depths.float()
    if not torch.all(lattice[1:] > lattice[:-1]):
        raise ValueError("候选逆深度必须严格递增")
    plans = []
    if neighbor_step < 1 or int(neighbor_step) != neighbor_step:
        raise ValueError("neighbor_step必须为正整数")
    for dy, dx in ((0, -neighbor_step), (0, neighbor_step), (-neighbor_step, 0), (neighbor_step, 0)):
        neighbor_rays = torch.roll(rays, shifts=(-dy, -dx), dims=(0, 1))
        numerator = torch.einsum("pc,hwc->phw", normals, neighbor_rays)
        safe = torch.where(denominator.abs() > 1e-6, denominator, 1)
        ratio = numerator / safe
        geometric = (denominator.abs() > 1e-6) & (ratio > 0)
        if mode == "same_depth":
            ratio = torch.ones_like(ratio)
            geometric = torch.ones_like(geometric)
        elif mode != "plane":
            raise ValueError("mode只能是plane或same_depth")
        query = lattice[None, :, None, None] * ratio[:, None]
        right = torch.searchsorted(lattice, query.contiguous()).clamp(1, len(lattice) - 1)
        left = right - 1
        fraction = (query - lattice[left]) / (lattice[right] - lattice[left])
        known = geometric[:, None] & (query >= lattice[0]) & (query <= lattice[-1])
        if dy < 0:
            known[..., : -dy, :] = False
        elif dy > 0:
            known[..., -dy:, :] = False
        if dx < 0:
            known[..., :, : -dx] = False
        elif dx > 0:
            known[..., :, -dx:] = False
        plans.append((dy, dx, left, right, fraction.clamp(0, 1), known))
    return plans


def transport_neighbors(features, valid, plans):
    """返回[B,C,plane,D,H,W]；无效间隙不插值，边缘不环绕。"""
    # 两个平面分开累加，只在最后拼接；每个元素的邻居求和顺序不变。
    messages = [torch.zeros_like(features), torch.zeros_like(features)]
    counts = [features.new_zeros(valid.shape), features.new_zeros(valid.shape)]
    for dy, dx, left, right, fraction, known in plans:
        # 四邻域只有一个非零位移，避免对零位移方向再复制整个候选场。
        dimension, shift = (-2, -dy) if dy else (-1, -dx)
        neighbor = torch.roll(features, shifts=shift, dims=dimension)
        neighbor_valid = torch.roll(valid, shifts=shift, dims=dimension)
        for plane in range(2):
            low = left[plane][None, None].expand(len(features), features.shape[1], -1, -1, -1)
            high = right[plane][None, None].expand_as(low)
            alpha = fraction[plane][None, None]
            sampled = torch.lerp(neighbor.gather(2, low), neighbor.gather(2, high), alpha)
            low_valid = neighbor_valid.gather(1, left[plane][None].expand(len(features), -1, -1, -1))
            high_valid = neighbor_valid.gather(1, right[plane][None].expand(len(features), -1, -1, -1))
            mask = known[plane][None] & valid
            mask = mask & (low_valid | (alpha[:, 0] >= 1 - 1e-6))
            mask = mask & (high_valid | (alpha[:, 0] <= 1e-6))
            messages[plane] = messages[plane] + sampled * mask[:, None]
            counts[plane] = counts[plane] + mask
    messages = torch.stack(messages, dim=2)
    counts = torch.stack(counts, dim=1)[:, None]
    return messages / counts.clamp_min(1), counts > 0


class CandidateMemoryBlock(nn.Module):
    def __init__(self, channels, normalization="group"):
        super().__init__()
        self.depth = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            candidate_norm(channels, normalization),
            nn.SiLU(),
        )
        self.plane_weights = nn.Conv3d(channels, 3, 1)
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            candidate_norm(channels, normalization),
        )

    def forward(self, features, valid, plans):
        features = F.silu(features + self.depth(features)) * valid[:, None]
        neighbors, known = transport_neighbors(features, valid, plans)
        logits = self.plane_weights(features)
        legal = torch.cat((known[:, 0], torch.ones_like(valid[:, None])), dim=1)
        weight = logits.masked_fill(~legal, -1e4).softmax(1)[:, :2]
        update = (neighbors * weight[:, None]).sum(2)
        return F.silu(features + self.spatial(update)) * valid[:, None]


class InitialSurfaceMemory(nn.Module):
    """每个深度候选共享同一个评分头，Hero评分保留直接通路。"""

    def __init__(
        self, cost_channels=12, channels=16, blocks=2, max_residual=8.0,
        normalization="group",
    ):
        super().__init__()
        self.max_residual = max_residual
        self.source_encoder = nn.Sequential(nn.Conv3d(2, 4, 1), nn.SiLU(), nn.Conv3d(4, 4, 1))
        # 默认group保持既有权重与运算；candidate仅用于独立归一化消融。
        self.input = nn.Sequential(
            nn.Conv3d(cost_channels + 12, channels, 1),
            candidate_norm(channels, normalization),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [CandidateMemoryBlock(channels, normalization) for _ in range(blocks)]
        )
        self.score = nn.Conv3d(channels, 1, 1)
        nn.init.normal_(self.score.weight, std=1e-3)
        nn.init.zeros_(self.score.bias)

    def forward(self, cost, hero_statistics, valid, inverse_depths, plans, source_cosine, source_valid):
        log_depth = -inverse_depths.float().log()
        coordinate = (log_depth - log_depth.mean()) / log_depth.std().clamp_min(1e-6)
        rho = inverse_depths.float() / inverse_depths.amax()
        coordinates = torch.stack((coordinate, rho))[None, :, :, None, None]
        coordinates = coordinates.expand(len(cost), -1, -1, *cost.shape[-2:])
        # 所有源视图共享编码器；先保留各源相对均值的偏差，再做置换不变聚合。
        description = torch.stack((source_cosine, source_cosine - hero_statistics[:, :1]), dim=2)
        encoded = self.source_encoder(description.flatten(0, 1)).unflatten(0, source_cosine.shape[:2])
        mask = source_valid[:, :, None]
        source_mean = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1)
        source_max = encoded.masked_fill(~mask, -1e4).amax(1)
        source_max = torch.where(mask.any(1), source_max, 0)
        inputs = torch.cat((cost.float(), hero_statistics.float(), coordinates, source_mean, source_max), dim=1)
        features = self.input(inputs) * valid[:, None]
        for block in self.blocks:
            features = block(features, valid, plans)
        residual = self.score(features).squeeze(1)
        residual = self.max_residual * torch.tanh(residual / self.max_residual)
        scores = 4 * hero_statistics[:, 0].float() + residual
        return scores.masked_fill(~valid, -1e4)
