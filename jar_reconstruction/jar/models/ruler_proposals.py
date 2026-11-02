"""在子射线上筛选候选，避免先裁剪父射线再复制其表面。

只混合相同公制刻度的后验，不跨深度刻度平均。邻居只提供候选供给，
候选仍须在目标射线上精确重投影；相邻像素相似不表示可见性认证。
"""

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, sample_maps


class SubrayTransport:
    """整数中心8j→4j的四邻域传递，无学习参数和全分辨率特征体。"""

    def __init__(self, coarse, prepared, mode):
        probability = coarse["probability"]
        batch, count, height, width = probability.shape
        device = probability.device
        y, x = torch.meshgrid(
            torch.arange(height * 2, device=device).float() / 2,
            torch.arange(width * 2, device=device).float() / 2,
            indexing="ij",
        )
        x0, y0 = x.floor().long(), y.floor().long()
        dx, dy = x - x0, y - y0
        self.indices = torch.stack([
            (y0 + oy).clamp_max(height - 1) * width + (x0 + ox).clamp_max(width - 1)
            for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1))
        ])
        spatial = torch.stack(((1 - dy) * (1 - dx), (1 - dy) * dx,
                               dy * (1 - dx), dy * dx))
        spatial = spatial[None].expand(batch, -1, -1, -1)
        if mode == "appearance":
            reference = prepared["matching"][:, prepared["reference"]]
            scale = prepared["scale"][:, None]
            fine_grid = pixel_grid(height * 2, width * 2, device=device, stride=4)
            fine_uv = (fine_grid.reshape(1, -1, 2) + 0.5) * scale / 4 - 0.5
            fine = sample_maps(reference, fine_uv).transpose(1, 2)
            parent_uv = torch.stack((self.indices % width, self.indices // width), -1).float() * 8
            parent_uv = (parent_uv.reshape(1, -1, 2) + 0.5) * scale / 4 - 0.5
            parent = sample_maps(reference, parent_uv).transpose(1, 2)
            parent = parent.reshape(batch, 16, 4, height * 2, width * 2)
            fine = fine.reshape(batch, 16, 1, height * 2, width * 2)
            cosine = (F.normalize(fine, dim=1) * F.normalize(parent, dim=1)).sum(1)
            spatial = spatial * ((cosine - 1) / 0.1).exp()
        elif mode != "bilinear":
            raise ValueError("proposal_transport必须为nearest/bilinear/appearance")
        self.spatial = spatial
        self.parent_probability = self.neighbors(probability)
        weights = self.parent_probability * spatial[:, None]
        self.probability = weights.sum(2)
        legal_weights = self.neighbors(coarse["valid"].float()) * spatial[:, None]
        legal_sum = legal_weights.sum(2, keepdim=True)
        fallback = legal_weights / legal_sum.clamp_min(torch.finfo(weights.dtype).tiny)
        denominator = self.probability[:, :, None]
        self.conditional = torch.where(
            denominator >= torch.finfo(weights.dtype).tiny,
            weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny),
            fallback,
        )
        self.valid = legal_sum[:, :, 0] > 0
        self.depth = (self.neighbors(coarse["candidate_depth"]) * self.conditional).sum(2)
        self.depth = torch.where(self.valid, self.depth, coarse["candidate_depth"].new_zeros(()))

    def neighbors(self, value):
        batch, count = value.shape[:2]
        index = self.indices.flatten()[None, None].expand(batch, count, -1)
        return value.flatten(2).gather(2, index).reshape(batch, count, *self.indices.shape)

    def selected(self, value, selected):
        """只读取选中的K个刻度；value为[B,C,D,H,W]，避免展开C×D细体。"""
        batch, channels, count, height, width = value.shape
        flat = value.flatten(2)
        result = value.new_zeros(batch, channels, *selected.shape[1:])
        for neighbor in range(4):
            index = selected * (height * width) + self.indices[neighbor]
            gathered = flat.gather(2, index.flatten(1)[:, None].expand(-1, channels, -1))
            gathered = gathered.reshape_as(result)
            weight = self.conditional[:, :, neighbor].gather(1, selected)
            result = result + gathered * weight[:, None]
        return result


def transport_proposals(model, coarse, features, observation, prepared, return_parent=False):
    transport = SubrayTransport(coarse, prepared, model.proposal_transport)
    candidates = transport.depth
    scores = transport.probability
    available = transport.valid.clone()
    chosen = []
    exclusion = int(model.settings.get("peak_exclusion_bins", 0))
    bins = torch.arange(scores.shape[1], device=scores.device)[None, :, None, None]
    for _ in range(model.peak_count):
        active = available.any(1, keepdim=True)
        index = scores.masked_fill(~available, -1).argmax(1, keepdim=True)
        index = torch.where(active, index, scores.argmax(1, keepdim=True))
        chosen.append(index)
        center = candidates.gather(1, index)
        available &= (candidates - center).abs() >= float(model.settings.get("peak_separation_m", 1.0))
        if exclusion > 0:
            # 粗刻度近邻通常属于同一宽峰，避免四个名额全部挤在一起。
            available &= (bins - index).abs() > exclusion
        available.scatter_(1, index, False)
    indices = torch.cat(chosen, 1)
    peaks = candidates.gather(1, indices)
    from .metric_ruler_network import depth_cell_edges

    edges = depth_cell_edges(model.depth_lattice)
    widths = (edges[1:] - edges[:-1])[indices] * 0.5
    derivative = torch.linalg.vector_norm(observation["ray_derivative"], dim=2)
    derivative = torch.where(observation["source_valid"], derivative, 0).amax(1)
    g = transport.selected(derivative[:, None], indices)[:, 0]
    # 插值导数只控制局部搜索半径；全部实际候选仍使用精确投影。
    steps = torch.maximum(widths, float(model.settings.get("fine_step_pixels", 0.5)) / g.clamp_min(1e-6))
    steps = steps.clamp(float(model.settings.get("fine_min_step_m", 0.05)),
                       float(model.settings.get("fine_max_step_m", 8.0)))
    offsets = peaks.new_tensor((-1., 0., 1.))[None, None, :, None, None]
    fine = (peaks[:, :, None] + steps[:, :, None] * offsets).flatten(1, 2).clamp(model.near, model.far)
    half_width = steps[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2) * 0.5
    inherited = transport.selected(features, indices).repeat_interleave(3, 2)
    if return_parent:
        # V6显式维护候选族身份；默认路径不增加张量计算，也不改变V2/V5输出。
        parent = scores.gather(1, indices).detach()
        return fine, half_width, inherited, parent
    return fine, half_width, inherited
