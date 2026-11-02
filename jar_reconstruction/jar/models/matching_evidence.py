"""冻结的官方匹配描述子与本地匹配分数融合，不改变几何支持判定。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.encoder import norm
from jar.pretrained_matching import sample_pretrained_matching


class PretrainedMatchingEvidence(nn.Module):
    """保留原生16维度量，仅学习按候选证据调节的正分数权重。"""

    def __init__(self, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 16))
        initial_weight = float(settings.get("initial_weight", 0.5))
        self.max_weight = float(settings.get("max_weight", 2.0))
        if hidden < 1 or not 0 < initial_weight < self.max_weight:
            raise ValueError("匹配证据要求hidden_dim>0，且0<initial_weight<max_weight")
        self.weight_network = nn.Sequential(
            nn.Conv3d(6, hidden, 1, bias=False),
            norm(hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            norm(hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, 1),
        )
        # 起点就使用已有匹配信息；小非零末层权重使前层第一步即可学习。
        nn.init.normal_(self.weight_network[-1].weight, std=1e-3)
        ratio = initial_weight / self.max_weight
        nn.init.constant_(self.weight_network[-1].bias, math.log(ratio / (1 - ratio)))
        per_view = settings.get("per_view", {})
        if per_view.get("enabled", False):
            from .per_view_matching import PerViewMatchingReliability

            with torch.random.fork_rng(devices=[]):
                self.per_view = PerViewMatchingReliability(per_view)

    @staticmethod
    def sample_cosine(features, reference_uv, reference, projected_uv, source_mask, image_scale=None):
        """保留[N,D,P]描述子相似度，投影合法性完全沿用调用方。"""
        views, depths, pixels = source_mask.shape
        if features.shape[:2] != (views, 16):
            raise ValueError("官方匹配特征须为[N,16,H4,W4]")
        ref = sample_pretrained_matching(features[reference : reference + 1], reference_uv[None], image_scale)[0]
        source = sample_pretrained_matching(features, projected_uv, image_scale).reshape(
            views, depths, pixels, 16
        )
        cosine = (
            F.normalize(source.float(), dim=-1, eps=1e-6)
            * F.normalize(ref.float(), dim=-1, eps=1e-6)[None, None]
        ).sum(-1)
        return torch.where(source_mask, cosine, 0)

    @staticmethod
    def sample_statistics(features, reference_uv, reference, projected_uv, source_mask, image_scale=None):
        """输入原图坐标，返回两项[D,P]统计；原均值路径保持不变。"""
        cosine = PretrainedMatchingEvidence.sample_cosine(
            features, reference_uv, reference, projected_uv, source_mask, image_scale
        )
        return PretrainedMatchingEvidence.statistics_from_cosine(cosine, source_mask)

    @staticmethod
    def statistics_from_cosine(cosine, source_mask):
        """已采样的逐源相似度复用同一掩码，不重复读取教师特征。"""
        count = source_mask.sum(0).clamp_min(1)
        mean = cosine.sum(0) / count
        variance = torch.where(source_mask, (cosine - mean[None]).square(), 0).sum(0) / count
        return torch.stack((mean, variance), 0)

    def forward(self, local_cost, prior_statistics, groups):
        """local_cost为[G组相关+4元数据]，返回分数修正及有界权重。"""
        description = torch.cat(
            (
                local_cost[:, :groups].mean(1, keepdim=True),
                prior_statistics,
                local_cost[:, groups : groups + 1],  # 有效源视图比例
                local_cost[:, groups + 2 : groups + 4],  # log视差导数、纹理
            ),
            dim=1,
        )
        weight = self.max_weight * self.weight_network(description).float().sigmoid().squeeze(1)
        correction = weight * prior_statistics[:, 0]
        return correction, weight
