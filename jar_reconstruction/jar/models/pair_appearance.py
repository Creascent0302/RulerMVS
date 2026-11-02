"""显式参考/源外观对的可选匹配残差；保留候选、视图及几何有效性。

候选对应已由标定精确投影确定。这里把两侧特征、逐通道差与积同时交给
网络，使其不必仅通过压缩后的相似度或 query-key 点积推断匹配关系。
每个候选/视图独立处理，随后仍使用原解码器的可靠性权重聚合。
"""

import math

import torch
from torch import nn


class PairResidualLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, features):
        return features + self.body(features)


class PairAppearance(nn.Module):
    """[Q,C] 与 [Q,D,N,C] → 有界的逐视图 logit 修正 [Q,D,N]。"""

    def __init__(self, channels, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 64))
        layers = int(settings.get("layers", 2))
        self.max_residual = float(settings.get("max_residual", 4.0))
        if hidden < 1 or layers < 1:
            raise ValueError("pair_appearance.hidden_dim/layers 必须为正")
        if not math.isfinite(self.max_residual) or self.max_residual <= 0:
            raise ValueError("pair_appearance.max_residual 必须有限且为正")
        # 相同归一化用于匹配两侧，避免引入参考与源的不同特征标尺。
        self.feature_norm = nn.LayerNorm(channels)
        self.input = nn.Sequential(nn.Linear(4 * channels + 8, hidden), nn.GELU())
        self.layers = nn.Sequential(*[PairResidualLayer(hidden) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, reference, source, metadata, valid):
        if reference is None or reference.shape != (source.shape[0], source.shape[-1]):
            raise ValueError("pair_appearance 需要实际参考图特征 reference_feature[Q,C]")
        # 在归一化前去除无效位置，既不读补零相机的内容，也避免 0*NaN。
        observed = valid.any(dim=(1, 2))
        reference = torch.where(observed[:, None], reference, 0.0)
        source = torch.where(valid[..., None], source, 0.0)
        metadata = torch.where(valid[..., None], metadata, 0.0)
        source = self.feature_norm(source)
        reference = self.feature_norm(reference)[:, None, None].expand_as(source)
        pair = torch.cat((reference, source, source - reference, source * reference, metadata), -1)
        features = self.layers(self.input(pair))
        residual = self.max_residual * self.output(features)[..., 0].float().tanh()
        return residual * valid
