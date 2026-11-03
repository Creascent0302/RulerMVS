"""粗候选的轻量多层重新融合；不产生新投影，也不提前回归唯一深度。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import safe_masked_softmax


def spatial_pool(value, valid):
    """仅在图像平面汇聚，保留每个深度候选及部分有效边缘。"""
    mask = valid[:, None].to(value.dtype)
    count = F.avg_pool3d(mask, (1, 2, 2), ceil_mode=True)
    total = F.avg_pool3d(torch.where(mask.bool(), value, 0), (1, 2, 2), ceil_mode=True)
    return total / count.clamp_min(1e-6), count[:, 0] > 0


class ImageEvidenceRound(nn.Module):
    """同一深度带内做全局空间交流，再读取一对不同层级的图像特征。

    全局注意力只作用于 stride16 的二维格点，不对整个候选体做两两注意力。
    深度带共享参数但不在此处互相平均，以保留多个互斥表面的解释。
    """

    def __init__(self, hidden, raw_channels, heads):
        super().__init__()
        self.heads = heads
        self.image = nn.Conv2d(2 * raw_channels, hidden, 1)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))

    def forward(self, tokens, raw_pair, valid):
        batch, bands, height, width, hidden = tokens.shape
        image = self.image(torch.cat(raw_pair, dim=1)).permute(0, 2, 3, 1)
        image = image[:, None].expand(-1, bands, -1, -1, -1)
        tokens = tokens + self.fusion(torch.cat((tokens, image), dim=-1))
        tokens = torch.where(valid[..., None], tokens, 0)
        sequence = tokens.reshape(batch * bands, height * width, hidden)
        sequence_valid = valid.reshape(batch * bands, height * width)
        qkv = self.qkv(self.norm(sequence)).reshape(
            batch * bands, height * width, 3, self.heads, hidden // self.heads
        ).permute(2, 0, 3, 1, 4)
        message = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=sequence_valid[:, None, None], dropout_p=0.
        ).transpose(1, 2).flatten(2)
        sequence = sequence + self.output(message)
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        sequence = torch.where(sequence_valid[..., None], sequence, 0)
        return sequence.reshape(batch, bands, height, width, hidden)


class CandidateRefusion(nn.Module):
    """96候选→8深度带上下文→原96候选评分残差。

    低秩支路只携带全局消息，原候选特征始终完整保留。两轮分别读取
    单图编码器浅两层、深两层；维度无关的标尺足迹与null质量共同参与
    汇聚。输出零初始化，开始训练时精确等价于V6；首步先学输出投影。
    """

    def __init__(self, channels, raw_channels, options=None):
        super().__init__()
        options = options or {}
        hidden = int(options.get("hidden", 32))
        heads = int(options.get("heads", 4))
        if hidden % heads:
            raise ValueError("refusion hidden必须能被heads整除")
        self.enabled = bool(options.get("enabled", True))
        self.bands = int(options.get("bands", 8))
        self.maximum_score_change = float(options.get("maximum_score_change", 4.))
        if self.bands < 1 or self.maximum_score_change <= 0:
            raise ValueError("bands和maximum_score_change必须为正")
        self.query = nn.Conv3d(channels, hidden, 1)
        self.geometry = nn.Conv3d(2, hidden, 1)
        self.pool_score = nn.Conv3d(hidden, 1, 1)
        self.position = nn.Linear(3, hidden)
        self.rounds = nn.ModuleList([
            ImageEvidenceRound(hidden, raw_channels, heads) for _ in range(2)
        ])
        self.readout = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.readout.weight)
        self.query_norm = nn.LayerNorm(hidden)

    def forward(self, features, raw, valid, null_mass, dimensionless_information):
        batch, _, count, height, width = features.shape
        # 任意候选数都选可整除的带数，不丢候选、不用填充造假深度。
        bands = max(number for number in range(1, min(self.bands, count) + 1) if count % number == 0)
        per_band = count // bands
        query = self.query(torch.where(valid[:, None], features, 0))
        pooled, pooled_valid = spatial_pool(query, valid)
        geometry, _ = spatial_pool(torch.cat((null_mass, dimensionless_information), dim=1), valid)
        pooled = pooled + self.geometry(geometry)
        small_height, small_width = pooled.shape[-2:]
        scores = self.pool_score(pooled)[:, 0].unflatten(1, (bands, per_band))
        grouped_valid = pooled_valid.unflatten(1, (bands, per_band))
        probability = safe_masked_softmax(scores.float(), grouped_valid, dim=2)
        tokens = (pooled.unflatten(2, (bands, per_band)) * probability[:, None]).sum(3)
        tokens = tokens.permute(0, 2, 3, 4, 1)
        band_valid = grouped_valid.any(2)
        z, y, x = torch.meshgrid(
            torch.linspace(0., 1., bands, device=features.device),
            torch.linspace(-1., 1., small_height, device=features.device),
            torch.linspace(-1., 1., small_width, device=features.device), indexing="ij"
        )
        tokens = tokens + self.position(torch.stack((x, y, z), dim=-1))[None]
        image_valid = valid.any(1, keepdim=True)
        pooled_raw = [spatial_pool(value[:, :, None], image_valid)[0][:, :, 0] for value in raw]
        for block, start in zip(self.rounds, (0, 2)):
            tokens = block(tokens, pooled_raw[start:start + 2], band_valid)
        context = self.readout(tokens).permute(0, 1, 4, 2, 3).flatten(0, 1)
        context = F.interpolate(context, (height, width), mode="bilinear", align_corners=False)
        context = context.reshape(batch, bands, -1, height, width)
        query = self.query_norm(query.movedim(1, -1)).movedim(-1, 1)
        query = query.unflatten(2, (bands, per_band)).permute(0, 2, 3, 1, 4, 5)
        delta = (query * context[:, :, None]).sum(3).flatten(1, 2) / math.sqrt(context.shape[2])
        limit = self.maximum_score_change
        return torch.where(valid, limit * torch.tanh(delta / limit), 0)
