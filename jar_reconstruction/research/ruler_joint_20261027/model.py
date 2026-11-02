"""独立研究分支：三点响应联合读出；不改变生产模型默认配置。"""

import torch
from torch import nn

from jar.geometry import safe_masked_softmax
from jar.pretraining.model import CalibratedDecoder, load_pretraining_state


class FineGroupRelation(nn.Module):
    """比较同一候选组的中心及两侧响应，不把三点认作同一真实表面。"""

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.hidden = nn.Conv3d(3 * channels + 9, hidden, 1)
        self.activation = nn.GELU()
        self.output = nn.Conv3d(hidden, 6, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, candidates, similarity, support, max_sources):
        batch, channels, count, height, width = features.shape
        groups = count // 3
        grouped = features.reshape(batch, channels, groups, 3, height, width)
        center = grouped[:, :, :, 1]
        left = grouped[:, :, :, 0] - center
        right = grouped[:, :, :, 2] - center
        positions = candidates.float().clamp_min(1e-6).log()
        positions = positions.reshape(batch, groups, 3, height, width)
        positions = (positions - positions[:, :, 1:2]).permute(0, 2, 1, 3, 4)
        available = support.float() / max(max_sources, 1)
        available = available.reshape(batch, groups, 3, height, width).permute(0, 2, 1, 3, 4)
        matching = similarity.reshape(batch, groups, 3, height, width).permute(0, 2, 1, 3, 4)
        message = torch.cat((center, left, right, positions.to(features.dtype),
                             available.to(features.dtype), matching.to(features.dtype)), dim=1)
        residual = self.output(self.activation(self.hidden(message)))
        residual = residual.reshape(batch, 2, 3, groups, height, width)
        residual = residual.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2, count, height, width)
        return residual[:, 0], residual[:, 1]


class JointDecoder(CalibratedDecoder):
    """原96粗/12细结构；仅在细层的评分与tanh前偏移加入组内残差。"""

    def __init__(self, settings):
        super().__init__(settings)
        self.relation = FineGroupRelation(settings['channels'], settings.get('relation_hidden', 32))

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        if observation['stride'] != 4:
            return super()._prediction(features, candidates, valid, similarity, support,
                                       lower, upper, observation, use_offset)
        score_head, offset_head = self._prediction_heads(observation)
        score_delta, offset_delta = self.relation(features, candidates, similarity, support,
                                                   observation['view_count'] - 1)
        if not self.settings.get('relation_score', True):
            score_delta = torch.zeros_like(score_delta)
        scores = score_head(features)[:, 0] + 4 * similarity + score_delta
        scores = scores.masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(scores.float(), valid, dim=1)
        fraction = (offset_head(features)[:, 0] + offset_delta).tanh()
        if not use_offset:
            fraction = torch.zeros_like(candidates)
        length = torch.where(fraction >= 0, upper - candidates, candidates - lower)
        continuous = (candidates + length * fraction).clamp(self.near, self.far)
        winner = scores.argmax(1, keepdim=True)
        depth = continuous.gather(1, winner)
        depth = torch.where(valid.any(1, keepdim=True), depth, 0)
        sources = observation['source_valid']
        if sources.shape[1] != observation['view_count']:
            reference = observation['reference_index']
            sources = torch.cat((sources[:, :reference], torch.zeros_like(sources[:, :1]),
                                 sources[:, reference:]), dim=1)
        return {'scores': scores, 'probability': probability, 'candidate_depth': continuous,
                'sampling_depth': candidates, 'valid': valid, 'depth': depth,
                'support': support, 'ref_valid': valid.any(1, keepdim=True),
                'selected_index': winner, 'confidence': probability.gather(1, winner),
                'source_valid': sources, 'lower': lower.expand_as(candidates),
                'upper': upper.expand_as(candidates)}


def make_decoder(saved, options, device):
    settings = {**saved['config']['model'], 'independent_fine_heads': True,
                **options.get('model_overrides', {})}
    state = {key.removeprefix('decoder.'): value for key, value in saved['model'].items()
             if key.startswith('decoder.')}
    base = CalibratedDecoder(settings)
    load_pretraining_state(base, state)
    if options.get('relation', False):
        model = JointDecoder(settings)
        # 唯一新增参数来自显式零输出初始化；旧子树严格加载。
        model.load_state_dict({**model.state_dict(), **base.state_dict()}, strict=True)
    else:
        model = base
    return model.to(device), settings
