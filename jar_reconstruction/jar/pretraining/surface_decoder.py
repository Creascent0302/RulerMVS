"""多表面像素解码：保留局部备选，以逆深度面片代替常深度混合。

它替换旧的144通道上采样头，不新增编码器或多视图采样。
局部四组与八个邻居提供12个面片，输出像素选择其中一个；
面片选择不是可见性认证，外部unknown语义保持不变。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import CalibratedDecoder


def evaluate_planes(depth, slope, offsets):
    """在目标像素评价局部逆Z仿射场；offset单位是4个主图像素。

    depth: [B,P,1,1,H,W]；slope: [B,2,P,1,1,H,W]；
    offsets: [2,P,4,4,1,1]。返回[B,P,4,4,H,W]。
    斜率L1<=.25且|offset|<=1.75时分母>=.5625，无需截断分母。
    """
    denominator = 1 + (slope * offsets[None]).sum(1)
    return depth / denominator


class SurfaceReadout(nn.Module):
    """64维候选→16维面片描述；stride2图像→8维子像素查询。"""

    def __init__(self, channels=64, hidden=16, key_dim=8, slope_limit=.25):
        super().__init__()
        if not 0 <= slope_limit < 1 / 1.75:
            raise ValueError('slope_limit必须保证整个3×3邻域内面片深度为正')
        self.key_dim = key_dim
        self.slope_limit = slope_limit
        self.return_render = False
        self.patch = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, key_dim + 4, 1),
        )
        self.query = nn.Sequential(nn.Conv2d(channels, key_dim * 4, 1), nn.PixelShuffle(2))
        # 保留原候选中心、初始平面正对相机；key保留小随机值使路由可训练。
        with torch.no_grad():
            self.patch[-1].weight[:3].zero_()
            self.patch[-1].bias[:3].zero_()
            self.patch[-1].weight[-1].zero_()
            self.patch[-1].bias[-1].zero_()
        self.register_buffer('neighbor_xy', torch.tensor(
            [(x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.)]
        ), persistent=False)

    @staticmethod
    def _gather(value, indices):
        if value.ndim == 5:
            indices = indices[:, None].expand(-1, value.shape[1], -1, -1, -1)
            return value.gather(2, indices)
        return value.gather(1, indices)

    @staticmethod
    def _neighbors(value):
        batch, channels, height, width = value.shape
        return F.unfold(value, 3, padding=1).reshape(batch, channels, 9, height, width)

    @staticmethod
    def _full(value):
        """[B,P,4,4,H,W]→[B,P,4H,4W]，严格沿用0,4,8格点锚。"""
        batch, count, _, _, height, width = value.shape
        return value.permute(0, 1, 4, 2, 5, 3).reshape(batch, count, height * 4, width * 4)

    def forward(self, fine, features, guide, image_mask, near, far):
        batch, channels, count, height, width = features.shape
        if count != 12 or guide.shape != (batch, channels, height * 2, width * 2):
            raise ValueError('面片版本要求四组×三候选，以及主图stride2参考特征')
        scores = fine['scores'].float().reshape(batch, 4, 3, height, width)
        local_valid = fine['valid'].reshape_as(scores)
        local = scores.masked_fill(~local_valid, -1e4).argmax(2)
        group_offset = torch.arange(4, device=features.device)[None, :, None, None] * 3
        indices = local + group_offset
        group_features = self._gather(features, indices)
        patch_input = group_features.permute(0, 2, 1, 3, 4).reshape(batch * 4, channels, height, width)
        parameters = self.patch(patch_input).float().reshape(batch, 4, self.key_dim + 4, height, width)
        parameters = parameters.permute(0, 2, 1, 3, 4)
        base_depth = self._gather(fine['candidate_depth'], indices).float()
        center = (base_depth * (.2 * parameters[:, 0].tanh()).exp()).clamp(near, far)
        slope = parameters[:, 1:3].tanh()
        slope = self.slope_limit * slope / slope.abs().sum(1, keepdim=True).clamp_min(1)
        key = parameters[:, 3:3 + self.key_dim]
        quality = parameters[:, -1]
        group_valid = local_valid.any(2)
        probability = fine['probability'].float().reshape(batch, 4, 3, height, width).sum(2)
        support = self._gather(fine['support'], indices).float()
        # 九邻居保留各自原全局winner；中心额外保留另外三组，不重复赢家。
        winning_group = fine['selected_index'] // 3
        alternatives = (winning_group + torch.arange(1, 4, device=features.device)[None, :, None, None]) % 4

        def supply(value):
            """value[B,C,4,H,W]→九邻域赢家＋三个本格备选。"""
            winner = self._gather(value, winning_group)[:, :, 0]
            return torch.cat((self._neighbors(winner), self._gather(value, alternatives)), 2)

        packet = torch.cat((center[:, None], slope, key, probability[:, None],
                            quality[:, None], group_valid[:, None].float(), support[:, None]), 1)
        packet = supply(packet)
        depth, slopes = packet[:, 0], packet[:, 1:3]
        keys = packet[:, 3:3 + self.key_dim]
        priors, qualities = packet[:, -4], packet[:, -3]
        available, supports = packet[:, -2] > .5, packet[:, -1]
        # 在图外的邻居不因unfold零填充获得合法深度或权重。
        depth = torch.where(available, depth, torch.ones_like(depth))
        xy = torch.cat((self.neighbor_xy, self.neighbor_xy.new_zeros(3, 2)))
        y, x = torch.meshgrid(torch.arange(4, device=features.device),
                              torch.arange(4, device=features.device), indexing='ij')
        offsets = torch.stack((x, y)).float()[:, None] / 4 - xy.T[:, :, None, None]
        offsets = offsets[..., None, None]
        query = self.query(guide).float().reshape(batch, self.key_dim, height, 4, width, 4)
        query = query.permute(0, 1, 3, 5, 2, 4)
        logits = torch.einsum('bcyxhw,bcphw->bpyxhw', query, keys) / math.sqrt(self.key_dim)
        spatial = offsets.square().sum(0)
        logits = logits + priors.clamp_min(1e-8).log()[:, :, None, None]
        logits = logits + qualities[:, :, None, None] - .25 * spatial[None]
        available = available[:, :, None, None].expand(-1, -1, 4, 4, -1, -1)
        logits = logits.masked_fill(~available, -1e4)
        probability_full = logits.softmax(1)
        winner = logits.argmax(1, keepdim=True)
        has_surface = available.any(1, keepdim=True)
        # 推理仅评价已选中的面片；训练另保留全部12面片供路由监督。
        selected_depth = depth[:, :, None, None].expand(-1, -1, 4, 4, -1, -1).gather(1, winner)
        selected_slope = slopes[:, :, :, None, None].expand(-1, -1, -1, 4, 4, -1, -1)
        selected_slope = selected_slope.gather(2, winner[:, None].expand(-1, 2, -1, -1, -1, -1, -1))
        selected_offset = offsets[None].expand(batch, -1, -1, -1, -1, height, width)
        selected_offset = selected_offset.gather(2, winner[:, None].expand(-1, 2, -1, -1, -1, -1, -1))
        selected_depth = selected_depth / (1 + (selected_slope * selected_offset).sum(1))
        selected_depth = selected_depth.clamp(near, far)
        selected_depth = torch.where(has_surface, selected_depth, math.sqrt(near * far))
        supports = supports[:, :, None, None].expand(-1, -1, 4, 4, -1, -1).gather(1, winner).long()
        valid = image_mask.bool()
        confidence = self._full(probability_full.gather(1, winner))
        has_surface = self._full(has_surface)
        supports = self._full(supports)
        output = {
            'depth': torch.where(valid, self._full(selected_depth), 0.), 'valid': valid,
            'confidence': torch.where(valid & has_surface, confidence, 0.),
            'support': torch.where(valid & has_surface, supports, 0),
            'unknown': torch.ones_like(valid), 'observed': torch.zeros_like(valid),
            'state': torch.full_like(supports, 2),
        }
        if self.training or self.return_render:
            plane = evaluate_planes(depth[:, :, None, None], slopes[:, :, :, None, None], offsets).clamp(near, far)
            output['render'] = {'plane_depth': self._full(plane), 'logits': self._full(logits),
                                'valid': self._full(available) & valid,
                                'selected_index': self._full(winner),
                                'group_depth': center, 'group_slope': slope}
        return output


class SurfaceDecoder(CalibratedDecoder):
    """保留粗/细取证，替换最终表面表示和像素选择的独立版本。"""

    def __init__(self, settings):
        super().__init__(settings)
        if self.peak_count != 4:
            raise ValueError('surface_transport版本固定4个候选组')
        del self.upsample_mask
        self.surface_readout = SurfaceReadout(
            settings['channels'], settings.get('surface_hidden', 16),
            settings.get('surface_key_dim', 8), settings.get('surface_slope_limit', .25))

    def forward(self, inputs):
        output = super().forward(inputs)
        output['support_semantics'] = (
            '所选面片锚点候选的可采样源视图数；'
            '未重新验证输出像素和修正深度，非可见性认证'
        )
        return output

    def _render(self, fine, features, image_mask, prepared):
        _, _, _, height, width = features.shape
        reference = prepared['matching'][:, prepared['reference']].float()
        # query只提供外观条件，由PixelShuffle学习像素归属，不把它当几何采样。
        # 默认匹配图为主图2倍，已有FPN恰好是主图stride2，直接复用。
        # 精确投影仍全部发生于_observe，面片仍在原像素0,1,...处求值。
        shape = (height * 2, width * 2)
        guide = reference if reference.shape[-2:] == shape else F.interpolate(
            reference, size=shape, mode='bilinear', align_corners=False)
        return self.surface_readout(fine, features, guide, image_mask, self.near, self.far)
