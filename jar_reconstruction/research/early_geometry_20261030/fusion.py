"""空间判断后的单目差分读取；只用于显式研究入口。

粗候选原入口融合保持不变。空间交互后，用更新后的候选再查询同一
四层图像bank，加入新旧读取的差分。共享原投影层，不增加第二套ViT。
"""

import math

import torch
from torch import nn

from jar.pretraining.fast_fusion import factorized_visual
from jar.pretraining.model import CalibratedDecoder


class LateReadingDecoder(CalibratedDecoder):
    def _read_visual(self, features, bank):
        if self.settings.get("factorized_fusion", False):
            return factorized_visual(features, bank, self.raw_query, self.raw_output)
        query = self.raw_query(features)
        logits = torch.einsum("bcdhw,bclhw->bldhw", query, bank).float() / math.sqrt(bank.shape[1])
        visual = torch.einsum("bldhw,bclhw->bcdhw", logits.softmax(1).to(bank.dtype), bank)
        return self.raw_output(visual)

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        # 保留生产入口的运算顺序；这里只把已有bank/visual留给本次粗层。
        metric = source["information"].clamp_min(0).log1p() * float(self.use_geometry)
        features = self.evidence(torch.cat((source["pooled"], source["null_mass"], metric), 1))
        position = (candidates.log() - math.log(self.near)) / math.log(self.far / self.near)
        position = torch.stack((position, (math.pi * position).sin(), (math.pi * position).cos()), -1)
        features = features + self.position(position).movedim(-1, 1)
        raw = [self._aligned(value, candidates.shape[-2:], stride) for value in prepared["raw"]]
        bank = torch.stack(raw, 2)
        visual = self._read_visual(features, bank)
        if stride == 8:
            observation["late_bank"] = bank
            observation["early_visual"] = visual
        valid = observation["ref_valid"].expand_as(candidates)
        features = self.entry_norm(features + visual) * valid[:, None]
        cosine = (observation["source"] * observation["reference"][:, None, :, None]).sum(2)
        similarity = (cosine * source["weights"]).sum(1)
        return features, raw, valid, similarity, observation["source_valid"].sum(1), source

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        if observation["stride"] == 8:
            later_visual = self._read_visual(features, observation["late_bank"])
            gain = 0.1 * self.late_gain.tanh()[None, :, None, None, None]
            difference = later_visual - observation["early_visual"]
            features = features + gain * difference * valid[:, None]
        return super()._prediction(features, candidates, valid, similarity, support,
                                   lower, upper, observation, use_offset)


def install_fusion(model, enabled=True):
    """加载原权重后显式安装；新增64个参数，零初始化保持原始函数。"""
    if not enabled:
        return model
    decoder = model.decoder
    if isinstance(decoder, LateReadingDecoder):
        return model
    if type(decoder) is not CalibratedDecoder:
        raise TypeError("本研究只组合原V2解码器，不与其他实验解码器隐式叠加")
    decoder.__class__ = LateReadingDecoder
    prototype = next(decoder.parameters())
    decoder.late_gain = nn.Parameter(prototype.new_zeros(decoder.settings["channels"]))
    return model
