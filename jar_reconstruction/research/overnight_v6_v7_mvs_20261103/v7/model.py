"""V7多层候选重融合：独立继承V6，保持原模型与权重可直接回退。"""

import torch

from jar.geometry import safe_masked_softmax
from jar.pretraining.fast_spatial import install_spatial_layout
from research.rulermvs_v6_20261101.model import V6Decoder, V6Model
from .refusion import CandidateRefusion


class V7Decoder(V6Decoder):
    def __init__(self, settings, handoff, options=None):
        super().__init__(settings, handoff)
        self.refusion = CandidateRefusion(settings["channels"], settings["raw_channels"], options)

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        result = super()._tokens_from_source(prepared, observation, candidates, stride, source)
        if stride == 8 and self.refusion.enabled:
            half = observation["cell_half_width"]
            if half.ndim == 1:
                half = half[None, None, :, None, None]
            else:
                half = half[:, None]
            # 导数单位与候选半宽相消；这里只描述可观测尺度，不能认证匹配正确。
            information = (source["information"].clamp_min(0) * half.square()).log1p().clamp_max(8.)
            observation["refusion_data"] = (result[1], source["null_mass"], information)
        return result

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        result = super()._prediction(features, candidates, valid, similarity, support,
                                     lower, upper, observation, use_offset)
        if self.refusion.enabled and observation["stride"] == 8:
            raw, null, information = observation["refusion_data"]
            delta = self.refusion(features, raw, valid, null, information)
            scores = (result["scores"] + delta).masked_fill(~valid, -1e4)
            probability = safe_masked_softmax(scores.float(), valid, dim=1)
            winner = scores.argmax(1, keepdim=True)
            depth = result["candidate_depth"].gather(1, winner) * result["ref_valid"]
            result.update(scores=scores, probability=probability, selected_index=winner,
                          depth=depth, confidence=probability.gather(1, winner),
                          refusion_delta=delta)
        return result


class V7Model(V6Model):
    architecture = "rulermvs_v7_refusion"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        original = self.decoder.state_dict()
        with torch.random.fork_rng(devices=[]):
            decoder = V7Decoder(config["model"], config.get("handoff", {}), config.get("refusion", {}))
        decoder.load_state_dict({**decoder.state_dict(), **original}, strict=True)
        self.decoder = decoder
        if config["model"].get("spatial_layout", False):
            install_spatial_layout(self)
        self.model_name = "RulerMVS-V7"

    def load_v6_state(self, state):
        expected = {key for key in self.state_dict() if not key.startswith("decoder.refusion.")}
        if set(state) != expected:
            raise ValueError("V7只接受完整V6底座；不会静默忽略不兼容参数")
        self.load_state_dict({**self.state_dict(), **state}, strict=True)


def model_factory(config, initialize=True):
    architecture = config["architecture"]
    if architecture == "rulermvs_v7_refusion":
        return V7Model(config, initialize)
    if architecture == "rulermvs_v6":
        return V6Model(config, initialize)
    raise ValueError(f"此入口不支持架构：{architecture}")
