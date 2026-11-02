"""V7研究分支：复用V6，新增模块独立开关，不修改V6源码和权重。"""

import copy

import torch

from jar.geometry import safe_masked_softmax
from jar.pretraining.fast_spatial import install_spatial_layout
from research.rulermvs_v6_20261101.model import V6Decoder, V6Model
from .center import ProjectiveCenter
from .prior import EvidencePrior


class V7Decoder(V6Decoder):
    def __init__(self, settings, handoff, options):
        super().__init__(settings, handoff)
        self.center = ProjectiveCenter() if options.get("projective_center", False) else None
        self.context_prior = EvidencePrior(
            raw_channels=settings["raw_channels"], candidates=settings["depth_bins"]
        ) if options.get("context_prior", False) else None

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        result = super()._tokens_from_source(prepared, observation, candidates, stride, source)
        if self.context_prior is not None and stride == 8:
            observation["prior_logits"] = self.context_prior(
                result[1], source["information"], source["null_mass"], candidates)
        return result

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        result = super()._prediction(features, candidates, valid, similarity, support,
                                     lower, upper, observation, use_offset)
        if "prior_logits" in observation:
            scores = (result["scores"] + observation["prior_logits"]).masked_fill(~valid, -1e4)
            probability = safe_masked_softmax(scores.float(), valid, dim=1)
            winner = scores.argmax(1, keepdim=True)
            depth = result["candidate_depth"].gather(1, winner) * result["ref_valid"]
            result.update(scores=scores, probability=probability, selected_index=winner,
                          depth=depth, confidence=probability.gather(1, winner))
        return result

    def _render(self, fine, features, image_mask, prepared):
        if self.center is not None:
            depth, gate, projected = self.center(fine, prepared)
            fine.update(group_depth=depth, center_gate=gate, projective_center=projected)
            fine["depth"] = depth.gather(1, fine["selected_group"]) * fine["ref_valid"]
        return super()._render(fine, features, image_mask, prepared)


class V7Model(V6Model):
    architecture = "rulermvs_v7"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        original = self.decoder.state_dict()
        with torch.random.fork_rng(devices=[]):
            self.decoder = V7Decoder(config["model"], config.get("handoff", {}), config["v7"])
        state = self.decoder.state_dict()
        state.update(original)
        self.decoder.load_state_dict(state, strict=True)
        if config["model"].get("spatial_layout", False):
            install_spatial_layout(self)
        self.model_name = "RulerMVS-V7-research"


def build_model(saved, variant="control"):
    config = copy.deepcopy(saved["config"])
    if saved["architecture"] == "rulermvs_v7":
        model = V7Model(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
    elif saved["architecture"] == "rulermvs_v6":
        if variant == "control":
            model = V6Model(config, initialize=False)
            model.load_state_dict(saved["model"], strict=True)
        elif variant in ("projective", "prior", "combined"):
            config.update(architecture="rulermvs_v7", v7={
                "projective_center": variant in ("projective", "combined"),
                "context_prior": variant in ("prior", "combined")})
            model = V7Model(config, initialize=False)
            extra = {name: value for name, value in model.state_dict().items()
                     if name.startswith(("decoder.center.", "decoder.context_prior."))}
            model.load_state_dict({**saved["model"], **extra}, strict=True)
        else:
            raise ValueError(f"未知V7实验分支: {variant}")
    else:
        raise ValueError("V7只从冻结V6或本版本权重加载")
    return model, config
