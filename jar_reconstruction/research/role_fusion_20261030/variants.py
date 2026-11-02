"""按职责融合：单目上下文评估源证据，粗细阶段协商表面归属。

仅研究入口显式安装；不修改编码器、匹配描述子或生产默认模型。
"""

import torch
from torch.utils.checkpoint import checkpoint

from jar.geometry import safe_masked_softmax
from jar.models.metric_ruler_network import depth_cell_edges
from jar.models.ruler_proposals import SubrayTransport
from jar.pretraining.model import CalibratedDecoder

from .context import install_context


def transport_with_prior(model, coarse, features, observation, prepared):
    """沿用原四邻域候选传递，额外保留所选父峰质量；不增加投影。

    选择顺序、间距、搜索步长和连续候选与原transport_proposals相同。
    父质量只作为后续估计之间的协调信号，不当作独立新观测。
    """
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
            available &= (bins - index).abs() > exclusion
        available.scatter_(1, index, False)
    indices = torch.cat(chosen, 1)
    peaks = candidates.gather(1, indices)
    edges = depth_cell_edges(model.depth_lattice)
    widths = (edges[1:] - edges[:-1])[indices] * 0.5
    derivative = torch.linalg.vector_norm(observation["ray_derivative"], dim=2)
    derivative = torch.where(observation["source_valid"], derivative, 0).amax(1)
    g = transport.selected(derivative[:, None], indices)[:, 0]
    steps = torch.maximum(widths, float(model.settings.get("fine_step_pixels", 0.5)) / g.clamp_min(1e-6))
    steps = steps.clamp(float(model.settings.get("fine_min_step_m", 0.05)),
                       float(model.settings.get("fine_max_step_m", 8.0)))
    offsets = peaks.new_tensor((-1., 0., 1.))[None, None, :, None, None]
    fine = (peaks[:, :, None] + steps[:, :, None] * offsets).flatten(1, 2).clamp(model.near, model.far)
    half_width = steps[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2) * 0.5
    inherited = transport.selected(features, indices).repeat_interleave(3, 2)
    # 梯度不借此重复优化粗后验；粗阶段仍接受原监督，特征继承梯度不变。
    parent_probability = scores.gather(1, indices).detach()
    return fine, half_width, inherited, parent_probability


class RoleDecoder(CalibratedDecoder):
    def _tokens(self, prepared, observation, candidates, stride):
        if not self.role_options["context"]:
            return super()._tokens(prepared, observation, candidates, stride)
        context = torch.cat([self._aligned(value, candidates.shape[-2:], stride)
                             for value in prepared["raw"]], 1)
        arguments = (observation["reference"], observation["source"],
                     observation["source_valid"], observation["ray_derivative"],
                     observation["cell_half_width"], self.use_geometry, context)
        if self.training:
            source = checkpoint(self.source, *arguments, use_reentrant=False)
        else:
            source = self.source(*arguments)
        return self._tokens_from_source(prepared, observation, candidates, stride, source)

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        if self.role_options["carry"] and stride == 4:
            observation["source_null_mass"] = source["null_mass"]
        return super()._tokens_from_source(prepared, observation, candidates, stride, source)

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        result = super()._prediction(features, candidates, valid, similarity, support,
                                     lower, upper, observation, use_offset)
        if not self.role_options["carry"] or observation["stride"] != 4:
            return result
        scores, gate = self.handoff_gate(
            features, result["scores"], observation["parent_probability"],
            observation["source_null_mass"], observation["source_valid"])
        scores = scores.masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(scores.float(), valid, dim=1)
        winner = scores.argmax(1, keepdim=True)
        depth = result["candidate_depth"].gather(1, winner)
        result.update(scores=scores, probability=probability, selected_index=winner,
                      depth=torch.where(valid.any(1, keepdim=True), depth, 0),
                      confidence=probability.gather(1, winner), handoff_gate=gate)
        return result

    def forward(self, inputs):
        if not self.role_options["carry"]:
            return super().forward(inputs)
        # 原forward只传特征而不传父后验；本研究在这一交接增加一个张量。
        prepared = self._prepare(inputs)
        batch = prepared["matching"].shape[0]
        height, width = prepared["image_size"]
        candidates = self.depth_lattice[None, :, None, None].expand(batch, -1, height // 8, width // 8)
        edges = depth_cell_edges(self.depth_lattice)
        half_width = (edges[1:] - edges[:-1]) * 0.5
        observation = self._observe(prepared, candidates, 8, half_width)
        features, raw, valid, similarity, support, _ = self._tokens(prepared, observation, candidates, 8)
        if self.local_kind == "ray":
            local = self.local(features)
        elif self.local_kind in {"volume", "scale_space"}:
            local = self.local(features, valid)
        else:
            reference = prepared["reference"]
            local = self.local(features, valid, self.depth_lattice, observation["points"],
                               prepared["K"][:, reference], prepared["T"][:, reference], 8)
        features = features + local * valid[:, None]
        coarse = self._prediction(features, candidates, valid, similarity, support,
                                  edges[:-1][None, :, None, None], edges[1:][None, :, None, None], observation)
        fine_candidates, fine_width, inherited, parent = transport_with_prior(
            self, coarse, features, observation, prepared)
        fine_observation = self._observe(prepared, fine_candidates, 4, fine_width)
        fine_observation["parent_probability"] = parent
        fine_features, _, fine_valid, fine_similarity, fine_support, _ = self._tokens(
            prepared, fine_observation, fine_candidates, 4)
        fine_features = self.entry_norm(fine_features + self.fine_inherit(inherited)) * fine_valid[:, None]
        fine = self._prediction(fine_features, fine_candidates, fine_valid, fine_similarity, fine_support,
                                fine_candidates - fine_width, fine_candidates + fine_width, fine_observation,
                                use_offset=self.use_fine_offset)
        if self.fine_readout == "mode_mean":
            fine["depth"] = self._mode_depth(fine)
        image_mask = prepared["image_mask"][:, prepared["reference"]:prepared["reference"] + 1]
        output = self._render(fine, fine_features, image_mask, prepared)
        output.update(coarse=coarse, fine=fine, scene=None,
                      reference_index=prepared["reference"], continuous_offset_used=True,
                      support_semantics="几何可采样的源视图数；非可见性认证")
        return output


def install(model, options):
    options = {"context": False, "carry": False, **options}
    if not (options["context"] or options["carry"]):
        return model
    decoder = model.decoder
    if type(decoder) is not CalibratedDecoder:
        raise TypeError("按职责融合须从原V2显式构建，不隐式叠加其他研究变体")
    if decoder.use_scene or (options["carry"] and decoder.proposal_transport == "nearest"):
        raise ValueError("此研究采用原V2无scene、四邻域候选传递协议")
    decoder.__class__ = RoleDecoder
    decoder.role_options = options
    if options["context"]:
        install_context(decoder.source, 4 * decoder.settings["raw_channels"])
    if options["carry"]:
        from .carry import HandoffGate

        reference = next(decoder.parameters())
        with torch.random.fork_rng(devices=[]):
            decoder.handoff_gate = HandoffGate(decoder.settings["channels"])
        decoder.handoff_gate.to(device=reference.device, dtype=reference.dtype)
    return model
