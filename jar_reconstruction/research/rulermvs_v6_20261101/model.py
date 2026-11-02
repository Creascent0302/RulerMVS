"""V6：保留V5前端，显式分开候选族选择与族内连续定位。"""

import copy

import torch

from jar.models.metric_ruler_network import depth_cell_edges
from jar.models.ruler_proposals import transport_proposals
from jar.pretraining.fast_spatial import install_spatial_layout
from jar.pretraining.model import CalibratedDecoder
from research.rulermvs_v5_20261031.model import V5Model
from .group_feature import pool_selected_group_features
from .handoff import HierarchicalHandoff


class V6Decoder(CalibratedDecoder):
    def __init__(self, settings, options):
        super().__init__(settings)
        self.handoff_options = dict(options)
        if options.get("within_group", "mean") not in ("mean", "winner"):
            raise ValueError("within_group须为mean或winner")
        if options.get("upsample_feature", "winner") not in ("winner", "group_mean"):
            raise ValueError("upsample_feature须为winner或group_mean")
        self.handoff_enabled = bool(options.get("enabled", True))
        self.handoff = HierarchicalHandoff(
            mode=options.get("mode", "parent"), hidden=16,
            gate_bias=float(options.get("gate_bias", -2.)))
        if self.use_scene or self.proposal_transport == "nearest":
            raise ValueError("V6沿用V5无scene的四邻域候选传递")

    def forward(self, inputs):
        if not self.handoff_enabled:
            return super().forward(inputs)
        prepared = self._prepare(inputs)
        batch = prepared["matching"].shape[0]
        height, width = prepared["image_size"]
        candidates = self.depth_lattice[None, :, None, None].expand(
            batch, -1, height // 8, width // 8)
        edges = depth_cell_edges(self.depth_lattice)
        half_width = (edges[1:] - edges[:-1]) * .5
        observation = self._observe(prepared, candidates, 8, half_width)
        features, _, valid, similarity, support, _ = self._tokens(
            prepared, observation, candidates, 8)
        if self.local_kind != "scale_space":
            raise ValueError("V6当前以已验证V5 scale_space局部模块为底座")
        features = features + self.local(features, valid) * valid[:, None]
        coarse = self._prediction(
            features, candidates, valid, similarity, support,
            edges[:-1][None, :, None, None], edges[1:][None, :, None, None], observation)
        fine_candidates, fine_width, inherited, parent = transport_proposals(
            self, coarse, features, observation, prepared, return_parent=True)
        fine_observation = self._observe(prepared, fine_candidates, 4, fine_width)
        fine_features, _, fine_valid, similarity, support, source = self._tokens(
            prepared, fine_observation, fine_candidates, 4)
        fine_features = self.entry_norm(
            fine_features + self.fine_inherit(inherited)) * fine_valid[:, None]
        fine = self._prediction(
            fine_features, fine_candidates, fine_valid, similarity, support,
            fine_candidates - fine_width, fine_candidates + fine_width,
            fine_observation, use_offset=self.use_fine_offset)
        # 细层只用已计算的证据调整候选族分布，不增加投影、图像编码或3D卷积。
        null = source["null_mass"].mean(2)
        available = fine_observation["source_valid"].float().mean((1, 2))[:, None]
        result = self.handoff(fine["scores"], fine["valid"], fine["candidate_depth"],
                              parent, null, available)
        if self.handoff_options.get("within_group", "mean") == "winner":
            # 保留每组已有的最佳连续点。其可选集合包含V5原赢家，
            # 避免未经校准的组三点均值先损坏候选供给；只学习组间选择。
            within = fine["scores"].unflatten(1, (-1, 3)).argmax(2, keepdim=True)
            depths = fine["candidate_depth"].unflatten(1, (-1, 3)).gather(2, within)[:, :, 0]
            result["group_depth"] = depths
            result["depth"] = depths.gather(1, result["selected_group"]) * result["valid"]
        fine.update(scores=result["scores"], probability=result["joint_probability"],
                    depth=result["depth"], selected_index=result["selected_index"],
                    confidence=result["confidence"], group_depth=result["group_depth"],
                    group_probability=result["group_probability"],
                    selected_group=result["selected_group"], parent_probability=parent,
                    handoff_gate=result["gate"])
        if self.handoff_options.get("upsample_feature", "winner") == "group_mean":
            fine["conditional_probability"] = result["conditional_probability"]
        reference = prepared["reference"]
        image_mask = prepared["image_mask"][:, reference:reference + 1]
        output = self._render(fine, fine_features, image_mask, prepared)
        output.update(coarse=coarse, fine=fine, scene=None, reference_index=reference,
                      continuous_offset_used=True,
                      support_semantics="几何可采样源数，不是表面可见性认证")
        return output

    def _render(self, fine, features, image_mask, prepared):
        if not self.handoff_enabled or self.handoff_options.get("upsample_feature", "winner") == "winner":
            return super()._render(fine, features, image_mask, prepared)
        selected = pool_selected_group_features(
            features, fine["selected_group"], fine["conditional_probability"], fine["valid"])
        return self._upsample(fine, features, image_mask, selected_features=selected)


class V6Model(V5Model):
    architecture = "rulermvs_v6"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        original = self.decoder.state_dict()
        with torch.random.fork_rng(devices=[]):
            self.decoder = V6Decoder(config["model"], config.get("handoff", {}))
        state = self.decoder.state_dict()
        state.update(original)
        self.decoder.load_state_dict(state, strict=True)
        if config["model"].get("spatial_layout", False):
            install_spatial_layout(self)
        self.model_name = "RulerMVS-V6"

    def load_v5_state(self, state):
        expected = {key for key in self.state_dict() if not key.startswith("decoder.handoff.")}
        if set(state) != expected:
            raise ValueError("V6只接受完整V5底座参数，不静默忽略不兼容键")
        self.load_state_dict({**self.state_dict(), **state}, strict=True)


def build_from_checkpoint(saved, variant="parent"):
    """研究入口显式选择V5对照或V6；不会覆盖源权重。"""
    config = copy.deepcopy(saved["config"])
    if variant == "v5":
        model = V5Model(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
    elif saved["architecture"] == "rulermvs_v6":
        model = V6Model(config, initialize=False)
        model.load_state_dict(saved["model"], strict=True)
    else:
        if saved["architecture"] != "rulermvs_v5":
            raise ValueError("V6增量实验须由V5权重初始化")
        mode = "parent" if variant == "parent_winner" else variant
        config.update(architecture="rulermvs_v6", handoff={
            "mode": mode, "enabled": True,
            "within_group": "winner" if variant == "parent_winner" else "mean"})
        model = V6Model(config, initialize=False)
        model.load_v5_state(saved["model"])
    return model, config
