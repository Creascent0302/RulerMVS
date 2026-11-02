"""MF-ET-JAR 的基础图像 token 分层射线解码器。

每层显式读取相应编码尺度的 token，继承父 token / 深度，再在已知相机
约束的候选位置做局部 Transformer 交叉注意力。所有候选精确投影；投影
导数只决定局部搜索步长。浅层新生 query 仍保留全深度候选，可反驳父层。
这是一套可训练研究基线；匹配证据不是严格遮挡真值，也不保证最佳性能。
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from jar.common import make_scene
from jar.geometry import (
    camera_centers,
    image_texture,
    pixel_grid,
    project,
    sample_maps,
    safe_masked_softmax,
)
from jar.geometry import reference_rays
from jar.fusion import enforce_roi, spatial_hash_fuse
from jar.depth_sampling import configured_inverse_depths


def _reference_sample(maps, uv, refs, stride=1):
    """Mixed-camera samples with the project's stride*j pixel convention."""
    sampled = sample_maps(maps, uv[None].expand(len(maps), -1, -1), stride)
    return sampled[refs, torch.arange(len(uv), device=uv.device)]


def _patch_maps(rgb, dilation=1):
    """3x3 RGB patches in physical image pixels, not learned texture evidence.

    Centering each colour independently removes local additive brightness;
    normalizing the whole patch removes a common positive contrast factor.
    Replicate padding avoids artificial texture on constant images. The raw
    gradient test below separately rejects genuinely constant image regions.
    dilation 只改变原图采样间距，保持 27 维描述子不变。
    """
    n, c, h, w = rgb.shape
    padded = F.pad(rgb.float(), (dilation,) * 4, mode="replicate")
    patches = F.unfold(padded, kernel_size=3, dilation=dilation).reshape(n, c, 9, h, w)
    patches = patches - patches.mean(2, keepdim=True)
    return patches.flatten(1, 2)


def _census_patch_maps(rgb, dilation=1):
    """Raw 5x5 gray samples. The nonlinear Census transform is applied later.

    Sampling precomputed binary/soft Census features is not equivalent to
    applying Census after subpixel image sampling. Only the linear grayscale
    conversion and extraction of integer-offset samples are cached here.
    dilation=2 使用 [-4,-2,0,2,4] 像素偏移，仍为 25 维且中心编号为 12。
    """
    n, _, h, w = rgb.shape
    luma = rgb.new_tensor([0.299, 0.587, 0.114], dtype=torch.float32)
    gray = (rgb.float() * luma[None, :, None, None]).sum(1, keepdim=True)
    padded = F.pad(gray, (2 * dilation,) * 4, mode="replicate")
    return F.unfold(padded, kernel_size=5, dilation=dilation).reshape(n, 25, h, w)


def _soft_census(sampled_patch):
    """Known soft rank descriptor, not a claimed novel matching algorithm."""
    return torch.tanh((sampled_patch - sampled_patch[..., 12:13]) / 0.02)


class RayTokenTransformer(nn.Module):
    """One query attends only to its own calibrated candidate/view tokens.

    The sequence length is D*N, never the global number of scene queries.
    Softmax is performed in float32 with explicit all-invalid handling.
    """

    def __init__(
        self,
        channels,
        hidden,
        heads,
        attention=True,
        query_conditioned_matching=False,
        matching_mode="learned",
        shared_view_weights=False,
        reasoning_settings=None,
        photo_score_scale=1.0,
        pair_settings=None,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError("et_attention_heads must divide hidden_dim")
        self.hidden, self.heads, self.attention = hidden, heads, attention
        # Opt-in research variant; no new parameter/checkpoint keys. The base
        # model leaves query/key interaction in token updating only.
        self.query_conditioned_matching = query_conditioned_matching
        if matching_mode not in ("learned", "photo_mean", "photo_consensus", "hybrid_consensus"):
            raise ValueError(
                "et_matching_mode must be learned/photo_mean/photo_consensus/hybrid_consensus"
            )
        self.matching_mode = matching_mode
        self.shared_view_weights = shared_view_weights
        self.photo_score_scale = float(photo_score_scale)
        if not math.isfinite(self.photo_score_scale) or not 0 <= self.photo_score_scale <= 1:
            raise ValueError("et_photo_score_scale 必须位于 [0,1]")
        self.image_query = nn.Linear(channels, hidden)
        self.geometry_query = nn.Sequential(
            nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.parent_projection = nn.Linear(hidden, hidden)
        self.birth_embedding = nn.Parameter(torch.zeros(hidden))
        self.query_norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.kv = nn.Linear(channels + 8, hidden * 2)
        self.output = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 3), nn.GELU(), nn.Linear(hidden * 3, hidden)
        )
        self.reliability = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1))
        self.match_score = nn.Sequential(nn.Linear(channels + 8, 32), nn.SiLU(), nn.Linear(32, 1))
        # Start close to a calibrated image matcher, then learn its correction.
        nn.init.zeros_(self.match_score[-1].weight)
        nn.init.zeros_(self.match_score[-1].bias)
        self.log_photo_weight = nn.Parameter(torch.tensor(1.5))
        self.log_feature_weight = nn.Parameter(torch.tensor(-1.0))
        self.residual_depth = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual_depth.weight)
        nn.init.zeros_(self.residual_depth.bias)
        self.normal = nn.Linear(hidden, 3)
        self.state = nn.Linear(hidden, 3)
        self.quality = nn.Linear(hidden, 1)
        self.reasoning = None
        if reasoning_settings and reasoning_settings.get("enabled", False):
            from .candidate_reasoning import CandidateReasoning

            # Keep initialization of every pre-existing decoder layer identical
            # between the optional branch and its disabled ablation.
            with torch.random.fork_rng(devices=[]):
                self.reasoning = CandidateReasoning(hidden, reasoning_settings, heads)
        self.pair_appearance = None
        if pair_settings and pair_settings.get("enabled", False):
            from .pair_appearance import PairAppearance

            with torch.random.fork_rng(devices=[]):
                self.pair_appearance = PairAppearance(channels, pair_settings)

    def query(self, reference, metadata, parent, born, encoder_queries):
        encoded = self.image_query(reference)
        if not encoder_queries:
            encoded = encoded * 0
        return self.query_norm(
            encoded
            + self.geometry_query(metadata)
            + self.parent_projection(parent)
            + born[:, None] * self.birth_embedding
        )

    @staticmethod
    def view_curve_consensus(photo_cost, valid, depth_candidates, global_start=9):
        """Per-query source-view weights from independent inverse-depth modes.

        This is a diagnostic use of known robust view-selection ideas, not a
        claimed new correspondence theorem. Global candidates have a uniform
        inverse-depth lattice; the nine local candidates are nonuniform in this
        coordinate. Prefer the global curve to avoid giving repeated local
        hypotheses extra votes. No GT, learned confidence or depth labels enter.
        """
        if depth_candidates is None or depth_candidates.shape != photo_cost.shape[:2]:
            raise ValueError("Consensus matching requires depth_candidates[Q,D]")
        if not torch.isfinite(depth_candidates).all() or (depth_candidates <= 0).any():
            raise ValueError("Consensus candidate depths must be finite and positive")
        q, d, n = photo_cost.shape
        inverse = depth_candidates.float().reciprocal()
        valid = valid.bool()
        has_global = d - global_start >= 3
        global_mask = torch.zeros_like(valid)
        if has_global:
            global_mask[:, global_start:] = valid[:, global_start:]
        use_global = global_mask.sum(1) >= 3
        curve_mask = torch.where(use_global[:, None], global_mask, valid)
        # Estimate one physical inverse-depth bin, not one arbitrary list index.
        lattice = inverse[:, global_start:] if has_global else inverse
        ordered = lattice.detach().sort(-1).values
        gaps = (ordered[:, 1:] - ordered[:, :-1]).abs()
        positive = gaps > 1e-9
        finite_gaps = gaps.masked_fill(~positive, float("inf"))
        # nanmedian is avoided for CPU/AMP portability and all-identical grids.
        sorted_gaps = finite_gaps.sort(-1).values
        median_index = ((positive.sum(-1) - 1).clamp_min(0) // 2)[:, None]
        if sorted_gaps.shape[-1]:
            bin_width = sorted_gaps.gather(-1, median_index).squeeze(-1)
        else:
            bin_width = inverse.new_full((q,), float("inf"))
        fallback = (inverse.detach().amax(-1) - inverse.detach().amin(-1)).clamp_min(1e-6) / max(
            d - 1, 1
        )
        bin_width = torch.where(torch.isfinite(bin_width), bin_width, fallback).clamp_min(1e-7)
        safe = photo_cost.detach().masked_fill(~curve_mask, 3.0)
        winner = safe.argmin(1)
        best = safe.gather(1, winner[:, None]).squeeze(1)
        mode = inverse.detach()[:, :, None].expand(-1, -1, n).gather(1, winner[:, None]).squeeze(1)
        # A distant equally good peak makes the single-view mode ambiguous.
        distant = curve_mask & (
            (inverse[:, :, None].detach() - mode[:, None]).abs() > 1.5 * bin_width[:, None, None]
        )
        alternate = photo_cost.detach().masked_fill(~distant, 3.0).amin(1)
        margin = alternate - best
        contrast = photo_cost.detach().masked_fill(~curve_mask, 0.0).amax(1) - best
        mode_valid = (curve_mask.sum(1) >= 3) & (contrast > 1e-4) & (best < 0.6)
        mode_valid &= (~distant.any(1)) | (margin > 1e-4)
        sigma = bin_width[:, None, None] * 1.5
        separation = (mode[:, :, None] - mode[:, None]) / sigma
        # Every independent valid view counts once, irrespective of how many
        # depth samples populated its local/global curves.
        density = (torch.exp(-0.5 * separation.square()) * mode_valid[:, None]).sum(-1)
        quality = (contrast.clamp_min(0) / (contrast.clamp_min(0) + 0.01)) / (1 + best.clamp_min(0))
        representative = (density + 0.001 * quality).masked_fill(~mode_valid, -1e4).argmax(-1)
        consensus = mode[torch.arange(q, device=mode.device), representative]
        disagreement = (mode - consensus[:, None]) / (bin_width[:, None] * 1.5)
        weights = (1 / (1 + disagreement.square())) * mode_valid
        # There is no reliable per-view mode to vote with. Fall back to the
        # baseline average; this does not bypass final ambiguity/texture masks.
        weights = torch.where(mode_valid.any(-1, keepdim=True), weights, torch.ones_like(weights))
        return dict(
            weights=weights,
            inverse_mode=mode,
            mode_valid=mode_valid,
            used_global=use_global,
            consensus_inverse_depth=consensus,
            inverse_bin_width=bin_width,
            peak_margin=margin,
        )

    def forward(
        self,
        query,
        source_feature,
        metadata,
        valid,
        depth_candidates=None,
        photo_cost=None,
        view_prior=None,
        reference_feature=None,
    ):
        # [query, depth, source_view, ...]; no reference-camera self match.
        combined = torch.cat((source_feature, metadata), -1)
        reliability = torch.sigmoid(self.reliability(metadata)[..., 0]).float()
        photo, feature = metadata[..., 0], metadata[..., 1]
        # 可选消融仅缩放显式光度先验；metadata、可靠性、稳健视图权重和
        # 最终真实图像核验仍读取原始 photo。下方 attention 同样使用 score，
        # 因此该开关同时改变 token 读取权重，并非只对最终 logits 做后处理。
        score = (
            -F.softplus(self.log_photo_weight) * photo * self.photo_score_scale
            - F.softplus(self.log_feature_weight) * feature
            + 0.25 * self.match_score(combined)[..., 0].float()
        )
        if self.pair_appearance is not None and self.matching_mode in (
            "learned",
            "hybrid_consensus",
        ):
            # 原候选分数最终乘8；这里先除8，使该分支的直接logit修正
            # 保持在配置的范围内。attention共享score，也会读取此匹配证据。
            score = (
                score + self.pair_appearance(reference_feature, source_feature, metadata, valid) / 8
            )
        weights = reliability * valid
        if view_prior is not None:
            weights = weights * view_prior
        # A bounded robust influence reduces single-view occlusion outliers.
        robust = 1 / (1 + (photo.detach() / 0.35).square())
        weights = weights * robust
        if self.shared_view_weights:
            # 每条射线的同一源视图共用权重，避免候选靠压低反对它的视角获胜。
            # 不同候选的真实投影有效性仍分别检查，不伪造图外观测。
            weights = weights.amax(1, keepdim=True) * valid
        candidate_score = (score * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
        candidate_valid = valid.any(-1)
        candidate_score = candidate_score.masked_fill(~candidate_valid, -1e4)
        if self.matching_mode != "learned":
            # Explicit raw cost is supplied by the decoder. In particular the
            # photo-only diagnostic does not accidentally use metadata[0]=0
            # when the separate legacy et_photo_matching switch is disabled.
            raw_photo = photo.float() if photo_cost is None else photo_cost.float()
            if raw_photo.shape != valid.shape:
                raise ValueError("photo_cost and matching validity must both have shape [Q,D,N]")
            if self.matching_mode == "photo_mean":
                weights = valid.to(raw_photo.dtype)
                candidate_score = -(raw_photo * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
            else:
                consensus = self.view_curve_consensus(raw_photo, valid, depth_candidates)
                if self.matching_mode == "photo_consensus":
                    weights = valid * consensus["weights"][:, None]
                    candidate_score = -(raw_photo * weights).sum(-1) / weights.sum(-1).clamp_min(
                        1e-8
                    )
                else:
                    weights = weights * consensus["weights"][:, None]
                    candidate_score = (score * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
            candidate_valid = candidate_valid & (weights.sum(-1) > 0)
            candidate_score = candidate_score.masked_fill(~candidate_valid, -1e4)
        count, depths, views = valid.shape
        keys, values = self.kv(combined).chunk(2, dim=-1)
        candidate_values = values
        head_dim = self.hidden // self.heads
        query_heads = self.q(query).reshape(count, self.heads, head_dim)
        keys = keys.reshape(count, depths * views, self.heads, head_dim).transpose(1, 2)
        values = values.reshape(count, depths * views, self.heads, head_dim).transpose(1, 2)
        compatibility = None
        if self.attention or self.query_conditioned_matching:
            compatibility = (query_heads.float()[:, :, None] * keys.float()).sum(-1) / math.sqrt(
                head_dim
            )
        if self.query_conditioned_matching and self.matching_mode in (
            "learned",
            "hybrid_consensus",
        ):
            # The encoder-injected query can now directly rank distance
            # hypotheses, instead of affecting geometry only through the small
            # final residual. Bounded, fixed influence retains the RGB prior.
            agreement = compatibility.tanh().mean(1).reshape(count, depths, views)
            correction = 0.25 * (agreement * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)
            candidate_score = (candidate_score + correction).masked_fill(~candidate_valid, -1e4)
        if self.attention:
            logits = compatibility
            logits = logits + score.flatten(1)[:, None] * 4
            logits = logits + reliability.clamp_min(1e-5).log().flatten(1)[:, None]
            attention = safe_masked_softmax(logits, valid.flatten(1)[:, None], -1)
        else:
            # Geometry-aware weighted pooling is the attention ablation.
            attention = safe_masked_softmax(
                score.flatten(1)[:, None] * 4, valid.flatten(1)[:, None], -1
            ).expand(-1, self.heads, -1)
        context = (attention[..., None] * values.float()).sum(-2).reshape(count, self.hidden)
        any_valid = valid.flatten(1).any(-1)
        features = query + self.output(context.to(query.dtype)) * any_valid[:, None]
        features = features + self.ffn(self.norm(features))
        # Candidate logits stay independent of token count within each view.
        logits = candidate_score * 8
        if self.reasoning is not None:
            # Crucially, classification reads the UPDATED query and value
            # branch. Its supervision therefore trains output/ffn as well as
            # the keys used by the original attention compatibility score.
            correction = self.reasoning(
                features, candidate_values, metadata, weights, valid, depth_candidates
            )
            logits = logits + correction * candidate_valid
        return features, logits, candidate_valid, reliability


class EncoderTokenHierarchyDecoder(nn.Module):
    """Four-scale image-query hierarchy with calibrated sparse ray attention."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.hidden = int(m.get("hidden_dim", 128))
        channels = int(m.get("feature_dim", 64))
        self.strides = (32, 16, 8, 4)
        self.query_budget = int(m.get("max_queries", 8192))
        self.stage_fractions = tuple(m.get("et_stage_fractions", (0.125, 0.25, 0.5, 1.0)))
        self.step_pixels = tuple(float(x) for x in m.get("et_step_pixels", (1.0, 1.0, 1.0, 1.0)))
        self.chunk = int(m.get("et_query_chunk", 256))
        self.global_count = int(m.get("et_global_candidates", 24))
        self.near, self.far = float(m.get("near", 2.0)), float(m.get("far", 80.0))
        self.min_step = float(m.get("refinement_min_step", 0.01))
        self.max_step = float(m.get("refinement_max_step", 4.0))
        self.encoder_queries = bool(m.get("et_encoder_queries", True))
        self.calibration_step = bool(m.get("et_calibration_step", True))
        self.detail_births = bool(m.get("et_detail_births", True))
        self.global_search = bool(m.get("et_global_search", True))
        self.parent_children = bool(m.get("et_parent_children", False))
        self.parent_fraction = float(m.get("et_parent_fraction", 0.75))
        self.edge_queries = bool(m.get("et_edge_queries", False))
        self.edge_ruler = bool(m.get("et_edge_ruler", False))
        self.protect_parent = bool(m.get("et_protect_parent", False))
        self.infinity_guard = bool(m.get("et_infinity_guard", False))
        self.shared_view_weights = bool(m.get("et_shared_view_weights", False))
        self.curve_context = m.get("et_curve_context", {})
        if self.curve_context.get("enabled", False):
            strength = float(self.curve_context.get("strength", 0.5))
            if not math.isfinite(strength) or strength < 0:
                raise ValueError("et_curve_context.strength 必须有限且非负")
            if (
                self.curve_context.get("radius", 32) <= 0
                or self.curve_context.get("neighbors", 8) < 1
            ):
                raise ValueError("曲线传递的邻居半径与数量必须为正")
            if any(stage not in range(4) for stage in self.curve_context.get("stages", [3])):
                raise ValueError("曲线传递仅有0..3四个解码阶段")
            if not self.global_search:
                raise ValueError("曲线传递需要启用全局深度候选")
            if cfg.get("camera", {}).get("mode") == "wide4_tele2":
                raise ValueError("曲线传递目前只验证共享深度范围的广角相机，尚未支持多焦距分段候选")
        self.pretrained_matching_weight = float(m.get("pretrained_matching", {}).get("weight", 0.0))
        if not 0 <= self.pretrained_matching_weight <= 1:
            raise ValueError("pretrained_matching.weight须在[0,1]")
        self.parent_accept_slack = float(m.get("et_parent_accept_slack", 0.0))
        if not 0 <= self.parent_fraction <= 1 or self.parent_accept_slack < 0:
            raise ValueError("parent_fraction须在[0,1]，parent_accept_slack须非负")
        self.photo_matching = bool(m.get("et_photo_matching", True))
        self.photo_descriptor = str(m.get("et_photo_descriptor", "zncc3"))
        if self.photo_descriptor not in ("zncc3", "census5"):
            raise ValueError("et_photo_descriptor must be 'zncc3' or 'census5'")
        self.patch_dilation = int(m.get("et_patch_dilation", 1))
        if self.patch_dilation < 1:
            raise ValueError("et_patch_dilation须为正整数")
        self.patch_radius = (1 if self.photo_descriptor == "zncc3" else 2) * self.patch_dilation
        self.exploration = float(m.get("query_explore", 0.1))
        self.texture_threshold = float(m.get("texture_threshold", 0.002))
        self.photo_threshold = float(m.get("et_photo_threshold", 0.60))
        self.contrast_threshold = float(m.get("et_min_cost_contrast", 1e-4))
        self.uniqueness_threshold = float(m.get("et_uniqueness_margin", 0.02))
        self.min_support = int(m.get("min_support", 2))
        self.free_space_source = m.get("et_free_space_source", "coarse")
        if self.free_space_source not in ("coarse", "verified_queries"):
            raise ValueError("et_free_space_source must be coarse or verified_queries")
        self.polish_stages = tuple(m.get("et_polish_stages", ()))
        if any(i not in range(4) for i in self.polish_stages):
            raise ValueError("et_polish_stages contains decoder indices 0..3")
        self.polisher = None
        if self.polish_stages:
            from jar.models.ray_polish import CalibratedRayPolisher

            self.polisher = CalibratedRayPolisher(cfg)
        self.oriented_stages = tuple(m.get("et_oriented_polish_stages", ()))
        if any(i not in range(4) for i in self.oriented_stages):
            raise ValueError("et_oriented_polish_stages contains decoder indices 0..3")
        if set(self.oriented_stages) & set(self.polish_stages):
            raise ValueError("choose one local polisher per decoder stage")
        self.oriented_normal_source = m.get("et_oriented_normal_source", "predicted")
        if self.oriented_normal_source not in ("predicted", "rig_up", "fronto"):
            raise ValueError("et_oriented_normal_source must be predicted/rig_up/fronto")
        self.oriented_polisher = None
        if self.oriented_stages:
            from jar.models.oriented_polish import OrientedRayPolisher

            self.oriented_polisher = OrientedRayPolisher(cfg)
        self.query_policy = m.get("et_query_policy", "legacy")
        self.roi_proposals = bool(m.get("et_roi_proposals", False))
        self.query_policy_stages = tuple(m.get("et_query_policy_stages", (3,)))
        if self.query_policy not in ("legacy", "phase_grid", "coverage_edge_pairs"):
            raise ValueError("unknown et_query_policy")
        if any(i not in range(4) for i in self.query_policy_stages):
            raise ValueError("et_query_policy_stages contains decoder indices 0..3")
        self.fusion_cell_size = float(m.get("fusion_cell_size", 0.05))
        self.roi = m.get("roi", [[0.0, 64.0], [-24.0, 24.0], [-2.0, 6.0]])
        if self.query_budget < 1 or self.chunk < 1 or self.global_count < 2:
            raise ValueError("ET query budget/chunk must be positive and global candidates >= 2")
        if len(self.stage_fractions) != 4 or not all(0 < x <= 1 for x in self.stage_fractions):
            raise ValueError("et_stage_fractions must contain four fractions in (0,1]")
        if (
            sorted(self.stage_fractions) != list(self.stage_fractions)
            or self.stage_fractions[-1] != 1
        ):
            raise ValueError("ET stage fractions must be nondecreasing and end at 1")
        if len(self.step_pixels) != 4 or not all(
            math.isfinite(x) and x > 0 for x in self.step_pixels
        ):
            raise ValueError("et_step_pixels must contain four finite positive feature-pixel steps")
        if not 0 < self.near < self.far or not 0 < self.min_step <= self.max_step:
            raise ValueError("ET depth and step bounds must be strictly positive and increasing")
        if not 0 <= self.exploration <= 1 or self.min_support < 2:
            raise ValueError("ET query exploration must be [0,1] and min_support >= 2")
        bounds = torch.as_tensor(self.roi)
        if bounds.shape != (3, 2) or not bool((bounds[:, 1] > bounds[:, 0]).all()):
            raise ValueError("roi must contain three increasing [minimum, maximum] pairs")
        heads = int(m.get("et_attention_heads", 4))
        reasoning_settings = dict(m.get("candidate_reasoning", {}))
        reasoning_settings.update(near=self.near, far=self.far)
        self.blocks = nn.ModuleList(
            [
                RayTokenTransformer(
                    channels,
                    self.hidden,
                    heads,
                    bool(m.get("et_attention", True)),
                    bool(m.get("et_query_conditioned_matching", False)),
                    m.get("et_matching_mode", "learned"),
                    self.shared_view_weights,
                    reasoning_settings,
                    m.get("et_photo_score_scale", 1.0),
                    m.get("pair_appearance", {}),
                )
                for _ in self.strides
            ]
        )
        classes = int(m.get("semantic_classes", 0))
        self.semantic = (
            nn.ModuleList([nn.Linear(self.hidden, classes) for _ in self.strides])
            if classes
            else None
        )
        self.register_buffer("offsets", torch.arange(-4.0, 5.0), persistent=False)
        self.register_buffer(
            "global_inverse_depth",
            configured_inverse_depths(m, self.global_count),
            persistent=False,
        )
        settings = m.get("hypothesis_context", {})
        self.hypothesis_stages = tuple(settings.get("stages", [1, 2, 3]))
        if settings.get("enabled", False):
            if not self.global_search or cfg.get("camera", {}).get("mode") == "wide4_tele2":
                raise ValueError("候选上下文目前要求广角共享的全局米制候选")
            if not self.hypothesis_stages or any(i not in range(4) for i in self.hypothesis_stages):
                raise ValueError("hypothesis_context.stages 必须是非空的0..3阶段集合")
            from .hypothesis_context import HypothesisContext

            # 新增模块不得推进旧层或数据采样器使用的随机数状态。
            with torch.random.fork_rng(devices=[]):
                self.hypothesis_context = HypothesisContext(cfg)

    def _describe_patch(self, sampled):
        if self.photo_descriptor == "census5":
            return _soft_census(sampled)
        return F.normalize(sampled, dim=-1, eps=1e-6)

    def _descriptor_cost(self, sample, reference):
        if self.photo_descriptor == "census5":
            # The centre comparison is identically zero; average over the
            # other 24 entries as in the standalone soft Census baseline.
            return ((sample - reference).abs().sum(-1) / 24).clamp(0, 2)
        return (1 - (sample * reference).sum(-1)).clamp(0, 2)

    def _select_births(self, inputs, pyramid, coarse, b, stride, count, old_refs=None, old_uv=None):
        """Predictive image evidence and coverage quotas, never labels.

        Existing anchors survive. Finer grids provide new image positions that
        did not exist at coarse resolution. Selection is intentionally discrete;
        selected encoder features are differentiable in the subsequent decoder.
        """
        rgb = inputs["rgb"][b]
        n, _, height, width = rgb.shape
        hh, ww = pyramid[stride].shape[-2:]
        uv = pixel_grid(hh, ww, rgb.device, stride=stride).reshape(-1, 2)
        inside = (uv[:, 0] < width) & (uv[:, 1] < height)
        available = inputs.get(
            "view_mask", torch.ones(inputs["rgb"].shape[:2], device=rgb.device, dtype=torch.bool)
        )[b]
        views = torch.nonzero(available, as_tuple=False).flatten()
        if len(views) == 0:
            views = torch.zeros(1, device=rgb.device, dtype=torch.long)
        texture = inputs["_texture"][b] if "_texture" in inputs else image_texture(rgb)
        edge = sample_maps(texture, uv[None].expand(n, -1, -1))[..., 0]
        confidence = sample_maps(
            coarse["confidence"][b, :, None].float(), uv[None].expand(n, -1, -1), 16
        )[..., 0]
        # Encoder contrast is a proposal cue only; it cannot certify 3-D hits.
        fmap = pyramid[stride][b].float()
        smooth = F.avg_pool2d(F.pad(fmap, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        contrast = (fmap - smooth).square().mean(1).sqrt().flatten(1).detach()
        result_uv, result_ref = [], []
        for number, view in enumerate(views):
            quota = count // len(views) + int(number < count % len(views))
            if quota == 0:
                continue
            valid = inside.clone()
            if self.roi_proposals:
                from jar.query_selection import ray_roi_mask

                reference_ids = torch.full(
                    (len(uv),), int(view), device=uv.device, dtype=torch.long
                )
                valid &= ray_roi_mask(inputs, b, reference_ids, uv, self.roi, self.near, self.far)
            if "image_mask" in inputs:
                valid &= (
                    sample_maps(inputs["image_mask"][b, view][None, None].float(), uv[None])[
                        0, :, 0
                    ]
                    > 0.999
                )
            if old_refs is not None and len(old_refs):
                old = old_uv[old_refs == view]
                # Prior anchors are multiples of the current finer stride.
                old_xy = (old / stride).round().long()
                key = old_xy[:, 1] * ww + old_xy[:, 0]
                valid[key.clamp(0, len(valid) - 1)] = False
            choices = torch.nonzero(valid, as_tuple=False).flatten()
            quota = min(quota, len(choices))
            if not quota:
                continue
            texture_score = edge[view, choices] / edge[view, choices].amax().clamp_min(1e-8)
            feature_score = contrast[view, choices] / contrast[view, choices].amax().clamp_min(1e-8)
            score = 0.60 * texture_score + 0.15 * feature_score + 0.25 * confidence[view, choices]
            explore_count = max(1, min(quota, int(math.ceil(quota * self.exploration))))
            selected = torch.zeros(len(choices), device=rgb.device, dtype=torch.bool)
            spread = (
                torch.linspace(0, len(choices) - 1, explore_count, device=rgb.device).round().long()
            )
            selected[spread] = True
            remain = quota - int(selected.sum())
            if remain:
                selected[
                    score.detach().masked_fill(selected, -float("inf")).topk(remain).indices
                ] = True
            chosen = choices[selected]
            result_uv.append(uv[chosen])
            result_ref.append(
                torch.full((len(chosen),), int(view), device=rgb.device, dtype=torch.long)
            )
        if not result_uv:
            return torch.empty(0, 2, device=rgb.device), torch.empty(
                0, dtype=torch.long, device=rgb.device
            )
        return torch.cat(result_uv), torch.cat(result_ref)

    @staticmethod
    def _parents(uv, refs, previous, previous_stride, image_hw):
        """O(image cells + queries) parent lookup, no dense pairwise attention."""
        height, width = image_hw
        hh, ww = math.ceil(height / previous_stride), math.ceil(width / previous_stride)
        views = int(torch.maximum(refs.max(), previous["refs"].max())) + 1
        table = torch.full((views * hh * ww,), -1, device=uv.device, dtype=torch.long)
        oldxy = (previous["uv"] / previous_stride).round().long()
        oldxy[:, 0].clamp_(0, ww - 1)
        oldxy[:, 1].clamp_(0, hh - 1)
        keys = (previous["refs"] * hh + oldxy[:, 1]) * ww + oldxy[:, 0]
        # Several inherited off-grid anchors can map into a parent bin. Use the
        # first stable index, not nondeterministic duplicate CUDA assignment.
        initial = torch.full_like(table, len(previous["uv"]))
        initial.scatter_reduce_(
            0, keys, torch.arange(len(keys), device=uv.device), reduce="amin", include_self=True
        )
        table = torch.where(initial < len(keys), initial, table)
        xy = (uv / previous_stride).round().long()
        xy[:, 0].clamp_(0, ww - 1)
        xy[:, 1].clamp_(0, hh - 1)
        return table[(refs * hh + xy[:, 1]) * ww + xy[:, 0]]

    def _stage_queries(self, inputs, pyramid, coarse, b, stage, previous):
        rgb = inputs["rgb"][b]
        budget = max(1, int(self.query_budget * self.stage_fractions[stage]))
        old_count = 0 if previous is None else len(previous["uv"])
        birth_count = budget - old_count if previous is None or self.detail_births else 0
        old_refs = None if previous is None else previous["refs"]
        old_uv = None if previous is None else previous["uv"]
        edge_level = inputs.get("_edge_levels", [{}] * len(inputs["rgb"]))[b].get(
            self.strides[stage]
        )
        children = None
        if self.parent_children and previous is not None:
            from jar.hierarchy_queries import child_queries

            children = child_queries(
                inputs,
                b,
                previous,
                int(max(birth_count, 0) * self.parent_fraction),
                self.strides[stage],
                edge_level if self.edge_queries else None,
                self.roi if self.roi_proposals else None,
                self.near,
                self.far,
            )
            birth_count -= len(children["uv"])
            old_refs = torch.cat((old_refs, children["refs"]))
            old_uv = torch.cat((old_uv, children["uv"]))
        if self.edge_queries or (
            self.query_policy != "legacy" and stage in self.query_policy_stages
        ):
            from jar.query_selection import select_query_candidates

            proposals = select_query_candidates(
                inputs,
                b,
                max(birth_count, 0),
                old_refs=old_refs,
                old_uv=old_uv,
                policy="coverage_edge_pairs" if self.edge_queries else self.query_policy,
                stride=self.strides[stage],
                coverage_fraction=self.exploration,
                roi=self.roi if self.roi_proposals else None,
                near=self.near,
                far=self.far,
                edge_level=edge_level if self.edge_queries else None,
                edge_cell_size=max(4, self.strides[stage]) if self.edge_queries else 8,
            )
            uv, refs = proposals["ref_uv"], proposals["ref_index"]
        else:
            uv, refs = self._select_births(
                inputs,
                pyramid,
                coarse,
                b,
                self.strides[stage],
                max(birth_count, 0),
                old_refs,
                old_uv,
            )
        child_parent = None
        if children is not None:
            child_parent = torch.cat((children["parent_id"], torch.full_like(refs, -1)))
            uv = torch.cat((children["uv"], uv))
            refs = torch.cat((children["refs"], refs))
        if previous is None and not len(uv):
            # Fully masked images still produce a finite unknown placeholder.
            uv = rgb.new_tensor([[0.0, 0.0]])
            refs = torch.zeros(1, device=rgb.device, dtype=torch.long)
        if len(uv) and self.parent_children:
            # 初始深度不跨像素平均前景/背景的不同峰。
            sampled = sample_maps(
                coarse["depth"][b, :, None].float(),
                uv[None].expand(len(rgb), -1, -1),
                16,
                mode="nearest",
                padding_mode="border",
            )
            initial_depth = sampled[refs, torch.arange(len(uv), device=uv.device), 0]
        elif len(uv):
            initial_depth = _reference_sample(coarse["depth"][b, :, None].float(), uv, refs, 16)[
                :, 0
            ]
        else:
            initial_depth = rgb.new_empty(0)
        initial_depth = initial_depth.clamp(self.near, self.far)
        parent = rgb.new_zeros((len(uv), self.hidden))
        parent_id = torch.full((len(uv),), -1, device=rgb.device, dtype=torch.long)
        if previous is not None and len(uv):
            parent_id = (
                child_parent
                if child_parent is not None
                else self._parents(uv, refs, previous, self.strides[stage - 1], rgb.shape[-2:])
            )
            has_parent = parent_id >= 0
            parent = torch.where(
                has_parent[:, None], previous["features"][parent_id.clamp_min(0)], parent
            )
            initial_depth = torch.where(
                has_parent, previous["depth"][parent_id.clamp_min(0)], initial_depth
            )
            if self.parent_children:
                from jar.hierarchy_queries import inherit_plane_depth

                inherited = inherit_plane_depth(
                    inputs, b, uv, refs, parent_id.clamp_min(0), previous, self.near, self.far
                )
                initial_depth = torch.where(has_parent, inherited, initial_depth)
        parent_valid = torch.zeros(len(uv), device=rgb.device, dtype=torch.bool)
        if previous is not None:
            # 传统基线可只重放查询位置，不提供匹配证据；此时不能视作有效父表面。
            previous_valid = previous.get(
                "valid", torch.zeros(old_count, device=rgb.device, dtype=torch.bool)
            )
            parent_valid = (parent_id >= 0) & previous_valid[parent_id.clamp_min(0)]
        born = torch.ones(len(uv), device=rgb.device, dtype=torch.bool)
        if previous is not None:
            uv = torch.cat((previous["uv"], uv))
            refs = torch.cat((previous["refs"], refs))
            parent = torch.cat((previous["features"], parent))
            initial_depth = torch.cat((previous["depth"], initial_depth))
            parent_id = torch.cat((torch.arange(old_count, device=rgb.device), parent_id))
            born = torch.cat((torch.zeros(old_count, device=rgb.device, dtype=torch.bool), born))
            parent_valid = torch.cat((previous_valid, parent_valid))
        return dict(
            uv=uv,
            refs=refs,
            depth=initial_depth,
            parent=parent,
            born=born,
            parent_id=parent_id,
            parent_valid=parent_valid,
        )

    def _match_chunk(self, inputs, pyramid, patches, patch_mask, texture, b, stage, query):
        rgb, K, T = inputs["rgb"][b], inputs["K"][b], inputs["T_cam_from_rig"][b]
        stride, block = self.strides[stage], self.blocks[stage]
        fmap = pyramid[stride][b]
        uv, refs, depth = query["uv"], query["refs"], query["depth"].float()
        n, _, height, width = rgb.shape
        view_mask = inputs.get("view_mask")
        view_mask = None if view_mask is None else view_mask[b]
        image_mask = inputs.get("image_mask")
        image_mask = None if image_mask is None else image_mask[b]
        centers, rays, _ = reference_rays(torch.zeros(len(uv), 3, device=uv.device), refs, uv, K, T)
        points_before = centers + rays * depth[:, None]
        base_uv, _, base_valid, jacobian = project(
            points_before, K, T, (height, width), view_mask, image_mask
        )
        # Original pixels/metre -> level pixels/metre; exact ray direction uses
        # optical depth, not unit-length Euclidean ray distance.
        motion = (jacobian.float() * rays[None, :, None]).sum(-1)
        g = motion.norm(dim=-1) / self._source_sampling_stride(
            inputs, b, points_before, refs, stride, jacobian
        )
        edge_level = inputs.get("_edge_levels", [{}] * len(inputs["rgb"]))[b].get(stride)
        if self.edge_ruler and edge_level is not None:
            from jar.edge_evidence import edge_motion_observability

            observation = edge_motion_observability(edge_level, base_uv, motion)
            # 沿边缘滑动的视图权重下降；普通纹理仍保留正权重，不硬判unknown。
            g = g * (0.25 + 0.75 * observation)
        source = torch.arange(n, device=uv.device)[:, None] != refs[None]
        # Reference pixels are exact input anchors. Re-projecting a boundary
        # anchor can round 0 to -1e-6 and make chunk-size-dependent masks.
        ref_valid = (
            (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        )
        if view_mask is not None:
            ref_valid &= view_mask[refs].bool()
        if image_mask is not None:
            ref_valid &= _reference_sample(image_mask[:, None].float(), uv, refs)[:, 0] > 0.999
        ref_valid &= _reference_sample(patch_mask, uv, refs)[:, 0] > 0.999
        informative = base_valid & source & (g > 1e-6) & ref_valid[None]
        ruler = g.masked_fill(~informative, 0).amax(0)
        step = (self.step_pixels[stage] / ruler.clamp_min(1e-8)).clamp(self.min_step, self.max_step)
        if not self.calibration_step:
            step = torch.full_like(step, 0.25).clamp(self.min_step, self.max_step)
        candidates = self._depth_candidates(inputs, b, query, step)
        count, candidate_count = candidates.shape
        positions = centers[:, None] + rays[:, None] * candidates[..., None]
        projected, _, valid, J = project(
            positions.reshape(-1, 3), K, T, (height, width), view_mask, image_mask
        )
        source_stride = self._source_sampling_stride(
            inputs,
            b,
            positions.reshape(-1, 3),
            refs[:, None].expand(-1, candidate_count).reshape(-1),
            stride,
            J,
        )
        if torch.is_tensor(source_stride):
            source_stride = source_stride.reshape(n, count, candidate_count)
        exact_g = (
            J.reshape(n, count, candidate_count, 2, 3).float() * rays[None, :, None, None]
        ).sum(-1).norm(dim=-1) / source_stride
        valid = (
            valid.reshape(n, count, candidate_count)
            & source[..., None]
            & (exact_g > 1e-6)
            & ref_valid[None, :, None]
        )
        valid &= (
            sample_maps(patch_mask, projected)[..., 0].reshape(n, count, candidate_count) > 0.999
        )
        if not self.global_search:
            valid[..., 9:] = False
        sampled = (
            sample_maps(fmap, projected, stride)
            .reshape(n, count, candidate_count, -1)
            .permute(1, 2, 0, 3)
        )
        reference = _reference_sample(fmap, uv, refs, stride)
        reference_norm = F.normalize(reference.float(), dim=-1, eps=1e-6)
        sample_norm = F.normalize(sampled.float(), dim=-1, eps=1e-6)
        feature_cost = (1 - (sample_norm * reference_norm[:, None, None]).sum(-1)).clamp(0, 2)
        ref_patch = self._describe_patch(_reference_sample(patches, uv, refs).float())
        sampled_patch = (
            sample_maps(patches, projected)
            .reshape(n, count, candidate_count, -1)
            .permute(1, 2, 0, 3)
        )
        sampled_patch = self._describe_patch(sampled_patch)
        photo_cost = self._descriptor_cost(sampled_patch, ref_patch[:, None, None])
        routed = self._candidate_observations(inputs, pyramid, b, stride, positions, refs, uv)
        if routed is not None:
            sampled, reference, photo_cost = (
                routed["sampled"],
                routed["reference"],
                routed["photo_cost"],
            )
            feature_cost = (
                1
                - (
                    F.normalize(sampled.float(), dim=-1, eps=1e-6)
                    * F.normalize(reference.float(), dim=-1, eps=1e-6)[:, None, None]
                ).sum(-1)
            ).clamp(0, 2)
            valid = valid & routed["valid"].permute(2, 0, 1)
        if self.pretrained_matching_weight > 0:
            from jar.pretrained_matching import pretrained_cosine_cost

            external_cost = pretrained_cosine_cost(
                inputs["_pretrained_matching_features"][b], uv, refs, projected, candidate_count
            )
            feature_cost = torch.lerp(feature_cost, external_cost, self.pretrained_matching_weight)
        texture_sample = (
            sample_maps(texture, projected)[..., 0]
            .reshape(n, count, candidate_count)
            .permute(1, 2, 0)
        )
        ref_texture = _reference_sample(texture, uv, refs)[:, 0]
        valid = valid.permute(1, 2, 0)
        valid = (
            valid
            & (texture_sample >= self.texture_threshold)
            & (ref_texture[:, None, None] >= self.texture_threshold)
        )
        # Photometric patches are supplemental matching evidence. Turning this
        # option off leaves learned appearance and raw texture observability.
        matcher_photo = photo_cost if self.photo_matching else torch.zeros_like(photo_cost)
        view_prior = None
        if self.edge_ruler and edge_level is not None:
            candidate_motion = (
                J.float() * rays[:, None].expand(-1, candidate_count, -1).reshape(1, -1, 1, 3)
            ).sum(-1)
            observation = edge_motion_observability(edge_level, projected, candidate_motion)
            observation = observation.reshape(n, count, candidate_count).permute(1, 2, 0)
            view_prior = 0.5 + 0.5 * observation
        derivative = exact_g.permute(1, 2, 0)
        displacement = ((candidates - depth[:, None]) / step[:, None].clamp_min(0.01)).clamp(
            -20, 20
        ) / 20
        baseline = (camera_centers(T)[None] - centers[:, None]).norm(dim=-1)
        metadata = torch.stack(
            (
                matcher_photo,
                feature_cost,
                derivative.log1p().clamp_max(10) / 10,
                texture_sample.clamp_min(1e-8).log().clamp(-18, 0) / 18,
                displacement[:, :, None].expand(-1, -1, n),
                (candidates.log1p() / 5)[:, :, None].expand(-1, -1, n),
                baseline[:, None].expand(-1, candidate_count, -1).clamp_max(10) / 10,
                query["born"][:, None, None].expand(-1, candidate_count, n).float(),
            ),
            -1,
        )
        geometry = torch.cat(
            (
                uv / uv.new_tensor([max(width - 1, 1), max(height - 1, 1)]) * 2 - 1,
                depth.log1p()[:, None] / 5,
                F.normalize(rays, dim=-1),
                depth.new_full((count, 1), stride / 32),
                (query["parent_id"] >= 0)[:, None].float(),
            ),
            -1,
        )
        query_reference = reference
        if "_image_context" in inputs:
            query_reference = query_reference + _reference_sample(
                inputs["_image_context"][stride][b], uv, refs, stride
            )
        token = block.query(
            query_reference, geometry, query["parent"], query["born"].float(), self.encoder_queries
        )
        token, logits, candidate_valid, reliability = block(
            token,
            sampled,
            metadata,
            valid,
            depth_candidates=candidates,
            photo_cost=photo_cost,
            view_prior=view_prior,
            reference_feature=reference,
        )
        result = dict(
            token=token,
            logits=logits,
            candidate_valid=candidate_valid,
            candidates=candidates,
            step=step,
            ref_valid=ref_valid,
            ref_patch=ref_patch,
            ref_texture=ref_texture,
            reference=reference,
            routed=routed,
        )
        if hasattr(self, "hypothesis_context") and stage in self.hypothesis_stages:
            result["hypothesis_features"] = self.hypothesis_context.encode_views(
                sampled, metadata, valid
            )
        return result

    def _decode_chunk(
        self, inputs, pyramid, patches, patch_mask, texture, b, stage, query, match=None
    ):
        """匹配与位置更新分开，允许整层交换证据后只执行一次深度更新。"""
        if match is None:
            match = self._match_chunk(
                inputs, pyramid, patches, patch_mask, texture, b, stage, query
            )
        rgb, K, T = inputs["rgb"][b], inputs["K"][b], inputs["T_cam_from_rig"][b]
        n, _, height, width = rgb.shape
        stride, block = self.strides[stage], self.blocks[stage]
        uv, refs, depth = query["uv"], query["refs"], query["depth"].float()
        view_mask = inputs["view_mask"][b] if "view_mask" in inputs else None
        image_mask = inputs["image_mask"][b] if "image_mask" in inputs else None
        centers, rays, _ = reference_rays(torch.zeros(len(uv), 3, device=uv.device), refs, uv, K, T)
        points_before = centers + rays * depth[:, None]
        source = torch.arange(n, device=uv.device)[:, None] != refs[None]
        token, logits = match["token"], match["logits"]
        candidate_valid, candidates = match["candidate_valid"], match["candidates"]
        step, ref_valid = match["step"], match["ref_valid"]
        ref_patch, ref_texture = match["ref_patch"], match["ref_texture"]
        routed = match["routed"]
        count, candidate_count = candidates.shape
        winner = logits.detach().argmax(-1)
        qi = torch.arange(count, device=uv.device)
        best_depth = candidates[qi, winner]
        # Blend only neighbours of one selected depth mode; do not average two
        # separated foreground/background peaks. Discrete mode choice is also
        # trainable through the candidate-class supervision exposed in aux.
        difference = (
            (candidates - best_depth[:, None]).abs().masked_fill(~candidate_valid, float("inf"))
        )
        close_ids = difference.topk(min(3, candidate_count), dim=-1, largest=False).indices
        local_mode = torch.zeros_like(candidate_valid).scatter_(1, close_ids, True)
        radius = torch.maximum(step, best_depth * 0.05)
        local_mode &= (difference <= radius[:, None]) & candidate_valid
        probability = safe_masked_softmax(logits, local_mode, -1)
        proposal = (probability * candidates).sum(-1)
        raw_logits = match.get("raw_logits", logits)
        contrast = raw_logits.masked_fill(~candidate_valid, -1e4).amax(-1) - raw_logits.masked_fill(
            ~candidate_valid, 1e4
        ).amin(-1)
        evidence = candidate_valid.sum(-1).ge(2) & (contrast > self.contrast_threshold)
        # Reject a separated, nearly equivalent solution. Max-minus-min alone
        # cannot distinguish a unique match from repeated texture. This checks
        # alternatives more than one calibrated local neighbourhood away.
        separate = candidate_valid & (
            (candidates - best_depth[:, None]).abs()
            > torch.maximum(step * 2, best_depth * 0.10)[:, None]
        )
        alternate = logits.masked_fill(~separate, -1e4).amax(-1)
        peak_margin = logits[qi, winner] - alternate
        unique = separate.any(-1) & (peak_margin > self.uniqueness_threshold)
        correction = 0.5 * step * torch.tanh(block.residual_depth(token)[:, 0].float())
        new_depth = self._clamp_depth(
            torch.where(evidence, proposal + correction, depth), refs, inputs, b
        )
        polish = None
        unpolished_depth = new_depth
        if stage in self.polish_stages:
            polish = self.polisher(inputs, b, refs, uv, new_depth)
            # A local optimum does not certify the globally selected depth
            # mode. Keep global ambiguity rejection below; polish only anchors
            # that had independent candidate evidence in the first place.
            new_depth = torch.where(evidence & polish["evidence"], polish["depth"], new_depth)
        if stage in self.oriented_stages:
            patch_normal = block.normal(token).float()
            patch_normal = F.normalize(
                patch_normal + patch_normal.new_tensor([0.0, 0.0, 0.1]), dim=-1, eps=1e-6
            )
            if self.oriented_normal_source == "rig_up":
                patch_normal = torch.zeros_like(patch_normal)
                patch_normal[:, 2] = 1.0
            elif self.oriented_normal_source == "fronto":
                patch_normal = T[refs, 2, :3].float()
            polish = self.oriented_polisher(inputs, b, refs, uv, new_depth, patch_normal)
            new_depth = torch.where(evidence & polish["evidence"], polish["depth"], new_depth)
        parent_kept = torch.zeros_like(evidence)
        if self.protect_parent and "parent_valid" in query:
            # 验收最终提案；同一组源视图、同一窗口形状，不能靠丢弃困难视图获胜。
            trusted = query["parent_valid"] & ~query["born"]
            trial_depth = torch.stack((depth, new_depth), 1)
            if stage in self.oriented_stages:
                scored = self.oriented_polisher._score(
                    inputs, b, refs, uv, trial_depth, polish["final_normal"], False
                )
                costs = scored["per_view_cost"][:, :, 0].permute(2, 0, 1)
                good = scored["per_view_valid"][:, :, 0].permute(2, 0, 1)
            else:
                trial = centers[:, None] + rays[:, None] * trial_depth[..., None]
                trial_uv, _, trial_valid, _ = project(
                    trial.reshape(-1, 3), K, T, (height, width), view_mask, image_mask
                )
                descriptors = self._describe_patch(sample_maps(patches, trial_uv))
                costs = self._descriptor_cost(
                    descriptors.reshape(n, count, 2, -1), ref_patch[None, :, None]
                )
                good = trial_valid.reshape(n, count, 2)
                good &= sample_maps(patch_mask, trial_uv)[..., 0].reshape(n, count, 2) > 0.999
            good = good & source[..., None]
            same_views = good.all(-1) & (costs[:, :, 0] < self.photo_threshold)
            mean_cost = (costs * same_views[..., None]).sum(0) / same_views.sum(0).clamp_min(1)[
                :, None
            ]
            parent_kept = trusted & (
                ~same_views.any(0) | (mean_cost[:, 1] > mean_cost[:, 0] + self.parent_accept_slack)
            )
            new_depth = torch.where(parent_kept, depth, new_depth)
            evidence = evidence | parent_kept
            unique = unique | parent_kept
        points = centers + rays * new_depth[:, None]
        # Certify the actual refined position, not only a search candidate and
        # not the stride-16 coarse depth (which may have missed a thin object).
        final_uv, _, final_valid, final_j = project(
            points, K, T, (height, width), view_mask, image_mask
        )
        final_descriptor = self._describe_patch(sample_maps(patches, final_uv))
        final_cost = self._descriptor_cost(final_descriptor, ref_patch[None])
        final_texture = sample_maps(texture, final_uv)[..., 0]
        final_derivative = (final_j.float() * rays[None, :, None]).sum(-1).norm(dim=-1) / stride
        final_informative = final_valid & source & ref_valid[None] & (final_derivative > 1e-6)
        final_informative &= sample_maps(patch_mask, final_uv)[..., 0] > 0.999
        final_informative &= (final_texture >= self.texture_threshold) & (
            ref_texture[None] >= self.texture_threshold
        )
        matching = final_informative & (final_cost < self.photo_threshold)
        if polish is not None:
            # Verify using the very descriptor/warp used for polishing. If no
            # local update was accepted, retain the old exact-position check.
            updated = evidence & polish["evidence"] & ~parent_kept
            final_cost = torch.where(updated[None], polish["per_view_cost"].T, final_cost)
            final_informative = torch.where(
                updated[None],
                polish["per_view_valid"].T & source & ref_valid[None],
                final_informative,
            )
            matching = final_informative & (final_cost < self.photo_threshold)
        actual = self._candidate_observations(
            inputs,
            pyramid,
            b,
            stride,
            points[:, None],
            refs,
            uv,
            polish["final_normal"] if polish is not None and "final_normal" in polish else None,
        )
        if actual is not None:
            final_cost = actual["photo_cost"][:, 0].T
            final_informative = actual["valid"][:, 0].T & source & ref_valid[None]
            matching = final_informative & (final_cost < self.photo_threshold)
        support = matching.sum(0) + ref_valid.long()
        surface = evidence & unique & (support >= self.min_support)
        infinity = None
        if self.infinity_guard and stage == len(self.strides) - 1:
            from jar.infinity_evidence import project_infinity, finite_depth_evidence

            infinity_uv, infinity_valid = project_infinity(
                rays, K, T, (height, width), view_mask, image_mask
            )
            infinity_descriptor = self._describe_patch(sample_maps(patches, infinity_uv))
            infinity_cost = self._descriptor_cost(infinity_descriptor, ref_patch[None])
            # 不拿倾斜平面精修窗口与正面窗口混比；两种假说都使用相同原图窗口。
            finite_cost = self._descriptor_cost(final_descriptor, ref_patch[None])
            common = matching & infinity_valid
            common &= sample_maps(patch_mask, final_uv)[..., 0] > 0.999
            common &= sample_maps(patch_mask, infinity_uv)[..., 0] > 0.999
            infinity = finite_depth_evidence(finite_cost, infinity_cost, common)
            surface = surface & infinity["accepted"]
        quality = block.quality(token)[:, 0].float()
        state_logits = block.state(token)
        # Correspondence residuals on the very same raw patches are stronger
        # evidence than a network's unconstrained confidence score alone.
        measured_quality = (
            1 - (final_cost * matching).sum(0) / matching.sum(0).clamp_min(1)
        ).clamp(0, 1)
        normal = block.normal(token).float()
        normal = F.normalize(normal + normal.new_tensor([0.0, 0.0, 0.1]), dim=-1, eps=1e-6)
        sign = torch.where((normal * (points - centers)).sum(-1) > 0, -1.0, 1.0)
        normal = normal * sign[:, None]
        footprint = stride / final_j.float().square().sum((-1, -2)).sqrt().masked_fill(
            ~final_valid, 0
        ).amax(0).clamp_min(1e-6)
        footprint = footprint.clamp(0.01, 2.0)
        ev = dict(
            surface=surface,
            free=torch.zeros_like(surface),
            known=surface,
            support=torch.where(surface, support, torch.zeros_like(support)),
            confidence=measured_quality * torch.sigmoid(quality) * surface,
        )
        scene = make_scene(
            points, normal, footprint[:, None].expand(-1, 3), token, state_logits, ev
        )
        scene["confidence_logits"] = quality
        if self.semantic is not None:
            scene["semantic_logits"] = self.semantic[stage](token)
        scene["aux"] = dict(
            ref_index=refs,
            ref_uv=uv,
            reference_depth=new_depth,
            initial_points=points_before,
            points_before=points_before,
            points_after=points,
            delta_xyz=points - points_before,
            candidate_depths=candidates,
            candidate_logits=logits,
            candidate_valid=candidate_valid,
            candidate_probabilities=probability,
            matcher_support=support,
            evidence=surface,
            calibrated_step=step,
            birth_mask=query["born"],
            separated_peak_margin=peak_margin,
            unique_match=unique,
            parent_index=query["parent_id"],
            stride=stride,
            roi=self.roi,
            independent_match_cost=final_cost.T,
            attention_token_count=candidate_count * n,
        )
        if "raw_logits" in match:
            raw_margin = raw_logits[qi, winner] - raw_logits.masked_fill(~separate, -1e4).amax(-1)
            scene["aux"]["context_selected_raw_margin"] = raw_margin
            scene["aux"]["context_resolved_ambiguity"] = unique & (
                raw_margin <= self.uniqueness_threshold
            )
            scene["aux"]["context_neighbor_mass"] = match["neighbor_mass"]
        if "hypothesis_residual" in match:
            scene["aux"]["hypothesis_logit_residual"] = match["hypothesis_residual"]
        if self.parent_children or self.edge_queries or self.edge_ruler or self.protect_parent:
            scene["aux"]["parent_update_rejected"] = parent_kept
        if infinity is not None:
            for key, value in infinity.items():
                scene["aux"]["infinity_" + key] = value
        if polish is not None:
            scene["aux"]["polish_initial_depth"] = unpolished_depth
            scene["aux"]["polish_final_depth"] = new_depth
            scene["aux"]["polish_evidence"] = polish["evidence"]
            scene["aux"]["polish_applied"] = evidence & polish["evidence"] & ~parent_kept
            if stage in self.oriented_stages:
                # The window's temporary plane hypothesis is distinct from
                # the supervised output normal. Fronto fallback is a matching
                # approximation, not a forced normal label for a small object.
                scene["aux"]["oriented_patch_normal"] = polish["final_normal"]
                scene["aux"]["oriented_applied"] = evidence & polish["evidence"] & ~parent_kept
        return self._finish_chunk(scene, inputs, b, stage, query, routed)

    def _source_sampling_stride(self, inputs, b, points, refs, stride, jacobian):
        return stride

    def _depth_candidates(self, inputs, b, query, step):
        depth = query["depth"].float()
        local = (depth[:, None] + self.offsets[None] * step[:, None]).clamp(self.near, self.far)
        global_depth = self.global_inverse_depth.reciprocal()[None].expand(len(depth), -1)
        return torch.cat((local, global_depth), -1)

    def _clamp_depth(self, depth, refs, inputs, b):
        return depth.clamp(self.near, self.far)

    def _candidate_observations(self, inputs, pyramid, b, stride, points, refs, uv, normals=None):
        return None

    def _finish_chunk(self, scene, inputs, b, stage, query, routed):
        return scene

    def _finish_stage(self, output, inputs, pyramid, b, stage):
        return output

    def _context_matches(self, inputs, pyramid, patches, patch_mask, texture, b, stage, queries):
        """同层稀疏查询共享候选曲线；不重复编码、投影或跨视图匹配。"""
        from jar.models.curve_transport import curve_messages

        matches = [
            self._match_chunk(inputs, pyramid, patches, patch_mask, texture, b, stage, sub)
            for sub in queries
        ]
        uv = torch.cat([sub["uv"] for sub in queries])
        refs = torch.cat([sub["refs"] for sub in queries])
        combined = {
            key: torch.cat([match[key] for match in matches])
            for key in ("candidates", "logits", "candidate_valid", "reference")
        }
        colors = _reference_sample(inputs["rgb"][b], uv, refs)
        settings = self.curve_context
        residual = None
        if hasattr(self, "hypothesis_context") and stage in self.hypothesis_stages:
            centers, rays, _ = reference_rays(
                torch.zeros(len(uv), 3, device=uv.device),
                refs,
                uv,
                inputs["K"][b],
                inputs["T_cam_from_rig"][b],
            )
            features = torch.cat([match["hypothesis_features"] for match in matches])
            residual = self.hypothesis_context(
                features,
                combined["logits"],
                combined["candidates"],
                combined["candidate_valid"],
                uv,
                refs,
                centers,
                rays,
                stage,
            )
            combined["logits"] = combined["logits"] + residual
        neighbor_mass = uv.new_zeros(len(uv))
        if settings.get("enabled", False) and stage in settings.get("stages", [3]):
            message, affinity = curve_messages(
                uv,
                refs,
                combined["candidates"],
                combined["logits"],
                combined["candidate_valid"],
                combined["reference"],
                colors,
                radius=float(settings.get("radius", 32)),
                neighbors=int(settings.get("neighbors", 8)),
            )
            combined["logits"] = combined["logits"] + float(settings.get("strength", 0.5)) * message
            neighbor_mass = affinity.sum(-1)
        offset = 0
        for match in matches:
            count = len(match["logits"])
            match["raw_logits"] = match["logits"]
            match["neighbor_mass"] = neighbor_mass[offset : offset + count]
            match["logits"] = combined["logits"][offset : offset + count]
            if residual is not None:
                match["hypothesis_residual"] = residual[offset : offset + count]
            offset += count
        return matches

    def _fuse_scene(self, stage, inputs, b):
        return spatial_hash_fuse(dict(stage), self.fusion_cell_size)

    def _ray_evidence(self, stage, inputs, b):
        from jar.ray_evidence import build_ray_evidence

        return build_ray_evidence(stage, inputs, b, self.roi)

    def forward(self, inputs, pyramid, coarse, teacher=None):
        if teacher is not None:
            raise ValueError("ET-JAR uses image queries; GT teacher forcing is not supported")
        if self.edge_queries or self.edge_ruler:
            from jar.edge_evidence import build_edge_pyramid

            inputs = dict(inputs)
            inputs["_edge_levels"] = [
                build_edge_pyramid(
                    inputs["rgb"][b],
                    inputs["view_mask"][b] if "view_mask" in inputs else None,
                    inputs["image_mask"][b] if "image_mask" in inputs else None,
                )
                for b in range(len(inputs["rgb"]))
            ]
        scenes = []
        for b in range(inputs["rgb"].shape[0]):
            patches = (
                _patch_maps(inputs["rgb"][b], self.patch_dilation)
                if self.photo_descriptor == "zncc3"
                else _census_patch_maps(inputs["rgb"][b], self.patch_dilation)
            )
            # Every pixel used by a patch must be valid. Replication at the
            # physical image border matches _patch_maps; interior missing-data
            # holes are eroded by the actual patch radius and cannot contaminate
            # descriptors. Census is nonlinear only AFTER subpixel sampling.
            mask = inputs.get("image_mask")
            if mask is None:
                patch_mask = torch.ones_like(inputs["rgb"][b, :, :1])
            else:
                missing = 1 - mask[b, :, None].float()
                radius = self.patch_radius
                patch_mask = 1 - F.max_pool2d(
                    F.pad(missing, (radius,) * 4, mode="replicate"), radius * 2 + 1, stride=1
                )
            texture = (
                inputs["_texture"][b] if "_texture" in inputs else image_texture(inputs["rgb"][b])
            )
            previous, stages = None, []
            for stage, stride in enumerate(self.strides):
                query = self._stage_queries(inputs, pyramid, coarse, b, stage, previous)
                queries = [
                    {key: value[start : start + self.chunk] for key, value in query.items()}
                    for start in range(0, len(query["uv"]), self.chunk)
                ]
                context_active = self.curve_context.get(
                    "enabled", False
                ) and stage in self.curve_context.get("stages", [3])
                context_active |= (
                    hasattr(self, "hypothesis_context") and stage in self.hypothesis_stages
                )
                matches = (
                    self._context_matches(
                        inputs, pyramid, patches, patch_mask, texture, b, stage, queries
                    )
                    if context_active
                    else None
                )
                parts = []
                for index, sub in enumerate(queries):
                    parts.append(
                        self._decode_chunk(
                            inputs,
                            pyramid,
                            patches,
                            patch_mask,
                            texture,
                            b,
                            stage,
                            sub,
                            **({"match": matches[index]} if matches is not None else {}),
                        )
                    )
                output = {}
                for key in parts[0]:
                    if key != "aux":
                        output[key] = torch.cat([part[key] for part in parts], 0)
                output["aux"] = {}
                for key, value in parts[0]["aux"].items():
                    output["aux"][key] = (
                        torch.cat([part["aux"][key] for part in parts], 0)
                        if torch.is_tensor(value)
                        else value
                    )
                # Chunk-local parent indices already refer to the full previous
                # stage. Persist all stages for deep supervision and inspection.
                output["stage_stride"] = stride
                output = self._finish_stage(output, inputs, pyramid, b, stage)
                stages.append(output)
                previous = dict(
                    uv=query["uv"],
                    refs=query["refs"],
                    depth=output["aux"]["reference_depth"],
                    features=output["features"],
                    points=output["points"],
                    normals=output["normals"],
                    valid=output["valid"],
                )
            final = self._fuse_scene(stages[-1], inputs, b)
            final = enforce_roi(final, self.roi)
            final["stages"] = stages
            if self.free_space_source == "verified_queries":
                final["ray_evidence"] = self._ray_evidence(stages[-1], inputs, b)
            # Final aux describes pre-fusion queries, as in JAR-Surface; keep an
            # explicit stage pointer for tools that need query-level metadata.
            final["aux"]["unfused_query_count"] = len(stages[-1]["points"])
            final["aux"]["architecture"] = "encoder_token_hierarchy"
            scenes.append(final)
        return dict(scenes=scenes, coarse=coarse)


# Short public alias; both names denote this independent implementation.
TokenHierarchyDecoder = EncoderTokenHierarchyDecoder
