"""在全局候选胜者附近重建局部逆深度曲线的可微精化器。

当前版本没有学习参数：图像采样、匹配代价与连续逆深度修正均可反向
传播，可接在可训练网络后面。query_features 为未来可靠性学习预留，
目前明确不参与计算。正面平面窗口变换和 Census 均是已有方法，不作
首创性声明；多视图曲线共识是可关闭的研究假设，不是遮挡真值。
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import project, sample_maps, camera_centers, pixel_grid


class CalibratedRayPolisher(nn.Module):
    """Differentiable photometric polish of image-anchored optical depths.

    Construction accepts a full project cfg, a model dict, or keyword overrides.
    All `per_view_*` outputs have shape [Q,N], with the reference view masked.
    `steps` is a list; each entry contains [Q,9] candidate arrays and [Q,N]
    consensus information. This module never reads targets or modifies K/T.

    Config: rp_rounds=3, rp_descriptor='census5', rp_window_warp='center',
    rp_curve_consensus=False, rp_span_shrink=.5, rp_initial_span_bins=1.,
    rp_global_count=(et_global_candidates or 48), rp_query_chunk=64,
    rp_texture_threshold=.006, rp_match_threshold=.6, rp_min_contrast=1e-5,
    rp_consensus_width=.5, rp_census_temperature=.02, near=2, far=80.
    """

    def __init__(self, cfg=None, **overrides):
        super().__init__()
        values = dict((cfg or {}).get("model", cfg or {}))
        values.update(overrides)
        self.near, self.far = float(values.get("near", 2.0)), float(values.get("far", 80.0))
        self.global_count = int(
            values.get(
                "rp_global_count",
                values.get("global_count", values.get("et_global_candidates", 48)),
            )
        )
        self.rounds = int(values.get("rp_rounds", 3))
        self.descriptor = values.get("rp_descriptor", "census5")
        self.window_warp = values.get("rp_window_warp", "center")
        self.curve_consensus = bool(values.get("rp_curve_consensus", False))
        self.shrink = float(values.get("rp_span_shrink", 0.5))
        self.initial_span_bins = float(values.get("rp_initial_span_bins", 1.0))
        self.chunk = int(values.get("rp_query_chunk", 64))
        self.texture_threshold = float(values.get("rp_texture_threshold", 0.006))
        self.match_threshold = float(values.get("rp_match_threshold", 0.6))
        self.min_contrast = float(values.get("rp_min_contrast", 1e-5))
        self.consensus_width = float(values.get("rp_consensus_width", 0.5))
        self.census_temperature = float(values.get("rp_census_temperature", 0.02))
        self.min_support = int(values.get("min_support", 2))
        self.parabola_slack = float(values.get("rp_parabola_slack", 0.002))
        if not 0 < self.near < self.far or self.global_count < 2:
            raise ValueError("Ray polish needs 0 < near < far and global_count >= 2")
        if (
            self.rounds < 1
            or self.chunk < 1
            or not 0 < self.shrink <= 1
            or self.initial_span_bins <= 0
        ):
            raise ValueError("Ray polish rounds/chunk/span must be positive; shrink must be (0,1]")
        if self.descriptor not in ("census5", "zncc") or self.window_warp not in (
            "center",
            "frontoparallel",
        ):
            raise ValueError("rp_descriptor: census5/zncc; rp_window_warp: center/frontoparallel")
        if (
            min(self.texture_threshold, self.consensus_width, self.census_temperature) <= 0
            or self.min_support < 2
        ):
            raise ValueError(
                "texture/consensus/Census scales must be positive and min_support >= 2"
            )
        if not 0 < self.match_threshold <= 2 or self.min_contrast < 0 or self.parabola_slack < 0:
            raise ValueError("Invalid polish contrast, matching threshold or parabola tolerance")
        self.radius = 2
        self.register_buffer(
            "patch_offsets", (pixel_grid(5, 5) - 2).reshape(-1, 2), persistent=False
        )
        self.register_buffer("search_offsets", torch.linspace(-1.0, 1.0, 9), persistent=False)

    @staticmethod
    def _mixed_rays(uv, refs, K, T):
        with torch.autocast(device_type=uv.device.type, enabled=False):
            uv, K, T = uv.float(), K.float(), T.float()
            shape = uv.shape
            hom = torch.cat((uv, torch.ones_like(uv[..., :1])), -1).reshape(len(refs), -1, 3)
            ray_cam = torch.linalg.solve(K[refs], hom.transpose(-1, -2)).transpose(-1, -2)
            ray_cam = ray_cam / ray_cam[..., 2:].clamp_min(1e-8)
            rays = torch.einsum("qpi,qij->qpj", ray_cam, T[refs, :3, :3])
            centers = camera_centers(T)[refs]
            return centers, rays.reshape(*shape[:-1], 3)

    def _prepare(self, inputs, b, refs, uv):
        rgb, K, T = (
            inputs["rgb"][b].float(),
            inputs["K"][b].float(),
            inputs["T_cam_from_rig"][b].float(),
        )
        n, _, h, w = rgb.shape
        vm = inputs.get("view_mask")
        vm = torch.ones(n, device=rgb.device, dtype=torch.bool) if vm is None else vm[b].bool()
        im = inputs.get("image_mask")
        im = (
            torch.ones(n, h, w, device=rgb.device, dtype=torch.bool) if im is None else im[b].bool()
        )
        pixels = uv[:, None] + self.patch_offsets[None]
        count, patches = pixels.shape[:2]
        sampled = sample_maps(rgb, pixels.reshape(1, -1, 2).expand(n, -1, -1)).reshape(
            n, count, patches, 3
        )
        reference = sampled[refs, torch.arange(count, device=rgb.device)]
        mask = sample_maps(im[:, None].float(), pixels.reshape(1, -1, 2).expand(n, -1, -1))[
            ..., 0
        ].reshape(n, count, patches)
        ref_valid = vm[refs] & (mask[refs, torch.arange(count, device=rgb.device)] > 0.999).all(-1)
        ref_valid &= (
            (pixels[..., 0] >= 0)
            & (pixels[..., 0] <= w - 1)
            & (pixels[..., 1] >= 0)
            & (pixels[..., 1] <= h - 1)
        ).all(-1)
        centered = reference - reference.mean(-2, keepdim=True)
        ref_texture = (centered.square().mean((-2, -1)) + 1e-12).sqrt()
        ref_valid &= ref_texture > self.texture_threshold
        centers, rays = self._mixed_rays(pixels, refs, K, T)
        return dict(
            rgb=rgb,
            K=K,
            T=T,
            view_mask=vm,
            image_mask=im,
            hw=(h, w),
            refs=refs,
            uv=uv,
            rays=rays,
            centers=centers,
            reference=reference,
            ref_valid=ref_valid,
            ref_texture=ref_texture,
        )

    def _costs(self, depths, context):
        """Exact image samples, returning cost/valid/derivative [Q,D,N]."""
        rgb, K, T = context["rgb"], context["K"], context["T"]
        n, q, d, p = len(rgb), len(depths), depths.shape[1], len(self.patch_offsets)
        hw, vm, im = context["hw"], context["view_mask"], context["image_mask"]
        center_ray = context["rays"][:, p // 2]
        if self.window_warp == "frontoparallel":
            # Every reference patch pixel gets the same optical depth. This is
            # an explicit local reference-frontoparallel plane hypothesis.
            points = (
                context["centers"][:, None, None]
                + context["rays"][:, None] * depths[:, :, None, None]
            )
            uv, _, valid, J = project(points.reshape(-1, 3), K, T, hw, vm, im)
            source = sample_maps(rgb, uv).reshape(n, q, d, p, 3).permute(1, 2, 0, 3, 4)
            valid = valid.reshape(n, q, d, p).all(-1).permute(1, 2, 0)
            center_J = J.reshape(n, q, d, p, 2, 3)[:, :, :, p // 2]
        else:
            # Only patch centers correspond geometrically; fixed image-offset
            # windows are a local descriptor, not full perspective correction.
            points = context["centers"][:, None] + center_ray[:, None] * depths[..., None]
            center_uv, _, center_valid, center_J = project(points.reshape(-1, 3), K, T, hw, vm, im)
            window = center_uv[:, :, None] + self.patch_offsets[None, None]
            source = (
                sample_maps(rgb, window.reshape(n, -1, 2))
                .reshape(n, q, d, p, 3)
                .permute(1, 2, 0, 3, 4)
            )
            h, w = hw
            window_valid = (
                (window[..., 0] >= 0)
                & (window[..., 0] <= w - 1)
                & (window[..., 1] >= 0)
                & (window[..., 1] <= h - 1)
            )
            mask = sample_maps(im[:, None].float(), window.reshape(n, -1, 2))[..., 0].reshape(
                n, q * d, p
            )
            valid = (
                (center_valid & (window_valid & (mask > 0.999)).all(-1))
                .reshape(n, q, d)
                .permute(1, 2, 0)
            )
            center_J = center_J.reshape(n, q, d, 2, 3)
        derivative = (
            (center_J.float() * center_ray[None, :, None, None])
            .sum(-1)
            .norm(dim=-1)
            .permute(1, 2, 0)
        )
        independent = torch.arange(n, device=rgb.device)[None] != context["refs"][:, None]
        centered = source - source.mean(-2, keepdim=True)
        texture = (centered.square().mean((-2, -1)) + 1e-12).sqrt()
        valid = valid & independent[:, None] & context["ref_valid"][:, None, None]
        valid &= (derivative > 1e-6) & (texture > self.texture_threshold)
        reference = context["reference"]
        if self.descriptor == "census5":
            luma = rgb.new_tensor([0.299, 0.587, 0.114])
            src_gray = (source * luma).sum(-1)
            ref_gray = (reference * luma).sum(-1)
            src_code = torch.tanh(
                (src_gray - src_gray[..., p // 2 : p // 2 + 1]) / self.census_temperature
            )
            ref_code = torch.tanh(
                (ref_gray - ref_gray[..., p // 2 : p // 2 + 1]) / self.census_temperature
            )
            costs = (src_code - ref_code[:, None, None]).abs().sum(-1) / (p - 1)
        else:
            ref_centered = reference - reference.mean(-2, keepdim=True)
            src_unit = F.normalize(centered.flatten(-2), dim=-1, eps=1e-6)
            ref_unit = F.normalize(ref_centered.flatten(-2), dim=-1, eps=1e-6)
            costs = 1 - (src_unit * ref_unit[:, None, None]).sum(-1)
        costs = costs.clamp(0, 2)
        return dict(
            cost=torch.where(valid, costs, torch.full_like(costs, 2.0)),
            valid=valid,
            derivative=derivative,
            texture=texture,
        )

    def aggregate_curves(self, costs, valid, inverse_depths, half_span):
        """[Q,D,N] per-view curves -> fused curve and [Q,N] view weights.

        Optional consensus first locates every source view's own local mode,
        then finds a mode supported by the largest number of independent views.
        Views whose modes disagree receive bounded Cauchy weights. This needs
        redundancy: with a single available source, it reduces to that source;
        with two disagreeing sources, no majority can establish ground truth.
        """
        q, d, n = costs.shape
        safe = costs.masked_fill(~valid, 2.0)
        winners = safe.detach().argmin(1)
        view_has = valid.sum(1) >= 2
        best = safe.gather(1, winners[:, None]).squeeze(1)
        maximum = costs.masked_fill(~valid, 0).amax(1)
        contrast = (maximum - best).clamp_min(0)
        mode = inverse_depths[:, :, None].expand(-1, -1, n).gather(1, winners[:, None]).squeeze(1)
        modes_ok = view_has & (contrast > self.min_contrast)
        weights = torch.ones_like(mode)
        chosen_mode = inverse_depths[:, inverse_depths.shape[1] // 2]
        if self.curve_consensus:
            width = (
                torch.as_tensor(half_span, device=mode.device, dtype=mode.dtype).clamp_min(1e-8)
                * self.consensus_width
            )
            distance = (mode[:, :, None] - mode[:, None]) / width
            neighbourhood = torch.exp(-0.5 * distance.detach().square()) * modes_ok[:, None]
            density = neighbourhood.sum(-1)
            # A small quality tie-break cannot override a larger support group.
            quality = (contrast / (contrast + 0.01)) / (1 + best)
            score = (density + 0.001 * quality.detach()).masked_fill(~modes_ok, -1e4)
            medoid = score.argmax(-1)
            chosen_mode = mode[torch.arange(q, device=mode.device), medoid]
            disagreement = ((mode - chosen_mode[:, None]) / width).detach()
            weights = (1 / (1 + disagreement.square())) * modes_ok
            # No identifiable per-view modes: ordinary valid-view average is a
            # finite fallback, and the fused curve still must pass contrast.
            weights = torch.where(modes_ok.any(-1, keepdim=True), weights, torch.ones_like(weights))
        effective = valid * weights[:, None]
        denominator = effective.sum(-1)
        aggregate = (costs * effective).sum(-1) / denominator.clamp_min(1e-8)
        candidate_valid = (valid.sum(-1) >= self.min_support - 1) & (denominator > 0)
        aggregate = torch.where(candidate_valid, aggregate, torch.full_like(aggregate, 2.0))
        return dict(
            cost=aggregate,
            valid=candidate_valid,
            view_weights=weights,
            view_mode=mode,
            view_mode_valid=modes_ok,
            consensus_mode=chosen_mode,
        )

    @staticmethod
    def _weighted_single(observation, weights, minimum_sources):
        valid = observation["valid"][:, 0]
        w = valid * weights
        cost = (observation["cost"][:, 0] * w).sum(-1) / w.sum(-1).clamp_min(1e-8)
        ok = (valid.sum(-1) >= minimum_sources) & (w.sum(-1) > 0)
        return torch.where(ok, cost, torch.full_like(cost, 2.0)), ok

    def _run_chunk(self, inputs, b, refs, uv, initial_depth):
        context = self._prepare(inputs, b, refs, uv)
        depth = initial_depth.float().clamp(self.near, self.far)
        initial_points = context["centers"] + context["rays"][:, 12] * depth[:, None]
        q = len(depth)
        index = torch.arange(q, device=depth.device)
        bin_width = (1 / self.near - 1 / self.far) / (self.global_count - 1)
        steps = []
        previous_evidence = torch.zeros(q, device=depth.device, dtype=torch.bool)
        for iteration in range(self.rounds):
            half_span = bin_width * self.initial_span_bins * self.shrink**iteration
            inverse = (depth.reciprocal()[:, None] + self.search_offsets[None] * half_span).clamp(
                1 / self.far, 1 / self.near
            )
            candidates = inverse.reciprocal()
            observed = self._costs(candidates, context)
            aggregate = self.aggregate_curves(
                observed["cost"], observed["valid"], inverse, half_span
            )
            cost, candidate_valid = aggregate["cost"], aggregate["valid"]
            winner = cost.detach().masked_fill(~candidate_valid, 1e4).argmin(-1)
            left, right = (winner - 1).clamp_min(0), (winner + 1).clamp_max(8)
            before_points = context["centers"] + context["rays"][:, 12] * depth[:, None]
            rho = inverse[index, winner]
            c0, cl, cr = cost[index, winner], cost[index, left], cost[index, right]
            hl, hr = rho - inverse[index, left], inverse[index, right] - rho
            curvature = cl - 2 * c0 + cr
            interior = (
                (winner > 0)
                & (winner < 8)
                & candidate_valid[index, left]
                & candidate_valid[index, right]
            )
            interior &= (
                (hl > 1e-9)
                & (hr > 1e-9)
                & ((hl - hr).abs() < half_span * 1e-3 + 1e-8)
                & (curvature > 1e-7)
            )
            offset = 0.5 * (cl - cr) / curvature.clamp_min(1e-7) * ((hl + hr) * 0.5)
            offset = torch.maximum(torch.minimum(offset, hr * 0.5), -hl * 0.5)
            refined_inverse = (rho + torch.where(interior, offset, torch.zeros_like(offset))).clamp(
                1 / self.far, 1 / self.near
            )
            proposal_depth = refined_inverse.reciprocal()
            actual = self._costs(proposal_depth[:, None], context)
            actual_cost, actual_valid = self._weighted_single(
                actual, aggregate["view_weights"], self.min_support - 1
            )
            accept_interpolation = (
                interior & actual_valid & (actual_cost <= c0 + self.parabola_slack)
            )
            proposal_depth = torch.where(
                accept_interpolation, proposal_depth, candidates[index, winner]
            )
            variation = cost.masked_fill(~candidate_valid, -1).amax(-1) - cost.masked_fill(
                ~candidate_valid, 3
            ).amin(-1)
            evidence = (candidate_valid.sum(-1) >= 3) & (variation > self.min_contrast)
            new_depth = torch.where(evidence, proposal_depth, depth)
            steps.append(
                dict(
                    points_before=before_points,
                    points_after=context["centers"] + context["rays"][:, 12] * new_depth[:, None],
                    candidate_depths=candidates,
                    candidate_inverse_depths=inverse,
                    candidate_costs=cost,
                    candidate_logits=-cost / 0.05,
                    candidate_valid=candidate_valid,
                    per_view_candidate_costs=observed["cost"],
                    per_view_candidate_valid=observed["valid"],
                    view_weights=aggregate["view_weights"],
                    view_mode=aggregate["view_mode"],
                    view_mode_valid=aggregate["view_mode_valid"],
                    consensus_mode=aggregate["consensus_mode"],
                    interpolation_accepted=accept_interpolation & evidence,
                    search_evidence=evidence,
                    half_span_inverse_m=depth.new_full((q,), half_span),
                )
            )
            depth = new_depth
            previous_evidence = evidence
        final = self._costs(depth[:, None], context)
        per_view_cost, per_view_valid = final["cost"][:, 0], final["valid"][:, 0]
        matching = per_view_valid & (per_view_cost < self.match_threshold)
        support = matching.sum(-1) + context["ref_valid"].long()
        evidence = previous_evidence & (support >= self.min_support)
        points = context["centers"] + context["rays"][:, 12] * depth[:, None]
        return dict(
            depth=depth,
            points=points,
            per_view_cost=per_view_cost,
            per_view_valid=per_view_valid,
            per_view_matching=matching,
            support=support,
            evidence=evidence,
            ref_valid=context["ref_valid"],
            initial_points=initial_points,
            delta_xyz=points - initial_points,
            steps=steps,
        )

    def forward(self, inputs, batch_index, ref_index, ref_uv, initial_depth, query_features=None):
        """Return [Q] depth and [Q,N] source evidence; no GT or free-space labels.

        query_features is intentionally unused in this parameter-free baseline.
        Refinement need not be correct if the initial winner is in a wrong,
        well-textured mode. The caller should retain its global ambiguity gate.
        """
        del query_features
        if (
            ref_index.ndim != 1
            or ref_uv.shape != (len(ref_index), 2)
            or initial_depth.shape != (len(ref_index),)
        ):
            raise ValueError("Expected refs[Q], uv[Q,2], initial_depth[Q]")
        n = inputs["rgb"].shape[1]
        if (
            ref_index.dtype not in (torch.int64, torch.int32)
            or ((ref_index < 0) | (ref_index >= n)).any()
        ):
            raise ValueError("Reference indices must be valid integer camera indices")
        if (
            not torch.isfinite(ref_uv).all()
            or not torch.isfinite(initial_depth).all()
            or (initial_depth <= 0).any()
        ):
            raise ValueError("Initial depths must be finite and positive, reference pixels finite")
        if len(ref_index) == 0:
            depth = initial_depth.float()
            return dict(
                depth=depth,
                points=depth.new_empty((0, 3)),
                per_view_cost=depth.new_empty((0, n)),
                per_view_valid=torch.empty(0, n, device=depth.device, dtype=torch.bool),
                per_view_matching=torch.empty(0, n, device=depth.device, dtype=torch.bool),
                support=torch.empty(0, device=depth.device, dtype=torch.long),
                evidence=torch.empty(0, device=depth.device, dtype=torch.bool),
                ref_valid=torch.empty(0, device=depth.device, dtype=torch.bool),
                initial_points=depth.new_empty((0, 3)),
                delta_xyz=depth.new_empty((0, 3)),
                steps=[],
            )
        pieces = [
            self._run_chunk(
                inputs,
                batch_index,
                ref_index[start : start + self.chunk].long(),
                ref_uv[start : start + self.chunk].float(),
                initial_depth[start : start + self.chunk],
            )
            for start in range(0, len(ref_index), self.chunk)
        ]
        result = {
            key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0] if key != "steps"
        }
        result["steps"] = [
            {
                key: torch.cat([piece["steps"][iteration][key] for piece in pieces])
                for key in pieces[0]["steps"][iteration]
            }
            for iteration in range(self.rounds)
        ]
        return result

    def score_candidates(self, inputs, batch_index, ref_index, ref_uv, depth_candidates):
        """Public exact-window scorer; cost/valid/derivative/texture [Q,D,N].

        Candidate depths are optical-axis metres. The same descriptor,
        reference/source masks and pure-rotation rejection as polish are used.
        This API does not impose a winner or a confidence threshold.
        """
        q = len(ref_index)
        n = inputs["rgb"].shape[1]
        if (
            ref_index.ndim != 1
            or ref_uv.shape != (q, 2)
            or depth_candidates.ndim != 2
            or depth_candidates.shape[0] != q
        ):
            raise ValueError("Expected refs[Q], uv[Q,2], candidate depths[Q,D]")
        if depth_candidates.shape[1] < 1:
            raise ValueError("At least one candidate depth is required")
        if (
            ref_index.dtype not in (torch.int64, torch.int32)
            or ((ref_index < 0) | (ref_index >= n)).any()
        ):
            raise ValueError("Reference indices must be valid integer camera indices")
        if (
            not torch.isfinite(ref_uv).all()
            or not torch.isfinite(depth_candidates).all()
            or (depth_candidates <= 0).any()
        ):
            raise ValueError(
                "Candidate depths must be finite and positive, reference pixels finite"
            )
        if not q:
            shape = (q, depth_candidates.shape[1], n)
            return dict(
                cost=depth_candidates.new_empty(shape),
                valid=torch.empty(shape, device=depth_candidates.device, dtype=torch.bool),
                derivative=depth_candidates.new_empty(shape),
                texture=depth_candidates.new_empty(shape),
            )
        parts = []
        for start in range(0, q, self.chunk):
            context = self._prepare(
                inputs,
                batch_index,
                ref_index[start : start + self.chunk].long(),
                ref_uv[start : start + self.chunk].float(),
            )
            parts.append(self._costs(depth_candidates[start : start + self.chunk].float(), context))
        return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


RayPolisher = CalibratedRayPolisher
