"""多焦距确定性验收：观测分辨率、独立基线、全局模式及反向匹配。

所有检查仅消费 RGB、标定和已验证的预测；不接受 targets。深度标准差是
局部一阶像素定位噪声传播估计，不是经过真实数据校准的概率保证。
"""

from __future__ import annotations

import torch
from torch import nn

from jar.geometry import camera_centers, project, image_texture, sample_maps
from jar.geometry import reference_rays

OUT_OF_FOV = 0
CONFIRMED_OCCLUDED = 1
INSUFFICIENT = 2
SUPPORT = 3
CONTRADICT = 4
OCCLUSION_UNKNOWN = 5
STATE_NAMES = (
    "out_of_fov",
    "confirmed_occluded",
    "insufficient_resolution_or_texture",
    "support",
    "contradict",
    "occlusion_unknown",
)
REASON_NAMES = (
    "accepted",
    "reference_invalid",
    "insufficient_independent_support",
    "observable_contradiction",
    "depth_uncertainty",
    "ambiguous_or_cycle_failure",
)


def accept_observations(
    *,
    refs,
    depth,
    centers,
    projected_valid,
    cost,
    texture,
    resolution_px,
    parallax,
    unique,
    cycle_px,
    reference_valid=None,
    patch_valid=None,
    visibility_known=None,
    confirmed_occluded=None,
    photo_threshold=0.15,
    texture_threshold=0.002,
    min_resolution_px=1.5,
    min_baseline_m=0.05,
    min_parallax=1e-5,
    max_cycle_px=0.75,
    pixel_sigma=0.5,
    max_depth_sigma_m=5.0,
    max_relative_sigma=0.10,
):
    """Pure tensor acceptance. All per-view tensors have shape [Q,N].

    A poor-resolution wide view is abstention, never a negative vote. A truly
    visible, textured and resolved contradiction blocks acceptance regardless
    of the number of agreeing views. Visibility cannot be inferred from a
    mismatch: absent a trusted first hit it is OCCLUSION_UNKNOWN. Duplicate
    camera centres cannot count twice. There is no learned acceptance weight.
    """
    q, n = cost.shape
    device = cost.device
    row = torch.arange(q, device=device)
    source = torch.arange(n, device=device)[None] != refs[:, None]
    baseline = (centers[None] - centers[refs, None]).norm(dim=-1)
    independent = source & (baseline >= min_baseline_m) & (parallax >= min_parallax)
    valid = projected_valid.bool() & torch.isfinite(cost) & torch.isfinite(texture)
    valid &= torch.isfinite(resolution_px) & torch.isfinite(parallax)
    known = torch.zeros_like(valid) if visibility_known is None else visibility_known.bool()
    occluded = torch.zeros_like(valid) if confirmed_occluded is None else confirmed_occluded.bool()
    observable = valid & (texture >= texture_threshold) & (resolution_px >= min_resolution_px)
    observable &= ~occluded
    if patch_valid is not None:
        observable &= patch_valid.bool()
    good = observable & (cost < photo_threshold) & unique.bool()
    good &= torch.isfinite(cycle_px) & (cycle_px <= max_cycle_px) & independent
    measured_support = good.clone()
    # Shared centres provide only one independent ray constraint. Prefer the
    # most precise observation (often tele), not the first/wide camera index.
    localization = pixel_sigma * (1 + torch.nan_to_num(cost.detach(), nan=1.0).clamp(0, 1))
    strength = torch.where(good, (parallax / localization.clamp_min(1e-6)).square(), 0.0)
    duplicate = (centers[:, None] - centers[None]).norm(dim=-1) < min_baseline_m
    ids = torch.arange(n, device=device)
    stronger = strength[:, None, :] > strength[:, :, None]
    tied_earlier = (strength[:, None, :] == strength[:, :, None]) & (ids[None, :] < ids[:, None])[
        None
    ]
    suppress = (duplicate[None] & (stronger | tied_earlier) & good[:, None, :]).any(-1)
    good = good & ~suppress
    contradiction = observable & known & source & (cost >= photo_threshold)
    state = torch.full((q, n), OUT_OF_FOV, device=device, dtype=torch.long)
    state[valid] = INSUFFICIENT
    state[observable & source] = OCCLUSION_UNKNOWN
    state[measured_support] = SUPPORT
    state[contradiction] = CONTRADICT
    state[valid & occluded] = CONFIRMED_OCCLUDED
    ref_ok = valid[row, refs] if reference_valid is None else reference_valid.bool()
    state[row, refs] = torch.where(ref_ok, SUPPORT, OUT_OF_FOV)
    source_count = good.sum(-1)
    # A large image footprint is not a position error bar. Only actual ray
    # parallax and matching localization noise enter depth uncertainty.
    information = torch.where(good, (parallax / localization.clamp_min(1e-6)).square(), 0).sum(-1)
    sigma = information.clamp_min(1e-12).rsqrt().clamp_max(1e6)
    sigma_limit = torch.minimum(
        torch.nan_to_num(depth.abs(), nan=0.0, posinf=0.0) * max_relative_sigma,
        torch.full_like(depth, max_depth_sigma_m),
    )
    enough = source_count >= 1
    precise = torch.isfinite(depth) & (depth > 0) & (sigma <= sigma_limit)
    accepted = ref_ok & enough & precise & ~contradiction.any(-1)
    count = source_count + ref_ok.long()
    quality = torch.where(good, (1 - cost).clamp(0, 1), 0).sum(-1) / source_count.clamp_min(1)
    confidence = quality * torch.exp(-sigma / sigma_limit.clamp_min(1e-6)) * accepted
    reason = torch.zeros(q, dtype=torch.long, device=device)
    reason[~precise] = 4
    failed_matching = (observable & independent).any(-1) & ~enough
    reason[failed_matching] = 5
    reason[~enough & ~failed_matching] = 2
    reason[contradiction.any(-1)] = 3
    reason[~ref_ok] = 1
    return dict(
        valid=accepted,
        state_per_view=state,
        support=count,
        confidence=confidence,
        depth_sigma=sigma,
        reason=reason,
        independent_support=good,
        per_view_support=measured_support,
        contradiction=contradiction,
        position_information=information,
        state_names=STATE_NAMES,
        reason_names=REASON_NAMES,
    )


def _global_unique(depth, candidates, costs, valid, margin, separation_fraction=0.1):
    """Compare measured final-position cost against separated global modes."""
    separation = (candidates - depth[:, None]).abs() > depth[:, None].abs().clamp_min(
        1.0
    ) * separation_fraction
    separated = valid & separation[..., None]
    alternative = costs.masked_fill(~separated, float("inf")).amin(1)
    enough = valid.sum(1) >= 3
    return alternative, separated.any(1) & enough


class ObservabilityVerifier(nn.Module):
    """Parameter-free final-position verifier; score_fn never receives labels.

    score_fn(inputs,b,refs,uv,depths,normals,stride=) returns raw photometric
    cost, exact patch validity, texture and projected resolution [Q,D,N].
    Reverse matching independently searches each source ray, then projects
    its best measured point back into the fixed reference image.
    """

    def __init__(self, cfg):
        super().__init__()
        model = cfg.get("model", cfg)
        m = model.get("multifocal", {})
        self.photo_threshold = float(
            m.get("photo_threshold", model.get("et_photo_threshold", 0.15))
        )
        self.texture_threshold = float(
            m.get("texture_threshold", model.get("texture_threshold", 0.002))
        )
        self.uniqueness_margin = float(m.get("uniqueness_margin", 0.015))
        self.min_resolution_px = float(m.get("min_resolution_px", 1.5))
        self.min_baseline_m = float(m.get("min_baseline_m", 0.05))
        self.max_cycle_px = float(m.get("max_cycle_px", 0.75))
        self.pixel_sigma = float(m.get("pixel_sigma", 0.5))
        self.max_sigma = float(m.get("max_depth_sigma_m", 5.0))
        self.relative_sigma = float(m.get("max_relative_sigma", 0.10))
        self.reverse_samples = int(m.get("reverse_samples", 9))
        if self.reverse_samples < 3 or self.reverse_samples % 2 != 1:
            raise ValueError("reverse_samples must be an odd integer >= 3")
        if (
            min(
                self.min_resolution_px,
                self.min_baseline_m,
                self.pixel_sigma,
                self.max_sigma,
                self.relative_sigma,
            )
            <= 0
        ):
            raise ValueError("observability geometric thresholds must be positive")

    @torch.no_grad()
    def forward(
        self,
        inputs,
        b,
        refs,
        uv,
        depth,
        normals,
        score_fn,
        *,
        candidate_depths,
        footprint_m=None,
        first_hits=None,
        stride=4,
    ):
        rgb, K, T = inputs["rgb"][b], inputs["K"][b], inputs["T_cam_from_rig"][b]
        q, n = len(depth), len(K)
        centers, rays, _ = reference_rays(torch.zeros(q, 3, device=depth.device), refs, uv, K, T)
        X = centers + rays * depth[:, None]
        vm = inputs.get("view_mask")
        vm = None if vm is None else vm[b]
        im = inputs.get("image_mask")
        im = None if im is None else im[b]
        projected, camera_depth, visible, J = project(X, K, T, rgb.shape[-2:], vm, im)
        parallax = (J * rays[None, :, None]).sum(-1).norm(dim=-1).T
        final = score_fn(inputs, b, refs, uv, depth[:, None], normals, stride=stride)
        costs = final["photo_cost"][:, 0].float()
        valid = final["valid"][:, 0].bool() & visible.T
        texture = final["texture"][:, 0].float()
        resolution = final["resolution_px"][:, 0].float()
        # Optional physical surfel support can only reduce observability. It
        # cannot make a large window imply a small target is resolved.
        if footprint_m is not None:
            width = torch.as_tensor(footprint_m, device=X.device).reshape(q, -1).amin(-1)
            projected_width = 2 * J.square().sum((-1, -2)).sqrt().T * width[:, None]
            resolution = torch.minimum(resolution, projected_width)
        global_score = score_fn(inputs, b, refs, uv, candidate_depths, normals, stride=stride)
        alternative, has_alternative = _global_unique(
            depth,
            candidate_depths,
            global_score["photo_cost"],
            global_score["valid"],
            self.uniqueness_margin,
        )
        unique = has_alternative & (costs + self.uniqueness_margin < alternative)
        cycle = torch.full_like(costs, 1e6)
        reverse_unique = torch.zeros_like(valid)
        row = torch.arange(q, device=depth.device)
        offsets = torch.linspace(-2, 2, self.reverse_samples, device=depth.device)
        # Only N fixed camera groups; never per-node recursion.
        for view in range(n):
            active = valid[:, view] & (refs != view) & unique[:, view]
            indices = active.nonzero().flatten()
            if not len(indices):
                continue
            source_refs = torch.full((len(indices),), view, device=refs.device, dtype=torch.long)
            source_uv = projected[view, indices]
            source_depth = camera_depth[view, indices]
            _, reverse_rays, _ = reference_rays(X[indices], source_refs, source_uv, K, T)
            reverse_g = (J[refs[indices], indices] * reverse_rays[:, None]).sum(-1).norm(dim=-1)
            step = (0.5 / reverse_g.clamp_min(1e-6)).clamp(0.001, 20.0)
            reverse_depths = torch.cat(
                (
                    (source_depth[:, None] + offsets[None] * step[:, None]).clamp_min(0.01),
                    candidate_depths[indices],
                ),
                -1,
            )
            reverse = score_fn(
                inputs, b, source_refs, source_uv, reverse_depths, normals[indices], stride=stride
            )
            rid = refs[indices]
            rr = torch.arange(len(indices), device=depth.device)
            rcost = reverse["photo_cost"][rr, :, rid]
            rvalid = reverse["valid"][rr, :, rid]
            ranked = rcost.masked_fill(~rvalid, float("inf"))
            winner = ranked.argmin(-1)
            best_depth = reverse_depths[rr, winner]
            best_cost = ranked[rr, winner]
            separate = (reverse_depths - source_depth[:, None]).abs() > source_depth[
                :, None
            ].abs().clamp_min(1.0) * 0.1
            alternate = ranked.masked_fill(~separate, float("inf")).amin(-1)
            reverse_ok = separate.logical_and(rvalid).any(-1) & (rvalid.sum(-1) >= 3)
            reverse_ok &= best_cost + self.uniqueness_margin < alternate
            reverse_ok &= best_cost < self.photo_threshold
            reverse_X = camera_centers(T)[view] + reverse_rays * best_depth[:, None]
            back, _, back_valid, _ = project(reverse_X, K, T, rgb.shape[-2:], vm, im)
            error = (back[rid, rr] - uv[indices]).norm(dim=-1)
            error = torch.where(reverse_ok & back_valid[rid, rr], error, 1e6)
            cycle[indices, view] = error
            reverse_unique[indices, view] = reverse_ok
        # The reference anchor does not need a self-correspondence from scorer.
        ref_texture = sample_maps(image_texture(rgb), projected)[..., 0].T[row, refs]
        ref_valid = visible.T[row, refs] & (ref_texture >= self.texture_threshold)
        valid[row, refs] = ref_valid
        known = torch.zeros_like(valid)
        occluded = torch.zeros_like(valid)
        if first_hits is not None:
            known, occluded = self._first_hit_visibility(
                inputs, b, projected, camera_depth, visible, first_hits
            )
        result = accept_observations(
            refs=refs,
            depth=depth,
            centers=camera_centers(T),
            projected_valid=visible.T,
            patch_valid=valid,
            cost=costs,
            texture=texture,
            resolution_px=resolution,
            parallax=parallax,
            unique=unique & reverse_unique,
            cycle_px=cycle,
            reference_valid=ref_valid,
            visibility_known=known,
            confirmed_occluded=occluded,
            photo_threshold=self.photo_threshold,
            texture_threshold=self.texture_threshold,
            min_resolution_px=self.min_resolution_px,
            min_baseline_m=self.min_baseline_m,
            max_cycle_px=self.max_cycle_px,
            pixel_sigma=self.pixel_sigma,
            max_depth_sigma_m=self.max_sigma,
            max_relative_sigma=self.relative_sigma,
        )
        result.update(
            final_cost=costs,
            projected_resolution_px=resolution,
            reverse_cycle_px=cycle,
            global_unique=unique,
            reverse_unique=reverse_unique,
            parallax_px_per_m=parallax,
            first_hit_visibility_known=known,
            final_texture=texture,
            final_patch_valid=valid,
            reference_valid=ref_valid,
        )
        return result

    @staticmethod
    def _first_hit_visibility(inputs, b, projected, camera_depth, visible, first_hits):
        if first_hits.get("source") not in (
            "verified_reference_queries_only",
            "mf_verified_reference_queries_only",
        ):
            raise ValueError(
                "occlusion needs verified predicted first hits, never coarse depth or labels"
            )
        h, w = inputs["rgb"].shape[-2:]
        xy = projected.round().long()
        px, py = xy[..., 0].clamp(0, w - 1), xy[..., 1].clamp(0, h - 1)
        ci = torch.arange(projected.shape[0], device=projected.device)[:, None]
        hit = first_hits["depth"][ci, py, px]
        reliable = first_hits["valid"][ci, py, px] & visible
        reliable &= (projected - xy).abs().amax(-1) <= min(
            float(first_hits.get("query_pixel_tolerance_px", 1e-3)), 1e-3
        )
        reliable &= first_hits["support"][ci, py, px] >= 2
        reliable &= torch.isfinite(hit) & (hit > 0)
        band = 0.05
        occluded = (reliable & (camera_depth > hit + band)).T
        known = (reliable & (camera_depth <= hit + band)).T
        return known, occluded

    @torch.no_grad()
    def recheck_visibility(self, result, inputs, b, refs, depth, points, first_hits):
        """Second pass using only independently verified preliminary first hits.

        No photometric rerun and no targets. Cached actual final-position costs,
        uncertainty information and cycle tests are unchanged. Newly established
        visibility makes a resolved negative observation an effective veto.
        The caller builds first_hits from preliminary accepted state-head-confirmed
        queries, never from all candidates or their coarse initialization.
        """
        view = inputs.get("view_mask")
        image = inputs.get("image_mask")
        projected, camera_depth, visible, _ = project(
            points,
            inputs["K"][b],
            inputs["T_cam_from_rig"][b],
            inputs["rgb"].shape[-2:],
            None if view is None else view[b],
            None if image is None else image[b],
        )
        known, occluded = self._first_hit_visibility(
            inputs, b, projected, camera_depth, visible, first_hits
        )
        updated = accept_observations(
            refs=refs,
            depth=depth,
            centers=camera_centers(inputs["T_cam_from_rig"][b]),
            projected_valid=visible.T,
            patch_valid=result["final_patch_valid"],
            cost=result["final_cost"],
            texture=result["final_texture"],
            resolution_px=result["projected_resolution_px"],
            parallax=result["parallax_px_per_m"],
            unique=result["global_unique"] & result["reverse_unique"],
            cycle_px=result["reverse_cycle_px"],
            reference_valid=result["reference_valid"],
            visibility_known=known,
            confirmed_occluded=occluded,
            photo_threshold=self.photo_threshold,
            texture_threshold=self.texture_threshold,
            min_resolution_px=self.min_resolution_px,
            min_baseline_m=self.min_baseline_m,
            max_cycle_px=self.max_cycle_px,
            pixel_sigma=self.pixel_sigma,
            max_depth_sigma_m=self.max_sigma,
            max_relative_sigma=self.relative_sigma,
        )
        return dict(result, **updated, first_hit_visibility_known=known)
