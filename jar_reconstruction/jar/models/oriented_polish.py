"""网络预测法线与正面平面共同参与的局部逆深度细化。

没有新增学习参数；假设选择离散，局部重心与精确重评分可微。所有相机
共享同一个平面假设，不能分别挑选法线。调用者仍须保留全局歧义拒绝逻辑。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from jar.geometry import camera_centers, pixel_grid, sample_maps, project
from jar.oriented_patch import oriented_patch_cost


class OrientedRayPolisher(nn.Module):
    """Config accepts full cfg or model dict; per_view arrays are [Q,N].

    op_rounds=2 (1..3), op_global_count=48, op_initial_span_bins=1.,
    op_span_shrink=.5, op_temperature=.03, op_query_chunk=64,
    op_descriptor='census', op_patch_radius=2, op_patch_dilation=1, op_detach_normals=True,
    op_match_threshold=.6, op_min_contrast=1e-5, op_accept_slack=.002.
    默认只阻断匹配投影到法线头的梯度；final_normal 仍允许直接法线监督。
    """

    def __init__(self, cfg=None, **overrides):
        super().__init__()
        values = dict((cfg or {}).get("model", cfg or {}))
        values.update(overrides)
        self.near, self.far = float(values.get("near", 2.0)), float(values.get("far", 80.0))
        self.rounds = int(values.get("op_rounds", 2))
        self.global_count = int(
            values.get("op_global_count", values.get("et_global_candidates", 48))
        )
        self.initial_span = float(values.get("op_initial_span_bins", 1.0))
        self.shrink = float(values.get("op_span_shrink", 0.5))
        self.temperature = float(values.get("op_temperature", 0.03))
        self.chunk = int(values.get("op_query_chunk", 64))
        self.descriptor = str(values.get("op_descriptor", "census"))
        self.radius = int(values.get("op_patch_radius", 2))
        self.dilation = int(values.get("op_patch_dilation", 1))
        self.detach_normals = bool(values.get("op_detach_normals", True))
        self.match_threshold = float(values.get("op_match_threshold", 0.6))
        self.min_contrast = float(values.get("op_min_contrast", 1e-5))
        self.accept_slack = float(values.get("op_accept_slack", 0.002))
        self.texture_threshold = float(values.get("op_texture_threshold", 0.006))
        self.census_temperature = float(values.get("op_census_temperature", 0.02))
        self.grazing_cosine = float(values.get("op_grazing_cosine", 0.025))
        self.calibrated_span = bool(values.get("op_calibrated_span", False))
        self.step_pixels = float(values.get("op_step_pixels", 1.0))
        self.min_step = float(values.get("refinement_min_step", 0.01))
        self.max_step = float(values.get("refinement_max_step", 4.0))
        if self.step_pixels <= 0 or not 0 < self.min_step <= self.max_step:
            raise ValueError("细化像素步长须正，米制步长须递增")
        if not 0 < self.near < self.far or self.global_count < 2:
            raise ValueError("require 0 < near < far and global_count >= 2")
        if self.rounds not in (1, 2, 3) or min(self.chunk, self.radius, self.dilation) < 1:
            raise ValueError("op_rounds must be 1,2,3; chunk/radius/dilation must be positive")
        if self.initial_span <= 0 or not 0 < self.shrink <= 1 or self.temperature <= 0:
            raise ValueError("invalid inverse-depth span or centroid temperature")
        if self.descriptor not in ("census", "zncc") or not 0 < self.match_threshold <= 2:
            raise ValueError("invalid descriptor or match threshold")
        if (
            min(self.min_contrast, self.accept_slack, self.texture_threshold) < 0
            or self.census_temperature <= 0
            or not 0 < self.grazing_cosine < 1
        ):
            raise ValueError("invalid matching options")
        self.register_buffer("offsets", torch.linspace(-1.0, 1.0, 9), persistent=False)

    def _score(self, inputs, b, refs, uv, depth, normals, include_fronto):
        return oriented_patch_cost(
            inputs,
            b,
            refs,
            uv,
            depth,
            normals,
            method=self.descriptor,
            patch_radius=self.radius,
            patch_dilation=self.dilation,
            include_fronto=include_fronto,
            query_chunk=self.chunk,
            texture_threshold=self.texture_threshold,
            census_temperature=self.census_temperature,
            agreement_cost=self.match_threshold,
            grazing_cosine=self.grazing_cosine,
        )

    @staticmethod
    def _rays(inputs, b, refs, uv):
        K, T = inputs["K"][b].float(), inputs["T_cam_from_rig"][b].float()
        hom = torch.cat((uv.float(), torch.ones_like(uv[:, :1]).float()), -1)
        cam = torch.linalg.solve(K[refs], hom[..., None])[..., 0]
        cam = cam / cam[:, 2:].clamp_min(1e-8)
        rays = torch.einsum("qi,qij->qj", cam, T[refs, :3, :3])
        return camera_centers(T)[refs], rays

    def _inverse_depth_span(self, inputs, b, refs, uv, depth, iteration):
        """默认保持原标量逆深度跨度；扩展可返回逐query的[Q]跨度。"""
        if self.calibrated_span:
            center, ray = self._rays(inputs, b, refs, uv)
            points = center + ray * depth[:, None]
            _, _, visible, jacobian = project(
                points,
                inputs["K"][b],
                inputs["T_cam_from_rig"][b],
                inputs["rgb"].shape[-2:],
                inputs["view_mask"][b] if "view_mask" in inputs else None,
                inputs["image_mask"][b] if "image_mask" in inputs else None,
            )
            source = torch.arange(len(jacobian), device=uv.device)[:, None] != refs[None]
            derivative = (jacobian * ray[None, :, None]).sum(-1).norm(dim=-1)
            ruler = derivative.masked_fill(~(visible & source), 0).amax(0)
            metric_step = (self.step_pixels / ruler.clamp_min(1e-8)).clamp(
                self.min_step, self.max_step
            )
            # d(1/Z)/dZ；它只定义搜索跨度，候选依旧逐一精确投影。
            return metric_step / depth.square().clamp_min(1e-8) * self.shrink**iteration
        bin_width = (1 / self.near - 1 / self.far) / (self.global_count - 1)
        return bin_width * self.initial_span * self.shrink**iteration

    def _run(self, inputs, b, refs, uv, initial_depth, normals):
        q = len(refs)
        index = torch.arange(q, device=uv.device)
        centers, rays = self._rays(inputs, b, refs, uv)
        depth = initial_depth.float()
        initial_points = centers + depth[:, None] * rays
        rgb = inputs["rgb"][b].float()
        h, w = rgb.shape[-2:]
        reference_mask = torch.isfinite(rgb).all(1)
        if "image_mask" in inputs:
            reference_mask &= inputs["image_mask"][b]
        if "view_mask" in inputs:
            reference_mask &= inputs["view_mask"][b, :, None, None]
        offsets = (
            pixel_grid(self.radius * 2 + 1, self.radius * 2 + 1, uv.device) - self.radius
        ).reshape(-1, 2) * self.dilation
        window = uv[:, None] + offsets[None]
        ref_valid = (
            (window[..., 0] >= 0)
            & (window[..., 0] <= w - 1)
            & (window[..., 1] >= 0)
            & (window[..., 1] <= h - 1)
        ).all(-1)
        for camera in range(len(rgb)):
            ids = (refs == camera).nonzero().flatten()
            if len(ids):
                mask = sample_maps(
                    reference_mask[camera][None, None].float(), window[ids].reshape(1, -1, 2)
                )[0, :, 0]
                ref_valid[ids] &= (mask.reshape(len(ids), -1) > 0.999).all(-1)
        raw_normal = torch.nan_to_num(normals.float(), nan=0.0, posinf=0.0, neginf=0.0)
        normal = F.normalize(raw_normal, dim=-1, eps=1e-8)
        normal_ok = torch.isfinite(normals).all(-1) & (raw_normal.norm(dim=-1) > 1e-8)
        fronto = inputs["T_cam_from_rig"][b, refs, 2, :3].float()
        normal = torch.where(normal_ok[:, None], normal, fronto)
        output_bank = torch.stack((normal, fronto), 1)
        cost_bank = output_bank.detach() if self.detach_normals else output_bank
        choice = torch.where(normal_ok, torch.zeros_like(refs), torch.ones_like(refs))
        seen_evidence = torch.zeros(q, dtype=torch.bool, device=uv.device)
        steps = []
        for iteration in range(self.rounds):
            span = self._inverse_depth_span(inputs, b, refs, uv, depth, iteration)
            candidate_span = span[:, None] if isinstance(span, torch.Tensor) else span
            inverse = (depth.reciprocal()[:, None] + self.offsets[None] * candidate_span).clamp(
                1 / self.far, 1 / self.near
            )
            candidates = inverse.reciprocal()
            scored = self._score(inputs, b, refs, uv, candidates, cost_bank, False)
            cost, valid = scored["cost"], scored["valid"]
            winner = cost.detach().masked_fill(~valid, 3.0).flatten(1).argmin(-1)
            winner_depth, winner_plane = torch.div(winner, 2, rounding_mode="floor"), winner % 2
            curve = cost[index, :, winner_plane]
            curve_valid = valid[index, :, winner_plane]
            unique_inverse = torch.cat(
                (
                    torch.ones(q, 1, dtype=torch.bool, device=uv.device),
                    inverse[:, 1:] - inverse[:, :-1] > 1e-9,
                ),
                1,
            )
            variation = curve.masked_fill(~curve_valid, -1.0).amax(-1) - curve.masked_fill(
                ~curve_valid, 3.0
            ).amin(-1)
            evidence = ((curve_valid & unique_inverse).sum(-1) >= 3) & (
                variation > self.min_contrast
            )
            # Centroid uses only adjacent samples of ONE chosen normal. It
            # cannot average depths from distant modes or different surfaces.
            local_index = (winner_depth[:, None] + winner_depth.new_tensor([-1, 0, 1])[None]).clamp(
                0, 8
            )
            local_cost = curve.gather(1, local_index)
            local_rho = inverse.gather(1, local_index)
            local_valid = curve_valid.gather(1, local_index)
            # Repeated clamped boundary samples must not gain double weight.
            local_valid = local_valid & torch.cat(
                (
                    torch.ones(q, 1, device=uv.device, dtype=torch.bool),
                    (local_rho[:, 1:] - local_rho[:, :-1]).abs() > 1e-9,
                ),
                1,
            )
            logits = -(local_cost - local_cost.min(-1, keepdim=True).values) / self.temperature
            weight = torch.softmax(logits.masked_fill(~local_valid, -1e4), -1) * local_valid
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            winner_rho = inverse[index, winner_depth]
            centroid = torch.where(local_valid.any(-1), (weight * local_rho).sum(-1), winner_rho)
            proposal = centroid.clamp(1 / self.far, 1 / self.near).reciprocal()
            chosen_normal = cost_bank[index, winner_plane]
            exact = self._score(inputs, b, refs, uv, proposal[:, None], chosen_normal, False)
            winner_cost = curve[index, winner_depth]
            accept = (
                evidence
                & exact["valid"][:, 0, 0]
                & (exact["cost"][:, 0, 0] <= winner_cost + self.accept_slack)
            )
            proposed_depth = torch.where(accept, proposal, candidates[index, winner_depth])
            next_depth = torch.where(evidence, proposed_depth, depth)
            choice = torch.where(evidence, winner_plane, choice)
            steps.append(
                dict(
                    candidate_depths=candidates,
                    candidate_inverse_depths=inverse,
                    candidate_costs=cost,
                    candidate_valid=valid,
                    candidate_logits=-cost / self.temperature,
                    per_view_candidate_costs=scored["per_view_cost"],
                    per_view_candidate_valid=scored["per_view_valid"],
                    winner_depth_index=winner_depth,
                    winner_normal_index=winner_plane,
                    centroid_accepted=accept,
                    search_evidence=evidence,
                    half_span_inverse_m=(
                        span if isinstance(span, torch.Tensor) else depth.new_full((q,), span)
                    ),
                    points_before=centers + depth[:, None] * rays,
                    points_after=centers + next_depth[:, None] * rays,
                )
            )
            depth = next_depth
            seen_evidence = seen_evidence | evidence
        final = self._score(inputs, b, refs, uv, depth[:, None], cost_bank[index, choice], False)
        per_view_cost = final["per_view_cost"][:, 0, 0]
        per_view_valid = final["per_view_valid"][:, 0, 0]
        matching = per_view_valid & (per_view_cost <= self.match_threshold)
        evidence = seen_evidence & final["valid"][:, 0, 0]
        points = centers + depth[:, None] * rays
        final_normal = output_bank[index, choice]
        # Sign does not affect matching; orient the output towards its camera.
        sign = torch.where((final_normal * rays).sum(-1) > 0, -1.0, 1.0)
        final_normal = final_normal * sign[:, None]
        return dict(
            depth=depth,
            points=points,
            evidence=evidence,
            per_view_cost=per_view_cost,
            per_view_valid=per_view_valid,
            per_view_matching=matching,
            support=final["support"][:, 0, 0],
            final_cost=final["cost"][:, 0, 0],
            final_normal=final_normal,
            normal_choice=choice,
            initial_points=initial_points,
            delta_xyz=points - initial_points,
            ref_valid=ref_valid,
            updated=evidence & ((depth - initial_depth).abs() > 1e-8),
            steps=steps,
        )

    def forward(self, inputs, b, refs, uv, initial_depth, normals):
        """局部细化不检查全局候选歧义，也不产生 free/surface 标签。

        输入深度必须有限且在配置范围内；无证据时保持它。default detach
        只限于用于匹配的法线；final_normal 可继续接受直接法线监督。
        """
        q, n = len(refs), inputs["rgb"].shape[1]
        if (
            refs.ndim != 1
            or uv.shape != (q, 2)
            or initial_depth.shape != (q,)
            or normals.shape != (q, 3)
        ):
            raise ValueError("expected refs[Q], uv[Q,2], initial_depth[Q], normals[Q,3]")
        if refs.dtype != torch.long or ((refs < 0) | (refs >= n)).any():
            raise ValueError("refs must contain valid integer camera indices")
        if (
            not torch.isfinite(uv).all()
            or not torch.isfinite(initial_depth).all()
            or ((initial_depth < self.near) | (initial_depth > self.far)).any()
        ):
            raise ValueError("pixels finite; initial optical depth finite and within [near,far]")
        if not q:
            floats = dict(
                depth=initial_depth.float(),
                points=initial_depth.new_empty(0, 3),
                final_normal=initial_depth.new_empty(0, 3),
                initial_points=initial_depth.new_empty(0, 3),
                delta_xyz=initial_depth.new_empty(0, 3),
                final_cost=initial_depth.new_empty(0),
                per_view_cost=initial_depth.new_empty(0, n),
            )
            flags = {
                key: torch.empty(0, dtype=torch.bool, device=uv.device)
                for key in ("evidence", "ref_valid", "updated")
            }
            flags.update(
                {
                    key: torch.empty(0, n, dtype=torch.bool, device=uv.device)
                    for key in ("per_view_valid", "per_view_matching")
                }
            )
            return dict(
                **floats, **flags, support=refs.clone(), normal_choice=refs.clone(), steps=[]
            )
        with torch.autocast(device_type=uv.device.type, enabled=False):
            pieces = [
                self._run(
                    inputs,
                    b,
                    refs[start : start + self.chunk],
                    uv[start : start + self.chunk],
                    initial_depth[start : start + self.chunk],
                    normals[start : start + self.chunk],
                )
                for start in range(0, q, self.chunk)
            ]
        result = {
            key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0] if key != "steps"
        }
        result["steps"] = [
            {
                key: torch.cat([piece["steps"][i][key] for piece in pieces])
                for key in pieces[0]["steps"][i]
            }
            for i in range(self.rounds)
        ]
        return result
