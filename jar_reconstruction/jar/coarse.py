"""Calibrated inverse-depth plane sweep, grouped correlation and explicit no-match."""

import torch
from torch import nn
import torch.nn.functional as F
from .geometry import (
    project,
    unproject,
    pixel_grid,
    sample_maps,
    image_texture,
    safe_masked_softmax,
)
from .encoder import norm
from .depth_sampling import configured_inverse_depths


class PlaneSweep(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.cfg = m
        c = m.get("feature_dim", 64)
        self.groups = next(
            g for g in range(min(m.get("correlation_groups", 8), c), 0, -1) if c % g == 0
        )
        self.bins = m.get("depth_bins", 48)
        near = m.get("near", 2.0)
        far = m.get("far", 80.0)
        if not (0 < near < far) or self.bins < 3:
            raise ValueError("depth range must satisfy 0 < near < far and bins >=3")
        self.register_buffer("inverse_depths", configured_inverse_depths(m, self.bins))
        self.regularizer = nn.Sequential(
            nn.Conv3d(self.groups + 4, 16, 3, padding=1),
            norm(16),
            nn.SiLU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )
        nn.init.normal_(self.regularizer[-1].weight, std=1e-3)
        nn.init.zeros_(self.regularizer[-1].bias)
        self.no_match = nn.Sequential(
            nn.Conv2d(c + 1, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 1)
        )
        nn.init.constant_(self.no_match[-1].bias, 0.0)
        settings = m.get("cost_context", {})
        if settings.get("enabled", False):
            from .models.cost_context import CostContext

            with torch.random.fork_rng(devices=[]):
                self.context = CostContext(self.groups + 4, settings)
        matching_settings = m.get("matching_evidence", {})
        self.primary_descriptor = bool(matching_settings.get("primary_descriptor", False))
        if self.primary_descriptor and not matching_settings.get("enabled", False):
            raise ValueError("primary_descriptor需要启用matching_evidence")
        if matching_settings.get("enabled", False):
            from .models.matching_evidence import PretrainedMatchingEvidence

            with torch.random.fork_rng(devices=[]):
                self.matching_evidence = PretrainedMatchingEvidence(matching_settings)
        calibration = m.get("cost_calibration", {})
        if calibration.get("enabled", False):
            if not matching_settings.get("enabled", False):
                raise ValueError("cost_calibration需要实际预训练匹配证据")
            from .models.cost_calibration import ObservationScaledCost

            self.cost_calibration = ObservationScaledCost(calibration.get("epsilon", 1e-6))
        distillation = m.get("matching_distillation", {})
        if distillation.get("enabled", False):
            if not matching_settings.get("enabled", False):
                raise ValueError("matching_distillation需要启用matching_evidence以复用冻结教师")
            from .models.matching_distillation import MatchingDistillation

            self.matching_distillation = MatchingDistillation(distillation)
        transport_settings = m.get("candidate_transport", {})
        if transport_settings.get("enabled", False):
            from .models.candidate_transport import CandidateSurfaceTransport

            with torch.random.fork_rng(devices=[]):
                self.candidate_transport = CandidateSurfaceTransport(
                    self.groups + 4, c, transport_settings
                )

    def forward(self, inputs, pyramid, stride=16, return_evidence=False):
        if stride not in (4, 8, 16, 32):
            raise ValueError("plane sweep stride must be 4, 8, 16 or 32")
        fmap = pyramid[stride]
        # 容器只属于本次完整 forward；保留原计算图供后续候选场复用。
        # 普通入口不提供容器，因此默认评分与返回接口完全不变。
        evidence_scope = inputs.get("_first_sweep_evidence")
        capture_evidence = (
            not return_evidence
            and evidence_scope is not None
            and evidence_scope["stride"] == stride
            and evidence_scope["entries"] is None
        )
        collect_evidence = return_evidence or capture_evidence
        B, N, C, h, w = fmap.shape
        references = inputs.get("_reference_indices", range(N))
        H, W = inputs["rgb"].shape[-2:]
        uvref = pixel_grid(h, w, device=fmap.device, stride=stride).reshape(-1, 2)
        P = h * w
        all_logits = []
        all_depth = []
        all_peaks = []
        all_conf = []
        all_support = []
        all_texture = []
        all_matching_weights = []
        all_matching_base_scores = []
        all_transport_updates = []
        all_per_view = []
        distillation_terms = []
        evidence_batches = []
        distill = self.training and hasattr(self, "matching_distillation")
        per_view = getattr(getattr(self, "matching_evidence", None), "per_view", None)
        prior_features = inputs.get("_pretrained_matching_features")
        matching_scale = inputs.get("_pretrained_matching_scale")
        if hasattr(self, "matching_evidence") and prior_features is None:
            raise ValueError("启用matching_evidence需要编码器提供_pretrained_matching_features")
        texture_threshold = self.cfg.get("texture_threshold", 0.002)
        for b in range(B):
            evidence_views = {}
            K = inputs["K"][b]
            T = inputs["T_cam_from_rig"][b]
            image_scale = None if matching_scale is None else matching_scale[b]
            vm = inputs.get("view_mask", torch.ones(B, N, device=fmap.device, dtype=torch.bool))[b]
            im = inputs.get("image_mask")
            im = None if im is None else im[b]
            tex = image_texture(inputs["rgb"][b])
            reftext = sample_maps(tex, uvref[None].expand(N, -1, -1))[..., 0]
            ref_logits = []
            ref_depth = []
            ref_peaks = []
            ref_conf = []
            ref_support = []
            ref_matching_weights = []
            ref_matching_base_scores = []
            ref_transport_updates = []
            ref_per_view = []
            for ref in references:
                source_cosines, source_masks = [], []
                grouped = []
                supports = []
                prior_statistics = []
                view_weights = []
                view_details = {"logits": [], "cosine": [], "valid": []}
                student_curves, teacher_curves, distillation_masks = [], [], []
                ref_feat = fmap[b, ref].permute(1, 2, 0).reshape(P, self.groups, C // self.groups)
                ref_feat = F.normalize(ref_feat.float(), dim=-1, eps=1e-6)
                for start in range(0, self.bins, self.cfg.get("sweep_chunk", 8)):
                    rho = self.inverse_depths[start : start + self.cfg.get("sweep_chunk", 8)]
                    D = len(rho)
                    z = (1 / rho)[:, None].expand(D, P)
                    points, ray = unproject(uvref[None].expand(D, -1, -1), z, K[ref], T[ref])
                    uv, zsrc, valid, J = project(points.reshape(-1, 3), K, T, (H, W), vm, im)
                    # Elementwise contraction keeps calibrated geometry fp32
                    # under autocast; a rounded GEMM can invent parallax.
                    g = (J * ray.reshape(1, -1, 1, 3)).sum(-1).norm(dim=-1)
                    local_tex = sample_maps(tex, uv)[..., 0]
                    srcmask = (
                        valid
                        & (g > 1e-5)
                        & (local_tex > texture_threshold)
                        & (reftext[ref][None].expand(D, -1).reshape(1, -1) > texture_threshold)
                    )
                    srcmask = (
                        srcmask & (torch.arange(N, device=fmap.device)[:, None] != ref) & vm[ref]
                    )
                    ref_valid = valid[ref]
                    srcmask = srcmask & ref_valid[None]
                    sampled = sample_maps(fmap[b], uv, stride=stride).reshape(
                        N, D, P, self.groups, C // self.groups
                    )
                    corr = (
                        F.normalize(sampled.float(), dim=-1, eps=1e-6) * ref_feat[None, None]
                    ).sum(-1)
                    mask = srcmask.reshape(N, D, P)
                    if hasattr(self, "matching_evidence"):
                        if per_view is None and not distill and not collect_evidence:
                            statistics = self.matching_evidence.sample_statistics(
                                prior_features[b], uvref, ref, uv, mask, image_scale
                            )
                        else:
                            cosine = self.matching_evidence.sample_cosine(
                                prior_features[b], uvref, ref, uv, mask, image_scale
                            )
                            if collect_evidence:
                                source_cosines.append(cosine)
                                source_masks.append(mask)
                            if per_view is None:
                                statistics = self.matching_evidence.statistics_from_cosine(
                                    cosine, mask
                                )
                            else:
                                statistics, weights, reliability = per_view(
                                    cosine,
                                    mask,
                                    corr.mean(-1),
                                    g.reshape(N, D, P),
                                    local_tex.reshape(N, D, P),
                                    reftext[ref][None, None],
                                )
                                view_weights.append(weights)
                                if per_view.diagnostics:
                                    view_details["logits"].append(reliability)
                                    view_details["cosine"].append(cosine)
                                    view_details["valid"].append(mask)
                            if distill:
                                student_curves.append(corr.mean(-1))
                                teacher_curves.append(cosine.detach())
                                distillation_masks.append(mask)
                        prior_statistics.append(statistics)
                    denom = mask.sum(0).clamp_min(1)
                    mean = (corr * mask[..., None]).sum(0) / denom[..., None]
                    variance = ((corr - mean[None]).square() * mask[..., None]).sum(0).mean(
                        -1
                    ) / denom
                    geom = (torch.log1p(g).reshape(N, D, P) * mask).sum(0) / denom
                    metadata = torch.stack(
                        [
                            mask.sum(0) / max(N - 1, 1),
                            variance,
                            geom,
                            reftext[ref][None].expand(D, -1),
                        ],
                        -1,
                    )
                    grouped.append(torch.cat([mean, metadata], -1))
                    supports.append(torch.where(mask.any(0), mask.sum(0) + 1, 0))
                if distill:
                    distillation_terms.append(
                        self.matching_distillation(
                            torch.cat(student_curves, dim=1),
                            torch.cat(teacher_curves, dim=1),
                            torch.cat(distillation_masks, dim=1),
                        )
                    )
                cost = (
                    torch.cat(grouped, 0)
                    .permute(2, 0, 1)
                    .reshape(1, self.groups + 4, self.bins, h, w)
                )
                sup = torch.cat(supports, 0).reshape(self.bins, h, w)
                if collect_evidence:
                    if not prior_statistics:
                        raise ValueError("候选证据场需要预训练匹配描述子")
                    evidence_views[ref] = {
                        "cost": cost,
                        "hero": torch.cat(prior_statistics, dim=1).reshape(1, 2, self.bins, h, w),
                        "source_cosine": torch.cat(source_cosines, dim=1).reshape(1, N, self.bins, h, w),
                        "source_valid": torch.cat(source_masks, dim=1).reshape(1, N, self.bins, h, w),
                        "candidate_valid": (sup >= self.cfg.get("min_support", 2))[None],
                        "inverse_depths": self.inverse_depths,
                        "intrinsic": K[ref],
                        "transform": T[ref],
                        "stride": stride,
                    }
                    if return_evidence:
                        # 独立取证入口不执行随后会被丢弃的旧评分器和深度头。
                        continue
                prior_volume = None
                base_scores = 4 * cost[:, : self.groups].mean(1)
                if self.primary_descriptor:
                    prior_volume = torch.cat(prior_statistics, dim=1).reshape(1, 2, self.bins, h, w)
                    # 专用度量负责原始相似度；本地CNN仍为正则化与上下文提供信息。
                    base_scores = 4 * prior_volume[:, 0]
                    ref_matching_base_scores.append(base_scores[0])
                scores = base_scores + self.regularizer(cost).squeeze(1)
                if hasattr(self, "context"):
                    scores = scores + self.context(cost, self.inverse_depths)
                minimum_support = self.cfg.get("min_support", 2)
                if hasattr(self, "matching_evidence"):
                    statistics = prior_volume
                    if statistics is None:
                        statistics = torch.cat(prior_statistics, dim=1).reshape(
                            1, 2, self.bins, h, w
                        )
                    correction, weight = self.matching_evidence(cost, statistics, self.groups)
                    scores = scores + correction
                    accepted = sup >= minimum_support
                    # 诊断仅统计仍合法的候选，不让权重成为新的支持或置信度定义。
                    average = (weight[0].detach() * accepted).sum(0) / accepted.sum(0).clamp_min(1)
                    ref_matching_weights.append(average)
                    if per_view is not None:
                        weights = torch.cat(view_weights, dim=1).reshape(N, self.bins, h, w)
                        averaged = (weights.detach() * accepted[None]).sum(1)
                        averaged = averaged / accepted.sum(0).clamp_min(1)[None]
                        details = {"weight_mean": averaged}
                        if per_view.diagnostics:
                            details["weights"] = weights
                            for key, chunks in view_details.items():
                                details[key] = torch.cat(chunks, dim=1).reshape(N, self.bins, h, w)
                        ref_per_view.append(details)
                if hasattr(self, "candidate_transport"):
                    accepted = (sup >= minimum_support)[None]
                    update = self.candidate_transport(
                        cost,
                        scores,
                        accepted,
                        self.inverse_depths,
                        K[ref],
                        T[ref],
                        stride,
                        fmap[b, ref : ref + 1],
                    )
                    scores = scores + update
                    average = (update[0].detach() * accepted[0]).sum(0)
                    average = average / accepted[0].sum(0).clamp_min(1)
                    ref_transport_updates.append(average)
                if hasattr(self, "cost_calibration"):
                    scores = self.cost_calibration(
                        scores, 4 * statistics[:, 0], (sup >= minimum_support)[None]
                    )
                scores = scores[0].masked_fill(sup < minimum_support, -1e4)
                no = self.no_match(
                    torch.cat([fmap[b, ref : ref + 1].float(), reftext[ref].reshape(1, 1, h, w)], 1)
                )[0]
                logits = torch.cat([scores, no], 0)
                prob = logits.softmax(0)
                conditional = scores.softmax(0)
                # Separated local peaks. Average only within +/-1 bin of each peak.
                peak_indices = []
                search = scores
                covered = torch.zeros_like(scores, dtype=torch.bool)
                axis = torch.arange(self.bins, device=scores.device)[:, None, None]
                peak_depths = []
                masses = []
                for _ in range(2):
                    idx = search.argmax(0)
                    window = ((axis - idx[None]).abs() <= 1) & ~covered
                    # Tiny D or absent support may leave no separated peak.
                    # In that case keep a finite duplicate, with zero confidence
                    # downstream if no source supports it.
                    window = torch.where(window.any(0)[None], window, axis == idx[None])
                    local = conditional * window
                    mass = local.sum(0)
                    weights = safe_masked_softmax(scores, window, dim=0)
                    rho = (weights * self.inverse_depths[:, None, None]).sum(0)
                    peak_depths.append(1 / rho.clamp_min(1e-6))
                    masses.append(mass)
                    peak_indices.append(idx)
                    covered = covered | window
                    search = scores.masked_fill(covered, -1e4)
                peaks = torch.stack(peak_depths, 0)
                conf = (1 - prob[-1]) * masses[0]
                selected_sup = sup.gather(0, peak_indices[0][None])[0]
                conf = conf * (selected_sup >= minimum_support)
                ref_logits.append(logits)
                ref_depth.append(peaks[0])
                ref_peaks.append(peaks)
                ref_conf.append(conf)
                ref_support.append(selected_sup)
            if collect_evidence:
                evidence_batches.append(evidence_views)
            if return_evidence:
                continue
            all_logits.append(torch.stack(ref_logits))
            all_depth.append(torch.stack(ref_depth))
            all_peaks.append(torch.stack(ref_peaks))
            all_conf.append(torch.stack(ref_conf))
            all_support.append(torch.stack(ref_support))
            all_texture.append(reftext.reshape(N, h, w))
            if ref_matching_weights:
                all_matching_weights.append(torch.stack(ref_matching_weights))
            if ref_matching_base_scores:
                all_matching_base_scores.append(torch.stack(ref_matching_base_scores))
            if ref_transport_updates:
                all_transport_updates.append(torch.stack(ref_transport_updates))
            if ref_per_view:
                all_per_view.append(
                    {
                        key: torch.stack([item[key] for item in ref_per_view])
                        for key in ref_per_view[0]
                    }
                )
        if return_evidence:
            return evidence_batches
        if capture_evidence:
            evidence_scope["entries"] = evidence_batches
        result = dict(
            logits=torch.stack(all_logits),
            depth=torch.stack(all_depth),
            peaks=torch.stack(all_peaks),
            confidence=torch.stack(all_conf),
            support=torch.stack(all_support),
            texture=torch.stack(all_texture),
            inverse_depths=self.inverse_depths,
            stride=stride,
        )
        if all_matching_weights:
            result["matching_evidence_weight"] = torch.stack(all_matching_weights)
        if all_matching_base_scores:
            # 未加regularizer/context/evidence及最终合法性屏蔽的原始分数。
            result["matching_base_scores"] = torch.stack(all_matching_base_scores)
        if all_transport_updates:
            result["candidate_transport_update"] = torch.stack(all_transport_updates)
        if all_per_view:
            # mean: [B,reference,source,Hc,Wc]；完整诊断额外保留depth轴。
            for key in all_per_view[0]:
                result[f"matching_per_view_{key}"] = torch.stack(
                    [item[key] for item in all_per_view]
                )
        if distill:
            numerator = sum(item[0] for item in distillation_terms)
            denominator = sum(item[1] for item in distillation_terms)
            result["matching_distillation_loss"] = numerator / denominator.clamp_min(1e-8)
        elif hasattr(self, "matching_distillation"):
            # 评估入口可能复用训练criterion；只保留零标量，不计算教师KL。
            result["matching_distillation_loss"] = result["logits"].reshape(-1)[:0].sum()
        if "_reference_indices" in inputs:
            # 保持[B,N,...]公共接口。未计算参考没有候选证据，不能作为监督或表面。
            indices = torch.as_tensor(references, device=fmap.device, dtype=torch.long)
            for key, value in list(result.items()):
                if (
                    key in ("texture", "inverse_depths")
                    or not torch.is_tensor(value)
                    or value.ndim < 2
                ):
                    continue
                if key == "logits":
                    restored = value.new_full((B, N, *value.shape[2:]), -1e4)
                    restored[:, :, -1] = 0
                else:
                    restored = value.new_zeros((B, N, *value.shape[2:]))
                result[key] = restored.index_copy(1, indices, value)
            supervised = torch.zeros(B, N, device=fmap.device, dtype=torch.bool)
            supervised[:, indices] = True
            result["supervised_views"] = supervised
            result["reference_indices"] = tuple(references)
        return result
