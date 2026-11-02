"""无需训练的已标定多视图参考算法；不是任何论文网络的官方复现。

每个候选深度上的整个参考小窗口都精确投影。前向平面假设在倾斜面、
细物体和遮挡边界会失效，这些限制必须与神经网络结果一起报告。
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..geometry import project, unproject, pixel_grid, sample_maps, validate_inputs


class ClassicalMVS(nn.Module):
    """RGB ZNCC 或软 Census 的分块平面扫描，支持 CPU/CUDA。

    ``cfg['baseline']``: stride=4, patch_radius=2, depth_chunk=8,
    query_chunk=256, refinement_bins=9, min_score=0.55/0.50,
    min_margin=0.005, texture_threshold=0.006, consistency=True。
    深度范围与 bins 默认沿用 ``cfg['model']``，无训练参数。
    """

    def __init__(self, cfg, method="zncc"):
        super().__init__()
        self.cfg = cfg
        self.options = dict(cfg.get("baseline", {}))
        self.method = method.removeprefix("classical_")
        if self.method not in ("zncc", "census"):
            raise ValueError("method must be zncc or census")
        m = cfg.get("model", {})
        self.near, self.far = float(m.get("near", 2)), float(m.get("far", 80))
        bins = int(self.options.get("depth_bins", m.get("depth_bins", 128)))
        self.stride = int(self.options.get("stride", 4))
        self.radius = int(self.options.get("patch_radius", 2))
        self.depth_chunk = int(self.options.get("depth_chunk", 8))
        self.query_chunk = int(self.options.get("query_chunk", 256))
        self.refine_bins = int(self.options.get("refinement_bins", 9))
        self.minimum_support = int(self.options.get("min_support", m.get("min_support", 2)))
        if not (0 < self.near < self.far and bins >= 3):
            raise ValueError("0 < near < far and depth_bins >= 3 required")
        if min(self.stride, self.radius, self.depth_chunk, self.query_chunk) < 1:
            raise ValueError("stride, patch radius and chunk sizes must be positive")
        if self.refine_bins and (self.refine_bins < 3 or self.refine_bins % 2 != 1):
            raise ValueError("refinement_bins must be zero or odd and >= 3")
        if not 0 < float(self.options.get("view_keep_fraction", 0.67)) <= 1:
            raise ValueError("view_keep_fraction must be in (0, 1]")
        if self.minimum_support < 2:
            raise ValueError("a calibrated matching reference needs at least two supporting views")
        self.register_buffer("inverse_depths", torch.linspace(1 / self.far, 1 / self.near, bins))
        offsets = pixel_grid(2 * self.radius + 1, 2 * self.radius + 1) - self.radius
        self.register_buffer("offsets", offsets.reshape(-1, 2))

    def _scores(self, rgb, K, T, vm, im, reference, uv, rhos):
        """Returns [D,P] cost and count; all points, including patch offsets, project exactly."""
        n, _, h, w = rgb.shape
        d, p = rhos.shape
        patches = uv[:, None] + self.offsets[None]
        count = patches.shape[1]
        z = rhos.reciprocal()[..., None].expand(d, p, count)
        points, directions = unproject(
            patches[None].expand(d, -1, -1, -1), z, K[reference], T[reference]
        )
        proj, _, mask, jacobian = project(points.reshape(-1, 3), K, T, (h, w), vm, im)
        values = sample_maps(rgb, proj).reshape(n, d, p, count, 3)
        all_valid = mask.reshape(n, d, p, count).all(-1)
        # Same optical center / pure rotation has zero depth information even
        # when texture matching looks perfect. Use the central ray derivative.
        c = count // 2
        derivative = (
            (jacobian.reshape(n, d, p, count, 2, 3)[:, :, :, c] * directions[None, :, :, c, None])
            .sum(-1)
            .norm(dim=-1)
        )
        valid = all_valid & all_valid[reference : reference + 1] & (derivative > 1e-5)
        valid[reference] = False
        reference_values = sample_maps(rgb[reference : reference + 1], patches.reshape(1, -1, 2))[0]
        reference_values = reference_values.reshape(p, count, 3)
        a = values - values.mean(-2, keepdim=True)
        b = reference_values - reference_values.mean(-2, keepdim=True)
        texture = b.square().mean((-2, -1)).sqrt()
        texture_src = a.square().mean((-2, -1)).sqrt()
        threshold = self.options.get("texture_threshold", 0.006)
        valid &= (texture[None, None] > threshold) & (texture_src > threshold)
        if self.method == "zncc":
            numerator = (a * b[None, None]).sum((-2, -1))
            denominator = (a.square().sum((-2, -1)) * b.square().sum((-2, -1))[None, None]).sqrt()
            pair_score = numerator / denominator.clamp_min(1e-8)
        else:
            # Soft Census preserves the rank comparison logic but avoids binary
            # discontinuities during subpixel interpolation; not a claim to
            # reproduce a particular Census+SGM implementation.
            luma = rgb.new_tensor([0.299, 0.587, 0.114])
            gray_src = (values * luma).sum(-1)
            gray_ref = (reference_values * luma).sum(-1)
            scale = float(self.options.get("census_temperature", 0.02))
            if scale <= 0:
                raise ValueError("census_temperature must be positive")
            src = torch.tanh((gray_src - gray_src[..., c : c + 1]) / scale)
            ref = torch.tanh((gray_ref - gray_ref[..., c : c + 1]) / scale)
            difference = (src - ref[None, None]).abs()
            # Exclude the trivially equal central comparison.
            pair_score = 1 - difference.sum(-1) / max(count - 1, 1)
        pair_score = pair_score.clamp(-1, 1).masked_fill(~valid, -1e4)
        # A lower-tail trimmed mean tolerates a minority occluded camera.
        ordered = pair_score.sort(0, descending=True).values
        available = valid.sum(0)
        keep = (
            torch.ceil(available.float() * float(self.options.get("view_keep_fraction", 0.67)))
            .long()
            .clamp_min(1)
        )
        ranks = torch.arange(n, device=rgb.device)[:, None, None]
        used = (ranks < keep[None]) & (ranks < available[None])
        score = torch.where(used, ordered, 0).sum(0) / used.sum(0).clamp_min(1)
        minimum = float(self.options.get("min_score", 0.55 if self.method == "zncc" else 0.50))
        support = ((pair_score >= minimum) & valid).sum(0)
        support = torch.where(support > 0, support + 1, 0)
        score = score.masked_fill(support < self.minimum_support, -1e4)
        return score, support, texture

    def _reference(self, rgb, K, T, vm, im, reference, uv):
        depths, confidence, supports, textures, accepted, volumes = [], [], [], [], [], []
        axis = torch.arange(len(self.inverse_depths), device=rgb.device)[:, None]
        rho_step = self.inverse_depths[1] - self.inverse_depths[0]
        for pixels in uv.split(self.query_chunk):
            scores, counts = [], []
            for rho in self.inverse_depths.split(self.depth_chunk):
                score, support, texture = self._scores(
                    rgb, K, T, vm, im, reference, pixels, rho[:, None].expand(-1, len(pixels))
                )
                scores.append(score)
                counts.append(support)
            scores, counts = torch.cat(scores), torch.cat(counts)
            best, index = scores.max(0)
            count = counts.gather(0, index[None])[0]
            # Do not count immediately adjacent bins as another surface mode.
            competitor = scores.masked_fill((axis - index[None]).abs() <= 1, -1e4).amax(0)
            margin = best - competitor
            valid = (best > -1e3) & (margin >= self.options.get("min_margin", 0.005))
            rho = self.inverse_depths[index]
            if self.refine_bins:
                local_rho = (
                    rho[None]
                    + torch.linspace(-1, 1, self.refine_bins, device=rgb.device)[:, None] * rho_step
                )
                local_rho = local_rho.clamp(self.inverse_depths[0], self.inverse_depths[-1])
                score, local_count, _ = self._scores(
                    rgb, K, T, vm, im, reference, pixels, local_rho
                )
                fine_best, fine_index = score.max(0)
                rho = local_rho.gather(0, fine_index[None])[0]
                fine_count = local_count.gather(0, fine_index[None])[0]
                count = torch.where(valid, fine_count, count)
                # Quadratic interpolation only within a finite concave local
                # bracket; exact-projection samples, not a geometric Taylor warp.
                left = (fine_index - 1).clamp_min(0)
                right = (fine_index + 1).clamp_max(self.refine_bins - 1)
                sl = score.gather(0, left[None])[0]
                sr = score.gather(0, right[None])[0]
                denominator = sl - 2 * fine_best + sr
                interior = (fine_index > 0) & (fine_index < self.refine_bins - 1)
                smooth = interior & (sl > -1e3) & (sr > -1e3) & (denominator < -1e-7)
                shift = (
                    0.5 * (sl - sr) / torch.where(smooth, denominator, torch.ones_like(denominator))
                )
                rho = rho + torch.where(smooth, shift.clamp(-0.5, 0.5), 0) * (
                    2 * rho_step / (self.refine_bins - 1)
                )
                rho = rho.clamp(self.inverse_depths[0], self.inverse_depths[-1])
                best = fine_best
                valid &= (fine_best > -1e3) & (fine_count >= self.minimum_support)
            conf = ((best + 1) * 0.5).clamp(0, 1) * (margin / 0.05).clamp(0, 1)
            depths.append(rho.clamp_min(1e-8).reciprocal())
            confidence.append(conf * valid)
            supports.append(torch.where(valid, count, 0))
            textures.append(texture)
            accepted.append(valid)
            volumes.append(scores)
        return tuple(
            torch.cat(v, -1) for v in (depths, confidence, supports, textures, accepted, volumes)
        )

    @torch.no_grad()
    def forward(self, inputs):
        validate_inputs(inputs)
        # Calibration math stays fp32 even if callers enter AMP.
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            return self._forward(inputs)

    def _forward(self, inputs):
        rgb = inputs["rgb"].float()
        batch, n, _, h, w = rgb.shape
        gh, gw = math.ceil(h / self.stride), math.ceil(w / self.stride)
        uv = pixel_grid(gh, gw, rgb.device, stride=self.stride).reshape(-1, 2)
        all_depth, all_conf, all_support, all_texture, all_valid, all_logits, scenes = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        all_raw_valid = []
        for b in range(batch):
            K, T = inputs["K"][b], inputs["T_cam_from_rig"][b]
            vm = inputs.get("view_mask", torch.ones(batch, n, dtype=torch.bool, device=rgb.device))[
                b
            ]
            im = inputs.get("image_mask")
            im = im[b] if im is not None else None
            result = [self._reference(rgb[b], K, T, vm, im, ref, uv) for ref in range(n)]
            depth, conf, support, texture, valid, logits = [
                torch.stack([r[i] for r in result]) for i in range(6)
            ]
            # Expose exactly the independently accepted inputs to the existing
            # one-pass cycle check. This adds diagnostics, not another filter.
            all_raw_valid.append(valid.reshape(n, gh, gw).clone())
            if self.options.get("consistency", True):
                # Cross-check independently inferred depths; occluded back
                # surfaces are never converted into free-space predictions.
                checked, verified_counts = [], []
                for ref in range(n):
                    points, _ = unproject(uv, depth[ref], K[ref], T[ref])
                    projected, expected, geometrical, _ = project(points, K, T, (h, w), vm, im)
                    other_depth = sample_maps(
                        depth.reshape(n, 1, gh, gw), projected, self.stride, mode="nearest"
                    )[..., 0]
                    other_valid = (
                        sample_maps(
                            valid.reshape(n, 1, gh, gw).float(),
                            projected,
                            self.stride,
                            mode="nearest",
                        )[..., 0]
                        > 0.5
                    )
                    tolerance = self.options.get(
                        "consistency_absolute_m", 0.20
                    ) + expected.abs() * self.options.get("consistency_relative", 0.04)
                    agreement = (
                        geometrical & other_valid & ((other_depth - expected).abs() <= tolerance)
                    )
                    agreement[ref] = False
                    count = agreement.sum(0) + 1
                    checked.append(valid[ref] & (count >= self.minimum_support))
                    verified_counts.append(count)
                valid = torch.stack(checked)
                support = torch.where(
                    valid, torch.minimum(support, torch.stack(verified_counts)), 0
                )
                conf = conf * valid
            points, normals, scales = [], [], []
            for ref in range(n):
                x, ray = unproject(uv, depth[ref], K[ref], T[ref])
                normal = -F.normalize(ray, dim=-1)
                image_x = x.reshape(gh, gw, 3)
                if gh > 2 and gw > 2:
                    dx = image_x[1:-1, 2:] - image_x[1:-1, :-2]
                    dy = image_x[2:, 1:-1] - image_x[:-2, 1:-1]
                    candidate = F.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1)
                    facing = (candidate * ray.reshape(gh, gw, 3)[1:-1, 1:-1]).sum(-1) > 0
                    candidate = torch.where(facing[..., None], -candidate, candidate)
                    mask = valid[ref].reshape(gh, gw)
                    neighbors_ok = (
                        mask[1:-1, 2:] & mask[1:-1, :-2] & mask[2:, 1:-1] & mask[:-2, 1:-1]
                    )
                    normal = normal.reshape(gh, gw, 3).clone()
                    normal[1:-1, 1:-1] = torch.where(
                        (neighbors_ok & (candidate.norm(dim=-1) > 0.5))[..., None],
                        candidate,
                        normal[1:-1, 1:-1],
                    )
                    normal = normal.reshape(-1, 3)
                scale = depth[ref, :, None] * self.stride / K[ref, :2, :2].diagonal().mean() * 0.5
                points.append(x)
                normals.append(normal)
                scales.append(scale.expand(-1, 3).clamp_min(0.005))
            points, normals, scales = torch.cat(points), torch.cat(normals), torch.cat(scales)
            state_valid = valid.reshape(-1)
            roi = self.cfg.get("model", {}).get("roi")
            if roi is not None:
                bounds = points.new_tensor(roi)
                state_valid = state_valid & (
                    (points >= bounds[:, 0]) & (points <= bounds[:, 1])
                ).all(-1)
            state = torch.where(state_valid, 1, 2)
            # Keep every rejected query as UNKNOWN. No unobserved free labeling.
            scene = dict(
                points=points,
                normals=normals,
                scales=scales,
                valid=state_valid,
                support=support.reshape(-1),
                confidence=conf.reshape(-1),
                state=state,
                state_logits=F.one_hot(state, 3).float() * 12 - 6,
                centers=points,
                sizes=scales * 2,
                features=rgb.new_empty(len(points), 0),
                split_logits=rgb.new_zeros(len(points)),
                baseline_method=self.method,
            )
            # Scene comparison can use the same point budget as neural models.
            # A fixed per-camera uniform lattice subset is independent of GT,
            # score and validity; rejected queries remain in that subset.
            source_count = len(points)
            cap = int(self.cfg.get("model", {}).get("max_queries", source_count))
            if cap < 1:
                raise ValueError("model.max_queries must be positive")
            if cap < source_count:
                per_camera, extra = divmod(cap, n)
                selected = []
                for camera in range(n):
                    quota = per_camera + int(camera < extra)
                    if quota:
                        local = torch.linspace(0, gh * gw - 1, quota, device=rgb.device).long()
                        selected.append(camera * gh * gw + local)
                selected = torch.cat(selected)
                scene = {
                    key: value[selected] if isinstance(value, torch.Tensor) else value
                    for key, value in scene.items()
                }
            scene["aux"] = dict(
                source_candidate_count=source_count,
                output_query_count=len(scene["points"]),
                selection="fixed_uniform_per_camera_including_unknown",
                depth_stride=self.stride,
                roi=roi,
            )
            scenes.append(scene)
            all_depth.append(depth.reshape(n, gh, gw))
            all_conf.append(conf.reshape(n, gh, gw))
            all_support.append(support.reshape(n, gh, gw))
            all_texture.append(texture.reshape(n, gh, gw))
            all_valid.append(valid.reshape(n, gh, gw))
            no_match = torch.where(valid, torch.full_like(depth, -1e4), torch.zeros_like(depth))
            all_logits.append(torch.cat([logits, no_match[:, None]], 1).reshape(n, -1, gh, gw))
        depth, conf, support, texture, valid = map(
            torch.stack, (all_depth, all_conf, all_support, all_texture, all_valid)
        )
        coarse = dict(
            depth=depth,
            confidence=conf,
            support=support,
            texture=texture,
            peaks=depth[:, :, None].expand(-1, -1, 2, -1, -1),
            logits=torch.stack(all_logits),
            inverse_depths=self.inverse_depths,
            stride=self.stride,
            valid=valid,
        )
        return dict(
            scenes=scenes,
            coarse=coarse,
            dense_depth=depth,
            dense_valid=valid,
            depth_stride=self.stride,
            raw_dense_depth=depth,
            raw_dense_valid=torch.stack(all_raw_valid),
        )


def build_baseline(cfg, method="zncc"):
    return ClassicalMVS(cfg, method)
