"""按需原分辨率多焦距有向精修；继承ET更新，分开广角和长焦深度范围。"""

from copy import deepcopy
import torch
from jar.geometry import project
from jar.models.oriented_polish import OrientedRayPolisher
from .matching import multifocal_patch_cost


class _PlanePolisher(OrientedRayPolisher):
    def __init__(self, cfg, calibrated_span=False):
        super().__init__(cfg)
        options = cfg.get("model", cfg).get("multifocal", {})
        self.tele_refinement = bool(options.get("tele_refinement", True))
        self.antialias = bool(options.get("common_scale_antialias", True))
        self.aa_samples = int(options.get("antialias_samples", 2))
        self.min_resolution_px = float(options.get("min_resolution_px", 1.5))
        self.calibrated_span = calibrated_span
        self.target_pixels = float(options.get("tele_refine_step_pixels", 1.0))
        self.max_step = float(options.get("tele_refine_max_step_m", 10.0))
        if self.tele_refinement:
            self.radius = int(options.get("tele_patch_radius", 1))

    def _score(self, inputs, b, refs, uv, depth, normals, include_fronto):
        if not self.tele_refinement:
            return super()._score(inputs, b, refs, uv, depth, normals, include_fronto)
        return multifocal_patch_cost(
            inputs,
            b,
            refs,
            uv,
            depth,
            normals,
            include_fronto=include_fronto,
            patch_radius=self.radius,
            query_chunk=self.chunk,
            antialias=self.antialias,
            aa_samples=self.aa_samples,
            texture_threshold=self.texture_threshold,
            census_temperature=self.census_temperature,
            grazing_cosine=self.grazing_cosine,
            agreement_cost=self.match_threshold,
            min_sources=1,
            method=self.descriptor,
            min_resolution_px=self.min_resolution_px,
        )

    def _inverse_depth_span(self, inputs, b, refs, uv, depth, iteration):
        if not self.calibrated_span or not self.tele_refinement:
            return super()._inverse_depth_span(inputs, b, refs, uv, depth, iteration)
        centers, rays = self._rays(inputs, b, refs, uv)
        point = centers + depth[:, None] * rays
        vm, im = inputs.get("view_mask"), inputs.get("image_mask")
        _, _, valid, J = project(
            point,
            inputs["K"][b],
            inputs["T_cam_from_rig"][b],
            inputs["rgb"].shape[-2:],
            None if vm is None else vm[b],
            None if im is None else im[b],
        )
        source = torch.arange(inputs["rgb"].shape[1], device=refs.device)[:, None] != refs[None]
        derivative = (J * rays[None, :, None]).sum(-1).norm(dim=-1)
        valid = valid & source & (derivative > 1e-5)
        ruler = derivative.masked_fill(~valid, 0.0).amax(0)
        delta = (self.target_pixels / ruler.clamp_min(1e-8)).clamp(0.005, self.max_step)
        # Convert a bounded metric update to inverse-depth spacing; no fixed
        # wide-camera inverse bin is reused at 150m. Degenerate rays stay put.
        span = delta / (depth * (depth + delta)).clamp_min(1e-6)
        return torch.where(
            valid.any(0), span * self.initial_span * self.shrink**iteration, 0.0
        ).detach()


class MFOrientedPolisher(_PlanePolisher):
    """原forward接口，内部按稳定camera_roles拆分near/far后恢复原query顺序。

    tele_refinement=False使用各自范围内的原版窗口算法；True启用共同带宽、
    小窗口、长焦标定步长。全局模式唯一性由原解码器及最终验收模块检查。
    """

    def __init__(self, cfg, sampler=None):
        super().__init__(cfg)
        tele_cfg = deepcopy(cfg)
        model = tele_cfg.setdefault("model", {})
        options = model.get("multifocal", {})
        model["near"] = float(options.get("tele_near", 10.0))
        model["far"] = float(options.get("tele_far", 150.0))
        self.tele_polisher = _PlanePolisher(tele_cfg, calibrated_span=True)

    def forward(self, inputs, b, refs, uv, initial_depth, normals):
        roles = inputs.get("camera_roles")
        if roles is None or not len(refs):
            return super().forward(inputs, b, refs, uv, initial_depth, normals)
        tele = roles[b, refs] == 1
        if not tele.any():
            return super().forward(inputs, b, refs, uv, initial_depth, normals)
        if tele.all():
            return self.tele_polisher(inputs, b, refs, uv, initial_depth, normals)
        chunks, order = [], []
        for selected, module in ((~tele, None), (tele, self.tele_polisher)):
            ids = selected.nonzero().flatten()
            args = (inputs, b, refs[ids], uv[ids], initial_depth[ids], normals[ids])
            chunks.append(super().forward(*args) if module is None else module(*args))
            order.append(ids)
        inverse = torch.cat(order).argsort()
        result = {
            key: torch.cat([chunk[key] for chunk in chunks])[inverse]
            for key in chunks[0]
            if key != "steps"
        }
        result["steps"] = [
            {
                key: torch.cat([chunk["steps"][i][key] for chunk in chunks])[inverse]
                for key in chunks[0]["steps"][i]
            }
            for i in range(self.rounds)
        ]
        return result
