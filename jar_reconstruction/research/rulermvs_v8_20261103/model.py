"""Independent V8 scalar-geometry experiments; every switch off is exact V6.

No additional learned parameters. ``information_scale`` is a dimensionless
numerical normalization, not a scene/unit estimate or a fitted metric scale.
The default 10 preserves the old information magnitude at internal Z=10 only;
it does not alter the unit invariance of Z*d(pixel)/dZ.
"""

import copy
import math

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, project, safe_masked_softmax, sample_maps, unproject
from jar.pretraining.model import CandidateViewAttention
from jar.pretraining.projective_ruler import compile_projective_ruler
from research.rulermvs_v6_20261101.model import V6Decoder, V6Model


DEFAULT_OPTIONS = {
    "information_mode": "legacy",
    "information_scale": 10.,
    "source_count_strength": 0.,
    "source_count_reference": 3.,
    "relative_validity": False,
    "relative_validity_threshold": 1e-5,
    "physical_thresholds": False,
    "proposal_scale_reference": "input",
    "position_mode": "range",
}


def validate_options(options=None):
    unknown = set(options or {}) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown V8 options: {sorted(unknown)}")
    result = {**DEFAULT_OPTIONS, **(options or {})}
    if result["information_mode"] not in ("legacy", "log_depth"):
        raise ValueError("information_mode must be legacy or log_depth")
    if result["position_mode"] != "range":
        raise ValueError("Only the original range position is implemented; parallax is a proposal")
    if result["proposal_scale_reference"] not in ("input", "nominal"):
        raise ValueError("proposal_scale_reference must be input or nominal")
    for key in ("information_scale", "source_count_reference", "relative_validity_threshold"):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = value
    strength = float(result["source_count_strength"])
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("source_count_strength must be in [0,1]")
    result["source_count_strength"] = strength
    for key in ("relative_validity", "physical_thresholds"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be boolean")
    return result


class V8SourceAttention(CandidateViewAttention):
    """The same four-head attention, with an optional source-count null prior.

    The enabled branch retains V6's compact-pair and reference-hoisting options.
    All-zero strength delegates to the original method, including roundoff order.
    """

    def configure(self, options):
        self.count_strength = options["source_count_strength"]
        self.count_reference = options["source_count_reference"]

    def forward(self, reference, source, source_valid, ray_derivative,
                cell_half_width, use_geometry=True):
        if self.count_strength == 0:
            return super().forward(reference, source, source_valid, ray_derivative,
                                   cell_half_width, use_geometry)
        batch, views, channels, depth, height, width = source.shape
        pixels = reference.permute(0, 2, 3, 1)
        anchor = pixels[:, None].expand(-1, depth, -1, -1, -1)
        projection_input = pixels if self.hoist_reference else anchor
        query = self.query(projection_input).unflatten(-1, (self.heads, channels // self.heads))
        null = self.null(projection_input).float()
        if self.hoist_reference:
            query = query[:, None]
            null = null[:, None].expand(-1, depth, -1, -1, -1)
        count = source_valid.sum(1).float().clamp_min(1)
        null = null + self.count_strength * (count / self.count_reference).log()[..., None]

        compact = self.compact_pair and not torch.is_autocast_enabled(source.device.type)
        if compact:
            weight = self.pair[0].weight
            grouped = weight[:, 2 * channels:2 * channels + self.groups]
            product_weight = weight[:, :channels] + grouped.repeat_interleave(channels // self.groups, 1)
            weight = torch.cat((product_weight, weight[:, channels:2 * channels], weight[:, -3:]), 1)
            padding = (-weight.shape[1]) % 16
            pair_weight = F.pad(weight, (0, padding))
        half = torch.as_tensor(cell_half_width, device=source.device).float()
        if half.ndim == 1:
            half = half[None, :, None, None]
        values, logits = [], []
        for view in range(views):
            # _observe already supplies finite zero descriptors for invalid
            # sources; avoid another C-channel candidate-volume allocation.
            sampled = source[:, view].permute(0, 2, 3, 4, 1)
            derivative = ray_derivative[:, view].float().movedim(1, -1)
            # Invalid observations never contribute a value, including 0*NaN.
            derivative = torch.where(source_valid[:, view, ..., None], derivative, 0.)
            length = torch.linalg.vector_norm(derivative, dim=-1)
            direction = derivative / length[..., None].clamp_min(1e-6)
            geometry = torch.cat((direction, (length * half).log1p().clamp_max(8)[..., None]), -1)
            geometry = geometry * float(use_geometry)
            product, difference = anchor * sampled, (anchor - sampled).abs()
            if compact:
                zeros = product.new_zeros((*product.shape[:-1], padding))
                pair = torch.cat((product, difference, geometry, zeros), -1)
                value = self.pair[2](self.pair[1](F.linear(pair, pair_weight, self.pair[0].bias)))
                del zeros, pair
            else:
                grouped = product.unflatten(-1, (self.groups, channels // self.groups)).sum(-1)
                value = self.pair(torch.cat((product, difference, grouped, geometry), -1))
                del grouped
            del product, difference, geometry
            key = self.key(sampled).unflatten(-1, (self.heads, channels // self.heads))
            score = (query.float() * key.float()).sum(-1) / math.sqrt(channels // self.heads)
            values.append(value)
            logits.append(score + self.reliability(value).float())
        valid = torch.cat((source_valid, torch.ones_like(source_valid[:, :1])), 1)
        weights = safe_masked_softmax(torch.cat((torch.stack(logits, 1), null[:, None]), 1),
                                      valid[..., None], dim=1)
        values = torch.stack(values, 1).unflatten(-1, (self.heads, channels // self.heads))
        pooled = (values * weights[:, :views, ..., None]).sum(1).flatten(-2).movedim(-1, 1)
        mean_weight = weights[:, :views].mean(-1)
        safe_derivative = torch.where(source_valid[:, :, None], ray_derivative, 0.)
        information = (mean_weight * safe_derivative.square().sum(2)).sum(1, keepdim=True)
        return {"pooled": pooled, "weights": mean_weight,
                "null_mass": weights[:, views:].mean(-1), "information": information}


class V8Decoder(V6Decoder):
    def __init__(self, settings, handoff, options=None):
        super().__init__(copy.deepcopy(settings), handoff)
        self.configure(options)

    def configure(self, options=None):
        self.v8_options = validate_options(options)
        self.settings = copy.deepcopy(self.settings)
        self._physical_settings = {
            "peak_separation_m": float(self.settings.get("peak_separation_m", 1.)),
            "fine_min_step_m": float(self.settings.get("fine_min_step_m", .05)),
            "fine_max_step_m": float(self.settings.get("fine_max_step_m", 8.)),
        }
        if any(not math.isfinite(x) or x < 0 for x in self._physical_settings.values()):
            raise ValueError("Physical candidate thresholds must be finite and nonnegative")
        if self._physical_settings["fine_max_step_m"] < self._physical_settings["fine_min_step_m"]:
            raise ValueError("fine_max_step_m must be >= fine_min_step_m")
        self.unit_scale = 1.
        self.source.__class__ = V8SourceAttention
        self.source.configure(self.v8_options)

    def set_unit_scale(self, scale, nominal_scale=None):
        """Set internal units; nominal_scale is the declared pre-jitter scale.

        If omitted, the current scale is treated as nominal. Callers applying
        unit jitter must carry the original nominal_depth_scale in the domain;
        changing depth_scale alone does not declare a pure unit conversion.
        """
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        self.unit_scale = scale
        if self.v8_options["physical_thresholds"]:
            # Private settings only: serialized config and previous domains stay intact.
            if self.v8_options["proposal_scale_reference"] == "nominal":
                nominal_scale = scale if nominal_scale is None else float(nominal_scale)
                if not math.isfinite(nominal_scale) or nominal_scale <= 0:
                    raise ValueError("nominal_depth_scale must be finite and positive")
                ratio = nominal_scale / scale
                for name, configured_value in self._physical_settings.items():
                    self.settings[name] = configured_value * ratio
            else:
                # Preserve the original V8 input-unit convention and operation order.
                for name, physical_value in self._physical_settings.items():
                    self.settings[name] = physical_value / scale

    def _tokens(self, prepared, observation, candidates, stride):
        if self.v8_options["information_mode"] == "legacy":
            return super()._tokens(prepared, observation, candidates, stride)
        # The source interface is unchanged. Using d(pixel)/d(log Z) and h/Z
        # preserves the existing projected half-span |g|h, while making its
        # direction epsilon and information independent of internal length units.
        half = observation["cell_half_width"]
        if half.ndim == 1:
            half = half[None, :, None, None]
        transformed = dict(observation,
                           ray_derivative=observation["ray_derivative"] * candidates[:, None, None],
                           cell_half_width=half / candidates)
        return super()._tokens(prepared, transformed, candidates, stride)

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        if self.v8_options["information_mode"] == "log_depth":
            source = dict(source, information=source["information"] / self.v8_options["information_scale"] ** 2)
        return super()._tokens_from_source(prepared, observation, candidates, stride, source)

    def _observe(self, prepared, candidates, stride, half_width):
        if not self.v8_options["relative_validity"]:
            return super()._observe(prepared, candidates, stride, half_width)
        # This localized override mirrors CalibratedDecoder._observe. Projection,
        # FPN routing and descriptor sampling are identical; only source
        # eligibility uses |d(pixel)/d(log Z)| instead of |d(pixel)/dZ|.
        batch, count, height, width = candidates.shape
        reference, channels = prepared["reference"], self.settings["channels"]
        views = prepared["K"].shape[1]
        compact = self.settings.get("compact_sources", False) and views > 1
        indices = [index for index in range(views) if not compact or index != reference]
        grid = pixel_grid(height, width, device=candidates.device, stride=stride).reshape(-1, 2)
        records = []
        for sample in range(batch):
            K, T = prepared["K"][sample], prepared["T"][sample]
            with torch.autocast(device_type=candidates.device.type, enabled=False):
                scale = prepared["scale"][sample]
                widths = half_width if half_width.ndim == 1 else half_width[sample]
                widths = widths[:, None].expand(-1, height * width).flatten() if widths.ndim == 1 else widths.flatten()
                z = candidates[sample].float().flatten()
                if self.settings.get("projective_ruler", False):
                    ruler = compile_projective_ruler(grid, K[reference], T[reference], K[indices], T[indices])
                    depths = candidates[sample].float().flatten(1)
                    geometry = ruler.project(depths, prepared["image_size"], prepared["view_mask"][sample, indices],
                                             prepared["image_mask"][sample, indices])
                    uv, valid = geometry["uv"].flatten(1, 2), geometry["valid"].flatten(1, 2)
                    derivative = geometry["depth_derivative"].flatten(1, 2)
                    footprint = geometry["pixel_derivative"].flatten(1, 2)
                    spans = torch.stack((
                        torch.linalg.vector_norm(footprint[..., 0] * scale, dim=-1) * stride,
                        torch.linalg.vector_norm(footprint[..., 1] * scale, dim=-1) * stride,
                        torch.linalg.vector_norm(derivative * scale, dim=-1) * widths * 2), -1)
                    needs_points = self.use_scene or self.local_kind in {"plane_flat", "plane_metric"}
                    points = ruler.world_points(depths) if needs_points else None
                else:
                    points, rays = unproject(grid[None].expand(count, -1, -1), candidates[sample].float().flatten(1),
                                             K[reference], T[reference])
                    uv, _, valid, jacobian = project(points.reshape(-1, 3), K[indices], T[indices], prepared["image_size"],
                                                     prepared["view_mask"][sample, indices], prepared["image_mask"][sample, indices])
                    derivative = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1)
                    dx = torch.linalg.inv(K[reference])[:, 0] @ T[reference, :3, :3]
                    dy = torch.linalg.inv(K[reference])[:, 1] @ T[reference, :3, :3]
                    spans = torch.stack((
                        torch.linalg.vector_norm((jacobian @ dx) * scale, dim=-1) * z * stride,
                        torch.linalg.vector_norm((jacobian @ dy) * scale, dim=-1) * z * stride,
                        torch.linalg.vector_norm(derivative * scale, dim=-1) * widths * 2), -1)
                level = (spans.amax(-1).clamp_min(1) / 16).log2().clamp(0, 2)
                if not self.settings.get("scale_routing", True):
                    level = torch.zeros_like(level)
                feature_uv = (uv + .5) * scale - .5
            sampled = 0
            for index, feature_stride in enumerate((4, 8, 16)):
                maps = prepared["pyramid"][feature_stride][sample, indices]
                weight = (1 - (level - index).abs()).clamp_min(0)
                sampled = sampled + sample_maps(maps.float(), feature_uv, stride=feature_stride, padding_mode="border") * weight[..., None]
            anchor_uv = (grid[None] + .5) * scale - .5
            anchor = sample_maps(prepared["pyramid"][4][sample, reference:reference + 1].float(),
                                 anchor_uv, stride=4, padding_mode="border")[0].T.reshape(channels, height, width)
            ref_valid = sample_maps(prepared["image_mask"][sample, reference:reference + 1, None].float(),
                                    grid[None])[0, :, 0] > .999
            ref_valid = ref_valid & prepared["view_mask"][sample, reference]
            sensitivity = torch.linalg.vector_norm(derivative * z[None, :, None], dim=-1)
            valid = valid & (sensitivity > self.v8_options["relative_validity_threshold"])
            if not compact:
                valid[reference] = False
            valid = valid.reshape(-1, count, height, width) & ref_valid.reshape(1, 1, height, width)
            source = sampled.transpose(1, 2).reshape(-1, channels, count, height, width)
            source = F.normalize(torch.where(valid[:, None], source, 0), dim=1, eps=1e-6)
            records.append({"reference": F.normalize(anchor, dim=0, eps=1e-6), "source": source,
                            "source_valid": valid, "ref_valid": ref_valid.reshape(1, height, width),
                            "ray_derivative": (derivative * scale / 4).transpose(1, 2).reshape(-1, 2, count, height, width),
                            "points": points.reshape(count, height, width, 3).permute(3, 0, 1, 2) if points is not None else None})
        result = {key: torch.stack([record[key] for record in records]) for key in records[0] if records[0][key] is not None}
        result.update(cell_half_width=half_width, stride=stride, reference_index=reference, view_count=views)
        return result


class V8Model(V6Model):
    architecture = "rulermvs_v8"

    def __init__(self, config, initialize=True):
        super().__init__(copy.deepcopy(config), initialize)
        # Preserve parameter identities, RNG, existing optimized module layouts,
        # and the complete V6 checkpoint schema. No second decoder initialization.
        self.decoder.__class__ = V8Decoder
        self.decoder.configure(self.config.get("v8", {}))
        self.model_name = "RulerMVS-V8"

    def load_v6_state(self, state):
        return self.load_state_dict(state, strict=True)

    def encode(self, inputs, domain, reference=0):
        self.decoder.set_unit_scale(domain["depth_scale"], domain.get("nominal_depth_scale"))
        return super().encode(inputs, domain, reference)

    def forward(self, inputs, domain, reference=0):
        self.decoder.set_unit_scale(domain["depth_scale"], domain.get("nominal_depth_scale"))
        return super().forward(inputs, domain, reference)

    @torch.no_grad()
    def forward_references(self, inputs, domain, references=None):
        self.decoder.set_unit_scale(domain["depth_scale"], domain.get("nominal_depth_scale"))
        return super().forward_references(inputs, domain, references)


def build_from_checkpoint(saved, options=None):
    if saved.get("architecture") not in ("rulermvs_v6", "rulermvs_v8"):
        raise ValueError("V8 requires a complete V6 or V8 checkpoint")
    config = copy.deepcopy(saved["config"])
    config["architecture"] = V8Model.architecture
    if options is not None:
        config["v8"] = dict(options)
    model = V8Model(config, initialize=False)
    model.load_v6_state(saved["model"])
    return model, config
