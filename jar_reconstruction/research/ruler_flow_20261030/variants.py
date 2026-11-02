"""独立的四峰局部条件流实验；保留生产代码和原始checkpoint。"""

import math

import torch

from jar.geometry import safe_masked_softmax
from jar.models.metric_ruler_network import depth_cell_edges
from jar.pretraining.model import CalibratedDecoder
from research.role_fusion_20261030.variants import transport_with_prior

from .direction import CHANNELS, directional_evidence, prepare_direction
from .flow import LocalVelocity, bridge_targets, clamp_state, local_coordinates, velocity_loss


class FlowDecoder(CalibratedDecoder):
    def _coarse_context(self, inputs):
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
        samples, half_width, inherited, prior = transport_with_prior(
            self, coarse, features, observation, prepared)
        anchor = samples[:, 1::3]
        # 原三点各有半步offset；联合区间保持，近远裁剪后不擅自扩窗。
        low = (samples[:, 0::3] - half_width[:, 0::3]).clamp(self.near, self.far)
        high = (samples[:, 2::3] + half_width[:, 2::3]).clamp(self.near, self.far)
        anchor_log, radius, low_state, high_state = local_coordinates(anchor, low, high)
        context = {
            "prepared": prepared, "coarse": coarse, "anchor": anchor,
            "anchor_log": anchor_log, "radius": radius, "lower": low, "upper": high,
            "low_state": low_state, "high_state": high_state,
            "half_width": half_width[:, 1::3], "inherited": inherited[:, :, 1::3],
            "prior": prior / prior.sum(1, keepdim=True).clamp_min(1e-8),
            "direction_cache": prepare_direction(prepared) if self.flow_options["direction"] else None,
        }
        return context

    def _read(self, context, state, time):
        prepared = context["prepared"]
        depth = (context["anchor_log"] + context["radius"] * state).exp().clamp(self.near, self.far)
        observation = self._observe(prepared, depth, 4, context["half_width"])
        features, _, valid, similarity, support, source = self._tokens(prepared, observation, depth, 4)
        features = self.entry_norm(features + self.fine_inherit(context["inherited"])) * valid[:, None]
        if self.flow_options["direction"]:
            # 方向证据作为本步测量值；不对线性化项反传，避免学习操纵GN证据。
            with torch.no_grad():
                direction = directional_evidence(
                    prepared, context["direction_cache"], depth.detach(), context["radius"].detach(),
                    source_weights=source["weights"].detach())
        else:
            direction = features.new_zeros((features.shape[0], len(CHANNELS), *depth.shape[1:]))
        position = (context["anchor_log"] - math.log(self.near)) / math.log(self.far / self.near)
        velocity, quality = self.flow_head(
            features, direction, state, time, position, context["radius"], context["prior"],
            similarity, source["null_mass"][:, 0], observation["source_valid"].float().mean(1), valid)
        return {"depth": depth, "features": features, "valid": valid,
                "similarity": similarity, "support": support, "observation": observation,
                "velocity": velocity, "quality": quality, "direction": direction}

    def _finish(self, context, state):
        # 更新后重新取证；实际证据调用次数是K+1，不能省略这一项声称加速。
        read = self._read(context, state, 1.)
        result = self._prediction(
            read["features"], read["depth"], read["valid"], read["similarity"], read["support"],
            context["lower"], context["upper"], read["observation"], use_offset=False)
        scores = (result["scores"] + read["quality"]).masked_fill(~read["valid"], -1e4)
        probability = safe_masked_softmax(scores.float(), read["valid"], dim=1)
        winner = scores.argmax(1, keepdim=True)
        result.update(scores=scores, probability=probability, selected_index=winner,
                      depth=torch.where(read["valid"].any(1, keepdim=True),
                                        read["depth"].gather(1, winner), 0),
                      confidence=probability.gather(1, winner), anchor_depth=context["anchor"])
        prepared = context["prepared"]
        reference = prepared["reference"]
        output = self._render(result, read["features"], prepared["image_mask"][:, reference:reference + 1], prepared)
        output.update(coarse=context["coarse"], fine=result, scene=None,
                      reference_index=reference, continuous_offset_used=True,
                      support_semantics="几何可采样源数，非可见性认证",
                      flow_state=state, flow_evidence_calls=int(self.flow_options["inference_steps"]) + 1)
        if torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.flow_head.parameters()):
            output["_flow_context"] = context
        return output

    def forward(self, inputs):
        context = self._coarse_context(inputs)
        state = torch.zeros_like(context["anchor"])
        sigma = float(self.flow_options.get("inference_noise_sigma", 0.))
        if sigma:
            generator = torch.Generator(device=state.device).manual_seed(
                int(self.flow_options.get("inference_seed", 0)))
            state = sigma * torch.randn(state.shape, device=state.device, generator=generator)
        state = clamp_state(state, context["low_state"], context["high_state"])
        steps = int(self.flow_options["inference_steps"])
        for step in range(steps):
            read = self._read(context, state, step / steps)
            state = clamp_state(state + read["velocity"] / steps,
                                context["low_state"], context["high_state"])
        return self._finish(context, state)


def install(model, options):
    if not options:
        return model
    options = {"objective": "flow", "inference_steps": 1, "direction": True,
               "hidden": 32, "sigma": .05, "learn_quality": True,
               "anchor_fraction": 0., **options}
    if options["objective"] not in {"flow", "direct"} or int(options["inference_steps"]) < 0:
        raise ValueError("objective须为direct/flow，inference_steps须非负")
    decoder = model.decoder
    if type(decoder) is not CalibratedDecoder or decoder.use_scene or decoder.proposal_transport == "nearest":
        raise ValueError("此独立实验仅从原V2四邻域候选传递架构安装，不叠加其他研究变体")
    decoder.__class__ = FlowDecoder
    decoder.flow_options = options
    parameter = next(decoder.parameters())
    with torch.random.fork_rng(devices=[]):
        decoder.flow_head = LocalVelocity(decoder.settings["channels"], len(CHANNELS),
                                         options["hidden"], options["learn_quality"])
    decoder.flow_head.to(device=parameter.device, dtype=parameter.dtype)
    return model


def training_loss(model, prediction, inputs, targets, domain, options=None):
    """GT只用于训练路径；不回写forward状态，不改变最终深度或推理候选。"""
    options = {} if options is None else options
    decoder = model.decoder
    context = prediction["_flow_context"]
    reference = prediction["reference_index"]
    truth = targets["depth"][:, reference:reference + 1, ::4, ::4].float() / prediction["unit_scale"]
    valid = targets["depth_valid"][:, reference:reference + 1, ::4, ::4].clone()
    valid &= inputs["image_mask"][:, reference:reference + 1, ::4, ::4]
    valid &= inputs["view_mask"][:, reference:reference + 1, None, None]
    valid &= torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        valid &= (truth * prediction["unit_scale"] >= domain["eval_range"][0])
        valid &= (truth * prediction["unit_scale"] <= domain["eval_range"][1])
    bridge = bridge_targets(context["anchor_log"].detach(), context["radius"].detach(),
                            context["low_state"].detach(), context["high_state"].detach(),
                            truth, valid, decoder.flow_options["objective"], decoder.flow_options["sigma"],
                            options.get("_generator"), decoder.flow_options["anchor_fraction"])
    read = decoder._read(context, bridge["state"], bridge["time"])
    assigned = bridge["assigned"] & read["valid"]
    loss = velocity_loss(read["velocity"], bridge["velocity"], assigned)
    return {"total": loss, "flow_mse": loss, "supervised_pixels": valid.sum(),
            "covered_branches": bridge["covered"].sum(), "assigned_branches": assigned.sum(),
            "covered_pixels": bridge["covered"].any(1).sum(),
            "target_velocity_abs": torch.where(assigned, bridge["velocity"].abs(), 0).sum() /
                                   assigned.sum().clamp_min(1)}
