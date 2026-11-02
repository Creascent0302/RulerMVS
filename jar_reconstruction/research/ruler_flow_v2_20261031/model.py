"""独立RulerFlow：原V2候选证据和读出保留，新增缓存证据平均流。"""

import torch

from jar.pretraining.fast_spatial import install_spatial_layout
from jar.pretraining.model import CalibratedDecoder
from research.v2_training_20261028.training import V2TrainingModel

from .field import EvidenceVelocity


class EvidenceFlowDecoder(CalibratedDecoder):
    def __init__(self, settings, flow_options=None):
        super().__init__(settings)
        self.flow_options = {"hidden": 48, "inference_steps": 1, **(flow_options or {})}
        if int(self.flow_options["inference_steps"]) < 1:
            raise ValueError("inference_steps必须为正整数")
        self.flow_field = EvidenceVelocity(settings["channels"], int(self.flow_options["hidden"]))

    def load_v2_state(self, state):
        expected = {key for key in self.state_dict() if not key.startswith("flow_field.")}
        if set(state) != expected:
            raise ValueError(f"旧解码器键不匹配，缺失{sorted(expected - set(state))}，多余{sorted(set(state) - expected)}")
        return self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def _tokens_from_source(self, prepared, observation, candidates, stride, source):
        if stride == 4:
            observation["flow_null_mass"] = source["null_mass"][:, 0]
        return super()._tokens_from_source(prepared, observation, candidates, stride, source)

    def _prediction(self, features, candidates, valid, similarity, support,
                    lower, upper, observation, use_offset=True):
        result = super()._prediction(features, candidates, valid, similarity, support,
                                     lower, upper, observation, use_offset)
        if observation["stride"] != 4:
            return result
        result["flow_similarity"] = similarity
        context = self.flow_field.compute_condition(
            features, candidates, lower, upper, result, observation, self.near, self.far)
        del result["flow_similarity"]
        state = context["anchor"]
        steps = int(self.flow_options["inference_steps"])
        for index in range(steps):
            t = 1 - index / steps
            r = 1 - (index + 1) / steps
            velocity = self.flow_field.velocity(context, state, r, t)
            if steps == 1:
                context["initial_velocity"] = velocity
            state = (state - (t - r) * velocity).clamp(-1, 1)
        # 差量写回避免0更新时先归一化再反归一化引入浮点变化。
        correction = .5 * context["width"] * (state - context["anchor"])
        continuous = result["candidate_depth"] + torch.where(context["valid"], correction, 0)
        result["candidate_depth"] = continuous
        result["depth"] = torch.where(
            valid.any(1, keepdim=True), continuous.gather(1, result["selected_index"]), 0)
        result["flow_state"] = state
        result["flow_anchor"] = context["anchor"]
        if torch.is_grad_enabled():
            result["_flow_context"] = context
        return result

    def forward(self, inputs):
        output = super().forward(inputs)
        if "_flow_context" in output["fine"]:
            output["_flow_context"] = output["fine"].pop("_flow_context")
        output["flow_evidence_calls"] = 1
        output["flow_observation_semantics"] = "仅原12细候选精确取证；连续端点是条件预测，不是新增观测"
        return output


class EvidenceFlowModel(V2TrainingModel):
    architecture = "ruler_flow_v2"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        old = self.decoder
        # 原模块照常初始化。新增结构不改变随后数据采样的全局随机流。
        with torch.random.fork_rng(devices=[]):
            self.decoder = EvidenceFlowDecoder(config["model"], config.get("flow"))
        self.decoder.load_v2_state(old.state_dict())
        if config["model"].get("spatial_layout", False):
            install_spatial_layout(self)
        self.model_name = "RulerFlow-V2"

    def load_v2_state(self, state):
        """只接受完整原V2；不容忍任意缺失或多余权重。"""
        expected = {key for key in self.state_dict() if not key.startswith("decoder.flow_field.")}
        if set(state) != expected:
            raise ValueError(f"旧V2键不匹配，缺失{sorted(expected - set(state))}，多余{sorted(set(state) - expected)}")
        return self.load_state_dict({**self.state_dict(), **state}, strict=True)

    @torch.no_grad()
    def forward_references(self, inputs, domain, references=None):
        """同一帧多参考推理：匹配编码一次，每个参考的ViT与解码独立执行。

        只在eval使用，不建立跨帧缓存，也不改变单参考forward。
        缺失参考仍返回其原生无效输出，不用其他相机偷偷替换。
        """
        if self.training:
            raise ValueError("forward_references只用于eval推理；训练沿用原参考采样协议")
        batch, views, _, height, width = inputs["rgb"].shape
        references = list(range(views)) if references is None else list(references)
        if len(set(references)) != len(references) or any(not 0 <= ref < views for ref in references):
            raise ValueError("references必须是无重复且在输入相机范围内的索引")
        scale = float(domain["depth_scale"])
        self.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
        transformed = inputs["T_cam_from_rig"].clone().float()
        transformed[..., :3, 3] = transformed[..., :3, 3] / scale
        match_height, match_width = inputs["matching_rgb"].shape[-2:]
        matching_scale = inputs["K"].new_tensor((match_width / width, match_height / height))
        shared = {
            "pyramid": self.matching(inputs),
            "K": inputs["K"],
            "T_cam_from_rig": transformed,
            "view_mask": inputs["view_mask"],
            "image_mask": inputs["image_mask"],
            "image_size": (height, width),
            "matching_scale": matching_scale[None].expand(batch, -1),
        }
        outputs = {}
        for reference in references:
            valid = inputs["image_mask"][:, reference] & inputs["view_mask"][:, reference, None, None]
            image = torch.where(valid[:, None], inputs["rgb"][:, reference], 0.)
            work = {**shared, "raw_maps": self.foundation(image), "reference_index": reference}
            prediction = self.decoder(work)
            prediction.update(pyramid=None, unit_scale=scale)
            outputs[reference] = prediction
        return outputs
