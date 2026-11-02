"""RulerMVS-V5：保留V2全部几何/解码，仅加入可拆卸CNN→ViT通路。"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research.v2_training_20261028.training import V2TrainingModel
from .bridge import CNNTokenBridge


class V5Model(V2TrainingModel):
    architecture = "rulermvs_v5"

    def __init__(self, config, initialize=True):
        super().__init__(config, initialize)
        with torch.random.fork_rng(devices=[]):
            self.bridge = CNNTokenBridge(config.get("bridge"))
        self.model_name = "RulerMVS-V5"

    def set_stage(self, number):
        super().set_stage(number)
        self.bridge.detach_backbone = bool(self.active_stage.get(
            "bridge_detach_backbone", self.bridge.options["detach_backbone"]))

    def load_v2_state(self, state):
        """严格接收V2完整参数，新增bridge保持自身恒等初始化。"""
        expected = {key for key in self.state_dict() if not key.startswith("bridge.")}
        if set(state) != expected:
            raise ValueError(f"V2键不匹配：缺失{sorted(expected - set(state))}，多余{sorted(set(state) - expected)}")
        return self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def v2_state_dict(self):
        """物理拆除桥后可交给原V2严格加载；训练后底座可能已更新。"""
        return {key: value for key, value in self.state_dict().items() if not key.startswith("bridge.")}

    def _matching_with_features(self, inputs):
        captured = {}
        handles = []
        batch, views = inputs["rgb"].shape[:2]

        def capture(name):
            def hook(module, arguments, output):
                captured[name] = output.unflatten(0, (batch, views))
            return hook

        for name in self.bridge.layers:
            handles.append(getattr(self.matching.backbone, name).register_forward_hook(capture(name)))
        try:
            pyramid = self.matching(inputs)
        finally:
            for handle in handles:
                handle.remove()
        return pyramid, captured

    def _foundation_with_bridge(self, image, features, matching_mask, image_mask):
        foundation = self.foundation
        value = (image - foundation.mean) / foundation.std
        value = F.interpolate(value, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        patch_shape = (value.shape[-2] // 14, value.shape[-1] // 14)

        def inject(module, arguments, output):
            return self.bridge(output, features, matching_mask, image_mask, patch_shape)

        handle = foundation.model.patch_embed.register_forward_hook(inject)
        try:
            tokens = foundation.model.prepare_tokens_with_masks(value)
        finally:
            handle.remove()
        outputs = []
        for index, block in enumerate(foundation.model.blocks):
            needs_grad = tokens.requires_grad or any(p.requires_grad for p in block.parameters())
            if torch.is_grad_enabled() and needs_grad:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in foundation.indices:
                normalized = foundation.model.norm(tokens)[:, 1:]
                outputs.append(normalized.transpose(1, 2).reshape(len(image), 768, *patch_shape))
        return outputs

    def _encode_reference(self, inputs, domain, reference, pyramid, features):
        batch, _, _, height, width = inputs["rgb"].shape
        scale = float(domain["depth_scale"])
        transformed = inputs["T_cam_from_rig"].clone().float()
        transformed[..., :3, 3] /= scale
        image_mask = inputs["image_mask"] & inputs["view_mask"][:, :, None, None]
        valid = image_mask[:, reference]
        image = torch.where(valid[:, None], inputs["rgb"][:, reference], 0.)
        matching_mask = inputs["matching_image_mask"][:, reference]
        matching_mask = matching_mask & inputs["view_mask"][:, reference, None, None]
        raw = self._foundation_with_bridge(
            image, {name: value[:, reference] for name, value in features.items()}, matching_mask, valid)
        match_h, match_w = inputs["matching_rgb"].shape[-2:]
        return {
            "pyramid": pyramid, "raw_maps": raw, "K": inputs["K"], "T_cam_from_rig": transformed,
            "view_mask": inputs["view_mask"], "image_mask": image_mask,
            "reference_index": reference, "image_size": (height, width),
            "matching_scale": inputs["K"].new_tensor((match_w / width, match_h / height))[None].expand(batch, -1),
        }

    def encode(self, inputs, domain, reference=0):
        if not self.bridge.enabled:
            masked = dict(inputs, image_mask=inputs["image_mask"] & inputs["view_mask"][:, :, None, None])
            return super().encode(masked, domain, reference)
        pyramid, features = self._matching_with_features(inputs)
        return self._encode_reference(inputs, domain, reference, pyramid, features)

    @torch.no_grad()
    def forward_references(self, inputs, domain, references=None):
        """eval时共享同帧匹配编码；不重复计算CNN，也不缓存跨帧特征。"""
        if self.training:
            raise ValueError("forward_references只用于eval；训练仍按原协议轮换参考")
        views = inputs["rgb"].shape[1]
        references = list(range(views)) if references is None else list(references)
        if len(set(references)) != len(references) or any(not 0 <= ref < views for ref in references):
            raise ValueError("references须为不重复的有效相机索引")
        if not self.bridge.enabled:
            return {ref: self(inputs, domain, ref) for ref in references}
        scale = float(domain["depth_scale"])
        self.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
        pyramid, features = self._matching_with_features(inputs)
        outputs = {}
        for reference in references:
            work = self._encode_reference(inputs, domain, reference, pyramid, features)
            prediction = self.decoder(work)
            prediction.update(pyramid=None, unit_scale=scale)
            outputs[reference] = prediction
        return outputs
