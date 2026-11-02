"""直接标尺解码的RGB入口：仅保留两个冻结预训练编码器。

不构建原CNN、学习式初值、稠密候选场、K9或迭代细化；不构建Hero。
原始标签与模型前向分离。可选多参考共享编码；未输出相机保持unknown。
"""

import copy

import torch
from torch import nn

from jar.multifocal.data import mask_camera_inputs
from jar.multifocal.view_order import canonicalize_wide_inputs, restore_wide_outputs
from jar.pretrained_matching import FrozenMVSMatching
from .deep_encoder import FrozenFoundation


class RulerDirectModel(nn.Module):
    architecture = "ruler_direct"

    def __init__(self, config, original_state=None, model_state=None):
        super().__init__()
        from .ruler_direct import RulerDirectDecoder

        self.cfg = copy.deepcopy(config)
        self.settings = self.cfg["direct"]
        self.reference_index = int(self.settings.get("reference_index", 2))
        if model_state is not None:
            foundation = {name[len("foundation."):]: value for name, value in model_state.items() if name.startswith("foundation.")}
            matching = {name[len("matching."):]: value for name, value in model_state.items() if name.startswith("matching.")}
        elif original_state is not None:
            foundation = {name[len("encoder.foundation."):]: value for name, value in original_state.items() if name.startswith("encoder.foundation.")}
            matching = {name[len("pretrained_matching."):]: value for name, value in original_state.items() if name.startswith("pretrained_matching.")}
        else:
            raise ValueError("需要完整原编码器权重；禁止随机前端或隐式联网下载")
        self.foundation = FrozenFoundation({}, state=foundation)
        self.matching = FrozenMVSMatching(state=matching)
        self.decoder = RulerDirectDecoder(self.settings)
        self.foundation.requires_grad_(False)
        self.matching.requires_grad_(False)
        if model_state is not None:
            self.load_state_dict(model_state, strict=True)

    def selected_references(self, views):
        """索引对应配置中的稳定相机顺序，与输入张量的临时排列无关。"""
        references = self.settings.get("reference_indices", [self.reference_index])
        if (
            not isinstance(references, (list, tuple))
            or not references
            or any(type(index) is not int or not 0 <= index < views for index in references)
            or len(set(references)) != len(references)
        ):
            raise ValueError("direct.reference_indices须为非空、不重复的有效相机索引列表")
        return tuple(references)

    def encode_references(self, inputs, references):
        """匹配编码只算一次，ViT只编码选中的有效参考图；解码器参数共享。"""
        inputs = mask_camera_inputs(inputs, self.cfg.get("camera", {}))
        batch, views, _, height, width = inputs["rgb"].shape
        image_mask = inputs["image_mask"] & inputs["view_mask"][:, :, None, None]
        work = {
            **inputs,
            "rgb": torch.where(image_mask[:, :, None], inputs["rgb"], 0),
            "image_mask": image_mask,
            "_reference_indices": references,
        }
        raw = self.foundation(work)
        raw = [value.reshape(batch, views, *value.shape[1:]) for value in raw]
        matching = self.matching(work)
        # 与研究特征缓存采用相同量化合同。完整时延包含量化开销。
        if self.settings.get("match_cache_quantization", True):
            matching = matching.half().float()
        matching_rgb = inputs.get("matching_rgb", inputs["rgb"])
        scale = matching.new_tensor([matching_rgb.shape[-1] / width, matching_rgb.shape[-2] / height])
        shared = {
            "matching_features": matching,
            "K": inputs["K"], "T_cam_from_rig": inputs["T_cam_from_rig"],
            "view_mask": inputs["view_mask"], "image_mask": image_mask,
            "matching_scale": scale[None].expand(batch, -1),
            "image_size": [height, width],
        }
        return {
            reference: {
                **shared,
                "raw_maps": [value[:, reference] for value in raw],
                "reference_index": reference,
            }
            for reference in references
        }

    def encode(self, inputs):
        """保留历史单参考特征缓存接口；多参考前向使用encode_references。"""
        if not 0 <= self.reference_index < inputs["rgb"].shape[1]:
            raise ValueError("reference_index超出实际相机数")
        # 缓存工具会将此无状态入口绑定到仅含两个编码器的轻量对象。
        return RulerDirectModel.encode_references(self, inputs, (self.reference_index,))[self.reference_index]

    def forward(self, inputs):
        ordered, restore = canonicalize_wide_inputs(inputs, self.cfg, force=True)
        batch, views = ordered["rgb"].shape[:2]
        references = self.selected_references(views)
        work = self.encode_references(ordered, references)
        results = {reference: self.decoder(work[reference]) for reference in references}
        primary = self.reference_index if self.reference_index in results else references[0]
        result = results[primary]
        height, width = result["depth"].shape[-2:]
        depth = result["depth"].new_zeros(batch, views, height, width)
        valid = torch.zeros_like(depth, dtype=torch.bool)
        confidence = torch.zeros_like(depth)
        support = torch.zeros_like(depth, dtype=torch.long)
        for reference, prediction in results.items():
            valid[:, reference] = prediction["valid"][:, 0]
            depth[:, reference] = torch.where(valid[:, reference], prediction["depth"][:, 0], 0)
            confidence[:, reference] = torch.where(valid[:, reference], prediction["confidence"][:, 0], 0)
            # 可投影且可取样的源数，不代表已通过遮挡或表面可见性验证。
            support[:, reference] = torch.where(valid[:, reference], prediction["support"][:, 0], 0).long()
        observed = torch.zeros_like(valid)
        # 原始targets不会在前向重排；为训练显式给出每个batch的原输入槽位。
        slots = torch.arange(views, device=depth.device)[None].expand(batch, -1)
        if restore is not None:
            slots = restore.argsort(1)
        output = {
            "direct": result, "dense_depth": depth, "dense_valid": valid,
            "direct_by_reference": results,
            "direct_camera_indices": slots[:, list(references)],
            "dense_confidence": confidence, "dense_observed": observed,
            "dense_support": support,
            "dense_state": torch.full_like(depth, 2, dtype=torch.long),
            "reference_indices": references, "depth_stride": 1,
        }
        output["dense"] = {
            name: output["dense_" + name]
            for name in ("depth", "valid", "confidence", "observed", "support", "state")
        }
        return restore_wide_outputs(output, restore)

    def training_loss(self, prediction, inputs, targets):
        from .ruler_direct_losses import multireference_loss
        import torch.distributed as dist

        losses = multireference_loss(prediction, inputs, targets, self.cfg.get("loss", {}))
        if dist.is_available() and dist.is_initialized():
            # DDP平均rank梯度；按有监督相机数补偿，不让缺标签rank稀释学习。
            local_count = losses["supervised_views"].detach().float()
            total_count = local_count.clone()
            dist.all_reduce(total_count)
            scale = dist.get_world_size() * local_count / total_count.clamp_min(1)
            losses = {key: value * scale if value.is_floating_point() else value
                      for key, value in losses.items()}
        return losses

    def load_checkpoint_weights(self, record):
        """只允许完整同构权重迁移；参考调度可改，学习结构不可静默替换。"""
        saved = record.get("config", {}).get("direct")
        if record.get("architecture") != self.architecture or saved is None:
            raise ValueError("ruler_direct初始化需要包含两个编码器的完整同架构checkpoint")
        scheduling = {"reference_index", "reference_indices"}
        before = {key: value for key, value in saved.items() if key not in scheduling}
        after = {key: value for key, value in self.settings.items() if key not in scheduling}
        if before != after:
            raise ValueError("ruler_direct的解码架构与checkpoint不一致，禁止静默部分加载")
        self.load_state_dict(record["model"], strict=True)
        return {
            "missing_keys": [], "unexpected_keys": [], "base_transfer": False,
            "reference_indices_before": saved.get("reference_indices", [saved.get("reference_index", 2)]),
            "reference_indices_after": self.settings.get("reference_indices", [self.reference_index]),
        }

    def checkpoint_record(self):
        return {"architecture": "ruler_direct", "config": self.cfg, "model": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, path):
        record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if record.get("architecture") != "ruler_direct":
            raise ValueError("需要ruler_direct完整checkpoint")
        return cls(record["config"], model_state=record["model"])
