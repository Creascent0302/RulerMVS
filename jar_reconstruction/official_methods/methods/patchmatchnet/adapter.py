"""作者指定论文版本的 PatchmatchNet；仅适配数据、标签和结果封装。"""

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import nn
import torch.nn.functional as F

UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "c6d2dce962be8109779afcdb3b79880d35ff9c83"


def native_module():
    # 上游 __init__ 使用全局 models 名称；隔离包名，避免与其他方法冲突。
    name = "_official_patchmatchnet_models"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(UPSTREAM / "models")]
        sys.modules[name] = package
    return import_module(f"{name}.net")


class Adapter(nn.Module):
    ddp_find_unused_parameters = False

    capabilities = {
        "depth": True,
        "confidence": "native_photometric_eval_only",
        "points": False,
        "occupancy": False,
        "semantics": False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        options = config.get("model", {})
        self.depth_min, self.depth_max = options.get("depth_range", [2.0, 80.0])
        if not 0 < self.depth_min < self.depth_max:
            raise ValueError("深度范围须为预先确定的正数；不能从测试标签推算。")
        self.native = native_module().PatchmatchNet()
        self.implementation_limitations = [
            "原生随机初始候选保留；重复比较必须记录并恢复随机状态。",
            "仅接收完整实际N视图，不用零图代替缺失相机。",
            "CARLA迁移不等于复现DTU论文指标；损失保留原生SmoothL1。",
        ]
        weights = options.get("weights", "")
        if weights and not config.get("_loading_shared_checkpoint", False):
            self.load_native_weights(weights)

    def load_native_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.native.load_state_dict(state, strict=True)

    def prepare_native(self, inputs, reference):
        rgb = inputs["rgb"].float()
        batch, views, _, height, width = rgb.shape
        if views < 2 or height % 8 or width % 8:
            raise ValueError("PatchmatchNet需要至少两个真实视图，图像尺寸须为8的倍数。")
        for key in ("view_mask", "image_mask"):
            if key in inputs and not bool(inputs[key].all()):
                raise ValueError(f"PatchmatchNet原模型不支持{key}中的缺失项；请使用共同完整子集。")
        if reference < 0 or reference >= views:
            raise ValueError("参考相机索引超出真实视图数量。")
        order = [reference] + [index for index in range(views) if index != reference]
        rgb = rgb[:, order]
        images, projections = {}, {}
        for stage in range(4):
            stride = 2 ** stage
            size = (height // stride, width // stride)
            images[f"stage_{stage}"] = (
                rgb if stage == 0 else F.interpolate(
                    rgb.flatten(0, 1), size, mode="bilinear", align_corners=False
                ).reshape(batch, views, 3, *size)
            )
            intrinsic = inputs["K"][:, order].float().clone()
            # 与原始特征卷积格点一致：整数0为中心，K的前两行直接除stride。
            intrinsic[..., :2, :] /= stride
            projection = inputs["T_cam_from_rig"][:, order].float().clone()
            projection[..., :3, :4] = intrinsic @ projection[..., :3, :4]
            projections[f"stage_{stage}"] = projection
        return (
            images, projections,
            rgb.new_full((batch,), self.depth_min),
            rgb.new_full((batch,), self.depth_max),
        )

    def forward(self, inputs):
        references = self.config.get("model", {}).get("reference_indices", [0])
        if not self.training:
            references = self.config.get("evaluation", {}).get("reference_indices", references)
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            native = [self.native(*self.prepare_native(inputs, ref)) for ref in references]
        depth = torch.cat([item["refined_depth"]["stage_0"] for item in native], dim=1)
        output = {
            "native": native,
            "depth": depth,
            "depth_K": inputs["K"][:, references].float().clone(),
            "reference_indices": list(references),
            "valid": torch.isfinite(depth) & (depth > 0),
        }
        if not self.training:
            output["confidence"] = torch.stack(
                [item["photometric_confidence"] for item in native], dim=1
            )
        return output

    def prepare_targets(self, targets, reference):
        depth = targets["depth"][:, reference:reference + 1].float()
        valid = torch.isfinite(depth) & (depth > 0)
        if "depth_valid" in targets:
            valid &= targets["depth_valid"][:, reference:reference + 1].bool()
        valid &= (depth >= self.depth_min) & (depth <= self.depth_max)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        labels, masks = {}, {}
        for stage in range(4):
            # 官方cv2 INTER_NEAREST与此nearest（非nearest-exact）的取样位置一致。
            size = tuple(axis // (2 ** stage) for axis in depth.shape[-2:])
            labels[f"stage_{stage}"] = F.interpolate(depth, size, mode="nearest")
            masks[f"stage_{stage}"] = F.interpolate(valid.float(), size, mode="nearest") > 0.5
        return labels, masks

    def loss(self, predictions, inputs, targets):
        losses = []
        for ref, native in zip(predictions["reference_indices"], predictions["native"]):
            labels, masks = self.prepare_targets(targets, ref)
            if all(mask.any() for mask in masks.values()):
                loss = native_module().patchmatchnet_loss(
                    native["depth_patchmatch"], native["refined_depth"], labels, masks
                )
            else:
                # 无监督像素时避免原生mean(empty)产生NaN；正常标签调用原函数。
                loss = native["refined_depth"]["stage_0"].sum() * 0
                for stage in range(4):
                    key = f"stage_{stage}"
                    estimates = ([native["refined_depth"][key]] if stage == 0
                                 else native["depth_patchmatch"][key])
                    for estimate in estimates:
                        loss = loss + (F.smooth_l1_loss(estimate[masks[key]], labels[key][masks[key]])
                                       if masks[key].any() else estimate.sum() * 0)
            losses.append(loss)
        total = torch.stack(losses).mean()
        return {"depth_multistage": total, "total": total}

    def configure_optimizer(self):
        settings = self.config.get("training", {})
        optimizer = torch.optim.Adam(
            self.parameters(), lr=settings.get("lr", 0.001),
            betas=(0.9, 0.999), weight_decay=settings.get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings.get("lr_milestones", [10, 12, 14]), gamma=0.5
        )
        return optimizer, scheduler
