"""保留官方冻结 DINOv2-B、跨视图旁路、粗代价 Transformer 与四级损失。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "bdd980cfbae9a863b5489e2ed973612b192d11a5"


def native_modules():
    """上游使用顶层 models 包；不同基线应按公共脚本在独立进程运行。"""
    existing = sys.modules.get("models")
    if existing is not None and not any(
        Path(path).resolve() == UPSTREAM / "models"
        for path in getattr(existing, "__path__", [])
    ):
        raise RuntimeError("其他基线已加载顶层 models 包，请在独立进程运行 MVSFormer++。")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet
    from models.losses import get_multi_stage_losses

    spec = importlib.util.spec_from_file_location("mvsformerpp_native_utils", UPSTREAM / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return DINOv2MVSNet, get_multi_stage_losses, utils


class Adapter(nn.Module):
    capabilities = {
        "depth": True,
        "confidence": "native_multistage_probability",
        "points": False,
        "occupancy": False,
        "semantics": False,
    }
    # 四级损失覆盖所有可训练参数；原生重入 checkpoint 要求禁用 unused 搜索。
    ddp_find_unused_parameters = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings = config.get("model", {})
        self.implementation_limitations = [
            "CARLA共同协议使用四目、固定米制范围与固定图像尺寸；原论文是五目及多尺度数据采样。",
            "公共训练器保留原生分级CE、AdamW分组和余弦预热；未复现原DTU/BlendedMVS训练数据增强及数据预算。",
            "原生3D位置编码的横纵极值跨batch计算；不同标定批处理可能影响结果，公平评估使用batch1。",
        ]
        if self.settings.get("allow_random_init") and not self.settings.get("weights"):
            self.implementation_limitations.append("当前是完整随机初始化架构的管线验证，不能作为论文预训练精度结果。")
        official = json.loads((UPSTREAM / "config" / "mvsformer++.json").read_text())
        self.native_args = deepcopy(official["arch"]["args"])
        # 完整发布架构不缩小层数或通道；分辨率与米制范围由数据协议确定。
        self.native_args["vit_path"] = ""
        self.native_args["freeze_vit"] = True
        self.loss_args = official["arch"]["loss"]
        self.optimizer_args = dict(official["optimizer"]["args"])
        for key in ("lr", "vit_lr", "weight_decay", "warmup_steps", "min_lr"):
            if key in config.get("training", {}):
                self.optimizer_args[key] = config["training"][key]
        self.depth_min, self.depth_max = self.settings.get("depth_range", [2.0, 80.0])
        if not 0 < self.depth_min < self.depth_max:
            raise ValueError("MVSFormer++ 需要固定米制范围 0 < near < far。")
        self.temperature = self.settings.get("temperature", [5.0, 5.0, 5.0, 1.0])
        if len(self.temperature) != 4 or min(self.temperature) <= 0:
            raise ValueError("temperature 需要四个正数，对应四级原生深度回归。")
        NativeModel, self.native_loss, self.native_utils = native_modules()
        self.native = NativeModel(self.native_args)
        self.native.vit.requires_grad_(False)
        resume = config.get("_loading_shared_checkpoint", False)
        if self.settings.get("weights") and not resume:
            self.load_native_weights(self.settings["weights"])
        elif self.settings.get("dinov2_weights") and not resume:
            state = torch.load(self.settings["dinov2_weights"], map_location="cpu", weights_only=False)
            state = state.get("state_dict", state.get("model", state))
            self.native.vit.load_state_dict(state, strict=True)
        elif not resume and not self.settings.get("allow_random_init", False):
            raise ValueError(
                "请配置完整官方 weights 或官方 dinov2_weights；仅合成结构测试可显式 allow_random_init=true。"
            )

    def load_native_weights(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model", state))
        # 官方 test.py/test_tt.py 同样排除旧版位置编码缓存；其余参数仍严格加载。
        state = {
            name.removeprefix("module."): value
            for name, value in state.items()
            if "pe_dict" not in name
        }
        self.native.load_state_dict(state, strict=True)

    def prepare_native(self, inputs, reference):
        rgb, intrinsic, transform = inputs["rgb"], inputs["K"], inputs["T_cam_from_rig"]
        batch, views, _, height, width = rgb.shape
        if views < 2 or height % 64 or width % 64:
            raise ValueError("官方 MVSFormer++ 需要至少两目，图像高宽为64的倍数。")
        if "view_mask" in inputs and not bool(inputs["view_mask"].all()):
            raise ValueError("MVSFormer++ 官方完整视图协议不支持缺失相机。")
        if "image_mask" in inputs and not bool(inputs["image_mask"].all()):
            raise ValueError("MVSFormer++ 官方网络不支持带孔图像，请先裁剪到有效矩形。")
        order = [reference] + [view for view in range(views) if view != reference]
        images = rgb[:, order].float()
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        images = (images - mean) / std
        projections = {}
        for stage, stride in enumerate((8, 4, 2, 1), start=1):
            camera = images.new_zeros((batch, views, 2, 4, 4))
            camera[:, :, 0] = transform[:, order].detach().float()
            camera[:, :, 1, :3, :3] = intrinsic[:, order].detach().float()
            # 官方卷积格点、反投影和 align_corners=True 均以整数像素为中心。
            camera[:, :, 1, :2, :3] /= stride
            projections[f"stage{stage}"] = camera
        depth_values = torch.linspace(
            self.depth_min, self.depth_max, 192, device=images.device, dtype=torch.float32
        ).expand(batch, -1)
        return images, projections, depth_values

    def call_native(self, prepared):
        return self.native(*prepared, tmp=self.temperature)

    def forward(self, inputs):
        references = self.settings.get("reference_indices", [2])
        if not self.training:
            references = self.config.get("evaluation", {}).get("reference_indices", references)
        refs = list(references)
        if not refs or len(set(refs)) != len(refs) or any(
            ref < 0 or ref >= inputs["rgb"].shape[1] for ref in refs
        ):
            raise ValueError("reference_indices 必须是不重复的有效相机编号。")
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            native = [self.call_native(self.prepare_native(inputs, ref)) for ref in refs]
        depth = torch.stack([result["refined_depth"] for result in native], dim=1)
        confidence = torch.stack([result["photometric_confidence"] for result in native], dim=1)
        return {
            "native": native,
            "depth": depth,
            "confidence": confidence,
            "depth_K": inputs["K"][:, refs].float().clone(),
            "reference_indices": refs,
            "valid": torch.isfinite(depth) & (depth > 0),
        }

    def loss(self, predictions, inputs, targets):
        if "depth" not in targets:
            raise ValueError("MVSFormer++ 原生训练需要光轴深度标签。")
        losses = {}
        depth = targets["depth"]
        valid = targets.get("depth_valid", torch.isfinite(depth) & (depth > 0)).bool()
        if depth.ndim == 5:
            depth, valid = depth.squeeze(2), valid.squeeze(2)
        valid = valid & torch.isfinite(depth) & (depth >= self.depth_min) & (depth <= self.depth_max)
        supervised = 0
        for ref, native in zip(predictions["reference_indices"], predictions["native"]):
            if not bool(valid[:, ref].any()):
                continue
            supervised += 1
            gt = torch.where(valid[:, ref], depth[:, ref], 0).float()[:, None]
            mask = valid[:, ref : ref + 1].float()
            depths, masks = {}, {}
            for stage in ("stage1", "stage2", "stage3", "stage4"):
                size = native[stage]["depth"].shape[-2:]
                # 官方 cv2.INTER_NEAREST 对应整数 stride 格点，不能换 nearest-exact。
                depths[stage] = F.interpolate(gt, size=size, mode="nearest")[:, 0]
                masks[stage] = F.interpolate(mask, size=size, mode="nearest")[:, 0]
            interval = gt.new_full((gt.shape[0],), (self.depth_max - self.depth_min) / 191)
            branch = self.native_loss(
                self.loss_args, self.native_args["depth_type"], native,
                depths, masks, interval, self.native_args["inverse_depth"],
            )
            for name, value in branch.items():
                losses[name] = losses.get(name, 0) + value
        if supervised:
            losses = {name: value / supervised for name, value in losses.items()}
        else:
            losses = {
                f"stage{stage}": sum(
                    result[f"stage{stage}"]["prob_volume_pre"].sum() * 0
                    for result in predictions["native"]
                ) for stage in range(1, 5)
            }
        losses["total"] = sum(losses.values())
        return losses

    def configure_optimizer(self):
        groups = self.native_utils.get_parameter_groups(self.optimizer_args, self.native, freeze_vit=True)
        return torch.optim.AdamW(
            groups, lr=self.optimizer_args["lr"], weight_decay=self.optimizer_args["weight_decay"]
        )

    def configure_scheduler(self, optimizer, total_steps):
        warmup = int(self.optimizer_args["warmup_steps"])
        if total_steps <= warmup:
            raise ValueError("训练总步数必须大于 warmup_steps；短冒烟测试请在配置中将其改为0。")
        return self.native_utils.get_lr_schedule_with_warmup(
            optimizer, warmup, total_steps, self.optimizer_args["min_lr"]
        )
