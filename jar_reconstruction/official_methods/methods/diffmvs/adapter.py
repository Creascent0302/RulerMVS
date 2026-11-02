"""DiffMVS/CasDiffMVS：原生逆深度初始化、置信度扩散和损失。"""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "cd10d5c282a9cabd45a2f64598cd2b990b408d35"


def native_modules():
    """使用私有包名，避免多个官方仓库的顶层 models 包互相污染。"""
    name = "_jar_official_diffmvs"
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(UPSTREAM / "models")]
        sys.modules[name] = package
    network = importlib.import_module(f"{name}.diffusion")
    losses = importlib.import_module(f"{name}.loss")
    return network.CasDiffMVS, losses.compute_inverse_loss


def official_options(variant="diffmvs"):
    """数值来自官方 scripts/train 与 scripts/test，不缩小网络做替身。"""
    common = dict(
        numdepth_initial=48,
        numdepth=384,
        timesteps=[1000, 1000, 1000],
        sampling_timesteps=[0, 1, 1],
        unet_dim=[0, 16, 8],
        conf_weight=0.05,
    )
    if variant == "diffmvs":
        common.update(
            stage_iters=[1, 4, 0],
            cost_dim_stage=[4, 4, 0],
            CostNum=[0, 6, 0],
            scale=[0, 0.5, 0],
            ddim_eta=[0, 1, 0],
            hidden_dim=[0, 32, 0],
            context_dim=[32, 32, 0],
            min_radius=0.25,
            max_radius=4,
        )
    elif variant == "casdiffmvs":
        common.update(
            stage_iters=[1, 3, 3],
            cost_dim_stage=[4, 4, 4],
            CostNum=[0, 4, 4],
            scale=[0, 0.5, 0.1],
            ddim_eta=[0, 1, 1],
            hidden_dim=[0, 32, 20],
            context_dim=[32, 32, 16],
            min_radius=0.125,
            max_radius=8,
        )
    else:
        raise ValueError(f"未知官方变体：{variant}")
    return common


class Adapter(nn.Module):
    capabilities = {
        "depth": True,
        "confidence": "native_multistage_confidence",
        "points": False,
        "occupancy": False,
        "semantics": False,
    }
    requires_training_targets = True
    # 两个原生变体的多阶段损失均覆盖全部可训练参数。
    ddp_find_unused_parameters = False

    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        self.options = self.config.get("model", {})
        self.variant = self.options.get("variant", "diffmvs")
        arguments = official_options(self.variant)
        if self.options.get("pretraining_domain", "dtu") == "blendedmvs" and self.variant == "casdiffmvs":
            arguments["scale"] = [0, 0.125, 0.025]
        self.args = SimpleNamespace(**arguments)
        self.depth_min, self.depth_max = self.options.get("depth_range", [2.0, 80.0])
        if not 0 < self.depth_min < self.depth_max:
            raise ValueError("固定深度范围必须满足 0 < near < far，不能从测试标签推定。")
        Native, self.native_loss = native_modules()
        self.native = Native(self.args, test=False)
        self.implementation_limitations = [
            "保留官方DiffMVS/CasDiffMVS架构；四目、尺寸、深度范围属于共同数据协议适配。",
            "推理保留原生随机噪声；重测/原生一致性比较必须复位同一随机数状态。",
            "未复现原论文DTU/BlendedMVS数据预算、数据增强和后处理阈值。",
            "缺失整幅源视图可移除，但同一batch的有效相机集合须一致；像素孔洞不受原生网络支持。",
        ]
        weights = self.options.get("weights", "")
        if weights and not self.config.get("_loading_shared_checkpoint", False):
            self.load_native_weights(weights)
        elif not self.config.get("_loading_shared_checkpoint", False):
            if not self.options.get("allow_random_init", False):
                raise ValueError("请设置官方weights；从头训练或合成测试需显式allow_random_init=true。")
            self.implementation_limitations.append("当前无官方预训练权重，不能作为论文精度复现结果。")

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "native"):
            self.native.test = not mode
        return self

    def load_native_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        self.native.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}, strict=True
        )

    def prepare_native(self, inputs, reference):
        rgb, intrinsic, extrinsic = inputs["rgb"], inputs["K"], inputs["T_cam_from_rig"]
        batch, views, _, height, width = rgb.shape
        if height % 32 or width % 32:
            raise ValueError("官方代价体要求高宽是32的倍数；由数据加载器统一调整并同步K。")
        available = inputs.get("view_mask", torch.ones(batch, views, dtype=torch.bool, device=rgb.device))
        if not torch.equal(available, available[:1].expand_as(available)):
            raise ValueError("有效相机集合不同的样本请分桶；batch=1可直接使用缺相机数据。")
        order = [reference] + [i for i in range(views) if i != reference and bool(available[0, i])]
        if not bool(available[:, reference].all()) or len(order) < 2:
            raise ValueError("参考相机必须存在，且至少有一个真实源相机；不复制图像。")
        if "image_mask" in inputs and not bool(inputs["image_mask"][:, order].all()):
            raise ValueError("原生DiffMVS不支持图内孔洞掩码，请统一裁剪有效矩形。")
        cameras = rgb.new_zeros(batch, len(order), 2, 4, 4, dtype=torch.float32)
        cameras[:, :, 0] = extrinsic[:, order].float()
        cameras[:, :, 1, :3, :3] = intrinsic[:, order].float()
        projections = {}
        for stage, scale in enumerate((0.125, 0.25, 0.5, 1.0), 1):
            value = cameras.clone()
            value[:, :, 1, :2] *= scale
            projections[f"stage{stage}"] = value
        depths = torch.linspace(
            1 / self.depth_max, 1 / self.depth_min, self.args.numdepth,
            device=rgb.device, dtype=torch.float32,
        ).expand(batch, -1)
        return {
            "imgs": [rgb[:, i].float() for i in order],
            "proj_matrices": projections,
            "depth_values": depths,
        }

    def prepare_targets(self, inputs, targets, reference):
        depth = targets["depth"][:, reference : reference + 1].float()
        valid = torch.isfinite(depth) & (depth >= self.depth_min) & (depth <= self.depth_max)
        if "depth_valid" in targets:
            valid &= targets["depth_valid"][:, reference : reference + 1].bool()
        if "image_mask" in inputs:
            valid &= inputs["image_mask"][:, reference : reference + 1].bool()
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        depth_ms, mask_ms = {}, {}
        for stage, stride in enumerate((8, 4, 2, 1), 1):
            size = (depth.shape[-2] // stride, depth.shape[-1] // stride)
            # 官方cv2.INTER_NEAREST使用左上取样，与F.interpolate(nearest)一致。
            depth_ms[f"stage{stage}"] = F.interpolate(depth, size, mode="nearest")[:, 0]
            mask_ms[f"stage{stage}"] = F.interpolate(valid.float(), size, mode="nearest")[:, 0]
        return depth_ms, mask_ms

    def native_forward(self, batch, depth_gt_ms=None):
        return self.native(**batch, depth_gt_ms=depth_gt_ms)

    def forward(self, inputs, targets=None):
        if self.training and (targets is None or "depth" not in targets):
            raise ValueError("扩散训练需要独立targets中的深度，以构造原生训练噪声；推理不需要。")
        if not self.training and targets is not None:
            raise ValueError("DiffMVS推理禁止传入targets。")
        refs = self.options.get("reference_indices", [0])
        if not self.training:
            refs = self.config.get("evaluation", {}).get("reference_indices", refs)
        results = []
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            for ref in refs:
                labels = self.prepare_targets(inputs, targets, ref)[0] if self.training else None
                results.append(self.native_forward(self.prepare_native(inputs, ref), labels))
        depth = torch.stack([item["depth"][-1] for item in results], 1)
        output = {
            "native": results,
            "depth": depth,
            "depth_K": inputs["K"][:, refs].float(),
            "reference_indices": list(refs),
            "valid": torch.isfinite(depth) & (depth > 0),
        }
        if not self.training:
            output["confidence"] = torch.stack([item["photometric_confidence"][-1] for item in results], 1)
        return output

    def loss(self, predictions, inputs, targets):
        if not self.training:
            raise ValueError("原生扩散损失只用于训练；评测使用无标签推理深度和统一指标。")
        losses = {}
        for ref, native in zip(predictions["reference_indices"], predictions["native"]):
            depth, masks = self.prepare_targets(inputs, targets, ref)
            if not all(mask.any() for mask in masks.values()):
                raise ValueError("该样本某层没有有效深度监督，不能计算原生均值损失。")
            value, details = self.native_loss(
                self.args, native["depth"], native["conf"], depth, masks,
                self.prepare_native(inputs, ref)["depth_values"],
                loss_rate=0.9, iters=self.args.stage_iters,
            )
            for key, term in {"total": value, **details}.items():
                losses[key] = losses.get(key, 0.0) + term / len(predictions["native"])
        return losses

    def configure_optimizer(self):
        settings = self.config.get("training", {})
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=settings.get("lr", 0.001),
            weight_decay=settings.get("weight_decay", 0.001), eps=1e-8,
        )
        return optimizer, None

    def configure_scheduler(self, optimizer, planned_steps):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.config.get("training", {}).get("lr", 0.001),
            total_steps=planned_steps + 100, pct_start=0.05,
            cycle_momentum=False, anneal_strategy="linear",
        )
