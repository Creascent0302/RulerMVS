"""Fast-MVSNet 官方三阶段网络的四目适配；不更换骨干或损失。"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "ccb686dda2717613c67d8a289dfe7b2aeb60e2fd"


def _native_classes():
    if not (UPSTREAM / "fastmvsnet" / "model.py").is_file():
        raise FileNotFoundError(f"Fast-MVSNet 官方源码缺失：{UPSTREAM}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from fastmvsnet.model import FastMVSNet, PointMVSNetLoss

    return FastMVSNet, PointMVSNetLoss


class Adapter(nn.Module):
    capabilities = {
        "depth": True,
        "confidence": "native_coarse_only",
        "points": False,
        "occupancy": False,
        "semantics": False,
    }
    # 官方 PropagationNet.conv1/conv2 永远未调用，GN 预热还会跳过 flow_img_conv。
    ddp_find_unused_parameters = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.options = config.get("model", {})
        self.epoch = 0
        FastMVSNet, PointMVSNetLoss = _native_classes()
        self.native = FastMVSNet()  # 官方默认通道 8/8，不以 tiny 网络替换。
        self.native_loss = PointMVSNetLoss(valid_threshold=8.0)
        self.depth_min, self.depth_max = self.options.get("depth_range", [2.0, 80.0])
        self.pretrain_epochs = int(config.get("training", {}).get("pretrain_epochs", 4))
        weights = self.options.get("weights", "")
        if weights:
            self.load_native_weights(weights)

    def load_native_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # 仓库自带权重使用旧模块名 upnet，发布源码更名为 propagation_net。
        state = {
            ("propagation_net." + k[len("upnet.") :] if k.startswith("upnet.") else k): v
            for k, v in state.items()
        }
        self.native.load_state_dict(state, strict=True)

    def set_epoch(self, epoch):
        """epoch 从零计数；0..3 仅稀疏估计+传播，4 起启用 GN。"""
        self.epoch = int(epoch)

    @property
    def use_gn(self):
        return not self.training or self.epoch >= self.pretrain_epochs

    def _depth_planes(self):
        key = "train_depth_planes" if self.training else "test_depth_planes"
        return int(self.options.get(key, 48 if self.training else 96))

    def prepare_native(self, inputs, reference):
        """只接受推理输入，返回官方 img_list/cam_params_list。"""
        rgb, K, T = inputs["rgb"], inputs["K"], inputs["T_cam_from_rig"]
        b, n, _, h, w = rgb.shape
        if n < 2 or h % 64 or w % 64:
            raise ValueError("Fast-MVSNet 需至少两视图，图像高宽为64的倍数（原始3D U-Net约束）。")
        if "view_mask" in inputs and not bool(inputs["view_mask"].all()):
            raise ValueError("官方 Fast-MVSNet 不支持缺失视图；请提供完整同步四目样本。")
        if "image_mask" in inputs and not bool(inputs["image_mask"].all()):
            raise ValueError("官方 Fast-MVSNet 不支持带孔图像掩码；请统一裁剪到有效矩形区域。")
        order = [reference] + [i for i in range(n) if i != reference]
        # cv2.imread 的官方通道顺序为 BGR；训练代码原样使用 0..255。
        images = rgb[:, order][:, :, [2, 1, 0]].float() * 255.0
        mode = self.options.get("normalization", "consistent_standardized")
        if mode == "consistent_standardized" or (mode == "official" and not self.training):
            mean = images.mean((-2, -1), keepdim=True)
            var = images.var((-2, -1), correction=0, keepdim=True)
            images = (images - mean) / (var.sqrt() + 1e-7)
        elif mode != "official":
            raise ValueError(f"未知 Fast-MVSNet normalization: {mode}")
        cameras = images.new_zeros((b, n, 2, 4, 4))
        cameras[:, :, 0] = T[:, order].float()
        intrinsic = K[:, order].float().clone()
        # 项目像素中心为整数，官方 get_pixel_grids 从0.5开始。
        intrinsic[..., :2, 2] += 0.5
        if self.training:
            intrinsic[..., :2, :] /= 4.0
        cameras[:, :, 1, :3, :3] = intrinsic
        planes = self._depth_planes()
        if planes < 8 or planes % 8 or not 0 < self.depth_min < self.depth_max:
            raise ValueError("深度平面数必须为8的正倍数，且0 < depth_min < depth_max。")
        cameras[:, :, 1, 3, 0] = self.depth_min
        cameras[:, :, 1, 3, 1] = (self.depth_max - self.depth_min) / (planes - 1)
        cameras[:, :, 1, 3, 2] = planes
        return {"img_list": images, "cam_params_list": cameras}

    def native_forward(self, batch):
        return self.native(
            batch,
            img_scales=(0.25, 0.5),
            inter_scales=(0.75, 0.375) if self.training else (0.75, 0.15),
            isGN=self.use_gn,
            isTest=not self.training,
        )

    def forward(self, inputs):
        # 几何逆矩阵及 GN 保持 float32；不静默将混合精度视为已验证模式。
        references = self.options.get("reference_indices", [0])
        if not self.training:
            references = self.config.get("evaluation", {}).get("reference_indices", references)
        refs = list(references)
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            results = [self.native_forward(self.prepare_native(inputs, ref)) for ref in refs]
        key = "flow2" if self.use_gn else "coarse_depth_map"
        depth = torch.cat([p[key] for p in results], dim=1)
        h, w = inputs["rgb"].shape[-2:]
        sy, sx = depth.shape[-2] / h, depth.shape[-1] / w
        depth_K = inputs["K"][:, refs].float().clone()
        depth_K[..., :2, 2] += 0.5
        depth_K[..., 0, :] *= sx
        depth_K[..., 1, :] *= sy
        depth_K[..., :2, 2] -= 0.5
        return {
            "native": results,
            "depth": depth,
            "depth_K": depth_K,
            "reference_indices": refs,
            "valid": torch.isfinite(depth) & (depth > 0),
            "coarse_confidence": torch.cat([p["coarse_prob_map"] for p in results], 1),
        }

    def loss(self, predictions, inputs, targets):
        if "depth" not in targets:
            raise ValueError("Fast-MVSNet 原生训练需要米制光轴深度标签。")
        losses = {}
        for ref, native in zip(predictions["reference_indices"], predictions["native"]):
            batch = self.prepare_native(inputs, ref)
            depth = targets["depth"][:, ref : ref + 1].float()
            valid = torch.isfinite(depth) & (depth > 0)
            if "depth_valid" in targets:
                valid &= targets["depth_valid"][:, ref : ref + 1].bool()
            interval = (self.depth_max - self.depth_min) / (self._depth_planes() - 1)
            # 官方训练加载器排除首尾深度平面附近的标签，不参与输入范围计算。
            valid &= (depth > self.depth_min + interval) & (depth <= self.depth_max - interval)
            labels = {
                "gt_depth_img": torch.where(valid, depth, torch.zeros_like(depth)),
                "cam_params_list": batch["cam_params_list"],
            }
            branch = self.native_loss(native, labels, self.use_gn)
            for name, value in branch.items():
                losses[name] = losses.get(name, 0) + value / len(predictions["native"])
        losses["total"] = sum(losses.values())
        return losses

    def configure_optimizer(self):
        cfg = self.config.get("training", {})
        decay, no_decay = [], []
        for name, value in self.native.named_parameters():
            (no_decay if ".bn." in name else decay).append(value)
        opt = torch.optim.RMSprop(
            [
                {"params": decay, "weight_decay": cfg.get("weight_decay", 0.001)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.get("lr", 0.0005),
            alpha=0.9,
        )
        return opt, torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.9)
