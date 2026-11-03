"""保留官方 Hero 架构，用通用预训练资产开始共同数据训练。

这里的 Hero 是架构名，不加载发布的 MVSAnywhere 任务 checkpoint。
原生四尺度损失、metadata MLP、两个 ViT 和 DPT 均保留。
"""

import hashlib
from pathlib import Path

import antialiased_cnns
import torch
from torch import nn

from official_methods.methods.mvsanywhere.adapter import Adapter


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MVSAnywhereTrainingModel(nn.Module):
    architecture = "mvsanywhere_retrained"

    def __init__(self, config, initialize=True):
        super().__init__()
        self.config = config
        height, width = config["image_size"]
        settings = config.get("mvsanywhere", {})
        initialization = config["initialization"]
        native_config = {
            "method": "mvsanywhere",
            "model": {
                "num_views": 4,
                "image_height": height,
                "image_width": width,
                "depth_bins": 64,
                "pretrained": False,
                "weights": "",
                "da_weights": initialization["foundation"] if initialize else "",
                "fast_cost_volume": False,
                "reference_indices": [0],
            },
            "training": {
                "lr": settings.get("matching_lr", 1e-4),
                "lr_da_encoder": settings.get("encoder_lr", 5e-6),
                "lr_da_decoder": settings.get("decoder_lr", 5e-5),
                "weight_decay": config.get("train", {}).get("weight_decay", 1e-4),
                "geometric_flip": settings.get("geometric_flip", True),
            },
        }
        self.adapter = Adapter(native_config)
        self.initialization_report = {"hero_task_weights_loaded": False}
        if initialize:
            self._initialize_generic(initialization)

    @property
    def native_model(self):
        return self.adapter.native_model

    def _initialize_generic(self, initialization):
        """严格加载官方使用的资产，构造模型本身绝不联网。"""
        dino_path = Path(initialization["dinov2"])
        resnet_path = Path(initialization["matching_antialiased"])
        dino = torch.load(dino_path, map_location="cpu", weights_only=True)
        self.native_model.cost_volume_net.model.load_state_dict(dino, strict=True)
        del dino

        source = antialiased_cnns.resnet18(pretrained=False)
        saved = torch.load(resnet_path, map_location="cpu", weights_only=True)
        source.load_state_dict(saved["state_dict"], strict=True)
        # 官方匹配头仅保留预训练ResNet的stem+layer1；后续16维描述头随机初始化。
        for index, name in enumerate(("conv1", "bn1", "relu", "maxpool", "layer1")):
            self.native_model.matching_model.net[index].load_state_dict(
                getattr(source, name).state_dict(), strict=True
            )
        del source, saved
        self.initialization_report.update({
            "foundation": {"path": initialization["foundation"],
                           "sha256": file_sha256(initialization["foundation"]),
                           "targets": ["reference ViT", "DPT shared layers"]},
            "dinov2": {"path": str(dino_path), "sha256": file_sha256(dino_path),
                       "targets": ["cost-volume ViT"]},
            "resnet18_antialiased": {
                "path": str(resnet_path), "sha256": file_sha256(resnet_path),
                "targets": ["matching stem and layer1"]},
            "random_modules": ["metadata MLP", "matching descriptor head", "cost-volume patchifier",
                               "mono/multi fusion projections", "depth/sky output heads"],
        })

    def forward(self, inputs, domain, reference=0, targets=None):
        if self.training and targets is None:
            raise ValueError("训练必须显式提供targets；推理请先model.eval()")
        self.adapter.reference_indices = [reference]
        self.native_model.run_opts.model_num_views = inputs["rgb"].shape[1]
        if self.training:
            work = self.adapter.prepare_training_inputs(inputs, targets)
            prediction = self.adapter(work)
            # 混合精度仅用于网络；log-depth与法线求导保持FP32，尤其保护平坦区域。
            with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
                prediction["native"] = [
                    {name: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                     for name, value in output.items()}
                    for output in prediction["native"]
                ]
                terms = self.adapter.loss(prediction, work, targets)
            return {
                "depth": prediction["depth"],
                "valid": prediction["depth_valid"],
                "loss": terms["total"],
                "loss_terms": terms,
                "unit_scale": 1.,
            }

        # eval不读取targets，不根据测试标签决定范围；共同固定TRAIN范围为默认。
        current, sources, _ = self.adapter.prepare_native(inputs, reference)
        current["min_depth"], current["max_depth"] = domain["near"], domain["far"]
        settings = self.config.get("mvsanywhere", {})
        native = self.native_model(
            "test", current, sources, unbatched_matching_encoder_forward=False,
            return_mask=True, estimate_depth_range=settings.get("estimate_depth_range", False),
            num_refinement_steps=int(settings.get("refinements", 1)),
        )
        depth = native["depth_pred_s0_b1hw"]
        return {"depth": depth, "valid": torch.isfinite(depth) & (depth > 0), "unit_scale": 1.}

    def configure_optimizer(self, total_steps=None):
        """原生三组AdamW；可显式按本轮短预算缩放调度，不改组间学习率。"""
        optimizer, _ = self.adapter.configure_optimizer()
        settings = self.config.get("mvsanywhere", {})
        if settings.get("schedule", "official") == "official":
            duration = 70000
            milestones = (70000, 80000)
        elif settings["schedule"] == "budget_scaled":
            if total_steps is None or total_steps < 1:
                raise ValueError("budget_scaled调度需要正整数total_steps")
            duration = int(total_steps)
            milestones = (int(total_steps * 140000 / 145000), int(total_steps) + 1)
        else:
            raise ValueError("schedule只能为official或budget_scaled")

        def matching(step):
            return 1. if step < milestones[0] else (.1 if step < milestones[1] else .01)

        def linear(step):
            return 1. - .999 * min(step, duration) / duration

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [matching, linear, linear])
        return optimizer, scheduler
