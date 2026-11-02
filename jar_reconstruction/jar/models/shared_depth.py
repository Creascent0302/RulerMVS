"""共享预训练初值与轻量米制残差，不重复运行旧初值网络。"""

import torch
from torch import nn

from ..geometry import unproject, pixel_grid, validate_inputs
from ..multifocal.data import mask_camera_inputs
from .shared_initializer import SharedPretrainedInitializer
from .metric_residual import MetricResidualHead, prepare_evidence


def metric_depth_loss(prediction, baseline, target, valid, settings):
    """平均误差、尾部误差与负修正惩罚；标签只进入此训练函数。"""
    valid = valid & torch.isfinite(target) & (target > 0)
    target = torch.where(valid, target.float(), 1.0)
    prediction = torch.where(valid, prediction.float(), 1.0)
    baseline = torch.where(valid, baseline.float(), 1.0)
    error = prediction - target
    base_error = baseline - target
    count = valid.sum().clamp_min(1)

    def mean(value):
        return torch.where(valid, value, 0.0).sum() / count

    terms = {
        "metric_l1": mean(error.abs()),
        "metric_mse": mean(error.square()),
        "harm": mean((error.abs() - base_error.abs() - settings.get("harm_margin_m", 0.02)).relu()),
        "residual_reg": mean((prediction - baseline).square()),
    }
    weights = {"metric_l1": 1.0, "metric_mse": 0.025, "harm": 0.5, "residual_reg": 0.002}
    terms["total"] = sum(settings.get(key, weight) * terms[key] for key, weight in weights.items())
    return terms


class SharedGeometryDepthModel(nn.Module):
    """一次预训练联合解码，保留全图深度，仅学习有界米制修正。

    当前输出深度假设，不宣称已验证占据或已观测表面。旧自研网络仍可独立选择。
    """

    def __init__(self, cfg, checkpoint_state=None):
        super().__init__()
        self.cfg = cfg
        settings = cfg["model"]["shared_depth"]
        self.refinement_settings = settings.get("refinement", {})
        use_refinement = self.refinement_settings.get("enabled", True)
        state = checkpoint_state
        initializer_state = None
        if state is not None:
            initializer_state = {
                key.removeprefix("initializer."): value
                for key, value in state.items() if key.startswith("initializer.")
            }
            if not initializer_state:
                raise ValueError("共享深度 checkpoint 必须完整包含初值器")
            if use_refinement and not any(key.startswith("metric_head.") for key in state):
                raise ValueError("启用米制细化时 checkpoint 必须包含对应模块")
        self.initializer = SharedPretrainedInitializer(settings["initializer"], initializer_state)
        self.metric_head = MetricResidualHead(
            context_channels=self.initializer.decoder_channels,
            settings=self.refinement_settings,
        ) if use_refinement else None
        self.reference_indices = (self.initializer.reference_index,)

    def forward(self, inputs, teacher=None):
        inputs = mask_camera_inputs(inputs, self.cfg.get("camera", {}))
        if self.cfg["model"].get("validate_inputs", True):
            validate_inputs(inputs)
        initial = self.initializer(inputs)
        work = {**inputs, "view_mask": initial["effective_view_mask"]}
        use_refinement = self.metric_head is not None and self.refinement_settings.get("enabled", True)
        if use_refinement:
            packet = prepare_evidence(
                work, initial["depth"], initial["matching_features"],
                reference_index=self.initializer.reference_index,
                settings=self.refinement_settings,
            )
            result = self.metric_head(packet, initial["decoder_features"], initial["depth"])
        else:
            result = {"depth": initial["depth"]}
        batch, views, _, height, width = inputs["rgb"].shape
        reference = self.initializer.reference_index
        depth = initial["depth"].new_zeros(batch, views, height, width)
        valid = torch.zeros_like(depth, dtype=torch.bool)
        depth[:, reference] = result["depth"][:, 0]
        valid[:, reference] = initial["valid"][:, 0]
        observed = torch.zeros_like(valid)
        return {
            "representation": "shared_metric_depth",
            "depth_stage": "metric_refined" if use_refinement else "initial",
            "initial_depth": initial["depth"],
            "depth": result["depth"],
            "depth_valid": initial["valid"],
            "depth_K": inputs["K"][:, reference : reference + 1],
            "reference_indices": [reference],
            "dense_depth": depth,
            "dense_valid": valid,
            "dense_observed": observed,
            "dense": {"depth": depth, "valid": valid, "observed": observed},
            "baseline_depth": initial["depth"],
            "metric_refinement": result,
            "scenes": [],
        }

    def training_loss(self, prediction, inputs, targets):
        depth = prediction["depth"]
        if "depth" not in targets:
            return {"total": depth.sum() * 0}
        reference = self.initializer.reference_index
        truth = targets["depth"]
        valid = targets.get("depth_valid", torch.isfinite(truth) & (truth > 0))
        if truth.ndim == 5:
            truth, valid = truth.squeeze(2), valid.squeeze(2)
        truth = truth[:, reference : reference + 1]
        valid = valid[:, reference : reference + 1].bool().clone()
        valid &= prediction["depth_valid"]
        valid &= inputs["view_mask"][:, reference, None, None, None]
        if "image_mask" in inputs:
            valid &= inputs["image_mask"][:, reference : reference + 1]
        model = self.cfg["model"]
        valid &= torch.isfinite(truth) & (truth >= model["near"]) & (truth <= model["far"])
        pixels = pixel_grid(*truth.shape[-2:], device=truth.device)
        bounds = truth.new_tensor(model["roi"])
        for batch in range(len(truth)):
            points, _ = unproject(
                pixels, torch.where(valid[batch, 0], truth[batch, 0], 1.0),
                inputs["K"][batch, reference], inputs["T_cam_from_rig"][batch, reference],
            )
            valid[batch, 0] &= ((points >= bounds[:, 0]) & (points <= bounds[:, 1])).all(-1)
        return metric_depth_loss(
            depth, prediction["baseline_depth"], truth, valid,
            self.cfg["model"]["shared_depth"].get("loss", {}),
        )
