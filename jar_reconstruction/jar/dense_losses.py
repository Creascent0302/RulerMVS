"""稠密深度的独立监督；误差置信度不等于匹配证据或空间可见性。"""

import math

import torch
from torch.nn import functional as F


def sample_stage_labels(depth, depth_valid, inputs, stride, shape, near, far):
    """stride*s 的整数像素中心采样；越界位置屏蔽，不复制边缘标签。"""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("dense stage.stride 必须为正整数")
    height, width = depth.shape[-2:]
    y = torch.arange(shape[-2], device=depth.device) * stride
    x = torch.arange(shape[-1], device=depth.device) * stride
    inside = (y[:, None] < height) & (x[None] < width)
    yy, xx = y.clamp_max(height - 1)[:, None], x.clamp_max(width - 1)[None]
    label = depth[..., yy, xx].float()
    valid = depth_valid[..., yy, xx].bool() & inside
    valid &= inputs["view_mask"][..., None, None]
    if "image_mask" in inputs:
        valid &= inputs["image_mask"][..., yy, xx]
    valid &= torch.isfinite(label) & (label >= near) & (label <= far)
    return torch.where(valid, label, 1.0), valid


def masked_mean(value, valid):
    return torch.where(valid, value, 0.0).sum() / valid.sum().clamp_min(1)


def dense_depth_losses(dense, inputs, targets, model_config, settings):
    """返回未乘总权重的三个项；不读取 dense.valid/confidence/observed 筛监督。

    dense_depth = 米制 Huber + dense_log_weight × 对数深度 L1。
    dense_gradient = 相邻有效标签的对数深度梯度 L1，除以真实像素间距。
    confidence 的目标 exp(-|预测误差|/容差) 使用 detach 深度，只表示预计
    深度误差大小；不能据此宣称该像素已被多视图观测。
    """
    stages = dense.get("stages") or [{"depth": dense["depth"], "stride": 1}]
    # Empty reductions retain a zero-gradient path without reading masked NaNs.
    zero = sum(stage["depth"].reshape(-1)[:0].sum() for stage in stages)
    for stage in stages:
        if "confidence_logits" in stage:
            zero = zero + stage["confidence_logits"].reshape(-1)[:0].sum()
    empty = {"dense_depth": zero, "dense_gradient": zero, "dense_confidence": zero}
    if "depth" not in targets:
        return empty
    weights = settings.get("dense_stage_weights", [1.0] * len(stages))
    if (
        len(weights) != len(stages)
        or any(not math.isfinite(w) or w < 0 for w in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("dense_stage_weights 须与实际阶段数一致、有限非负且总和为正")
    near, far = model_config.get("near", 2.0), model_config.get("far", 80.0)
    beta = float(settings.get("dense_metric_beta_m", 0.1))
    log_weight = float(settings.get("dense_log_weight", 1.0))
    confidence_absolute = float(settings.get("dense_confidence_absolute_m", 0.1))
    confidence_relative = float(settings.get("dense_confidence_relative", 0.02))
    if beta < 0 or log_weight < 0 or confidence_absolute <= 0 or confidence_relative < 0:
        raise ValueError("dense loss 的误差尺度和权重须非负，置信度绝对容差须为正")
    truth = targets["depth"]
    depth_valid = targets.get("depth_valid", truth > 0)
    totals = dict(empty)
    for stage, weight in zip(stages, weights):
        prediction = stage["depth"].float()
        stride = stage["stride"]
        target, valid = sample_stage_labels(
            truth, depth_valid, inputs, stride, prediction.shape, near, far
        )
        if prediction.shape != target.shape:
            raise ValueError("dense stage.depth 必须为 [B,N,Hs,Ws]")
        safe = torch.where(valid, prediction, 1.0)
        metric = F.smooth_l1_loss(safe, target, reduction="none", beta=beta)
        log_prediction = safe.clamp_min(1e-6).log()
        log_target = target.log()
        depth_loss = masked_mean(metric + log_weight * (log_prediction - log_target).abs(), valid)
        gradient_sum = zero
        edge_count = valid.new_zeros((), dtype=torch.long)
        for axis in (-2, -1):
            left = [slice(None)] * 4
            right = [slice(None)] * 4
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            edge_valid = valid[left] & valid[right]
            prediction_gradient = log_prediction[right] - log_prediction[left]
            target_gradient = log_target[right] - log_target[left]
            error = (prediction_gradient - target_gradient).abs() / stride
            gradient_sum = gradient_sum + torch.where(edge_valid, error, 0.0).sum()
            edge_count = edge_count + edge_valid.sum()
        gradient_loss = gradient_sum / edge_count.clamp_min(1)
        confidence_loss = zero
        if "confidence_logits" in stage:
            logits = stage["confidence_logits"].float()
            if logits.shape != target.shape:
                raise ValueError("dense confidence_logits 必须与该级depth同形状")
            scale = confidence_absolute + confidence_relative * target
            desired = (-(safe.detach() - target).abs() / scale).exp()
            safe_logits = torch.where(valid, logits, 0.0)
            confidence_error = F.binary_cross_entropy_with_logits(
                safe_logits, desired, reduction="none"
            )
            confidence_loss = masked_mean(confidence_error, valid)
        totals["dense_depth"] = totals["dense_depth"] + weight * depth_loss
        totals["dense_gradient"] = totals["dense_gradient"] + weight * gradient_loss
        totals["dense_confidence"] = totals["dense_confidence"] + weight * confidence_loss
    result = {key: value / sum(weights) for key, value in totals.items()}
    if settings.get("dense_initial_depth", 0.0) > 0:
        if "refinement" not in dense:
            raise ValueError("dense_initial_depth 需要启用局部细化器")
        # s4 的 stage.depth 已被细化后深度覆盖；单独监督原始初值，
        # 不让局部匹配成功掩盖初值本身的误差。标签仍只在 loss 中读取。
        initial = dense["refinement"]["depth_before"].float()
        target, valid = sample_stage_labels(
            truth, depth_valid, inputs, 4, initial.shape, near, far
        )
        safe = torch.where(valid, initial, 1.0)
        metric = F.smooth_l1_loss(safe, target, reduction="none", beta=beta)
        log_error = (safe.clamp_min(1e-6).log() - target.log()).abs()
        result["dense_initial_depth"] = masked_mean(metric + log_weight * log_error, valid)
    if settings.get("dense_surface_recovery", 0.0) > 0:
        recovery = dense.get("surface_recovery")
        if recovery is None:
            raise ValueError("dense_surface_recovery 需要启用初始表面选择器")
        candidates = recovery["candidate_depth"].float().detach()
        candidate_valid = recovery["candidate_valid"]
        logits = recovery["logits"].float()
        target, valid = sample_stage_labels(
            truth, depth_valid, inputs, 4, candidates.shape[:2] + candidates.shape[-2:], near, far
        )
        valid &= candidate_valid[:, :, 0]
        risk = (candidates - target[:, :, None]).abs()
        risk = risk.clamp_max(40) + 3 * (risk - 8).clamp_min(0).clamp_max(25)
        risk = risk.masked_fill(~candidate_valid, 1e4)
        choices = risk.argmin(2)
        expected = (logits.softmax(2) * risk).sum(2)
        classification = F.cross_entropy(
            logits.permute(0, 1, 3, 4, 2)[valid], choices[valid], reduction="mean"
        ) if valid.any() else zero
        result["dense_surface_recovery"] = masked_mean(expected, valid) + 0.5 * classification
    return result
