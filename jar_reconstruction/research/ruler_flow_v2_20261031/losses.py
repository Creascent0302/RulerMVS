"""固定几何证据上的平均速度训练；真值仅进入监督，不进入模型推理。

时间从旧预测 t=1 走到目标 t=0。MeanFlow 的 JVP 只作用于小型
条件速度头，不穿过投影、grid_sample 或原图编码器。
"""

import torch


def masked_mean(value, mask):
    return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def target_coordinates(context, truth, truth_valid):
    """只监督最近的可达候选；未覆盖像素仍由原生全图损失监督。"""
    anchor = context["anchor"].detach().float()
    lower = context["lower"].detach().float()
    upper = context["upper"].detach().float()
    width = context["width"].detach().float()
    samples = context["sampling_depth"].detach().float()
    valid = context["valid"].detach().bool()
    safe_truth = torch.where(truth_valid & torch.isfinite(truth) & (truth > 0), truth, 1.)
    covered = valid & truth_valid & (width > 1e-8)
    covered &= torch.isfinite(truth) & (truth > 0) & (truth >= lower) & (truth <= upper)
    distance = (samples.clamp_min(1e-8).log() - safe_truth.log()).abs()
    nearest = distance.masked_fill(~covered, torch.inf).argmin(1, keepdim=True)
    assigned = torch.zeros_like(covered).scatter_(1, nearest, True) & covered
    coordinate = 2. * (safe_truth - lower) / width.clamp_min(1e-8) - 1.
    endpoint = torch.where(assigned, coordinate, anchor)
    native = distance.masked_fill(~valid, torch.inf).argmin(1, keepdim=True)
    disagreement = covered.any(1, keepdim=True) & (native != nearest)
    return {"anchor": anchor, "endpoint": endpoint, "assigned": assigned,
            "covered": covered, "responsibility_disagreement": disagreement}


def bridge_samples(targets, generator=None, noise_sigma=0., instantaneous_fraction=.5):
    """direct 对照也调用此函数，保证随机数流与 MeanFlow 一致。"""
    anchor, endpoint = targets["anchor"], targets["endpoint"]
    shape = (anchor.shape[0], 1, 1, 1)
    draws = torch.rand((3, *shape), device=anchor.device, generator=generator)
    t = torch.maximum(draws[0], draws[1])
    r = torch.minimum(draws[0], draws[1])
    r = torch.where(draws[2] < instantaneous_fraction, t, r)
    # 默认零噪声是条件纠错，而不是从标准高斯采样的生成模型。
    noise = torch.randn(anchor.shape, device=anchor.device, generator=generator) * noise_sigma
    start = (anchor + noise).clamp(-1., 1.)
    start = torch.where(targets["assigned"], start, anchor)
    state = (1. - t) * endpoint + t * start
    tangent = start - endpoint
    return {"state": state, "r": r, "t": t, "tangent": tangent, "start": start}


def meanflow_pair(velocity, state, r, t, tangent):
    """u_target = stopgrad[v - (t-r) (v·∂q u + ∂t u)]。

    velocity 的闭包条件保持固定。参数梯度通过 u 保留；目标全部停止
    梯度，避免二阶参数优化，也不使用差分近似冒充精确 JVP。
    """
    estimate, derivative = torch.func.jvp(
        velocity, (state, r, t),
        (tangent, torch.zeros_like(r), torch.ones_like(t)),
    )
    target = (tangent - (t - r) * derivative).detach()
    return estimate, target


def flow_objective(model, prediction, inputs, targets, domain, options=None):
    """返回新增损失；原深度/选面/风险损失由训练入口继续计算。"""
    options = options or {}
    objective = options.get("objective", "meanflow")
    if objective not in {"direct", "flow", "meanflow"}:
        raise ValueError("objective 必须是 direct、flow 或 meanflow")
    context = prediction["_flow_context"]
    reference = prediction["reference_index"]
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, reference:reference + 1, ::4, ::4].float() / scale
    valid = targets["depth_valid"][:, reference:reference + 1, ::4, ::4].clone()
    valid &= inputs["image_mask"][:, reference:reference + 1, ::4, ::4]
    valid &= inputs["view_mask"][:, reference:reference + 1, None, None]
    valid &= torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        valid &= (truth * scale >= low) & (truth * scale <= high)
    coordinates = target_coordinates(context, truth, valid)
    bridge = bridge_samples(coordinates, options.get("_generator"),
                            float(options.get("noise_sigma", 0.)),
                            float(options.get("instantaneous_fraction", .5)))
    assigned = coordinates["assigned"]
    start_target = coordinates["anchor"] - coordinates["endpoint"]
    # 主 forward 已求出真实起点的一步平均速度，不额外重新采样图像。
    initial = context["initial_velocity"].float()
    start_loss = masked_mean((initial - start_target).square(), assigned)
    bridge_loss = initial.sum() * 0.
    target_magnitude = bridge_loss.detach()
    if objective != "direct":
        field = model.decoder.flow_field

        def velocity(state, r, t):
            return field.velocity(context, state, r, t).float()

        if objective == "meanflow":
            estimate, target = meanflow_pair(
                velocity, bridge["state"], bridge["r"], bridge["t"], bridge["tangent"])
        else:
            estimate = velocity(bridge["state"], bridge["t"], bridge["t"])
            target = bridge["tangent"]
        bridge_loss = masked_mean((estimate - target).square(), assigned)
        target_magnitude = masked_mean(target.abs(), assigned).detach()
    total = float(options.get("start_weight", 1.)) * start_loss
    total = total + float(options.get("bridge_weight", .25)) * bridge_loss
    return {
        "total": total, "start_mse": start_loss, "bridge_mse": bridge_loss,
        "bridge_target_abs": target_magnitude,
        "supervised_pixels": valid.sum(), "reachable_pixels": coordinates["covered"].any(1).sum(),
        "assigned_cells": assigned.sum(),
        "responsibility_disagreement": coordinates["responsibility_disagreement"].sum(),
    }
