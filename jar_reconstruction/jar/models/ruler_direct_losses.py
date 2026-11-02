"""公制候选解码的逐阶段监督；仅训练端读取深度标签。"""

import torch
from torch.nn import functional as F

def initializer_loss(prediction, targets, weights):
    """绝对距离保证米制定位，log项兼顾近距，梯度项保留空间变化与边界。"""
    valid = targets["valid"].bool()
    if not bool(valid.any()):
        raise ValueError("训练样本没有有效深度监督")
    truth = torch.where(valid, targets["depth"].float(), 1.0)
    estimate = prediction.float()
    logarithm, target_log = estimate.clamp_min(1e-6).log(), truth.log()
    absolute = (estimate - truth).abs()[valid].mean() / weights["metric_normalizer_m"]
    log_absolute = (logarithm - target_log).abs()[valid].mean()
    gradient_terms = []
    for axis in (-1, -2):
        first, second = [slice(None)] * 4, [slice(None)] * 4
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        supported = valid[first] & valid[second]
        if bool(supported.any()):
            residual = logarithm[first] - logarithm[second] - target_log[first] + target_log[second]
            gradient_terms.append(residual.abs()[supported].mean())
    gradient = torch.stack(gradient_terms).mean() if gradient_terms else estimate.sum() * 0
    total = absolute + weights["log_weight"] * log_absolute + weights["gradient_weight"] * gradient
    return {"total": total, "metric": absolute, "log": log_absolute, "gradient": gradient}


def stage_loss(stage, truth, mask, label_depth="continuous"):
    scores, depths = stage["scores"], stage["candidate_depth"]
    valid = stage["valid"].bool()
    distance = (depths - truth).abs()
    if label_depth == "continuous":
        label_distance = distance.detach()
    elif label_depth == "sampling":
        # 非重叠公制单元由固定刻度划分；偏移不能改变类别归属。
        label_distance = (stage["sampling_depth"].detach() - truth).abs()
    else:
        raise ValueError("label_depth须为continuous或sampling")
    indices = label_distance.masked_fill(~valid, torch.inf).argmin(1)
    supervised = mask[:, 0] & valid.any(1)
    zero = scores.sum() * 0
    if not bool(supervised.any()):
        return {"classification": zero, "risk": zero, "location": zero}
    classification = F.cross_entropy(scores, indices, reduction="none")[supervised].mean()
    probability = stage["probability"]
    risk = (probability * distance).sum(1)[supervised].mean() / 20
    closest = depths.gather(1, indices[:, None])
    location = F.smooth_l1_loss(closest[:, 0][supervised], truth[:, 0][supervised], beta=0.5) / 20
    return {"classification": classification, "risk": risk, "location": location}


def subray_distribution(stage, truth, mask):
    """每个stride8粗格负责(0/4, 0/4)四条子射线，返回它们的表面分布。

    分类标签使用固定sampling_depth的最近格点；与旧center损失按预测的
    continuous深度重选类别不同，避免offset更新同时移动分类标签。
    这里的GT只构造训练目标，不生成或修改推理候选。
    """
    children = torch.cat([truth[..., dy::8, dx::8] for dy in (0, 4) for dx in (0, 4)], 1)
    child_valid = torch.cat([mask[..., dy::8, dx::8] for dy in (0, 4) for dx in (0, 4)], 1).bool()
    child_valid = child_valid & torch.isfinite(children) & (children > 0)
    child_valid = child_valid & stage["valid"].any(1, keepdim=True)
    children = torch.where(child_valid, children, 1.)
    distance = (stage["sampling_depth"].detach()[:, :, None] - children[:, None]).abs()
    index = distance.masked_fill(~stage["valid"][:, :, None], torch.inf).argmin(1)
    distribution = torch.zeros_like(stage["scores"])
    distribution.scatter_add_(1, index, child_valid.to(distribution.dtype))
    distribution = distribution / child_valid.sum(1, keepdim=True).clamp_min(1)
    return distribution, children, child_valid


def subray_set_loss(stage, truth, mask):
    """粗层负责覆盖子射线的表面集合；细层仍逐射线选唯一表面。"""
    distribution, children, child_valid = subray_distribution(stage, truth, mask)
    supervised = child_valid.any(1)
    scores, depths = stage["scores"], stage["candidate_depth"]
    zero = (scores.sum() + depths.sum()) * 0
    if not bool(supervised.any()):
        return {"classification": zero, "risk": zero, "location": zero}
    classification = -(distribution * F.log_softmax(scores, dim=1)).sum(1)[supervised].mean()
    distance = (depths[:, :, None] - children[:, None]).abs()
    set_distance = distance.masked_fill(~child_valid[:, None], torch.inf).amin(2)
    set_distance = torch.where(supervised[:, None], set_distance, 0)
    risk = (stage["probability"] * set_distance).sum(1)[supervised].mean() / 20
    index = distance.detach().masked_fill(~stage["valid"][:, :, None], torch.inf).argmin(1)
    closest = depths.gather(1, index)
    location = F.smooth_l1_loss(closest[child_valid], children[child_valid], beta=0.5) / 20
    return {"classification": classification, "risk": risk, "location": location}


def direct_loss(result, targets, settings):
    coarse_supervision = settings.get("coarse_supervision", "center")
    if coarse_supervision not in ("center", "subray_set"):
        raise ValueError("coarse_supervision必须为center或subray_set")
    if coarse_supervision == "subray_set" and not bool(targets["valid"].any()):
        zero = result["depth"].sum() * 0
        loss = {name: zero for name in ("total", "metric", "log", "gradient")}
    else:
        loss = initializer_loss(result["depth"], targets, settings)
    for name, stride in (("coarse", 8), ("fine", 4)):
        if name == "coarse" and coarse_supervision == "subray_set":
            stage = subray_set_loss(result[name], targets["depth"], targets["valid"])
        else:
            stage = stage_loss(result[name], targets["depth"][..., ::stride, ::stride], targets["valid"][..., ::stride, ::stride])
        loss.update({name + "_" + key: value for key, value in stage.items()})
        loss["total"] = loss["total"] + settings.get(name + "_weight", 1.0) * (
            settings.get("classification_weight", 1.0) * stage["classification"]
            + settings.get("risk_weight", 1.0) * stage["risk"] + stage["location"]
        )
    return loss


def multireference_loss(prediction, inputs, targets, settings):
    """等权平均有监督的样本-相机，缺标签不当零深度且不稀释loss。

    用于真实RGB入口；模型本身不读取targets。每个样本单独归一化，
    避免像素更多的相机/帧掩盖其它视角。监督范围仅限制loss。
    """
    from .ruler_direct_training import select_reference_targets

    references = prediction["reference_indices"]
    losses = []
    samples = set()
    valid_pixels = 0
    defaults = dict(metric_normalizer_m=20., log_weight=0.2, gradient_weight=0.05)
    options = {**defaults, **settings}
    for position, reference in enumerate(references):
        result = prediction["direct_by_reference"][reference]
        selected = select_reference_targets(
            inputs, targets, prediction["direct_camera_indices"][:, position],
            depth_range=options.get("depth_range_m", (2., 80.)),
            prediction_valid=result["valid"],
        )
        for sample in range(inputs["rgb"].shape[0]):
            if not bool(selected["valid"][sample].any()):
                continue
            single = {"depth": result["depth"][sample:sample + 1]}
            for stage in ("coarse", "fine"):
                single[stage] = {key: value[sample:sample + 1] for key, value in result[stage].items()
                                 if torch.is_tensor(value)}
            truth = {key: value[sample:sample + 1] for key, value in selected.items()}
            losses.append(direct_loss(single, truth, options))
            samples.add(sample)
            valid_pixels += int(truth["valid"].sum())
    if losses:
        answer = {key: torch.stack([loss[key] for loss in losses]).mean() for key in losses[0]}
    else:
        zero = sum(result["depth"].sum() * 0 for result in prediction["direct_by_reference"].values())
        answer = {"total": zero}
    device = inputs["rgb"].device
    answer["supervised_samples"] = torch.tensor(len(samples), device=device)
    answer["supervised_views"] = torch.tensor(len(losses), device=device)
    answer["supervised_pixels"] = torch.tensor(valid_pixels, device=device)
    return answer


