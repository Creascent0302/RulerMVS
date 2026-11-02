"""仅训练期使用的候选风险、距离均衡与条件GT误差；不改变推理结构。"""

import torch

from jar.pretraining.losses import masked_mean


def fine_actual_regret(prediction, inputs, targets, domain):
    """用连续修正后的候选误差监督评分，风险目标中的候选位置与GT停止梯度。

    old_risk保留采样位置误差的期望，new_risk改用实际候选误差。
    regret减去每条射线的最小实际误差；概率归一化后，它对评分的
    梯度与new_risk相同，不能将减去常数称为另一种梯度优化机制。
    梯度仍可沿评分特征回传；此函数不改CE、location或原有位置分支目标。
    """
    scale = float(prediction["unit_scale"])
    truth = targets["depth"][:, :1].detach().float() / scale
    mask = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    mask = mask & inputs["view_mask"][:, :1, None, None]
    mask = mask & torch.isfinite(truth) & (truth > 0)
    if domain.get("eval_range"):
        low, high = domain["eval_range"]
        mask = mask & (truth * scale >= low) & (truth * scale <= high)
    truth = torch.where(mask, truth, 1.)[..., ::4, ::4]
    stage = prediction["fine"]
    mask = mask[..., ::4, ::4] & stage["ref_valid"]

    sampling = stage["sampling_depth"].detach().float()
    actual = stage["candidate_depth"].detach().float()
    probability = stage["probability"].float()
    legal = stage["valid"] & torch.isfinite(sampling) & (sampling > 0)
    legal = legal & torch.isfinite(actual) & (actual > 0)
    legal = legal & torch.isfinite(probability) & (probability >= 0)
    # 先替换非法值再取log，避免被mask掉的NaN经反向传播污染评分。
    sampling = torch.where(legal, sampling, 1.)
    actual = torch.where(legal, actual, 1.)
    old_error = (sampling.clamp_min(1e-6).log() - truth.log()).abs().clamp_max(4.)
    new_error = (actual.clamp_min(1e-6).log() - truth.log()).abs().clamp_max(4.)
    weight = torch.where(legal, probability, 0.)
    mass = weight.sum(1, keepdim=True)
    weight = weight / mass.clamp_min(1e-12)
    mask = mask & legal.any(1, keepdim=True) & (mass > 0)
    minimum = new_error.masked_fill(~legal, 4.).amin(1, keepdim=True)

    return {
        "old_risk": masked_mean((weight * old_error).sum(1, keepdim=True), mask),
        "new_risk": masked_mean((weight * new_error).sum(1, keepdim=True), mask),
        "regret": masked_mean((weight * (new_error - minimum)).sum(1, keepdim=True), mask),
        "valid_pixels": mask.sum(),
    }


def balanced_depth_error(prediction, truth, mask, limits):
    """近/中/远非空距离段等权；truth和limits须处于相同训练归一单位。

    每像素误差沿用log绝对误差+.2倍截断相对误差，不加入梯度/候选损失。
    这里的等权只用于优化；正式指标仍使用全部GT像素的既定分母。
    """
    truth = truth.detach().float()
    mask = mask & torch.isfinite(truth) & (truth > 0)
    target = torch.where(mask, truth, 1.)
    estimate = torch.where(mask, prediction.float(), 1.).clamp_min(1e-6)
    error = (estimate.log() - target.log()).abs()
    error = error + .2 * ((estimate - target).abs() / target.clamp_min(.1)).clamp_max(100.)
    low, high = limits
    bands = (mask & (truth < low), mask & (truth >= low) & (truth < high), mask & (truth >= high))
    losses = torch.stack([masked_mean(error, band) for band in bands])
    present = torch.stack([band.any() for band in bands]).to(losses.dtype)
    return losses.sum() / present.sum().clamp_min(1.)


def excess_error(estimate, teacher, truth, mask):
    """仅学生比固定teacher更差时，增加其到GT的log误差梯度。

    这是条件GT重加权，不要求学生输出等于teacher，也不保证防止遗忘。
    teacher和GT均停止梯度；学生优于或等于teacher时该项为0。
    调用方可传入各距离段mask再进行均衡聚合。
    """
    teacher = teacher.detach().float()
    truth = truth.detach().float()
    mask = mask & torch.isfinite(truth) & (truth > 0)
    mask = mask & torch.isfinite(teacher) & (teacher > 0)
    target = torch.where(mask, truth, 1.)
    reference = torch.where(mask, teacher, 1.)
    student = torch.where(mask, estimate.float(), 1.).clamp_min(1e-6)
    student_error = (student.log() - target.log()).abs()
    teacher_error = (reference.log() - target.log()).abs()
    return masked_mean((student_error - teacher_error).relu(), mask)
