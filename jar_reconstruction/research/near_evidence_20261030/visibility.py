"""训练期可见性责任监督；不改变生产网络，不增加推理参数。

用法：
    with capture_attention(model) as captured:
        prediction = model(inputs, domain, reference=0)
    auxiliary = auxiliary_loss(prediction, inputs, targets, domain, captured)
    loss = original_loss + 0.02 * auxiliary["loss"]

context 必须在 backward 前退出：checkpoint 重算时 hook 已移除，不会重复记录。
默认只否决 GT 明确遮挡的源证据；几何可见、未知和 null 均可承接剩余质量。
正/负 source-vs-null BCE 仅作为诊断，显式 mode="bce" 才用于训练。
"""

from contextlib import AbstractContextManager

import torch
from torch.nn import functional as F

from jar.geometry import pixel_grid, project, sample_maps, unproject


class AttentionCapture(AbstractContextManager):
    """仅保存两次前向现有的轻量权重张量，保留其 autograd 图。"""

    def __init__(self, model):
        model = getattr(model, "module", model)
        self.source = model.decoder.source
        self.records = []
        self.handle = None

    def __enter__(self):
        if self.handle is not None or self.records:
            raise RuntimeError("每次模型前向使用新的 AttentionCapture")
        self.handle = self.source.register_forward_hook(self._record)
        return self

    def _record(self, module, arguments, output):
        self.records.append({
            "weights": output["weights"],
            "null_mass": output["null_mass"],
            "source_valid": arguments[2].detach(),
        })

    def __exit__(self, *exception):
        self.handle.remove()
        self.handle = None
        return False


def capture_attention(model):
    return AttentionCapture(model)


def source_indices(view_count, source_count, reference):
    """与 _observe 的 compact_sources 顺序相同，不按相机名称猜槽位。"""
    if source_count == view_count:
        return list(range(view_count))
    if source_count == view_count - 1:
        return [index for index in range(view_count) if index != reference]
    raise ValueError("捕获的源数量与输入相机数量不一致")


@torch.no_grad()
def visibility_targets(prediction, inputs, targets, stage_name, captured, stride,
                       projection_limit=0.5, depth_tolerance=0.02):
    """返回最近 sampling 候选索引和逐源 {-1未知, 0遮挡, 1可见} 标签。

    GT 必须在该候选的回归区间内。候选和 GT 的源图投影距离不超过
    0.5 个 matching-stride4 像素；两处源 GT 必须给出相同可见/遮挡状态。
    这样不把细层已经失去的真实表面标签转交给任意远处候选。
    标签描述 GT 表面在候选采样邻域的可观测性，不把光度匹配视为已认证。
    """
    stage = prediction[stage_name]
    scale = float(prediction["unit_scale"])
    sampling = stage["sampling_depth"].detach().float() * scale
    lower = stage["lower"].detach().float() * scale
    upper = stage["upper"].detach().float() * scale
    batch, _, height, width = sampling.shape
    reference = int(prediction["reference_index"])
    source_count = captured["weights"].shape[1]
    view_count = inputs["K"].shape[1]
    indices = source_indices(view_count, source_count, reference)
    truth = targets["depth"][:, reference, ::stride, ::stride].detach().float()
    if truth.shape != (batch, height, width):
        raise ValueError("GT 整数 stride 格点与候选大小不一致")
    reference_valid = targets["depth_valid"][:, reference, ::stride, ::stride].bool().clone()
    reference_valid &= inputs["image_mask"][:, reference, ::stride, ::stride].bool()
    reference_valid &= inputs["view_mask"][:, reference, None, None].bool()
    reference_valid &= torch.isfinite(truth) & (truth > 0)
    if prediction.get("visibility_eval_range"):
        near, far = prediction["visibility_eval_range"]
        reference_valid &= (truth >= near) & (truth <= far)
    safe_truth = torch.where(reference_valid, truth, 1.)
    legal_sampling = stage["valid"] & torch.isfinite(sampling) & (sampling > 0)
    distance = (sampling.clamp_min(1e-6).log() - safe_truth[:, None].log()).abs()
    nearest = distance.masked_fill(~legal_sampling, torch.inf).argmin(1, keepdim=True)
    selected = sampling.gather(1, nearest)[:, 0]
    selected_lower = lower.expand_as(sampling).gather(1, nearest)[:, 0]
    selected_upper = upper.expand_as(sampling).gather(1, nearest)[:, 0]
    covered = reference_valid & legal_sampling.gather(1, nearest)[:, 0]
    covered &= (truth >= selected_lower) & (truth <= selected_upper)
    gather_index = nearest[:, None].expand(-1, source_count, -1, -1, -1)
    valid_selected = captured["source_valid"].gather(2, gather_index)[:, :, 0]
    labels = torch.full((batch, source_count, height, width), -1,
                        device=sampling.device, dtype=torch.int8)
    eligible = torch.zeros_like(labels, dtype=torch.bool)
    image_hw = inputs["rgb"].shape[-2:]
    matching_hw = inputs["matching_rgb"].shape[-2:]
    ratio = sampling.new_tensor((matching_hw[1] / image_hw[1], matching_hw[0] / image_hw[0])) / 4
    grid = pixel_grid(height, width, stride=stride, device=sampling.device).reshape(-1, 2)
    for sample in range(batch):
        K = inputs["K"][sample].float()
        T = inputs["T_cam_from_rig"][sample].float()
        points, _ = unproject(grid, safe_truth[sample].flatten(), K[reference], T[reference])
        candidate_points, _ = unproject(grid, selected[sample].flatten(), K[reference], T[reference])
        arguments = (K[indices], T[indices], image_hw,
                     inputs["view_mask"][sample, indices], inputs["image_mask"][sample, indices])
        uv, source_z, legal, _ = project(points, *arguments)
        candidate_uv, _, candidate_legal, _ = project(candidate_points, *arguments)
        separation = torch.linalg.vector_norm((candidate_uv - uv) * ratio, dim=-1)
        can_transfer = legal & candidate_legal & (separation <= projection_limit)
        can_transfer &= covered[sample].flatten()[None] & valid_selected[sample].flatten(1)
        if reference in indices:
            can_transfer[indices.index(reference)] = False
        depths = targets["depth"][sample, indices, None].float()
        masks = targets["depth_valid"][sample, indices, None].float()
        observed = sample_maps(depths, uv, mode="nearest")[..., 0]
        candidate_observed = sample_maps(depths, candidate_uv, mode="nearest")[..., 0]
        known = sample_maps(masks, uv, mode="nearest")[..., 0] > .5
        candidate_known = sample_maps(masks, candidate_uv, mode="nearest")[..., 0] > .5
        known &= torch.isfinite(observed) & (observed > 0)
        known &= candidate_known & torch.isfinite(candidate_observed) & (candidate_observed > 0)
        tolerance = depth_tolerance * source_z.abs().clamp_min(1e-3)
        visible = ((observed - source_z).abs() <= tolerance)
        visible &= (candidate_observed - source_z).abs() <= tolerance
        occluded = (observed < source_z - tolerance) & (candidate_observed < source_z - tolerance)
        label = labels[sample].flatten(1)
        label[can_transfer & known & visible] = 1
        label[can_transfer & known & occluded] = 0
        eligible[sample] = can_transfer.reshape(source_count, height, width)
    return {"nearest": nearest, "labels": labels, "eligible": eligible,
            "reference_valid": reference_valid, "covered": covered}


def auxiliary_loss(prediction, inputs, targets, domain, captured, mode="veto",
                   projection_limit=0.5, depth_tolerance=0.02):
    """默认按含负例的阶段等权平均，避免细层像素更多而自动占四倍权重。

    loss 是未乘系数的值，建议调用处固定 0.02。bce 只诊断或显式研究；
    未知不作负例。无可监督项返回连图零值，允许正常 backward。
    """
    if mode not in ("veto", "bce"):
        raise ValueError("mode 必须为 veto 或 bce")
    if captured.handle is not None:
        raise RuntimeError("先退出 capture context，再构造 loss 和 backward")
    zero = prediction["depth"].sum() * 0
    result = {"loss": zero, "veto": zero, "bce": zero, "stages": {},
              "positive_pairs": 0, "occluded_pairs": 0, "supervised_rays": 0,
              "known_pairs": 0, "skipped": False,
              "null_mass_mean": zero.detach(), "occluded_mass_all_rays": zero.detach()}
    if not domain.get("correspondence", True) or not {"depth", "depth_valid"}.issubset(targets):
        result["skipped"] = True
        return result
    if len(captured.records) != 2:
        raise ValueError("需要恰好捕获 coarse 和 fine 两次源注意力前向")
    geometric_prediction = dict(prediction, visibility_eval_range=domain.get("eval_range"))
    veto_terms, bce_terms, all_nulls, all_occluded = [], [], [], []
    for (name, stride), observation in zip((("coarse", 8), ("fine", 4)), captured.records):
        with torch.autocast(device_type=prediction["depth"].device.type, enabled=False):
            supervision = visibility_targets(geometric_prediction, inputs, targets, name, observation,
                                             stride, projection_limit, depth_tolerance)
        index = supervision["nearest"][:, None]
        count = observation["weights"].shape[1]
        weights = observation["weights"].float().gather(2, index.expand(-1, count, -1, -1, -1))[:, :, 0]
        null = observation["null_mass"].float().gather(2, index)[:, 0, 0]
        labels = supervision["labels"]
        known, occluded = labels >= 0, labels == 0
        negative_rays = occluded.any(1)
        occluded_mass = (weights * occluded).sum(1)
        # 与 1 - occluded_mass 等价，但避免质量接近1时相减损失精度。
        accepted = null + (weights * ~occluded).sum(1)
        total = null + weights.sum(1)
        veto = -((accepted / total.clamp_min(1e-12)).clamp_min(1e-12)).log()
        if bool(negative_rays.any()):
            veto_terms.append(veto[negative_rays].mean())
        pair_probability = weights / (weights + null[:, None]).clamp_min(1e-12)
        if bool(known.any()):
            bce_terms.append(F.binary_cross_entropy(pair_probability[known].clamp(1e-6, 1 - 1e-6),
                                                   labels[known].float()))
        valid_rays = supervision["reference_valid"]
        if bool(valid_rays.any()):
            all_nulls.append(null[valid_rays].detach().mean())
            all_occluded.append(occluded_mass[valid_rays].detach().mean())
        details = {"positive_pairs": int((labels == 1).sum()), "occluded_pairs": int(occluded.sum()),
                   "known_pairs": int(known.sum()), "supervised_rays": int(negative_rays.sum()),
                   "eligible_pairs": int(supervision["eligible"].sum()),
                   "unknown_eligible_pairs": int((supervision["eligible"] & ~known).sum()),
                   "reference_rays": int(valid_rays.sum()), "covered_rays": int(supervision["covered"].sum()),
                   "null_mass_mean": null[valid_rays].detach().mean() if bool(valid_rays.any()) else zero.detach(),
                   "occluded_mass_mean": occluded_mass[negative_rays].detach().mean()
                   if bool(negative_rays.any()) else zero.detach()}
        result["stages"][name] = details
        for key in ("positive_pairs", "occluded_pairs", "known_pairs", "supervised_rays"):
            result[key] += details[key]
    result["veto"] = torch.stack(veto_terms).mean() if veto_terms else zero
    result["bce"] = torch.stack(bce_terms).mean() if bce_terms else zero
    result["loss"] = result[mode]
    result["null_mass_mean"] = torch.stack(all_nulls).mean() if all_nulls else zero.detach()
    result["occluded_mass_all_rays"] = torch.stack(all_occluded).mean() if all_occluded else zero.detach()
    return result
