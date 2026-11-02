"""新增模块的显式GT监督；推理模块不能调用这些函数。"""

import torch
from torch.nn import functional as F
from jar.geometry import pixel_grid, unproject, project


def auxiliary_losses(outputs, inputs, targets):
    terms = {}
    status = {}
    depth = targets.get("depth")
    tele = outputs["coarse"].get("tele")
    if depth is None:
        return terms, {
            "tele_initializer": "skipped_missing_depth",
            "birth_score": "skipped_missing_depth",
        }
    valid = (
        targets.get("depth_valid", depth > 0)
        & inputs["view_mask"][..., None, None]
        & inputs["image_mask"]
    )
    if tele is not None:
        h, w = tele["depth"].shape[-2:]
        uv = pixel_grid(h, w, depth.device, stride=16).reshape(-1, 2)
        xx = uv[:, 0].long().clamp(max=depth.shape[-1] - 1)
        yy = uv[:, 1].long().clamp(max=depth.shape[-2] - 1)
        d = depth[:, 4:6, yy, xx].reshape(depth.shape[0], 2, h, w)
        ok = valid[:, 4:6, yy, xx].reshape_as(d) & (d > 0)
        match = torch.zeros_like(ok)
        for b in range(len(depth)):
            for ref in (4, 5):
                other = 9 - ref
                X, _ = unproject(
                    uv,
                    d[b, ref - 4].reshape(-1),
                    inputs["K"][b, ref],
                    inputs["T_cam_from_rig"][b, ref],
                )
                loc, z, vis, J = project(
                    X,
                    inputs["K"][b, other : other + 1],
                    inputs["T_cam_from_rig"][b, other : other + 1],
                    depth.shape[-2:],
                    inputs["view_mask"][b, other : other + 1],
                    inputs["image_mask"][b, other : other + 1],
                )
                x = loc[0, :, 0].round().long().clamp(0, depth.shape[-1] - 1)
                y = loc[0, :, 1].round().long().clamp(0, depth.shape[-2] - 1)
                gt = depth[b, other, y, x]
                agree = vis[0] & valid[b, other, y, x] & ((gt - z[0]).abs() < 0.05 + 0.005 * gt)
                match[b, ref - 4] = agree.reshape(h, w)
        inv = tele["inverse_depths"]
        rho = d.clamp_min(1e-6).reciprocal()
        nearest = (rho[..., None] - inv).abs().argmin(-1)
        match &= (rho >= inv.min()) & (rho <= inv.max())
        match &= tele["logits"][:, :, :-1].detach().gather(2, nearest[:, :, None]).squeeze(2) > -1e3
        label = torch.where(match, nearest, len(inv))
        ce = F.cross_entropy(tele["logits"].flatten(0, 1), label.flatten(0, 1), reduction="none")
        terms["mf_tele_class"] = (ce * ok.flatten(0, 1)).sum() / ok.sum().clamp_min(1)
        status["tele_initializer"] = (
            "supervised_depth_and_visible_correspondence"
            if ok.any()
            else "skipped_no_labeled_tele_pixels"
        )
    else:
        status["tele_initializer"] = "disabled"
    if "mf_birth_logits" in outputs:
        logits = outputs["mf_birth_logits"]
        h, w = logits.shape[-2:]
        # 深度跳变邻域是细结构候选监督，不使用实例ID猜正对应。
        rho = torch.where(valid, depth.clamp_min(1e-5).reciprocal(), 0.0)
        b, n, H, W = rho.shape
        mx = F.max_pool2d(rho.reshape(b * n, 1, H, W), 5, 1, 2).reshape_as(rho)
        mn = -F.max_pool2d(-rho.reshape(b * n, 1, H, W), 5, 1, 2).reshape_as(rho)
        desired = ((mx - mn) > 0.02 * rho.clamp_min(1e-4)).float()
        y = (torch.arange(h, device=depth.device) * 4).clamp(max=H - 1)
        x = (torch.arange(w, device=depth.device) * 4).clamp(max=W - 1)
        labels = desired[..., y[:, None], x[None, :]]
        mask = valid[..., y[:, None], x[None, :]] & (inputs["camera_roles"] == 1)[..., None, None]
        positives = (labels * mask).sum()
        negatives = ((1 - labels) * mask).sum()
        weight = (negatives / positives.clamp_min(1)).clamp(1, 10).detach()
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), labels, reduction="none", pos_weight=weight
        )
        terms["mf_birth_score"] = (loss * mask).sum() / mask.sum().clamp_min(1)
        status["birth_score"] = (
            "supervised_depth_discontinuities" if mask.any() else "skipped_no_labeled_tele_pixels"
        )
    else:
        status["birth_score"] = "disabled"
    return terms, status
