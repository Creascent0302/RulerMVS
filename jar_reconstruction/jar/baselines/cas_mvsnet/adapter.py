"""作者 CasMVSNet 网络的本地适配器；不是同骨干消融，也非论文成绩复现。

源码、版本和兼容改动记录于 third_party/cas_mvsnet。网络保持原作者
三阶段 CNN 结构；适配器只转换标定、轮换参考视图和统一评估输出。
完整 FP32 执行，避免对旧代码不受控的混合精度改动。
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ...geometry import (
    pixel_grid,
    unproject,
    project,
    sample_maps,
    image_texture,
    camera_centers,
    validate_inputs,
)
from ...fusion import enforce_roi


class OfficialCasMVSNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from third_party.cas_mvsnet.cas_mvsnet import CascadeMVSNet

        self.cfg = cfg
        m = cfg["model"]
        self.near, self.far = float(m["near"]), float(m["far"])
        self.budget = int(m["max_queries"])
        self.net = CascadeMVSNet(
            refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], cr_base_chs=[8, 8, 8]
        )
        self.register_buffer("depth_values", torch.linspace(self.near, self.far, 192))

    def forward(self, inputs, teacher=None):
        if teacher is not None:
            raise ValueError("CasMVSNet adapter never accepts GT teacher forcing")
        validate_inputs(inputs)
        with torch.autocast(device_type=inputs["rgb"].device.type, enabled=False):
            return self._forward_fp32(inputs)

    def _forward_fp32(self, inputs):
        rgb = inputs["rgb"].float()
        B, N, _, H, W = rgb.shape
        if H % 32 or W % 32 or min(H, W) < 32:
            raise ValueError("官方 CasMVSNet 三维卷积要求高宽为32的倍数；实验建议64×96以上")
        depth = rgb.new_zeros(B, N, H, W) + next(self.net.parameters()).sum() * 0
        confidence = torch.zeros_like(depth)
        stage_records = []
        for b in range(B):
            available = inputs["view_mask"][b].nonzero().flatten().tolist()
            if len(available) < 2:
                continue
            for ref in available:
                order = [ref] + [n for n in available if n != ref]
                images = rgb[b : b + 1, order]
                if "image_mask" in inputs:
                    images = images * inputs["image_mask"][b : b + 1, order, None]
                matrices = {}
                for stage, stride in enumerate((4, 2, 1), 1):
                    packed = rgb.new_zeros(1, len(order), 2, 4, 4)
                    packed[0, :, 0] = inputs["T_cam_from_rig"][b, order].float()
                    K = inputs["K"][b, order].float().clone()
                    K[:, :2] /= stride
                    packed[0, :, 1, :3, :3] = K
                    matrices[f"stage{stage}"] = packed
                result = self.net(images, matrices, self.depth_values[None])
                depth[b, ref] = result["depth"][0]
                confidence[b, ref] = result["photometric_confidence"][0]
                stage_records.append((b, ref, result))
        # The network predicts depth; this common post-filter checks calibrated
        # agreement before presenting a point as an observed surface. It cannot
        # certify that an ambiguous repeated-texture match is correct.
        valid_depth = (depth >= self.near) & (depth <= self.far) & torch.isfinite(depth)
        valid_depth &= inputs["view_mask"][..., None, None]
        if "image_mask" in inputs:
            valid_depth &= inputs["image_mask"]
        all_support = torch.zeros_like(depth, dtype=torch.long)
        texture = image_texture(rgb.flatten(0, 1)).reshape(B, N, H, W)
        grid = pixel_grid(H, W, device=rgb.device).reshape(-1, 2)
        scenes = []
        for b in range(B):
            points_list, refs_list = [], []
            for ref in range(N):
                X, _ = unproject(
                    grid,
                    depth[b, ref].flatten(),
                    inputs["K"][b, ref],
                    inputs["T_cam_from_rig"][b, ref],
                )
                points_list.append(X)
                refs_list.append(torch.full((H * W,), ref, device=rgb.device, dtype=torch.long))
                with torch.no_grad():
                    uv, z, visible, _ = project(
                        X.detach(),
                        inputs["K"][b],
                        inputs["T_cam_from_rig"][b],
                        (H, W),
                        inputs["view_mask"][b],
                        inputs.get("image_mask", None)[b] if "image_mask" in inputs else None,
                    )
                    other_depth = sample_maps(depth[b, :, None].detach(), uv, mode="nearest")[
                        ..., 0
                    ]
                    other_ok = (
                        sample_maps(valid_depth[b, :, None].float(), uv, mode="nearest")[..., 0]
                        > 0.5
                    )
                    centers = camera_centers(inputs["T_cam_from_rig"][b])
                    independent = (centers - centers[ref]).norm(dim=-1) > 1e-4
                    agree = visible & other_ok & ((other_depth - z).abs() < 0.2 + 0.01 * z.abs())
                    agree &= independent[:, None]
                    source_texture = sample_maps(texture[b, :, None], uv)[..., 0]
                    agree &= source_texture > self.cfg["model"].get("texture_threshold", 0.002)
                    support = torch.where(agree.any(0), agree.sum(0) + 1, 0).reshape(H, W)
                    support *= valid_depth[b, ref] & (
                        texture[b, ref] > self.cfg["model"].get("texture_threshold", 0.002)
                    )
                    all_support[b, ref] = support
            points = torch.cat(points_list)
            refs = torch.cat(refs_list)
            pixels = grid.repeat(N, 1)
            total = len(points)
            # Fixed, label-independent cap, shared across methods at evaluation.
            select = (
                torch.linspace(0, total - 1, min(total, self.budget), device=rgb.device)
                .round()
                .long()
            )
            points, refs, pixels = points[select], refs[select], pixels[select]
            sup = all_support[b].flatten()[select]
            conf = confidence[b].flatten()[select]
            valid = (sup >= self.cfg["model"].get("min_support", 2)) & (conf >= 0.1)
            # Normals are derived from this model's own depth only; no labels.
            p_grid = torch.stack(points_list).reshape(N, H, W, 3)
            dx = F.pad(
                (p_grid[:, :, 2:] - p_grid[:, :, :-2]).permute(0, 3, 1, 2),
                (1, 1, 0, 0),
                mode="replicate",
            ).permute(0, 2, 3, 1)
            dy = F.pad(
                (p_grid[:, 2:] - p_grid[:, :-2]).permute(0, 3, 1, 2), (0, 0, 1, 1), mode="replicate"
            ).permute(0, 2, 3, 1)
            normals = F.normalize(
                torch.linalg.cross(dx, dy, dim=-1).reshape(-1, 3)[select], dim=-1, eps=1e-6
            )
            fallback = torch.zeros_like(normals)
            fallback[:, 2] = 1
            normals = torch.where(normals.norm(dim=-1, keepdim=True) > 0.5, normals, fallback)
            scale = (
                depth[b].flatten()[select]
                / inputs["K"][b, refs, 0, 0]
                * (total / max(len(select), 1)) ** 0.5
            ).clamp(0.01, 2.0)
            state = torch.where(valid, 1, 2)
            scene = dict(
                points=points,
                normals=normals,
                scales=scale[:, None].expand(-1, 3),
                valid=valid,
                support=sup,
                state=state,
                confidence=conf * valid,
                centers=points,
                sizes=scale[:, None].expand(-1, 3) * 2,
                features=points.new_zeros(len(points), 1),
                state_logits=F.one_hot(state, 3).float() * 5,
                split_logits=scale * 0,
                aux={
                    "ref_index": refs,
                    "ref_uv": pixels,
                    "reference_depth": depth[b].flatten()[select],
                    "roi": self.cfg["model"]["roi"],
                },
            )
            scenes.append(enforce_roi(scene, self.cfg["model"]["roi"]))
        coarse = dict(
            depth=depth[..., ::16, ::16],
            confidence=confidence[..., ::16, ::16],
            support=all_support[..., ::16, ::16],
            texture=texture[..., ::16, ::16],
            evidence_threshold=0.1,
            texture_threshold=self.cfg["model"].get("texture_threshold", 0.002),
            min_support=self.cfg["model"].get("min_support", 2),
        )
        return dict(
            scenes=scenes,
            coarse=coarse,
            dense_depth=depth,
            depth_stride=1,
            dense_valid=valid_depth & (all_support >= 2),
            official_stages=stage_records,
        )


class OfficialCasMVSLoss(nn.Module):
    """原作者每级米制 SmoothL1 思路；空标签安全，权重0.5/1/2。"""

    def forward(self, outputs, inputs, targets):
        zero = outputs["dense_depth"].sum() * 0
        if "depth" not in targets:
            return {"total": zero}
        losses = []
        for b, ref, result in outputs["official_stages"]:
            for stage, (stride, weight) in enumerate(zip((4, 2, 1), (0.5, 1.0, 2.0)), 1):
                prediction = result[f"stage{stage}"]["depth"][0]
                gt = targets["depth"][b, ref, ::stride, ::stride]
                mask = targets.get("depth_valid", targets["depth"] > 0)[b, ref, ::stride, ::stride]
                mask = mask & torch.isfinite(gt) & (gt > 0)
                if "image_mask" in inputs:
                    mask &= inputs["image_mask"][b, ref, ::stride, ::stride]
                error = F.smooth_l1_loss(prediction, gt, reduction="none")
                losses.append(weight * torch.where(mask, error, 0).sum() / mask.sum().clamp_min(1))
        total = (
            torch.stack(losses).sum() / max(len(outputs["official_stages"]), 1) if losses else zero
        )
        return {"official_depth": total, "total": total}


def build_model_and_loss(cfg):
    return OfficialCasMVSNet(cfg), OfficialCasMVSLoss()
