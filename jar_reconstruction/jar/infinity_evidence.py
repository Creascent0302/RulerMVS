"""用纯旋转重投影检验有限距离；不识别天空语义，也不产生 free 标签。"""

import torch

from jar.geometry import project


def project_infinity(rays, intrinsics, transforms, image_hw, view_mask=None, image_mask=None):
    """rig 射线方向在无限远处的精确投影，平移在该极限中消失。"""
    rotation_only = transforms.clone()
    rotation_only[:, :3, 3] = 0
    uv, _, valid, _ = project(rays, intrinsics, rotation_only, image_hw, view_mask, image_mask)
    return uv, valid


def finite_depth_evidence(finite_cost, infinity_cost, common, margin=0.01):
    """相同描述子、相同源视图逐一比较；至少一个源视图明显支持有限距离。

    若所有可比较视图都无法排除无限远，则保留 unknown。没有共同有效视图
    时不以无限远假说否决原匹配（例如无限远投影恰好落到图外）。这只是
    保守的额外拒绝器，并不证明通过的点正确，也不证明拒绝的点是天空。
    """
    improvement = infinity_cost.float() - finite_cost.float()
    votes = common & torch.isfinite(improvement) & (improvement > margin)
    compared = common.sum(0)
    return {
        "accepted": (compared == 0) | votes.any(0),
        "votes": votes.sum(0),
        "compared": compared,
        "best_margin": improvement.masked_fill(~common, -2.0).amax(0),
    }
