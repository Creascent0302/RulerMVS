"""让组内深度与上采样条件使用同一组三点分布，不新增网络参数。

V6 mean 深度是选中组的条件期望；目前凸上采样却读取该组单点赢家的
特征。这里仅替换后者：先选一个组，再对该组三个特征加权，不跨组融合。
固定组选取和有效掩码时，这个读出对特征及组内分数连续；跨组 argmax
仍然离散，不能据此称完整网络连续或已证明精度提高。

最小集成方式（本文件不自动安装、不修改运行中的模型）：

    selected_features = pool_selected_group_features(
        fine_features, result["selected_group"],
        result["conditional_probability"], fine_valid,
    )

在旧 _upsample 中增加可选 selected_features 参数。参数缺省时仍执行
原 selected_index 的特征 gather；非空时直接交给原 upsample_mask。
后面的邻域有效性、凸权重、深度/置信度混合及支持数读取保持原样。
_render 仅显式转发这一可选参数，不能把 selected_index 改成伪造索引。

建议只对 within_group="mean" 打开；winner 读出继续使用旧赢家特征。
这个修改不改粗层、细候选、组概率、细深度或原参数形状，但原上采样头
接收到的特征分布会变化，可能需要同预算适配，不能承诺无训练就变好。
"""

import torch


def pool_selected_group_features(features, selected_group, conditional_probability,
                                 candidate_valid=None):
    """返回 [B,C,H,W] 的已选组条件特征，保留组内权重及特征梯度。

    features: [B,C,3K,H,W]，每个父组的三个候选连续排列。
    selected_group: [B,1,H,W] 的整数组索引，通常来自组概率 argmax。
    conditional_probability: [B,K,3,H,W]，应直接使用 handoff 已算的条件概率。
    candidate_valid: 可选 [B,3K,H,W]；它是合法候选掩码，非可见性认证。

    只 gather 已选组三个特征，避免展开或混合所有候选组。无效/非有限
    特征、非有限/非正权重不参与；剩余权重重新归一化。没有有效质量
    （总和 <= 1e-12）时返回零特征，调用者仍须保留原几何和图像掩码。
    累加采用 FP32，最后恢复 features 的 dtype 供原卷积读取。
    """
    if features.ndim != 5 or features.shape[2] == 0 or features.shape[2] % 3:
        raise ValueError("features须为[B,C,3K,H,W]，至少包含一个组")
    batch, channels, count, height, width = features.shape
    if selected_group.shape != (batch, 1, height, width):
        raise ValueError("selected_group须为[B,1,H,W]")
    if conditional_probability.shape != (batch, count // 3, 3, height, width):
        raise ValueError("conditional_probability须为[B,K,3,H,W]")
    if candidate_valid is not None and candidate_valid.shape != (batch, count, height, width):
        raise ValueError("candidate_valid须为[B,3K,H,W]")

    offsets = torch.arange(3, device=features.device)[None, :, None, None]
    indices = selected_group * 3 + offsets
    selected = features.gather(2, indices[:, None].expand(-1, channels, -1, -1, -1))
    group_indices = selected_group[:, :, None].expand(-1, -1, 3, -1, -1)
    weights = conditional_probability.gather(1, group_indices)[:, 0].float()

    valid = torch.isfinite(selected).all(1) & torch.isfinite(weights) & (weights > 0)
    if candidate_valid is not None:
        valid = valid & candidate_valid.gather(1, indices).bool()
    weights = torch.where(valid, weights, 0.)
    mass = weights.sum(1, keepdim=True)
    weights = torch.where(mass > 1e-12, weights / mass.clamp_min(1e-12), 0.)
    selected = torch.where(valid[:, None], selected, 0.).float()
    pooled = (selected * weights[:, None]).sum(2)
    return pooled.to(features.dtype)
