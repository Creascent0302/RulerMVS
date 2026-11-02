"""以物理相机为观测单位的鲁棒外观共识与几何信息。

先消去各视图共享的未知外观，再计算几何更新。不会把同一张图参与
多个相机对误当成多次独立测量。方差聚合、IRLS与Schur消元都是已有
方法；此处是独立研究算子，尚未替换生产网络或证明精度收益。
"""

import torch
from torch import nn


def unique_camera_mask(valid, quality, camera_ids):
    """同一物理相机只取质量最高的一份表示；并列时保留最先一份。

    输入为[B,V,Q]。同源多尺度应更新原观测，而不作为独立票重复累加。
    """
    if camera_ids is None:
        return valid
    ids = torch.as_tensor(camera_ids, device=valid.device)
    if ids.ndim == 1:
        ids = ids[None].expand(valid.shape[0], -1)
    effective = torch.where(valid, quality, -torch.inf)
    order = torch.arange(valid.shape[1], device=valid.device)
    repeated = ids[:, :, None] == ids[:, None, :]
    better = effective[:, None] > effective[:, :, None]
    tied_first = (effective[:, None] == effective[:, :, None]) & (order[None, :] < order[:, None])[
        None, :, :, None
    ]
    superseded = (repeated[..., None] & (better | tied_first)).any(2)
    return valid & ~superseded


class CameraConsensus(nn.Module):
    """features[B,V,Q,C]，valid/quality[B,V,Q]。

    derivatives为dF/d参数[B,V,Q,C,P]；Z细化时P=1，局部平面时可为3。
    纹理特征的单位由调用方固定，inlier_radius是该单位下的残差半径。
    noise_sigma只规定局部线性化噪声模型，不自动给出正确匹配概率。
    identity_energy用于比较候选，information用于定位，两者不相乘。
    """

    def __init__(
        self,
        inlier_radius=0.6,
        noise_sigma=0.1,
        iterations=2,
        damping=1e-4,
        stop_weight_gradient=True,
        score_loss="m_estimator",
    ):
        super().__init__()
        if inlier_radius <= 0 or noise_sigma <= 0 or iterations < 0 or damping <= 0:
            raise ValueError("残差半径、噪声和阻尼须正，迭代次数须非负")
        self.inlier_radius = float(inlier_radius)
        self.noise_sigma = float(noise_sigma)
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.stop_weight_gradient = stop_weight_gradient
        if score_loss not in ("m_estimator", "reweighted_variance"):
            raise ValueError("score_loss须为m_estimator或研究消融reweighted_variance")
        self.score_loss = score_loss

    def forward(self, features, valid, *, quality=None, derivatives=None, camera_ids=None):
        # 小矩阵求解与鲁棒残差用FP32/64；可嵌入BF16训练而不产生半精度solve。
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._forward(
                features.to(dtype),
                valid,
                quality=None if quality is None else quality.to(dtype),
                derivatives=None if derivatives is None else derivatives.to(dtype),
                camera_ids=camera_ids,
            )

    def _forward(self, features, valid, *, quality=None, derivatives=None, camera_ids=None):
        if features.ndim != 4 or valid.shape != features.shape[:-1]:
            raise ValueError("features要求[B,V,Q,C]，valid要求[B,V,Q]")
        if quality is None:
            quality = torch.ones_like(valid, dtype=features.dtype)
        legal = valid & torch.isfinite(features).all(-1) & torch.isfinite(quality) & (quality > 0)
        legal = unique_camera_mask(legal, quality, camera_ids)
        values = torch.where(legal[..., None], features, 0.0)
        base = torch.where(legal, quality.clamp(0, 1), 0.0)
        # 只在少量相机之间求medoid，不产生全场景query两两attention。
        distance = (values[:, :, None] - values[:, None]).square().sum(-1)
        distance = (distance * base[:, None]).sum(2) / base.sum(1, keepdim=True).clamp_min(1e-12)
        medoid = distance.masked_fill(~legal, torch.inf).argmin(1)
        center = values.gather(
            1, medoid[:, None, :, None].expand(-1, 1, -1, values.shape[-1])
        ).squeeze(1)
        weights = base
        for _ in range(self.iterations):
            squared_residual = (values - center[:, None]).square().sum(-1)
            fraction = squared_residual / self.inlier_radius**2
            robust = (1 - fraction).clamp_min(0).square()
            weights = base * robust
            if self.stop_weight_gradient:
                weights = weights.detach()
            mass = weights.sum(1)
            center = (weights[..., None] * values).sum(1) / mass[..., None].clamp_min(1e-12)
        mass = weights.sum(1)
        center = (weights[..., None] * values).sum(1) / mass[..., None].clamp_min(1e-12)
        residual = values - center[:, None]
        squared_residual = residual.square().sum(-1)
        inlier_energy = (weights * squared_residual).sum(1) / mass.clamp_min(1e-12)
        energy = inlier_energy
        if self.iterations > 0 and self.score_loss == "m_estimator":
            # IRLS权重是鲁棒损失的导数，不能把被拒绝的残差从目标函数中免费删掉。
            fraction = squared_residual / self.inlier_radius**2
            bounded_loss = self.inlier_radius**2 / 3 * (1 - (1 - fraction).clamp_min(0).pow(3))
            energy = (base * bounded_loss).sum(1) / base.sum(1).clamp_min(1e-12)
        # 拒绝只有一个自洽视图的零残差；自证不能建立多视图表面。
        support = (weights > 1e-6).sum(1)
        supported = support >= 2
        result = {
            "identity_energy": torch.where(supported, energy, 0.0),
            "inlier_energy": inlier_energy,
            "valid": supported,
            "unknown": ~supported,
            "support": support,
            "weights": weights,
            "mean_feature": center,
            "weight_mass": mass,
            "hard_invalid": ~legal,
        }
        if derivatives is None:
            return result
        if derivatives.shape[:-1] != features.shape:
            raise ValueError("derivatives要求[B,V,Q,C,P]")
        derivative_valid = torch.isfinite(derivatives).all((-2, -1))
        geometric_weights = weights * derivative_valid
        derivative = torch.where((legal & derivative_valid)[..., None, None], derivatives, 0.0)
        total = geometric_weights.sum(1).clamp_min(1e-12)
        average = (geometric_weights[..., None, None] * derivative).sum(1) / total[..., None, None]
        centered_derivative = derivative - average[:, None]
        information = (
            torch.einsum(
                "bvq,bvqcp,bvqcr->bqpr", geometric_weights, centered_derivative, centered_derivative
            )
            / self.noise_sigma**2
        )
        numerator = (
            torch.einsum("bvq,bvqcp,bvqc->bqp", geometric_weights, centered_derivative, residual)
            / self.noise_sigma**2
        )
        parameters = derivatives.shape[-1]
        identity = torch.eye(parameters, device=features.device, dtype=features.dtype)
        update = -torch.linalg.solve(
            information + self.damping * identity, numerator[..., None]
        ).squeeze(-1)
        geometry_supported = supported & ((geometric_weights > 1e-6).sum(1) >= 2)
        update = torch.where(geometry_supported[..., None], update, 0.0)
        result.update(
            information=information,
            update=update,
            mean_derivative=average,
            geometric_weight_mass=total,
            geometric_valid=geometry_supported,
        )
        return result


def marginal_information_gain(
    weight_mass, mean_derivative, new_weight, new_derivative, noise_sigma=1.0
):
    """固定可靠性权重时，新独立相机对Schur信息矩阵的精确增量。

    所有维度为[...,C,P]或相应前缀。该式用于分配计算，不能证明新相机
    匹配正确；同camera_id的重复表示必须在调用前拒绝。
    """
    difference = new_derivative - mean_derivative
    factor = weight_mass * new_weight / (weight_mass + new_weight).clamp_min(1e-12)
    return (
        factor[..., None, None]
        * torch.einsum("...cp,...cr->...pr", difference, difference)
        / noise_sigma**2
    )
