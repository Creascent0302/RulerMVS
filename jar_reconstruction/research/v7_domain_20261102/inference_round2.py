"""连续深度码的推理计算复用；不修改训练模块、参数或候选。

粗深度格由expand产生，所有空间位置使用同一组候选。仅在空间
stride均为0时将深度MLP提升到每个候选一次，einsum显式广播。
任意非共享候选完全回退原forward；关闭时恢复原forward方法。
"""

import math
from types import MethodType

import torch

from .coordinate_prior import CoordinateEvidencePrior


def coordinate_reuse_forward(self, raw, information, null_mass, candidates):
    if candidates.stride(-2) != 0 or candidates.stride(-1) != 0:
        return self._coordinate_original_forward(raw, information, null_mass, candidates)
    if len(raw) != self.levels:
        raise ValueError('上下文特征层数与CoordinateEvidencePrior配置不符')
    features = self.project(torch.cat(raw, dim=1))
    spatial = (self.local(features) + self.context(features)) * .5
    coefficients = self.output(self.activation(features + spatial)).float()
    code = self.encode_depth(candidates[..., :1, :1]).float()
    logits = torch.einsum('bchw,bdhwc->bdhw', coefficients, code) / math.sqrt(self.code_channels)
    return logits * self.quality_gate(information, null_mass, candidates)


def prepare_coordinate_reuse(model_or_prior, enabled=True):
    """显式开关，不序列化方法绑定；训练/可求梯度模型不能安装部署重排。"""
    prior = (model_or_prior if isinstance(model_or_prior, CoordinateEvidencePrior)
             else model_or_prior.decoder.context_prior)
    if not isinstance(prior, CoordinateEvidencePrior):
        raise TypeError('此优化只适用于CoordinateEvidencePrior')
    keys = tuple(prior.state_dict())
    installed = hasattr(prior, '_coordinate_original_forward')
    if enabled and not installed:
        if prior.training or any(parameter.requires_grad for parameter in prior.parameters()):
            raise ValueError('先eval并冻结，再安装坐标码推理复用')
        prior._coordinate_original_forward = prior.forward
        prior.forward = MethodType(coordinate_reuse_forward, prior)
    elif not enabled and installed:
        prior.forward = prior._coordinate_original_forward
        del prior._coordinate_original_forward
    if tuple(prior.state_dict()) != keys:
        raise RuntimeError('部署优化不能修改state_dict键')
    return model_or_prior
