"""单目上下文只参与源证据仲裁，不改变跨视图匹配描述子。

对已计算的pair evidence加入低秩双线性可靠性残差，并单独调节null。
它不声称单目上下文能认证遮挡；该可靠性仍由深度训练目标学习。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.pretraining.model import CandidateViewAttention


class ContextualViewAttention(CandidateViewAttention):
    def forward(self, reference, source, source_valid, ray_derivative,
                cell_half_width, use_geometry=True, context=None):
        """context为[B,4*raw_channels,H,W]，与当前候选xy网格精确对齐。

        checkpoint必须显式传context，不能从粗/细阶段共享的可变属性读取。
        旧query/key/pair/value与描述子保持不变，context只改变source/null权重。
        """
        if context is None:
            raise ValueError("上下文仲裁需要当前阶段对齐后的context")
        batch, _, _, _, height, width = source.shape
        if context.shape != (batch, self.context_projection.in_channels, height, width):
            raise ValueError("context必须与source的batch及xy网格一致")
        head_dim = self.channels // self.heads
        query = self.context_projection(context).permute(0, 2, 3, 1)
        query = query.unflatten(-1, (self.heads, head_dim))[:, None]
        null_delta = 0.5 * self.context_null(context).permute(0, 2, 3, 1).tanh()

        def reliability_hook(module, arguments, output):
            evidence = arguments[0]
            # 只用实际pair evidence与上下文交互；没有独立的候选位置分数旁路。
            evidence = F.layer_norm(evidence.float(), (self.channels,))
            evidence = evidence.unflatten(-1, (self.heads, head_dim))
            correction = (evidence * query.float()).sum(-1) / math.sqrt(head_dim)
            correction = 0.5 * correction.tanh()
            return output + correction.to(output.dtype)

        def null_hook(module, arguments, output):
            correction = null_delta if output.ndim == 4 else null_delta[:, None]
            return output + correction.to(output.dtype)

        reliability_handle = self.reliability.register_forward_hook(reliability_hook)
        null_handle = self.null.register_forward_hook(null_hook)
        try:
            return super().forward(reference, source, source_valid, ray_derivative,
                                   cell_half_width, use_geometry)
        finally:
            reliability_handle.remove()
            null_handle.remove()


def install_context(source, context_channels):
    """原位安装，旧参数对象/键不变，零初始化严格保留原始函数。"""
    if isinstance(source, ContextualViewAttention):
        if source.context_projection.in_channels != context_channels:
            raise ValueError("已安装的context通道数不同")
        return source
    if type(source) is not CandidateViewAttention:
        raise TypeError("上下文原型仅安装在原CandidateViewAttention上")
    prototype = next(source.parameters())
    with torch.random.fork_rng(devices=[]):
        projection = nn.Conv2d(context_channels, source.channels, 1)
        null = nn.Conv2d(context_channels, source.heads, 1)
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
        nn.init.zeros_(null.weight)
        nn.init.zeros_(null.bias)
    source.context_projection = projection.to(device=prototype.device, dtype=prototype.dtype)
    source.context_null = null.to(device=prototype.device, dtype=prototype.dtype)
    source.__class__ = ContextualViewAttention
    return source
