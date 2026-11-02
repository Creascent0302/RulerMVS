"""把线性投影放在四层图像特征端，避免对每个深度候选重复投影。"""

import math

import torch
from torch.nn import functional as F


def factorized_visual(features, bank, query, output):
    """与 output(softmax(query(features)^T bank) bank) 代数等价。

    features: [B,C,D,H,W]；bank: [B,R,4,H,W]。
    query/output 必须是原模型的带偏置 1×1×1 卷积，不增加参数。
    浮点归约顺序不同；训练与部署均保留原参数上的梯度。
    """
    keys = F.conv3d(bank, query.weight.transpose(0, 1))
    logits = torch.einsum("bcdhw,bclhw->bldhw", features, keys)
    if query.bias is not None:
        bias = torch.einsum("c,bclhw->blhw", query.bias, bank)
        logits = logits + bias[:, :, None]
    weights = (logits.float() / math.sqrt(bank.shape[1])).softmax(1)
    # 权重和为一，所以 output 的偏置也能提前放到四层值特征上。
    values = output(bank)
    return torch.einsum("bldhw,bclhw->bcdhw", weights.to(values.dtype), values)
