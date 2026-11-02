"""沿同一源射线比较候选证据，以及仅训练期的局部错误证据干预。"""

import math

import torch


def relative_response(cosine, valid, temperature=0.07):
    """[B,D,H,W]：减去该源合法候选响应的log-mean-exp共同分量。

    等于 temperature * log(softmax(cosine / temperature) * 合法候选数)。
    这是相对匹配响应，不是已校准的真实表面概率。常数响应返回零；
    所有候选无效时也返回零，不通过log(0)或-inf相减产生NaN。
    """
    count = valid.sum(1, keepdim=True)
    logits = (cosine.float() / temperature).masked_fill(~valid, -1e4)
    background = temperature * (torch.logsumexp(logits, dim=1, keepdim=True)
                                - count.clamp_min(1).float().log())
    return torch.where(valid & (count > 0), cosine.float() - background, 0.)


def corrupt_local_source(source, valid, probability, generator):
    """仅改变一台源相机局部候选的外观，保留标定和合法mask。

    source为[B,V,C,D,H,W]。一半宽×一半高区域内沿候选维循环移动，
    模拟投影合法但对应错误的证据；不改变参考图、单图bank或GT。
    随机数来自专用CPU generator，不改变数据采样或模型全局随机流。
    """
    batch, views, _, depth, height, width = source.shape
    result = source
    for sample in range(batch):
        if float(torch.rand((), generator=generator)) >= probability or depth < 2:
            continue
        usable = valid[sample].flatten(1).any(1).nonzero().flatten().tolist()
        if not usable:
            continue
        if result is source:
            result = source.clone()
        view = usable[int(torch.randint(len(usable), (), generator=generator))]
        patch_h, patch_w = max(1, height // 2), max(1, width // 2)
        top = int(torch.randint(height - patch_h + 1, (), generator=generator))
        left = int(torch.randint(width - patch_w + 1, (), generator=generator))
        shift = int(torch.randint(max(1, math.ceil(depth / 4)), max(2, math.ceil(3 * depth / 4)),
                                  (), generator=generator))
        patch = source[sample, view, :, :, top:top + patch_h, left:left + patch_w]
        result[sample, view, :, :, top:top + patch_h, left:left + patch_w] = patch.roll(shift, dims=1)
    return result
