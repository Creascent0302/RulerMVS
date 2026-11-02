"""候选局部卷积的布局优化，不改变计算图、参数或卷积公式。"""

from torch import nn


class ContiguousDepthwise3d(nn.Conv3d):
    """让深度可分离3D卷积读取NCDHW，避开当前GPU较慢的NDHWC路径。"""

    def forward(self, value):
        return super().forward(value.contiguous())


def install_spatial_layout(model):
    """原位替换三个局部depthwise层；state_dict键与参数对象都保持不变。"""
    local = model.decoder.local
    changed = 0
    for block in (local, *local.scales):
        source = block.depthwise
        if isinstance(source, ContiguousDepthwise3d):
            continue
        target = ContiguousDepthwise3d(
            source.in_channels, source.out_channels, source.kernel_size,
            source.stride, source.padding, source.dilation, source.groups,
            source.bias is not None, source.padding_mode,
            device=source.weight.device, dtype=source.weight.dtype,
        )
        target.weight = source.weight
        target.bias = source.bias
        target.train(source.training)
        block.depthwise = target
        changed += 1
    return changed
