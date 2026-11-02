"""Shared lightweight CNN/FPN with explicit, aligned stride lattices."""

import torch
from torch import nn
import torch.nn.functional as F
from .geometry import pixel_grid, sample_maps


def norm(channels):
    max_groups = min(8, max(1, channels // 4))
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            norm(cout),  # 每张图像独立进行分组归一化
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, groups=cout, bias=False),  # 逐通道卷积
            norm(cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 1, bias=False),  # 1*1卷积核重组不同通道的信息
            norm(cout),
        )
        if cin != cout or stride != 1:
            self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class ImageFPN(nn.Module):
    """每台相机独立编码，输出步长为 4、8、16、32 的对齐特征。"""

    def __init__(self, cfg):
        super().__init__()
        model_config = cfg["model"]
        base_channels = model_config.get("backbone_width", 24)
        feature_channels = model_config.get("feature_dim", 64)

        # 五个 Block 的输出通道数，以及它们相对原图的采样步长。
        stage_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
            base_channels * 4,
        ]
        self.block_strides = (2, 4, 8, 16, 32)
        self.feature_strides = (4, 8, 16, 32)

        self.blocks = nn.ModuleList()
        input_channels = 3
        for output_channels in stage_channels:
            block = Block(input_channels, output_channels, stride=2)
            self.blocks.append(block)
            input_channels = output_channels

        # 第二至第五个 Block 的输出，分别映射到相同的特征通道数。
        self.lateral = nn.ModuleDict()
        for stride, input_channels in zip(self.feature_strides, stage_channels[1:]):
            self.lateral[str(stride)] = nn.Conv2d(input_channels, feature_channels, 1)

        # 保持先构建全部 lateral、再构建全部 smooth 的初始化顺序。
        self.smooth = nn.ModuleDict()
        for stride in self.feature_strides:
            self.smooth[str(stride)] = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=False),
                norm(feature_channels),
                nn.SiLU(),
            )

    def forward(self, inputs):
        # 1. 屏蔽无效输入，将相机维并入批次维。
        x = inputs["rgb"]  # [B, N, 3, H, W]
        batch_size, num_views, channels, height, width = x.shape
        view_mask = inputs.get(
            "view_mask",
            torch.ones(batch_size, num_views, device=x.device, dtype=torch.bool),
        )
        x = x * view_mask[:, :, None, None, None]
        if "image_mask" in inputs:
            x = x * inputs["image_mask"][:, :, None]
        x = x.reshape(batch_size * num_views, channels, height, width)

        # 2. 按 2、4、8、16、32 的顺序编码，保存后四层的侧向特征。
        raw = {}
        for stage, (stride, block) in enumerate(zip(self.block_strides, self.blocks)):
            x = block(x)
            x = self.refine_stage(stage, x)
            if stride == 2:
                continue
            raw[stride] = self.lateral[str(stride)](x)

        # 3. 从粗到细融合：32 -> 16 -> 8 -> 4。
        result = {}
        coarser_features = None
        coarser_stride = None
        for stride in reversed(self.feature_strides):
            current = raw[stride]  # [B*N, C, 当前高度, 当前宽度]

            if coarser_features is not None:
                feature_height, feature_width = current.shape[-2:]

                # 当前特征的采样位置，用原图像素坐标表达。
                uv = pixel_grid(
                    feature_height,
                    feature_width,
                    device=current.device,
                    dtype=torch.float32,
                    stride=stride,
                )  # [当前高度, 当前宽度, 2]
                uv = uv.reshape(1, -1, 2)  # [1, 位置数量, 2]
                uv = uv.expand(batch_size * num_views, -1, -1)

                # 在上一层读取这些位置，再恢复为当前层的空间尺寸。
                upsampled = sample_maps(
                    coarser_features, uv, stride=coarser_stride, padding_mode="border"
                )  # [B*N, 位置数量, C]
                upsampled = upsampled.transpose(1, 2)  # [B*N, C, 位置数量]
                upsampled = upsampled.reshape_as(current)
                current = current + upsampled

            # 下一层继承相加后的特征；smooth 只作用于本层输出。
            coarser_features = current
            coarser_stride = stride
            output = self.smooth[str(stride)](current)
            output = output.reshape(batch_size, num_views, *current.shape[1:])
            result[stride] = output * view_mask[:, :, None, None, None]

        return result

    def refine_stage(self, stage, features):
        """实验编码器的扩展点；原编码器不增加参数或计算。"""
        return features
