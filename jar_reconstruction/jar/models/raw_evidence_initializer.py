"""只替换自研初值生成：完整单图特征、候选证据注入和预训练 DPT。

借用范围仅为 MVSAnywhere 的 DPT 深度解码器与对应权重；不构造其图像
编码器、元数据代价体、融合 ViT 或完整网络。四层 768 维特征由原自研
编码器提供，后续选面和标定细化仍由调用方执行。

新增几何注入为零初始化，起点严格等于同一个 DPT 直接读取原始特征，
不等于原自研初值。几何可采样和单图预测均不表示表面已被观测确认。
"""

import math
from pathlib import Path
import sys

import torch
from torch import nn

from jar.encoder import norm
from jar.geometry import pixel_grid, sample_maps
from .shared_initializer import PrimaryDepthHead


def _build_depth_decoder():
    """直接构造官方 DPT；不构造完整 MVSAnywhere，不联网下载。"""
    source = Path(__file__).resolve().parents[2] / "official_methods/upstream/mvsanywhere/src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mvsanywhere.modules.depth_anything_blocks import DPTHead

    return PrimaryDepthHead(
        DPTHead(model_name="dinov2_vitb14", nclass=1, prediction_scale=1.0)
    )


def align_candidate_packet(packet, coarse_stride, patch_size):
    """候选格点 stride*i → 原始特征中心 7.5+16*i，边缘采用 border。"""
    uv = pixel_grid(*patch_size, device=packet.device, stride=16) + 7.5
    uv = uv.reshape(1, -1, 2).expand(len(packet), -1, -1)
    sampled = sample_maps(packet, uv, stride=coarse_stride, padding_mode="border")
    return sampled.transpose(1, 2).reshape(len(packet), packet.shape[1], *patch_size)


class RawEvidenceInitializer(nn.Module):
    """原始四层特征解码；默认只训练原输出头与新增几何注入。

    raw_maps 为四个 [B,768,Hp,Wp]；packet 为 [B,4*bins+5,Hc,Wc]，
    采用 DenseDepthDecoder._candidate_packet 的既有合同。state 为本模块
    完整 state_dict；提供时严格恢复且不读取 settings.weights。

    本模块不接收 targets 或标定真值。参考图缺失、原图有效区域和最终
    观测状态由调用方独立处理；不能把本模块的连续深度直接标为已知表面。
    """

    def __init__(self, settings, bins, near, far, state=None):
        super().__init__()
        self.bins = int(bins)
        self.near, self.far = float(near), float(far)
        self.use_geometry = bool(settings.get("use_geometry", True))
        hidden = int(settings.get("hidden_dim", 128))
        if self.bins < 2 or hidden < 1 or not 0 < self.near < self.far < math.inf:
            raise ValueError("候选数、通道和米制深度范围不合法")
        self.depth_decoder = _build_depth_decoder()
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(4 * self.bins + 5, hidden, 1, bias=False),
            norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            norm(hidden),
            nn.GELU(),
        )
        self.geometry_projections = nn.ModuleList(
            [nn.Conv2d(hidden, 768, 1) for _ in range(4)]
        )
        for projection in self.geometry_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        if state is not None:
            self.load_state_dict(state, strict=True)
        else:
            weights = settings.get("weights", "")
            if not weights:
                raise ValueError("初值解码器需要已有 DPT weights 或完整 state，禁止随机初始化")
            record = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
            saved = record.get("state_dict", record.get("model", record))
            prefix = "native_model.depth_decoder." if any(
                key.startswith("native_model.depth_decoder.") for key in saved
            ) else "depth_decoder."
            decoder_state = {
                key[len(prefix):]: value for key, value in saved.items() if key.startswith(prefix)
            }
            if not decoder_state:
                raise ValueError("权重中缺少完整 depth_decoder 参数")
            self.depth_decoder.load_state_dict(decoder_state, strict=True)
        self.configure_trainable(settings.get("train_decoder", "head"))

    def configure_trainable(self, mode="head"):
        """head：只调原主输出头；decoder：调完整主解码；none：冻结 DPT。"""
        if mode not in ("head", "decoder", "none"):
            raise ValueError("train_decoder 必须为 head、decoder 或 none")
        self.requires_grad_(False)
        if self.use_geometry:
            self.geometry_encoder.requires_grad_(True)
            self.geometry_projections.requires_grad_(True)
        if mode == "head":
            self.depth_decoder.output_convs[0].requires_grad_(True)
        elif mode == "decoder":
            self.depth_decoder.requires_grad_(True)
            # PrimaryDepthHead 不执行三个辅助深度头或 sky 头。
            for head in self.depth_decoder.output_convs[1:]:
                head.requires_grad_(False)
            self.depth_decoder.sky_convs.requires_grad_(False)
            # 最深融合块只有单输入，resConfUnit1 的 skip 分支从不执行。
            self.depth_decoder.scratch.refinenet4.resConfUnit1.requires_grad_(False)

    def forward(self, raw_maps, packet, coarse_stride, output_size):
        if len(raw_maps) != 4 or raw_maps[0].ndim != 4 or raw_maps[0].shape[1] != 768:
            raise ValueError("raw_maps 必须为四个 [B,768,Hp,Wp] 张量")
        shape = raw_maps[0].shape
        if any(value.shape != shape for value in raw_maps):
            raise ValueError("四层原始特征的尺寸必须相同")
        if packet.ndim != 4 or packet.shape[:2] != (shape[0], 4 * self.bins + 5):
            raise ValueError("packet 必须为 [B,4*bins+5,Hc,Wc]")
        if coarse_stride not in (4, 8, 16, 32):
            raise ValueError("coarse_stride 必须为 4、8、16 或 32")
        patch_size = shape[-2:]
        maps = [torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for value in raw_maps]
        if self.use_geometry:
            clean = torch.nan_to_num(packet.float(), nan=0.0, posinf=0.0, neginf=0.0)
            aligned = align_candidate_packet(clean, coarse_stride, patch_size)
            # valid 流位于第 3 组 bins 通道；无几何支持时不注入偏置。
            supported = aligned[:, 2 * self.bins:3 * self.bins].amax(1, keepdim=True) > 0.999
            geometry = self.geometry_encoder(aligned)
            maps = [
                value + projection(geometry).to(value.dtype) * supported
                for value, projection in zip(maps, self.geometry_projections)
            ]
        # 官方 use_clstoken=False；保留原接口但 class token 不参与预测。
        features = [(value.flatten(2).transpose(1, 2), value.new_zeros(len(value), 768)) for value in maps]
        logits = self.depth_decoder(features, *patch_size)["log_depth_pred_s0_b1hw"].float()
        output_h, output_w = map(int, output_size)
        if min(output_h, output_w) < 1 or output_h > logits.shape[-2] or output_w > logits.shape[-1]:
            raise ValueError("output_size 必须是 DPT 原生全图范围内的正尺寸")
        # 默认完整 RGB 输出；非 16 整除输入只裁去右/下 padding，不做隐式 resize。
        depth = (math.log(self.near) + math.log(self.far / self.near) * logits.sigmoid()).exp()
        return {
            "depth": depth[..., :output_h, :output_w],
            "logits": logits[..., :output_h, :output_w],
        }
