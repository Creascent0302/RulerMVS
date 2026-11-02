"""标定引导的图像上下文，只供查询读取，不改变匹配特征或相机参数。"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from jar.geometry import pixel_grid, safe_masked_softmax, sample_maps


class MaskedAttention(nn.Module):
    """有界 token 集合的注意力，全部 key 无效时严格返回零。"""

    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.output = nn.Linear(channels, channels)
        self.query_norm = nn.LayerNorm(self.head_dim)
        self.key_norm = nn.LayerNorm(self.head_dim)

    def forward(self, query, memory, valid):
        batch, count, channels = query.shape
        query = self.query(query).reshape(batch, count, self.heads, self.head_dim)
        key, value = self.key_value(memory).reshape(
            batch, memory.shape[1], 2, self.heads, self.head_dim
        ).unbind(2)
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        score = query.float() @ key.float().transpose(-1, -2)
        score = score / math.sqrt(self.head_dim)
        weights = safe_masked_softmax(score, valid[:, None, None], dim=-1)
        attended = (weights.to(value.dtype) @ value).transpose(1, 2)
        output = self.output(attended.reshape(batch, count, channels))
        return output * valid.any(-1)[:, None, None]


class ContextLayer(nn.Module):
    """先整理每张图像，再通过少量场景 token 交换跨相机信息。"""

    def __init__(self, channels, heads, window_size, pooled_grid, max_global_tokens):
        super().__init__()
        self.window_size = window_size
        self.pooled_grid = pooled_grid
        self.max_global_tokens = max_global_tokens
        self.local_norm = nn.LayerNorm(channels)
        self.scene_norm = nn.LayerNorm(channels)
        self.read_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.local = MaskedAttention(channels, heads)
        self.scene = MaskedAttention(channels, heads)
        self.read = MaskedAttention(channels, heads)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def _local_windows(self, tokens, valid):
        batch, views, height, width, channels = tokens.shape
        window = self.window_size
        pad_h = (-height) % window
        pad_w = (-width) % window
        padded = F.pad(tokens, (0, 0, 0, pad_w, 0, pad_h))
        mask = F.pad(valid, (0, pad_w, 0, pad_h))
        rows, cols = padded.shape[2] // window, padded.shape[3] // window
        windows = padded.reshape(batch, views, rows, window, cols, window, channels)
        windows = windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(-1, window * window, channels)
        mask = mask.reshape(batch, views, rows, window, cols, window)
        mask = mask.permute(0, 1, 2, 4, 3, 5).reshape(-1, window * window)
        normalized = self.local_norm(windows)
        windows = windows + self.local(normalized, normalized, mask)
        windows = windows * mask[..., None]
        restored = windows.reshape(batch, views, rows, cols, window, window, channels)
        restored = restored.permute(0, 1, 2, 4, 3, 5, 6)
        return restored.reshape(batch, views, rows * window, cols * window, channels)[
            :, :, :height, :width
        ]

    def forward(self, tokens, valid):
        batch, views, height, width, channels = tokens.shape
        tokens = self._local_windows(tokens, valid)
        # 每台相机至少保留一个槽位；总槽位数严格不超过配置预算。
        per_view = self.max_global_tokens // views
        rows = min(height, self.pooled_grid, max(1, int(math.sqrt(per_view))))
        cols = min(width, self.pooled_grid, per_view // rows)
        maps = tokens.permute(0, 1, 4, 2, 3).reshape(batch * views, channels, height, width)
        mask = valid.reshape(batch * views, 1, height, width).to(maps.dtype)
        support = F.adaptive_avg_pool2d(mask, (rows, cols))
        pooled = F.adaptive_avg_pool2d(maps * mask, (rows, cols))
        pooled = pooled / support.clamp_min(1e-6)
        scene = pooled.reshape(batch, views, channels, rows * cols)
        scene = scene.permute(0, 1, 3, 2).reshape(batch, -1, channels)
        scene_valid = support.reshape(batch, -1) > 0
        normalized = self.scene_norm(scene)
        scene = scene + self.scene(normalized, normalized, scene_valid)
        scene = scene * scene_valid[..., None]
        query = tokens.reshape(batch, -1, channels)
        query = query + self.read(self.read_norm(query), scene, scene_valid)
        query = query + self.ffn(self.ffn_norm(query))
        return query.reshape_as(tokens) * valid[..., None]


class CalibratedImageContext(nn.Module):
    """图像局部/全局交替上下文，返回与原 FPN 格点严格对齐的增量。

    输入沿用项目的 rgb、K、T_cam_from_rig、view_mask、image_mask。
    输出字典的尺寸与 pyramid 相同，通道数为 feature_dim。
    零初始化末层保证刚加载旧模型时增量为零。匹配特征没有被改写。
    """

    def __init__(self, cfg):
        super().__init__()
        model = cfg["model"]
        settings = model.get("image_context", {})
        channels = model.get("feature_dim", 64)
        hidden = int(settings.get("hidden_dim", 64))
        heads = int(settings.get("heads", 4))
        layers = int(settings.get("layers", 2))
        window = int(settings.get("window_size", 4))
        grid = int(settings.get("pooled_grid", 4))
        self.max_global_tokens = int(settings.get("max_global_tokens", 128))
        self.center_scale = float(settings.get("center_scale", 10.0))
        if min(hidden, heads, layers, window, grid, self.max_global_tokens) < 1:
            raise ValueError("image_context 的维度、层数和预算必须为正整数")
        if hidden % heads:
            raise ValueError("image_context.hidden_dim 必须被 heads 整除")
        if not math.isfinite(self.center_scale) or self.center_scale <= 0:
            raise ValueError("image_context.center_scale 必须为有限正数")
        self.input_projection = nn.Linear(channels, hidden)
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList(
            ContextLayer(hidden, heads, window, grid, self.max_global_tokens)
            for _ in range(layers)
        )
        self.output_projection = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _valid_pixels(inputs, height, width, stride):
        batch, views, _, image_h, image_w = inputs["rgb"].shape
        device = inputs["rgb"].device
        uv = pixel_grid(height, width, device=device, stride=stride)
        valid = (uv[..., 0] < image_w) & (uv[..., 1] < image_h)
        valid = valid.expand(batch, views, height, width)
        view_mask = inputs.get("view_mask", torch.ones(batch, views, device=device, dtype=torch.bool))
        valid = valid & view_mask[:, :, None, None]
        if "image_sizes" in inputs:
            sizes = inputs["image_sizes"]
            valid = valid & (uv[..., 1] < sizes[..., 0, None, None])
            valid = valid & (uv[..., 0] < sizes[..., 1, None, None])
        if "image_mask" in inputs:
            maps = inputs["image_mask"].reshape(batch * views, 1, image_h, image_w)
            sampled = sample_maps(maps.float(), uv.reshape(1, -1, 2).expand(batch * views, -1, -1))
            valid = valid & (sampled[..., 0].reshape_as(valid) > 0.999)
        return uv, valid

    def _geometry(self, inputs, uv, valid):
        batch, views = inputs["rgb"].shape[:2]
        with torch.autocast(device_type=uv.device.type, enabled=False):
            # 标定是固定输入；任何上游误设 requires_grad 均不允许改变相机。
            intrinsic = inputs["K"].detach().float()
            transform = inputs["T_cam_from_rig"].detach().float()
            present = valid.flatten(2).any(-1)
            intrinsic = torch.where(present[..., None, None], intrinsic, torch.eye(3, device=uv.device))
            transform = torch.where(present[..., None, None], transform, torch.eye(4, device=uv.device))
            homogeneous = torch.cat((uv, torch.ones_like(uv[..., :1])), -1).reshape(-1, 3)
            camera_rays = torch.linalg.solve(intrinsic, homogeneous.T.expand(batch, views, 3, -1))
            rotation = transform[..., :3, :3]
            rig_rays = (rotation.transpose(-1, -2) @ camera_rays).transpose(-1, -2)
            rig_rays = F.normalize(rig_rays, dim=-1)
            center = -(rotation.transpose(-1, -2) @ transform[..., :3, 3:4]).squeeze(-1)
            center = center[..., None, :].expand_as(rig_rays) / self.center_scale
            return torch.cat((rig_rays, center), -1).reshape(*valid.shape, 6)

    def forward(self, inputs, pyramid):
        batch, views, channels, height, width = pyramid[16].shape
        if views > self.max_global_tokens:
            raise ValueError("相机数不能大于 image_context.max_global_tokens")
        uv, valid = self._valid_pixels(inputs, height, width, 16)
        features = pyramid[16].permute(0, 1, 3, 4, 2)
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        geometry = self._geometry(inputs, uv, valid)
        tokens = self.input_projection(features) + self.geometry_projection(geometry)
        tokens = tokens * valid[..., None]
        for layer in self.layers:
            tokens = layer(tokens, valid)
        context = self.output_projection(tokens) * valid[..., None]
        context = context.permute(0, 1, 4, 2, 3).reshape(batch * views, channels, height, width)
        result = {}
        for stride, feature in pyramid.items():
            target_h, target_w = feature.shape[-2:]
            target_uv, target_valid = self._valid_pixels(inputs, target_h, target_w, stride)
            sampled = sample_maps(
                context, target_uv.reshape(1, -1, 2).expand(batch * views, -1, -1),
                stride=16, padding_mode="border",
            )
            sampled = sampled.transpose(1, 2).reshape(batch, views, channels, target_h, target_w)
            result[stride] = sampled * target_valid[:, :, None]
        return result
