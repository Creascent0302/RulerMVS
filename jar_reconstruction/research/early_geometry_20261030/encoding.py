"""早期相机条件研究原型；不读取标签，不改变生产编码器。

7维条件是相机局部单位射线、两方向角足迹的log、两方向导数夹角余弦，
以及可用阵列横向基线RMS/depth_scale的log1p。角足迹不是已知物体尺寸，
基线只是潜在几何杠杆，不认证可见性，也不能解除共心相机的尺度退化。
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from jar.geometry import pixel_grid, sample_maps
from jar.pretraining.encoders import MatchingPyramid, TrainableFoundation
from jar.pretraining.model import PretrainingModel


@torch.no_grad()
def camera_descriptor(inputs, domain, grid, footprint):
    """主图像素坐标grid[H,W,2]和足迹(dx,dy) -> [B,N,7,H,W]。

局部微分用于描述采样角尺度；此处没有以一阶投影替代候选精确投影。
缺图相机先替换标定，再计算逆矩阵和基线，不能只在计算后乘零。
"""
    original = inputs["K"]
    dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    device = original.device
    K = original.to(dtype)
    T = inputs["T_cam_from_rig"].to(dtype)
    available = inputs["view_mask"].bool()
    if "image_mask" in inputs:
        available = available & inputs["image_mask"].flatten(2).any(2)
    K = torch.where(available[..., None, None], K, torch.eye(3, device=device, dtype=dtype))
    T = torch.where(available[..., None, None], T, torch.eye(4, device=device, dtype=dtype))
    inverse = torch.linalg.inv(K)
    pixels = grid.to(device=device, dtype=dtype).reshape(-1, 2)
    pixels = torch.cat((pixels, torch.ones_like(pixels[:, :1])), -1)
    q = torch.einsum("bnij,pj->bnpi", inverse, pixels)
    length = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    ray = q / length
    derivatives = []
    for axis in (0, 1):
        direction = inverse[..., :, axis][:, :, None]
        derivatives.append((direction - ray * (direction * ray).sum(-1, keepdim=True)) / length)
    du, dv = derivatives
    norm_u = torch.linalg.vector_norm(du, dim=-1).clamp_min(1e-12)
    norm_v = torch.linalg.vector_norm(dv, dim=-1).clamp_min(1e-12)
    footprint = torch.as_tensor(footprint, device=device, dtype=dtype)
    angular = torch.stack(((norm_u * footprint[0]).log(), (norm_v * footprint[1]).log()), -1)
    correlation = ((du * dv).sum(-1) / (norm_u * norm_v)).clamp(-1, 1)[..., None]

    rotation, translation = T[..., :3, :3], T[..., :3, 3]
    centers = -(rotation.transpose(-1, -2) @ translation[..., None])[..., 0]
    displacement = centers[:, None] - centers[:, :, None]
    local_baseline = torch.einsum("bnij,bnmj->bnmi", rotation, displacement)
    along = torch.einsum("bnmc,bnpc->bnmp", local_baseline, ray)
    transverse_squared = (local_baseline.square().sum(-1)[..., None] - along.square()).clamp_min(0)
    views = K.shape[1]
    pairs = available[:, :, None] & available[:, None, :]
    pairs &= ~torch.eye(views, device=device, dtype=torch.bool)[None]
    count = pairs.sum(2).clamp_min(1)
    rms = ((transverse_squared * pairs[..., None]).sum(2) / count[..., None]).sqrt()
    baseline = (rms / float(domain["depth_scale"])).log1p()[..., None]
    descriptor = torch.cat((ray, angular, correlation, baseline), -1)
    descriptor = torch.where(available[:, :, None, None], descriptor, 0.)
    return descriptor.movedim(-1, 2).reshape(K.shape[0], views, 7, *grid.shape[:2])


def matching_camera(inputs, domain, shape, stride=4):
    main_h, main_w = inputs["rgb"].shape[-2:]
    match_h, match_w = inputs["matching_rgb"].shape[-2:]
    scale = inputs["K"].new_tensor((match_w / main_w, match_h / main_h))
    grid = pixel_grid(*shape, device=scale.device, stride=stride)
    grid = (grid + .5) / scale - .5
    return camera_descriptor(inputs, domain, grid, stride / scale)


def foundation_camera(inputs, domain, resized_shape, patch=14):
    main_h, main_w = inputs["rgb"].shape[-2:]
    height, width = resized_shape
    inverse_scale = inputs["K"].new_tensor((main_w / width, main_h / height))
    grid = pixel_grid(height // patch, width // patch, device=inverse_scale.device, stride=patch)
    grid = (grid + patch / 2) * inverse_scale - .5
    return camera_descriptor(inputs, domain, grid, patch * inverse_scale)


class CameraAdapter(nn.Module):
    """通道相关乘性残差；初始恒等，幅度限制在±10%。"""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value, descriptor, tokens=False):
        conditions = descriptor.movedim(1, -1)
        if tokens:
            conditions = conditions.flatten(1, 2)
        gain = .1 * self.net(conditions.to(value.dtype)).tanh()
        if not tokens:
            gain = gain.movedim(-1, 1)
        return value * (1 + gain)


def conditioned_map(descriptor, camera_mode):
    return torch.zeros_like(descriptor) if camera_mode == "constant" else descriptor


class ConditionedMatching(MatchingPyramid):
    def forward(self, inputs, domain, camera_mode="geometry"):
        image = inputs["matching_rgb"]
        batch, views = image.shape[:2]
        valid = inputs["view_mask"][:, :, None, None] & inputs["matching_image_mask"]
        image = torch.where(valid[:, :, None], image, 0).flatten(0, 1)
        image = (image - self.mean) / self.std
        net = self.backbone
        value = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        camera = matching_camera(inputs, domain, value.shape[-2:]).flatten(0, 1)
        value = self.camera_adapter(value, conditioned_map(camera, camera_mode))
        raw = []
        for layer, projection in zip((net.layer1, net.layer2, net.layer3, net.layer4), self.lateral):
            needs_gradient = value.requires_grad or any(p.requires_grad for p in layer.parameters())
            if torch.is_grad_enabled() and needs_gradient:
                value = checkpoint(layer, value, use_reentrant=False)
            else:
                value = layer(value)
            raw.append(projection(value))
        maps, top = {}, None
        for index in (3, 2, 1, 0):
            stride = 4 * 2 ** index
            value = raw[index]
            if top is not None:
                grid = pixel_grid(*value.shape[-2:], device=value.device, stride=stride)
                uv = grid.reshape(1, -1, 2).expand(batch * views, -1, -1)
                up = sample_maps(top.float(), uv, stride=stride * 2, padding_mode="border")
                value = value + up.transpose(1, 2).reshape_as(value).to(value.dtype)
            top = value
            if index < 3:
                maps[stride] = self.smooth[index](value).unflatten(0, (batch, views))
        return maps


class ConditionedFoundation(TrainableFoundation):
    def forward(self, image, inputs, domain, reference=0, camera_mode="geometry"):
        image = (image - self.mean) / self.std
        image = F.interpolate(image, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        camera = foundation_camera(inputs, domain, image.shape[-2:])[:, reference]
        camera = conditioned_map(camera, camera_mode)

        def inject(module, arguments, output):
            return self.camera_adapter(output, camera, tokens=True)

        # hook只包住patch嵌入；退出后块级checkpoint不会再次插入条件。
        handle = self.model.patch_embed.register_forward_hook(inject)
        try:
            tokens = self.model.prepare_tokens_with_masks(image)
        finally:
            handle.remove()
        height, width = image.shape[-2] // 14, image.shape[-1] // 14
        outputs = []
        for index, block in enumerate(self.model.blocks):
            needs_gradient = tokens.requires_grad or any(p.requires_grad for p in block.parameters())
            if torch.is_grad_enabled() and needs_gradient:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self.indices:
                normalized = self.model.norm(tokens)[:, 1:]
                outputs.append(normalized.transpose(1, 2).reshape(len(image), 768, height, width))
        return outputs


class EarlyModel(PretrainingModel):
    def encode(self, inputs, domain, reference=0):
        batch, _, _, height, width = inputs["rgb"].shape
        scale = float(domain["depth_scale"])
        transformed = inputs["T_cam_from_rig"].clone().float()
        transformed[..., :3, 3] = transformed[..., :3, 3] / scale
        reference_valid = inputs["image_mask"][:, reference] & inputs["view_mask"][:, reference, None, None]
        image = torch.where(reference_valid[:, None], inputs["rgb"][:, reference], 0.)
        mode = self._encoding_options["camera_mode"]
        if isinstance(self.foundation, ConditionedFoundation):
            raw = self.foundation(image, inputs, domain, reference, mode)
        else:
            raw = self.foundation(image)
        if isinstance(self.matching, ConditionedMatching):
            pyramid = self.matching(inputs, domain, mode)
        else:
            pyramid = self.matching(inputs)
        match_hw = inputs["matching_rgb"].shape[-2:]
        return {"pyramid": pyramid, "raw_maps": raw, "K": inputs["K"], "T_cam_from_rig": transformed,
                "view_mask": inputs["view_mask"], "image_mask": inputs["image_mask"],
                "reference_index": reference, "image_size": (height, width),
                "matching_scale": inputs["K"].new_tensor((match_hw[1] / width, match_hw[0] / height))[None].expand(batch, -1)}


def install_encoding(model, options):
    """加载旧权重后显式安装；新参数只含camera_adapter，其余参数对象不变。

训练prefix为matching.camera_adapter.、foundation.camera_adapter.；检查冻结
encoder哈希时排除含camera_adapter的键。constant对照保留完全相同参数量。
"""
    options = {"matching": False, "foundation": False, "camera_mode": "geometry", **options}
    if options["camera_mode"] not in {"geometry", "constant"}:
        raise ValueError("camera_mode须为geometry或constant")
    if not (options["matching"] or options["foundation"]):
        return model
    if hasattr(model, "_encoding_options"):
        if model._encoding_options != options:
            raise ValueError("请重新构建模型来切换早期条件配置")
        return model
    if type(model) is not PretrainingModel:
        raise TypeError("此研究安装器需要原PretrainingModel实例")
    model.__class__ = EarlyModel
    model._encoding_options = options
    for name, target, channels in (("matching", ConditionedMatching, 64),
                                   ("foundation", ConditionedFoundation, 768)):
        if not options[name]:
            continue
        module = getattr(model, name)
        parameter = next(module.parameters())
        with torch.random.fork_rng(devices=[]):
            adapter = CameraAdapter(channels)
        module.camera_adapter = adapter.to(device=parameter.device, dtype=parameter.dtype)
        module.__class__ = target
    return model
